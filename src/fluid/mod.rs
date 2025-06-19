use itertools::Itertools;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

type Float = ordered_float::OrderedFloat<f64>;

pub fn find_shortest_path<T: Eq + Hash + Clone + Copy>(
    a: T,
    b: T,
    neighbor_function: impl Fn(T) -> Vec<T>,
    weight_function: impl Fn(T, T) -> Float,
) -> Option<(Vec<T>, Float)> {
    pathfinding::prelude::dijkstra(
        &a,
        |&elem| {
            let neighbors = neighbor_function(elem)
                .iter()
                .map(|&neighbor| (neighbor, weight_function(elem, neighbor)))
                .collect_vec();
            neighbors
        },
        |&elem| elem == b,
    )
}

pub fn find_shortest_path_astar<T: Eq + Hash + Clone + Copy>(
    a: T,
    b: T,
    neighbor_function: impl Fn(T) -> Vec<T>,
    weight_function: impl Fn(T, T) -> Float,
    heuristic_function: impl Fn(T, T) -> Float,
) -> Option<(Vec<T>, Float)> {
    pathfinding::directed::astar::astar(
        &a,
        |&elem| {
            let neighbors = neighbor_function(elem)
                .iter()
                .map(|&neighbor| (neighbor, weight_function(elem, neighbor)))
                .collect_vec();
            neighbors
        },
        |&elem| heuristic_function(elem, b),
        |&elem| elem == b,
    )
}

pub fn find_shortest_cycle<T: Eq + Hash + Clone + Copy>(
    a: T,
    neighbor_function: impl Fn(T) -> Vec<T>,
    weight_function: impl Fn(T, T) -> Float,
) -> Option<(Vec<T>, Float)> {
    neighbor_function(a)
        .iter()
        .filter_map(|&neighbor| find_shortest_path(neighbor, a, &neighbor_function, &weight_function))
        .sorted_by(|(_, cost1), (_, cost2)| cost1.cmp(cost2))
        .next()
        .map(|(path, score)| {
            let (last, rest) = path.split_last().unwrap();
            ([&[*last], rest].concat(), score + weight_function(a, *path.first().unwrap()))
        })
}

pub fn find_ccs<T>(nodes: &[T], neighbor_function: impl Fn(T) -> Vec<T>) -> Vec<HashSet<T>>
where
    T: Eq + Hash + Clone + Copy,
{
    let mut visited = HashSet::new();
    let mut ccs = vec![];
    for &node in nodes {
        if visited.contains(&node) {
            continue;
        }
        let cc = find_cc(node, &neighbor_function);
        visited.extend(cc.clone());
        ccs.push(cc);
    }
    ccs.into_iter().collect()
}

pub fn find_cc<T>(node: T, neighbor_function: impl Fn(T) -> Vec<T>) -> HashSet<T>
where
    T: Eq + Hash + Copy,
{
    pathfinding::directed::bfs::bfs_reach(node, |&x| neighbor_function(x)).collect()
}

// Should do this for each connected component (degree of freedom!)
pub fn two_color<T>(nodes: &[T], neighbor_function: impl Fn(T) -> Vec<T>) -> Option<(HashSet<T>, HashSet<T>)>
where
    T: Eq + Hash + Clone + Copy + Debug,
{
    let mut pool = nodes.to_vec();
    let mut color1 = HashSet::new();
    let mut color2 = HashSet::new();

    while let Some(s) = pool.pop() {
        let mut queue = vec![s];

        while let Some(node) = queue.pop() {
            pool.retain(|x| x != &node);
            if color1.contains(&node) || color2.contains(&node) {
                continue;
            }

            let neighbors = neighbor_function(node);

            if neighbors.iter().any(|x| color1.contains(x)) {
                if neighbors.iter().any(|x| color2.contains(x)) {
                    return None;
                }
                color2.insert(node);
            } else if neighbors.iter().any(|x| color2.contains(x)) {
                if neighbors.iter().any(|x| color1.contains(x)) {
                    return None;
                }

                color1.insert(node);
            } else {
                // Degree of freedom.
                color2.insert(node);
            }

            queue.extend(neighbors);
        }
    }
    Some((color1, color2))
}

pub fn topological_sort<T>(nodes: &[T], neighbor_function: impl Fn(T) -> Vec<T>) -> Option<Vec<T>>
where
    T: Eq + Hash + Clone + Copy,
{
    pathfinding::directed::topological_sort::topological_sort(nodes, |&x| neighbor_function(x)).ok()
}
