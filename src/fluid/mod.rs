use itertools::Itertools;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

type Float = ordered_float::OrderedFloat<f64>;

// Define a struct that holds a neighbor function. (a fluid graph)
#[derive(Clone, Debug)]
pub struct FluidGraph<T: Eq + Hash + Clone + Copy> {
    pub neighbor_function: fn(T) -> Vec<T>,
}

// Implement all methods for the FluidGraph struct.
impl<T: Eq + Hash + Clone + Copy> FluidGraph<T> {
    pub fn new(neighbor_function: fn(T) -> Vec<T>) -> Self {
        Self { neighbor_function }
    }

    pub fn neighbors(&self, node: T) -> Vec<T> {
        (self.neighbor_function)(node)
    }

    pub fn shortest_path(&self, a: T, b: T, weight_function: impl Fn(T, T) -> Float) -> Option<(Vec<T>, Float)> {
        pathfinding::prelude::dijkstra(
            &a,
            |&elem| {
                let neighbors = self
                    .neighbors(elem)
                    .iter()
                    .map(|&neighbor| (neighbor, weight_function(elem, neighbor)))
                    .collect_vec();
                neighbors
            },
            |&elem| elem == b,
        )
    }

    pub fn shortest_path_heuristic(
        &self,
        a: T,
        b: T,
        weight_function: impl Fn(T, T) -> Float,
        heuristic_function: impl Fn(T, T) -> Float,
    ) -> Option<(Vec<T>, Float)> {
        pathfinding::directed::astar::astar(
            &a,
            |&elem| {
                let neighbors = self
                    .neighbors(elem)
                    .iter()
                    .map(|&neighbor| (neighbor, weight_function(elem, neighbor)))
                    .collect_vec();
                neighbors
            },
            |&elem| heuristic_function(elem, b),
            |&elem| elem == b,
        )
    }

    pub fn shortest_cycle(&self, a: T, weight_function: impl Fn(T, T) -> Float) -> Option<(Vec<T>, Float)> {
        self.neighbors(a)
            .iter()
            .filter_map(|&neighbor| self.shortest_path(neighbor, a, &weight_function))
            .sorted_by(|(_, cost1), (_, cost2)| cost1.cmp(cost2))
            .next()
            .map(|(path, score)| {
                let (last, rest) = path.split_last().unwrap();
                ([&[*last], rest].concat(), score + weight_function(a, *path.first().unwrap()))
            })
    }

    pub fn connected_components(&self, nodes: &[T]) -> Vec<HashSet<T>> {
        let mut visited = HashSet::new();
        let mut ccs = vec![];
        for &node in nodes {
            if visited.contains(&node) {
                continue;
            }
            let cc = self.connected_component(node);
            visited.extend(cc.clone());
            ccs.push(cc);
        }
        ccs.into_iter().collect()
    }

    pub fn connected_component(&self, node: T) -> HashSet<T> {
        pathfinding::directed::bfs::bfs_reach(node, |&x| self.neighbors(x)).collect()
    }

    pub fn two_coloring(&self, nodes: &[T]) -> Option<(HashSet<T>, HashSet<T>)> {
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

                let neighbors = self.neighbors(node);

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

    pub fn topological_sort(&self, nodes: &[T]) -> Option<Vec<T>> {
        pathfinding::directed::topological_sort::topological_sort(nodes, |&x| self.neighbors(x)).ok()
    }
}
