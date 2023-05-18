# Check the decomposition of the topological sort.
# This decomposition is only performed by the incremental scheduler.
# OPTIONS: --no-schedule-whole-component
domain: { A[]; B[]; C[] }
validity: { A[] -> B[]; B[] -> C[] }
