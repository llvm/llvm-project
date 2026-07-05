# Check the decomposition of the topological sort.
# This decomposition is only performed by the incremental scheduler.
# OPTIONS: --no-schedule-whole-component
domain: { A[]; B[]; C[]; D[]; E[]; F[]; G[] }
validity:
    { A[] -> C[]; B[] -> C[]; C[] -> E[]; D[] -> E[]; E[] -> F[]; E[] -> G[];
      A[] -> G[]; B[] -> E[] }
