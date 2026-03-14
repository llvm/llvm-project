# Check that the Feautrier schedule is not confused by
# compressed nodes in a subgraph of the original dependence graph.
# OPTIONS: --schedule-algorithm=feautrier
domain: { A[]; B[0]; C[] }
validity: { A[] -> B[0] }
