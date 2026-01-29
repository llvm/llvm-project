# This is a specialized version of leyin2?.sc (for N = 32),
# showing that a single band is computed.
domain: { A[]; B[0:31]; C[] }
validity: { A[] -> B[0]; B[31] -> C[] }
proximity: { A[] -> B[0]; B[31] -> C[] }
