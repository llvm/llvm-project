# Check that nodes are fused when not maximizing coincidence.
# This option is only effective for the incremental scheduler.
# OPTIONS: --no-schedule-whole-component --no-schedule-maximize-coincidence
domain: [n] -> { A[i,j,k] : 0 <= i,j,k < n; B[i,j,k] : 0 <= i,j,k < n }
validity: { A[i,j,k] -> B[i,k,j] }
proximity: { A[i,j,k] -> A[i,j,k+1]; A[i,j,k] -> B[i,k,j] }
coincidence: { A[i,j,k] -> A[i,j,k+1]; B[i,j,k] -> B[i,j,k+1] }
