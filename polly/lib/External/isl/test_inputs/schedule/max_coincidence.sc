# Check that nodes are not fused when maximizing coincidence.
# This option is only effective for the incremental scheduler.
# OPTIONS: --no-schedule-whole-component --schedule-maximize-coincidence
domain: [n] -> { A[i,j,k] : 0 <= i,j,k < n; B[i,j,k] : 0 <= i,j,k < n }
validity: { A[i,j,k] -> B[i,k,j] }
proximity: { A[i,j,k] -> B[i,k,j] }
coincidence: { A[i,j,k] -> A[i,j,k+1]; B[i,j,k] -> B[i,j,k+1] }
