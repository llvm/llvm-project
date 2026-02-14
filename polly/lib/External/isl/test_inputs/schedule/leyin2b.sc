# This is a generalized version of leyin1.sc,
# with an extra dependence that simplifies away when N = 32.
# However, since N is known to be non-negative, this should still
# produce a similar schedule (with a single band).
# The exact form of the schedule depends on whether the whole-component or
# the incremental scheduler is used.
# This is the whole-component scheduler version.
# OPTIONS: --schedule-whole-component
domain: [N] -> { A[]; B[0:N-1]; C[] }
context: [N] -> { : N >= 0 }
validity: [N] -> { A[] -> C[] : N <= 0;
		   A[] -> B[0] : N >= 1; B[N-1] -> C[] : N >= 1 }
proximity: [N] -> { A[] -> C[] : N <= 0;
		   A[] -> B[0] : N >= 1; B[N-1] -> C[] : N >= 1 }
