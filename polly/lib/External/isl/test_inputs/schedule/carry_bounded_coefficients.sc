# Check that the dependence carrying step is not confused by
# a bound on the coefficient size.
# In particular, force the scheduler to move to a dependence carrying
# step by demanding outer coincidence and bound the size of
# the coefficients.  Earlier versions of isl would take this
# bound into account while carrying dependences, breaking
# fundamental assumptions.
# On the other hand, the dependence carrying step now tries
# to prevent loop coalescing by default, so check that indeed
# no loop coalescing occurs by comparing the computed schedule
# to the expected non-coalescing schedule.
# OPTIONS: --schedule-outer-coincidence --schedule-max-coefficient=20
domain: { C[i0, i1] : 2 <= i0 <= 3999 and 0 <= i1 <= -1 + i0 }
validity: { C[i0, i1] -> C[i0, 1 + i1] : i0 <= 3999 and i1 >= 0 and
						i1 <= -2 + i0;
		C[i0, -1 + i0] -> C[1 + i0, 0] : i0 <= 3998 and i0 >= 1 }
coincidence: { C[i0, i1] -> C[i0, 1 + i1] : i0 <= 3999 and i1 >= 0 and
						i1 <= -2 + i0;
		C[i0, -1 + i0] -> C[1 + i0, 0] : i0 <= 3998 and i0 >= 1 }
