#include <isl/constraint.h>

/* This function was replaced by isl_constraint_alloc_equality.
 */
__isl_give isl_constraint *isl_equality_alloc(__isl_take isl_local_space *ls)
{
	return isl_constraint_alloc_equality(ls);
}

/* This function was replaced by isl_constraint_alloc_inequality.
 */
__isl_give isl_constraint *isl_inequality_alloc(__isl_take isl_local_space *ls)
{
	return isl_constraint_alloc_inequality(ls);
}
