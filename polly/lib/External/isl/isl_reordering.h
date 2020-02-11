#ifndef ISL_REORDERING_H
#define ISL_REORDERING_H

#include <isl/space.h>

/* pos maps original dimensions to new dimensions.
 * The final space is given by "space".
 * The number of dimensions (i.e., the range of values) in the result
 * may be larger than the number of dimensions in the input.
 * In particular, the possible values of the entries in pos ranges from 0 to
 * the total dimension of dim - 1, unless isl_reordering_extend
 * has been called.
 */
struct isl_reordering {
	int ref;
	isl_space *space;
	unsigned len;
	int pos[1];
};
typedef struct isl_reordering isl_reordering;

isl_ctx *isl_reordering_get_ctx(__isl_keep isl_reordering *r);
__isl_keep isl_space *isl_reordering_peek_space(__isl_keep isl_reordering *r);
__isl_give isl_space *isl_reordering_get_space(__isl_keep isl_reordering *r);
__isl_give isl_reordering *isl_parameter_alignment_reordering(
	__isl_keep isl_space *alignee, __isl_keep isl_space *aligner);
__isl_give isl_reordering *isl_reordering_unbind_params_insert_domain(
	__isl_keep isl_space *space, __isl_keep isl_multi_id *tuple);
__isl_give isl_reordering *isl_reordering_copy(__isl_keep isl_reordering *exp);
__isl_null isl_reordering *isl_reordering_free(__isl_take isl_reordering *exp);
__isl_give isl_reordering *isl_reordering_extend_space(
	__isl_take isl_reordering *exp, __isl_take isl_space *space);
__isl_give isl_reordering *isl_reordering_extend(__isl_take isl_reordering *exp,
	unsigned extra);

#endif
