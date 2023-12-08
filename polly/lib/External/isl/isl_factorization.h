#ifndef ISL_FACTORIZATION_H
#define ISL_FACTORIZATION_H

#include <isl/set.h>
#include <isl_morph.h>

#if defined(__cplusplus)
extern "C" {
#endif

/* Data for factorizing the basic set "bset".
 * After applying "morph" to the basic set, there are "n_group"
 * groups of consecutive set variables, each of length "len[i]",
 * with 0 <= i < n_group.
 * If no factorization is possible, then "n_group" is set to 0.
 */
struct isl_factorizer {
	isl_basic_set	*bset;
	isl_morph	*morph;
	int		n_group;
	int		*len;
};
typedef struct isl_factorizer isl_factorizer;

__isl_give isl_factorizer *isl_basic_set_factorizer(
	__isl_keep isl_basic_set *bset);

isl_ctx *isl_factorizer_get_ctx(__isl_keep isl_factorizer *f);

__isl_null isl_factorizer *isl_factorizer_free(__isl_take isl_factorizer *f);
void isl_factorizer_dump(__isl_take isl_factorizer *f);

__isl_give isl_bool isl_factorizer_every_factor_basic_set(
	__isl_keep isl_factorizer *f,
	isl_bool (*test)(__isl_keep isl_basic_set *bset, void *user),
	void *user);

#if defined(__cplusplus)
}
#endif

#endif
