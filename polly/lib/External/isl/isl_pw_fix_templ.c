#include <isl_pw_macro.h>

/* Fix the value of the given parameter or domain dimension of "pw"
 * to be equal to "value".
 */
__isl_give PW *FN(PW,fix_si)(__isl_take PW *pw, enum isl_dim_type type,
	unsigned pos, int value)
{
	int i;
	isl_size n;

	n = FN(PW,n_piece)(pw);
	if (n < 0)
		return FN(PW,free)(pw);

	if (type == isl_dim_out)
		isl_die(FN(PW,get_ctx)(pw), isl_error_invalid,
			"cannot fix output dimension", return FN(PW,free)(pw));

	if (type == isl_dim_in)
		type = isl_dim_set;

	for (i = n - 1; i >= 0; --i) {
		isl_set *domain;

		domain = FN(PW,take_domain_at)(pw, i);
		domain = isl_set_fix_si(domain, type, pos, value);
		pw = FN(PW,restore_domain_at)(pw, i, domain);
		pw = FN(PW,exploit_equalities_and_remove_if_empty)(pw, i);
	}

	return pw;
}

/* Fix the value of the variable at position "pos" of type "type" of "pw"
 * to be equal to "v".
 */
__isl_give PW *FN(PW,fix_val)(__isl_take PW *pw,
	enum isl_dim_type type, unsigned pos, __isl_take isl_val *v)
{
	int i;
	isl_size n;

	if (!v)
		return FN(PW,free)(pw);
	if (!isl_val_is_int(v))
		isl_die(FN(PW,get_ctx)(pw), isl_error_invalid,
			"expecting integer value", goto error);

	n = FN(PW,n_piece)(pw);
	if (n < 0)
		goto error;

	if (type == isl_dim_in)
		type = isl_dim_set;

	for (i = 0; i < n; ++i) {
		isl_set *domain;

		domain = FN(PW,take_domain_at)(pw, i);
		domain = isl_set_fix(domain, type, pos, v->n);
		pw = FN(PW,restore_domain_at)(pw, i, domain);
		pw = FN(PW,exploit_equalities_and_remove_if_empty)(pw, i);
	}

	isl_val_free(v);
	return pw;
error:
	isl_val_free(v);
	return FN(PW,free)(pw);
}
