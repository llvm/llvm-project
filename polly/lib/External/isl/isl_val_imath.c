#include <isl_val_private.h>

/* Return a reference to an isl_val representing the unsigned
 * integer value stored in the "n" chunks of size "size" at "chunks".
 * The least significant chunk is assumed to be stored first.
 */
__isl_give isl_val *isl_val_int_from_chunks(isl_ctx *ctx, size_t n,
	size_t size, const void *chunks)
{
	isl_val *v;

	v = isl_val_alloc(ctx);
	if (!v)
		return NULL;

	impz_import(v->n, n, -1, size, 0, 0, chunks);
	isl_int_set_si(v->d, 1);

	return v;
}

/* Store a representation of the absolute value of the numerator of "v"
 * in terms of chunks of size "size" at "chunks".
 * The least significant chunk is stored first.
 * The number of chunks in the result can be obtained by calling
 * isl_val_n_abs_num_chunks.  The user is responsible for allocating
 * enough memory to store the results.
 *
 * In the special case of a zero value, isl_val_n_abs_num_chunks will
 * return one, while impz_export will not fill in any chunks.  We therefore
 * do it ourselves.
 */
isl_stat isl_val_get_abs_num_chunks(__isl_keep isl_val *v, size_t size,
	void *chunks)
{
	if (!v || !chunks)
		return isl_stat_error;

	if (!isl_val_is_rat(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting rational value", return isl_stat_error);

	impz_export(chunks, NULL, -1, size, 0, 0, v->n);
	if (isl_val_is_zero(v))
		memset(chunks, 0, size);

	return isl_stat_ok;
}

/* Return the number of chunks of size "size" required to
 * store the absolute value of the numerator of "v".
 */
isl_size isl_val_n_abs_num_chunks(__isl_keep isl_val *v, size_t size)
{
	if (!v)
		return isl_size_error;

	if (!isl_val_is_rat(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting rational value", return isl_size_error);

	size *= 8;
	return (impz_sizeinbase(v->n, 2) + size - 1) / size;
}
