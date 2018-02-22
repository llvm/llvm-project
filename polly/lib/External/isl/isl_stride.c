/*
 * Copyright 2012-2013 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/val.h>
#include <isl/aff.h>
#include <isl/constraint.h>
#include <isl/set.h>

/* Stride information about a specific set dimension.
 * The values of the set dimension are equal to
 * "offset" plus a multiple of "stride".
 */
struct isl_stride_info {
	isl_val *stride;
	isl_aff *offset;
};

/* Free "si" and return NULL.
 */
__isl_null isl_stride_info *isl_stride_info_free(
	__isl_take isl_stride_info *si)
{
	if (!si)
		return NULL;
	isl_val_free(si->stride);
	isl_aff_free(si->offset);
	free(si);
	return NULL;
}

/* Construct an isl_stride_info object with given offset and stride.
 */
__isl_give isl_stride_info *isl_stride_info_alloc(
	__isl_take isl_val *stride, __isl_take isl_aff *offset)
{
	struct isl_stride_info *si;

	if (!stride || !offset)
		goto error;
	si = isl_alloc_type(isl_val_get_ctx(stride), struct isl_stride_info);
	if (!si)
		goto error;
	si->stride = stride;
	si->offset = offset;
	return si;
error:
	isl_val_free(stride);
	isl_aff_free(offset);
	return NULL;
}

/* Return the stride of "si".
 */
__isl_give isl_val *isl_stride_info_get_stride(__isl_keep isl_stride_info *si)
{
	if (!si)
		return NULL;
	return isl_val_copy(si->stride);
}

/* Return the offset of "si".
 */
__isl_give isl_aff *isl_stride_info_get_offset(__isl_keep isl_stride_info *si)
{
	if (!si)
		return NULL;
	return isl_aff_copy(si->offset);
}

/* Information used inside detect_stride.
 *
 * "pos" is the set dimension at which the stride is being determined.
 * "want_offset" is set if the offset should be computed.
 * "found" is set if some stride was found already.
 * "stride" and "offset" contain the (combined) stride and offset
 * found so far and are NULL when "found" is not set.
 * If "want_offset" is not set, then "offset" remains NULL.
 */
struct isl_detect_stride_data {
	int pos;
	int want_offset;
	int found;
	isl_val *stride;
	isl_aff *offset;
};

/* Set the stride and offset of data->pos to the given
 * value and expression.
 *
 * If we had already found a stride before, then the two strides
 * are combined into a single stride.
 *
 * In particular, if the new stride information is of the form
 *
 *	i = f + s (...)
 *
 * and the old stride information is of the form
 *
 *	i = f2 + s2 (...)
 *
 * then we compute the extended gcd of s and s2
 *
 *	a s + b s2 = g,
 *
 * with g = gcd(s,s2), multiply the first equation with t1 = b s2/g
 * and the second with t2 = a s1/g.
 * This results in
 *
 *	i = (b s2 + a s1)/g i = t1 f + t2 f2 + (s s2)/g (...)
 *
 * so that t1 f + t2 f2 is the combined offset and (s s2)/g = lcm(s,s2)
 * is the combined stride.
 */
static isl_stat set_stride(struct isl_detect_stride_data *data,
	__isl_take isl_val *stride, __isl_take isl_aff *offset)
{
	int pos;

	if (!stride || !offset)
		goto error;

	pos = data->pos;

	if (data->found) {
		isl_val *stride2, *a, *b, *g;
		isl_aff *offset2;

		stride2 = data->stride;
		g = isl_val_gcdext(isl_val_copy(stride), isl_val_copy(stride2),
					&a, &b);
		a = isl_val_mul(a, isl_val_copy(stride));
		a = isl_val_div(a, isl_val_copy(g));
		stride2 = isl_val_div(stride2, g);
		b = isl_val_mul(b, isl_val_copy(stride2));
		stride = isl_val_mul(stride, stride2);

		if (!data->want_offset) {
			isl_val_free(a);
			isl_val_free(b);
		} else {
			offset2 = data->offset;
			offset2 = isl_aff_scale_val(offset2, a);
			offset = isl_aff_scale_val(offset, b);
			offset = isl_aff_add(offset, offset2);
		}
	}

	data->found = 1;
	data->stride = stride;
	if (data->want_offset)
		data->offset = offset;
	else
		isl_aff_free(offset);
	if (!data->stride || (data->want_offset && !data->offset))
		return isl_stat_error;

	return isl_stat_ok;
error:
	isl_val_free(stride);
	isl_aff_free(offset);
	return isl_stat_error;
}

/* Check if constraint "c" imposes any stride on dimension data->pos
 * and, if so, update the stride information in "data".
 *
 * In order to impose a stride on the dimension, "c" needs to be an equality
 * and it needs to involve the dimension.  Note that "c" may also be
 * a div constraint and thus an inequality that we cannot use.
 *
 * Let c be of the form
 *
 *	h(p) + g * v * i + g * stride * f(alpha) = 0
 *
 * with h(p) an expression in terms of the parameters and other dimensions
 * and f(alpha) an expression in terms of the existentially quantified
 * variables.
 *
 * If "stride" is not zero and not one, then it represents a non-trivial stride
 * on "i".  We compute a and b such that
 *
 *	a v + b stride = 1
 *
 * We have
 *
 *	g v i = -h(p) + g stride f(alpha)
 *
 *	a g v i = -a h(p) + g stride f(alpha)
 *
 *	a g v i + b g stride i = -a h(p) + g stride * (...)
 *
 *	g i = -a h(p) + g stride * (...)
 *
 *	i = -a h(p)/g + stride * (...)
 *
 * The expression "-a h(p)/g" can therefore be used as offset.
 */
static isl_stat detect_stride(__isl_take isl_constraint *c, void *user)
{
	struct isl_detect_stride_data *data = user;
	int i, n_div;
	isl_ctx *ctx;
	isl_stat r = isl_stat_ok;
	isl_val *v, *stride, *m;

	if (!isl_constraint_is_equality(c) ||
	    !isl_constraint_involves_dims(c, isl_dim_set, data->pos, 1)) {
		isl_constraint_free(c);
		return isl_stat_ok;
	}

	ctx = isl_constraint_get_ctx(c);
	stride = isl_val_zero(ctx);
	n_div = isl_constraint_dim(c, isl_dim_div);
	for (i = 0; i < n_div; ++i) {
		v = isl_constraint_get_coefficient_val(c, isl_dim_div, i);
		stride = isl_val_gcd(stride, v);
	}

	v = isl_constraint_get_coefficient_val(c, isl_dim_set, data->pos);
	m = isl_val_gcd(isl_val_copy(stride), isl_val_copy(v));
	stride = isl_val_div(stride, isl_val_copy(m));
	v = isl_val_div(v, isl_val_copy(m));

	if (!isl_val_is_zero(stride) && !isl_val_is_one(stride)) {
		isl_aff *aff;
		isl_val *gcd, *a, *b;

		gcd = isl_val_gcdext(v, isl_val_copy(stride), &a, &b);
		isl_val_free(gcd);
		isl_val_free(b);

		aff = isl_constraint_get_aff(c);
		for (i = 0; i < n_div; ++i)
			aff = isl_aff_set_coefficient_si(aff,
							 isl_dim_div, i, 0);
		aff = isl_aff_set_coefficient_si(aff, isl_dim_in, data->pos, 0);
		a = isl_val_neg(a);
		aff = isl_aff_scale_val(aff, a);
		aff = isl_aff_scale_down_val(aff, m);
		r = set_stride(data, stride, aff);
	} else {
		isl_val_free(stride);
		isl_val_free(m);
		isl_val_free(v);
	}

	isl_constraint_free(c);
	return r;
}

/* Check if the constraints in "set" imply any stride on set dimension "pos" and
 * store the results in data->stride and data->offset.
 *
 * In particular, compute the affine hull and then check if
 * any of the constraints in the hull impose any stride on the dimension.
 * If no such constraint can be found, then the offset is taken
 * to be the zero expression and the stride is taken to be one.
 */
static void set_detect_stride(__isl_keep isl_set *set, int pos,
	struct isl_detect_stride_data *data)
{
	isl_basic_set *hull;

	hull = isl_set_affine_hull(isl_set_copy(set));

	data->pos = pos;
	data->found = 0;
	data->stride = NULL;
	data->offset = NULL;
	if (isl_basic_set_foreach_constraint(hull, &detect_stride, data) < 0)
		goto error;

	if (!data->found) {
		data->stride = isl_val_one(isl_set_get_ctx(set));
		if (data->want_offset) {
			isl_space *space;
			isl_local_space *ls;

			space = isl_set_get_space(set);
			ls = isl_local_space_from_space(space);
			data->offset = isl_aff_zero_on_domain(ls);
		}
	}
	isl_basic_set_free(hull);
	return;
error:
	isl_basic_set_free(hull);
	data->stride = isl_val_free(data->stride);
	data->offset = isl_aff_free(data->offset);
}

/* Check if the constraints in "set" imply any stride on set dimension "pos" and
 * return the results in the form of an offset and a stride.
 */
__isl_give isl_stride_info *isl_set_get_stride_info(__isl_keep isl_set *set,
	int pos)
{
	struct isl_detect_stride_data data;

	data.want_offset = 1;
	set_detect_stride(set, pos, &data);

	return isl_stride_info_alloc(data.stride, data.offset);
}

/* Check if the constraints in "set" imply any stride on set dimension "pos" and
 * return this stride.
 */
__isl_give isl_val *isl_set_get_stride(__isl_keep isl_set *set, int pos)
{
	struct isl_detect_stride_data data;

	data.want_offset = 0;
	set_detect_stride(set, pos, &data);

	return data.stride;
}
