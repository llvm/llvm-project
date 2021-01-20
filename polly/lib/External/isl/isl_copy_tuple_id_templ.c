/*
 * Copyright 2019      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Copy the identifier of tuple "src_type" in "src"
 * to that of "dst_type" in "dst", if there is any such identifier.
 */
__isl_give TYPE *FN(TYPE,copy_tuple_id)(__isl_take TYPE *dst,
	enum isl_dim_type dst_type, __isl_keep isl_space *src,
	enum isl_dim_type src_type)
{
	isl_bool has_id;
	isl_id *id;

	has_id = isl_space_has_tuple_id(src, src_type);
	if (has_id < 0)
		return FN(TYPE,free)(dst);
	if (!has_id)
		return dst;

	id = isl_space_get_tuple_id(src, src_type);
	dst = FN(TYPE,set_tuple_id)(dst, dst_type, id);

	return dst;
}
