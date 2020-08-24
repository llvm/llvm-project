#ifndef ISL_MULTI_H
#define ISL_MULTI_H

#include <isl/val_type.h>
#include <isl/space_type.h>
#include <isl/list.h>
#include <isl/set_type.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define ISL_DECLARE_MULTI(BASE)						\
isl_ctx *isl_multi_##BASE##_get_ctx(					\
	__isl_keep isl_multi_##BASE *multi);				\
__isl_export								\
__isl_give isl_space *isl_multi_##BASE##_get_space(			\
	__isl_keep isl_multi_##BASE *multi);				\
__isl_give isl_space *isl_multi_##BASE##_get_domain_space(		\
	__isl_keep isl_multi_##BASE *multi);				\
__isl_export								\
__isl_give isl_##BASE##_list *isl_multi_##BASE##_get_list(		\
	__isl_keep isl_multi_##BASE *multi);				\
__isl_constructor							\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_from_##BASE##_list(	\
	__isl_take isl_space *space, __isl_take isl_##BASE##_list *list); \
__isl_give isl_multi_##BASE *isl_multi_##BASE##_copy(			\
	__isl_keep isl_multi_##BASE *multi);				\
__isl_null isl_multi_##BASE *isl_multi_##BASE##_free(			\
	__isl_take isl_multi_##BASE *multi);				\
__isl_export								\
isl_bool isl_multi_##BASE##_plain_is_equal(				\
	__isl_keep isl_multi_##BASE *multi1,				\
	__isl_keep isl_multi_##BASE *multi2);				\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_reset_user(		\
	__isl_take isl_multi_##BASE *multi);				\
__isl_export								\
isl_size isl_multi_##BASE##_size(__isl_keep isl_multi_##BASE *multi);	\
__isl_export								\
__isl_give isl_##BASE *isl_multi_##BASE##_get_at(			\
	__isl_keep isl_multi_##BASE *multi, int pos);			\
__isl_give isl_##BASE *isl_multi_##BASE##_get_##BASE(			\
	__isl_keep isl_multi_##BASE *multi, int pos);			\
__isl_export								\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_set_at(			\
	__isl_take isl_multi_##BASE *multi, int pos,			\
	__isl_take isl_##BASE *el);					\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_set_##BASE(		\
	__isl_take isl_multi_##BASE *multi, int pos,			\
	__isl_take isl_##BASE *el);					\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_range_splice(		\
	__isl_take isl_multi_##BASE *multi1, unsigned pos,		\
	__isl_take isl_multi_##BASE *multi2);				\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_flatten_range(		\
	__isl_take isl_multi_##BASE *multi);				\
__isl_export								\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_flat_range_product(	\
	__isl_take isl_multi_##BASE *multi1,				\
	__isl_take isl_multi_##BASE *multi2);				\
__isl_export								\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_range_product(		\
	__isl_take isl_multi_##BASE *multi1,				\
	__isl_take isl_multi_##BASE *multi2);				\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_factor_range(		\
	__isl_take isl_multi_##BASE *multi);				\
isl_bool isl_multi_##BASE##_range_is_wrapping(				\
	__isl_keep isl_multi_##BASE *multi);				\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_range_factor_domain(	\
	__isl_take isl_multi_##BASE *multi);				\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_range_factor_range(	\
	__isl_take isl_multi_##BASE *multi);				\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_align_params(		\
	__isl_take isl_multi_##BASE *multi,				\
	__isl_take isl_space *model);					\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_from_range(		\
	__isl_take isl_multi_##BASE *multi);

#define ISL_DECLARE_MULTI_IDENTITY(BASE)				\
__isl_overload								\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_identity_multi_##BASE(	\
	__isl_take isl_multi_##BASE *multi);				\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_identity(		\
	__isl_take isl_space *space);					\
__isl_overload								\
__isl_give isl_multi_##BASE *						\
isl_multi_##BASE##_identity_on_domain_space(				\
	__isl_take isl_space *space);

#define ISL_DECLARE_MULTI_CMP(BASE)					\
int isl_multi_##BASE##_plain_cmp(__isl_keep isl_multi_##BASE *multi1,	\
	__isl_keep isl_multi_##BASE *multi2);

#define ISL_DECLARE_MULTI_ARITH(BASE)					\
__isl_overload								\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_scale_val(		\
	__isl_take isl_multi_##BASE *multi, __isl_take isl_val *v);	\
__isl_overload								\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_scale_down_val(		\
	__isl_take isl_multi_##BASE *multi, __isl_take isl_val *v);	\
__isl_overload								\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_scale_multi_val(	\
	__isl_take isl_multi_##BASE *multi,				\
	__isl_take isl_multi_val *mv);					\
__isl_overload								\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_scale_down_multi_val(	\
	__isl_take isl_multi_##BASE *multi,				\
	__isl_take isl_multi_val *mv);					\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_mod_multi_val(		\
	__isl_take isl_multi_##BASE *multi,				\
	__isl_take isl_multi_val *mv);					\
__isl_export								\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_add(			\
	__isl_take isl_multi_##BASE *multi1,				\
	__isl_take isl_multi_##BASE *multi2);				\
__isl_export								\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_sub(			\
	__isl_take isl_multi_##BASE *multi1,				\
	__isl_take isl_multi_##BASE *multi2);				\
__isl_export								\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_neg(		 	\
	__isl_take isl_multi_##BASE *multi);

#define ISL_DECLARE_MULTI_MIN_MAX(BASE)					\
__isl_export								\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_min(			\
	__isl_take isl_multi_##BASE *multi1,				\
	__isl_take isl_multi_##BASE *multi2);				\
__isl_export								\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_max(			\
	__isl_take isl_multi_##BASE *multi1,				\
	__isl_take isl_multi_##BASE *multi2);

#define ISL_DECLARE_MULTI_ADD_CONSTANT(BASE)				\
__isl_overload								\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_add_constant_val(	\
	__isl_take isl_multi_##BASE *mpa, __isl_take isl_val *v);	\
__isl_overload								\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_add_constant_multi_val(	\
	__isl_take isl_multi_##BASE *mpa, __isl_take isl_multi_val *mv);

#define ISL_DECLARE_MULTI_ZERO(BASE)					\
__isl_export								\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_zero(			\
	__isl_take isl_space *space);

#define ISL_DECLARE_MULTI_NAN(BASE)					\
isl_bool isl_multi_##BASE##_involves_nan(				\
	__isl_keep isl_multi_##BASE *multi);

#define ISL_DECLARE_MULTI_DROP_DIMS(BASE)				\
isl_size isl_multi_##BASE##_dim(__isl_keep isl_multi_##BASE *multi,	\
	enum isl_dim_type type);					\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_drop_dims(		\
	__isl_take isl_multi_##BASE *multi, enum isl_dim_type type,	\
	unsigned first, unsigned n);
#define ISL_DECLARE_MULTI_DIMS(BASE)					\
ISL_DECLARE_MULTI_DROP_DIMS(BASE)					\
isl_bool isl_multi_##BASE##_involves_dims(				\
	__isl_keep isl_multi_##BASE *multi, enum isl_dim_type type,	\
	unsigned first, unsigned n);					\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_insert_dims(		\
	__isl_take isl_multi_##BASE *multi, enum isl_dim_type type,	\
	unsigned first, unsigned n);					\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_add_dims(		\
	__isl_take isl_multi_##BASE *multi, enum isl_dim_type type,	\
	unsigned n);							\
__isl_give isl_multi_##BASE *						\
isl_multi_##BASE##_project_domain_on_params(				\
	__isl_take isl_multi_##BASE *multi);

#define ISL_DECLARE_MULTI_INSERT_DOMAIN(BASE)				\
__isl_export								\
__isl_give isl_multi_##BASE *						\
isl_multi_##BASE##_insert_domain(__isl_take isl_multi_##BASE *multi,	\
	__isl_take isl_space *domain);

#define ISL_DECLARE_MULTI_LOCALS(BASE)					\
__isl_export								\
isl_bool isl_multi_##BASE##_involves_locals(				\
	__isl_keep isl_multi_##BASE *multi);

#define ISL_DECLARE_MULTI_DIM_ID(BASE)					\
int isl_multi_##BASE##_find_dim_by_name(				\
	__isl_keep isl_multi_##BASE *multi,				\
	enum isl_dim_type type, const char *name);			\
int isl_multi_##BASE##_find_dim_by_id(					\
	__isl_keep isl_multi_##BASE *multi, enum isl_dim_type type,	\
	__isl_keep isl_id *id);						\
__isl_give isl_id *isl_multi_##BASE##_get_dim_id(			\
	__isl_keep isl_multi_##BASE *multi,				\
	enum isl_dim_type type, unsigned pos);				\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_set_dim_name(		\
	__isl_take isl_multi_##BASE *multi,				\
	enum isl_dim_type type, unsigned pos, const char *s);		\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_set_dim_id(		\
	__isl_take isl_multi_##BASE *multi,				\
	enum isl_dim_type type, unsigned pos, __isl_take isl_id *id);

#define ISL_DECLARE_MULTI_TUPLE_ID(BASE)				\
const char *isl_multi_##BASE##_get_tuple_name(				\
	__isl_keep isl_multi_##BASE *multi, enum isl_dim_type type);	\
isl_bool isl_multi_##BASE##_has_tuple_id(				\
	__isl_keep isl_multi_##BASE *multi, enum isl_dim_type type);	\
__isl_give isl_id *isl_multi_##BASE##_get_tuple_id(			\
	__isl_keep isl_multi_##BASE *multi, enum isl_dim_type type);	\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_set_tuple_name(		\
	__isl_take isl_multi_##BASE *multi,				\
	enum isl_dim_type type, const char *s);				\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_set_tuple_id(		\
	__isl_take isl_multi_##BASE *multi,				\
	enum isl_dim_type type, __isl_take isl_id *id);			\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_reset_tuple_id(		\
	__isl_take isl_multi_##BASE *multi, enum isl_dim_type type);

#define ISL_DECLARE_MULTI_WITH_DOMAIN(BASE)				\
__isl_export								\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_product(		\
	__isl_take isl_multi_##BASE *multi1,				\
	__isl_take isl_multi_##BASE *multi2);				\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_splice(			\
	__isl_take isl_multi_##BASE *multi1, unsigned in_pos,		\
	unsigned out_pos, __isl_take isl_multi_##BASE *multi2);

#define ISL_DECLARE_MULTI_BIND_DOMAIN(BASE)				\
__isl_export								\
__isl_give isl_multi_##BASE *isl_multi_##BASE##_bind_domain(		\
	__isl_take isl_multi_##BASE *multi,				\
	__isl_take isl_multi_id *tuple);				\
__isl_export								\
__isl_give isl_multi_##BASE *						\
isl_multi_##BASE##_bind_domain_wrapped_domain(				\
	__isl_take isl_multi_##BASE *multi,				\
	__isl_take isl_multi_id *tuple);

#define ISL_DECLARE_MULTI_UNBIND_PARAMS(BASE)				\
__isl_export								\
__isl_give isl_multi_##BASE *						\
isl_multi_##BASE##_unbind_params_insert_domain(				\
	__isl_take isl_multi_##BASE *multi,				\
	__isl_take isl_multi_id *domain);

#define ISL_DECLARE_MULTI_PARAM(BASE)					\
__isl_overload								\
isl_bool isl_multi_##BASE##_involves_param_id(				\
	__isl_keep isl_multi_##BASE *multi, __isl_keep isl_id *id);	\
__isl_overload								\
isl_bool isl_multi_##BASE##_involves_param_id_list(			\
	__isl_keep isl_multi_##BASE *multi,				\
	__isl_keep isl_id_list *list);

#if defined(__cplusplus)
}
#endif

#endif
