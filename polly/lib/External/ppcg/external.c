#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <pet.h>
#include "cpu.h"
#include "opencl.h"


#define die() { \
  fprintf(stderr, "Dummy function %s called\n", __FUNCTION__); \
  abort(); \
}

__isl_give isl_union_map *pet_scop_compute_outer_to_any(
  __isl_keep pet_scop *scop) {
  die();
}
__isl_give isl_union_map *pet_scop_compute_outer_to_inner(
  __isl_keep pet_scop *scop) {
  die();
}
enum pet_tree_type pet_tree_get_type(__isl_keep pet_tree *tree) {
  die();
}
int pet_tree_foreach_access_expr(__isl_keep pet_tree *tree,
  int (*fn)(__isl_keep pet_expr *expr, void *user), void *user) {
  die();
}
isl_ctx *pet_expr_get_ctx(__isl_keep pet_expr *expr) {
  die();
}
isl_bool pet_expr_access_is_read(__isl_keep pet_expr *expr) {
  die();
}
isl_bool pet_expr_access_is_write(__isl_keep pet_expr *expr) {
  die();
}
__isl_give isl_union_map *pet_expr_access_get_tagged_may_read(
  __isl_keep pet_expr *expr) {
  die();
}
__isl_give isl_union_map *pet_expr_access_get_tagged_may_write(
  __isl_keep pet_expr *expr) {
  die();
}
__isl_give isl_union_map *pet_expr_access_get_must_write(
  __isl_keep pet_expr *expr) {
  die();
}
__isl_give isl_multi_pw_aff *pet_expr_access_get_index(
  __isl_keep pet_expr *expr) {
  die();
}
__isl_give isl_id *pet_expr_access_get_ref_id(__isl_keep pet_expr *expr) {
  die();
}
__isl_give isl_printer *print_cpu(__isl_take isl_printer *p,
  struct ppcg_scop *ps, struct ppcg_options *options) {
  die();
}

__isl_give isl_printer *pet_stmt_print_body(struct pet_stmt *stmt,
  __isl_take isl_printer *p, __isl_keep isl_id_to_ast_expr *ref2expr) {
  die();
}
unsigned pet_loc_get_start(__isl_keep pet_loc *loc) {
  die();
}
unsigned pet_loc_get_end(__isl_keep pet_loc *loc) {
  die();
}
int pet_transform_C_source(isl_ctx *ctx, const char *input, FILE *output,
  __isl_give isl_printer *(*transform)(__isl_take isl_printer *p,
    __isl_take pet_scop *scop, void *user), void *user) {
  die();
}
__isl_give isl_printer *pet_scop_print_original(__isl_keep pet_scop *scop,
  __isl_take isl_printer *p) {
  die();
}
__isl_null pet_scop *pet_scop_free(__isl_take pet_scop *scop) {
  die();
}
__isl_give pet_scop *pet_scop_align_params(__isl_take pet_scop *scop) {
  die();
}
int pet_scop_can_build_ast_exprs(__isl_keep pet_scop *scop) {
  die();
}
int pet_scop_has_data_dependent_conditions(__isl_keep pet_scop *scop) {
  die();
}
int pet_tree_foreach_expr(__isl_keep pet_tree *tree,
  int (*fn)(__isl_keep pet_expr *expr, void *user), void *user) {
  die();
}
int pet_expr_foreach_call_expr(__isl_keep pet_expr *expr,
  int (*fn)(__isl_keep pet_expr *expr, void *user), void *user) {
  die();
}
int pet_stmt_is_kill(struct pet_stmt *stmt) {
  die();
}
struct isl_args pet_options_args;
const char *ppcg_version(void) {
  die();
}
int pet_options_set_encapsulate_dynamic_control(isl_ctx *ctx, int val) {
  die();
}
int generate_opencl(isl_ctx *ctx, struct ppcg_options *options,
  const char *input, const char *output) {
  die();
}
int generate_cpu(isl_ctx *ctx, struct ppcg_options *options,
  const char *input, const char *output) {
  die();
}
__isl_give isl_id_to_ast_expr *pet_stmt_build_ast_exprs(struct pet_stmt *stmt,
  __isl_keep isl_ast_build *build,
  __isl_give isl_multi_pw_aff *(*fn_index)(
    __isl_take isl_multi_pw_aff *mpa, __isl_keep isl_id *id,
    void *user), void *user_index,
  __isl_give isl_ast_expr *(*fn_expr)(__isl_take isl_ast_expr *expr,
    __isl_keep isl_id *id, void *user), void *user_expr) {
  die();
}
__isl_give isl_union_map *pet_scop_get_tagged_may_reads(
  __isl_keep pet_scop *scop) {
  die();
}
__isl_give isl_union_map *pet_scop_get_may_reads(__isl_keep pet_scop *scop) {
  die();
}
__isl_give isl_union_map *pet_scop_get_may_writes(__isl_keep pet_scop *scop) {
  die();
}
__isl_give isl_union_map *pet_scop_get_must_writes(__isl_keep pet_scop *scop) {
  die();
}
__isl_give isl_union_map *pet_scop_get_tagged_may_writes(
  __isl_keep pet_scop *scop) {
  die();
}
__isl_give isl_union_map *pet_scop_get_tagged_must_writes(
  __isl_keep pet_scop *scop) {
  die();
}
__isl_give isl_union_map *pet_scop_get_must_kills(__isl_keep pet_scop *scop) {
  die();
}
__isl_give isl_union_map *pet_scop_get_tagged_must_kills(
  __isl_keep pet_scop *scop) {
  die();
}
__isl_keep const char *pet_expr_call_get_name(__isl_keep pet_expr *expr) {
  die();
}
__isl_give pet_expr *pet_expr_call_set_name(__isl_take pet_expr *expr,
  __isl_keep const char *name) {
  die();
}
__isl_give pet_expr *pet_expr_get_arg(__isl_keep pet_expr *expr, int pos) {
  die();
}
__isl_give pet_expr *pet_expr_new_cast(const char *type_name,
  __isl_take pet_expr *arg) {
  die();
}
__isl_give pet_expr *pet_expr_set_arg(__isl_take pet_expr *expr, int pos,
  __isl_take pet_expr *arg) {
  die();
}
__isl_give pet_tree *pet_tree_copy(__isl_keep pet_tree *tree) {
  die();
}
__isl_null pet_tree *pet_tree_free(__isl_take pet_tree *tree) {
  die();
}
__isl_give pet_tree *pet_tree_map_call_expr(__isl_take pet_tree *tree,
  __isl_give pet_expr *(*fn)(__isl_take pet_expr *expr, void *user),
  void *user) {
  die();
}
__isl_give isl_union_map *pet_expr_access_get_may_read(
  __isl_keep pet_expr *expr) {
  die();
}
__isl_give isl_union_map *pet_expr_access_get_may_write(
  __isl_keep pet_expr *expr) {
  die();
}
