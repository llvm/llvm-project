#include "assert.h"
#include "stdio.h"
#include "stdlib.h"

#define die() { \
  fprintf(stderr, "Dummy function %s called\n", __FUNCTION__); \
  abort(); \
}

void ppcg_start_block() {
  die();
}
void ppcg_end_block(){
  die();
}
void ppcg_print_macros(){
  die();
}
void pet_scop_compute_outer_to_any(){
  die();
}
void pet_scop_compute_outer_to_inner(){
  die();
}
void pet_tree_get_type(){
  die();
}
void pet_tree_foreach_access_expr(){
  die();
}
void pet_expr_get_ctx(){
  die();
}
void pet_expr_access_is_read(){
  die();
}
void pet_expr_access_is_write(){
  die();
}
void pet_expr_access_get_tagged_may_read(){
  die();
}
void pet_expr_access_get_tagged_may_write(){
  die();
}
void pet_expr_access_get_must_write(){
  die();
}
void pet_expr_access_get_index(){
  die();
}
void pet_expr_access_get_ref_id(){
  die();
}
void print_cpu(){
  die();
}
void ppcg_print_exposed_declarations(){
  die();
}
void ppcg_print_declaration(){
  die();
}
void pet_stmt_print_body(){
  die();
}
void pet_loc_get_start(){
  die();
}
void pet_loc_get_end(){
  die();
}
void pet_scop_collect_tagged_may_reads(){
  die();
}
void pet_scop_collect_may_reads(){
  die();
}
void pet_scop_collect_tagged_may_writes(){
  die();
}
void pet_scop_collect_may_writes(){
  die();
}
void pet_scop_collect_tagged_must_writes(){
  die();
}
void pet_scop_collect_must_writes(){
  die();
}
void pet_scop_collect_tagged_must_kills(){
  die();
}
void pet_transform_C_source(){
  die();
}
void pet_scop_print_original(){
  die();
}
void pet_scop_free(){
  die();
}
void pet_scop_align_params(){
  die();
}
void pet_scop_can_build_ast_exprs(){
  die();
}
void pet_scop_has_data_dependent_conditions(){
  die();
}
void pet_tree_foreach_expr(){
  die();
}
void pet_expr_foreach_call_expr(){
  die();
}
void pet_stmt_is_kill(){
  die();
}
void pet_options_args() {
  die();
}
void ppcg_print_guarded() {
  die();
}
void ppcg_version() {
  die();
}
void pet_options_set_encapsulate_dynamic_control() {
  die();
}
void generate_opencl() {
  die();
}
void generate_cpu() {
  die();
}
void pet_stmt_build_ast_exprs() {
  die();
}
