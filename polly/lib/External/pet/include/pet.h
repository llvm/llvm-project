#ifndef PET_H
#define PET_H

#include <isl/aff.h>
#include <isl/arg.h>
#include <isl/ast_build.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/printer.h>
#include <isl/id_to_ast_expr.h>
#include <isl/id_to_pw_aff.h>
#include <isl/schedule.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct pet_options;
ISL_ARG_DECL(pet_options, struct pet_options, pet_options_args)

/* Create an isl_ctx that references the pet options. */
isl_ctx *isl_ctx_alloc_with_pet_options();

/* If autodetect is set, any valid scop is extracted.
 * Otherwise, the scop needs to be delimited by pragmas.
 */
int pet_options_set_autodetect(isl_ctx *ctx, int val);
int pet_options_get_autodetect(isl_ctx *ctx);

int pet_options_set_detect_conditional_assignment(isl_ctx *ctx, int val);
int pet_options_get_detect_conditional_assignment(isl_ctx *ctx);

/* If encapsulate-dynamic-control is set, then any dynamic control
 * in the input program will be encapsulated in macro statements.
 * This means in particular that no statements with arguments
 * will be created.
 */
int pet_options_set_encapsulate_dynamic_control(isl_ctx *ctx, int val);
int pet_options_get_encapsulate_dynamic_control(isl_ctx *ctx);

#define	PET_OVERFLOW_AVOID	0
#define	PET_OVERFLOW_IGNORE	1
int pet_options_set_signed_overflow(isl_ctx *ctx, int val);
int pet_options_get_signed_overflow(isl_ctx *ctx);

struct pet_loc;
typedef struct pet_loc pet_loc;

/* Return an additional reference to "loc". */
__isl_give pet_loc *pet_loc_copy(__isl_keep pet_loc *loc);
/* Free a reference to "loc". */
pet_loc *pet_loc_free(__isl_take pet_loc *loc);

/* Return the offset in the input file of the start of "loc". */
unsigned pet_loc_get_start(__isl_keep pet_loc *loc);
/* Return the offset in the input file of the character after "loc". */
unsigned pet_loc_get_end(__isl_keep pet_loc *loc);
/* Return the line number of a line within the "loc" region. */
int pet_loc_get_line(__isl_keep pet_loc *loc);
/* Return the indentation of the "loc" region. */
__isl_keep const char *pet_loc_get_indent(__isl_keep pet_loc *loc);

enum pet_expr_type {
	pet_expr_error = -1,
	pet_expr_access,
	pet_expr_call,
	pet_expr_cast,
	pet_expr_int,
	pet_expr_double,
	pet_expr_op
};

enum pet_op_type {
	/* only compound assignments operators before assignment */
	pet_op_add_assign,
	pet_op_sub_assign,
	pet_op_mul_assign,
	pet_op_div_assign,
	pet_op_and_assign,
	pet_op_xor_assign,
	pet_op_or_assign,
	pet_op_assign,
	pet_op_add,
	pet_op_sub,
	pet_op_mul,
	pet_op_div,
	pet_op_mod,
	pet_op_shl,
	pet_op_shr,
	pet_op_eq,
	pet_op_ne,
	pet_op_le,
	pet_op_ge,
	pet_op_lt,
	pet_op_gt,
	pet_op_minus,
	pet_op_post_inc,
	pet_op_post_dec,
	pet_op_pre_inc,
	pet_op_pre_dec,
	pet_op_address_of,
	pet_op_assume,
	pet_op_kill,
	pet_op_and,
	pet_op_xor,
	pet_op_or,
	pet_op_not,
	pet_op_land,
	pet_op_lor,
	pet_op_lnot,
	pet_op_cond,
	pet_op_last
};

/* Index into the pet_expr->args array when pet_expr->type == pet_expr_unary
 */
enum pet_un_arg_type {
	pet_un_arg
};

/* Indices into the pet_expr->args array when
 * pet_expr->type == pet_expr_binary
 */
enum pet_bin_arg_type {
	pet_bin_lhs,
	pet_bin_rhs
};

/* Indices into the pet_expr->args array when
 * pet_expr->type == pet_expr_ternary
 */
enum pet_ter_arg_type {
	pet_ter_cond,
	pet_ter_true,
	pet_ter_false
};

struct pet_expr;
typedef struct pet_expr pet_expr;

/* Return an additional reference to "expr". */
__isl_give pet_expr *pet_expr_copy(__isl_keep pet_expr *expr);
/* Free a reference to "expr". */
__isl_null pet_expr *pet_expr_free(__isl_take pet_expr *expr);

/* Return the isl_ctx in which "expr" was created. */
isl_ctx *pet_expr_get_ctx(__isl_keep pet_expr *expr);

/* Return the type of "expr". */
enum pet_expr_type pet_expr_get_type(__isl_keep pet_expr *expr);
/* Return the number of arguments of "expr". */
int pet_expr_get_n_arg(__isl_keep pet_expr *expr);
/* Set the number of arguments of "expr" to "n". */
__isl_give pet_expr *pet_expr_set_n_arg(__isl_take pet_expr *expr, int n);
/* Return the argument of "expr" at position "pos". */
__isl_give pet_expr *pet_expr_get_arg(__isl_keep pet_expr *expr, int pos);
/* Replace the argument of "expr" at position "pos" by "arg". */
__isl_give pet_expr *pet_expr_set_arg(__isl_take pet_expr *expr, int pos,
	__isl_take pet_expr *arg);

/* Return the operation type of operation expression "expr". */
enum pet_op_type pet_expr_op_get_type(__isl_keep pet_expr *expr);
/* Replace the operation type of operation expression "expr" by "type". */
__isl_give pet_expr *pet_expr_op_set_type(__isl_take pet_expr *expr,
	enum pet_op_type type);

/* Construct a (read) access pet_expr from an index expression. */
__isl_give pet_expr *pet_expr_from_index(__isl_take isl_multi_pw_aff *index);

/* Does "expr" represent an affine expression? */
isl_bool pet_expr_is_affine(__isl_keep pet_expr *expr);
/* Does the access expression "expr" read the accessed elements? */
isl_bool pet_expr_access_is_read(__isl_keep pet_expr *expr);
/* Does the access expression "expr" write to the accessed elements? */
isl_bool pet_expr_access_is_write(__isl_keep pet_expr *expr);
/* Does the access expression "expr" kill the accessed elements? */
isl_bool pet_expr_access_is_kill(__isl_keep pet_expr *expr);
/* Mark "expr" as a read depending on "read". */
__isl_give pet_expr *pet_expr_access_set_read(__isl_take pet_expr *expr,
	int read);
/* Mark "expr" as a write depending on "write". */
__isl_give pet_expr *pet_expr_access_set_write(__isl_take pet_expr *expr,
	int write);
/* Mark "expr" as a kill depending on "kill". */
__isl_give pet_expr *pet_expr_access_set_kill(__isl_take pet_expr *expr,
	int kill);
/* Return the reference identifier of access expression "expr". */
__isl_give isl_id *pet_expr_access_get_ref_id(__isl_keep pet_expr *expr);
/* Replace the reference identifier of access expression "expr" by "ref_id". */
__isl_give pet_expr *pet_expr_access_set_ref_id(__isl_take pet_expr *expr,
	__isl_take isl_id *ref_id);
/* Return the identifier of the outer array accessed by "expr". */
__isl_give isl_id *pet_expr_access_get_id(__isl_keep pet_expr *expr);
/* Return the index expression of access expression "expr". */
__isl_give isl_multi_pw_aff *pet_expr_access_get_index(
	__isl_keep pet_expr *expr);

/* Return the potential read access relation of access expression "expr". */
__isl_give isl_union_map *pet_expr_access_get_may_read(
	__isl_keep pet_expr *expr);
/* Return the potential write access relation of access expression "expr". */
__isl_give isl_union_map *pet_expr_access_get_may_write(
	__isl_keep pet_expr *expr);
/* Return the definite write access relation of access expression "expr". */
__isl_give isl_union_map *pet_expr_access_get_must_write(
	__isl_keep pet_expr *expr);
/* Return the argument dependent potential read access relation of "expr". */
__isl_give isl_union_map *pet_expr_access_get_dependent_may_read(
	__isl_keep pet_expr *expr);
/* Return the argument dependent potential write access relation of "expr". */
__isl_give isl_union_map *pet_expr_access_get_dependent_may_write(
	__isl_keep pet_expr *expr);
/* Return the argument dependent definite write access relation of "expr". */
__isl_give isl_union_map *pet_expr_access_get_dependent_must_write(
	__isl_keep pet_expr *expr);
/* Return the tagged potential read access relation of access "expr". */
__isl_give isl_union_map *pet_expr_access_get_tagged_may_read(
	__isl_keep pet_expr *expr);
/* Return the tagged potential write access relation of access "expr". */
__isl_give isl_union_map *pet_expr_access_get_tagged_may_write(
	__isl_keep pet_expr *expr);

/* Return the name of the function called by "expr". */
__isl_keep const char *pet_expr_call_get_name(__isl_keep pet_expr *expr);
/* Replace the name of the function called by "expr" by "name". */
__isl_give pet_expr *pet_expr_call_set_name(__isl_take pet_expr *expr,
	__isl_keep const char *name);

/* Create a pet_expr representing a cast of "arg" to "type_name". */
__isl_give pet_expr *pet_expr_new_cast(const char *type_name,
	__isl_take pet_expr *arg);
/* Replace the type of the cast performed by "expr" by "name". */
__isl_give pet_expr *pet_expr_cast_set_type_name(__isl_take pet_expr *expr,
	__isl_keep const char *name);

/* Return the value of the integer represented by "expr". */
__isl_give isl_val *pet_expr_int_get_val(__isl_keep pet_expr *expr);
/* Replace the value of the integer represented by "expr" by "v". */
__isl_give pet_expr *pet_expr_int_set_val(__isl_take pet_expr *expr,
	__isl_take isl_val *v);

/* Return a string representation of the double expression "expr". */
__isl_give char *pet_expr_double_get_str(__isl_keep pet_expr *expr);
/* Replace value and string representation of the double expression "expr" */
__isl_give pet_expr *pet_expr_double_set(__isl_take pet_expr *expr,
	double d, __isl_keep const char *s);

/* Call "fn" on each of the subexpressions of "expr" of type pet_expr_access. */
int pet_expr_foreach_access_expr(__isl_keep pet_expr *expr,
	int (*fn)(__isl_keep pet_expr *expr, void *user), void *user);
/* Call "fn" on each of the subexpressions of "expr" of type pet_expr_call. */
int pet_expr_foreach_call_expr(__isl_keep pet_expr *expr,
	int (*fn)(__isl_keep pet_expr *expr, void *user), void *user);

struct pet_context;
typedef struct pet_context pet_context;

/* Create a context with the given domain. */
__isl_give pet_context *pet_context_alloc(__isl_take isl_set *domain);
/* Return an additional reference to "pc". */
__isl_give pet_context *pet_context_copy(__isl_keep pet_context *pc);
/* Free a reference to "pc". */
__isl_null pet_context *pet_context_free(__isl_take pet_context *pc);

/* Return the isl_ctx in which "pc" was created. */
isl_ctx *pet_context_get_ctx(__isl_keep pet_context *pc);

/* Extract an affine expression defined over the domain of "pc" from "expr"
 * or return NaN.
 */
__isl_give isl_pw_aff *pet_expr_extract_affine(__isl_keep pet_expr *expr,
	__isl_keep pet_context *pc);

void pet_expr_dump(__isl_keep pet_expr *expr);

enum pet_tree_type {
	pet_tree_error = -1,
	pet_tree_expr,
	pet_tree_block,
	pet_tree_break,
	pet_tree_continue,
	pet_tree_decl,		/* A declaration without initialization */
	pet_tree_decl_init,	/* A declaration with initialization */
	pet_tree_if,		/* An if without an else branch */
	pet_tree_if_else,	/* An if with an else branch */
	pet_tree_for,
	pet_tree_infinite_loop,
	pet_tree_while,
	pet_tree_return,
};

struct pet_tree;
typedef struct pet_tree pet_tree;

/* Return the isl_ctx in which "tree" was created. */
isl_ctx *pet_tree_get_ctx(__isl_keep pet_tree *tree);

/* Return an additional reference to "tree". */
__isl_give pet_tree *pet_tree_copy(__isl_keep pet_tree *tree);
/* Free a reference to "tree". */
__isl_null pet_tree *pet_tree_free(__isl_take pet_tree *tree);

/* Return the location of "tree". */
__isl_give pet_loc *pet_tree_get_loc(__isl_keep pet_tree *tree);

/* Return the type of "tree". */
enum pet_tree_type pet_tree_get_type(__isl_keep pet_tree *tree);

/* Return the expression of the expression tree "tree". */
__isl_give pet_expr *pet_tree_expr_get_expr(__isl_keep pet_tree *tree);

/* Return the expression returned by the return tree "tree". */
__isl_give pet_expr *pet_tree_return_get_expr(__isl_keep pet_tree *tree);

/* Return the number of children of the block tree "tree". */
int pet_tree_block_n_child(__isl_keep pet_tree *tree);
/* Return child "pos" of the block tree "tree". */
__isl_give pet_tree *pet_tree_block_get_child(__isl_keep pet_tree *tree,
	int pos);

/* Is "tree" a declaration (with or without initialization)? */
int pet_tree_is_decl(__isl_keep pet_tree *tree);
/* Return the variable declared by the declaration tree "tree". */
__isl_give pet_expr *pet_tree_decl_get_var(__isl_keep pet_tree *tree);
/* Return the initial value of the pet_tree_decl_init tree "tree". */
__isl_give pet_expr *pet_tree_decl_get_init(__isl_keep pet_tree *tree);

/* Return the condition of the if tree "tree". */
__isl_give pet_expr *pet_tree_if_get_cond(__isl_keep pet_tree *tree);
/* Return the then branch of the if tree "tree". */
__isl_give pet_tree *pet_tree_if_get_then(__isl_keep pet_tree *tree);
/* Return the else branch of the if tree with else branch "tree". */
__isl_give pet_tree *pet_tree_if_get_else(__isl_keep pet_tree *tree);

/* Is "tree" a for loop, a while loop or an infinite loop? */
int pet_tree_is_loop(__isl_keep pet_tree *tree);
/* Return the induction variable of the for loop "tree" */
__isl_give pet_expr *pet_tree_loop_get_var(__isl_keep pet_tree *tree);
/* Return the initial value of the induction variable of the for loop "tree" */
__isl_give pet_expr *pet_tree_loop_get_init(__isl_keep pet_tree *tree);
/* Return the condition of the loop tree "tree" */
__isl_give pet_expr *pet_tree_loop_get_cond(__isl_keep pet_tree *tree);
/* Return the induction variable of the for loop "tree" */
__isl_give pet_expr *pet_tree_loop_get_inc(__isl_keep pet_tree *tree);
/* Return the body of the loop tree "tree" */
__isl_give pet_tree *pet_tree_loop_get_body(__isl_keep pet_tree *tree);

/* Call "fn" on each top-level expression in the nodes of "tree" */
int pet_tree_foreach_expr(__isl_keep pet_tree *tree,
	int (*fn)(__isl_keep pet_expr *expr, void *user), void *user);
/* Call "fn" on each access subexpression in the nodes of "tree" */
int pet_tree_foreach_access_expr(__isl_keep pet_tree *tree,
	int (*fn)(__isl_keep pet_expr *expr, void *user), void *user);
/* Modify all call subexpressions in the nodes of "tree" through "fn". */
__isl_give pet_tree *pet_tree_map_call_expr(__isl_take pet_tree *tree,
	__isl_give pet_expr *(*fn)(__isl_take pet_expr *expr, void *user),
	void *user);

void pet_tree_dump(__isl_keep pet_tree *tree);

/* "loc" represents the region of the source code that is represented
 * by this statement.
 *
 * If the statement has arguments, i.e., n_arg != 0, then
 * "domain" is a wrapped map, mapping the iteration domain
 * to the values of the arguments for which this statement
 * is executed.
 * Otherwise, it is simply the iteration domain.
 *
 * If one of the arguments is an access expression that accesses
 * more than one element for a given iteration, then the constraints
 * on the value of this argument (encoded in "domain") should be satisfied
 * for all of those accessed elements.
 */
struct pet_stmt {
	pet_loc *loc;
	isl_set *domain;
	pet_tree *body;

	unsigned n_arg;
	pet_expr **args;
};

/* Return the iteration space of "stmt". */
__isl_give isl_space *pet_stmt_get_space(struct pet_stmt *stmt);

/* Is "stmt" an assignment statement? */
int pet_stmt_is_assign(struct pet_stmt *stmt);
/* Is "stmt" a kill statement? */
int pet_stmt_is_kill(struct pet_stmt *stmt);

/* pet_stmt_build_ast_exprs is currently limited to only handle
 * some forms of data dependent accesses.
 * If pet_stmt_can_build_ast_exprs returns 1, then pet_stmt_build_ast_exprs
 * can safely be called on "stmt".
 */
int pet_stmt_can_build_ast_exprs(struct pet_stmt *stmt);
/* Construct an associative array from reference identifiers of
 * access expressions in "stmt" to the corresponding isl_ast_expr.
 * Each index expression is first transformed through "fn_index"
 * (if not NULL).  Then an AST expression is generated using "build".
 * Finally, the AST expression is transformed using "fn_expr"
 * (if not NULL).
 */
__isl_give isl_id_to_ast_expr *pet_stmt_build_ast_exprs(struct pet_stmt *stmt,
	__isl_keep isl_ast_build *build,
	__isl_give isl_multi_pw_aff *(*fn_index)(
		__isl_take isl_multi_pw_aff *mpa, __isl_keep isl_id *id,
		void *user), void *user_index,
	__isl_give isl_ast_expr *(*fn_expr)(__isl_take isl_ast_expr *expr,
		__isl_keep isl_id *id, void *user), void *user_expr);

/* Print "stmt" to "p".
 *
 * The access expressions in "stmt" are replaced by the isl_ast_expr
 * associated to its reference identifier in "ref2expr".
 */
__isl_give isl_printer *pet_stmt_print_body(struct pet_stmt *stmt,
	__isl_take isl_printer *p, __isl_keep isl_id_to_ast_expr *ref2expr);

/* This structure represents a defined type.
 * "name" is the name of the type, while "definition" is a string
 * representation of its definition.
 */
struct pet_type {
	char *name;
	char *definition;
};

/* context holds constraints on the parameter that ensure that
 * this array has a valid (i.e., non-negative) size
 *
 * extent holds constraints on the indices
 *
 * value_bounds holds constraints on the elements of the array
 * and may be NULL if no such constraints were specified by the user
 *
 * element_size is the size in bytes of each array element
 * element_type is the type of the array elements.
 * element_is_record is set if this type is a record type.
 *
 * live_out is set if the array appears in a live-out pragma
 *
 * if uniquely_defined is set then the array is written by a single access
 * such that any element that is ever read
 * is known to be assigned exactly once before the read
 *
 * declared is set if the array was declared somewhere inside the scop.
 * exposed is set if the declared array is visible outside the scop.
 * outer is set if the type of the array elements is a record and
 * the fields of this record are represented by separate pet_array structures.
 */
struct pet_array {
	isl_set *context;
	isl_set *extent;
	isl_set *value_bounds;
	char *element_type;
	int element_is_record;
	int element_size;
	int live_out;
	int uniquely_defined;
	int declared;
	int exposed;
	int outer;
};

/* This structure represents an implication on a boolean filter.
 * In particular, if the filter value of an element in the domain
 * of "extension" is equal to "satisfied", then the filter values
 * of the corresponding images in "extension" are also equal
 * to "satisfied".
 */
struct pet_implication {
	int satisfied;
	isl_map *extension;
};

/* This structure represents an independence implied by a for loop
 * that is marked as independent in the source code.
 * "filter" contains pairs of statement instances that are guaranteed
 * not to be dependent on each other based on the independent for loop,
 * assuming that no dependences carried by this loop are implied
 * by the variables in "local".
 * "local" contains the variables that are local to the loop that was
 * marked independent.
 */
struct pet_independence {
	isl_union_map *filter;
	isl_union_set *local;
};

/* "loc" represents the region of the source code that is represented
 * by this scop.
 * If the scop was detected based on scop and endscop pragmas, then
 * the lines containing these pragmas are included in this region.
 * In the final result, the context describes the set of parameter values
 * for which the scop can be executed.
 * During the construction of the pet_scop, the context lives in a set space
 * where each dimension refers to an outer loop.
 * context_value describes assignments to the parameters (if any)
 * outside of the scop.
 *
 * "schedule" is the schedule of the statements in the scop.
 *
 * The n_type types define types that may be referenced from by the arrays.
 *
 * The n_implication implications describe implications on boolean filters.
 *
 * The n_independence independences describe independences implied
 * by for loops that are marked independent in the source code.
 */
struct pet_scop {
	pet_loc *loc;

	isl_set *context;
	isl_set *context_value;
	isl_schedule *schedule;

	int n_type;
	struct pet_type **types;

	int n_array;
	struct pet_array **arrays;

	int n_stmt;
	struct pet_stmt **stmts;

	int n_implication;
	struct pet_implication **implications;

	int n_independence;
	struct pet_independence **independences;
};
typedef struct pet_scop pet_scop;

/* Return a textual representation of the operator. */
const char *pet_op_str(enum pet_op_type op);
int pet_op_is_inc_dec(enum pet_op_type op);

/* Extract a pet_scop from a C source file.
 * If function is not NULL, then the pet_scop is extracted from
 * a function with that name.
 */
__isl_give pet_scop *pet_scop_extract_from_C_source(isl_ctx *ctx,
	const char *filename, const char *function);

/* Transform the C source file "input" by rewriting each scop
 * When autodetecting scops, at most one scop per function is rewritten.
 * The transformed C code is written to "output".
 */
int pet_transform_C_source(isl_ctx *ctx, const char *input, FILE *output,
	__isl_give isl_printer *(*transform)(__isl_take isl_printer *p,
		__isl_take pet_scop *scop, void *user), void *user);
/* Given a scop and a printer passed to a pet_transform_C_source callback,
 * print the original corresponding code to the printer.
 */
__isl_give isl_printer *pet_scop_print_original(__isl_keep pet_scop *scop,
	__isl_take isl_printer *p);

/* Update all isl_sets and isl_maps such that they all have the same
 * parameters in the same order.
 */
__isl_give pet_scop *pet_scop_align_params(__isl_take pet_scop *scop);

/* Does "scop" contain any data dependent accesses? */
int pet_scop_has_data_dependent_accesses(__isl_keep pet_scop *scop);
/* Does "scop" contain any data dependent conditions? */
int pet_scop_has_data_dependent_conditions(__isl_keep pet_scop *scop);
/* pet_stmt_build_ast_exprs is currently limited to only handle
 * some forms of data dependent accesses.
 * If pet_scop_can_build_ast_exprs returns 1, then pet_stmt_build_ast_exprs
 * can safely be called on all statements in the scop.
 */
int pet_scop_can_build_ast_exprs(__isl_keep pet_scop *scop);

void pet_scop_dump(__isl_keep pet_scop *scop);
__isl_null pet_scop *pet_scop_free(__isl_take pet_scop *scop);

/* Return the context of "scop". */
__isl_give isl_set *pet_scop_get_context(__isl_keep pet_scop *scop);
/* Return the schedule of "scop". */
__isl_give isl_schedule *pet_scop_get_schedule(__isl_keep pet_scop *scop);
/* Return the set of all statement instances. */
__isl_give isl_union_set *pet_scop_get_instance_set(__isl_keep pet_scop *scop);
/* Return the potential read access relation. */
__isl_give isl_union_map *pet_scop_get_may_reads(__isl_keep pet_scop *scop);
/* Return the tagged potential read access relation. */
__isl_give isl_union_map *pet_scop_get_tagged_may_reads(
	__isl_keep pet_scop *scop);
/* Return the potential write access relation. */
__isl_give isl_union_map *pet_scop_get_may_writes(__isl_keep pet_scop *scop);
/* Return the definite write access relation. */
__isl_give isl_union_map *pet_scop_get_must_writes(__isl_keep pet_scop *scop);
/* Return the tagged potential write access relation. */
__isl_give isl_union_map *pet_scop_get_tagged_may_writes(
	__isl_keep pet_scop *scop);
/* Return the tagged definite write access relation. */
__isl_give isl_union_map *pet_scop_get_tagged_must_writes(
	__isl_keep pet_scop *scop);
/* Return the definite kill access relation. */
__isl_give isl_union_map *pet_scop_get_must_kills(__isl_keep pet_scop *scop);
/* Return the tagged definite kill access relation. */
__isl_give isl_union_map *pet_scop_get_tagged_must_kills(
	__isl_keep pet_scop *scop);

/* Compute a mapping from all outermost arrays (of structs) in scop
 * to their innermost members.
 */
__isl_give isl_union_map *pet_scop_compute_outer_to_inner(
	__isl_keep pet_scop *scop);
/* Compute a mapping from all outermost arrays (of structs) in scop
 * to their members, including the outermost arrays themselves.
 */
__isl_give isl_union_map *pet_scop_compute_outer_to_any(
	__isl_keep pet_scop *scop);

#if defined(__cplusplus)
}
#endif

#endif
