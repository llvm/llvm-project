/// These are automatically generated checked C++ bindings for isl.
///
/// isl is a library for computing with integer sets and maps described by
/// Presburger formulas. On top of this, isl provides various tools for
/// polyhedral compilation, ranging from dependence analysis over scheduling
/// to AST generation.

#ifndef ISL_CPP_CHECKED
#define ISL_CPP_CHECKED

#include <isl/val.h>
#include <isl/aff.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/id.h>
#include <isl/map.h>
#include <isl/vec.h>
#include <isl/ilp.h>
#include <isl/union_set.h>
#include <isl/union_map.h>
#include <isl/flow.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/ast_build.h>
#include <isl/constraint.h>
#include <isl/polynomial.h>
#include <isl/mat.h>
#include <isl/fixed_box.h>

#include <stdio.h>
#include <stdlib.h>

#include <functional>
#include <string>

namespace isl {
inline namespace noexceptions {

#define ISLPP_STRINGIZE_(X) #X
#define ISLPP_STRINGIZE(X) ISLPP_STRINGIZE_(X)

#define ISLPP_ASSERT(test, message)                          \
  do {                                                       \
    if (test)                                                \
      break;                                                 \
    fputs("Assertion \"" #test "\" failed at " __FILE__      \
      ":" ISLPP_STRINGIZE(__LINE__) "\n  " message "\n",     \
      stderr);                                               \
    abort();                                                 \
  } while (0)

class boolean {
private:
  mutable bool checked = false;
  isl_bool val;

  friend boolean manage(isl_bool val);
  boolean(isl_bool val): val(val) {}
public:
  boolean()
      : val(isl_bool_error) {}
  ~boolean() {
    // ISLPP_ASSERT(checked, "IMPLEMENTATION ERROR: Unchecked state");
  }

  /* implicit */ boolean(bool val)
      : val(val ? isl_bool_true : isl_bool_false) {}

  bool is_error() const { checked = true; return val == isl_bool_error; }
  bool is_false() const { checked = true; return val == isl_bool_false; }
  bool is_true() const { checked = true; return val == isl_bool_true; }

  operator bool() const {
    // ISLPP_ASSERT(checked, "IMPLEMENTATION ERROR: Unchecked error state");
    ISLPP_ASSERT(!is_error(), "IMPLEMENTATION ERROR: Unhandled error state");
    return is_true();
  }

  boolean operator!() const {
    if (is_error())
      return *this;
    return !is_true();
  }
};

inline boolean manage(isl_bool val) {
  return boolean(val);
}

class ctx {
  isl_ctx *ptr;
public:
  /* implicit */ ctx(isl_ctx *ctx)
      : ptr(ctx) {}
  isl_ctx *release() {
    auto tmp = ptr;
    ptr = nullptr;
    return tmp;
  }
  isl_ctx *get() {
    return ptr;
  }
};

/* Class encapsulating an isl_stat value.
 */
class stat {
private:
	mutable bool checked = false;
	isl_stat val;

	friend stat manage(isl_stat val);
public:
	constexpr stat(isl_stat val) : val(val) {}
	static stat ok() {
		return stat(isl_stat_ok);
	}
	static stat error() {
		return stat(isl_stat_error);
	}
	stat() : val(isl_stat_error) {}
	~stat() {
		// ISLPP_ASSERT(checked, "IMPLEMENTATION ERROR: Unchecked state");
	}

	isl_stat release() {
		checked = true;
		return val;
	}

	bool is_error() const {
		checked = true;
		return val == isl_stat_error;
	}
	bool is_ok() const {
		checked = true;
		return val == isl_stat_ok;
	}
};


inline stat manage(isl_stat val)
{
	return stat(val);
}

enum class dim {
  cst = isl_dim_cst,
  param = isl_dim_param,
  in = isl_dim_in,
  out = isl_dim_out,
  set = isl_dim_set,
  div = isl_dim_div,
  all = isl_dim_all
};

}
} // namespace isl

namespace isl {

inline namespace noexceptions {

// forward declarations
class aff;
class aff_list;
class ast_build;
class ast_expr;
class ast_expr_list;
class ast_node;
class ast_node_list;
class basic_map;
class basic_map_list;
class basic_set;
class basic_set_list;
class constraint;
class constraint_list;
class fixed_box;
class id;
class id_list;
class id_to_ast_expr;
class local_space;
class map;
class map_list;
class mat;
class multi_aff;
class multi_pw_aff;
class multi_union_pw_aff;
class multi_val;
class point;
class pw_aff;
class pw_aff_list;
class pw_multi_aff;
class pw_multi_aff_list;
class pw_qpolynomial;
class pw_qpolynomial_fold_list;
class pw_qpolynomial_list;
class qpolynomial;
class schedule;
class schedule_constraints;
class schedule_node;
class set;
class set_list;
class space;
class term;
class union_access_info;
class union_flow;
class union_map;
class union_map_list;
class union_pw_aff;
class union_pw_aff_list;
class union_pw_multi_aff;
class union_pw_multi_aff_list;
class union_pw_qpolynomial;
class union_set;
class union_set_list;
class val;
class val_list;
class vec;

// declarations for isl::aff
inline aff manage(__isl_take isl_aff *ptr);
inline aff manage_copy(__isl_keep isl_aff *ptr);

class aff {
  friend inline aff manage(__isl_take isl_aff *ptr);
  friend inline aff manage_copy(__isl_keep isl_aff *ptr);

  isl_aff *ptr = nullptr;

  inline explicit aff(__isl_take isl_aff *ptr);

public:
  inline /* implicit */ aff();
  inline /* implicit */ aff(const aff &obj);
  inline /* implicit */ aff(std::nullptr_t);
  inline explicit aff(local_space ls);
  inline explicit aff(local_space ls, val val);
  inline explicit aff(ctx ctx, const std::string &str);
  inline aff &operator=(aff obj);
  inline ~aff();
  inline __isl_give isl_aff *copy() const &;
  inline __isl_give isl_aff *copy() && = delete;
  inline __isl_keep isl_aff *get() const;
  inline __isl_give isl_aff *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline aff add(aff aff2) const;
  inline aff add_coefficient_si(isl::dim type, int pos, int v) const;
  inline aff add_coefficient_val(isl::dim type, int pos, val v) const;
  inline aff add_constant_num_si(int v) const;
  inline aff add_constant_si(int v) const;
  inline aff add_constant_val(val v) const;
  inline aff add_dims(isl::dim type, unsigned int n) const;
  inline aff align_params(space model) const;
  inline aff ceil() const;
  inline int coefficient_sgn(isl::dim type, int pos) const;
  inline int dim(isl::dim type) const;
  inline aff div(aff aff2) const;
  inline aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline basic_set eq_basic_set(aff aff2) const;
  inline set eq_set(aff aff2) const;
  inline val eval(point pnt) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline aff floor() const;
  inline aff from_range() const;
  inline basic_set ge_basic_set(aff aff2) const;
  inline set ge_set(aff aff2) const;
  inline val get_coefficient_val(isl::dim type, int pos) const;
  inline val get_constant_val() const;
  inline val get_denominator_val() const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline aff get_div(int pos) const;
  inline local_space get_domain_local_space() const;
  inline space get_domain_space() const;
  inline uint32_t get_hash() const;
  inline local_space get_local_space() const;
  inline space get_space() const;
  inline aff gist(set context) const;
  inline aff gist_params(set context) const;
  inline basic_set gt_basic_set(aff aff2) const;
  inline set gt_set(aff aff2) const;
  inline aff insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean is_cst() const;
  inline boolean is_nan() const;
  inline basic_set le_basic_set(aff aff2) const;
  inline set le_set(aff aff2) const;
  inline basic_set lt_basic_set(aff aff2) const;
  inline set lt_set(aff aff2) const;
  inline aff mod(val mod) const;
  inline aff move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline aff mul(aff aff2) const;
  static inline aff nan_on_domain(local_space ls);
  inline set ne_set(aff aff2) const;
  inline aff neg() const;
  inline basic_set neg_basic_set() const;
  static inline aff param_on_domain_space_id(space space, id id);
  inline boolean plain_is_equal(const aff &aff2) const;
  inline boolean plain_is_zero() const;
  inline aff project_domain_on_params() const;
  inline aff pullback(multi_aff ma) const;
  inline aff pullback_aff(aff aff2) const;
  inline aff scale(val v) const;
  inline aff scale_down(val v) const;
  inline aff scale_down_ui(unsigned int f) const;
  inline aff set_coefficient_si(isl::dim type, int pos, int v) const;
  inline aff set_coefficient_val(isl::dim type, int pos, val v) const;
  inline aff set_constant_si(int v) const;
  inline aff set_constant_val(val v) const;
  inline aff set_dim_id(isl::dim type, unsigned int pos, id id) const;
  inline aff set_tuple_id(isl::dim type, id id) const;
  inline aff sub(aff aff2) const;
  static inline aff var_on_domain(local_space ls, isl::dim type, unsigned int pos);
  inline basic_set zero_basic_set() const;
};

// declarations for isl::aff_list
inline aff_list manage(__isl_take isl_aff_list *ptr);
inline aff_list manage_copy(__isl_keep isl_aff_list *ptr);

class aff_list {
  friend inline aff_list manage(__isl_take isl_aff_list *ptr);
  friend inline aff_list manage_copy(__isl_keep isl_aff_list *ptr);

  isl_aff_list *ptr = nullptr;

  inline explicit aff_list(__isl_take isl_aff_list *ptr);

public:
  inline /* implicit */ aff_list();
  inline /* implicit */ aff_list(const aff_list &obj);
  inline /* implicit */ aff_list(std::nullptr_t);
  inline aff_list &operator=(aff_list obj);
  inline ~aff_list();
  inline __isl_give isl_aff_list *copy() const &;
  inline __isl_give isl_aff_list *copy() && = delete;
  inline __isl_keep isl_aff_list *get() const;
  inline __isl_give isl_aff_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline aff_list add(aff el) const;
  static inline aff_list alloc(ctx ctx, int n);
  inline aff_list concat(aff_list list2) const;
  inline aff_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(aff)> &fn) const;
  static inline aff_list from_aff(aff el);
  inline aff get_aff(int index) const;
  inline aff get_at(int index) const;
  inline aff_list insert(unsigned int pos, aff el) const;
  inline int n_aff() const;
  inline aff_list reverse() const;
  inline aff_list set_aff(int index, aff el) const;
  inline int size() const;
  inline aff_list swap(unsigned int pos1, unsigned int pos2) const;
};

// declarations for isl::ast_build
inline ast_build manage(__isl_take isl_ast_build *ptr);
inline ast_build manage_copy(__isl_keep isl_ast_build *ptr);

class ast_build {
  friend inline ast_build manage(__isl_take isl_ast_build *ptr);
  friend inline ast_build manage_copy(__isl_keep isl_ast_build *ptr);

  isl_ast_build *ptr = nullptr;

  inline explicit ast_build(__isl_take isl_ast_build *ptr);

public:
  inline /* implicit */ ast_build();
  inline /* implicit */ ast_build(const ast_build &obj);
  inline /* implicit */ ast_build(std::nullptr_t);
  inline explicit ast_build(ctx ctx);
  inline ast_build &operator=(ast_build obj);
  inline ~ast_build();
  inline __isl_give isl_ast_build *copy() const &;
  inline __isl_give isl_ast_build *copy() && = delete;
  inline __isl_keep isl_ast_build *get() const;
  inline __isl_give isl_ast_build *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

  inline ast_expr access_from(pw_multi_aff pma) const;
  inline ast_expr access_from(multi_pw_aff mpa) const;
  inline ast_node ast_from_schedule(union_map schedule) const;
  inline ast_expr call_from(pw_multi_aff pma) const;
  inline ast_expr call_from(multi_pw_aff mpa) const;
  inline ast_expr expr_from(set set) const;
  inline ast_expr expr_from(pw_aff pa) const;
  static inline ast_build from_context(set set);
  inline union_map get_schedule() const;
  inline space get_schedule_space() const;
  inline ast_node node_from_schedule(schedule schedule) const;
  inline ast_node node_from_schedule_map(union_map schedule) const;
  inline ast_build restrict(set set) const;
};

// declarations for isl::ast_expr
inline ast_expr manage(__isl_take isl_ast_expr *ptr);
inline ast_expr manage_copy(__isl_keep isl_ast_expr *ptr);

class ast_expr {
  friend inline ast_expr manage(__isl_take isl_ast_expr *ptr);
  friend inline ast_expr manage_copy(__isl_keep isl_ast_expr *ptr);

  isl_ast_expr *ptr = nullptr;

  inline explicit ast_expr(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr();
  inline /* implicit */ ast_expr(const ast_expr &obj);
  inline /* implicit */ ast_expr(std::nullptr_t);
  inline ast_expr &operator=(ast_expr obj);
  inline ~ast_expr();
  inline __isl_give isl_ast_expr *copy() const &;
  inline __isl_give isl_ast_expr *copy() && = delete;
  inline __isl_keep isl_ast_expr *get() const;
  inline __isl_give isl_ast_expr *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline ast_expr access(ast_expr_list indices) const;
  inline ast_expr add(ast_expr expr2) const;
  inline ast_expr address_of() const;
  inline ast_expr call(ast_expr_list arguments) const;
  inline ast_expr div(ast_expr expr2) const;
  inline ast_expr eq(ast_expr expr2) const;
  static inline ast_expr from_id(id id);
  static inline ast_expr from_val(val v);
  inline ast_expr ge(ast_expr expr2) const;
  inline id get_id() const;
  inline ast_expr get_op_arg(int pos) const;
  inline int get_op_n_arg() const;
  inline val get_val() const;
  inline ast_expr gt(ast_expr expr2) const;
  inline boolean is_equal(const ast_expr &expr2) const;
  inline ast_expr le(ast_expr expr2) const;
  inline ast_expr lt(ast_expr expr2) const;
  inline ast_expr mul(ast_expr expr2) const;
  inline ast_expr neg() const;
  inline ast_expr pdiv_q(ast_expr expr2) const;
  inline ast_expr pdiv_r(ast_expr expr2) const;
  inline ast_expr set_op_arg(int pos, ast_expr arg) const;
  inline ast_expr sub(ast_expr expr2) const;
  inline ast_expr substitute_ids(id_to_ast_expr id2expr) const;
  inline std::string to_C_str() const;
};

// declarations for isl::ast_expr_list
inline ast_expr_list manage(__isl_take isl_ast_expr_list *ptr);
inline ast_expr_list manage_copy(__isl_keep isl_ast_expr_list *ptr);

class ast_expr_list {
  friend inline ast_expr_list manage(__isl_take isl_ast_expr_list *ptr);
  friend inline ast_expr_list manage_copy(__isl_keep isl_ast_expr_list *ptr);

  isl_ast_expr_list *ptr = nullptr;

  inline explicit ast_expr_list(__isl_take isl_ast_expr_list *ptr);

public:
  inline /* implicit */ ast_expr_list();
  inline /* implicit */ ast_expr_list(const ast_expr_list &obj);
  inline /* implicit */ ast_expr_list(std::nullptr_t);
  inline ast_expr_list &operator=(ast_expr_list obj);
  inline ~ast_expr_list();
  inline __isl_give isl_ast_expr_list *copy() const &;
  inline __isl_give isl_ast_expr_list *copy() && = delete;
  inline __isl_keep isl_ast_expr_list *get() const;
  inline __isl_give isl_ast_expr_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline ast_expr_list add(ast_expr el) const;
  static inline ast_expr_list alloc(ctx ctx, int n);
  inline ast_expr_list concat(ast_expr_list list2) const;
  inline ast_expr_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(ast_expr)> &fn) const;
  static inline ast_expr_list from_ast_expr(ast_expr el);
  inline ast_expr get_ast_expr(int index) const;
  inline ast_expr get_at(int index) const;
  inline ast_expr_list insert(unsigned int pos, ast_expr el) const;
  inline int n_ast_expr() const;
  inline ast_expr_list reverse() const;
  inline ast_expr_list set_ast_expr(int index, ast_expr el) const;
  inline int size() const;
  inline ast_expr_list swap(unsigned int pos1, unsigned int pos2) const;
};

// declarations for isl::ast_node
inline ast_node manage(__isl_take isl_ast_node *ptr);
inline ast_node manage_copy(__isl_keep isl_ast_node *ptr);

class ast_node {
  friend inline ast_node manage(__isl_take isl_ast_node *ptr);
  friend inline ast_node manage_copy(__isl_keep isl_ast_node *ptr);

  isl_ast_node *ptr = nullptr;

  inline explicit ast_node(__isl_take isl_ast_node *ptr);

public:
  inline /* implicit */ ast_node();
  inline /* implicit */ ast_node(const ast_node &obj);
  inline /* implicit */ ast_node(std::nullptr_t);
  inline ast_node &operator=(ast_node obj);
  inline ~ast_node();
  inline __isl_give isl_ast_node *copy() const &;
  inline __isl_give isl_ast_node *copy() && = delete;
  inline __isl_keep isl_ast_node *get() const;
  inline __isl_give isl_ast_node *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  static inline ast_node alloc_user(ast_expr expr);
  inline ast_node_list block_get_children() const;
  inline ast_node for_get_body() const;
  inline ast_expr for_get_cond() const;
  inline ast_expr for_get_inc() const;
  inline ast_expr for_get_init() const;
  inline ast_expr for_get_iterator() const;
  inline boolean for_is_degenerate() const;
  inline id get_annotation() const;
  inline ast_expr if_get_cond() const;
  inline ast_node if_get_else() const;
  inline ast_node if_get_then() const;
  inline boolean if_has_else() const;
  inline id mark_get_id() const;
  inline ast_node mark_get_node() const;
  inline ast_node set_annotation(id annotation) const;
  inline std::string to_C_str() const;
  inline ast_expr user_get_expr() const;
};

// declarations for isl::ast_node_list
inline ast_node_list manage(__isl_take isl_ast_node_list *ptr);
inline ast_node_list manage_copy(__isl_keep isl_ast_node_list *ptr);

class ast_node_list {
  friend inline ast_node_list manage(__isl_take isl_ast_node_list *ptr);
  friend inline ast_node_list manage_copy(__isl_keep isl_ast_node_list *ptr);

  isl_ast_node_list *ptr = nullptr;

  inline explicit ast_node_list(__isl_take isl_ast_node_list *ptr);

public:
  inline /* implicit */ ast_node_list();
  inline /* implicit */ ast_node_list(const ast_node_list &obj);
  inline /* implicit */ ast_node_list(std::nullptr_t);
  inline ast_node_list &operator=(ast_node_list obj);
  inline ~ast_node_list();
  inline __isl_give isl_ast_node_list *copy() const &;
  inline __isl_give isl_ast_node_list *copy() && = delete;
  inline __isl_keep isl_ast_node_list *get() const;
  inline __isl_give isl_ast_node_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline ast_node_list add(ast_node el) const;
  static inline ast_node_list alloc(ctx ctx, int n);
  inline ast_node_list concat(ast_node_list list2) const;
  inline ast_node_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(ast_node)> &fn) const;
  static inline ast_node_list from_ast_node(ast_node el);
  inline ast_node get_ast_node(int index) const;
  inline ast_node get_at(int index) const;
  inline ast_node_list insert(unsigned int pos, ast_node el) const;
  inline int n_ast_node() const;
  inline ast_node_list reverse() const;
  inline ast_node_list set_ast_node(int index, ast_node el) const;
  inline int size() const;
  inline ast_node_list swap(unsigned int pos1, unsigned int pos2) const;
};

// declarations for isl::basic_map
inline basic_map manage(__isl_take isl_basic_map *ptr);
inline basic_map manage_copy(__isl_keep isl_basic_map *ptr);

class basic_map {
  friend inline basic_map manage(__isl_take isl_basic_map *ptr);
  friend inline basic_map manage_copy(__isl_keep isl_basic_map *ptr);

  isl_basic_map *ptr = nullptr;

  inline explicit basic_map(__isl_take isl_basic_map *ptr);

public:
  inline /* implicit */ basic_map();
  inline /* implicit */ basic_map(const basic_map &obj);
  inline /* implicit */ basic_map(std::nullptr_t);
  inline explicit basic_map(ctx ctx, const std::string &str);
  inline basic_map &operator=(basic_map obj);
  inline ~basic_map();
  inline __isl_give isl_basic_map *copy() const &;
  inline __isl_give isl_basic_map *copy() && = delete;
  inline __isl_keep isl_basic_map *get() const;
  inline __isl_give isl_basic_map *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline basic_map add_constraint(constraint constraint) const;
  inline basic_map add_dims(isl::dim type, unsigned int n) const;
  inline basic_map affine_hull() const;
  inline basic_map align_params(space model) const;
  inline basic_map apply_domain(basic_map bmap2) const;
  inline basic_map apply_range(basic_map bmap2) const;
  inline boolean can_curry() const;
  inline boolean can_uncurry() const;
  inline boolean can_zip() const;
  inline basic_map curry() const;
  inline basic_set deltas() const;
  inline basic_map deltas_map() const;
  inline basic_map detect_equalities() const;
  inline unsigned int dim(isl::dim type) const;
  inline basic_set domain() const;
  inline basic_map domain_map() const;
  inline basic_map domain_product(basic_map bmap2) const;
  inline basic_map drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline basic_map drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline basic_map drop_unused_params() const;
  inline basic_map eliminate(isl::dim type, unsigned int first, unsigned int n) const;
  static inline basic_map empty(space space);
  static inline basic_map equal(space dim, unsigned int n_equal);
  inline mat equalities_matrix(isl::dim c1, isl::dim c2, isl::dim c3, isl::dim c4, isl::dim c5) const;
  inline basic_map equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline basic_map fix_si(isl::dim type, unsigned int pos, int value) const;
  inline basic_map fix_val(isl::dim type, unsigned int pos, val v) const;
  inline basic_map flat_product(basic_map bmap2) const;
  inline basic_map flat_range_product(basic_map bmap2) const;
  inline basic_map flatten() const;
  inline basic_map flatten_domain() const;
  inline basic_map flatten_range() const;
  inline stat foreach_constraint(const std::function<stat(constraint)> &fn) const;
  static inline basic_map from_aff(aff aff);
  static inline basic_map from_aff_list(space domain_space, aff_list list);
  static inline basic_map from_constraint(constraint constraint);
  static inline basic_map from_domain(basic_set bset);
  static inline basic_map from_domain_and_range(basic_set domain, basic_set range);
  static inline basic_map from_multi_aff(multi_aff maff);
  static inline basic_map from_qpolynomial(qpolynomial qp);
  static inline basic_map from_range(basic_set bset);
  inline constraint_list get_constraint_list() const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline aff get_div(int pos) const;
  inline local_space get_local_space() const;
  inline space get_space() const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline basic_map gist(basic_map context) const;
  inline basic_map gist_domain(basic_set context) const;
  inline boolean has_dim_id(isl::dim type, unsigned int pos) const;
  static inline basic_map identity(space dim);
  inline boolean image_is_bounded() const;
  inline mat inequalities_matrix(isl::dim c1, isl::dim c2, isl::dim c3, isl::dim c4, isl::dim c5) const;
  inline basic_map insert_dims(isl::dim type, unsigned int pos, unsigned int n) const;
  inline basic_map intersect(basic_map bmap2) const;
  inline basic_map intersect_domain(basic_set bset) const;
  inline basic_map intersect_range(basic_set bset) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean is_disjoint(const basic_map &bmap2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const basic_map &bmap2) const;
  inline boolean is_rational() const;
  inline boolean is_single_valued() const;
  inline boolean is_strict_subset(const basic_map &bmap2) const;
  inline boolean is_subset(const basic_map &bmap2) const;
  inline boolean is_universe() const;
  static inline basic_map less_at(space dim, unsigned int pos);
  inline map lexmax() const;
  inline map lexmin() const;
  inline pw_multi_aff lexmin_pw_multi_aff() const;
  inline basic_map lower_bound_si(isl::dim type, unsigned int pos, int value) const;
  static inline basic_map more_at(space dim, unsigned int pos);
  inline basic_map move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline int n_constraint() const;
  static inline basic_map nat_universe(space dim);
  inline basic_map neg() const;
  inline basic_map order_ge(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline basic_map order_gt(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline val plain_get_val_if_fixed(isl::dim type, unsigned int pos) const;
  inline boolean plain_is_empty() const;
  inline boolean plain_is_universe() const;
  inline basic_map preimage_domain_multi_aff(multi_aff ma) const;
  inline basic_map preimage_range_multi_aff(multi_aff ma) const;
  inline basic_map product(basic_map bmap2) const;
  inline basic_map project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline basic_set range() const;
  inline basic_map range_map() const;
  inline basic_map range_product(basic_map bmap2) const;
  inline basic_map remove_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline basic_map remove_divs() const;
  inline basic_map remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline basic_map remove_redundancies() const;
  inline basic_map reverse() const;
  inline basic_map sample() const;
  inline basic_map set_tuple_id(isl::dim type, id id) const;
  inline basic_map set_tuple_name(isl::dim type, const std::string &s) const;
  inline basic_map sum(basic_map bmap2) const;
  inline basic_map uncurry() const;
  inline map unite(basic_map bmap2) const;
  static inline basic_map universe(space space);
  inline basic_map upper_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline basic_set wrap() const;
  inline basic_map zip() const;
};

// declarations for isl::basic_map_list
inline basic_map_list manage(__isl_take isl_basic_map_list *ptr);
inline basic_map_list manage_copy(__isl_keep isl_basic_map_list *ptr);

class basic_map_list {
  friend inline basic_map_list manage(__isl_take isl_basic_map_list *ptr);
  friend inline basic_map_list manage_copy(__isl_keep isl_basic_map_list *ptr);

  isl_basic_map_list *ptr = nullptr;

  inline explicit basic_map_list(__isl_take isl_basic_map_list *ptr);

public:
  inline /* implicit */ basic_map_list();
  inline /* implicit */ basic_map_list(const basic_map_list &obj);
  inline /* implicit */ basic_map_list(std::nullptr_t);
  inline basic_map_list &operator=(basic_map_list obj);
  inline ~basic_map_list();
  inline __isl_give isl_basic_map_list *copy() const &;
  inline __isl_give isl_basic_map_list *copy() && = delete;
  inline __isl_keep isl_basic_map_list *get() const;
  inline __isl_give isl_basic_map_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline basic_map_list add(basic_map el) const;
  static inline basic_map_list alloc(ctx ctx, int n);
  inline basic_map_list concat(basic_map_list list2) const;
  inline basic_map_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(basic_map)> &fn) const;
  static inline basic_map_list from_basic_map(basic_map el);
  inline basic_map get_at(int index) const;
  inline basic_map get_basic_map(int index) const;
  inline basic_map_list insert(unsigned int pos, basic_map el) const;
  inline int n_basic_map() const;
  inline basic_map_list reverse() const;
  inline basic_map_list set_basic_map(int index, basic_map el) const;
  inline int size() const;
  inline basic_map_list swap(unsigned int pos1, unsigned int pos2) const;
};

// declarations for isl::basic_set
inline basic_set manage(__isl_take isl_basic_set *ptr);
inline basic_set manage_copy(__isl_keep isl_basic_set *ptr);

class basic_set {
  friend inline basic_set manage(__isl_take isl_basic_set *ptr);
  friend inline basic_set manage_copy(__isl_keep isl_basic_set *ptr);

  isl_basic_set *ptr = nullptr;

  inline explicit basic_set(__isl_take isl_basic_set *ptr);

public:
  inline /* implicit */ basic_set();
  inline /* implicit */ basic_set(const basic_set &obj);
  inline /* implicit */ basic_set(std::nullptr_t);
  inline explicit basic_set(ctx ctx, const std::string &str);
  inline /* implicit */ basic_set(point pnt);
  inline basic_set &operator=(basic_set obj);
  inline ~basic_set();
  inline __isl_give isl_basic_set *copy() const &;
  inline __isl_give isl_basic_set *copy() && = delete;
  inline __isl_keep isl_basic_set *get() const;
  inline __isl_give isl_basic_set *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline basic_set affine_hull() const;
  inline basic_set align_params(space model) const;
  inline basic_set apply(basic_map bmap) const;
  static inline basic_set box_from_points(point pnt1, point pnt2);
  inline basic_set coefficients() const;
  inline basic_set detect_equalities() const;
  inline unsigned int dim(isl::dim type) const;
  inline val dim_max_val(int pos) const;
  inline basic_set drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline basic_set drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline basic_set drop_unused_params() const;
  inline basic_set eliminate(isl::dim type, unsigned int first, unsigned int n) const;
  static inline basic_set empty(space space);
  inline mat equalities_matrix(isl::dim c1, isl::dim c2, isl::dim c3, isl::dim c4) const;
  inline basic_set fix_si(isl::dim type, unsigned int pos, int value) const;
  inline basic_set fix_val(isl::dim type, unsigned int pos, val v) const;
  inline basic_set flat_product(basic_set bset2) const;
  inline basic_set flatten() const;
  inline stat foreach_bound_pair(isl::dim type, unsigned int pos, const std::function<stat(constraint, constraint, basic_set)> &fn) const;
  inline stat foreach_constraint(const std::function<stat(constraint)> &fn) const;
  static inline basic_set from_constraint(constraint constraint);
  static inline basic_set from_multi_aff(multi_aff ma);
  inline basic_set from_params() const;
  inline constraint_list get_constraint_list() const;
  inline id get_dim_id(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline aff get_div(int pos) const;
  inline local_space get_local_space() const;
  inline space get_space() const;
  inline std::string get_tuple_name() const;
  inline basic_set gist(basic_set context) const;
  inline mat inequalities_matrix(isl::dim c1, isl::dim c2, isl::dim c3, isl::dim c4) const;
  inline basic_set insert_dims(isl::dim type, unsigned int pos, unsigned int n) const;
  inline basic_set intersect(basic_set bset2) const;
  inline basic_set intersect_params(basic_set bset2) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean is_bounded() const;
  inline boolean is_disjoint(const basic_set &bset2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const basic_set &bset2) const;
  inline int is_rational() const;
  inline boolean is_subset(const basic_set &bset2) const;
  inline boolean is_universe() const;
  inline boolean is_wrapping() const;
  inline set lexmax() const;
  inline set lexmin() const;
  inline basic_set lower_bound_val(isl::dim type, unsigned int pos, val value) const;
  inline val max_val(const aff &obj) const;
  inline basic_set move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline int n_constraint() const;
  inline unsigned int n_dim() const;
  static inline basic_set nat_universe(space dim);
  inline basic_set neg() const;
  inline basic_set params() const;
  inline boolean plain_is_empty() const;
  inline boolean plain_is_equal(const basic_set &bset2) const;
  inline boolean plain_is_universe() const;
  static inline basic_set positive_orthant(space space);
  inline basic_set preimage_multi_aff(multi_aff ma) const;
  inline basic_set project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline mat reduced_basis() const;
  inline basic_set remove_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline basic_set remove_divs() const;
  inline basic_set remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline basic_set remove_redundancies() const;
  inline basic_set remove_unknown_divs() const;
  inline basic_set sample() const;
  inline point sample_point() const;
  inline basic_set set_tuple_id(id id) const;
  inline basic_set set_tuple_name(const std::string &s) const;
  inline basic_set solutions() const;
  inline set unite(basic_set bset2) const;
  static inline basic_set universe(space space);
  inline basic_map unwrap() const;
  inline basic_set upper_bound_val(isl::dim type, unsigned int pos, val value) const;
};

// declarations for isl::basic_set_list
inline basic_set_list manage(__isl_take isl_basic_set_list *ptr);
inline basic_set_list manage_copy(__isl_keep isl_basic_set_list *ptr);

class basic_set_list {
  friend inline basic_set_list manage(__isl_take isl_basic_set_list *ptr);
  friend inline basic_set_list manage_copy(__isl_keep isl_basic_set_list *ptr);

  isl_basic_set_list *ptr = nullptr;

  inline explicit basic_set_list(__isl_take isl_basic_set_list *ptr);

public:
  inline /* implicit */ basic_set_list();
  inline /* implicit */ basic_set_list(const basic_set_list &obj);
  inline /* implicit */ basic_set_list(std::nullptr_t);
  inline basic_set_list &operator=(basic_set_list obj);
  inline ~basic_set_list();
  inline __isl_give isl_basic_set_list *copy() const &;
  inline __isl_give isl_basic_set_list *copy() && = delete;
  inline __isl_keep isl_basic_set_list *get() const;
  inline __isl_give isl_basic_set_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline basic_set_list add(basic_set el) const;
  static inline basic_set_list alloc(ctx ctx, int n);
  inline basic_set_list coefficients() const;
  inline basic_set_list concat(basic_set_list list2) const;
  inline basic_set_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(basic_set)> &fn) const;
  static inline basic_set_list from_basic_set(basic_set el);
  inline basic_set get_at(int index) const;
  inline basic_set get_basic_set(int index) const;
  inline basic_set_list insert(unsigned int pos, basic_set el) const;
  inline int n_basic_set() const;
  inline basic_set_list reverse() const;
  inline basic_set_list set_basic_set(int index, basic_set el) const;
  inline int size() const;
  inline basic_set_list swap(unsigned int pos1, unsigned int pos2) const;
};

// declarations for isl::constraint
inline constraint manage(__isl_take isl_constraint *ptr);
inline constraint manage_copy(__isl_keep isl_constraint *ptr);

class constraint {
  friend inline constraint manage(__isl_take isl_constraint *ptr);
  friend inline constraint manage_copy(__isl_keep isl_constraint *ptr);

  isl_constraint *ptr = nullptr;

  inline explicit constraint(__isl_take isl_constraint *ptr);

public:
  inline /* implicit */ constraint();
  inline /* implicit */ constraint(const constraint &obj);
  inline /* implicit */ constraint(std::nullptr_t);
  inline constraint &operator=(constraint obj);
  inline ~constraint();
  inline __isl_give isl_constraint *copy() const &;
  inline __isl_give isl_constraint *copy() && = delete;
  inline __isl_keep isl_constraint *get() const;
  inline __isl_give isl_constraint *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  static inline constraint alloc_equality(local_space ls);
  static inline constraint alloc_inequality(local_space ls);
  inline int cmp_last_non_zero(const constraint &c2) const;
  inline aff get_aff() const;
  inline aff get_bound(isl::dim type, int pos) const;
  inline val get_coefficient_val(isl::dim type, int pos) const;
  inline val get_constant_val() const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline aff get_div(int pos) const;
  inline local_space get_local_space() const;
  inline space get_space() const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline int is_div_constraint() const;
  inline boolean is_lower_bound(isl::dim type, unsigned int pos) const;
  inline boolean is_upper_bound(isl::dim type, unsigned int pos) const;
  inline int plain_cmp(const constraint &c2) const;
  inline constraint set_coefficient_si(isl::dim type, int pos, int v) const;
  inline constraint set_coefficient_val(isl::dim type, int pos, val v) const;
  inline constraint set_constant_si(int v) const;
  inline constraint set_constant_val(val v) const;
};

// declarations for isl::constraint_list
inline constraint_list manage(__isl_take isl_constraint_list *ptr);
inline constraint_list manage_copy(__isl_keep isl_constraint_list *ptr);

class constraint_list {
  friend inline constraint_list manage(__isl_take isl_constraint_list *ptr);
  friend inline constraint_list manage_copy(__isl_keep isl_constraint_list *ptr);

  isl_constraint_list *ptr = nullptr;

  inline explicit constraint_list(__isl_take isl_constraint_list *ptr);

public:
  inline /* implicit */ constraint_list();
  inline /* implicit */ constraint_list(const constraint_list &obj);
  inline /* implicit */ constraint_list(std::nullptr_t);
  inline constraint_list &operator=(constraint_list obj);
  inline ~constraint_list();
  inline __isl_give isl_constraint_list *copy() const &;
  inline __isl_give isl_constraint_list *copy() && = delete;
  inline __isl_keep isl_constraint_list *get() const;
  inline __isl_give isl_constraint_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline constraint_list add(constraint el) const;
  static inline constraint_list alloc(ctx ctx, int n);
  inline constraint_list concat(constraint_list list2) const;
  inline constraint_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(constraint)> &fn) const;
  static inline constraint_list from_constraint(constraint el);
  inline constraint get_at(int index) const;
  inline constraint get_constraint(int index) const;
  inline constraint_list insert(unsigned int pos, constraint el) const;
  inline int n_constraint() const;
  inline constraint_list reverse() const;
  inline constraint_list set_constraint(int index, constraint el) const;
  inline int size() const;
  inline constraint_list swap(unsigned int pos1, unsigned int pos2) const;
};

// declarations for isl::fixed_box
inline fixed_box manage(__isl_take isl_fixed_box *ptr);
inline fixed_box manage_copy(__isl_keep isl_fixed_box *ptr);

class fixed_box {
  friend inline fixed_box manage(__isl_take isl_fixed_box *ptr);
  friend inline fixed_box manage_copy(__isl_keep isl_fixed_box *ptr);

  isl_fixed_box *ptr = nullptr;

  inline explicit fixed_box(__isl_take isl_fixed_box *ptr);

public:
  inline /* implicit */ fixed_box();
  inline /* implicit */ fixed_box(const fixed_box &obj);
  inline /* implicit */ fixed_box(std::nullptr_t);
  inline fixed_box &operator=(fixed_box obj);
  inline ~fixed_box();
  inline __isl_give isl_fixed_box *copy() const &;
  inline __isl_give isl_fixed_box *copy() && = delete;
  inline __isl_keep isl_fixed_box *get() const;
  inline __isl_give isl_fixed_box *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

  inline multi_aff get_offset() const;
  inline multi_val get_size() const;
  inline space get_space() const;
  inline boolean is_valid() const;
};

// declarations for isl::id
inline id manage(__isl_take isl_id *ptr);
inline id manage_copy(__isl_keep isl_id *ptr);

class id {
  friend inline id manage(__isl_take isl_id *ptr);
  friend inline id manage_copy(__isl_keep isl_id *ptr);

  isl_id *ptr = nullptr;

  inline explicit id(__isl_take isl_id *ptr);

public:
  inline /* implicit */ id();
  inline /* implicit */ id(const id &obj);
  inline /* implicit */ id(std::nullptr_t);
  inline id &operator=(id obj);
  inline ~id();
  inline __isl_give isl_id *copy() const &;
  inline __isl_give isl_id *copy() && = delete;
  inline __isl_keep isl_id *get() const;
  inline __isl_give isl_id *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  static inline id alloc(ctx ctx, const std::string &name, void * user);
  inline uint32_t get_hash() const;
  inline std::string get_name() const;
  inline void * get_user() const;
};

// declarations for isl::id_list
inline id_list manage(__isl_take isl_id_list *ptr);
inline id_list manage_copy(__isl_keep isl_id_list *ptr);

class id_list {
  friend inline id_list manage(__isl_take isl_id_list *ptr);
  friend inline id_list manage_copy(__isl_keep isl_id_list *ptr);

  isl_id_list *ptr = nullptr;

  inline explicit id_list(__isl_take isl_id_list *ptr);

public:
  inline /* implicit */ id_list();
  inline /* implicit */ id_list(const id_list &obj);
  inline /* implicit */ id_list(std::nullptr_t);
  inline id_list &operator=(id_list obj);
  inline ~id_list();
  inline __isl_give isl_id_list *copy() const &;
  inline __isl_give isl_id_list *copy() && = delete;
  inline __isl_keep isl_id_list *get() const;
  inline __isl_give isl_id_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline id_list add(id el) const;
  static inline id_list alloc(ctx ctx, int n);
  inline id_list concat(id_list list2) const;
  inline id_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(id)> &fn) const;
  static inline id_list from_id(id el);
  inline id get_at(int index) const;
  inline id get_id(int index) const;
  inline id_list insert(unsigned int pos, id el) const;
  inline int n_id() const;
  inline id_list reverse() const;
  inline id_list set_id(int index, id el) const;
  inline int size() const;
  inline id_list swap(unsigned int pos1, unsigned int pos2) const;
};

// declarations for isl::id_to_ast_expr
inline id_to_ast_expr manage(__isl_take isl_id_to_ast_expr *ptr);
inline id_to_ast_expr manage_copy(__isl_keep isl_id_to_ast_expr *ptr);

class id_to_ast_expr {
  friend inline id_to_ast_expr manage(__isl_take isl_id_to_ast_expr *ptr);
  friend inline id_to_ast_expr manage_copy(__isl_keep isl_id_to_ast_expr *ptr);

  isl_id_to_ast_expr *ptr = nullptr;

  inline explicit id_to_ast_expr(__isl_take isl_id_to_ast_expr *ptr);

public:
  inline /* implicit */ id_to_ast_expr();
  inline /* implicit */ id_to_ast_expr(const id_to_ast_expr &obj);
  inline /* implicit */ id_to_ast_expr(std::nullptr_t);
  inline id_to_ast_expr &operator=(id_to_ast_expr obj);
  inline ~id_to_ast_expr();
  inline __isl_give isl_id_to_ast_expr *copy() const &;
  inline __isl_give isl_id_to_ast_expr *copy() && = delete;
  inline __isl_keep isl_id_to_ast_expr *get() const;
  inline __isl_give isl_id_to_ast_expr *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  static inline id_to_ast_expr alloc(ctx ctx, int min_size);
  inline id_to_ast_expr drop(id key) const;
  inline stat foreach(const std::function<stat(id, ast_expr)> &fn) const;
  inline ast_expr get(id key) const;
  inline boolean has(const id &key) const;
  inline id_to_ast_expr set(id key, ast_expr val) const;
};

// declarations for isl::local_space
inline local_space manage(__isl_take isl_local_space *ptr);
inline local_space manage_copy(__isl_keep isl_local_space *ptr);

class local_space {
  friend inline local_space manage(__isl_take isl_local_space *ptr);
  friend inline local_space manage_copy(__isl_keep isl_local_space *ptr);

  isl_local_space *ptr = nullptr;

  inline explicit local_space(__isl_take isl_local_space *ptr);

public:
  inline /* implicit */ local_space();
  inline /* implicit */ local_space(const local_space &obj);
  inline /* implicit */ local_space(std::nullptr_t);
  inline explicit local_space(space dim);
  inline local_space &operator=(local_space obj);
  inline ~local_space();
  inline __isl_give isl_local_space *copy() const &;
  inline __isl_give isl_local_space *copy() && = delete;
  inline __isl_keep isl_local_space *get() const;
  inline __isl_give isl_local_space *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline local_space add_dims(isl::dim type, unsigned int n) const;
  inline int dim(isl::dim type) const;
  inline local_space domain() const;
  inline local_space drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline local_space flatten_domain() const;
  inline local_space flatten_range() const;
  inline local_space from_domain() const;
  inline id get_dim_id(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline aff get_div(int pos) const;
  inline space get_space() const;
  inline boolean has_dim_id(isl::dim type, unsigned int pos) const;
  inline boolean has_dim_name(isl::dim type, unsigned int pos) const;
  inline local_space insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline local_space intersect(local_space ls2) const;
  inline boolean is_equal(const local_space &ls2) const;
  inline boolean is_params() const;
  inline boolean is_set() const;
  inline local_space range() const;
  inline local_space set_dim_id(isl::dim type, unsigned int pos, id id) const;
  inline local_space set_from_params() const;
  inline local_space set_tuple_id(isl::dim type, id id) const;
  inline local_space wrap() const;
};

// declarations for isl::map
inline map manage(__isl_take isl_map *ptr);
inline map manage_copy(__isl_keep isl_map *ptr);

class map {
  friend inline map manage(__isl_take isl_map *ptr);
  friend inline map manage_copy(__isl_keep isl_map *ptr);

  isl_map *ptr = nullptr;

  inline explicit map(__isl_take isl_map *ptr);

public:
  inline /* implicit */ map();
  inline /* implicit */ map(const map &obj);
  inline /* implicit */ map(std::nullptr_t);
  inline explicit map(ctx ctx, const std::string &str);
  inline /* implicit */ map(basic_map bmap);
  inline map &operator=(map obj);
  inline ~map();
  inline __isl_give isl_map *copy() const &;
  inline __isl_give isl_map *copy() && = delete;
  inline __isl_keep isl_map *get() const;
  inline __isl_give isl_map *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline map add_constraint(constraint constraint) const;
  inline map add_dims(isl::dim type, unsigned int n) const;
  inline basic_map affine_hull() const;
  inline map align_params(space model) const;
  inline map apply_domain(map map2) const;
  inline map apply_range(map map2) const;
  inline boolean can_curry() const;
  inline boolean can_range_curry() const;
  inline boolean can_uncurry() const;
  inline boolean can_zip() const;
  inline map coalesce() const;
  inline map complement() const;
  inline basic_map convex_hull() const;
  inline map curry() const;
  inline set deltas() const;
  inline map deltas_map() const;
  inline map detect_equalities() const;
  inline unsigned int dim(isl::dim type) const;
  inline pw_aff dim_max(int pos) const;
  inline pw_aff dim_min(int pos) const;
  inline set domain() const;
  inline map domain_factor_domain() const;
  inline map domain_factor_range() const;
  inline boolean domain_is_wrapping() const;
  inline map domain_map() const;
  inline map domain_product(map map2) const;
  inline map drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline map drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline map drop_unused_params() const;
  inline map eliminate(isl::dim type, unsigned int first, unsigned int n) const;
  static inline map empty(space space);
  inline map equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline map factor_domain() const;
  inline map factor_range() const;
  inline int find_dim_by_id(isl::dim type, const id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline map fix_si(isl::dim type, unsigned int pos, int value) const;
  inline map fix_val(isl::dim type, unsigned int pos, val v) const;
  inline map fixed_power_val(val exp) const;
  inline map flat_domain_product(map map2) const;
  inline map flat_product(map map2) const;
  inline map flat_range_product(map map2) const;
  inline map flatten() const;
  inline map flatten_domain() const;
  inline map flatten_range() const;
  inline map floordiv_val(val d) const;
  inline stat foreach_basic_map(const std::function<stat(basic_map)> &fn) const;
  static inline map from_aff(aff aff);
  static inline map from_domain(set set);
  static inline map from_domain_and_range(set domain, set range);
  static inline map from_multi_aff(multi_aff maff);
  static inline map from_multi_pw_aff(multi_pw_aff mpa);
  static inline map from_pw_aff(pw_aff pwaff);
  static inline map from_pw_multi_aff(pw_multi_aff pma);
  static inline map from_range(set set);
  static inline map from_union_map(union_map umap);
  inline basic_map_list get_basic_map_list() const;
  inline id get_dim_id(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline uint32_t get_hash() const;
  inline fixed_box get_range_simple_fixed_box_hull() const;
  inline space get_space() const;
  inline id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline map gist(map context) const;
  inline map gist_basic_map(basic_map context) const;
  inline map gist_domain(set context) const;
  inline map gist_params(set context) const;
  inline map gist_range(set context) const;
  inline boolean has_dim_id(isl::dim type, unsigned int pos) const;
  inline boolean has_dim_name(isl::dim type, unsigned int pos) const;
  inline boolean has_equal_space(const map &map2) const;
  inline boolean has_tuple_id(isl::dim type) const;
  inline boolean has_tuple_name(isl::dim type) const;
  static inline map identity(space dim);
  inline map insert_dims(isl::dim type, unsigned int pos, unsigned int n) const;
  inline map intersect(map map2) const;
  inline map intersect_domain(set set) const;
  inline map intersect_domain_factor_range(map factor) const;
  inline map intersect_params(set params) const;
  inline map intersect_range(set set) const;
  inline map intersect_range_factor_range(map factor) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean is_bijective() const;
  inline boolean is_disjoint(const map &map2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const map &map2) const;
  inline boolean is_identity() const;
  inline boolean is_injective() const;
  inline boolean is_product() const;
  inline boolean is_single_valued() const;
  inline boolean is_strict_subset(const map &map2) const;
  inline boolean is_subset(const map &map2) const;
  inline int is_translation() const;
  static inline map lex_ge(space set_dim);
  static inline map lex_ge_first(space dim, unsigned int n);
  inline map lex_ge_map(map map2) const;
  static inline map lex_gt(space set_dim);
  static inline map lex_gt_first(space dim, unsigned int n);
  inline map lex_gt_map(map map2) const;
  static inline map lex_le(space set_dim);
  static inline map lex_le_first(space dim, unsigned int n);
  inline map lex_le_map(map map2) const;
  static inline map lex_lt(space set_dim);
  static inline map lex_lt_first(space dim, unsigned int n);
  inline map lex_lt_map(map map2) const;
  inline map lexmax() const;
  inline pw_multi_aff lexmax_pw_multi_aff() const;
  inline map lexmin() const;
  inline pw_multi_aff lexmin_pw_multi_aff() const;
  inline map lower_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline map move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline int n_basic_map() const;
  static inline map nat_universe(space dim);
  inline map neg() const;
  inline map oppose(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline map order_ge(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline map order_gt(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline map order_le(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline map order_lt(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline set params() const;
  inline val plain_get_val_if_fixed(isl::dim type, unsigned int pos) const;
  inline boolean plain_is_empty() const;
  inline boolean plain_is_equal(const map &map2) const;
  inline boolean plain_is_injective() const;
  inline boolean plain_is_single_valued() const;
  inline boolean plain_is_universe() const;
  inline basic_map plain_unshifted_simple_hull() const;
  inline basic_map polyhedral_hull() const;
  inline map preimage_domain_multi_aff(multi_aff ma) const;
  inline map preimage_domain_multi_pw_aff(multi_pw_aff mpa) const;
  inline map preimage_domain_pw_multi_aff(pw_multi_aff pma) const;
  inline map preimage_range_multi_aff(multi_aff ma) const;
  inline map preimage_range_pw_multi_aff(pw_multi_aff pma) const;
  inline map product(map map2) const;
  inline map project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline set range() const;
  inline map range_curry() const;
  inline map range_factor_domain() const;
  inline map range_factor_range() const;
  inline boolean range_is_wrapping() const;
  inline map range_map() const;
  inline map range_product(map map2) const;
  inline map remove_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline map remove_divs() const;
  inline map remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline map remove_redundancies() const;
  inline map remove_unknown_divs() const;
  inline map reset_tuple_id(isl::dim type) const;
  inline map reset_user() const;
  inline map reverse() const;
  inline basic_map sample() const;
  inline map set_dim_id(isl::dim type, unsigned int pos, id id) const;
  inline map set_tuple_id(isl::dim type, id id) const;
  inline map set_tuple_name(isl::dim type, const std::string &s) const;
  inline basic_map simple_hull() const;
  inline map subtract(map map2) const;
  inline map subtract_domain(set dom) const;
  inline map subtract_range(set dom) const;
  inline map sum(map map2) const;
  inline map uncurry() const;
  inline map unite(map map2) const;
  static inline map universe(space space);
  inline basic_map unshifted_simple_hull() const;
  inline basic_map unshifted_simple_hull_from_map_list(map_list list) const;
  inline map upper_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline set wrap() const;
  inline map zip() const;
};

// declarations for isl::map_list
inline map_list manage(__isl_take isl_map_list *ptr);
inline map_list manage_copy(__isl_keep isl_map_list *ptr);

class map_list {
  friend inline map_list manage(__isl_take isl_map_list *ptr);
  friend inline map_list manage_copy(__isl_keep isl_map_list *ptr);

  isl_map_list *ptr = nullptr;

  inline explicit map_list(__isl_take isl_map_list *ptr);

public:
  inline /* implicit */ map_list();
  inline /* implicit */ map_list(const map_list &obj);
  inline /* implicit */ map_list(std::nullptr_t);
  inline map_list &operator=(map_list obj);
  inline ~map_list();
  inline __isl_give isl_map_list *copy() const &;
  inline __isl_give isl_map_list *copy() && = delete;
  inline __isl_keep isl_map_list *get() const;
  inline __isl_give isl_map_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline map_list add(map el) const;
  static inline map_list alloc(ctx ctx, int n);
  inline map_list concat(map_list list2) const;
  inline map_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(map)> &fn) const;
  static inline map_list from_map(map el);
  inline map get_at(int index) const;
  inline map get_map(int index) const;
  inline map_list insert(unsigned int pos, map el) const;
  inline int n_map() const;
  inline map_list reverse() const;
  inline map_list set_map(int index, map el) const;
  inline int size() const;
  inline map_list swap(unsigned int pos1, unsigned int pos2) const;
};

// declarations for isl::mat
inline mat manage(__isl_take isl_mat *ptr);
inline mat manage_copy(__isl_keep isl_mat *ptr);

class mat {
  friend inline mat manage(__isl_take isl_mat *ptr);
  friend inline mat manage_copy(__isl_keep isl_mat *ptr);

  isl_mat *ptr = nullptr;

  inline explicit mat(__isl_take isl_mat *ptr);

public:
  inline /* implicit */ mat();
  inline /* implicit */ mat(const mat &obj);
  inline /* implicit */ mat(std::nullptr_t);
  inline mat &operator=(mat obj);
  inline ~mat();
  inline __isl_give isl_mat *copy() const &;
  inline __isl_give isl_mat *copy() && = delete;
  inline __isl_keep isl_mat *get() const;
  inline __isl_give isl_mat *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline mat add_rows(unsigned int n) const;
  inline mat add_zero_cols(unsigned int n) const;
  inline mat add_zero_rows(unsigned int n) const;
  inline mat aff_direct_sum(mat right) const;
  static inline mat alloc(ctx ctx, unsigned int n_row, unsigned int n_col);
  inline int cols() const;
  inline mat concat(mat bot) const;
  inline mat diagonal(mat mat2) const;
  inline mat drop_cols(unsigned int col, unsigned int n) const;
  inline mat drop_rows(unsigned int row, unsigned int n) const;
  static inline mat from_row_vec(vec vec);
  inline val get_element_val(int row, int col) const;
  inline boolean has_linearly_independent_rows(const mat &mat2) const;
  inline int initial_non_zero_cols() const;
  inline mat insert_cols(unsigned int col, unsigned int n) const;
  inline mat insert_rows(unsigned int row, unsigned int n) const;
  inline mat insert_zero_cols(unsigned int first, unsigned int n) const;
  inline mat insert_zero_rows(unsigned int row, unsigned int n) const;
  inline mat inverse_product(mat right) const;
  inline boolean is_equal(const mat &mat2) const;
  inline mat lin_to_aff() const;
  inline mat move_cols(unsigned int dst_col, unsigned int src_col, unsigned int n) const;
  inline mat normalize() const;
  inline mat normalize_row(int row) const;
  inline mat product(mat right) const;
  inline int rank() const;
  inline mat right_inverse() const;
  inline mat right_kernel() const;
  inline mat row_basis() const;
  inline mat row_basis_extension(mat mat2) const;
  inline int rows() const;
  inline mat set_element_si(int row, int col, int v) const;
  inline mat set_element_val(int row, int col, val v) const;
  inline mat swap_cols(unsigned int i, unsigned int j) const;
  inline mat swap_rows(unsigned int i, unsigned int j) const;
  inline mat transpose() const;
  inline mat unimodular_complete(int row) const;
  inline mat vec_concat(vec bot) const;
  inline vec vec_inverse_product(vec vec) const;
  inline vec vec_product(vec vec) const;
};

// declarations for isl::multi_aff
inline multi_aff manage(__isl_take isl_multi_aff *ptr);
inline multi_aff manage_copy(__isl_keep isl_multi_aff *ptr);

class multi_aff {
  friend inline multi_aff manage(__isl_take isl_multi_aff *ptr);
  friend inline multi_aff manage_copy(__isl_keep isl_multi_aff *ptr);

  isl_multi_aff *ptr = nullptr;

  inline explicit multi_aff(__isl_take isl_multi_aff *ptr);

public:
  inline /* implicit */ multi_aff();
  inline /* implicit */ multi_aff(const multi_aff &obj);
  inline /* implicit */ multi_aff(std::nullptr_t);
  inline /* implicit */ multi_aff(aff aff);
  inline explicit multi_aff(ctx ctx, const std::string &str);
  inline multi_aff &operator=(multi_aff obj);
  inline ~multi_aff();
  inline __isl_give isl_multi_aff *copy() const &;
  inline __isl_give isl_multi_aff *copy() && = delete;
  inline __isl_keep isl_multi_aff *get() const;
  inline __isl_give isl_multi_aff *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline multi_aff add(multi_aff multi2) const;
  inline multi_aff add_dims(isl::dim type, unsigned int n) const;
  inline multi_aff align_params(space model) const;
  inline unsigned int dim(isl::dim type) const;
  static inline multi_aff domain_map(space space);
  inline multi_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline multi_aff factor_range() const;
  inline int find_dim_by_id(isl::dim type, const id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline multi_aff flat_range_product(multi_aff multi2) const;
  inline multi_aff flatten_domain() const;
  inline multi_aff flatten_range() const;
  inline multi_aff floor() const;
  static inline multi_aff from_aff_list(space space, aff_list list);
  inline multi_aff from_range() const;
  inline aff get_aff(int pos) const;
  inline id get_dim_id(isl::dim type, unsigned int pos) const;
  inline space get_domain_space() const;
  inline space get_space() const;
  inline id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline multi_aff gist(set context) const;
  inline multi_aff gist_params(set context) const;
  inline boolean has_tuple_id(isl::dim type) const;
  static inline multi_aff identity(space space);
  inline multi_aff insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_nan() const;
  inline set lex_ge_set(multi_aff ma2) const;
  inline set lex_gt_set(multi_aff ma2) const;
  inline set lex_le_set(multi_aff ma2) const;
  inline set lex_lt_set(multi_aff ma2) const;
  inline multi_aff mod_multi_val(multi_val mv) const;
  inline multi_aff move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  static inline multi_aff multi_val_on_space(space space, multi_val mv);
  inline multi_aff neg() const;
  inline int plain_cmp(const multi_aff &multi2) const;
  inline boolean plain_is_equal(const multi_aff &multi2) const;
  inline multi_aff product(multi_aff multi2) const;
  inline multi_aff project_domain_on_params() const;
  static inline multi_aff project_out_map(space space, isl::dim type, unsigned int first, unsigned int n);
  inline multi_aff pullback(multi_aff ma2) const;
  inline multi_aff range_factor_domain() const;
  inline multi_aff range_factor_range() const;
  inline boolean range_is_wrapping() const;
  static inline multi_aff range_map(space space);
  inline multi_aff range_product(multi_aff multi2) const;
  inline multi_aff range_splice(unsigned int pos, multi_aff multi2) const;
  inline multi_aff reset_tuple_id(isl::dim type) const;
  inline multi_aff reset_user() const;
  inline multi_aff scale_down_multi_val(multi_val mv) const;
  inline multi_aff scale_down_val(val v) const;
  inline multi_aff scale_multi_val(multi_val mv) const;
  inline multi_aff scale_val(val v) const;
  inline multi_aff set_aff(int pos, aff el) const;
  inline multi_aff set_dim_id(isl::dim type, unsigned int pos, id id) const;
  inline multi_aff set_tuple_id(isl::dim type, id id) const;
  inline multi_aff set_tuple_name(isl::dim type, const std::string &s) const;
  inline multi_aff splice(unsigned int in_pos, unsigned int out_pos, multi_aff multi2) const;
  inline multi_aff sub(multi_aff multi2) const;
  static inline multi_aff zero(space space);
};

// declarations for isl::multi_pw_aff
inline multi_pw_aff manage(__isl_take isl_multi_pw_aff *ptr);
inline multi_pw_aff manage_copy(__isl_keep isl_multi_pw_aff *ptr);

class multi_pw_aff {
  friend inline multi_pw_aff manage(__isl_take isl_multi_pw_aff *ptr);
  friend inline multi_pw_aff manage_copy(__isl_keep isl_multi_pw_aff *ptr);

  isl_multi_pw_aff *ptr = nullptr;

  inline explicit multi_pw_aff(__isl_take isl_multi_pw_aff *ptr);

public:
  inline /* implicit */ multi_pw_aff();
  inline /* implicit */ multi_pw_aff(const multi_pw_aff &obj);
  inline /* implicit */ multi_pw_aff(std::nullptr_t);
  inline /* implicit */ multi_pw_aff(multi_aff ma);
  inline /* implicit */ multi_pw_aff(pw_aff pa);
  inline /* implicit */ multi_pw_aff(pw_multi_aff pma);
  inline explicit multi_pw_aff(ctx ctx, const std::string &str);
  inline multi_pw_aff &operator=(multi_pw_aff obj);
  inline ~multi_pw_aff();
  inline __isl_give isl_multi_pw_aff *copy() const &;
  inline __isl_give isl_multi_pw_aff *copy() && = delete;
  inline __isl_keep isl_multi_pw_aff *get() const;
  inline __isl_give isl_multi_pw_aff *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline multi_pw_aff add(multi_pw_aff multi2) const;
  inline multi_pw_aff add_dims(isl::dim type, unsigned int n) const;
  inline multi_pw_aff align_params(space model) const;
  inline multi_pw_aff coalesce() const;
  inline unsigned int dim(isl::dim type) const;
  inline set domain() const;
  inline multi_pw_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline map eq_map(multi_pw_aff mpa2) const;
  inline multi_pw_aff factor_range() const;
  inline int find_dim_by_id(isl::dim type, const id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline multi_pw_aff flat_range_product(multi_pw_aff multi2) const;
  inline multi_pw_aff flatten_range() const;
  static inline multi_pw_aff from_pw_aff_list(space space, pw_aff_list list);
  inline multi_pw_aff from_range() const;
  inline id get_dim_id(isl::dim type, unsigned int pos) const;
  inline space get_domain_space() const;
  inline uint32_t get_hash() const;
  inline pw_aff get_pw_aff(int pos) const;
  inline space get_space() const;
  inline id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline multi_pw_aff gist(set set) const;
  inline multi_pw_aff gist_params(set set) const;
  inline boolean has_tuple_id(isl::dim type) const;
  static inline multi_pw_aff identity(space space);
  inline multi_pw_aff insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline multi_pw_aff intersect_domain(set domain) const;
  inline multi_pw_aff intersect_params(set set) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_nan() const;
  inline boolean is_cst() const;
  inline boolean is_equal(const multi_pw_aff &mpa2) const;
  inline map lex_gt_map(multi_pw_aff mpa2) const;
  inline map lex_lt_map(multi_pw_aff mpa2) const;
  inline multi_pw_aff mod_multi_val(multi_val mv) const;
  inline multi_pw_aff move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline multi_pw_aff neg() const;
  inline boolean plain_is_equal(const multi_pw_aff &multi2) const;
  inline multi_pw_aff product(multi_pw_aff multi2) const;
  inline multi_pw_aff project_domain_on_params() const;
  inline multi_pw_aff pullback(multi_aff ma) const;
  inline multi_pw_aff pullback(pw_multi_aff pma) const;
  inline multi_pw_aff pullback(multi_pw_aff mpa2) const;
  inline multi_pw_aff range_factor_domain() const;
  inline multi_pw_aff range_factor_range() const;
  inline boolean range_is_wrapping() const;
  inline multi_pw_aff range_product(multi_pw_aff multi2) const;
  inline multi_pw_aff range_splice(unsigned int pos, multi_pw_aff multi2) const;
  inline multi_pw_aff reset_tuple_id(isl::dim type) const;
  inline multi_pw_aff reset_user() const;
  inline multi_pw_aff scale_down_multi_val(multi_val mv) const;
  inline multi_pw_aff scale_down_val(val v) const;
  inline multi_pw_aff scale_multi_val(multi_val mv) const;
  inline multi_pw_aff scale_val(val v) const;
  inline multi_pw_aff set_dim_id(isl::dim type, unsigned int pos, id id) const;
  inline multi_pw_aff set_pw_aff(int pos, pw_aff el) const;
  inline multi_pw_aff set_tuple_id(isl::dim type, id id) const;
  inline multi_pw_aff set_tuple_name(isl::dim type, const std::string &s) const;
  inline multi_pw_aff splice(unsigned int in_pos, unsigned int out_pos, multi_pw_aff multi2) const;
  inline multi_pw_aff sub(multi_pw_aff multi2) const;
  static inline multi_pw_aff zero(space space);
};

// declarations for isl::multi_union_pw_aff
inline multi_union_pw_aff manage(__isl_take isl_multi_union_pw_aff *ptr);
inline multi_union_pw_aff manage_copy(__isl_keep isl_multi_union_pw_aff *ptr);

class multi_union_pw_aff {
  friend inline multi_union_pw_aff manage(__isl_take isl_multi_union_pw_aff *ptr);
  friend inline multi_union_pw_aff manage_copy(__isl_keep isl_multi_union_pw_aff *ptr);

  isl_multi_union_pw_aff *ptr = nullptr;

  inline explicit multi_union_pw_aff(__isl_take isl_multi_union_pw_aff *ptr);

public:
  inline /* implicit */ multi_union_pw_aff();
  inline /* implicit */ multi_union_pw_aff(const multi_union_pw_aff &obj);
  inline /* implicit */ multi_union_pw_aff(std::nullptr_t);
  inline /* implicit */ multi_union_pw_aff(union_pw_aff upa);
  inline /* implicit */ multi_union_pw_aff(multi_pw_aff mpa);
  inline explicit multi_union_pw_aff(union_pw_multi_aff upma);
  inline explicit multi_union_pw_aff(ctx ctx, const std::string &str);
  inline multi_union_pw_aff &operator=(multi_union_pw_aff obj);
  inline ~multi_union_pw_aff();
  inline __isl_give isl_multi_union_pw_aff *copy() const &;
  inline __isl_give isl_multi_union_pw_aff *copy() && = delete;
  inline __isl_keep isl_multi_union_pw_aff *get() const;
  inline __isl_give isl_multi_union_pw_aff *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline multi_union_pw_aff add(multi_union_pw_aff multi2) const;
  inline multi_union_pw_aff align_params(space model) const;
  inline union_pw_aff apply_aff(aff aff) const;
  inline union_pw_aff apply_pw_aff(pw_aff pa) const;
  inline multi_union_pw_aff apply_pw_multi_aff(pw_multi_aff pma) const;
  inline multi_union_pw_aff coalesce() const;
  inline unsigned int dim(isl::dim type) const;
  inline union_set domain() const;
  inline multi_union_pw_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline multi_pw_aff extract_multi_pw_aff(space space) const;
  inline multi_union_pw_aff factor_range() const;
  inline int find_dim_by_id(isl::dim type, const id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline multi_union_pw_aff flat_range_product(multi_union_pw_aff multi2) const;
  inline multi_union_pw_aff flatten_range() const;
  inline multi_union_pw_aff floor() const;
  static inline multi_union_pw_aff from_multi_aff(multi_aff ma);
  inline multi_union_pw_aff from_range() const;
  static inline multi_union_pw_aff from_union_map(union_map umap);
  static inline multi_union_pw_aff from_union_pw_aff_list(space space, union_pw_aff_list list);
  inline id get_dim_id(isl::dim type, unsigned int pos) const;
  inline space get_domain_space() const;
  inline space get_space() const;
  inline id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline union_pw_aff get_union_pw_aff(int pos) const;
  inline multi_union_pw_aff gist(union_set context) const;
  inline multi_union_pw_aff gist_params(set context) const;
  inline boolean has_tuple_id(isl::dim type) const;
  inline multi_union_pw_aff intersect_domain(union_set uset) const;
  inline multi_union_pw_aff intersect_params(set params) const;
  inline multi_union_pw_aff intersect_range(set set) const;
  inline boolean involves_nan() const;
  inline multi_val max_multi_val() const;
  inline multi_val min_multi_val() const;
  inline multi_union_pw_aff mod_multi_val(multi_val mv) const;
  static inline multi_union_pw_aff multi_aff_on_domain(union_set domain, multi_aff ma);
  static inline multi_union_pw_aff multi_val_on_domain(union_set domain, multi_val mv);
  inline multi_union_pw_aff neg() const;
  inline boolean plain_is_equal(const multi_union_pw_aff &multi2) const;
  inline multi_union_pw_aff pullback(union_pw_multi_aff upma) const;
  static inline multi_union_pw_aff pw_multi_aff_on_domain(union_set domain, pw_multi_aff pma);
  inline multi_union_pw_aff range_factor_domain() const;
  inline multi_union_pw_aff range_factor_range() const;
  inline boolean range_is_wrapping() const;
  inline multi_union_pw_aff range_product(multi_union_pw_aff multi2) const;
  inline multi_union_pw_aff range_splice(unsigned int pos, multi_union_pw_aff multi2) const;
  inline multi_union_pw_aff reset_tuple_id(isl::dim type) const;
  inline multi_union_pw_aff reset_user() const;
  inline multi_union_pw_aff scale_down_multi_val(multi_val mv) const;
  inline multi_union_pw_aff scale_down_val(val v) const;
  inline multi_union_pw_aff scale_multi_val(multi_val mv) const;
  inline multi_union_pw_aff scale_val(val v) const;
  inline multi_union_pw_aff set_dim_id(isl::dim type, unsigned int pos, id id) const;
  inline multi_union_pw_aff set_tuple_id(isl::dim type, id id) const;
  inline multi_union_pw_aff set_tuple_name(isl::dim type, const std::string &s) const;
  inline multi_union_pw_aff set_union_pw_aff(int pos, union_pw_aff el) const;
  inline multi_union_pw_aff sub(multi_union_pw_aff multi2) const;
  inline multi_union_pw_aff union_add(multi_union_pw_aff mupa2) const;
  static inline multi_union_pw_aff zero(space space);
  inline union_set zero_union_set() const;
};

// declarations for isl::multi_val
inline multi_val manage(__isl_take isl_multi_val *ptr);
inline multi_val manage_copy(__isl_keep isl_multi_val *ptr);

class multi_val {
  friend inline multi_val manage(__isl_take isl_multi_val *ptr);
  friend inline multi_val manage_copy(__isl_keep isl_multi_val *ptr);

  isl_multi_val *ptr = nullptr;

  inline explicit multi_val(__isl_take isl_multi_val *ptr);

public:
  inline /* implicit */ multi_val();
  inline /* implicit */ multi_val(const multi_val &obj);
  inline /* implicit */ multi_val(std::nullptr_t);
  inline multi_val &operator=(multi_val obj);
  inline ~multi_val();
  inline __isl_give isl_multi_val *copy() const &;
  inline __isl_give isl_multi_val *copy() && = delete;
  inline __isl_keep isl_multi_val *get() const;
  inline __isl_give isl_multi_val *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline multi_val add(multi_val multi2) const;
  inline multi_val add_dims(isl::dim type, unsigned int n) const;
  inline multi_val add_val(val v) const;
  inline multi_val align_params(space model) const;
  inline unsigned int dim(isl::dim type) const;
  inline multi_val drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline multi_val factor_range() const;
  inline int find_dim_by_id(isl::dim type, const id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline multi_val flat_range_product(multi_val multi2) const;
  inline multi_val flatten_range() const;
  inline multi_val from_range() const;
  static inline multi_val from_val_list(space space, val_list list);
  inline id get_dim_id(isl::dim type, unsigned int pos) const;
  inline space get_domain_space() const;
  inline space get_space() const;
  inline id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline val get_val(int pos) const;
  inline boolean has_tuple_id(isl::dim type) const;
  inline multi_val insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_nan() const;
  inline multi_val mod_multi_val(multi_val mv) const;
  inline multi_val mod_val(val v) const;
  inline multi_val neg() const;
  inline boolean plain_is_equal(const multi_val &multi2) const;
  inline multi_val product(multi_val multi2) const;
  inline multi_val project_domain_on_params() const;
  inline multi_val range_factor_domain() const;
  inline multi_val range_factor_range() const;
  inline boolean range_is_wrapping() const;
  inline multi_val range_product(multi_val multi2) const;
  inline multi_val range_splice(unsigned int pos, multi_val multi2) const;
  static inline multi_val read_from_str(ctx ctx, const std::string &str);
  inline multi_val reset_tuple_id(isl::dim type) const;
  inline multi_val reset_user() const;
  inline multi_val scale_down_multi_val(multi_val mv) const;
  inline multi_val scale_down_val(val v) const;
  inline multi_val scale_multi_val(multi_val mv) const;
  inline multi_val scale_val(val v) const;
  inline multi_val set_dim_id(isl::dim type, unsigned int pos, id id) const;
  inline multi_val set_tuple_id(isl::dim type, id id) const;
  inline multi_val set_tuple_name(isl::dim type, const std::string &s) const;
  inline multi_val set_val(int pos, val el) const;
  inline multi_val splice(unsigned int in_pos, unsigned int out_pos, multi_val multi2) const;
  inline multi_val sub(multi_val multi2) const;
  static inline multi_val zero(space space);
};

// declarations for isl::point
inline point manage(__isl_take isl_point *ptr);
inline point manage_copy(__isl_keep isl_point *ptr);

class point {
  friend inline point manage(__isl_take isl_point *ptr);
  friend inline point manage_copy(__isl_keep isl_point *ptr);

  isl_point *ptr = nullptr;

  inline explicit point(__isl_take isl_point *ptr);

public:
  inline /* implicit */ point();
  inline /* implicit */ point(const point &obj);
  inline /* implicit */ point(std::nullptr_t);
  inline explicit point(space dim);
  inline point &operator=(point obj);
  inline ~point();
  inline __isl_give isl_point *copy() const &;
  inline __isl_give isl_point *copy() && = delete;
  inline __isl_keep isl_point *get() const;
  inline __isl_give isl_point *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline point add_ui(isl::dim type, int pos, unsigned int val) const;
  inline val get_coordinate_val(isl::dim type, int pos) const;
  inline space get_space() const;
  inline point set_coordinate_val(isl::dim type, int pos, val v) const;
  inline point sub_ui(isl::dim type, int pos, unsigned int val) const;
};

// declarations for isl::pw_aff
inline pw_aff manage(__isl_take isl_pw_aff *ptr);
inline pw_aff manage_copy(__isl_keep isl_pw_aff *ptr);

class pw_aff {
  friend inline pw_aff manage(__isl_take isl_pw_aff *ptr);
  friend inline pw_aff manage_copy(__isl_keep isl_pw_aff *ptr);

  isl_pw_aff *ptr = nullptr;

  inline explicit pw_aff(__isl_take isl_pw_aff *ptr);

public:
  inline /* implicit */ pw_aff();
  inline /* implicit */ pw_aff(const pw_aff &obj);
  inline /* implicit */ pw_aff(std::nullptr_t);
  inline /* implicit */ pw_aff(aff aff);
  inline explicit pw_aff(local_space ls);
  inline explicit pw_aff(set domain, val v);
  inline explicit pw_aff(ctx ctx, const std::string &str);
  inline pw_aff &operator=(pw_aff obj);
  inline ~pw_aff();
  inline __isl_give isl_pw_aff *copy() const &;
  inline __isl_give isl_pw_aff *copy() && = delete;
  inline __isl_keep isl_pw_aff *get() const;
  inline __isl_give isl_pw_aff *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline pw_aff add(pw_aff pwaff2) const;
  inline pw_aff add_dims(isl::dim type, unsigned int n) const;
  inline pw_aff align_params(space model) const;
  static inline pw_aff alloc(set set, aff aff);
  inline pw_aff ceil() const;
  inline pw_aff coalesce() const;
  inline pw_aff cond(pw_aff pwaff_true, pw_aff pwaff_false) const;
  inline unsigned int dim(isl::dim type) const;
  inline pw_aff div(pw_aff pa2) const;
  inline set domain() const;
  inline pw_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline pw_aff drop_unused_params() const;
  static inline pw_aff empty(space dim);
  inline map eq_map(pw_aff pa2) const;
  inline set eq_set(pw_aff pwaff2) const;
  inline val eval(point pnt) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline pw_aff floor() const;
  inline stat foreach_piece(const std::function<stat(set, aff)> &fn) const;
  inline pw_aff from_range() const;
  inline set ge_set(pw_aff pwaff2) const;
  inline id get_dim_id(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline space get_domain_space() const;
  inline uint32_t get_hash() const;
  inline space get_space() const;
  inline id get_tuple_id(isl::dim type) const;
  inline pw_aff gist(set context) const;
  inline pw_aff gist_params(set context) const;
  inline map gt_map(pw_aff pa2) const;
  inline set gt_set(pw_aff pwaff2) const;
  inline boolean has_dim_id(isl::dim type, unsigned int pos) const;
  inline boolean has_tuple_id(isl::dim type) const;
  inline pw_aff insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline pw_aff intersect_domain(set set) const;
  inline pw_aff intersect_params(set set) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_nan() const;
  inline boolean is_cst() const;
  inline boolean is_empty() const;
  inline boolean is_equal(const pw_aff &pa2) const;
  inline set le_set(pw_aff pwaff2) const;
  inline map lt_map(pw_aff pa2) const;
  inline set lt_set(pw_aff pwaff2) const;
  inline pw_aff max(pw_aff pwaff2) const;
  inline pw_aff min(pw_aff pwaff2) const;
  inline pw_aff mod(val mod) const;
  inline pw_aff move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline pw_aff mul(pw_aff pwaff2) const;
  inline int n_piece() const;
  static inline pw_aff nan_on_domain(local_space ls);
  inline set ne_set(pw_aff pwaff2) const;
  inline pw_aff neg() const;
  inline set non_zero_set() const;
  inline set nonneg_set() const;
  inline set params() const;
  inline int plain_cmp(const pw_aff &pa2) const;
  inline boolean plain_is_equal(const pw_aff &pwaff2) const;
  inline set pos_set() const;
  inline pw_aff project_domain_on_params() const;
  inline pw_aff pullback(multi_aff ma) const;
  inline pw_aff pullback(pw_multi_aff pma) const;
  inline pw_aff pullback(multi_pw_aff mpa) const;
  inline pw_aff reset_tuple_id(isl::dim type) const;
  inline pw_aff reset_user() const;
  inline pw_aff scale(val v) const;
  inline pw_aff scale_down(val f) const;
  inline pw_aff set_dim_id(isl::dim type, unsigned int pos, id id) const;
  inline pw_aff set_tuple_id(isl::dim type, id id) const;
  inline pw_aff sub(pw_aff pwaff2) const;
  inline pw_aff subtract_domain(set set) const;
  inline pw_aff tdiv_q(pw_aff pa2) const;
  inline pw_aff tdiv_r(pw_aff pa2) const;
  inline pw_aff union_add(pw_aff pwaff2) const;
  inline pw_aff union_max(pw_aff pwaff2) const;
  inline pw_aff union_min(pw_aff pwaff2) const;
  static inline pw_aff var_on_domain(local_space ls, isl::dim type, unsigned int pos);
  inline set zero_set() const;
};

// declarations for isl::pw_aff_list
inline pw_aff_list manage(__isl_take isl_pw_aff_list *ptr);
inline pw_aff_list manage_copy(__isl_keep isl_pw_aff_list *ptr);

class pw_aff_list {
  friend inline pw_aff_list manage(__isl_take isl_pw_aff_list *ptr);
  friend inline pw_aff_list manage_copy(__isl_keep isl_pw_aff_list *ptr);

  isl_pw_aff_list *ptr = nullptr;

  inline explicit pw_aff_list(__isl_take isl_pw_aff_list *ptr);

public:
  inline /* implicit */ pw_aff_list();
  inline /* implicit */ pw_aff_list(const pw_aff_list &obj);
  inline /* implicit */ pw_aff_list(std::nullptr_t);
  inline pw_aff_list &operator=(pw_aff_list obj);
  inline ~pw_aff_list();
  inline __isl_give isl_pw_aff_list *copy() const &;
  inline __isl_give isl_pw_aff_list *copy() && = delete;
  inline __isl_keep isl_pw_aff_list *get() const;
  inline __isl_give isl_pw_aff_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline pw_aff_list add(pw_aff el) const;
  static inline pw_aff_list alloc(ctx ctx, int n);
  inline pw_aff_list concat(pw_aff_list list2) const;
  inline pw_aff_list drop(unsigned int first, unsigned int n) const;
  inline set eq_set(pw_aff_list list2) const;
  inline stat foreach(const std::function<stat(pw_aff)> &fn) const;
  static inline pw_aff_list from_pw_aff(pw_aff el);
  inline set ge_set(pw_aff_list list2) const;
  inline pw_aff get_at(int index) const;
  inline pw_aff get_pw_aff(int index) const;
  inline set gt_set(pw_aff_list list2) const;
  inline pw_aff_list insert(unsigned int pos, pw_aff el) const;
  inline set le_set(pw_aff_list list2) const;
  inline set lt_set(pw_aff_list list2) const;
  inline pw_aff max() const;
  inline pw_aff min() const;
  inline int n_pw_aff() const;
  inline set ne_set(pw_aff_list list2) const;
  inline pw_aff_list reverse() const;
  inline pw_aff_list set_pw_aff(int index, pw_aff el) const;
  inline int size() const;
  inline pw_aff_list swap(unsigned int pos1, unsigned int pos2) const;
};

// declarations for isl::pw_multi_aff
inline pw_multi_aff manage(__isl_take isl_pw_multi_aff *ptr);
inline pw_multi_aff manage_copy(__isl_keep isl_pw_multi_aff *ptr);

class pw_multi_aff {
  friend inline pw_multi_aff manage(__isl_take isl_pw_multi_aff *ptr);
  friend inline pw_multi_aff manage_copy(__isl_keep isl_pw_multi_aff *ptr);

  isl_pw_multi_aff *ptr = nullptr;

  inline explicit pw_multi_aff(__isl_take isl_pw_multi_aff *ptr);

public:
  inline /* implicit */ pw_multi_aff();
  inline /* implicit */ pw_multi_aff(const pw_multi_aff &obj);
  inline /* implicit */ pw_multi_aff(std::nullptr_t);
  inline /* implicit */ pw_multi_aff(multi_aff ma);
  inline /* implicit */ pw_multi_aff(pw_aff pa);
  inline explicit pw_multi_aff(ctx ctx, const std::string &str);
  inline pw_multi_aff &operator=(pw_multi_aff obj);
  inline ~pw_multi_aff();
  inline __isl_give isl_pw_multi_aff *copy() const &;
  inline __isl_give isl_pw_multi_aff *copy() && = delete;
  inline __isl_keep isl_pw_multi_aff *get() const;
  inline __isl_give isl_pw_multi_aff *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline pw_multi_aff add(pw_multi_aff pma2) const;
  inline pw_multi_aff align_params(space model) const;
  static inline pw_multi_aff alloc(set set, multi_aff maff);
  inline pw_multi_aff coalesce() const;
  inline unsigned int dim(isl::dim type) const;
  inline set domain() const;
  inline pw_multi_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline pw_multi_aff drop_unused_params() const;
  static inline pw_multi_aff empty(space space);
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline pw_multi_aff fix_si(isl::dim type, unsigned int pos, int value) const;
  inline pw_multi_aff flat_range_product(pw_multi_aff pma2) const;
  inline stat foreach_piece(const std::function<stat(set, multi_aff)> &fn) const;
  static inline pw_multi_aff from_domain(set set);
  static inline pw_multi_aff from_map(map map);
  static inline pw_multi_aff from_multi_pw_aff(multi_pw_aff mpa);
  static inline pw_multi_aff from_set(set set);
  inline id get_dim_id(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline space get_domain_space() const;
  inline pw_aff get_pw_aff(int pos) const;
  inline space get_space() const;
  inline id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline pw_multi_aff gist(set set) const;
  inline pw_multi_aff gist_params(set set) const;
  inline boolean has_tuple_id(isl::dim type) const;
  inline boolean has_tuple_name(isl::dim type) const;
  static inline pw_multi_aff identity(space space);
  inline pw_multi_aff intersect_domain(set set) const;
  inline pw_multi_aff intersect_params(set set) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_nan() const;
  inline boolean is_equal(const pw_multi_aff &pma2) const;
  static inline pw_multi_aff multi_val_on_domain(set domain, multi_val mv);
  inline int n_piece() const;
  inline pw_multi_aff neg() const;
  inline boolean plain_is_equal(const pw_multi_aff &pma2) const;
  inline pw_multi_aff product(pw_multi_aff pma2) const;
  inline pw_multi_aff project_domain_on_params() const;
  static inline pw_multi_aff project_out_map(space space, isl::dim type, unsigned int first, unsigned int n);
  inline pw_multi_aff pullback(multi_aff ma) const;
  inline pw_multi_aff pullback(pw_multi_aff pma2) const;
  static inline pw_multi_aff range_map(space space);
  inline pw_multi_aff range_product(pw_multi_aff pma2) const;
  inline pw_multi_aff reset_tuple_id(isl::dim type) const;
  inline pw_multi_aff reset_user() const;
  inline pw_multi_aff scale_down_val(val v) const;
  inline pw_multi_aff scale_multi_val(multi_val mv) const;
  inline pw_multi_aff scale_val(val v) const;
  inline pw_multi_aff set_dim_id(isl::dim type, unsigned int pos, id id) const;
  inline pw_multi_aff set_pw_aff(unsigned int pos, pw_aff pa) const;
  inline pw_multi_aff set_tuple_id(isl::dim type, id id) const;
  inline pw_multi_aff sub(pw_multi_aff pma2) const;
  inline pw_multi_aff subtract_domain(set set) const;
  inline pw_multi_aff union_add(pw_multi_aff pma2) const;
  inline pw_multi_aff union_lexmax(pw_multi_aff pma2) const;
  inline pw_multi_aff union_lexmin(pw_multi_aff pma2) const;
  static inline pw_multi_aff zero(space space);
};

// declarations for isl::pw_multi_aff_list
inline pw_multi_aff_list manage(__isl_take isl_pw_multi_aff_list *ptr);
inline pw_multi_aff_list manage_copy(__isl_keep isl_pw_multi_aff_list *ptr);

class pw_multi_aff_list {
  friend inline pw_multi_aff_list manage(__isl_take isl_pw_multi_aff_list *ptr);
  friend inline pw_multi_aff_list manage_copy(__isl_keep isl_pw_multi_aff_list *ptr);

  isl_pw_multi_aff_list *ptr = nullptr;

  inline explicit pw_multi_aff_list(__isl_take isl_pw_multi_aff_list *ptr);

public:
  inline /* implicit */ pw_multi_aff_list();
  inline /* implicit */ pw_multi_aff_list(const pw_multi_aff_list &obj);
  inline /* implicit */ pw_multi_aff_list(std::nullptr_t);
  inline pw_multi_aff_list &operator=(pw_multi_aff_list obj);
  inline ~pw_multi_aff_list();
  inline __isl_give isl_pw_multi_aff_list *copy() const &;
  inline __isl_give isl_pw_multi_aff_list *copy() && = delete;
  inline __isl_keep isl_pw_multi_aff_list *get() const;
  inline __isl_give isl_pw_multi_aff_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline pw_multi_aff_list add(pw_multi_aff el) const;
  static inline pw_multi_aff_list alloc(ctx ctx, int n);
  inline pw_multi_aff_list concat(pw_multi_aff_list list2) const;
  inline pw_multi_aff_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(pw_multi_aff)> &fn) const;
  static inline pw_multi_aff_list from_pw_multi_aff(pw_multi_aff el);
  inline pw_multi_aff get_at(int index) const;
  inline pw_multi_aff get_pw_multi_aff(int index) const;
  inline pw_multi_aff_list insert(unsigned int pos, pw_multi_aff el) const;
  inline int n_pw_multi_aff() const;
  inline pw_multi_aff_list reverse() const;
  inline pw_multi_aff_list set_pw_multi_aff(int index, pw_multi_aff el) const;
  inline int size() const;
  inline pw_multi_aff_list swap(unsigned int pos1, unsigned int pos2) const;
};

// declarations for isl::pw_qpolynomial
inline pw_qpolynomial manage(__isl_take isl_pw_qpolynomial *ptr);
inline pw_qpolynomial manage_copy(__isl_keep isl_pw_qpolynomial *ptr);

class pw_qpolynomial {
  friend inline pw_qpolynomial manage(__isl_take isl_pw_qpolynomial *ptr);
  friend inline pw_qpolynomial manage_copy(__isl_keep isl_pw_qpolynomial *ptr);

  isl_pw_qpolynomial *ptr = nullptr;

  inline explicit pw_qpolynomial(__isl_take isl_pw_qpolynomial *ptr);

public:
  inline /* implicit */ pw_qpolynomial();
  inline /* implicit */ pw_qpolynomial(const pw_qpolynomial &obj);
  inline /* implicit */ pw_qpolynomial(std::nullptr_t);
  inline explicit pw_qpolynomial(ctx ctx, const std::string &str);
  inline pw_qpolynomial &operator=(pw_qpolynomial obj);
  inline ~pw_qpolynomial();
  inline __isl_give isl_pw_qpolynomial *copy() const &;
  inline __isl_give isl_pw_qpolynomial *copy() && = delete;
  inline __isl_keep isl_pw_qpolynomial *get() const;
  inline __isl_give isl_pw_qpolynomial *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline pw_qpolynomial add(pw_qpolynomial pwqp2) const;
  inline pw_qpolynomial add_dims(isl::dim type, unsigned int n) const;
  static inline pw_qpolynomial alloc(set set, qpolynomial qp);
  inline pw_qpolynomial coalesce() const;
  inline unsigned int dim(isl::dim type) const;
  inline set domain() const;
  inline pw_qpolynomial drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline pw_qpolynomial drop_unused_params() const;
  inline val eval(point pnt) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline pw_qpolynomial fix_val(isl::dim type, unsigned int n, val v) const;
  inline stat foreach_piece(const std::function<stat(set, qpolynomial)> &fn) const;
  static inline pw_qpolynomial from_pw_aff(pw_aff pwaff);
  static inline pw_qpolynomial from_qpolynomial(qpolynomial qp);
  inline pw_qpolynomial from_range() const;
  inline space get_domain_space() const;
  inline space get_space() const;
  inline pw_qpolynomial gist(set context) const;
  inline pw_qpolynomial gist_params(set context) const;
  inline boolean has_equal_space(const pw_qpolynomial &pwqp2) const;
  inline pw_qpolynomial insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline pw_qpolynomial intersect_domain(set set) const;
  inline pw_qpolynomial intersect_params(set set) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_nan() const;
  inline boolean is_zero() const;
  inline val max() const;
  inline val min() const;
  inline pw_qpolynomial move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline pw_qpolynomial mul(pw_qpolynomial pwqp2) const;
  inline int n_piece() const;
  inline pw_qpolynomial neg() const;
  inline boolean plain_is_equal(const pw_qpolynomial &pwqp2) const;
  inline pw_qpolynomial pow(unsigned int exponent) const;
  inline pw_qpolynomial project_domain_on_params() const;
  inline pw_qpolynomial reset_domain_space(space dim) const;
  inline pw_qpolynomial reset_user() const;
  inline pw_qpolynomial scale_down_val(val v) const;
  inline pw_qpolynomial scale_val(val v) const;
  inline pw_qpolynomial split_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline pw_qpolynomial split_periods(int max_periods) const;
  inline pw_qpolynomial sub(pw_qpolynomial pwqp2) const;
  inline pw_qpolynomial subtract_domain(set set) const;
  inline pw_qpolynomial to_polynomial(int sign) const;
  static inline pw_qpolynomial zero(space dim);
};

// declarations for isl::pw_qpolynomial_fold_list
inline pw_qpolynomial_fold_list manage(__isl_take isl_pw_qpolynomial_fold_list *ptr);
inline pw_qpolynomial_fold_list manage_copy(__isl_keep isl_pw_qpolynomial_fold_list *ptr);

class pw_qpolynomial_fold_list {
  friend inline pw_qpolynomial_fold_list manage(__isl_take isl_pw_qpolynomial_fold_list *ptr);
  friend inline pw_qpolynomial_fold_list manage_copy(__isl_keep isl_pw_qpolynomial_fold_list *ptr);

  isl_pw_qpolynomial_fold_list *ptr = nullptr;

  inline explicit pw_qpolynomial_fold_list(__isl_take isl_pw_qpolynomial_fold_list *ptr);

public:
  inline /* implicit */ pw_qpolynomial_fold_list();
  inline /* implicit */ pw_qpolynomial_fold_list(const pw_qpolynomial_fold_list &obj);
  inline /* implicit */ pw_qpolynomial_fold_list(std::nullptr_t);
  inline pw_qpolynomial_fold_list &operator=(pw_qpolynomial_fold_list obj);
  inline ~pw_qpolynomial_fold_list();
  inline __isl_give isl_pw_qpolynomial_fold_list *copy() const &;
  inline __isl_give isl_pw_qpolynomial_fold_list *copy() && = delete;
  inline __isl_keep isl_pw_qpolynomial_fold_list *get() const;
  inline __isl_give isl_pw_qpolynomial_fold_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

};

// declarations for isl::pw_qpolynomial_list
inline pw_qpolynomial_list manage(__isl_take isl_pw_qpolynomial_list *ptr);
inline pw_qpolynomial_list manage_copy(__isl_keep isl_pw_qpolynomial_list *ptr);

class pw_qpolynomial_list {
  friend inline pw_qpolynomial_list manage(__isl_take isl_pw_qpolynomial_list *ptr);
  friend inline pw_qpolynomial_list manage_copy(__isl_keep isl_pw_qpolynomial_list *ptr);

  isl_pw_qpolynomial_list *ptr = nullptr;

  inline explicit pw_qpolynomial_list(__isl_take isl_pw_qpolynomial_list *ptr);

public:
  inline /* implicit */ pw_qpolynomial_list();
  inline /* implicit */ pw_qpolynomial_list(const pw_qpolynomial_list &obj);
  inline /* implicit */ pw_qpolynomial_list(std::nullptr_t);
  inline pw_qpolynomial_list &operator=(pw_qpolynomial_list obj);
  inline ~pw_qpolynomial_list();
  inline __isl_give isl_pw_qpolynomial_list *copy() const &;
  inline __isl_give isl_pw_qpolynomial_list *copy() && = delete;
  inline __isl_keep isl_pw_qpolynomial_list *get() const;
  inline __isl_give isl_pw_qpolynomial_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline pw_qpolynomial_list add(pw_qpolynomial el) const;
  static inline pw_qpolynomial_list alloc(ctx ctx, int n);
  inline pw_qpolynomial_list concat(pw_qpolynomial_list list2) const;
  inline pw_qpolynomial_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(pw_qpolynomial)> &fn) const;
  static inline pw_qpolynomial_list from_pw_qpolynomial(pw_qpolynomial el);
  inline pw_qpolynomial get_at(int index) const;
  inline pw_qpolynomial get_pw_qpolynomial(int index) const;
  inline pw_qpolynomial_list insert(unsigned int pos, pw_qpolynomial el) const;
  inline int n_pw_qpolynomial() const;
  inline pw_qpolynomial_list reverse() const;
  inline pw_qpolynomial_list set_pw_qpolynomial(int index, pw_qpolynomial el) const;
  inline int size() const;
  inline pw_qpolynomial_list swap(unsigned int pos1, unsigned int pos2) const;
};

// declarations for isl::qpolynomial
inline qpolynomial manage(__isl_take isl_qpolynomial *ptr);
inline qpolynomial manage_copy(__isl_keep isl_qpolynomial *ptr);

class qpolynomial {
  friend inline qpolynomial manage(__isl_take isl_qpolynomial *ptr);
  friend inline qpolynomial manage_copy(__isl_keep isl_qpolynomial *ptr);

  isl_qpolynomial *ptr = nullptr;

  inline explicit qpolynomial(__isl_take isl_qpolynomial *ptr);

public:
  inline /* implicit */ qpolynomial();
  inline /* implicit */ qpolynomial(const qpolynomial &obj);
  inline /* implicit */ qpolynomial(std::nullptr_t);
  inline qpolynomial &operator=(qpolynomial obj);
  inline ~qpolynomial();
  inline __isl_give isl_qpolynomial *copy() const &;
  inline __isl_give isl_qpolynomial *copy() && = delete;
  inline __isl_keep isl_qpolynomial *get() const;
  inline __isl_give isl_qpolynomial *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline qpolynomial add(qpolynomial qp2) const;
  inline qpolynomial add_dims(isl::dim type, unsigned int n) const;
  inline qpolynomial align_params(space model) const;
  inline stat as_polynomial_on_domain(const basic_set &bset, const std::function<stat(basic_set, qpolynomial)> &fn) const;
  inline unsigned int dim(isl::dim type) const;
  inline qpolynomial drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline val eval(point pnt) const;
  inline stat foreach_term(const std::function<stat(term)> &fn) const;
  static inline qpolynomial from_aff(aff aff);
  static inline qpolynomial from_constraint(constraint c, isl::dim type, unsigned int pos);
  static inline qpolynomial from_term(term term);
  inline val get_constant_val() const;
  inline space get_domain_space() const;
  inline space get_space() const;
  inline qpolynomial gist(set context) const;
  inline qpolynomial gist_params(set context) const;
  inline qpolynomial homogenize() const;
  static inline qpolynomial infty_on_domain(space dim);
  inline qpolynomial insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean is_infty() const;
  inline boolean is_nan() const;
  inline boolean is_neginfty() const;
  inline boolean is_zero() const;
  inline qpolynomial move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline qpolynomial mul(qpolynomial qp2) const;
  static inline qpolynomial nan_on_domain(space dim);
  inline qpolynomial neg() const;
  static inline qpolynomial neginfty_on_domain(space dim);
  static inline qpolynomial one_on_domain(space dim);
  inline boolean plain_is_equal(const qpolynomial &qp2) const;
  inline qpolynomial pow(unsigned int power) const;
  inline qpolynomial project_domain_on_params() const;
  inline qpolynomial scale_down_val(val v) const;
  inline qpolynomial scale_val(val v) const;
  inline int sgn() const;
  inline qpolynomial sub(qpolynomial qp2) const;
  static inline qpolynomial val_on_domain(space space, val val);
  static inline qpolynomial var_on_domain(space dim, isl::dim type, unsigned int pos);
  static inline qpolynomial zero_on_domain(space dim);
};

// declarations for isl::schedule
inline schedule manage(__isl_take isl_schedule *ptr);
inline schedule manage_copy(__isl_keep isl_schedule *ptr);

class schedule {
  friend inline schedule manage(__isl_take isl_schedule *ptr);
  friend inline schedule manage_copy(__isl_keep isl_schedule *ptr);

  isl_schedule *ptr = nullptr;

  inline explicit schedule(__isl_take isl_schedule *ptr);

public:
  inline /* implicit */ schedule();
  inline /* implicit */ schedule(const schedule &obj);
  inline /* implicit */ schedule(std::nullptr_t);
  inline explicit schedule(ctx ctx, const std::string &str);
  inline schedule &operator=(schedule obj);
  inline ~schedule();
  inline __isl_give isl_schedule *copy() const &;
  inline __isl_give isl_schedule *copy() && = delete;
  inline __isl_keep isl_schedule *get() const;
  inline __isl_give isl_schedule *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline schedule align_params(space space) const;
  static inline schedule empty(space space);
  static inline schedule from_domain(union_set domain);
  inline union_set get_domain() const;
  inline union_map get_map() const;
  inline schedule_node get_root() const;
  inline schedule gist_domain_params(set context) const;
  inline schedule insert_context(set context) const;
  inline schedule insert_guard(set guard) const;
  inline schedule insert_partial_schedule(multi_union_pw_aff partial) const;
  inline schedule intersect_domain(union_set domain) const;
  inline boolean plain_is_equal(const schedule &schedule2) const;
  inline schedule pullback(union_pw_multi_aff upma) const;
  inline schedule reset_user() const;
  inline schedule sequence(schedule schedule2) const;
};

// declarations for isl::schedule_constraints
inline schedule_constraints manage(__isl_take isl_schedule_constraints *ptr);
inline schedule_constraints manage_copy(__isl_keep isl_schedule_constraints *ptr);

class schedule_constraints {
  friend inline schedule_constraints manage(__isl_take isl_schedule_constraints *ptr);
  friend inline schedule_constraints manage_copy(__isl_keep isl_schedule_constraints *ptr);

  isl_schedule_constraints *ptr = nullptr;

  inline explicit schedule_constraints(__isl_take isl_schedule_constraints *ptr);

public:
  inline /* implicit */ schedule_constraints();
  inline /* implicit */ schedule_constraints(const schedule_constraints &obj);
  inline /* implicit */ schedule_constraints(std::nullptr_t);
  inline explicit schedule_constraints(ctx ctx, const std::string &str);
  inline schedule_constraints &operator=(schedule_constraints obj);
  inline ~schedule_constraints();
  inline __isl_give isl_schedule_constraints *copy() const &;
  inline __isl_give isl_schedule_constraints *copy() && = delete;
  inline __isl_keep isl_schedule_constraints *get() const;
  inline __isl_give isl_schedule_constraints *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline schedule_constraints apply(union_map umap) const;
  inline schedule compute_schedule() const;
  inline union_map get_coincidence() const;
  inline union_map get_conditional_validity() const;
  inline union_map get_conditional_validity_condition() const;
  inline set get_context() const;
  inline union_set get_domain() const;
  inline union_map get_proximity() const;
  inline union_map get_validity() const;
  static inline schedule_constraints on_domain(union_set domain);
  inline schedule_constraints set_coincidence(union_map coincidence) const;
  inline schedule_constraints set_conditional_validity(union_map condition, union_map validity) const;
  inline schedule_constraints set_context(set context) const;
  inline schedule_constraints set_proximity(union_map proximity) const;
  inline schedule_constraints set_validity(union_map validity) const;
};

// declarations for isl::schedule_node
inline schedule_node manage(__isl_take isl_schedule_node *ptr);
inline schedule_node manage_copy(__isl_keep isl_schedule_node *ptr);

class schedule_node {
  friend inline schedule_node manage(__isl_take isl_schedule_node *ptr);
  friend inline schedule_node manage_copy(__isl_keep isl_schedule_node *ptr);

  isl_schedule_node *ptr = nullptr;

  inline explicit schedule_node(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node();
  inline /* implicit */ schedule_node(const schedule_node &obj);
  inline /* implicit */ schedule_node(std::nullptr_t);
  inline schedule_node &operator=(schedule_node obj);
  inline ~schedule_node();
  inline __isl_give isl_schedule_node *copy() const &;
  inline __isl_give isl_schedule_node *copy() && = delete;
  inline __isl_keep isl_schedule_node *get() const;
  inline __isl_give isl_schedule_node *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline schedule_node align_params(space space) const;
  inline schedule_node ancestor(int generation) const;
  inline boolean band_member_get_coincident(int pos) const;
  inline schedule_node band_member_set_coincident(int pos, int coincident) const;
  inline schedule_node band_set_ast_build_options(union_set options) const;
  inline schedule_node child(int pos) const;
  inline set context_get_context() const;
  inline schedule_node cut() const;
  inline union_set domain_get_domain() const;
  inline union_pw_multi_aff expansion_get_contraction() const;
  inline union_map expansion_get_expansion() const;
  inline union_map extension_get_extension() const;
  inline union_set filter_get_filter() const;
  inline schedule_node first_child() const;
  inline stat foreach_ancestor_top_down(const std::function<stat(schedule_node)> &fn) const;
  static inline schedule_node from_domain(union_set domain);
  static inline schedule_node from_extension(union_map extension);
  inline int get_ancestor_child_position(const schedule_node &ancestor) const;
  inline schedule_node get_child(int pos) const;
  inline int get_child_position() const;
  inline union_set get_domain() const;
  inline multi_union_pw_aff get_prefix_schedule_multi_union_pw_aff() const;
  inline union_map get_prefix_schedule_relation() const;
  inline union_map get_prefix_schedule_union_map() const;
  inline union_pw_multi_aff get_prefix_schedule_union_pw_multi_aff() const;
  inline schedule get_schedule() const;
  inline int get_schedule_depth() const;
  inline schedule_node get_shared_ancestor(const schedule_node &node2) const;
  inline union_pw_multi_aff get_subtree_contraction() const;
  inline union_map get_subtree_expansion() const;
  inline union_map get_subtree_schedule_union_map() const;
  inline int get_tree_depth() const;
  inline union_set get_universe_domain() const;
  inline schedule_node graft_after(schedule_node graft) const;
  inline schedule_node graft_before(schedule_node graft) const;
  inline schedule_node group(id group_id) const;
  inline set guard_get_guard() const;
  inline boolean has_children() const;
  inline boolean has_next_sibling() const;
  inline boolean has_parent() const;
  inline boolean has_previous_sibling() const;
  inline schedule_node insert_context(set context) const;
  inline schedule_node insert_filter(union_set filter) const;
  inline schedule_node insert_guard(set context) const;
  inline schedule_node insert_mark(id mark) const;
  inline schedule_node insert_partial_schedule(multi_union_pw_aff schedule) const;
  inline schedule_node insert_sequence(union_set_list filters) const;
  inline schedule_node insert_set(union_set_list filters) const;
  inline boolean is_equal(const schedule_node &node2) const;
  inline boolean is_subtree_anchored() const;
  inline id mark_get_id() const;
  inline int n_children() const;
  inline schedule_node next_sibling() const;
  inline schedule_node order_after(union_set filter) const;
  inline schedule_node order_before(union_set filter) const;
  inline schedule_node parent() const;
  inline schedule_node previous_sibling() const;
  inline schedule_node reset_user() const;
  inline schedule_node root() const;
  inline schedule_node sequence_splice_child(int pos) const;
};

// declarations for isl::set
inline set manage(__isl_take isl_set *ptr);
inline set manage_copy(__isl_keep isl_set *ptr);

class set {
  friend inline set manage(__isl_take isl_set *ptr);
  friend inline set manage_copy(__isl_keep isl_set *ptr);

  isl_set *ptr = nullptr;

  inline explicit set(__isl_take isl_set *ptr);

public:
  inline /* implicit */ set();
  inline /* implicit */ set(const set &obj);
  inline /* implicit */ set(std::nullptr_t);
  inline explicit set(ctx ctx, const std::string &str);
  inline /* implicit */ set(basic_set bset);
  inline /* implicit */ set(point pnt);
  inline explicit set(union_set uset);
  inline set &operator=(set obj);
  inline ~set();
  inline __isl_give isl_set *copy() const &;
  inline __isl_give isl_set *copy() && = delete;
  inline __isl_keep isl_set *get() const;
  inline __isl_give isl_set *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline set add_constraint(constraint constraint) const;
  inline set add_dims(isl::dim type, unsigned int n) const;
  inline basic_set affine_hull() const;
  inline set align_params(space model) const;
  inline set apply(map map) const;
  inline basic_set bounded_simple_hull() const;
  static inline set box_from_points(point pnt1, point pnt2);
  inline set coalesce() const;
  inline basic_set coefficients() const;
  inline set complement() const;
  inline basic_set convex_hull() const;
  inline val count_val() const;
  inline set detect_equalities() const;
  inline unsigned int dim(isl::dim type) const;
  inline boolean dim_has_any_lower_bound(isl::dim type, unsigned int pos) const;
  inline boolean dim_has_any_upper_bound(isl::dim type, unsigned int pos) const;
  inline boolean dim_has_lower_bound(isl::dim type, unsigned int pos) const;
  inline boolean dim_has_upper_bound(isl::dim type, unsigned int pos) const;
  inline boolean dim_is_bounded(isl::dim type, unsigned int pos) const;
  inline pw_aff dim_max(int pos) const;
  inline pw_aff dim_min(int pos) const;
  inline set drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline set drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline set drop_unused_params() const;
  inline set eliminate(isl::dim type, unsigned int first, unsigned int n) const;
  static inline set empty(space space);
  inline set equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline int find_dim_by_id(isl::dim type, const id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline set fix_si(isl::dim type, unsigned int pos, int value) const;
  inline set fix_val(isl::dim type, unsigned int pos, val v) const;
  inline set flat_product(set set2) const;
  inline set flatten() const;
  inline map flatten_map() const;
  inline int follows_at(const set &set2, int pos) const;
  inline stat foreach_basic_set(const std::function<stat(basic_set)> &fn) const;
  inline stat foreach_point(const std::function<stat(point)> &fn) const;
  static inline set from_multi_aff(multi_aff ma);
  static inline set from_multi_pw_aff(multi_pw_aff mpa);
  inline set from_params() const;
  static inline set from_pw_aff(pw_aff pwaff);
  static inline set from_pw_multi_aff(pw_multi_aff pma);
  inline basic_set_list get_basic_set_list() const;
  inline id get_dim_id(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline space get_space() const;
  inline val get_stride(int pos) const;
  inline id get_tuple_id() const;
  inline std::string get_tuple_name() const;
  inline set gist(set context) const;
  inline set gist_basic_set(basic_set context) const;
  inline set gist_params(set context) const;
  inline boolean has_dim_id(isl::dim type, unsigned int pos) const;
  inline boolean has_dim_name(isl::dim type, unsigned int pos) const;
  inline boolean has_equal_space(const set &set2) const;
  inline boolean has_tuple_id() const;
  inline boolean has_tuple_name() const;
  inline map identity() const;
  inline pw_aff indicator_function() const;
  inline set insert_dims(isl::dim type, unsigned int pos, unsigned int n) const;
  inline set intersect(set set2) const;
  inline set intersect_params(set params) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean is_bounded() const;
  inline boolean is_box() const;
  inline boolean is_disjoint(const set &set2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const set &set2) const;
  inline boolean is_params() const;
  inline boolean is_singleton() const;
  inline boolean is_strict_subset(const set &set2) const;
  inline boolean is_subset(const set &set2) const;
  inline boolean is_wrapping() const;
  inline map lex_ge_set(set set2) const;
  inline map lex_gt_set(set set2) const;
  inline map lex_le_set(set set2) const;
  inline map lex_lt_set(set set2) const;
  inline set lexmax() const;
  inline pw_multi_aff lexmax_pw_multi_aff() const;
  inline set lexmin() const;
  inline pw_multi_aff lexmin_pw_multi_aff() const;
  inline set lower_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline set lower_bound_val(isl::dim type, unsigned int pos, val value) const;
  inline val max_val(const aff &obj) const;
  inline val min_val(const aff &obj) const;
  inline set move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline int n_basic_set() const;
  inline unsigned int n_dim() const;
  static inline set nat_universe(space dim);
  inline set neg() const;
  inline set params() const;
  inline int plain_cmp(const set &set2) const;
  inline val plain_get_val_if_fixed(isl::dim type, unsigned int pos) const;
  inline boolean plain_is_disjoint(const set &set2) const;
  inline boolean plain_is_empty() const;
  inline boolean plain_is_equal(const set &set2) const;
  inline boolean plain_is_universe() const;
  inline basic_set plain_unshifted_simple_hull() const;
  inline basic_set polyhedral_hull() const;
  inline set preimage_multi_aff(multi_aff ma) const;
  inline set preimage_multi_pw_aff(multi_pw_aff mpa) const;
  inline set preimage_pw_multi_aff(pw_multi_aff pma) const;
  inline set product(set set2) const;
  inline map project_onto_map(isl::dim type, unsigned int first, unsigned int n) const;
  inline set project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline set remove_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline set remove_divs() const;
  inline set remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline set remove_redundancies() const;
  inline set remove_unknown_divs() const;
  inline set reset_space(space dim) const;
  inline set reset_tuple_id() const;
  inline set reset_user() const;
  inline basic_set sample() const;
  inline point sample_point() const;
  inline set set_dim_id(isl::dim type, unsigned int pos, id id) const;
  inline set set_tuple_id(id id) const;
  inline set set_tuple_name(const std::string &s) const;
  inline basic_set simple_hull() const;
  inline int size() const;
  inline basic_set solutions() const;
  inline set split_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline set subtract(set set2) const;
  inline set sum(set set2) const;
  inline set unite(set set2) const;
  static inline set universe(space space);
  inline basic_set unshifted_simple_hull() const;
  inline basic_set unshifted_simple_hull_from_set_list(set_list list) const;
  inline map unwrap() const;
  inline set upper_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline set upper_bound_val(isl::dim type, unsigned int pos, val value) const;
  inline map wrapped_domain_map() const;
};

// declarations for isl::set_list
inline set_list manage(__isl_take isl_set_list *ptr);
inline set_list manage_copy(__isl_keep isl_set_list *ptr);

class set_list {
  friend inline set_list manage(__isl_take isl_set_list *ptr);
  friend inline set_list manage_copy(__isl_keep isl_set_list *ptr);

  isl_set_list *ptr = nullptr;

  inline explicit set_list(__isl_take isl_set_list *ptr);

public:
  inline /* implicit */ set_list();
  inline /* implicit */ set_list(const set_list &obj);
  inline /* implicit */ set_list(std::nullptr_t);
  inline set_list &operator=(set_list obj);
  inline ~set_list();
  inline __isl_give isl_set_list *copy() const &;
  inline __isl_give isl_set_list *copy() && = delete;
  inline __isl_keep isl_set_list *get() const;
  inline __isl_give isl_set_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline set_list add(set el) const;
  static inline set_list alloc(ctx ctx, int n);
  inline set_list concat(set_list list2) const;
  inline set_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(set)> &fn) const;
  static inline set_list from_set(set el);
  inline set get_at(int index) const;
  inline set get_set(int index) const;
  inline set_list insert(unsigned int pos, set el) const;
  inline int n_set() const;
  inline set_list reverse() const;
  inline set_list set_set(int index, set el) const;
  inline int size() const;
  inline set_list swap(unsigned int pos1, unsigned int pos2) const;
  inline set unite() const;
};

// declarations for isl::space
inline space manage(__isl_take isl_space *ptr);
inline space manage_copy(__isl_keep isl_space *ptr);

class space {
  friend inline space manage(__isl_take isl_space *ptr);
  friend inline space manage_copy(__isl_keep isl_space *ptr);

  isl_space *ptr = nullptr;

  inline explicit space(__isl_take isl_space *ptr);

public:
  inline /* implicit */ space();
  inline /* implicit */ space(const space &obj);
  inline /* implicit */ space(std::nullptr_t);
  inline explicit space(ctx ctx, unsigned int nparam, unsigned int n_in, unsigned int n_out);
  inline explicit space(ctx ctx, unsigned int nparam, unsigned int dim);
  inline space &operator=(space obj);
  inline ~space();
  inline __isl_give isl_space *copy() const &;
  inline __isl_give isl_space *copy() && = delete;
  inline __isl_keep isl_space *get() const;
  inline __isl_give isl_space *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline space add_dims(isl::dim type, unsigned int n) const;
  inline space add_param_id(id id) const;
  inline space align_params(space dim2) const;
  inline boolean can_curry() const;
  inline boolean can_range_curry() const;
  inline boolean can_uncurry() const;
  inline boolean can_zip() const;
  inline space curry() const;
  inline unsigned int dim(isl::dim type) const;
  inline space domain() const;
  inline space domain_factor_domain() const;
  inline space domain_factor_range() const;
  inline boolean domain_is_wrapping() const;
  inline space domain_map() const;
  inline space domain_product(space right) const;
  inline space drop_dims(isl::dim type, unsigned int first, unsigned int num) const;
  inline space factor_domain() const;
  inline space factor_range() const;
  inline int find_dim_by_id(isl::dim type, const id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline space flatten_domain() const;
  inline space flatten_range() const;
  inline space from_domain() const;
  inline space from_range() const;
  inline id get_dim_id(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline boolean has_dim_id(isl::dim type, unsigned int pos) const;
  inline boolean has_dim_name(isl::dim type, unsigned int pos) const;
  inline boolean has_equal_params(const space &space2) const;
  inline boolean has_equal_tuples(const space &space2) const;
  inline boolean has_tuple_id(isl::dim type) const;
  inline boolean has_tuple_name(isl::dim type) const;
  inline space insert_dims(isl::dim type, unsigned int pos, unsigned int n) const;
  inline boolean is_domain(const space &space2) const;
  inline boolean is_equal(const space &space2) const;
  inline boolean is_map() const;
  inline boolean is_params() const;
  inline boolean is_product() const;
  inline boolean is_range(const space &space2) const;
  inline boolean is_set() const;
  inline boolean is_wrapping() const;
  inline space join(space right) const;
  inline space map_from_domain_and_range(space range) const;
  inline space map_from_set() const;
  inline space move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline space params() const;
  static inline space params_alloc(ctx ctx, unsigned int nparam);
  inline space product(space right) const;
  inline space range() const;
  inline space range_curry() const;
  inline space range_factor_domain() const;
  inline space range_factor_range() const;
  inline boolean range_is_wrapping() const;
  inline space range_map() const;
  inline space range_product(space right) const;
  inline space reset_tuple_id(isl::dim type) const;
  inline space reset_user() const;
  inline space reverse() const;
  inline space set_dim_id(isl::dim type, unsigned int pos, id id) const;
  inline space set_from_params() const;
  inline space set_tuple_id(isl::dim type, id id) const;
  inline space set_tuple_name(isl::dim type, const std::string &s) const;
  inline boolean tuple_is_equal(isl::dim type1, const space &space2, isl::dim type2) const;
  inline space uncurry() const;
  inline space unwrap() const;
  inline space wrap() const;
  inline space zip() const;
};

// declarations for isl::term
inline term manage(__isl_take isl_term *ptr);
inline term manage_copy(__isl_keep isl_term *ptr);

class term {
  friend inline term manage(__isl_take isl_term *ptr);
  friend inline term manage_copy(__isl_keep isl_term *ptr);

  isl_term *ptr = nullptr;

  inline explicit term(__isl_take isl_term *ptr);

public:
  inline /* implicit */ term();
  inline /* implicit */ term(const term &obj);
  inline /* implicit */ term(std::nullptr_t);
  inline term &operator=(term obj);
  inline ~term();
  inline __isl_give isl_term *copy() const &;
  inline __isl_give isl_term *copy() && = delete;
  inline __isl_keep isl_term *get() const;
  inline __isl_give isl_term *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

  inline unsigned int dim(isl::dim type) const;
  inline val get_coefficient_val() const;
  inline aff get_div(unsigned int pos) const;
  inline int get_exp(isl::dim type, unsigned int pos) const;
};

// declarations for isl::union_access_info
inline union_access_info manage(__isl_take isl_union_access_info *ptr);
inline union_access_info manage_copy(__isl_keep isl_union_access_info *ptr);

class union_access_info {
  friend inline union_access_info manage(__isl_take isl_union_access_info *ptr);
  friend inline union_access_info manage_copy(__isl_keep isl_union_access_info *ptr);

  isl_union_access_info *ptr = nullptr;

  inline explicit union_access_info(__isl_take isl_union_access_info *ptr);

public:
  inline /* implicit */ union_access_info();
  inline /* implicit */ union_access_info(const union_access_info &obj);
  inline /* implicit */ union_access_info(std::nullptr_t);
  inline explicit union_access_info(union_map sink);
  inline union_access_info &operator=(union_access_info obj);
  inline ~union_access_info();
  inline __isl_give isl_union_access_info *copy() const &;
  inline __isl_give isl_union_access_info *copy() && = delete;
  inline __isl_keep isl_union_access_info *get() const;
  inline __isl_give isl_union_access_info *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline union_flow compute_flow() const;
  inline union_access_info set_kill(union_map kill) const;
  inline union_access_info set_may_source(union_map may_source) const;
  inline union_access_info set_must_source(union_map must_source) const;
  inline union_access_info set_schedule(schedule schedule) const;
  inline union_access_info set_schedule_map(union_map schedule_map) const;
};

// declarations for isl::union_flow
inline union_flow manage(__isl_take isl_union_flow *ptr);
inline union_flow manage_copy(__isl_keep isl_union_flow *ptr);

class union_flow {
  friend inline union_flow manage(__isl_take isl_union_flow *ptr);
  friend inline union_flow manage_copy(__isl_keep isl_union_flow *ptr);

  isl_union_flow *ptr = nullptr;

  inline explicit union_flow(__isl_take isl_union_flow *ptr);

public:
  inline /* implicit */ union_flow();
  inline /* implicit */ union_flow(const union_flow &obj);
  inline /* implicit */ union_flow(std::nullptr_t);
  inline union_flow &operator=(union_flow obj);
  inline ~union_flow();
  inline __isl_give isl_union_flow *copy() const &;
  inline __isl_give isl_union_flow *copy() && = delete;
  inline __isl_keep isl_union_flow *get() const;
  inline __isl_give isl_union_flow *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline union_map get_full_may_dependence() const;
  inline union_map get_full_must_dependence() const;
  inline union_map get_may_dependence() const;
  inline union_map get_may_no_source() const;
  inline union_map get_must_dependence() const;
  inline union_map get_must_no_source() const;
};

// declarations for isl::union_map
inline union_map manage(__isl_take isl_union_map *ptr);
inline union_map manage_copy(__isl_keep isl_union_map *ptr);

class union_map {
  friend inline union_map manage(__isl_take isl_union_map *ptr);
  friend inline union_map manage_copy(__isl_keep isl_union_map *ptr);

  isl_union_map *ptr = nullptr;

  inline explicit union_map(__isl_take isl_union_map *ptr);

public:
  inline /* implicit */ union_map();
  inline /* implicit */ union_map(const union_map &obj);
  inline /* implicit */ union_map(std::nullptr_t);
  inline explicit union_map(union_pw_multi_aff upma);
  inline /* implicit */ union_map(basic_map bmap);
  inline /* implicit */ union_map(map map);
  inline explicit union_map(ctx ctx, const std::string &str);
  inline union_map &operator=(union_map obj);
  inline ~union_map();
  inline __isl_give isl_union_map *copy() const &;
  inline __isl_give isl_union_map *copy() && = delete;
  inline __isl_keep isl_union_map *get() const;
  inline __isl_give isl_union_map *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline union_map add_map(map map) const;
  inline union_map affine_hull() const;
  inline union_map align_params(space model) const;
  inline union_map apply_domain(union_map umap2) const;
  inline union_map apply_range(union_map umap2) const;
  inline union_map coalesce() const;
  inline boolean contains(const space &space) const;
  inline union_map curry() const;
  inline union_set deltas() const;
  inline union_map deltas_map() const;
  inline union_map detect_equalities() const;
  inline unsigned int dim(isl::dim type) const;
  inline union_set domain() const;
  inline union_map domain_factor_domain() const;
  inline union_map domain_factor_range() const;
  inline union_map domain_map() const;
  inline union_pw_multi_aff domain_map_union_pw_multi_aff() const;
  inline union_map domain_product(union_map umap2) const;
  static inline union_map empty(space space);
  inline union_map eq_at(multi_union_pw_aff mupa) const;
  inline map extract_map(space dim) const;
  inline union_map factor_domain() const;
  inline union_map factor_range() const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline union_map fixed_power(val exp) const;
  inline union_map flat_domain_product(union_map umap2) const;
  inline union_map flat_range_product(union_map umap2) const;
  inline stat foreach_map(const std::function<stat(map)> &fn) const;
  static inline union_map from(multi_union_pw_aff mupa);
  static inline union_map from_domain(union_set uset);
  static inline union_map from_domain_and_range(union_set domain, union_set range);
  static inline union_map from_range(union_set uset);
  static inline union_map from_union_pw_aff(union_pw_aff upa);
  inline id get_dim_id(isl::dim type, unsigned int pos) const;
  inline uint32_t get_hash() const;
  inline map_list get_map_list() const;
  inline space get_space() const;
  inline union_map gist(union_map context) const;
  inline union_map gist_domain(union_set uset) const;
  inline union_map gist_params(set set) const;
  inline union_map gist_range(union_set uset) const;
  inline union_map intersect(union_map umap2) const;
  inline union_map intersect_domain(union_set uset) const;
  inline union_map intersect_params(set set) const;
  inline union_map intersect_range(union_set uset) const;
  inline union_map intersect_range_factor_range(union_map factor) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean is_bijective() const;
  inline boolean is_disjoint(const union_map &umap2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const union_map &umap2) const;
  inline boolean is_identity() const;
  inline boolean is_injective() const;
  inline boolean is_single_valued() const;
  inline boolean is_strict_subset(const union_map &umap2) const;
  inline boolean is_subset(const union_map &umap2) const;
  inline union_map lex_ge_union_map(union_map umap2) const;
  inline union_map lex_gt_at_multi_union_pw_aff(multi_union_pw_aff mupa) const;
  inline union_map lex_gt_union_map(union_map umap2) const;
  inline union_map lex_le_union_map(union_map umap2) const;
  inline union_map lex_lt_at_multi_union_pw_aff(multi_union_pw_aff mupa) const;
  inline union_map lex_lt_union_map(union_map umap2) const;
  inline union_map lexmax() const;
  inline union_map lexmin() const;
  inline int n_map() const;
  inline set params() const;
  inline boolean plain_is_empty() const;
  inline boolean plain_is_injective() const;
  inline union_map polyhedral_hull() const;
  inline union_map preimage_domain_multi_aff(multi_aff ma) const;
  inline union_map preimage_domain_multi_pw_aff(multi_pw_aff mpa) const;
  inline union_map preimage_domain_pw_multi_aff(pw_multi_aff pma) const;
  inline union_map preimage_domain_union_pw_multi_aff(union_pw_multi_aff upma) const;
  inline union_map preimage_range_multi_aff(multi_aff ma) const;
  inline union_map preimage_range_pw_multi_aff(pw_multi_aff pma) const;
  inline union_map preimage_range_union_pw_multi_aff(union_pw_multi_aff upma) const;
  inline union_map product(union_map umap2) const;
  inline union_map project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline union_map project_out_all_params() const;
  inline union_set range() const;
  inline union_map range_curry() const;
  inline union_map range_factor_domain() const;
  inline union_map range_factor_range() const;
  inline union_map range_map() const;
  inline union_map range_product(union_map umap2) const;
  inline union_map remove_divs() const;
  inline union_map remove_redundancies() const;
  inline union_map reset_user() const;
  inline union_map reverse() const;
  inline basic_map sample() const;
  inline union_map simple_hull() const;
  inline union_map subtract(union_map umap2) const;
  inline union_map subtract_domain(union_set dom) const;
  inline union_map subtract_range(union_set dom) const;
  inline union_map uncurry() const;
  inline union_map unite(union_map umap2) const;
  inline union_map universe() const;
  inline union_set wrap() const;
  inline union_map zip() const;
};

// declarations for isl::union_map_list
inline union_map_list manage(__isl_take isl_union_map_list *ptr);
inline union_map_list manage_copy(__isl_keep isl_union_map_list *ptr);

class union_map_list {
  friend inline union_map_list manage(__isl_take isl_union_map_list *ptr);
  friend inline union_map_list manage_copy(__isl_keep isl_union_map_list *ptr);

  isl_union_map_list *ptr = nullptr;

  inline explicit union_map_list(__isl_take isl_union_map_list *ptr);

public:
  inline /* implicit */ union_map_list();
  inline /* implicit */ union_map_list(const union_map_list &obj);
  inline /* implicit */ union_map_list(std::nullptr_t);
  inline union_map_list &operator=(union_map_list obj);
  inline ~union_map_list();
  inline __isl_give isl_union_map_list *copy() const &;
  inline __isl_give isl_union_map_list *copy() && = delete;
  inline __isl_keep isl_union_map_list *get() const;
  inline __isl_give isl_union_map_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline union_map_list add(union_map el) const;
  static inline union_map_list alloc(ctx ctx, int n);
  inline union_map_list concat(union_map_list list2) const;
  inline union_map_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(union_map)> &fn) const;
  static inline union_map_list from_union_map(union_map el);
  inline union_map get_at(int index) const;
  inline union_map get_union_map(int index) const;
  inline union_map_list insert(unsigned int pos, union_map el) const;
  inline int n_union_map() const;
  inline union_map_list reverse() const;
  inline union_map_list set_union_map(int index, union_map el) const;
  inline int size() const;
  inline union_map_list swap(unsigned int pos1, unsigned int pos2) const;
};

// declarations for isl::union_pw_aff
inline union_pw_aff manage(__isl_take isl_union_pw_aff *ptr);
inline union_pw_aff manage_copy(__isl_keep isl_union_pw_aff *ptr);

class union_pw_aff {
  friend inline union_pw_aff manage(__isl_take isl_union_pw_aff *ptr);
  friend inline union_pw_aff manage_copy(__isl_keep isl_union_pw_aff *ptr);

  isl_union_pw_aff *ptr = nullptr;

  inline explicit union_pw_aff(__isl_take isl_union_pw_aff *ptr);

public:
  inline /* implicit */ union_pw_aff();
  inline /* implicit */ union_pw_aff(const union_pw_aff &obj);
  inline /* implicit */ union_pw_aff(std::nullptr_t);
  inline /* implicit */ union_pw_aff(pw_aff pa);
  inline explicit union_pw_aff(union_set domain, val v);
  inline explicit union_pw_aff(ctx ctx, const std::string &str);
  inline union_pw_aff &operator=(union_pw_aff obj);
  inline ~union_pw_aff();
  inline __isl_give isl_union_pw_aff *copy() const &;
  inline __isl_give isl_union_pw_aff *copy() && = delete;
  inline __isl_keep isl_union_pw_aff *get() const;
  inline __isl_give isl_union_pw_aff *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline union_pw_aff add(union_pw_aff upa2) const;
  inline union_pw_aff add_pw_aff(pw_aff pa) const;
  static inline union_pw_aff aff_on_domain(union_set domain, aff aff);
  inline union_pw_aff align_params(space model) const;
  inline union_pw_aff coalesce() const;
  inline unsigned int dim(isl::dim type) const;
  inline union_set domain() const;
  inline union_pw_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  static inline union_pw_aff empty(space space);
  inline pw_aff extract_pw_aff(space space) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline union_pw_aff floor() const;
  inline stat foreach_pw_aff(const std::function<stat(pw_aff)> &fn) const;
  inline pw_aff_list get_pw_aff_list() const;
  inline space get_space() const;
  inline union_pw_aff gist(union_set context) const;
  inline union_pw_aff gist_params(set context) const;
  inline union_pw_aff intersect_domain(union_set uset) const;
  inline union_pw_aff intersect_params(set set) const;
  inline boolean involves_nan() const;
  inline val max_val() const;
  inline val min_val() const;
  inline union_pw_aff mod_val(val f) const;
  inline int n_pw_aff() const;
  inline union_pw_aff neg() const;
  static inline union_pw_aff param_on_domain_id(union_set domain, id id);
  inline boolean plain_is_equal(const union_pw_aff &upa2) const;
  inline union_pw_aff pullback(union_pw_multi_aff upma) const;
  static inline union_pw_aff pw_aff_on_domain(union_set domain, pw_aff pa);
  inline union_pw_aff reset_user() const;
  inline union_pw_aff scale_down_val(val v) const;
  inline union_pw_aff scale_val(val v) const;
  inline union_pw_aff sub(union_pw_aff upa2) const;
  inline union_pw_aff subtract_domain(union_set uset) const;
  inline union_pw_aff union_add(union_pw_aff upa2) const;
  inline union_set zero_union_set() const;
};

// declarations for isl::union_pw_aff_list
inline union_pw_aff_list manage(__isl_take isl_union_pw_aff_list *ptr);
inline union_pw_aff_list manage_copy(__isl_keep isl_union_pw_aff_list *ptr);

class union_pw_aff_list {
  friend inline union_pw_aff_list manage(__isl_take isl_union_pw_aff_list *ptr);
  friend inline union_pw_aff_list manage_copy(__isl_keep isl_union_pw_aff_list *ptr);

  isl_union_pw_aff_list *ptr = nullptr;

  inline explicit union_pw_aff_list(__isl_take isl_union_pw_aff_list *ptr);

public:
  inline /* implicit */ union_pw_aff_list();
  inline /* implicit */ union_pw_aff_list(const union_pw_aff_list &obj);
  inline /* implicit */ union_pw_aff_list(std::nullptr_t);
  inline union_pw_aff_list &operator=(union_pw_aff_list obj);
  inline ~union_pw_aff_list();
  inline __isl_give isl_union_pw_aff_list *copy() const &;
  inline __isl_give isl_union_pw_aff_list *copy() && = delete;
  inline __isl_keep isl_union_pw_aff_list *get() const;
  inline __isl_give isl_union_pw_aff_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline union_pw_aff_list add(union_pw_aff el) const;
  static inline union_pw_aff_list alloc(ctx ctx, int n);
  inline union_pw_aff_list concat(union_pw_aff_list list2) const;
  inline union_pw_aff_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(union_pw_aff)> &fn) const;
  static inline union_pw_aff_list from_union_pw_aff(union_pw_aff el);
  inline union_pw_aff get_at(int index) const;
  inline union_pw_aff get_union_pw_aff(int index) const;
  inline union_pw_aff_list insert(unsigned int pos, union_pw_aff el) const;
  inline int n_union_pw_aff() const;
  inline union_pw_aff_list reverse() const;
  inline union_pw_aff_list set_union_pw_aff(int index, union_pw_aff el) const;
  inline int size() const;
  inline union_pw_aff_list swap(unsigned int pos1, unsigned int pos2) const;
};

// declarations for isl::union_pw_multi_aff
inline union_pw_multi_aff manage(__isl_take isl_union_pw_multi_aff *ptr);
inline union_pw_multi_aff manage_copy(__isl_keep isl_union_pw_multi_aff *ptr);

class union_pw_multi_aff {
  friend inline union_pw_multi_aff manage(__isl_take isl_union_pw_multi_aff *ptr);
  friend inline union_pw_multi_aff manage_copy(__isl_keep isl_union_pw_multi_aff *ptr);

  isl_union_pw_multi_aff *ptr = nullptr;

  inline explicit union_pw_multi_aff(__isl_take isl_union_pw_multi_aff *ptr);

public:
  inline /* implicit */ union_pw_multi_aff();
  inline /* implicit */ union_pw_multi_aff(const union_pw_multi_aff &obj);
  inline /* implicit */ union_pw_multi_aff(std::nullptr_t);
  inline /* implicit */ union_pw_multi_aff(aff aff);
  inline /* implicit */ union_pw_multi_aff(pw_multi_aff pma);
  inline explicit union_pw_multi_aff(union_set uset);
  inline explicit union_pw_multi_aff(union_map umap);
  inline explicit union_pw_multi_aff(ctx ctx, const std::string &str);
  inline /* implicit */ union_pw_multi_aff(union_pw_aff upa);
  inline explicit union_pw_multi_aff(multi_union_pw_aff mupa);
  inline union_pw_multi_aff &operator=(union_pw_multi_aff obj);
  inline ~union_pw_multi_aff();
  inline __isl_give isl_union_pw_multi_aff *copy() const &;
  inline __isl_give isl_union_pw_multi_aff *copy() && = delete;
  inline __isl_keep isl_union_pw_multi_aff *get() const;
  inline __isl_give isl_union_pw_multi_aff *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline union_pw_multi_aff add(union_pw_multi_aff upma2) const;
  inline union_pw_multi_aff add_pw_multi_aff(pw_multi_aff pma) const;
  inline union_pw_multi_aff align_params(space model) const;
  inline union_pw_multi_aff coalesce() const;
  inline unsigned int dim(isl::dim type) const;
  inline union_set domain() const;
  inline union_pw_multi_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  static inline union_pw_multi_aff empty(space space);
  inline pw_multi_aff extract_pw_multi_aff(space space) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline union_pw_multi_aff flat_range_product(union_pw_multi_aff upma2) const;
  inline stat foreach_pw_multi_aff(const std::function<stat(pw_multi_aff)> &fn) const;
  static inline union_pw_multi_aff from_union_set(union_set uset);
  inline pw_multi_aff_list get_pw_multi_aff_list() const;
  inline space get_space() const;
  inline union_pw_aff get_union_pw_aff(int pos) const;
  inline union_pw_multi_aff gist(union_set context) const;
  inline union_pw_multi_aff gist_params(set context) const;
  inline union_pw_multi_aff intersect_domain(union_set uset) const;
  inline union_pw_multi_aff intersect_params(set set) const;
  inline boolean involves_nan() const;
  static inline union_pw_multi_aff multi_val_on_domain(union_set domain, multi_val mv);
  inline int n_pw_multi_aff() const;
  inline union_pw_multi_aff neg() const;
  inline boolean plain_is_equal(const union_pw_multi_aff &upma2) const;
  inline union_pw_multi_aff pullback(union_pw_multi_aff upma2) const;
  inline union_pw_multi_aff reset_user() const;
  inline union_pw_multi_aff scale_down_val(val val) const;
  inline union_pw_multi_aff scale_multi_val(multi_val mv) const;
  inline union_pw_multi_aff scale_val(val val) const;
  inline union_pw_multi_aff sub(union_pw_multi_aff upma2) const;
  inline union_pw_multi_aff subtract_domain(union_set uset) const;
  inline union_pw_multi_aff union_add(union_pw_multi_aff upma2) const;
};

// declarations for isl::union_pw_multi_aff_list
inline union_pw_multi_aff_list manage(__isl_take isl_union_pw_multi_aff_list *ptr);
inline union_pw_multi_aff_list manage_copy(__isl_keep isl_union_pw_multi_aff_list *ptr);

class union_pw_multi_aff_list {
  friend inline union_pw_multi_aff_list manage(__isl_take isl_union_pw_multi_aff_list *ptr);
  friend inline union_pw_multi_aff_list manage_copy(__isl_keep isl_union_pw_multi_aff_list *ptr);

  isl_union_pw_multi_aff_list *ptr = nullptr;

  inline explicit union_pw_multi_aff_list(__isl_take isl_union_pw_multi_aff_list *ptr);

public:
  inline /* implicit */ union_pw_multi_aff_list();
  inline /* implicit */ union_pw_multi_aff_list(const union_pw_multi_aff_list &obj);
  inline /* implicit */ union_pw_multi_aff_list(std::nullptr_t);
  inline union_pw_multi_aff_list &operator=(union_pw_multi_aff_list obj);
  inline ~union_pw_multi_aff_list();
  inline __isl_give isl_union_pw_multi_aff_list *copy() const &;
  inline __isl_give isl_union_pw_multi_aff_list *copy() && = delete;
  inline __isl_keep isl_union_pw_multi_aff_list *get() const;
  inline __isl_give isl_union_pw_multi_aff_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline union_pw_multi_aff_list add(union_pw_multi_aff el) const;
  static inline union_pw_multi_aff_list alloc(ctx ctx, int n);
  inline union_pw_multi_aff_list concat(union_pw_multi_aff_list list2) const;
  inline union_pw_multi_aff_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(union_pw_multi_aff)> &fn) const;
  static inline union_pw_multi_aff_list from_union_pw_multi_aff(union_pw_multi_aff el);
  inline union_pw_multi_aff get_at(int index) const;
  inline union_pw_multi_aff get_union_pw_multi_aff(int index) const;
  inline union_pw_multi_aff_list insert(unsigned int pos, union_pw_multi_aff el) const;
  inline int n_union_pw_multi_aff() const;
  inline union_pw_multi_aff_list reverse() const;
  inline union_pw_multi_aff_list set_union_pw_multi_aff(int index, union_pw_multi_aff el) const;
  inline int size() const;
  inline union_pw_multi_aff_list swap(unsigned int pos1, unsigned int pos2) const;
};

// declarations for isl::union_pw_qpolynomial
inline union_pw_qpolynomial manage(__isl_take isl_union_pw_qpolynomial *ptr);
inline union_pw_qpolynomial manage_copy(__isl_keep isl_union_pw_qpolynomial *ptr);

class union_pw_qpolynomial {
  friend inline union_pw_qpolynomial manage(__isl_take isl_union_pw_qpolynomial *ptr);
  friend inline union_pw_qpolynomial manage_copy(__isl_keep isl_union_pw_qpolynomial *ptr);

  isl_union_pw_qpolynomial *ptr = nullptr;

  inline explicit union_pw_qpolynomial(__isl_take isl_union_pw_qpolynomial *ptr);

public:
  inline /* implicit */ union_pw_qpolynomial();
  inline /* implicit */ union_pw_qpolynomial(const union_pw_qpolynomial &obj);
  inline /* implicit */ union_pw_qpolynomial(std::nullptr_t);
  inline explicit union_pw_qpolynomial(ctx ctx, const std::string &str);
  inline union_pw_qpolynomial &operator=(union_pw_qpolynomial obj);
  inline ~union_pw_qpolynomial();
  inline __isl_give isl_union_pw_qpolynomial *copy() const &;
  inline __isl_give isl_union_pw_qpolynomial *copy() && = delete;
  inline __isl_keep isl_union_pw_qpolynomial *get() const;
  inline __isl_give isl_union_pw_qpolynomial *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline union_pw_qpolynomial add(union_pw_qpolynomial upwqp2) const;
  inline union_pw_qpolynomial add_pw_qpolynomial(pw_qpolynomial pwqp) const;
  inline union_pw_qpolynomial align_params(space model) const;
  inline union_pw_qpolynomial coalesce() const;
  inline unsigned int dim(isl::dim type) const;
  inline union_set domain() const;
  inline union_pw_qpolynomial drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline val eval(point pnt) const;
  inline pw_qpolynomial extract_pw_qpolynomial(space dim) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline stat foreach_pw_qpolynomial(const std::function<stat(pw_qpolynomial)> &fn) const;
  static inline union_pw_qpolynomial from_pw_qpolynomial(pw_qpolynomial pwqp);
  inline pw_qpolynomial_list get_pw_qpolynomial_list() const;
  inline space get_space() const;
  inline union_pw_qpolynomial gist(union_set context) const;
  inline union_pw_qpolynomial gist_params(set context) const;
  inline union_pw_qpolynomial intersect_domain(union_set uset) const;
  inline union_pw_qpolynomial intersect_params(set set) const;
  inline boolean involves_nan() const;
  inline union_pw_qpolynomial mul(union_pw_qpolynomial upwqp2) const;
  inline int n_pw_qpolynomial() const;
  inline union_pw_qpolynomial neg() const;
  inline boolean plain_is_equal(const union_pw_qpolynomial &upwqp2) const;
  inline union_pw_qpolynomial reset_user() const;
  inline union_pw_qpolynomial scale_down_val(val v) const;
  inline union_pw_qpolynomial scale_val(val v) const;
  inline union_pw_qpolynomial sub(union_pw_qpolynomial upwqp2) const;
  inline union_pw_qpolynomial subtract_domain(union_set uset) const;
  inline union_pw_qpolynomial to_polynomial(int sign) const;
  static inline union_pw_qpolynomial zero(space dim);
};

// declarations for isl::union_set
inline union_set manage(__isl_take isl_union_set *ptr);
inline union_set manage_copy(__isl_keep isl_union_set *ptr);

class union_set {
  friend inline union_set manage(__isl_take isl_union_set *ptr);
  friend inline union_set manage_copy(__isl_keep isl_union_set *ptr);

  isl_union_set *ptr = nullptr;

  inline explicit union_set(__isl_take isl_union_set *ptr);

public:
  inline /* implicit */ union_set();
  inline /* implicit */ union_set(const union_set &obj);
  inline /* implicit */ union_set(std::nullptr_t);
  inline /* implicit */ union_set(basic_set bset);
  inline /* implicit */ union_set(set set);
  inline /* implicit */ union_set(point pnt);
  inline explicit union_set(ctx ctx, const std::string &str);
  inline union_set &operator=(union_set obj);
  inline ~union_set();
  inline __isl_give isl_union_set *copy() const &;
  inline __isl_give isl_union_set *copy() && = delete;
  inline __isl_keep isl_union_set *get() const;
  inline __isl_give isl_union_set *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline union_set add_set(set set) const;
  inline union_set affine_hull() const;
  inline union_set align_params(space model) const;
  inline union_set apply(union_map umap) const;
  inline union_set coalesce() const;
  inline union_set coefficients() const;
  inline schedule compute_schedule(union_map validity, union_map proximity) const;
  inline boolean contains(const space &space) const;
  inline union_set detect_equalities() const;
  inline unsigned int dim(isl::dim type) const;
  static inline union_set empty(space space);
  inline set extract_set(space dim) const;
  inline stat foreach_point(const std::function<stat(point)> &fn) const;
  inline stat foreach_set(const std::function<stat(set)> &fn) const;
  inline basic_set_list get_basic_set_list() const;
  inline uint32_t get_hash() const;
  inline set_list get_set_list() const;
  inline space get_space() const;
  inline union_set gist(union_set context) const;
  inline union_set gist_params(set set) const;
  inline union_map identity() const;
  inline union_pw_multi_aff identity_union_pw_multi_aff() const;
  inline union_set intersect(union_set uset2) const;
  inline union_set intersect_params(set set) const;
  inline boolean is_disjoint(const union_set &uset2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const union_set &uset2) const;
  inline boolean is_params() const;
  inline boolean is_strict_subset(const union_set &uset2) const;
  inline boolean is_subset(const union_set &uset2) const;
  inline union_map lex_ge_union_set(union_set uset2) const;
  inline union_map lex_gt_union_set(union_set uset2) const;
  inline union_map lex_le_union_set(union_set uset2) const;
  inline union_map lex_lt_union_set(union_set uset2) const;
  inline union_set lexmax() const;
  inline union_set lexmin() const;
  inline multi_val min_multi_union_pw_aff(const multi_union_pw_aff &obj) const;
  inline int n_set() const;
  inline set params() const;
  inline union_set polyhedral_hull() const;
  inline union_set preimage(multi_aff ma) const;
  inline union_set preimage(pw_multi_aff pma) const;
  inline union_set preimage(union_pw_multi_aff upma) const;
  inline union_set product(union_set uset2) const;
  inline union_set project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline union_set remove_divs() const;
  inline union_set remove_redundancies() const;
  inline union_set reset_user() const;
  inline basic_set sample() const;
  inline point sample_point() const;
  inline union_set simple_hull() const;
  inline union_set solutions() const;
  inline union_set subtract(union_set uset2) const;
  inline union_set unite(union_set uset2) const;
  inline union_set universe() const;
  inline union_map unwrap() const;
  inline union_map wrapped_domain_map() const;
};

// declarations for isl::union_set_list
inline union_set_list manage(__isl_take isl_union_set_list *ptr);
inline union_set_list manage_copy(__isl_keep isl_union_set_list *ptr);

class union_set_list {
  friend inline union_set_list manage(__isl_take isl_union_set_list *ptr);
  friend inline union_set_list manage_copy(__isl_keep isl_union_set_list *ptr);

  isl_union_set_list *ptr = nullptr;

  inline explicit union_set_list(__isl_take isl_union_set_list *ptr);

public:
  inline /* implicit */ union_set_list();
  inline /* implicit */ union_set_list(const union_set_list &obj);
  inline /* implicit */ union_set_list(std::nullptr_t);
  inline union_set_list &operator=(union_set_list obj);
  inline ~union_set_list();
  inline __isl_give isl_union_set_list *copy() const &;
  inline __isl_give isl_union_set_list *copy() && = delete;
  inline __isl_keep isl_union_set_list *get() const;
  inline __isl_give isl_union_set_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline union_set_list add(union_set el) const;
  static inline union_set_list alloc(ctx ctx, int n);
  inline union_set_list concat(union_set_list list2) const;
  inline union_set_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(union_set)> &fn) const;
  static inline union_set_list from_union_set(union_set el);
  inline union_set get_at(int index) const;
  inline union_set get_union_set(int index) const;
  inline union_set_list insert(unsigned int pos, union_set el) const;
  inline int n_union_set() const;
  inline union_set_list reverse() const;
  inline union_set_list set_union_set(int index, union_set el) const;
  inline int size() const;
  inline union_set_list swap(unsigned int pos1, unsigned int pos2) const;
  inline union_set unite() const;
};

// declarations for isl::val
inline val manage(__isl_take isl_val *ptr);
inline val manage_copy(__isl_keep isl_val *ptr);

class val {
  friend inline val manage(__isl_take isl_val *ptr);
  friend inline val manage_copy(__isl_keep isl_val *ptr);

  isl_val *ptr = nullptr;

  inline explicit val(__isl_take isl_val *ptr);

public:
  inline /* implicit */ val();
  inline /* implicit */ val(const val &obj);
  inline /* implicit */ val(std::nullptr_t);
  inline explicit val(ctx ctx, const std::string &str);
  inline explicit val(ctx ctx, long i);
  inline val &operator=(val obj);
  inline ~val();
  inline __isl_give isl_val *copy() const &;
  inline __isl_give isl_val *copy() && = delete;
  inline __isl_keep isl_val *get() const;
  inline __isl_give isl_val *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline val abs() const;
  inline boolean abs_eq(const val &v2) const;
  inline val add(val v2) const;
  inline val add_ui(unsigned long v2) const;
  inline val ceil() const;
  inline int cmp_si(long i) const;
  inline val div(val v2) const;
  inline val div_ui(unsigned long v2) const;
  inline boolean eq(const val &v2) const;
  inline val floor() const;
  inline val gcd(val v2) const;
  inline boolean ge(const val &v2) const;
  inline uint32_t get_hash() const;
  inline long get_num_si() const;
  inline boolean gt(const val &v2) const;
  inline boolean gt_si(long i) const;
  static inline val infty(ctx ctx);
  static inline val int_from_ui(ctx ctx, unsigned long u);
  inline val inv() const;
  inline boolean is_divisible_by(const val &v2) const;
  inline boolean is_infty() const;
  inline boolean is_int() const;
  inline boolean is_nan() const;
  inline boolean is_neg() const;
  inline boolean is_neginfty() const;
  inline boolean is_negone() const;
  inline boolean is_nonneg() const;
  inline boolean is_nonpos() const;
  inline boolean is_one() const;
  inline boolean is_pos() const;
  inline boolean is_rat() const;
  inline boolean is_zero() const;
  inline boolean le(const val &v2) const;
  inline boolean lt(const val &v2) const;
  inline val max(val v2) const;
  inline val min(val v2) const;
  inline val mod(val v2) const;
  inline val mul(val v2) const;
  inline val mul_ui(unsigned long v2) const;
  inline size_t n_abs_num_chunks(size_t size) const;
  static inline val nan(ctx ctx);
  inline boolean ne(const val &v2) const;
  inline val neg() const;
  static inline val neginfty(ctx ctx);
  static inline val negone(ctx ctx);
  static inline val one(ctx ctx);
  inline val pow2() const;
  inline val set_si(long i) const;
  inline int sgn() const;
  inline val sub(val v2) const;
  inline val sub_ui(unsigned long v2) const;
  inline val trunc() const;
  static inline val zero(ctx ctx);
};

// declarations for isl::val_list
inline val_list manage(__isl_take isl_val_list *ptr);
inline val_list manage_copy(__isl_keep isl_val_list *ptr);

class val_list {
  friend inline val_list manage(__isl_take isl_val_list *ptr);
  friend inline val_list manage_copy(__isl_keep isl_val_list *ptr);

  isl_val_list *ptr = nullptr;

  inline explicit val_list(__isl_take isl_val_list *ptr);

public:
  inline /* implicit */ val_list();
  inline /* implicit */ val_list(const val_list &obj);
  inline /* implicit */ val_list(std::nullptr_t);
  inline val_list &operator=(val_list obj);
  inline ~val_list();
  inline __isl_give isl_val_list *copy() const &;
  inline __isl_give isl_val_list *copy() && = delete;
  inline __isl_keep isl_val_list *get() const;
  inline __isl_give isl_val_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline val_list add(val el) const;
  static inline val_list alloc(ctx ctx, int n);
  inline val_list concat(val_list list2) const;
  inline val_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(val)> &fn) const;
  static inline val_list from_val(val el);
  inline val get_at(int index) const;
  inline val get_val(int index) const;
  inline val_list insert(unsigned int pos, val el) const;
  inline int n_val() const;
  inline val_list reverse() const;
  inline val_list set_val(int index, val el) const;
  inline int size() const;
  inline val_list swap(unsigned int pos1, unsigned int pos2) const;
};

// declarations for isl::vec
inline vec manage(__isl_take isl_vec *ptr);
inline vec manage_copy(__isl_keep isl_vec *ptr);

class vec {
  friend inline vec manage(__isl_take isl_vec *ptr);
  friend inline vec manage_copy(__isl_keep isl_vec *ptr);

  isl_vec *ptr = nullptr;

  inline explicit vec(__isl_take isl_vec *ptr);

public:
  inline /* implicit */ vec();
  inline /* implicit */ vec(const vec &obj);
  inline /* implicit */ vec(std::nullptr_t);
  inline vec &operator=(vec obj);
  inline ~vec();
  inline __isl_give isl_vec *copy() const &;
  inline __isl_give isl_vec *copy() && = delete;
  inline __isl_keep isl_vec *get() const;
  inline __isl_give isl_vec *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline void dump() const;

  inline vec add(vec vec2) const;
  inline vec add_els(unsigned int n) const;
  static inline vec alloc(ctx ctx, unsigned int size);
  inline vec ceil() const;
  inline vec clr() const;
  inline int cmp_element(const vec &vec2, int pos) const;
  inline vec concat(vec vec2) const;
  inline vec drop_els(unsigned int pos, unsigned int n) const;
  inline vec extend(unsigned int size) const;
  inline val get_element_val(int pos) const;
  inline vec insert_els(unsigned int pos, unsigned int n) const;
  inline vec insert_zero_els(unsigned int pos, unsigned int n) const;
  inline boolean is_equal(const vec &vec2) const;
  inline vec mat_product(mat mat) const;
  inline vec move_els(unsigned int dst_col, unsigned int src_col, unsigned int n) const;
  inline vec neg() const;
  inline vec set_element_si(int pos, int v) const;
  inline vec set_element_val(int pos, val v) const;
  inline vec set_si(int v) const;
  inline vec set_val(val v) const;
  inline int size() const;
  inline vec sort() const;
  static inline vec zero(ctx ctx, unsigned int size);
  inline vec zero_extend(unsigned int size) const;
};

// implementations for isl::aff
aff manage(__isl_take isl_aff *ptr) {
  return aff(ptr);
}
aff manage_copy(__isl_keep isl_aff *ptr) {
  ptr = isl_aff_copy(ptr);
  return aff(ptr);
}

aff::aff()
    : ptr(nullptr) {}

aff::aff(const aff &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
aff::aff(std::nullptr_t)
    : ptr(nullptr) {}


aff::aff(__isl_take isl_aff *ptr)
    : ptr(ptr) {}

aff::aff(local_space ls)
{
  auto res = isl_aff_zero_on_domain(ls.release());
  ptr = res;
}
aff::aff(local_space ls, val val)
{
  auto res = isl_aff_val_on_domain(ls.release(), val.release());
  ptr = res;
}
aff::aff(ctx ctx, const std::string &str)
{
  auto res = isl_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

aff &aff::operator=(aff obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

aff::~aff() {
  if (ptr)
    isl_aff_free(ptr);
}

__isl_give isl_aff *aff::copy() const & {
  return isl_aff_copy(ptr);
}

__isl_keep isl_aff *aff::get() const {
  return ptr;
}

__isl_give isl_aff *aff::release() {
  isl_aff *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool aff::is_null() const {
  return ptr == nullptr;
}
aff::operator bool() const {
  return !is_null();
}


ctx aff::get_ctx() const {
  return ctx(isl_aff_get_ctx(ptr));
}
std::string aff::to_str() const {
  char *Tmp = isl_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void aff::dump() const {
  isl_aff_dump(get());
}


aff aff::add(aff aff2) const
{
  auto res = isl_aff_add(copy(), aff2.release());
  return manage(res);
}

aff aff::add_coefficient_si(isl::dim type, int pos, int v) const
{
  auto res = isl_aff_add_coefficient_si(copy(), static_cast<enum isl_dim_type>(type), pos, v);
  return manage(res);
}

aff aff::add_coefficient_val(isl::dim type, int pos, val v) const
{
  auto res = isl_aff_add_coefficient_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

aff aff::add_constant_num_si(int v) const
{
  auto res = isl_aff_add_constant_num_si(copy(), v);
  return manage(res);
}

aff aff::add_constant_si(int v) const
{
  auto res = isl_aff_add_constant_si(copy(), v);
  return manage(res);
}

aff aff::add_constant_val(val v) const
{
  auto res = isl_aff_add_constant_val(copy(), v.release());
  return manage(res);
}

aff aff::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_aff_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

aff aff::align_params(space model) const
{
  auto res = isl_aff_align_params(copy(), model.release());
  return manage(res);
}

aff aff::ceil() const
{
  auto res = isl_aff_ceil(copy());
  return manage(res);
}

int aff::coefficient_sgn(isl::dim type, int pos) const
{
  auto res = isl_aff_coefficient_sgn(get(), static_cast<enum isl_dim_type>(type), pos);
  return res;
}

int aff::dim(isl::dim type) const
{
  auto res = isl_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

aff aff::div(aff aff2) const
{
  auto res = isl_aff_div(copy(), aff2.release());
  return manage(res);
}

aff aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

basic_set aff::eq_basic_set(aff aff2) const
{
  auto res = isl_aff_eq_basic_set(copy(), aff2.release());
  return manage(res);
}

set aff::eq_set(aff aff2) const
{
  auto res = isl_aff_eq_set(copy(), aff2.release());
  return manage(res);
}

val aff::eval(point pnt) const
{
  auto res = isl_aff_eval(copy(), pnt.release());
  return manage(res);
}

int aff::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

aff aff::floor() const
{
  auto res = isl_aff_floor(copy());
  return manage(res);
}

aff aff::from_range() const
{
  auto res = isl_aff_from_range(copy());
  return manage(res);
}

basic_set aff::ge_basic_set(aff aff2) const
{
  auto res = isl_aff_ge_basic_set(copy(), aff2.release());
  return manage(res);
}

set aff::ge_set(aff aff2) const
{
  auto res = isl_aff_ge_set(copy(), aff2.release());
  return manage(res);
}

val aff::get_coefficient_val(isl::dim type, int pos) const
{
  auto res = isl_aff_get_coefficient_val(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

val aff::get_constant_val() const
{
  auto res = isl_aff_get_constant_val(get());
  return manage(res);
}

val aff::get_denominator_val() const
{
  auto res = isl_aff_get_denominator_val(get());
  return manage(res);
}

std::string aff::get_dim_name(isl::dim type, unsigned int pos) const
{
  auto res = isl_aff_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

aff aff::get_div(int pos) const
{
  auto res = isl_aff_get_div(get(), pos);
  return manage(res);
}

local_space aff::get_domain_local_space() const
{
  auto res = isl_aff_get_domain_local_space(get());
  return manage(res);
}

space aff::get_domain_space() const
{
  auto res = isl_aff_get_domain_space(get());
  return manage(res);
}

uint32_t aff::get_hash() const
{
  auto res = isl_aff_get_hash(get());
  return res;
}

local_space aff::get_local_space() const
{
  auto res = isl_aff_get_local_space(get());
  return manage(res);
}

space aff::get_space() const
{
  auto res = isl_aff_get_space(get());
  return manage(res);
}

aff aff::gist(set context) const
{
  auto res = isl_aff_gist(copy(), context.release());
  return manage(res);
}

aff aff::gist_params(set context) const
{
  auto res = isl_aff_gist_params(copy(), context.release());
  return manage(res);
}

basic_set aff::gt_basic_set(aff aff2) const
{
  auto res = isl_aff_gt_basic_set(copy(), aff2.release());
  return manage(res);
}

set aff::gt_set(aff aff2) const
{
  auto res = isl_aff_gt_set(copy(), aff2.release());
  return manage(res);
}

aff aff::insert_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_aff_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean aff::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_aff_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean aff::is_cst() const
{
  auto res = isl_aff_is_cst(get());
  return manage(res);
}

boolean aff::is_nan() const
{
  auto res = isl_aff_is_nan(get());
  return manage(res);
}

basic_set aff::le_basic_set(aff aff2) const
{
  auto res = isl_aff_le_basic_set(copy(), aff2.release());
  return manage(res);
}

set aff::le_set(aff aff2) const
{
  auto res = isl_aff_le_set(copy(), aff2.release());
  return manage(res);
}

basic_set aff::lt_basic_set(aff aff2) const
{
  auto res = isl_aff_lt_basic_set(copy(), aff2.release());
  return manage(res);
}

set aff::lt_set(aff aff2) const
{
  auto res = isl_aff_lt_set(copy(), aff2.release());
  return manage(res);
}

aff aff::mod(val mod) const
{
  auto res = isl_aff_mod_val(copy(), mod.release());
  return manage(res);
}

aff aff::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_aff_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

aff aff::mul(aff aff2) const
{
  auto res = isl_aff_mul(copy(), aff2.release());
  return manage(res);
}

aff aff::nan_on_domain(local_space ls)
{
  auto res = isl_aff_nan_on_domain(ls.release());
  return manage(res);
}

set aff::ne_set(aff aff2) const
{
  auto res = isl_aff_ne_set(copy(), aff2.release());
  return manage(res);
}

aff aff::neg() const
{
  auto res = isl_aff_neg(copy());
  return manage(res);
}

basic_set aff::neg_basic_set() const
{
  auto res = isl_aff_neg_basic_set(copy());
  return manage(res);
}

aff aff::param_on_domain_space_id(space space, id id)
{
  auto res = isl_aff_param_on_domain_space_id(space.release(), id.release());
  return manage(res);
}

boolean aff::plain_is_equal(const aff &aff2) const
{
  auto res = isl_aff_plain_is_equal(get(), aff2.get());
  return manage(res);
}

boolean aff::plain_is_zero() const
{
  auto res = isl_aff_plain_is_zero(get());
  return manage(res);
}

aff aff::project_domain_on_params() const
{
  auto res = isl_aff_project_domain_on_params(copy());
  return manage(res);
}

aff aff::pullback(multi_aff ma) const
{
  auto res = isl_aff_pullback_multi_aff(copy(), ma.release());
  return manage(res);
}

aff aff::pullback_aff(aff aff2) const
{
  auto res = isl_aff_pullback_aff(copy(), aff2.release());
  return manage(res);
}

aff aff::scale(val v) const
{
  auto res = isl_aff_scale_val(copy(), v.release());
  return manage(res);
}

aff aff::scale_down(val v) const
{
  auto res = isl_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

aff aff::scale_down_ui(unsigned int f) const
{
  auto res = isl_aff_scale_down_ui(copy(), f);
  return manage(res);
}

aff aff::set_coefficient_si(isl::dim type, int pos, int v) const
{
  auto res = isl_aff_set_coefficient_si(copy(), static_cast<enum isl_dim_type>(type), pos, v);
  return manage(res);
}

aff aff::set_coefficient_val(isl::dim type, int pos, val v) const
{
  auto res = isl_aff_set_coefficient_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

aff aff::set_constant_si(int v) const
{
  auto res = isl_aff_set_constant_si(copy(), v);
  return manage(res);
}

aff aff::set_constant_val(val v) const
{
  auto res = isl_aff_set_constant_val(copy(), v.release());
  return manage(res);
}

aff aff::set_dim_id(isl::dim type, unsigned int pos, id id) const
{
  auto res = isl_aff_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

aff aff::set_tuple_id(isl::dim type, id id) const
{
  auto res = isl_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

aff aff::sub(aff aff2) const
{
  auto res = isl_aff_sub(copy(), aff2.release());
  return manage(res);
}

aff aff::var_on_domain(local_space ls, isl::dim type, unsigned int pos)
{
  auto res = isl_aff_var_on_domain(ls.release(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

basic_set aff::zero_basic_set() const
{
  auto res = isl_aff_zero_basic_set(copy());
  return manage(res);
}

// implementations for isl::aff_list
aff_list manage(__isl_take isl_aff_list *ptr) {
  return aff_list(ptr);
}
aff_list manage_copy(__isl_keep isl_aff_list *ptr) {
  ptr = isl_aff_list_copy(ptr);
  return aff_list(ptr);
}

aff_list::aff_list()
    : ptr(nullptr) {}

aff_list::aff_list(const aff_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
aff_list::aff_list(std::nullptr_t)
    : ptr(nullptr) {}


aff_list::aff_list(__isl_take isl_aff_list *ptr)
    : ptr(ptr) {}


aff_list &aff_list::operator=(aff_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

aff_list::~aff_list() {
  if (ptr)
    isl_aff_list_free(ptr);
}

__isl_give isl_aff_list *aff_list::copy() const & {
  return isl_aff_list_copy(ptr);
}

__isl_keep isl_aff_list *aff_list::get() const {
  return ptr;
}

__isl_give isl_aff_list *aff_list::release() {
  isl_aff_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool aff_list::is_null() const {
  return ptr == nullptr;
}
aff_list::operator bool() const {
  return !is_null();
}


ctx aff_list::get_ctx() const {
  return ctx(isl_aff_list_get_ctx(ptr));
}

void aff_list::dump() const {
  isl_aff_list_dump(get());
}


aff_list aff_list::add(aff el) const
{
  auto res = isl_aff_list_add(copy(), el.release());
  return manage(res);
}

aff_list aff_list::alloc(ctx ctx, int n)
{
  auto res = isl_aff_list_alloc(ctx.release(), n);
  return manage(res);
}

aff_list aff_list::concat(aff_list list2) const
{
  auto res = isl_aff_list_concat(copy(), list2.release());
  return manage(res);
}

aff_list aff_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_aff_list_drop(copy(), first, n);
  return manage(res);
}

stat aff_list::foreach(const std::function<stat(aff)> &fn) const
{
  struct fn_data {
    const std::function<stat(aff)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_aff_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

aff_list aff_list::from_aff(aff el)
{
  auto res = isl_aff_list_from_aff(el.release());
  return manage(res);
}

aff aff_list::get_aff(int index) const
{
  auto res = isl_aff_list_get_aff(get(), index);
  return manage(res);
}

aff aff_list::get_at(int index) const
{
  auto res = isl_aff_list_get_at(get(), index);
  return manage(res);
}

aff_list aff_list::insert(unsigned int pos, aff el) const
{
  auto res = isl_aff_list_insert(copy(), pos, el.release());
  return manage(res);
}

int aff_list::n_aff() const
{
  auto res = isl_aff_list_n_aff(get());
  return res;
}

aff_list aff_list::reverse() const
{
  auto res = isl_aff_list_reverse(copy());
  return manage(res);
}

aff_list aff_list::set_aff(int index, aff el) const
{
  auto res = isl_aff_list_set_aff(copy(), index, el.release());
  return manage(res);
}

int aff_list::size() const
{
  auto res = isl_aff_list_size(get());
  return res;
}

aff_list aff_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_aff_list_swap(copy(), pos1, pos2);
  return manage(res);
}

// implementations for isl::ast_build
ast_build manage(__isl_take isl_ast_build *ptr) {
  return ast_build(ptr);
}
ast_build manage_copy(__isl_keep isl_ast_build *ptr) {
  ptr = isl_ast_build_copy(ptr);
  return ast_build(ptr);
}

ast_build::ast_build()
    : ptr(nullptr) {}

ast_build::ast_build(const ast_build &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
ast_build::ast_build(std::nullptr_t)
    : ptr(nullptr) {}


ast_build::ast_build(__isl_take isl_ast_build *ptr)
    : ptr(ptr) {}

ast_build::ast_build(ctx ctx)
{
  auto res = isl_ast_build_alloc(ctx.release());
  ptr = res;
}

ast_build &ast_build::operator=(ast_build obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

ast_build::~ast_build() {
  if (ptr)
    isl_ast_build_free(ptr);
}

__isl_give isl_ast_build *ast_build::copy() const & {
  return isl_ast_build_copy(ptr);
}

__isl_keep isl_ast_build *ast_build::get() const {
  return ptr;
}

__isl_give isl_ast_build *ast_build::release() {
  isl_ast_build *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool ast_build::is_null() const {
  return ptr == nullptr;
}
ast_build::operator bool() const {
  return !is_null();
}


ctx ast_build::get_ctx() const {
  return ctx(isl_ast_build_get_ctx(ptr));
}


ast_expr ast_build::access_from(pw_multi_aff pma) const
{
  auto res = isl_ast_build_access_from_pw_multi_aff(get(), pma.release());
  return manage(res);
}

ast_expr ast_build::access_from(multi_pw_aff mpa) const
{
  auto res = isl_ast_build_access_from_multi_pw_aff(get(), mpa.release());
  return manage(res);
}

ast_node ast_build::ast_from_schedule(union_map schedule) const
{
  auto res = isl_ast_build_ast_from_schedule(get(), schedule.release());
  return manage(res);
}

ast_expr ast_build::call_from(pw_multi_aff pma) const
{
  auto res = isl_ast_build_call_from_pw_multi_aff(get(), pma.release());
  return manage(res);
}

ast_expr ast_build::call_from(multi_pw_aff mpa) const
{
  auto res = isl_ast_build_call_from_multi_pw_aff(get(), mpa.release());
  return manage(res);
}

ast_expr ast_build::expr_from(set set) const
{
  auto res = isl_ast_build_expr_from_set(get(), set.release());
  return manage(res);
}

ast_expr ast_build::expr_from(pw_aff pa) const
{
  auto res = isl_ast_build_expr_from_pw_aff(get(), pa.release());
  return manage(res);
}

ast_build ast_build::from_context(set set)
{
  auto res = isl_ast_build_from_context(set.release());
  return manage(res);
}

union_map ast_build::get_schedule() const
{
  auto res = isl_ast_build_get_schedule(get());
  return manage(res);
}

space ast_build::get_schedule_space() const
{
  auto res = isl_ast_build_get_schedule_space(get());
  return manage(res);
}

ast_node ast_build::node_from_schedule(schedule schedule) const
{
  auto res = isl_ast_build_node_from_schedule(get(), schedule.release());
  return manage(res);
}

ast_node ast_build::node_from_schedule_map(union_map schedule) const
{
  auto res = isl_ast_build_node_from_schedule_map(get(), schedule.release());
  return manage(res);
}

ast_build ast_build::restrict(set set) const
{
  auto res = isl_ast_build_restrict(copy(), set.release());
  return manage(res);
}

// implementations for isl::ast_expr
ast_expr manage(__isl_take isl_ast_expr *ptr) {
  return ast_expr(ptr);
}
ast_expr manage_copy(__isl_keep isl_ast_expr *ptr) {
  ptr = isl_ast_expr_copy(ptr);
  return ast_expr(ptr);
}

ast_expr::ast_expr()
    : ptr(nullptr) {}

ast_expr::ast_expr(const ast_expr &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
ast_expr::ast_expr(std::nullptr_t)
    : ptr(nullptr) {}


ast_expr::ast_expr(__isl_take isl_ast_expr *ptr)
    : ptr(ptr) {}


ast_expr &ast_expr::operator=(ast_expr obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

ast_expr::~ast_expr() {
  if (ptr)
    isl_ast_expr_free(ptr);
}

__isl_give isl_ast_expr *ast_expr::copy() const & {
  return isl_ast_expr_copy(ptr);
}

__isl_keep isl_ast_expr *ast_expr::get() const {
  return ptr;
}

__isl_give isl_ast_expr *ast_expr::release() {
  isl_ast_expr *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool ast_expr::is_null() const {
  return ptr == nullptr;
}
ast_expr::operator bool() const {
  return !is_null();
}


ctx ast_expr::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}
std::string ast_expr::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void ast_expr::dump() const {
  isl_ast_expr_dump(get());
}


ast_expr ast_expr::access(ast_expr_list indices) const
{
  auto res = isl_ast_expr_access(copy(), indices.release());
  return manage(res);
}

ast_expr ast_expr::add(ast_expr expr2) const
{
  auto res = isl_ast_expr_add(copy(), expr2.release());
  return manage(res);
}

ast_expr ast_expr::address_of() const
{
  auto res = isl_ast_expr_address_of(copy());
  return manage(res);
}

ast_expr ast_expr::call(ast_expr_list arguments) const
{
  auto res = isl_ast_expr_call(copy(), arguments.release());
  return manage(res);
}

ast_expr ast_expr::div(ast_expr expr2) const
{
  auto res = isl_ast_expr_div(copy(), expr2.release());
  return manage(res);
}

ast_expr ast_expr::eq(ast_expr expr2) const
{
  auto res = isl_ast_expr_eq(copy(), expr2.release());
  return manage(res);
}

ast_expr ast_expr::from_id(id id)
{
  auto res = isl_ast_expr_from_id(id.release());
  return manage(res);
}

ast_expr ast_expr::from_val(val v)
{
  auto res = isl_ast_expr_from_val(v.release());
  return manage(res);
}

ast_expr ast_expr::ge(ast_expr expr2) const
{
  auto res = isl_ast_expr_ge(copy(), expr2.release());
  return manage(res);
}

id ast_expr::get_id() const
{
  auto res = isl_ast_expr_get_id(get());
  return manage(res);
}

ast_expr ast_expr::get_op_arg(int pos) const
{
  auto res = isl_ast_expr_get_op_arg(get(), pos);
  return manage(res);
}

int ast_expr::get_op_n_arg() const
{
  auto res = isl_ast_expr_get_op_n_arg(get());
  return res;
}

val ast_expr::get_val() const
{
  auto res = isl_ast_expr_get_val(get());
  return manage(res);
}

ast_expr ast_expr::gt(ast_expr expr2) const
{
  auto res = isl_ast_expr_gt(copy(), expr2.release());
  return manage(res);
}

boolean ast_expr::is_equal(const ast_expr &expr2) const
{
  auto res = isl_ast_expr_is_equal(get(), expr2.get());
  return manage(res);
}

ast_expr ast_expr::le(ast_expr expr2) const
{
  auto res = isl_ast_expr_le(copy(), expr2.release());
  return manage(res);
}

ast_expr ast_expr::lt(ast_expr expr2) const
{
  auto res = isl_ast_expr_lt(copy(), expr2.release());
  return manage(res);
}

ast_expr ast_expr::mul(ast_expr expr2) const
{
  auto res = isl_ast_expr_mul(copy(), expr2.release());
  return manage(res);
}

ast_expr ast_expr::neg() const
{
  auto res = isl_ast_expr_neg(copy());
  return manage(res);
}

ast_expr ast_expr::pdiv_q(ast_expr expr2) const
{
  auto res = isl_ast_expr_pdiv_q(copy(), expr2.release());
  return manage(res);
}

ast_expr ast_expr::pdiv_r(ast_expr expr2) const
{
  auto res = isl_ast_expr_pdiv_r(copy(), expr2.release());
  return manage(res);
}

ast_expr ast_expr::set_op_arg(int pos, ast_expr arg) const
{
  auto res = isl_ast_expr_set_op_arg(copy(), pos, arg.release());
  return manage(res);
}

ast_expr ast_expr::sub(ast_expr expr2) const
{
  auto res = isl_ast_expr_sub(copy(), expr2.release());
  return manage(res);
}

ast_expr ast_expr::substitute_ids(id_to_ast_expr id2expr) const
{
  auto res = isl_ast_expr_substitute_ids(copy(), id2expr.release());
  return manage(res);
}

std::string ast_expr::to_C_str() const
{
  auto res = isl_ast_expr_to_C_str(get());
  std::string tmp(res);
  free(res);
  return tmp;
}

// implementations for isl::ast_expr_list
ast_expr_list manage(__isl_take isl_ast_expr_list *ptr) {
  return ast_expr_list(ptr);
}
ast_expr_list manage_copy(__isl_keep isl_ast_expr_list *ptr) {
  ptr = isl_ast_expr_list_copy(ptr);
  return ast_expr_list(ptr);
}

ast_expr_list::ast_expr_list()
    : ptr(nullptr) {}

ast_expr_list::ast_expr_list(const ast_expr_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
ast_expr_list::ast_expr_list(std::nullptr_t)
    : ptr(nullptr) {}


ast_expr_list::ast_expr_list(__isl_take isl_ast_expr_list *ptr)
    : ptr(ptr) {}


ast_expr_list &ast_expr_list::operator=(ast_expr_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

ast_expr_list::~ast_expr_list() {
  if (ptr)
    isl_ast_expr_list_free(ptr);
}

__isl_give isl_ast_expr_list *ast_expr_list::copy() const & {
  return isl_ast_expr_list_copy(ptr);
}

__isl_keep isl_ast_expr_list *ast_expr_list::get() const {
  return ptr;
}

__isl_give isl_ast_expr_list *ast_expr_list::release() {
  isl_ast_expr_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool ast_expr_list::is_null() const {
  return ptr == nullptr;
}
ast_expr_list::operator bool() const {
  return !is_null();
}


ctx ast_expr_list::get_ctx() const {
  return ctx(isl_ast_expr_list_get_ctx(ptr));
}

void ast_expr_list::dump() const {
  isl_ast_expr_list_dump(get());
}


ast_expr_list ast_expr_list::add(ast_expr el) const
{
  auto res = isl_ast_expr_list_add(copy(), el.release());
  return manage(res);
}

ast_expr_list ast_expr_list::alloc(ctx ctx, int n)
{
  auto res = isl_ast_expr_list_alloc(ctx.release(), n);
  return manage(res);
}

ast_expr_list ast_expr_list::concat(ast_expr_list list2) const
{
  auto res = isl_ast_expr_list_concat(copy(), list2.release());
  return manage(res);
}

ast_expr_list ast_expr_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_ast_expr_list_drop(copy(), first, n);
  return manage(res);
}

stat ast_expr_list::foreach(const std::function<stat(ast_expr)> &fn) const
{
  struct fn_data {
    const std::function<stat(ast_expr)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_ast_expr *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_ast_expr_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

ast_expr_list ast_expr_list::from_ast_expr(ast_expr el)
{
  auto res = isl_ast_expr_list_from_ast_expr(el.release());
  return manage(res);
}

ast_expr ast_expr_list::get_ast_expr(int index) const
{
  auto res = isl_ast_expr_list_get_ast_expr(get(), index);
  return manage(res);
}

ast_expr ast_expr_list::get_at(int index) const
{
  auto res = isl_ast_expr_list_get_at(get(), index);
  return manage(res);
}

ast_expr_list ast_expr_list::insert(unsigned int pos, ast_expr el) const
{
  auto res = isl_ast_expr_list_insert(copy(), pos, el.release());
  return manage(res);
}

int ast_expr_list::n_ast_expr() const
{
  auto res = isl_ast_expr_list_n_ast_expr(get());
  return res;
}

ast_expr_list ast_expr_list::reverse() const
{
  auto res = isl_ast_expr_list_reverse(copy());
  return manage(res);
}

ast_expr_list ast_expr_list::set_ast_expr(int index, ast_expr el) const
{
  auto res = isl_ast_expr_list_set_ast_expr(copy(), index, el.release());
  return manage(res);
}

int ast_expr_list::size() const
{
  auto res = isl_ast_expr_list_size(get());
  return res;
}

ast_expr_list ast_expr_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_ast_expr_list_swap(copy(), pos1, pos2);
  return manage(res);
}

// implementations for isl::ast_node
ast_node manage(__isl_take isl_ast_node *ptr) {
  return ast_node(ptr);
}
ast_node manage_copy(__isl_keep isl_ast_node *ptr) {
  ptr = isl_ast_node_copy(ptr);
  return ast_node(ptr);
}

ast_node::ast_node()
    : ptr(nullptr) {}

ast_node::ast_node(const ast_node &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
ast_node::ast_node(std::nullptr_t)
    : ptr(nullptr) {}


ast_node::ast_node(__isl_take isl_ast_node *ptr)
    : ptr(ptr) {}


ast_node &ast_node::operator=(ast_node obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

ast_node::~ast_node() {
  if (ptr)
    isl_ast_node_free(ptr);
}

__isl_give isl_ast_node *ast_node::copy() const & {
  return isl_ast_node_copy(ptr);
}

__isl_keep isl_ast_node *ast_node::get() const {
  return ptr;
}

__isl_give isl_ast_node *ast_node::release() {
  isl_ast_node *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool ast_node::is_null() const {
  return ptr == nullptr;
}
ast_node::operator bool() const {
  return !is_null();
}


ctx ast_node::get_ctx() const {
  return ctx(isl_ast_node_get_ctx(ptr));
}
std::string ast_node::to_str() const {
  char *Tmp = isl_ast_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void ast_node::dump() const {
  isl_ast_node_dump(get());
}


ast_node ast_node::alloc_user(ast_expr expr)
{
  auto res = isl_ast_node_alloc_user(expr.release());
  return manage(res);
}

ast_node_list ast_node::block_get_children() const
{
  auto res = isl_ast_node_block_get_children(get());
  return manage(res);
}

ast_node ast_node::for_get_body() const
{
  auto res = isl_ast_node_for_get_body(get());
  return manage(res);
}

ast_expr ast_node::for_get_cond() const
{
  auto res = isl_ast_node_for_get_cond(get());
  return manage(res);
}

ast_expr ast_node::for_get_inc() const
{
  auto res = isl_ast_node_for_get_inc(get());
  return manage(res);
}

ast_expr ast_node::for_get_init() const
{
  auto res = isl_ast_node_for_get_init(get());
  return manage(res);
}

ast_expr ast_node::for_get_iterator() const
{
  auto res = isl_ast_node_for_get_iterator(get());
  return manage(res);
}

boolean ast_node::for_is_degenerate() const
{
  auto res = isl_ast_node_for_is_degenerate(get());
  return manage(res);
}

id ast_node::get_annotation() const
{
  auto res = isl_ast_node_get_annotation(get());
  return manage(res);
}

ast_expr ast_node::if_get_cond() const
{
  auto res = isl_ast_node_if_get_cond(get());
  return manage(res);
}

ast_node ast_node::if_get_else() const
{
  auto res = isl_ast_node_if_get_else(get());
  return manage(res);
}

ast_node ast_node::if_get_then() const
{
  auto res = isl_ast_node_if_get_then(get());
  return manage(res);
}

boolean ast_node::if_has_else() const
{
  auto res = isl_ast_node_if_has_else(get());
  return manage(res);
}

id ast_node::mark_get_id() const
{
  auto res = isl_ast_node_mark_get_id(get());
  return manage(res);
}

ast_node ast_node::mark_get_node() const
{
  auto res = isl_ast_node_mark_get_node(get());
  return manage(res);
}

ast_node ast_node::set_annotation(id annotation) const
{
  auto res = isl_ast_node_set_annotation(copy(), annotation.release());
  return manage(res);
}

std::string ast_node::to_C_str() const
{
  auto res = isl_ast_node_to_C_str(get());
  std::string tmp(res);
  free(res);
  return tmp;
}

ast_expr ast_node::user_get_expr() const
{
  auto res = isl_ast_node_user_get_expr(get());
  return manage(res);
}

// implementations for isl::ast_node_list
ast_node_list manage(__isl_take isl_ast_node_list *ptr) {
  return ast_node_list(ptr);
}
ast_node_list manage_copy(__isl_keep isl_ast_node_list *ptr) {
  ptr = isl_ast_node_list_copy(ptr);
  return ast_node_list(ptr);
}

ast_node_list::ast_node_list()
    : ptr(nullptr) {}

ast_node_list::ast_node_list(const ast_node_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
ast_node_list::ast_node_list(std::nullptr_t)
    : ptr(nullptr) {}


ast_node_list::ast_node_list(__isl_take isl_ast_node_list *ptr)
    : ptr(ptr) {}


ast_node_list &ast_node_list::operator=(ast_node_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

ast_node_list::~ast_node_list() {
  if (ptr)
    isl_ast_node_list_free(ptr);
}

__isl_give isl_ast_node_list *ast_node_list::copy() const & {
  return isl_ast_node_list_copy(ptr);
}

__isl_keep isl_ast_node_list *ast_node_list::get() const {
  return ptr;
}

__isl_give isl_ast_node_list *ast_node_list::release() {
  isl_ast_node_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool ast_node_list::is_null() const {
  return ptr == nullptr;
}
ast_node_list::operator bool() const {
  return !is_null();
}


ctx ast_node_list::get_ctx() const {
  return ctx(isl_ast_node_list_get_ctx(ptr));
}

void ast_node_list::dump() const {
  isl_ast_node_list_dump(get());
}


ast_node_list ast_node_list::add(ast_node el) const
{
  auto res = isl_ast_node_list_add(copy(), el.release());
  return manage(res);
}

ast_node_list ast_node_list::alloc(ctx ctx, int n)
{
  auto res = isl_ast_node_list_alloc(ctx.release(), n);
  return manage(res);
}

ast_node_list ast_node_list::concat(ast_node_list list2) const
{
  auto res = isl_ast_node_list_concat(copy(), list2.release());
  return manage(res);
}

ast_node_list ast_node_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_ast_node_list_drop(copy(), first, n);
  return manage(res);
}

stat ast_node_list::foreach(const std::function<stat(ast_node)> &fn) const
{
  struct fn_data {
    const std::function<stat(ast_node)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_ast_node *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_ast_node_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

ast_node_list ast_node_list::from_ast_node(ast_node el)
{
  auto res = isl_ast_node_list_from_ast_node(el.release());
  return manage(res);
}

ast_node ast_node_list::get_ast_node(int index) const
{
  auto res = isl_ast_node_list_get_ast_node(get(), index);
  return manage(res);
}

ast_node ast_node_list::get_at(int index) const
{
  auto res = isl_ast_node_list_get_at(get(), index);
  return manage(res);
}

ast_node_list ast_node_list::insert(unsigned int pos, ast_node el) const
{
  auto res = isl_ast_node_list_insert(copy(), pos, el.release());
  return manage(res);
}

int ast_node_list::n_ast_node() const
{
  auto res = isl_ast_node_list_n_ast_node(get());
  return res;
}

ast_node_list ast_node_list::reverse() const
{
  auto res = isl_ast_node_list_reverse(copy());
  return manage(res);
}

ast_node_list ast_node_list::set_ast_node(int index, ast_node el) const
{
  auto res = isl_ast_node_list_set_ast_node(copy(), index, el.release());
  return manage(res);
}

int ast_node_list::size() const
{
  auto res = isl_ast_node_list_size(get());
  return res;
}

ast_node_list ast_node_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_ast_node_list_swap(copy(), pos1, pos2);
  return manage(res);
}

// implementations for isl::basic_map
basic_map manage(__isl_take isl_basic_map *ptr) {
  return basic_map(ptr);
}
basic_map manage_copy(__isl_keep isl_basic_map *ptr) {
  ptr = isl_basic_map_copy(ptr);
  return basic_map(ptr);
}

basic_map::basic_map()
    : ptr(nullptr) {}

basic_map::basic_map(const basic_map &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
basic_map::basic_map(std::nullptr_t)
    : ptr(nullptr) {}


basic_map::basic_map(__isl_take isl_basic_map *ptr)
    : ptr(ptr) {}

basic_map::basic_map(ctx ctx, const std::string &str)
{
  auto res = isl_basic_map_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

basic_map &basic_map::operator=(basic_map obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

basic_map::~basic_map() {
  if (ptr)
    isl_basic_map_free(ptr);
}

__isl_give isl_basic_map *basic_map::copy() const & {
  return isl_basic_map_copy(ptr);
}

__isl_keep isl_basic_map *basic_map::get() const {
  return ptr;
}

__isl_give isl_basic_map *basic_map::release() {
  isl_basic_map *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool basic_map::is_null() const {
  return ptr == nullptr;
}
basic_map::operator bool() const {
  return !is_null();
}


ctx basic_map::get_ctx() const {
  return ctx(isl_basic_map_get_ctx(ptr));
}
std::string basic_map::to_str() const {
  char *Tmp = isl_basic_map_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void basic_map::dump() const {
  isl_basic_map_dump(get());
}


basic_map basic_map::add_constraint(constraint constraint) const
{
  auto res = isl_basic_map_add_constraint(copy(), constraint.release());
  return manage(res);
}

basic_map basic_map::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_basic_map_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

basic_map basic_map::affine_hull() const
{
  auto res = isl_basic_map_affine_hull(copy());
  return manage(res);
}

basic_map basic_map::align_params(space model) const
{
  auto res = isl_basic_map_align_params(copy(), model.release());
  return manage(res);
}

basic_map basic_map::apply_domain(basic_map bmap2) const
{
  auto res = isl_basic_map_apply_domain(copy(), bmap2.release());
  return manage(res);
}

basic_map basic_map::apply_range(basic_map bmap2) const
{
  auto res = isl_basic_map_apply_range(copy(), bmap2.release());
  return manage(res);
}

boolean basic_map::can_curry() const
{
  auto res = isl_basic_map_can_curry(get());
  return manage(res);
}

boolean basic_map::can_uncurry() const
{
  auto res = isl_basic_map_can_uncurry(get());
  return manage(res);
}

boolean basic_map::can_zip() const
{
  auto res = isl_basic_map_can_zip(get());
  return manage(res);
}

basic_map basic_map::curry() const
{
  auto res = isl_basic_map_curry(copy());
  return manage(res);
}

basic_set basic_map::deltas() const
{
  auto res = isl_basic_map_deltas(copy());
  return manage(res);
}

basic_map basic_map::deltas_map() const
{
  auto res = isl_basic_map_deltas_map(copy());
  return manage(res);
}

basic_map basic_map::detect_equalities() const
{
  auto res = isl_basic_map_detect_equalities(copy());
  return manage(res);
}

unsigned int basic_map::dim(isl::dim type) const
{
  auto res = isl_basic_map_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

basic_set basic_map::domain() const
{
  auto res = isl_basic_map_domain(copy());
  return manage(res);
}

basic_map basic_map::domain_map() const
{
  auto res = isl_basic_map_domain_map(copy());
  return manage(res);
}

basic_map basic_map::domain_product(basic_map bmap2) const
{
  auto res = isl_basic_map_domain_product(copy(), bmap2.release());
  return manage(res);
}

basic_map basic_map::drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_map_drop_constraints_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

basic_map basic_map::drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_map_drop_constraints_not_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

basic_map basic_map::drop_unused_params() const
{
  auto res = isl_basic_map_drop_unused_params(copy());
  return manage(res);
}

basic_map basic_map::eliminate(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_map_eliminate(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

basic_map basic_map::empty(space space)
{
  auto res = isl_basic_map_empty(space.release());
  return manage(res);
}

basic_map basic_map::equal(space dim, unsigned int n_equal)
{
  auto res = isl_basic_map_equal(dim.release(), n_equal);
  return manage(res);
}

mat basic_map::equalities_matrix(isl::dim c1, isl::dim c2, isl::dim c3, isl::dim c4, isl::dim c5) const
{
  auto res = isl_basic_map_equalities_matrix(get(), static_cast<enum isl_dim_type>(c1), static_cast<enum isl_dim_type>(c2), static_cast<enum isl_dim_type>(c3), static_cast<enum isl_dim_type>(c4), static_cast<enum isl_dim_type>(c5));
  return manage(res);
}

basic_map basic_map::equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_basic_map_equate(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

int basic_map::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_basic_map_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

basic_map basic_map::fix_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_basic_map_fix_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

basic_map basic_map::fix_val(isl::dim type, unsigned int pos, val v) const
{
  auto res = isl_basic_map_fix_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

basic_map basic_map::flat_product(basic_map bmap2) const
{
  auto res = isl_basic_map_flat_product(copy(), bmap2.release());
  return manage(res);
}

basic_map basic_map::flat_range_product(basic_map bmap2) const
{
  auto res = isl_basic_map_flat_range_product(copy(), bmap2.release());
  return manage(res);
}

basic_map basic_map::flatten() const
{
  auto res = isl_basic_map_flatten(copy());
  return manage(res);
}

basic_map basic_map::flatten_domain() const
{
  auto res = isl_basic_map_flatten_domain(copy());
  return manage(res);
}

basic_map basic_map::flatten_range() const
{
  auto res = isl_basic_map_flatten_range(copy());
  return manage(res);
}

stat basic_map::foreach_constraint(const std::function<stat(constraint)> &fn) const
{
  struct fn_data {
    const std::function<stat(constraint)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_constraint *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_basic_map_foreach_constraint(get(), fn_lambda, &fn_data);
  return manage(res);
}

basic_map basic_map::from_aff(aff aff)
{
  auto res = isl_basic_map_from_aff(aff.release());
  return manage(res);
}

basic_map basic_map::from_aff_list(space domain_space, aff_list list)
{
  auto res = isl_basic_map_from_aff_list(domain_space.release(), list.release());
  return manage(res);
}

basic_map basic_map::from_constraint(constraint constraint)
{
  auto res = isl_basic_map_from_constraint(constraint.release());
  return manage(res);
}

basic_map basic_map::from_domain(basic_set bset)
{
  auto res = isl_basic_map_from_domain(bset.release());
  return manage(res);
}

basic_map basic_map::from_domain_and_range(basic_set domain, basic_set range)
{
  auto res = isl_basic_map_from_domain_and_range(domain.release(), range.release());
  return manage(res);
}

basic_map basic_map::from_multi_aff(multi_aff maff)
{
  auto res = isl_basic_map_from_multi_aff(maff.release());
  return manage(res);
}

basic_map basic_map::from_qpolynomial(qpolynomial qp)
{
  auto res = isl_basic_map_from_qpolynomial(qp.release());
  return manage(res);
}

basic_map basic_map::from_range(basic_set bset)
{
  auto res = isl_basic_map_from_range(bset.release());
  return manage(res);
}

constraint_list basic_map::get_constraint_list() const
{
  auto res = isl_basic_map_get_constraint_list(get());
  return manage(res);
}

std::string basic_map::get_dim_name(isl::dim type, unsigned int pos) const
{
  auto res = isl_basic_map_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

aff basic_map::get_div(int pos) const
{
  auto res = isl_basic_map_get_div(get(), pos);
  return manage(res);
}

local_space basic_map::get_local_space() const
{
  auto res = isl_basic_map_get_local_space(get());
  return manage(res);
}

space basic_map::get_space() const
{
  auto res = isl_basic_map_get_space(get());
  return manage(res);
}

std::string basic_map::get_tuple_name(isl::dim type) const
{
  auto res = isl_basic_map_get_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  std::string tmp(res);
  return tmp;
}

basic_map basic_map::gist(basic_map context) const
{
  auto res = isl_basic_map_gist(copy(), context.release());
  return manage(res);
}

basic_map basic_map::gist_domain(basic_set context) const
{
  auto res = isl_basic_map_gist_domain(copy(), context.release());
  return manage(res);
}

boolean basic_map::has_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_basic_map_has_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

basic_map basic_map::identity(space dim)
{
  auto res = isl_basic_map_identity(dim.release());
  return manage(res);
}

boolean basic_map::image_is_bounded() const
{
  auto res = isl_basic_map_image_is_bounded(get());
  return manage(res);
}

mat basic_map::inequalities_matrix(isl::dim c1, isl::dim c2, isl::dim c3, isl::dim c4, isl::dim c5) const
{
  auto res = isl_basic_map_inequalities_matrix(get(), static_cast<enum isl_dim_type>(c1), static_cast<enum isl_dim_type>(c2), static_cast<enum isl_dim_type>(c3), static_cast<enum isl_dim_type>(c4), static_cast<enum isl_dim_type>(c5));
  return manage(res);
}

basic_map basic_map::insert_dims(isl::dim type, unsigned int pos, unsigned int n) const
{
  auto res = isl_basic_map_insert_dims(copy(), static_cast<enum isl_dim_type>(type), pos, n);
  return manage(res);
}

basic_map basic_map::intersect(basic_map bmap2) const
{
  auto res = isl_basic_map_intersect(copy(), bmap2.release());
  return manage(res);
}

basic_map basic_map::intersect_domain(basic_set bset) const
{
  auto res = isl_basic_map_intersect_domain(copy(), bset.release());
  return manage(res);
}

basic_map basic_map::intersect_range(basic_set bset) const
{
  auto res = isl_basic_map_intersect_range(copy(), bset.release());
  return manage(res);
}

boolean basic_map::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_map_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean basic_map::is_disjoint(const basic_map &bmap2) const
{
  auto res = isl_basic_map_is_disjoint(get(), bmap2.get());
  return manage(res);
}

boolean basic_map::is_empty() const
{
  auto res = isl_basic_map_is_empty(get());
  return manage(res);
}

boolean basic_map::is_equal(const basic_map &bmap2) const
{
  auto res = isl_basic_map_is_equal(get(), bmap2.get());
  return manage(res);
}

boolean basic_map::is_rational() const
{
  auto res = isl_basic_map_is_rational(get());
  return manage(res);
}

boolean basic_map::is_single_valued() const
{
  auto res = isl_basic_map_is_single_valued(get());
  return manage(res);
}

boolean basic_map::is_strict_subset(const basic_map &bmap2) const
{
  auto res = isl_basic_map_is_strict_subset(get(), bmap2.get());
  return manage(res);
}

boolean basic_map::is_subset(const basic_map &bmap2) const
{
  auto res = isl_basic_map_is_subset(get(), bmap2.get());
  return manage(res);
}

boolean basic_map::is_universe() const
{
  auto res = isl_basic_map_is_universe(get());
  return manage(res);
}

basic_map basic_map::less_at(space dim, unsigned int pos)
{
  auto res = isl_basic_map_less_at(dim.release(), pos);
  return manage(res);
}

map basic_map::lexmax() const
{
  auto res = isl_basic_map_lexmax(copy());
  return manage(res);
}

map basic_map::lexmin() const
{
  auto res = isl_basic_map_lexmin(copy());
  return manage(res);
}

pw_multi_aff basic_map::lexmin_pw_multi_aff() const
{
  auto res = isl_basic_map_lexmin_pw_multi_aff(copy());
  return manage(res);
}

basic_map basic_map::lower_bound_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_basic_map_lower_bound_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

basic_map basic_map::more_at(space dim, unsigned int pos)
{
  auto res = isl_basic_map_more_at(dim.release(), pos);
  return manage(res);
}

basic_map basic_map::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_basic_map_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

int basic_map::n_constraint() const
{
  auto res = isl_basic_map_n_constraint(get());
  return res;
}

basic_map basic_map::nat_universe(space dim)
{
  auto res = isl_basic_map_nat_universe(dim.release());
  return manage(res);
}

basic_map basic_map::neg() const
{
  auto res = isl_basic_map_neg(copy());
  return manage(res);
}

basic_map basic_map::order_ge(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_basic_map_order_ge(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

basic_map basic_map::order_gt(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_basic_map_order_gt(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

val basic_map::plain_get_val_if_fixed(isl::dim type, unsigned int pos) const
{
  auto res = isl_basic_map_plain_get_val_if_fixed(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

boolean basic_map::plain_is_empty() const
{
  auto res = isl_basic_map_plain_is_empty(get());
  return manage(res);
}

boolean basic_map::plain_is_universe() const
{
  auto res = isl_basic_map_plain_is_universe(get());
  return manage(res);
}

basic_map basic_map::preimage_domain_multi_aff(multi_aff ma) const
{
  auto res = isl_basic_map_preimage_domain_multi_aff(copy(), ma.release());
  return manage(res);
}

basic_map basic_map::preimage_range_multi_aff(multi_aff ma) const
{
  auto res = isl_basic_map_preimage_range_multi_aff(copy(), ma.release());
  return manage(res);
}

basic_map basic_map::product(basic_map bmap2) const
{
  auto res = isl_basic_map_product(copy(), bmap2.release());
  return manage(res);
}

basic_map basic_map::project_out(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_map_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

basic_set basic_map::range() const
{
  auto res = isl_basic_map_range(copy());
  return manage(res);
}

basic_map basic_map::range_map() const
{
  auto res = isl_basic_map_range_map(copy());
  return manage(res);
}

basic_map basic_map::range_product(basic_map bmap2) const
{
  auto res = isl_basic_map_range_product(copy(), bmap2.release());
  return manage(res);
}

basic_map basic_map::remove_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_map_remove_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

basic_map basic_map::remove_divs() const
{
  auto res = isl_basic_map_remove_divs(copy());
  return manage(res);
}

basic_map basic_map::remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_map_remove_divs_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

basic_map basic_map::remove_redundancies() const
{
  auto res = isl_basic_map_remove_redundancies(copy());
  return manage(res);
}

basic_map basic_map::reverse() const
{
  auto res = isl_basic_map_reverse(copy());
  return manage(res);
}

basic_map basic_map::sample() const
{
  auto res = isl_basic_map_sample(copy());
  return manage(res);
}

basic_map basic_map::set_tuple_id(isl::dim type, id id) const
{
  auto res = isl_basic_map_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

basic_map basic_map::set_tuple_name(isl::dim type, const std::string &s) const
{
  auto res = isl_basic_map_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

basic_map basic_map::sum(basic_map bmap2) const
{
  auto res = isl_basic_map_sum(copy(), bmap2.release());
  return manage(res);
}

basic_map basic_map::uncurry() const
{
  auto res = isl_basic_map_uncurry(copy());
  return manage(res);
}

map basic_map::unite(basic_map bmap2) const
{
  auto res = isl_basic_map_union(copy(), bmap2.release());
  return manage(res);
}

basic_map basic_map::universe(space space)
{
  auto res = isl_basic_map_universe(space.release());
  return manage(res);
}

basic_map basic_map::upper_bound_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_basic_map_upper_bound_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

basic_set basic_map::wrap() const
{
  auto res = isl_basic_map_wrap(copy());
  return manage(res);
}

basic_map basic_map::zip() const
{
  auto res = isl_basic_map_zip(copy());
  return manage(res);
}

// implementations for isl::basic_map_list
basic_map_list manage(__isl_take isl_basic_map_list *ptr) {
  return basic_map_list(ptr);
}
basic_map_list manage_copy(__isl_keep isl_basic_map_list *ptr) {
  ptr = isl_basic_map_list_copy(ptr);
  return basic_map_list(ptr);
}

basic_map_list::basic_map_list()
    : ptr(nullptr) {}

basic_map_list::basic_map_list(const basic_map_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
basic_map_list::basic_map_list(std::nullptr_t)
    : ptr(nullptr) {}


basic_map_list::basic_map_list(__isl_take isl_basic_map_list *ptr)
    : ptr(ptr) {}


basic_map_list &basic_map_list::operator=(basic_map_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

basic_map_list::~basic_map_list() {
  if (ptr)
    isl_basic_map_list_free(ptr);
}

__isl_give isl_basic_map_list *basic_map_list::copy() const & {
  return isl_basic_map_list_copy(ptr);
}

__isl_keep isl_basic_map_list *basic_map_list::get() const {
  return ptr;
}

__isl_give isl_basic_map_list *basic_map_list::release() {
  isl_basic_map_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool basic_map_list::is_null() const {
  return ptr == nullptr;
}
basic_map_list::operator bool() const {
  return !is_null();
}


ctx basic_map_list::get_ctx() const {
  return ctx(isl_basic_map_list_get_ctx(ptr));
}

void basic_map_list::dump() const {
  isl_basic_map_list_dump(get());
}


basic_map_list basic_map_list::add(basic_map el) const
{
  auto res = isl_basic_map_list_add(copy(), el.release());
  return manage(res);
}

basic_map_list basic_map_list::alloc(ctx ctx, int n)
{
  auto res = isl_basic_map_list_alloc(ctx.release(), n);
  return manage(res);
}

basic_map_list basic_map_list::concat(basic_map_list list2) const
{
  auto res = isl_basic_map_list_concat(copy(), list2.release());
  return manage(res);
}

basic_map_list basic_map_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_basic_map_list_drop(copy(), first, n);
  return manage(res);
}

stat basic_map_list::foreach(const std::function<stat(basic_map)> &fn) const
{
  struct fn_data {
    const std::function<stat(basic_map)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_basic_map *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_basic_map_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

basic_map_list basic_map_list::from_basic_map(basic_map el)
{
  auto res = isl_basic_map_list_from_basic_map(el.release());
  return manage(res);
}

basic_map basic_map_list::get_at(int index) const
{
  auto res = isl_basic_map_list_get_at(get(), index);
  return manage(res);
}

basic_map basic_map_list::get_basic_map(int index) const
{
  auto res = isl_basic_map_list_get_basic_map(get(), index);
  return manage(res);
}

basic_map_list basic_map_list::insert(unsigned int pos, basic_map el) const
{
  auto res = isl_basic_map_list_insert(copy(), pos, el.release());
  return manage(res);
}

int basic_map_list::n_basic_map() const
{
  auto res = isl_basic_map_list_n_basic_map(get());
  return res;
}

basic_map_list basic_map_list::reverse() const
{
  auto res = isl_basic_map_list_reverse(copy());
  return manage(res);
}

basic_map_list basic_map_list::set_basic_map(int index, basic_map el) const
{
  auto res = isl_basic_map_list_set_basic_map(copy(), index, el.release());
  return manage(res);
}

int basic_map_list::size() const
{
  auto res = isl_basic_map_list_size(get());
  return res;
}

basic_map_list basic_map_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_basic_map_list_swap(copy(), pos1, pos2);
  return manage(res);
}

// implementations for isl::basic_set
basic_set manage(__isl_take isl_basic_set *ptr) {
  return basic_set(ptr);
}
basic_set manage_copy(__isl_keep isl_basic_set *ptr) {
  ptr = isl_basic_set_copy(ptr);
  return basic_set(ptr);
}

basic_set::basic_set()
    : ptr(nullptr) {}

basic_set::basic_set(const basic_set &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
basic_set::basic_set(std::nullptr_t)
    : ptr(nullptr) {}


basic_set::basic_set(__isl_take isl_basic_set *ptr)
    : ptr(ptr) {}

basic_set::basic_set(ctx ctx, const std::string &str)
{
  auto res = isl_basic_set_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}
basic_set::basic_set(point pnt)
{
  auto res = isl_basic_set_from_point(pnt.release());
  ptr = res;
}

basic_set &basic_set::operator=(basic_set obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

basic_set::~basic_set() {
  if (ptr)
    isl_basic_set_free(ptr);
}

__isl_give isl_basic_set *basic_set::copy() const & {
  return isl_basic_set_copy(ptr);
}

__isl_keep isl_basic_set *basic_set::get() const {
  return ptr;
}

__isl_give isl_basic_set *basic_set::release() {
  isl_basic_set *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool basic_set::is_null() const {
  return ptr == nullptr;
}
basic_set::operator bool() const {
  return !is_null();
}


ctx basic_set::get_ctx() const {
  return ctx(isl_basic_set_get_ctx(ptr));
}
std::string basic_set::to_str() const {
  char *Tmp = isl_basic_set_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void basic_set::dump() const {
  isl_basic_set_dump(get());
}


basic_set basic_set::affine_hull() const
{
  auto res = isl_basic_set_affine_hull(copy());
  return manage(res);
}

basic_set basic_set::align_params(space model) const
{
  auto res = isl_basic_set_align_params(copy(), model.release());
  return manage(res);
}

basic_set basic_set::apply(basic_map bmap) const
{
  auto res = isl_basic_set_apply(copy(), bmap.release());
  return manage(res);
}

basic_set basic_set::box_from_points(point pnt1, point pnt2)
{
  auto res = isl_basic_set_box_from_points(pnt1.release(), pnt2.release());
  return manage(res);
}

basic_set basic_set::coefficients() const
{
  auto res = isl_basic_set_coefficients(copy());
  return manage(res);
}

basic_set basic_set::detect_equalities() const
{
  auto res = isl_basic_set_detect_equalities(copy());
  return manage(res);
}

unsigned int basic_set::dim(isl::dim type) const
{
  auto res = isl_basic_set_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

val basic_set::dim_max_val(int pos) const
{
  auto res = isl_basic_set_dim_max_val(copy(), pos);
  return manage(res);
}

basic_set basic_set::drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_set_drop_constraints_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

basic_set basic_set::drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_set_drop_constraints_not_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

basic_set basic_set::drop_unused_params() const
{
  auto res = isl_basic_set_drop_unused_params(copy());
  return manage(res);
}

basic_set basic_set::eliminate(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_set_eliminate(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

basic_set basic_set::empty(space space)
{
  auto res = isl_basic_set_empty(space.release());
  return manage(res);
}

mat basic_set::equalities_matrix(isl::dim c1, isl::dim c2, isl::dim c3, isl::dim c4) const
{
  auto res = isl_basic_set_equalities_matrix(get(), static_cast<enum isl_dim_type>(c1), static_cast<enum isl_dim_type>(c2), static_cast<enum isl_dim_type>(c3), static_cast<enum isl_dim_type>(c4));
  return manage(res);
}

basic_set basic_set::fix_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_basic_set_fix_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

basic_set basic_set::fix_val(isl::dim type, unsigned int pos, val v) const
{
  auto res = isl_basic_set_fix_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

basic_set basic_set::flat_product(basic_set bset2) const
{
  auto res = isl_basic_set_flat_product(copy(), bset2.release());
  return manage(res);
}

basic_set basic_set::flatten() const
{
  auto res = isl_basic_set_flatten(copy());
  return manage(res);
}

stat basic_set::foreach_bound_pair(isl::dim type, unsigned int pos, const std::function<stat(constraint, constraint, basic_set)> &fn) const
{
  struct fn_data {
    const std::function<stat(constraint, constraint, basic_set)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_constraint *arg_0, isl_constraint *arg_1, isl_basic_set *arg_2, void *arg_3) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_3);
    stat ret = (*data->func)(manage(arg_0), manage(arg_1), manage(arg_2));
    return ret.release();
  };
  auto res = isl_basic_set_foreach_bound_pair(get(), static_cast<enum isl_dim_type>(type), pos, fn_lambda, &fn_data);
  return manage(res);
}

stat basic_set::foreach_constraint(const std::function<stat(constraint)> &fn) const
{
  struct fn_data {
    const std::function<stat(constraint)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_constraint *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_basic_set_foreach_constraint(get(), fn_lambda, &fn_data);
  return manage(res);
}

basic_set basic_set::from_constraint(constraint constraint)
{
  auto res = isl_basic_set_from_constraint(constraint.release());
  return manage(res);
}

basic_set basic_set::from_multi_aff(multi_aff ma)
{
  auto res = isl_basic_set_from_multi_aff(ma.release());
  return manage(res);
}

basic_set basic_set::from_params() const
{
  auto res = isl_basic_set_from_params(copy());
  return manage(res);
}

constraint_list basic_set::get_constraint_list() const
{
  auto res = isl_basic_set_get_constraint_list(get());
  return manage(res);
}

id basic_set::get_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_basic_set_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

std::string basic_set::get_dim_name(isl::dim type, unsigned int pos) const
{
  auto res = isl_basic_set_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

aff basic_set::get_div(int pos) const
{
  auto res = isl_basic_set_get_div(get(), pos);
  return manage(res);
}

local_space basic_set::get_local_space() const
{
  auto res = isl_basic_set_get_local_space(get());
  return manage(res);
}

space basic_set::get_space() const
{
  auto res = isl_basic_set_get_space(get());
  return manage(res);
}

std::string basic_set::get_tuple_name() const
{
  auto res = isl_basic_set_get_tuple_name(get());
  std::string tmp(res);
  return tmp;
}

basic_set basic_set::gist(basic_set context) const
{
  auto res = isl_basic_set_gist(copy(), context.release());
  return manage(res);
}

mat basic_set::inequalities_matrix(isl::dim c1, isl::dim c2, isl::dim c3, isl::dim c4) const
{
  auto res = isl_basic_set_inequalities_matrix(get(), static_cast<enum isl_dim_type>(c1), static_cast<enum isl_dim_type>(c2), static_cast<enum isl_dim_type>(c3), static_cast<enum isl_dim_type>(c4));
  return manage(res);
}

basic_set basic_set::insert_dims(isl::dim type, unsigned int pos, unsigned int n) const
{
  auto res = isl_basic_set_insert_dims(copy(), static_cast<enum isl_dim_type>(type), pos, n);
  return manage(res);
}

basic_set basic_set::intersect(basic_set bset2) const
{
  auto res = isl_basic_set_intersect(copy(), bset2.release());
  return manage(res);
}

basic_set basic_set::intersect_params(basic_set bset2) const
{
  auto res = isl_basic_set_intersect_params(copy(), bset2.release());
  return manage(res);
}

boolean basic_set::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_set_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean basic_set::is_bounded() const
{
  auto res = isl_basic_set_is_bounded(get());
  return manage(res);
}

boolean basic_set::is_disjoint(const basic_set &bset2) const
{
  auto res = isl_basic_set_is_disjoint(get(), bset2.get());
  return manage(res);
}

boolean basic_set::is_empty() const
{
  auto res = isl_basic_set_is_empty(get());
  return manage(res);
}

boolean basic_set::is_equal(const basic_set &bset2) const
{
  auto res = isl_basic_set_is_equal(get(), bset2.get());
  return manage(res);
}

int basic_set::is_rational() const
{
  auto res = isl_basic_set_is_rational(get());
  return res;
}

boolean basic_set::is_subset(const basic_set &bset2) const
{
  auto res = isl_basic_set_is_subset(get(), bset2.get());
  return manage(res);
}

boolean basic_set::is_universe() const
{
  auto res = isl_basic_set_is_universe(get());
  return manage(res);
}

boolean basic_set::is_wrapping() const
{
  auto res = isl_basic_set_is_wrapping(get());
  return manage(res);
}

set basic_set::lexmax() const
{
  auto res = isl_basic_set_lexmax(copy());
  return manage(res);
}

set basic_set::lexmin() const
{
  auto res = isl_basic_set_lexmin(copy());
  return manage(res);
}

basic_set basic_set::lower_bound_val(isl::dim type, unsigned int pos, val value) const
{
  auto res = isl_basic_set_lower_bound_val(copy(), static_cast<enum isl_dim_type>(type), pos, value.release());
  return manage(res);
}

val basic_set::max_val(const aff &obj) const
{
  auto res = isl_basic_set_max_val(get(), obj.get());
  return manage(res);
}

basic_set basic_set::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_basic_set_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

int basic_set::n_constraint() const
{
  auto res = isl_basic_set_n_constraint(get());
  return res;
}

unsigned int basic_set::n_dim() const
{
  auto res = isl_basic_set_n_dim(get());
  return res;
}

basic_set basic_set::nat_universe(space dim)
{
  auto res = isl_basic_set_nat_universe(dim.release());
  return manage(res);
}

basic_set basic_set::neg() const
{
  auto res = isl_basic_set_neg(copy());
  return manage(res);
}

basic_set basic_set::params() const
{
  auto res = isl_basic_set_params(copy());
  return manage(res);
}

boolean basic_set::plain_is_empty() const
{
  auto res = isl_basic_set_plain_is_empty(get());
  return manage(res);
}

boolean basic_set::plain_is_equal(const basic_set &bset2) const
{
  auto res = isl_basic_set_plain_is_equal(get(), bset2.get());
  return manage(res);
}

boolean basic_set::plain_is_universe() const
{
  auto res = isl_basic_set_plain_is_universe(get());
  return manage(res);
}

basic_set basic_set::positive_orthant(space space)
{
  auto res = isl_basic_set_positive_orthant(space.release());
  return manage(res);
}

basic_set basic_set::preimage_multi_aff(multi_aff ma) const
{
  auto res = isl_basic_set_preimage_multi_aff(copy(), ma.release());
  return manage(res);
}

basic_set basic_set::project_out(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_set_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

mat basic_set::reduced_basis() const
{
  auto res = isl_basic_set_reduced_basis(get());
  return manage(res);
}

basic_set basic_set::remove_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_set_remove_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

basic_set basic_set::remove_divs() const
{
  auto res = isl_basic_set_remove_divs(copy());
  return manage(res);
}

basic_set basic_set::remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_set_remove_divs_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

basic_set basic_set::remove_redundancies() const
{
  auto res = isl_basic_set_remove_redundancies(copy());
  return manage(res);
}

basic_set basic_set::remove_unknown_divs() const
{
  auto res = isl_basic_set_remove_unknown_divs(copy());
  return manage(res);
}

basic_set basic_set::sample() const
{
  auto res = isl_basic_set_sample(copy());
  return manage(res);
}

point basic_set::sample_point() const
{
  auto res = isl_basic_set_sample_point(copy());
  return manage(res);
}

basic_set basic_set::set_tuple_id(id id) const
{
  auto res = isl_basic_set_set_tuple_id(copy(), id.release());
  return manage(res);
}

basic_set basic_set::set_tuple_name(const std::string &s) const
{
  auto res = isl_basic_set_set_tuple_name(copy(), s.c_str());
  return manage(res);
}

basic_set basic_set::solutions() const
{
  auto res = isl_basic_set_solutions(copy());
  return manage(res);
}

set basic_set::unite(basic_set bset2) const
{
  auto res = isl_basic_set_union(copy(), bset2.release());
  return manage(res);
}

basic_set basic_set::universe(space space)
{
  auto res = isl_basic_set_universe(space.release());
  return manage(res);
}

basic_map basic_set::unwrap() const
{
  auto res = isl_basic_set_unwrap(copy());
  return manage(res);
}

basic_set basic_set::upper_bound_val(isl::dim type, unsigned int pos, val value) const
{
  auto res = isl_basic_set_upper_bound_val(copy(), static_cast<enum isl_dim_type>(type), pos, value.release());
  return manage(res);
}

// implementations for isl::basic_set_list
basic_set_list manage(__isl_take isl_basic_set_list *ptr) {
  return basic_set_list(ptr);
}
basic_set_list manage_copy(__isl_keep isl_basic_set_list *ptr) {
  ptr = isl_basic_set_list_copy(ptr);
  return basic_set_list(ptr);
}

basic_set_list::basic_set_list()
    : ptr(nullptr) {}

basic_set_list::basic_set_list(const basic_set_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
basic_set_list::basic_set_list(std::nullptr_t)
    : ptr(nullptr) {}


basic_set_list::basic_set_list(__isl_take isl_basic_set_list *ptr)
    : ptr(ptr) {}


basic_set_list &basic_set_list::operator=(basic_set_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

basic_set_list::~basic_set_list() {
  if (ptr)
    isl_basic_set_list_free(ptr);
}

__isl_give isl_basic_set_list *basic_set_list::copy() const & {
  return isl_basic_set_list_copy(ptr);
}

__isl_keep isl_basic_set_list *basic_set_list::get() const {
  return ptr;
}

__isl_give isl_basic_set_list *basic_set_list::release() {
  isl_basic_set_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool basic_set_list::is_null() const {
  return ptr == nullptr;
}
basic_set_list::operator bool() const {
  return !is_null();
}


ctx basic_set_list::get_ctx() const {
  return ctx(isl_basic_set_list_get_ctx(ptr));
}

void basic_set_list::dump() const {
  isl_basic_set_list_dump(get());
}


basic_set_list basic_set_list::add(basic_set el) const
{
  auto res = isl_basic_set_list_add(copy(), el.release());
  return manage(res);
}

basic_set_list basic_set_list::alloc(ctx ctx, int n)
{
  auto res = isl_basic_set_list_alloc(ctx.release(), n);
  return manage(res);
}

basic_set_list basic_set_list::coefficients() const
{
  auto res = isl_basic_set_list_coefficients(copy());
  return manage(res);
}

basic_set_list basic_set_list::concat(basic_set_list list2) const
{
  auto res = isl_basic_set_list_concat(copy(), list2.release());
  return manage(res);
}

basic_set_list basic_set_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_basic_set_list_drop(copy(), first, n);
  return manage(res);
}

stat basic_set_list::foreach(const std::function<stat(basic_set)> &fn) const
{
  struct fn_data {
    const std::function<stat(basic_set)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_basic_set *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_basic_set_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

basic_set_list basic_set_list::from_basic_set(basic_set el)
{
  auto res = isl_basic_set_list_from_basic_set(el.release());
  return manage(res);
}

basic_set basic_set_list::get_at(int index) const
{
  auto res = isl_basic_set_list_get_at(get(), index);
  return manage(res);
}

basic_set basic_set_list::get_basic_set(int index) const
{
  auto res = isl_basic_set_list_get_basic_set(get(), index);
  return manage(res);
}

basic_set_list basic_set_list::insert(unsigned int pos, basic_set el) const
{
  auto res = isl_basic_set_list_insert(copy(), pos, el.release());
  return manage(res);
}

int basic_set_list::n_basic_set() const
{
  auto res = isl_basic_set_list_n_basic_set(get());
  return res;
}

basic_set_list basic_set_list::reverse() const
{
  auto res = isl_basic_set_list_reverse(copy());
  return manage(res);
}

basic_set_list basic_set_list::set_basic_set(int index, basic_set el) const
{
  auto res = isl_basic_set_list_set_basic_set(copy(), index, el.release());
  return manage(res);
}

int basic_set_list::size() const
{
  auto res = isl_basic_set_list_size(get());
  return res;
}

basic_set_list basic_set_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_basic_set_list_swap(copy(), pos1, pos2);
  return manage(res);
}

// implementations for isl::constraint
constraint manage(__isl_take isl_constraint *ptr) {
  return constraint(ptr);
}
constraint manage_copy(__isl_keep isl_constraint *ptr) {
  ptr = isl_constraint_copy(ptr);
  return constraint(ptr);
}

constraint::constraint()
    : ptr(nullptr) {}

constraint::constraint(const constraint &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
constraint::constraint(std::nullptr_t)
    : ptr(nullptr) {}


constraint::constraint(__isl_take isl_constraint *ptr)
    : ptr(ptr) {}


constraint &constraint::operator=(constraint obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

constraint::~constraint() {
  if (ptr)
    isl_constraint_free(ptr);
}

__isl_give isl_constraint *constraint::copy() const & {
  return isl_constraint_copy(ptr);
}

__isl_keep isl_constraint *constraint::get() const {
  return ptr;
}

__isl_give isl_constraint *constraint::release() {
  isl_constraint *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool constraint::is_null() const {
  return ptr == nullptr;
}
constraint::operator bool() const {
  return !is_null();
}


ctx constraint::get_ctx() const {
  return ctx(isl_constraint_get_ctx(ptr));
}

void constraint::dump() const {
  isl_constraint_dump(get());
}


constraint constraint::alloc_equality(local_space ls)
{
  auto res = isl_constraint_alloc_equality(ls.release());
  return manage(res);
}

constraint constraint::alloc_inequality(local_space ls)
{
  auto res = isl_constraint_alloc_inequality(ls.release());
  return manage(res);
}

int constraint::cmp_last_non_zero(const constraint &c2) const
{
  auto res = isl_constraint_cmp_last_non_zero(get(), c2.get());
  return res;
}

aff constraint::get_aff() const
{
  auto res = isl_constraint_get_aff(get());
  return manage(res);
}

aff constraint::get_bound(isl::dim type, int pos) const
{
  auto res = isl_constraint_get_bound(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

val constraint::get_coefficient_val(isl::dim type, int pos) const
{
  auto res = isl_constraint_get_coefficient_val(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

val constraint::get_constant_val() const
{
  auto res = isl_constraint_get_constant_val(get());
  return manage(res);
}

std::string constraint::get_dim_name(isl::dim type, unsigned int pos) const
{
  auto res = isl_constraint_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

aff constraint::get_div(int pos) const
{
  auto res = isl_constraint_get_div(get(), pos);
  return manage(res);
}

local_space constraint::get_local_space() const
{
  auto res = isl_constraint_get_local_space(get());
  return manage(res);
}

space constraint::get_space() const
{
  auto res = isl_constraint_get_space(get());
  return manage(res);
}

boolean constraint::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_constraint_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

int constraint::is_div_constraint() const
{
  auto res = isl_constraint_is_div_constraint(get());
  return res;
}

boolean constraint::is_lower_bound(isl::dim type, unsigned int pos) const
{
  auto res = isl_constraint_is_lower_bound(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

boolean constraint::is_upper_bound(isl::dim type, unsigned int pos) const
{
  auto res = isl_constraint_is_upper_bound(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

int constraint::plain_cmp(const constraint &c2) const
{
  auto res = isl_constraint_plain_cmp(get(), c2.get());
  return res;
}

constraint constraint::set_coefficient_si(isl::dim type, int pos, int v) const
{
  auto res = isl_constraint_set_coefficient_si(copy(), static_cast<enum isl_dim_type>(type), pos, v);
  return manage(res);
}

constraint constraint::set_coefficient_val(isl::dim type, int pos, val v) const
{
  auto res = isl_constraint_set_coefficient_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

constraint constraint::set_constant_si(int v) const
{
  auto res = isl_constraint_set_constant_si(copy(), v);
  return manage(res);
}

constraint constraint::set_constant_val(val v) const
{
  auto res = isl_constraint_set_constant_val(copy(), v.release());
  return manage(res);
}

// implementations for isl::constraint_list
constraint_list manage(__isl_take isl_constraint_list *ptr) {
  return constraint_list(ptr);
}
constraint_list manage_copy(__isl_keep isl_constraint_list *ptr) {
  ptr = isl_constraint_list_copy(ptr);
  return constraint_list(ptr);
}

constraint_list::constraint_list()
    : ptr(nullptr) {}

constraint_list::constraint_list(const constraint_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
constraint_list::constraint_list(std::nullptr_t)
    : ptr(nullptr) {}


constraint_list::constraint_list(__isl_take isl_constraint_list *ptr)
    : ptr(ptr) {}


constraint_list &constraint_list::operator=(constraint_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

constraint_list::~constraint_list() {
  if (ptr)
    isl_constraint_list_free(ptr);
}

__isl_give isl_constraint_list *constraint_list::copy() const & {
  return isl_constraint_list_copy(ptr);
}

__isl_keep isl_constraint_list *constraint_list::get() const {
  return ptr;
}

__isl_give isl_constraint_list *constraint_list::release() {
  isl_constraint_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool constraint_list::is_null() const {
  return ptr == nullptr;
}
constraint_list::operator bool() const {
  return !is_null();
}


ctx constraint_list::get_ctx() const {
  return ctx(isl_constraint_list_get_ctx(ptr));
}

void constraint_list::dump() const {
  isl_constraint_list_dump(get());
}


constraint_list constraint_list::add(constraint el) const
{
  auto res = isl_constraint_list_add(copy(), el.release());
  return manage(res);
}

constraint_list constraint_list::alloc(ctx ctx, int n)
{
  auto res = isl_constraint_list_alloc(ctx.release(), n);
  return manage(res);
}

constraint_list constraint_list::concat(constraint_list list2) const
{
  auto res = isl_constraint_list_concat(copy(), list2.release());
  return manage(res);
}

constraint_list constraint_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_constraint_list_drop(copy(), first, n);
  return manage(res);
}

stat constraint_list::foreach(const std::function<stat(constraint)> &fn) const
{
  struct fn_data {
    const std::function<stat(constraint)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_constraint *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_constraint_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

constraint_list constraint_list::from_constraint(constraint el)
{
  auto res = isl_constraint_list_from_constraint(el.release());
  return manage(res);
}

constraint constraint_list::get_at(int index) const
{
  auto res = isl_constraint_list_get_at(get(), index);
  return manage(res);
}

constraint constraint_list::get_constraint(int index) const
{
  auto res = isl_constraint_list_get_constraint(get(), index);
  return manage(res);
}

constraint_list constraint_list::insert(unsigned int pos, constraint el) const
{
  auto res = isl_constraint_list_insert(copy(), pos, el.release());
  return manage(res);
}

int constraint_list::n_constraint() const
{
  auto res = isl_constraint_list_n_constraint(get());
  return res;
}

constraint_list constraint_list::reverse() const
{
  auto res = isl_constraint_list_reverse(copy());
  return manage(res);
}

constraint_list constraint_list::set_constraint(int index, constraint el) const
{
  auto res = isl_constraint_list_set_constraint(copy(), index, el.release());
  return manage(res);
}

int constraint_list::size() const
{
  auto res = isl_constraint_list_size(get());
  return res;
}

constraint_list constraint_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_constraint_list_swap(copy(), pos1, pos2);
  return manage(res);
}

// implementations for isl::fixed_box
fixed_box manage(__isl_take isl_fixed_box *ptr) {
  return fixed_box(ptr);
}
fixed_box manage_copy(__isl_keep isl_fixed_box *ptr) {
  ptr = isl_fixed_box_copy(ptr);
  return fixed_box(ptr);
}

fixed_box::fixed_box()
    : ptr(nullptr) {}

fixed_box::fixed_box(const fixed_box &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
fixed_box::fixed_box(std::nullptr_t)
    : ptr(nullptr) {}


fixed_box::fixed_box(__isl_take isl_fixed_box *ptr)
    : ptr(ptr) {}


fixed_box &fixed_box::operator=(fixed_box obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

fixed_box::~fixed_box() {
  if (ptr)
    isl_fixed_box_free(ptr);
}

__isl_give isl_fixed_box *fixed_box::copy() const & {
  return isl_fixed_box_copy(ptr);
}

__isl_keep isl_fixed_box *fixed_box::get() const {
  return ptr;
}

__isl_give isl_fixed_box *fixed_box::release() {
  isl_fixed_box *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool fixed_box::is_null() const {
  return ptr == nullptr;
}
fixed_box::operator bool() const {
  return !is_null();
}


ctx fixed_box::get_ctx() const {
  return ctx(isl_fixed_box_get_ctx(ptr));
}


multi_aff fixed_box::get_offset() const
{
  auto res = isl_fixed_box_get_offset(get());
  return manage(res);
}

multi_val fixed_box::get_size() const
{
  auto res = isl_fixed_box_get_size(get());
  return manage(res);
}

space fixed_box::get_space() const
{
  auto res = isl_fixed_box_get_space(get());
  return manage(res);
}

boolean fixed_box::is_valid() const
{
  auto res = isl_fixed_box_is_valid(get());
  return manage(res);
}

// implementations for isl::id
id manage(__isl_take isl_id *ptr) {
  return id(ptr);
}
id manage_copy(__isl_keep isl_id *ptr) {
  ptr = isl_id_copy(ptr);
  return id(ptr);
}

id::id()
    : ptr(nullptr) {}

id::id(const id &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
id::id(std::nullptr_t)
    : ptr(nullptr) {}


id::id(__isl_take isl_id *ptr)
    : ptr(ptr) {}


id &id::operator=(id obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

id::~id() {
  if (ptr)
    isl_id_free(ptr);
}

__isl_give isl_id *id::copy() const & {
  return isl_id_copy(ptr);
}

__isl_keep isl_id *id::get() const {
  return ptr;
}

__isl_give isl_id *id::release() {
  isl_id *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool id::is_null() const {
  return ptr == nullptr;
}
id::operator bool() const {
  return !is_null();
}


ctx id::get_ctx() const {
  return ctx(isl_id_get_ctx(ptr));
}
std::string id::to_str() const {
  char *Tmp = isl_id_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void id::dump() const {
  isl_id_dump(get());
}


id id::alloc(ctx ctx, const std::string &name, void * user)
{
  auto res = isl_id_alloc(ctx.release(), name.c_str(), user);
  return manage(res);
}

uint32_t id::get_hash() const
{
  auto res = isl_id_get_hash(get());
  return res;
}

std::string id::get_name() const
{
  auto res = isl_id_get_name(get());
  std::string tmp(res);
  return tmp;
}

void * id::get_user() const
{
  auto res = isl_id_get_user(get());
  return res;
}

// implementations for isl::id_list
id_list manage(__isl_take isl_id_list *ptr) {
  return id_list(ptr);
}
id_list manage_copy(__isl_keep isl_id_list *ptr) {
  ptr = isl_id_list_copy(ptr);
  return id_list(ptr);
}

id_list::id_list()
    : ptr(nullptr) {}

id_list::id_list(const id_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
id_list::id_list(std::nullptr_t)
    : ptr(nullptr) {}


id_list::id_list(__isl_take isl_id_list *ptr)
    : ptr(ptr) {}


id_list &id_list::operator=(id_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

id_list::~id_list() {
  if (ptr)
    isl_id_list_free(ptr);
}

__isl_give isl_id_list *id_list::copy() const & {
  return isl_id_list_copy(ptr);
}

__isl_keep isl_id_list *id_list::get() const {
  return ptr;
}

__isl_give isl_id_list *id_list::release() {
  isl_id_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool id_list::is_null() const {
  return ptr == nullptr;
}
id_list::operator bool() const {
  return !is_null();
}


ctx id_list::get_ctx() const {
  return ctx(isl_id_list_get_ctx(ptr));
}

void id_list::dump() const {
  isl_id_list_dump(get());
}


id_list id_list::add(id el) const
{
  auto res = isl_id_list_add(copy(), el.release());
  return manage(res);
}

id_list id_list::alloc(ctx ctx, int n)
{
  auto res = isl_id_list_alloc(ctx.release(), n);
  return manage(res);
}

id_list id_list::concat(id_list list2) const
{
  auto res = isl_id_list_concat(copy(), list2.release());
  return manage(res);
}

id_list id_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_id_list_drop(copy(), first, n);
  return manage(res);
}

stat id_list::foreach(const std::function<stat(id)> &fn) const
{
  struct fn_data {
    const std::function<stat(id)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_id *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_id_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

id_list id_list::from_id(id el)
{
  auto res = isl_id_list_from_id(el.release());
  return manage(res);
}

id id_list::get_at(int index) const
{
  auto res = isl_id_list_get_at(get(), index);
  return manage(res);
}

id id_list::get_id(int index) const
{
  auto res = isl_id_list_get_id(get(), index);
  return manage(res);
}

id_list id_list::insert(unsigned int pos, id el) const
{
  auto res = isl_id_list_insert(copy(), pos, el.release());
  return manage(res);
}

int id_list::n_id() const
{
  auto res = isl_id_list_n_id(get());
  return res;
}

id_list id_list::reverse() const
{
  auto res = isl_id_list_reverse(copy());
  return manage(res);
}

id_list id_list::set_id(int index, id el) const
{
  auto res = isl_id_list_set_id(copy(), index, el.release());
  return manage(res);
}

int id_list::size() const
{
  auto res = isl_id_list_size(get());
  return res;
}

id_list id_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_id_list_swap(copy(), pos1, pos2);
  return manage(res);
}

// implementations for isl::id_to_ast_expr
id_to_ast_expr manage(__isl_take isl_id_to_ast_expr *ptr) {
  return id_to_ast_expr(ptr);
}
id_to_ast_expr manage_copy(__isl_keep isl_id_to_ast_expr *ptr) {
  ptr = isl_id_to_ast_expr_copy(ptr);
  return id_to_ast_expr(ptr);
}

id_to_ast_expr::id_to_ast_expr()
    : ptr(nullptr) {}

id_to_ast_expr::id_to_ast_expr(const id_to_ast_expr &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
id_to_ast_expr::id_to_ast_expr(std::nullptr_t)
    : ptr(nullptr) {}


id_to_ast_expr::id_to_ast_expr(__isl_take isl_id_to_ast_expr *ptr)
    : ptr(ptr) {}


id_to_ast_expr &id_to_ast_expr::operator=(id_to_ast_expr obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

id_to_ast_expr::~id_to_ast_expr() {
  if (ptr)
    isl_id_to_ast_expr_free(ptr);
}

__isl_give isl_id_to_ast_expr *id_to_ast_expr::copy() const & {
  return isl_id_to_ast_expr_copy(ptr);
}

__isl_keep isl_id_to_ast_expr *id_to_ast_expr::get() const {
  return ptr;
}

__isl_give isl_id_to_ast_expr *id_to_ast_expr::release() {
  isl_id_to_ast_expr *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool id_to_ast_expr::is_null() const {
  return ptr == nullptr;
}
id_to_ast_expr::operator bool() const {
  return !is_null();
}


ctx id_to_ast_expr::get_ctx() const {
  return ctx(isl_id_to_ast_expr_get_ctx(ptr));
}

void id_to_ast_expr::dump() const {
  isl_id_to_ast_expr_dump(get());
}


id_to_ast_expr id_to_ast_expr::alloc(ctx ctx, int min_size)
{
  auto res = isl_id_to_ast_expr_alloc(ctx.release(), min_size);
  return manage(res);
}

id_to_ast_expr id_to_ast_expr::drop(id key) const
{
  auto res = isl_id_to_ast_expr_drop(copy(), key.release());
  return manage(res);
}

stat id_to_ast_expr::foreach(const std::function<stat(id, ast_expr)> &fn) const
{
  struct fn_data {
    const std::function<stat(id, ast_expr)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_id *arg_0, isl_ast_expr *arg_1, void *arg_2) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_2);
    stat ret = (*data->func)(manage(arg_0), manage(arg_1));
    return ret.release();
  };
  auto res = isl_id_to_ast_expr_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

ast_expr id_to_ast_expr::get(id key) const
{
  auto res = isl_id_to_ast_expr_get(get(), key.release());
  return manage(res);
}

boolean id_to_ast_expr::has(const id &key) const
{
  auto res = isl_id_to_ast_expr_has(get(), key.get());
  return manage(res);
}

id_to_ast_expr id_to_ast_expr::set(id key, ast_expr val) const
{
  auto res = isl_id_to_ast_expr_set(copy(), key.release(), val.release());
  return manage(res);
}

// implementations for isl::local_space
local_space manage(__isl_take isl_local_space *ptr) {
  return local_space(ptr);
}
local_space manage_copy(__isl_keep isl_local_space *ptr) {
  ptr = isl_local_space_copy(ptr);
  return local_space(ptr);
}

local_space::local_space()
    : ptr(nullptr) {}

local_space::local_space(const local_space &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
local_space::local_space(std::nullptr_t)
    : ptr(nullptr) {}


local_space::local_space(__isl_take isl_local_space *ptr)
    : ptr(ptr) {}

local_space::local_space(space dim)
{
  auto res = isl_local_space_from_space(dim.release());
  ptr = res;
}

local_space &local_space::operator=(local_space obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

local_space::~local_space() {
  if (ptr)
    isl_local_space_free(ptr);
}

__isl_give isl_local_space *local_space::copy() const & {
  return isl_local_space_copy(ptr);
}

__isl_keep isl_local_space *local_space::get() const {
  return ptr;
}

__isl_give isl_local_space *local_space::release() {
  isl_local_space *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool local_space::is_null() const {
  return ptr == nullptr;
}
local_space::operator bool() const {
  return !is_null();
}


ctx local_space::get_ctx() const {
  return ctx(isl_local_space_get_ctx(ptr));
}

void local_space::dump() const {
  isl_local_space_dump(get());
}


local_space local_space::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_local_space_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

int local_space::dim(isl::dim type) const
{
  auto res = isl_local_space_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

local_space local_space::domain() const
{
  auto res = isl_local_space_domain(copy());
  return manage(res);
}

local_space local_space::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_local_space_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

int local_space::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_local_space_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

local_space local_space::flatten_domain() const
{
  auto res = isl_local_space_flatten_domain(copy());
  return manage(res);
}

local_space local_space::flatten_range() const
{
  auto res = isl_local_space_flatten_range(copy());
  return manage(res);
}

local_space local_space::from_domain() const
{
  auto res = isl_local_space_from_domain(copy());
  return manage(res);
}

id local_space::get_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_local_space_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

std::string local_space::get_dim_name(isl::dim type, unsigned int pos) const
{
  auto res = isl_local_space_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

aff local_space::get_div(int pos) const
{
  auto res = isl_local_space_get_div(get(), pos);
  return manage(res);
}

space local_space::get_space() const
{
  auto res = isl_local_space_get_space(get());
  return manage(res);
}

boolean local_space::has_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_local_space_has_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

boolean local_space::has_dim_name(isl::dim type, unsigned int pos) const
{
  auto res = isl_local_space_has_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

local_space local_space::insert_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_local_space_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

local_space local_space::intersect(local_space ls2) const
{
  auto res = isl_local_space_intersect(copy(), ls2.release());
  return manage(res);
}

boolean local_space::is_equal(const local_space &ls2) const
{
  auto res = isl_local_space_is_equal(get(), ls2.get());
  return manage(res);
}

boolean local_space::is_params() const
{
  auto res = isl_local_space_is_params(get());
  return manage(res);
}

boolean local_space::is_set() const
{
  auto res = isl_local_space_is_set(get());
  return manage(res);
}

local_space local_space::range() const
{
  auto res = isl_local_space_range(copy());
  return manage(res);
}

local_space local_space::set_dim_id(isl::dim type, unsigned int pos, id id) const
{
  auto res = isl_local_space_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

local_space local_space::set_from_params() const
{
  auto res = isl_local_space_set_from_params(copy());
  return manage(res);
}

local_space local_space::set_tuple_id(isl::dim type, id id) const
{
  auto res = isl_local_space_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

local_space local_space::wrap() const
{
  auto res = isl_local_space_wrap(copy());
  return manage(res);
}

// implementations for isl::map
map manage(__isl_take isl_map *ptr) {
  return map(ptr);
}
map manage_copy(__isl_keep isl_map *ptr) {
  ptr = isl_map_copy(ptr);
  return map(ptr);
}

map::map()
    : ptr(nullptr) {}

map::map(const map &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
map::map(std::nullptr_t)
    : ptr(nullptr) {}


map::map(__isl_take isl_map *ptr)
    : ptr(ptr) {}

map::map(ctx ctx, const std::string &str)
{
  auto res = isl_map_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}
map::map(basic_map bmap)
{
  auto res = isl_map_from_basic_map(bmap.release());
  ptr = res;
}

map &map::operator=(map obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

map::~map() {
  if (ptr)
    isl_map_free(ptr);
}

__isl_give isl_map *map::copy() const & {
  return isl_map_copy(ptr);
}

__isl_keep isl_map *map::get() const {
  return ptr;
}

__isl_give isl_map *map::release() {
  isl_map *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool map::is_null() const {
  return ptr == nullptr;
}
map::operator bool() const {
  return !is_null();
}


ctx map::get_ctx() const {
  return ctx(isl_map_get_ctx(ptr));
}
std::string map::to_str() const {
  char *Tmp = isl_map_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void map::dump() const {
  isl_map_dump(get());
}


map map::add_constraint(constraint constraint) const
{
  auto res = isl_map_add_constraint(copy(), constraint.release());
  return manage(res);
}

map map::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_map_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

basic_map map::affine_hull() const
{
  auto res = isl_map_affine_hull(copy());
  return manage(res);
}

map map::align_params(space model) const
{
  auto res = isl_map_align_params(copy(), model.release());
  return manage(res);
}

map map::apply_domain(map map2) const
{
  auto res = isl_map_apply_domain(copy(), map2.release());
  return manage(res);
}

map map::apply_range(map map2) const
{
  auto res = isl_map_apply_range(copy(), map2.release());
  return manage(res);
}

boolean map::can_curry() const
{
  auto res = isl_map_can_curry(get());
  return manage(res);
}

boolean map::can_range_curry() const
{
  auto res = isl_map_can_range_curry(get());
  return manage(res);
}

boolean map::can_uncurry() const
{
  auto res = isl_map_can_uncurry(get());
  return manage(res);
}

boolean map::can_zip() const
{
  auto res = isl_map_can_zip(get());
  return manage(res);
}

map map::coalesce() const
{
  auto res = isl_map_coalesce(copy());
  return manage(res);
}

map map::complement() const
{
  auto res = isl_map_complement(copy());
  return manage(res);
}

basic_map map::convex_hull() const
{
  auto res = isl_map_convex_hull(copy());
  return manage(res);
}

map map::curry() const
{
  auto res = isl_map_curry(copy());
  return manage(res);
}

set map::deltas() const
{
  auto res = isl_map_deltas(copy());
  return manage(res);
}

map map::deltas_map() const
{
  auto res = isl_map_deltas_map(copy());
  return manage(res);
}

map map::detect_equalities() const
{
  auto res = isl_map_detect_equalities(copy());
  return manage(res);
}

unsigned int map::dim(isl::dim type) const
{
  auto res = isl_map_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

pw_aff map::dim_max(int pos) const
{
  auto res = isl_map_dim_max(copy(), pos);
  return manage(res);
}

pw_aff map::dim_min(int pos) const
{
  auto res = isl_map_dim_min(copy(), pos);
  return manage(res);
}

set map::domain() const
{
  auto res = isl_map_domain(copy());
  return manage(res);
}

map map::domain_factor_domain() const
{
  auto res = isl_map_domain_factor_domain(copy());
  return manage(res);
}

map map::domain_factor_range() const
{
  auto res = isl_map_domain_factor_range(copy());
  return manage(res);
}

boolean map::domain_is_wrapping() const
{
  auto res = isl_map_domain_is_wrapping(get());
  return manage(res);
}

map map::domain_map() const
{
  auto res = isl_map_domain_map(copy());
  return manage(res);
}

map map::domain_product(map map2) const
{
  auto res = isl_map_domain_product(copy(), map2.release());
  return manage(res);
}

map map::drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_map_drop_constraints_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

map map::drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_map_drop_constraints_not_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

map map::drop_unused_params() const
{
  auto res = isl_map_drop_unused_params(copy());
  return manage(res);
}

map map::eliminate(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_map_eliminate(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

map map::empty(space space)
{
  auto res = isl_map_empty(space.release());
  return manage(res);
}

map map::equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_map_equate(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

map map::factor_domain() const
{
  auto res = isl_map_factor_domain(copy());
  return manage(res);
}

map map::factor_range() const
{
  auto res = isl_map_factor_range(copy());
  return manage(res);
}

int map::find_dim_by_id(isl::dim type, const id &id) const
{
  auto res = isl_map_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int map::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_map_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

map map::fix_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_map_fix_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

map map::fix_val(isl::dim type, unsigned int pos, val v) const
{
  auto res = isl_map_fix_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

map map::fixed_power_val(val exp) const
{
  auto res = isl_map_fixed_power_val(copy(), exp.release());
  return manage(res);
}

map map::flat_domain_product(map map2) const
{
  auto res = isl_map_flat_domain_product(copy(), map2.release());
  return manage(res);
}

map map::flat_product(map map2) const
{
  auto res = isl_map_flat_product(copy(), map2.release());
  return manage(res);
}

map map::flat_range_product(map map2) const
{
  auto res = isl_map_flat_range_product(copy(), map2.release());
  return manage(res);
}

map map::flatten() const
{
  auto res = isl_map_flatten(copy());
  return manage(res);
}

map map::flatten_domain() const
{
  auto res = isl_map_flatten_domain(copy());
  return manage(res);
}

map map::flatten_range() const
{
  auto res = isl_map_flatten_range(copy());
  return manage(res);
}

map map::floordiv_val(val d) const
{
  auto res = isl_map_floordiv_val(copy(), d.release());
  return manage(res);
}

stat map::foreach_basic_map(const std::function<stat(basic_map)> &fn) const
{
  struct fn_data {
    const std::function<stat(basic_map)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_basic_map *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_map_foreach_basic_map(get(), fn_lambda, &fn_data);
  return manage(res);
}

map map::from_aff(aff aff)
{
  auto res = isl_map_from_aff(aff.release());
  return manage(res);
}

map map::from_domain(set set)
{
  auto res = isl_map_from_domain(set.release());
  return manage(res);
}

map map::from_domain_and_range(set domain, set range)
{
  auto res = isl_map_from_domain_and_range(domain.release(), range.release());
  return manage(res);
}

map map::from_multi_aff(multi_aff maff)
{
  auto res = isl_map_from_multi_aff(maff.release());
  return manage(res);
}

map map::from_multi_pw_aff(multi_pw_aff mpa)
{
  auto res = isl_map_from_multi_pw_aff(mpa.release());
  return manage(res);
}

map map::from_pw_aff(pw_aff pwaff)
{
  auto res = isl_map_from_pw_aff(pwaff.release());
  return manage(res);
}

map map::from_pw_multi_aff(pw_multi_aff pma)
{
  auto res = isl_map_from_pw_multi_aff(pma.release());
  return manage(res);
}

map map::from_range(set set)
{
  auto res = isl_map_from_range(set.release());
  return manage(res);
}

map map::from_union_map(union_map umap)
{
  auto res = isl_map_from_union_map(umap.release());
  return manage(res);
}

basic_map_list map::get_basic_map_list() const
{
  auto res = isl_map_get_basic_map_list(get());
  return manage(res);
}

id map::get_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_map_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

std::string map::get_dim_name(isl::dim type, unsigned int pos) const
{
  auto res = isl_map_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

uint32_t map::get_hash() const
{
  auto res = isl_map_get_hash(get());
  return res;
}

fixed_box map::get_range_simple_fixed_box_hull() const
{
  auto res = isl_map_get_range_simple_fixed_box_hull(get());
  return manage(res);
}

space map::get_space() const
{
  auto res = isl_map_get_space(get());
  return manage(res);
}

id map::get_tuple_id(isl::dim type) const
{
  auto res = isl_map_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

std::string map::get_tuple_name(isl::dim type) const
{
  auto res = isl_map_get_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  std::string tmp(res);
  return tmp;
}

map map::gist(map context) const
{
  auto res = isl_map_gist(copy(), context.release());
  return manage(res);
}

map map::gist_basic_map(basic_map context) const
{
  auto res = isl_map_gist_basic_map(copy(), context.release());
  return manage(res);
}

map map::gist_domain(set context) const
{
  auto res = isl_map_gist_domain(copy(), context.release());
  return manage(res);
}

map map::gist_params(set context) const
{
  auto res = isl_map_gist_params(copy(), context.release());
  return manage(res);
}

map map::gist_range(set context) const
{
  auto res = isl_map_gist_range(copy(), context.release());
  return manage(res);
}

boolean map::has_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_map_has_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

boolean map::has_dim_name(isl::dim type, unsigned int pos) const
{
  auto res = isl_map_has_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

boolean map::has_equal_space(const map &map2) const
{
  auto res = isl_map_has_equal_space(get(), map2.get());
  return manage(res);
}

boolean map::has_tuple_id(isl::dim type) const
{
  auto res = isl_map_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

boolean map::has_tuple_name(isl::dim type) const
{
  auto res = isl_map_has_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

map map::identity(space dim)
{
  auto res = isl_map_identity(dim.release());
  return manage(res);
}

map map::insert_dims(isl::dim type, unsigned int pos, unsigned int n) const
{
  auto res = isl_map_insert_dims(copy(), static_cast<enum isl_dim_type>(type), pos, n);
  return manage(res);
}

map map::intersect(map map2) const
{
  auto res = isl_map_intersect(copy(), map2.release());
  return manage(res);
}

map map::intersect_domain(set set) const
{
  auto res = isl_map_intersect_domain(copy(), set.release());
  return manage(res);
}

map map::intersect_domain_factor_range(map factor) const
{
  auto res = isl_map_intersect_domain_factor_range(copy(), factor.release());
  return manage(res);
}

map map::intersect_params(set params) const
{
  auto res = isl_map_intersect_params(copy(), params.release());
  return manage(res);
}

map map::intersect_range(set set) const
{
  auto res = isl_map_intersect_range(copy(), set.release());
  return manage(res);
}

map map::intersect_range_factor_range(map factor) const
{
  auto res = isl_map_intersect_range_factor_range(copy(), factor.release());
  return manage(res);
}

boolean map::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_map_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean map::is_bijective() const
{
  auto res = isl_map_is_bijective(get());
  return manage(res);
}

boolean map::is_disjoint(const map &map2) const
{
  auto res = isl_map_is_disjoint(get(), map2.get());
  return manage(res);
}

boolean map::is_empty() const
{
  auto res = isl_map_is_empty(get());
  return manage(res);
}

boolean map::is_equal(const map &map2) const
{
  auto res = isl_map_is_equal(get(), map2.get());
  return manage(res);
}

boolean map::is_identity() const
{
  auto res = isl_map_is_identity(get());
  return manage(res);
}

boolean map::is_injective() const
{
  auto res = isl_map_is_injective(get());
  return manage(res);
}

boolean map::is_product() const
{
  auto res = isl_map_is_product(get());
  return manage(res);
}

boolean map::is_single_valued() const
{
  auto res = isl_map_is_single_valued(get());
  return manage(res);
}

boolean map::is_strict_subset(const map &map2) const
{
  auto res = isl_map_is_strict_subset(get(), map2.get());
  return manage(res);
}

boolean map::is_subset(const map &map2) const
{
  auto res = isl_map_is_subset(get(), map2.get());
  return manage(res);
}

int map::is_translation() const
{
  auto res = isl_map_is_translation(get());
  return res;
}

map map::lex_ge(space set_dim)
{
  auto res = isl_map_lex_ge(set_dim.release());
  return manage(res);
}

map map::lex_ge_first(space dim, unsigned int n)
{
  auto res = isl_map_lex_ge_first(dim.release(), n);
  return manage(res);
}

map map::lex_ge_map(map map2) const
{
  auto res = isl_map_lex_ge_map(copy(), map2.release());
  return manage(res);
}

map map::lex_gt(space set_dim)
{
  auto res = isl_map_lex_gt(set_dim.release());
  return manage(res);
}

map map::lex_gt_first(space dim, unsigned int n)
{
  auto res = isl_map_lex_gt_first(dim.release(), n);
  return manage(res);
}

map map::lex_gt_map(map map2) const
{
  auto res = isl_map_lex_gt_map(copy(), map2.release());
  return manage(res);
}

map map::lex_le(space set_dim)
{
  auto res = isl_map_lex_le(set_dim.release());
  return manage(res);
}

map map::lex_le_first(space dim, unsigned int n)
{
  auto res = isl_map_lex_le_first(dim.release(), n);
  return manage(res);
}

map map::lex_le_map(map map2) const
{
  auto res = isl_map_lex_le_map(copy(), map2.release());
  return manage(res);
}

map map::lex_lt(space set_dim)
{
  auto res = isl_map_lex_lt(set_dim.release());
  return manage(res);
}

map map::lex_lt_first(space dim, unsigned int n)
{
  auto res = isl_map_lex_lt_first(dim.release(), n);
  return manage(res);
}

map map::lex_lt_map(map map2) const
{
  auto res = isl_map_lex_lt_map(copy(), map2.release());
  return manage(res);
}

map map::lexmax() const
{
  auto res = isl_map_lexmax(copy());
  return manage(res);
}

pw_multi_aff map::lexmax_pw_multi_aff() const
{
  auto res = isl_map_lexmax_pw_multi_aff(copy());
  return manage(res);
}

map map::lexmin() const
{
  auto res = isl_map_lexmin(copy());
  return manage(res);
}

pw_multi_aff map::lexmin_pw_multi_aff() const
{
  auto res = isl_map_lexmin_pw_multi_aff(copy());
  return manage(res);
}

map map::lower_bound_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_map_lower_bound_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

map map::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_map_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

int map::n_basic_map() const
{
  auto res = isl_map_n_basic_map(get());
  return res;
}

map map::nat_universe(space dim)
{
  auto res = isl_map_nat_universe(dim.release());
  return manage(res);
}

map map::neg() const
{
  auto res = isl_map_neg(copy());
  return manage(res);
}

map map::oppose(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_map_oppose(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

map map::order_ge(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_map_order_ge(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

map map::order_gt(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_map_order_gt(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

map map::order_le(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_map_order_le(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

map map::order_lt(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_map_order_lt(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

set map::params() const
{
  auto res = isl_map_params(copy());
  return manage(res);
}

val map::plain_get_val_if_fixed(isl::dim type, unsigned int pos) const
{
  auto res = isl_map_plain_get_val_if_fixed(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

boolean map::plain_is_empty() const
{
  auto res = isl_map_plain_is_empty(get());
  return manage(res);
}

boolean map::plain_is_equal(const map &map2) const
{
  auto res = isl_map_plain_is_equal(get(), map2.get());
  return manage(res);
}

boolean map::plain_is_injective() const
{
  auto res = isl_map_plain_is_injective(get());
  return manage(res);
}

boolean map::plain_is_single_valued() const
{
  auto res = isl_map_plain_is_single_valued(get());
  return manage(res);
}

boolean map::plain_is_universe() const
{
  auto res = isl_map_plain_is_universe(get());
  return manage(res);
}

basic_map map::plain_unshifted_simple_hull() const
{
  auto res = isl_map_plain_unshifted_simple_hull(copy());
  return manage(res);
}

basic_map map::polyhedral_hull() const
{
  auto res = isl_map_polyhedral_hull(copy());
  return manage(res);
}

map map::preimage_domain_multi_aff(multi_aff ma) const
{
  auto res = isl_map_preimage_domain_multi_aff(copy(), ma.release());
  return manage(res);
}

map map::preimage_domain_multi_pw_aff(multi_pw_aff mpa) const
{
  auto res = isl_map_preimage_domain_multi_pw_aff(copy(), mpa.release());
  return manage(res);
}

map map::preimage_domain_pw_multi_aff(pw_multi_aff pma) const
{
  auto res = isl_map_preimage_domain_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

map map::preimage_range_multi_aff(multi_aff ma) const
{
  auto res = isl_map_preimage_range_multi_aff(copy(), ma.release());
  return manage(res);
}

map map::preimage_range_pw_multi_aff(pw_multi_aff pma) const
{
  auto res = isl_map_preimage_range_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

map map::product(map map2) const
{
  auto res = isl_map_product(copy(), map2.release());
  return manage(res);
}

map map::project_out(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_map_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

set map::range() const
{
  auto res = isl_map_range(copy());
  return manage(res);
}

map map::range_curry() const
{
  auto res = isl_map_range_curry(copy());
  return manage(res);
}

map map::range_factor_domain() const
{
  auto res = isl_map_range_factor_domain(copy());
  return manage(res);
}

map map::range_factor_range() const
{
  auto res = isl_map_range_factor_range(copy());
  return manage(res);
}

boolean map::range_is_wrapping() const
{
  auto res = isl_map_range_is_wrapping(get());
  return manage(res);
}

map map::range_map() const
{
  auto res = isl_map_range_map(copy());
  return manage(res);
}

map map::range_product(map map2) const
{
  auto res = isl_map_range_product(copy(), map2.release());
  return manage(res);
}

map map::remove_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_map_remove_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

map map::remove_divs() const
{
  auto res = isl_map_remove_divs(copy());
  return manage(res);
}

map map::remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_map_remove_divs_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

map map::remove_redundancies() const
{
  auto res = isl_map_remove_redundancies(copy());
  return manage(res);
}

map map::remove_unknown_divs() const
{
  auto res = isl_map_remove_unknown_divs(copy());
  return manage(res);
}

map map::reset_tuple_id(isl::dim type) const
{
  auto res = isl_map_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

map map::reset_user() const
{
  auto res = isl_map_reset_user(copy());
  return manage(res);
}

map map::reverse() const
{
  auto res = isl_map_reverse(copy());
  return manage(res);
}

basic_map map::sample() const
{
  auto res = isl_map_sample(copy());
  return manage(res);
}

map map::set_dim_id(isl::dim type, unsigned int pos, id id) const
{
  auto res = isl_map_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

map map::set_tuple_id(isl::dim type, id id) const
{
  auto res = isl_map_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

map map::set_tuple_name(isl::dim type, const std::string &s) const
{
  auto res = isl_map_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

basic_map map::simple_hull() const
{
  auto res = isl_map_simple_hull(copy());
  return manage(res);
}

map map::subtract(map map2) const
{
  auto res = isl_map_subtract(copy(), map2.release());
  return manage(res);
}

map map::subtract_domain(set dom) const
{
  auto res = isl_map_subtract_domain(copy(), dom.release());
  return manage(res);
}

map map::subtract_range(set dom) const
{
  auto res = isl_map_subtract_range(copy(), dom.release());
  return manage(res);
}

map map::sum(map map2) const
{
  auto res = isl_map_sum(copy(), map2.release());
  return manage(res);
}

map map::uncurry() const
{
  auto res = isl_map_uncurry(copy());
  return manage(res);
}

map map::unite(map map2) const
{
  auto res = isl_map_union(copy(), map2.release());
  return manage(res);
}

map map::universe(space space)
{
  auto res = isl_map_universe(space.release());
  return manage(res);
}

basic_map map::unshifted_simple_hull() const
{
  auto res = isl_map_unshifted_simple_hull(copy());
  return manage(res);
}

basic_map map::unshifted_simple_hull_from_map_list(map_list list) const
{
  auto res = isl_map_unshifted_simple_hull_from_map_list(copy(), list.release());
  return manage(res);
}

map map::upper_bound_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_map_upper_bound_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

set map::wrap() const
{
  auto res = isl_map_wrap(copy());
  return manage(res);
}

map map::zip() const
{
  auto res = isl_map_zip(copy());
  return manage(res);
}

// implementations for isl::map_list
map_list manage(__isl_take isl_map_list *ptr) {
  return map_list(ptr);
}
map_list manage_copy(__isl_keep isl_map_list *ptr) {
  ptr = isl_map_list_copy(ptr);
  return map_list(ptr);
}

map_list::map_list()
    : ptr(nullptr) {}

map_list::map_list(const map_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
map_list::map_list(std::nullptr_t)
    : ptr(nullptr) {}


map_list::map_list(__isl_take isl_map_list *ptr)
    : ptr(ptr) {}


map_list &map_list::operator=(map_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

map_list::~map_list() {
  if (ptr)
    isl_map_list_free(ptr);
}

__isl_give isl_map_list *map_list::copy() const & {
  return isl_map_list_copy(ptr);
}

__isl_keep isl_map_list *map_list::get() const {
  return ptr;
}

__isl_give isl_map_list *map_list::release() {
  isl_map_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool map_list::is_null() const {
  return ptr == nullptr;
}
map_list::operator bool() const {
  return !is_null();
}


ctx map_list::get_ctx() const {
  return ctx(isl_map_list_get_ctx(ptr));
}

void map_list::dump() const {
  isl_map_list_dump(get());
}


map_list map_list::add(map el) const
{
  auto res = isl_map_list_add(copy(), el.release());
  return manage(res);
}

map_list map_list::alloc(ctx ctx, int n)
{
  auto res = isl_map_list_alloc(ctx.release(), n);
  return manage(res);
}

map_list map_list::concat(map_list list2) const
{
  auto res = isl_map_list_concat(copy(), list2.release());
  return manage(res);
}

map_list map_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_map_list_drop(copy(), first, n);
  return manage(res);
}

stat map_list::foreach(const std::function<stat(map)> &fn) const
{
  struct fn_data {
    const std::function<stat(map)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_map *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_map_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

map_list map_list::from_map(map el)
{
  auto res = isl_map_list_from_map(el.release());
  return manage(res);
}

map map_list::get_at(int index) const
{
  auto res = isl_map_list_get_at(get(), index);
  return manage(res);
}

map map_list::get_map(int index) const
{
  auto res = isl_map_list_get_map(get(), index);
  return manage(res);
}

map_list map_list::insert(unsigned int pos, map el) const
{
  auto res = isl_map_list_insert(copy(), pos, el.release());
  return manage(res);
}

int map_list::n_map() const
{
  auto res = isl_map_list_n_map(get());
  return res;
}

map_list map_list::reverse() const
{
  auto res = isl_map_list_reverse(copy());
  return manage(res);
}

map_list map_list::set_map(int index, map el) const
{
  auto res = isl_map_list_set_map(copy(), index, el.release());
  return manage(res);
}

int map_list::size() const
{
  auto res = isl_map_list_size(get());
  return res;
}

map_list map_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_map_list_swap(copy(), pos1, pos2);
  return manage(res);
}

// implementations for isl::mat
mat manage(__isl_take isl_mat *ptr) {
  return mat(ptr);
}
mat manage_copy(__isl_keep isl_mat *ptr) {
  ptr = isl_mat_copy(ptr);
  return mat(ptr);
}

mat::mat()
    : ptr(nullptr) {}

mat::mat(const mat &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
mat::mat(std::nullptr_t)
    : ptr(nullptr) {}


mat::mat(__isl_take isl_mat *ptr)
    : ptr(ptr) {}


mat &mat::operator=(mat obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

mat::~mat() {
  if (ptr)
    isl_mat_free(ptr);
}

__isl_give isl_mat *mat::copy() const & {
  return isl_mat_copy(ptr);
}

__isl_keep isl_mat *mat::get() const {
  return ptr;
}

__isl_give isl_mat *mat::release() {
  isl_mat *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool mat::is_null() const {
  return ptr == nullptr;
}
mat::operator bool() const {
  return !is_null();
}


ctx mat::get_ctx() const {
  return ctx(isl_mat_get_ctx(ptr));
}

void mat::dump() const {
  isl_mat_dump(get());
}


mat mat::add_rows(unsigned int n) const
{
  auto res = isl_mat_add_rows(copy(), n);
  return manage(res);
}

mat mat::add_zero_cols(unsigned int n) const
{
  auto res = isl_mat_add_zero_cols(copy(), n);
  return manage(res);
}

mat mat::add_zero_rows(unsigned int n) const
{
  auto res = isl_mat_add_zero_rows(copy(), n);
  return manage(res);
}

mat mat::aff_direct_sum(mat right) const
{
  auto res = isl_mat_aff_direct_sum(copy(), right.release());
  return manage(res);
}

mat mat::alloc(ctx ctx, unsigned int n_row, unsigned int n_col)
{
  auto res = isl_mat_alloc(ctx.release(), n_row, n_col);
  return manage(res);
}

int mat::cols() const
{
  auto res = isl_mat_cols(get());
  return res;
}

mat mat::concat(mat bot) const
{
  auto res = isl_mat_concat(copy(), bot.release());
  return manage(res);
}

mat mat::diagonal(mat mat2) const
{
  auto res = isl_mat_diagonal(copy(), mat2.release());
  return manage(res);
}

mat mat::drop_cols(unsigned int col, unsigned int n) const
{
  auto res = isl_mat_drop_cols(copy(), col, n);
  return manage(res);
}

mat mat::drop_rows(unsigned int row, unsigned int n) const
{
  auto res = isl_mat_drop_rows(copy(), row, n);
  return manage(res);
}

mat mat::from_row_vec(vec vec)
{
  auto res = isl_mat_from_row_vec(vec.release());
  return manage(res);
}

val mat::get_element_val(int row, int col) const
{
  auto res = isl_mat_get_element_val(get(), row, col);
  return manage(res);
}

boolean mat::has_linearly_independent_rows(const mat &mat2) const
{
  auto res = isl_mat_has_linearly_independent_rows(get(), mat2.get());
  return manage(res);
}

int mat::initial_non_zero_cols() const
{
  auto res = isl_mat_initial_non_zero_cols(get());
  return res;
}

mat mat::insert_cols(unsigned int col, unsigned int n) const
{
  auto res = isl_mat_insert_cols(copy(), col, n);
  return manage(res);
}

mat mat::insert_rows(unsigned int row, unsigned int n) const
{
  auto res = isl_mat_insert_rows(copy(), row, n);
  return manage(res);
}

mat mat::insert_zero_cols(unsigned int first, unsigned int n) const
{
  auto res = isl_mat_insert_zero_cols(copy(), first, n);
  return manage(res);
}

mat mat::insert_zero_rows(unsigned int row, unsigned int n) const
{
  auto res = isl_mat_insert_zero_rows(copy(), row, n);
  return manage(res);
}

mat mat::inverse_product(mat right) const
{
  auto res = isl_mat_inverse_product(copy(), right.release());
  return manage(res);
}

boolean mat::is_equal(const mat &mat2) const
{
  auto res = isl_mat_is_equal(get(), mat2.get());
  return manage(res);
}

mat mat::lin_to_aff() const
{
  auto res = isl_mat_lin_to_aff(copy());
  return manage(res);
}

mat mat::move_cols(unsigned int dst_col, unsigned int src_col, unsigned int n) const
{
  auto res = isl_mat_move_cols(copy(), dst_col, src_col, n);
  return manage(res);
}

mat mat::normalize() const
{
  auto res = isl_mat_normalize(copy());
  return manage(res);
}

mat mat::normalize_row(int row) const
{
  auto res = isl_mat_normalize_row(copy(), row);
  return manage(res);
}

mat mat::product(mat right) const
{
  auto res = isl_mat_product(copy(), right.release());
  return manage(res);
}

int mat::rank() const
{
  auto res = isl_mat_rank(get());
  return res;
}

mat mat::right_inverse() const
{
  auto res = isl_mat_right_inverse(copy());
  return manage(res);
}

mat mat::right_kernel() const
{
  auto res = isl_mat_right_kernel(copy());
  return manage(res);
}

mat mat::row_basis() const
{
  auto res = isl_mat_row_basis(copy());
  return manage(res);
}

mat mat::row_basis_extension(mat mat2) const
{
  auto res = isl_mat_row_basis_extension(copy(), mat2.release());
  return manage(res);
}

int mat::rows() const
{
  auto res = isl_mat_rows(get());
  return res;
}

mat mat::set_element_si(int row, int col, int v) const
{
  auto res = isl_mat_set_element_si(copy(), row, col, v);
  return manage(res);
}

mat mat::set_element_val(int row, int col, val v) const
{
  auto res = isl_mat_set_element_val(copy(), row, col, v.release());
  return manage(res);
}

mat mat::swap_cols(unsigned int i, unsigned int j) const
{
  auto res = isl_mat_swap_cols(copy(), i, j);
  return manage(res);
}

mat mat::swap_rows(unsigned int i, unsigned int j) const
{
  auto res = isl_mat_swap_rows(copy(), i, j);
  return manage(res);
}

mat mat::transpose() const
{
  auto res = isl_mat_transpose(copy());
  return manage(res);
}

mat mat::unimodular_complete(int row) const
{
  auto res = isl_mat_unimodular_complete(copy(), row);
  return manage(res);
}

mat mat::vec_concat(vec bot) const
{
  auto res = isl_mat_vec_concat(copy(), bot.release());
  return manage(res);
}

vec mat::vec_inverse_product(vec vec) const
{
  auto res = isl_mat_vec_inverse_product(copy(), vec.release());
  return manage(res);
}

vec mat::vec_product(vec vec) const
{
  auto res = isl_mat_vec_product(copy(), vec.release());
  return manage(res);
}

// implementations for isl::multi_aff
multi_aff manage(__isl_take isl_multi_aff *ptr) {
  return multi_aff(ptr);
}
multi_aff manage_copy(__isl_keep isl_multi_aff *ptr) {
  ptr = isl_multi_aff_copy(ptr);
  return multi_aff(ptr);
}

multi_aff::multi_aff()
    : ptr(nullptr) {}

multi_aff::multi_aff(const multi_aff &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
multi_aff::multi_aff(std::nullptr_t)
    : ptr(nullptr) {}


multi_aff::multi_aff(__isl_take isl_multi_aff *ptr)
    : ptr(ptr) {}

multi_aff::multi_aff(aff aff)
{
  auto res = isl_multi_aff_from_aff(aff.release());
  ptr = res;
}
multi_aff::multi_aff(ctx ctx, const std::string &str)
{
  auto res = isl_multi_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

multi_aff &multi_aff::operator=(multi_aff obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

multi_aff::~multi_aff() {
  if (ptr)
    isl_multi_aff_free(ptr);
}

__isl_give isl_multi_aff *multi_aff::copy() const & {
  return isl_multi_aff_copy(ptr);
}

__isl_keep isl_multi_aff *multi_aff::get() const {
  return ptr;
}

__isl_give isl_multi_aff *multi_aff::release() {
  isl_multi_aff *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool multi_aff::is_null() const {
  return ptr == nullptr;
}
multi_aff::operator bool() const {
  return !is_null();
}


ctx multi_aff::get_ctx() const {
  return ctx(isl_multi_aff_get_ctx(ptr));
}
std::string multi_aff::to_str() const {
  char *Tmp = isl_multi_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void multi_aff::dump() const {
  isl_multi_aff_dump(get());
}


multi_aff multi_aff::add(multi_aff multi2) const
{
  auto res = isl_multi_aff_add(copy(), multi2.release());
  return manage(res);
}

multi_aff multi_aff::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_multi_aff_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

multi_aff multi_aff::align_params(space model) const
{
  auto res = isl_multi_aff_align_params(copy(), model.release());
  return manage(res);
}

unsigned int multi_aff::dim(isl::dim type) const
{
  auto res = isl_multi_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

multi_aff multi_aff::domain_map(space space)
{
  auto res = isl_multi_aff_domain_map(space.release());
  return manage(res);
}

multi_aff multi_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_multi_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

multi_aff multi_aff::factor_range() const
{
  auto res = isl_multi_aff_factor_range(copy());
  return manage(res);
}

int multi_aff::find_dim_by_id(isl::dim type, const id &id) const
{
  auto res = isl_multi_aff_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int multi_aff::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_multi_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

multi_aff multi_aff::flat_range_product(multi_aff multi2) const
{
  auto res = isl_multi_aff_flat_range_product(copy(), multi2.release());
  return manage(res);
}

multi_aff multi_aff::flatten_domain() const
{
  auto res = isl_multi_aff_flatten_domain(copy());
  return manage(res);
}

multi_aff multi_aff::flatten_range() const
{
  auto res = isl_multi_aff_flatten_range(copy());
  return manage(res);
}

multi_aff multi_aff::floor() const
{
  auto res = isl_multi_aff_floor(copy());
  return manage(res);
}

multi_aff multi_aff::from_aff_list(space space, aff_list list)
{
  auto res = isl_multi_aff_from_aff_list(space.release(), list.release());
  return manage(res);
}

multi_aff multi_aff::from_range() const
{
  auto res = isl_multi_aff_from_range(copy());
  return manage(res);
}

aff multi_aff::get_aff(int pos) const
{
  auto res = isl_multi_aff_get_aff(get(), pos);
  return manage(res);
}

id multi_aff::get_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_multi_aff_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

space multi_aff::get_domain_space() const
{
  auto res = isl_multi_aff_get_domain_space(get());
  return manage(res);
}

space multi_aff::get_space() const
{
  auto res = isl_multi_aff_get_space(get());
  return manage(res);
}

id multi_aff::get_tuple_id(isl::dim type) const
{
  auto res = isl_multi_aff_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

std::string multi_aff::get_tuple_name(isl::dim type) const
{
  auto res = isl_multi_aff_get_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  std::string tmp(res);
  return tmp;
}

multi_aff multi_aff::gist(set context) const
{
  auto res = isl_multi_aff_gist(copy(), context.release());
  return manage(res);
}

multi_aff multi_aff::gist_params(set context) const
{
  auto res = isl_multi_aff_gist_params(copy(), context.release());
  return manage(res);
}

boolean multi_aff::has_tuple_id(isl::dim type) const
{
  auto res = isl_multi_aff_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

multi_aff multi_aff::identity(space space)
{
  auto res = isl_multi_aff_identity(space.release());
  return manage(res);
}

multi_aff multi_aff::insert_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_multi_aff_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean multi_aff::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_multi_aff_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean multi_aff::involves_nan() const
{
  auto res = isl_multi_aff_involves_nan(get());
  return manage(res);
}

set multi_aff::lex_ge_set(multi_aff ma2) const
{
  auto res = isl_multi_aff_lex_ge_set(copy(), ma2.release());
  return manage(res);
}

set multi_aff::lex_gt_set(multi_aff ma2) const
{
  auto res = isl_multi_aff_lex_gt_set(copy(), ma2.release());
  return manage(res);
}

set multi_aff::lex_le_set(multi_aff ma2) const
{
  auto res = isl_multi_aff_lex_le_set(copy(), ma2.release());
  return manage(res);
}

set multi_aff::lex_lt_set(multi_aff ma2) const
{
  auto res = isl_multi_aff_lex_lt_set(copy(), ma2.release());
  return manage(res);
}

multi_aff multi_aff::mod_multi_val(multi_val mv) const
{
  auto res = isl_multi_aff_mod_multi_val(copy(), mv.release());
  return manage(res);
}

multi_aff multi_aff::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_multi_aff_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

multi_aff multi_aff::multi_val_on_space(space space, multi_val mv)
{
  auto res = isl_multi_aff_multi_val_on_space(space.release(), mv.release());
  return manage(res);
}

multi_aff multi_aff::neg() const
{
  auto res = isl_multi_aff_neg(copy());
  return manage(res);
}

int multi_aff::plain_cmp(const multi_aff &multi2) const
{
  auto res = isl_multi_aff_plain_cmp(get(), multi2.get());
  return res;
}

boolean multi_aff::plain_is_equal(const multi_aff &multi2) const
{
  auto res = isl_multi_aff_plain_is_equal(get(), multi2.get());
  return manage(res);
}

multi_aff multi_aff::product(multi_aff multi2) const
{
  auto res = isl_multi_aff_product(copy(), multi2.release());
  return manage(res);
}

multi_aff multi_aff::project_domain_on_params() const
{
  auto res = isl_multi_aff_project_domain_on_params(copy());
  return manage(res);
}

multi_aff multi_aff::project_out_map(space space, isl::dim type, unsigned int first, unsigned int n)
{
  auto res = isl_multi_aff_project_out_map(space.release(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

multi_aff multi_aff::pullback(multi_aff ma2) const
{
  auto res = isl_multi_aff_pullback_multi_aff(copy(), ma2.release());
  return manage(res);
}

multi_aff multi_aff::range_factor_domain() const
{
  auto res = isl_multi_aff_range_factor_domain(copy());
  return manage(res);
}

multi_aff multi_aff::range_factor_range() const
{
  auto res = isl_multi_aff_range_factor_range(copy());
  return manage(res);
}

boolean multi_aff::range_is_wrapping() const
{
  auto res = isl_multi_aff_range_is_wrapping(get());
  return manage(res);
}

multi_aff multi_aff::range_map(space space)
{
  auto res = isl_multi_aff_range_map(space.release());
  return manage(res);
}

multi_aff multi_aff::range_product(multi_aff multi2) const
{
  auto res = isl_multi_aff_range_product(copy(), multi2.release());
  return manage(res);
}

multi_aff multi_aff::range_splice(unsigned int pos, multi_aff multi2) const
{
  auto res = isl_multi_aff_range_splice(copy(), pos, multi2.release());
  return manage(res);
}

multi_aff multi_aff::reset_tuple_id(isl::dim type) const
{
  auto res = isl_multi_aff_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

multi_aff multi_aff::reset_user() const
{
  auto res = isl_multi_aff_reset_user(copy());
  return manage(res);
}

multi_aff multi_aff::scale_down_multi_val(multi_val mv) const
{
  auto res = isl_multi_aff_scale_down_multi_val(copy(), mv.release());
  return manage(res);
}

multi_aff multi_aff::scale_down_val(val v) const
{
  auto res = isl_multi_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

multi_aff multi_aff::scale_multi_val(multi_val mv) const
{
  auto res = isl_multi_aff_scale_multi_val(copy(), mv.release());
  return manage(res);
}

multi_aff multi_aff::scale_val(val v) const
{
  auto res = isl_multi_aff_scale_val(copy(), v.release());
  return manage(res);
}

multi_aff multi_aff::set_aff(int pos, aff el) const
{
  auto res = isl_multi_aff_set_aff(copy(), pos, el.release());
  return manage(res);
}

multi_aff multi_aff::set_dim_id(isl::dim type, unsigned int pos, id id) const
{
  auto res = isl_multi_aff_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

multi_aff multi_aff::set_tuple_id(isl::dim type, id id) const
{
  auto res = isl_multi_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

multi_aff multi_aff::set_tuple_name(isl::dim type, const std::string &s) const
{
  auto res = isl_multi_aff_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

multi_aff multi_aff::splice(unsigned int in_pos, unsigned int out_pos, multi_aff multi2) const
{
  auto res = isl_multi_aff_splice(copy(), in_pos, out_pos, multi2.release());
  return manage(res);
}

multi_aff multi_aff::sub(multi_aff multi2) const
{
  auto res = isl_multi_aff_sub(copy(), multi2.release());
  return manage(res);
}

multi_aff multi_aff::zero(space space)
{
  auto res = isl_multi_aff_zero(space.release());
  return manage(res);
}

// implementations for isl::multi_pw_aff
multi_pw_aff manage(__isl_take isl_multi_pw_aff *ptr) {
  return multi_pw_aff(ptr);
}
multi_pw_aff manage_copy(__isl_keep isl_multi_pw_aff *ptr) {
  ptr = isl_multi_pw_aff_copy(ptr);
  return multi_pw_aff(ptr);
}

multi_pw_aff::multi_pw_aff()
    : ptr(nullptr) {}

multi_pw_aff::multi_pw_aff(const multi_pw_aff &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
multi_pw_aff::multi_pw_aff(std::nullptr_t)
    : ptr(nullptr) {}


multi_pw_aff::multi_pw_aff(__isl_take isl_multi_pw_aff *ptr)
    : ptr(ptr) {}

multi_pw_aff::multi_pw_aff(multi_aff ma)
{
  auto res = isl_multi_pw_aff_from_multi_aff(ma.release());
  ptr = res;
}
multi_pw_aff::multi_pw_aff(pw_aff pa)
{
  auto res = isl_multi_pw_aff_from_pw_aff(pa.release());
  ptr = res;
}
multi_pw_aff::multi_pw_aff(pw_multi_aff pma)
{
  auto res = isl_multi_pw_aff_from_pw_multi_aff(pma.release());
  ptr = res;
}
multi_pw_aff::multi_pw_aff(ctx ctx, const std::string &str)
{
  auto res = isl_multi_pw_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

multi_pw_aff &multi_pw_aff::operator=(multi_pw_aff obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

multi_pw_aff::~multi_pw_aff() {
  if (ptr)
    isl_multi_pw_aff_free(ptr);
}

__isl_give isl_multi_pw_aff *multi_pw_aff::copy() const & {
  return isl_multi_pw_aff_copy(ptr);
}

__isl_keep isl_multi_pw_aff *multi_pw_aff::get() const {
  return ptr;
}

__isl_give isl_multi_pw_aff *multi_pw_aff::release() {
  isl_multi_pw_aff *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool multi_pw_aff::is_null() const {
  return ptr == nullptr;
}
multi_pw_aff::operator bool() const {
  return !is_null();
}


ctx multi_pw_aff::get_ctx() const {
  return ctx(isl_multi_pw_aff_get_ctx(ptr));
}
std::string multi_pw_aff::to_str() const {
  char *Tmp = isl_multi_pw_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void multi_pw_aff::dump() const {
  isl_multi_pw_aff_dump(get());
}


multi_pw_aff multi_pw_aff::add(multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_add(copy(), multi2.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_multi_pw_aff_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

multi_pw_aff multi_pw_aff::align_params(space model) const
{
  auto res = isl_multi_pw_aff_align_params(copy(), model.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::coalesce() const
{
  auto res = isl_multi_pw_aff_coalesce(copy());
  return manage(res);
}

unsigned int multi_pw_aff::dim(isl::dim type) const
{
  auto res = isl_multi_pw_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

set multi_pw_aff::domain() const
{
  auto res = isl_multi_pw_aff_domain(copy());
  return manage(res);
}

multi_pw_aff multi_pw_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_multi_pw_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

map multi_pw_aff::eq_map(multi_pw_aff mpa2) const
{
  auto res = isl_multi_pw_aff_eq_map(copy(), mpa2.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::factor_range() const
{
  auto res = isl_multi_pw_aff_factor_range(copy());
  return manage(res);
}

int multi_pw_aff::find_dim_by_id(isl::dim type, const id &id) const
{
  auto res = isl_multi_pw_aff_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int multi_pw_aff::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_multi_pw_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

multi_pw_aff multi_pw_aff::flat_range_product(multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_flat_range_product(copy(), multi2.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::flatten_range() const
{
  auto res = isl_multi_pw_aff_flatten_range(copy());
  return manage(res);
}

multi_pw_aff multi_pw_aff::from_pw_aff_list(space space, pw_aff_list list)
{
  auto res = isl_multi_pw_aff_from_pw_aff_list(space.release(), list.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::from_range() const
{
  auto res = isl_multi_pw_aff_from_range(copy());
  return manage(res);
}

id multi_pw_aff::get_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_multi_pw_aff_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

space multi_pw_aff::get_domain_space() const
{
  auto res = isl_multi_pw_aff_get_domain_space(get());
  return manage(res);
}

uint32_t multi_pw_aff::get_hash() const
{
  auto res = isl_multi_pw_aff_get_hash(get());
  return res;
}

pw_aff multi_pw_aff::get_pw_aff(int pos) const
{
  auto res = isl_multi_pw_aff_get_pw_aff(get(), pos);
  return manage(res);
}

space multi_pw_aff::get_space() const
{
  auto res = isl_multi_pw_aff_get_space(get());
  return manage(res);
}

id multi_pw_aff::get_tuple_id(isl::dim type) const
{
  auto res = isl_multi_pw_aff_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

std::string multi_pw_aff::get_tuple_name(isl::dim type) const
{
  auto res = isl_multi_pw_aff_get_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  std::string tmp(res);
  return tmp;
}

multi_pw_aff multi_pw_aff::gist(set set) const
{
  auto res = isl_multi_pw_aff_gist(copy(), set.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::gist_params(set set) const
{
  auto res = isl_multi_pw_aff_gist_params(copy(), set.release());
  return manage(res);
}

boolean multi_pw_aff::has_tuple_id(isl::dim type) const
{
  auto res = isl_multi_pw_aff_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

multi_pw_aff multi_pw_aff::identity(space space)
{
  auto res = isl_multi_pw_aff_identity(space.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::insert_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_multi_pw_aff_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

multi_pw_aff multi_pw_aff::intersect_domain(set domain) const
{
  auto res = isl_multi_pw_aff_intersect_domain(copy(), domain.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::intersect_params(set set) const
{
  auto res = isl_multi_pw_aff_intersect_params(copy(), set.release());
  return manage(res);
}

boolean multi_pw_aff::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_multi_pw_aff_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean multi_pw_aff::involves_nan() const
{
  auto res = isl_multi_pw_aff_involves_nan(get());
  return manage(res);
}

boolean multi_pw_aff::is_cst() const
{
  auto res = isl_multi_pw_aff_is_cst(get());
  return manage(res);
}

boolean multi_pw_aff::is_equal(const multi_pw_aff &mpa2) const
{
  auto res = isl_multi_pw_aff_is_equal(get(), mpa2.get());
  return manage(res);
}

map multi_pw_aff::lex_gt_map(multi_pw_aff mpa2) const
{
  auto res = isl_multi_pw_aff_lex_gt_map(copy(), mpa2.release());
  return manage(res);
}

map multi_pw_aff::lex_lt_map(multi_pw_aff mpa2) const
{
  auto res = isl_multi_pw_aff_lex_lt_map(copy(), mpa2.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::mod_multi_val(multi_val mv) const
{
  auto res = isl_multi_pw_aff_mod_multi_val(copy(), mv.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_multi_pw_aff_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

multi_pw_aff multi_pw_aff::neg() const
{
  auto res = isl_multi_pw_aff_neg(copy());
  return manage(res);
}

boolean multi_pw_aff::plain_is_equal(const multi_pw_aff &multi2) const
{
  auto res = isl_multi_pw_aff_plain_is_equal(get(), multi2.get());
  return manage(res);
}

multi_pw_aff multi_pw_aff::product(multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_product(copy(), multi2.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::project_domain_on_params() const
{
  auto res = isl_multi_pw_aff_project_domain_on_params(copy());
  return manage(res);
}

multi_pw_aff multi_pw_aff::pullback(multi_aff ma) const
{
  auto res = isl_multi_pw_aff_pullback_multi_aff(copy(), ma.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::pullback(pw_multi_aff pma) const
{
  auto res = isl_multi_pw_aff_pullback_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::pullback(multi_pw_aff mpa2) const
{
  auto res = isl_multi_pw_aff_pullback_multi_pw_aff(copy(), mpa2.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::range_factor_domain() const
{
  auto res = isl_multi_pw_aff_range_factor_domain(copy());
  return manage(res);
}

multi_pw_aff multi_pw_aff::range_factor_range() const
{
  auto res = isl_multi_pw_aff_range_factor_range(copy());
  return manage(res);
}

boolean multi_pw_aff::range_is_wrapping() const
{
  auto res = isl_multi_pw_aff_range_is_wrapping(get());
  return manage(res);
}

multi_pw_aff multi_pw_aff::range_product(multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_range_product(copy(), multi2.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::range_splice(unsigned int pos, multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_range_splice(copy(), pos, multi2.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::reset_tuple_id(isl::dim type) const
{
  auto res = isl_multi_pw_aff_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

multi_pw_aff multi_pw_aff::reset_user() const
{
  auto res = isl_multi_pw_aff_reset_user(copy());
  return manage(res);
}

multi_pw_aff multi_pw_aff::scale_down_multi_val(multi_val mv) const
{
  auto res = isl_multi_pw_aff_scale_down_multi_val(copy(), mv.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::scale_down_val(val v) const
{
  auto res = isl_multi_pw_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::scale_multi_val(multi_val mv) const
{
  auto res = isl_multi_pw_aff_scale_multi_val(copy(), mv.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::scale_val(val v) const
{
  auto res = isl_multi_pw_aff_scale_val(copy(), v.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::set_dim_id(isl::dim type, unsigned int pos, id id) const
{
  auto res = isl_multi_pw_aff_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::set_pw_aff(int pos, pw_aff el) const
{
  auto res = isl_multi_pw_aff_set_pw_aff(copy(), pos, el.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::set_tuple_id(isl::dim type, id id) const
{
  auto res = isl_multi_pw_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::set_tuple_name(isl::dim type, const std::string &s) const
{
  auto res = isl_multi_pw_aff_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

multi_pw_aff multi_pw_aff::splice(unsigned int in_pos, unsigned int out_pos, multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_splice(copy(), in_pos, out_pos, multi2.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::sub(multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_sub(copy(), multi2.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::zero(space space)
{
  auto res = isl_multi_pw_aff_zero(space.release());
  return manage(res);
}

// implementations for isl::multi_union_pw_aff
multi_union_pw_aff manage(__isl_take isl_multi_union_pw_aff *ptr) {
  return multi_union_pw_aff(ptr);
}
multi_union_pw_aff manage_copy(__isl_keep isl_multi_union_pw_aff *ptr) {
  ptr = isl_multi_union_pw_aff_copy(ptr);
  return multi_union_pw_aff(ptr);
}

multi_union_pw_aff::multi_union_pw_aff()
    : ptr(nullptr) {}

multi_union_pw_aff::multi_union_pw_aff(const multi_union_pw_aff &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
multi_union_pw_aff::multi_union_pw_aff(std::nullptr_t)
    : ptr(nullptr) {}


multi_union_pw_aff::multi_union_pw_aff(__isl_take isl_multi_union_pw_aff *ptr)
    : ptr(ptr) {}

multi_union_pw_aff::multi_union_pw_aff(union_pw_aff upa)
{
  auto res = isl_multi_union_pw_aff_from_union_pw_aff(upa.release());
  ptr = res;
}
multi_union_pw_aff::multi_union_pw_aff(multi_pw_aff mpa)
{
  auto res = isl_multi_union_pw_aff_from_multi_pw_aff(mpa.release());
  ptr = res;
}
multi_union_pw_aff::multi_union_pw_aff(union_pw_multi_aff upma)
{
  auto res = isl_multi_union_pw_aff_from_union_pw_multi_aff(upma.release());
  ptr = res;
}
multi_union_pw_aff::multi_union_pw_aff(ctx ctx, const std::string &str)
{
  auto res = isl_multi_union_pw_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

multi_union_pw_aff &multi_union_pw_aff::operator=(multi_union_pw_aff obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

multi_union_pw_aff::~multi_union_pw_aff() {
  if (ptr)
    isl_multi_union_pw_aff_free(ptr);
}

__isl_give isl_multi_union_pw_aff *multi_union_pw_aff::copy() const & {
  return isl_multi_union_pw_aff_copy(ptr);
}

__isl_keep isl_multi_union_pw_aff *multi_union_pw_aff::get() const {
  return ptr;
}

__isl_give isl_multi_union_pw_aff *multi_union_pw_aff::release() {
  isl_multi_union_pw_aff *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool multi_union_pw_aff::is_null() const {
  return ptr == nullptr;
}
multi_union_pw_aff::operator bool() const {
  return !is_null();
}


ctx multi_union_pw_aff::get_ctx() const {
  return ctx(isl_multi_union_pw_aff_get_ctx(ptr));
}
std::string multi_union_pw_aff::to_str() const {
  char *Tmp = isl_multi_union_pw_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void multi_union_pw_aff::dump() const {
  isl_multi_union_pw_aff_dump(get());
}


multi_union_pw_aff multi_union_pw_aff::add(multi_union_pw_aff multi2) const
{
  auto res = isl_multi_union_pw_aff_add(copy(), multi2.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::align_params(space model) const
{
  auto res = isl_multi_union_pw_aff_align_params(copy(), model.release());
  return manage(res);
}

union_pw_aff multi_union_pw_aff::apply_aff(aff aff) const
{
  auto res = isl_multi_union_pw_aff_apply_aff(copy(), aff.release());
  return manage(res);
}

union_pw_aff multi_union_pw_aff::apply_pw_aff(pw_aff pa) const
{
  auto res = isl_multi_union_pw_aff_apply_pw_aff(copy(), pa.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::apply_pw_multi_aff(pw_multi_aff pma) const
{
  auto res = isl_multi_union_pw_aff_apply_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::coalesce() const
{
  auto res = isl_multi_union_pw_aff_coalesce(copy());
  return manage(res);
}

unsigned int multi_union_pw_aff::dim(isl::dim type) const
{
  auto res = isl_multi_union_pw_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

union_set multi_union_pw_aff::domain() const
{
  auto res = isl_multi_union_pw_aff_domain(copy());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_multi_union_pw_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

multi_pw_aff multi_union_pw_aff::extract_multi_pw_aff(space space) const
{
  auto res = isl_multi_union_pw_aff_extract_multi_pw_aff(get(), space.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::factor_range() const
{
  auto res = isl_multi_union_pw_aff_factor_range(copy());
  return manage(res);
}

int multi_union_pw_aff::find_dim_by_id(isl::dim type, const id &id) const
{
  auto res = isl_multi_union_pw_aff_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int multi_union_pw_aff::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_multi_union_pw_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

multi_union_pw_aff multi_union_pw_aff::flat_range_product(multi_union_pw_aff multi2) const
{
  auto res = isl_multi_union_pw_aff_flat_range_product(copy(), multi2.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::flatten_range() const
{
  auto res = isl_multi_union_pw_aff_flatten_range(copy());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::floor() const
{
  auto res = isl_multi_union_pw_aff_floor(copy());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::from_multi_aff(multi_aff ma)
{
  auto res = isl_multi_union_pw_aff_from_multi_aff(ma.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::from_range() const
{
  auto res = isl_multi_union_pw_aff_from_range(copy());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::from_union_map(union_map umap)
{
  auto res = isl_multi_union_pw_aff_from_union_map(umap.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::from_union_pw_aff_list(space space, union_pw_aff_list list)
{
  auto res = isl_multi_union_pw_aff_from_union_pw_aff_list(space.release(), list.release());
  return manage(res);
}

id multi_union_pw_aff::get_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_multi_union_pw_aff_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

space multi_union_pw_aff::get_domain_space() const
{
  auto res = isl_multi_union_pw_aff_get_domain_space(get());
  return manage(res);
}

space multi_union_pw_aff::get_space() const
{
  auto res = isl_multi_union_pw_aff_get_space(get());
  return manage(res);
}

id multi_union_pw_aff::get_tuple_id(isl::dim type) const
{
  auto res = isl_multi_union_pw_aff_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

std::string multi_union_pw_aff::get_tuple_name(isl::dim type) const
{
  auto res = isl_multi_union_pw_aff_get_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  std::string tmp(res);
  return tmp;
}

union_pw_aff multi_union_pw_aff::get_union_pw_aff(int pos) const
{
  auto res = isl_multi_union_pw_aff_get_union_pw_aff(get(), pos);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::gist(union_set context) const
{
  auto res = isl_multi_union_pw_aff_gist(copy(), context.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::gist_params(set context) const
{
  auto res = isl_multi_union_pw_aff_gist_params(copy(), context.release());
  return manage(res);
}

boolean multi_union_pw_aff::has_tuple_id(isl::dim type) const
{
  auto res = isl_multi_union_pw_aff_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::intersect_domain(union_set uset) const
{
  auto res = isl_multi_union_pw_aff_intersect_domain(copy(), uset.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::intersect_params(set params) const
{
  auto res = isl_multi_union_pw_aff_intersect_params(copy(), params.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::intersect_range(set set) const
{
  auto res = isl_multi_union_pw_aff_intersect_range(copy(), set.release());
  return manage(res);
}

boolean multi_union_pw_aff::involves_nan() const
{
  auto res = isl_multi_union_pw_aff_involves_nan(get());
  return manage(res);
}

multi_val multi_union_pw_aff::max_multi_val() const
{
  auto res = isl_multi_union_pw_aff_max_multi_val(copy());
  return manage(res);
}

multi_val multi_union_pw_aff::min_multi_val() const
{
  auto res = isl_multi_union_pw_aff_min_multi_val(copy());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::mod_multi_val(multi_val mv) const
{
  auto res = isl_multi_union_pw_aff_mod_multi_val(copy(), mv.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::multi_aff_on_domain(union_set domain, multi_aff ma)
{
  auto res = isl_multi_union_pw_aff_multi_aff_on_domain(domain.release(), ma.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::multi_val_on_domain(union_set domain, multi_val mv)
{
  auto res = isl_multi_union_pw_aff_multi_val_on_domain(domain.release(), mv.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::neg() const
{
  auto res = isl_multi_union_pw_aff_neg(copy());
  return manage(res);
}

boolean multi_union_pw_aff::plain_is_equal(const multi_union_pw_aff &multi2) const
{
  auto res = isl_multi_union_pw_aff_plain_is_equal(get(), multi2.get());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::pullback(union_pw_multi_aff upma) const
{
  auto res = isl_multi_union_pw_aff_pullback_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::pw_multi_aff_on_domain(union_set domain, pw_multi_aff pma)
{
  auto res = isl_multi_union_pw_aff_pw_multi_aff_on_domain(domain.release(), pma.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::range_factor_domain() const
{
  auto res = isl_multi_union_pw_aff_range_factor_domain(copy());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::range_factor_range() const
{
  auto res = isl_multi_union_pw_aff_range_factor_range(copy());
  return manage(res);
}

boolean multi_union_pw_aff::range_is_wrapping() const
{
  auto res = isl_multi_union_pw_aff_range_is_wrapping(get());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::range_product(multi_union_pw_aff multi2) const
{
  auto res = isl_multi_union_pw_aff_range_product(copy(), multi2.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::range_splice(unsigned int pos, multi_union_pw_aff multi2) const
{
  auto res = isl_multi_union_pw_aff_range_splice(copy(), pos, multi2.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::reset_tuple_id(isl::dim type) const
{
  auto res = isl_multi_union_pw_aff_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::reset_user() const
{
  auto res = isl_multi_union_pw_aff_reset_user(copy());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::scale_down_multi_val(multi_val mv) const
{
  auto res = isl_multi_union_pw_aff_scale_down_multi_val(copy(), mv.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::scale_down_val(val v) const
{
  auto res = isl_multi_union_pw_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::scale_multi_val(multi_val mv) const
{
  auto res = isl_multi_union_pw_aff_scale_multi_val(copy(), mv.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::scale_val(val v) const
{
  auto res = isl_multi_union_pw_aff_scale_val(copy(), v.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::set_dim_id(isl::dim type, unsigned int pos, id id) const
{
  auto res = isl_multi_union_pw_aff_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::set_tuple_id(isl::dim type, id id) const
{
  auto res = isl_multi_union_pw_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::set_tuple_name(isl::dim type, const std::string &s) const
{
  auto res = isl_multi_union_pw_aff_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::set_union_pw_aff(int pos, union_pw_aff el) const
{
  auto res = isl_multi_union_pw_aff_set_union_pw_aff(copy(), pos, el.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::sub(multi_union_pw_aff multi2) const
{
  auto res = isl_multi_union_pw_aff_sub(copy(), multi2.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::union_add(multi_union_pw_aff mupa2) const
{
  auto res = isl_multi_union_pw_aff_union_add(copy(), mupa2.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::zero(space space)
{
  auto res = isl_multi_union_pw_aff_zero(space.release());
  return manage(res);
}

union_set multi_union_pw_aff::zero_union_set() const
{
  auto res = isl_multi_union_pw_aff_zero_union_set(copy());
  return manage(res);
}

// implementations for isl::multi_val
multi_val manage(__isl_take isl_multi_val *ptr) {
  return multi_val(ptr);
}
multi_val manage_copy(__isl_keep isl_multi_val *ptr) {
  ptr = isl_multi_val_copy(ptr);
  return multi_val(ptr);
}

multi_val::multi_val()
    : ptr(nullptr) {}

multi_val::multi_val(const multi_val &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
multi_val::multi_val(std::nullptr_t)
    : ptr(nullptr) {}


multi_val::multi_val(__isl_take isl_multi_val *ptr)
    : ptr(ptr) {}


multi_val &multi_val::operator=(multi_val obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

multi_val::~multi_val() {
  if (ptr)
    isl_multi_val_free(ptr);
}

__isl_give isl_multi_val *multi_val::copy() const & {
  return isl_multi_val_copy(ptr);
}

__isl_keep isl_multi_val *multi_val::get() const {
  return ptr;
}

__isl_give isl_multi_val *multi_val::release() {
  isl_multi_val *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool multi_val::is_null() const {
  return ptr == nullptr;
}
multi_val::operator bool() const {
  return !is_null();
}


ctx multi_val::get_ctx() const {
  return ctx(isl_multi_val_get_ctx(ptr));
}
std::string multi_val::to_str() const {
  char *Tmp = isl_multi_val_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void multi_val::dump() const {
  isl_multi_val_dump(get());
}


multi_val multi_val::add(multi_val multi2) const
{
  auto res = isl_multi_val_add(copy(), multi2.release());
  return manage(res);
}

multi_val multi_val::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_multi_val_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

multi_val multi_val::add_val(val v) const
{
  auto res = isl_multi_val_add_val(copy(), v.release());
  return manage(res);
}

multi_val multi_val::align_params(space model) const
{
  auto res = isl_multi_val_align_params(copy(), model.release());
  return manage(res);
}

unsigned int multi_val::dim(isl::dim type) const
{
  auto res = isl_multi_val_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

multi_val multi_val::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_multi_val_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

multi_val multi_val::factor_range() const
{
  auto res = isl_multi_val_factor_range(copy());
  return manage(res);
}

int multi_val::find_dim_by_id(isl::dim type, const id &id) const
{
  auto res = isl_multi_val_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int multi_val::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_multi_val_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

multi_val multi_val::flat_range_product(multi_val multi2) const
{
  auto res = isl_multi_val_flat_range_product(copy(), multi2.release());
  return manage(res);
}

multi_val multi_val::flatten_range() const
{
  auto res = isl_multi_val_flatten_range(copy());
  return manage(res);
}

multi_val multi_val::from_range() const
{
  auto res = isl_multi_val_from_range(copy());
  return manage(res);
}

multi_val multi_val::from_val_list(space space, val_list list)
{
  auto res = isl_multi_val_from_val_list(space.release(), list.release());
  return manage(res);
}

id multi_val::get_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_multi_val_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

space multi_val::get_domain_space() const
{
  auto res = isl_multi_val_get_domain_space(get());
  return manage(res);
}

space multi_val::get_space() const
{
  auto res = isl_multi_val_get_space(get());
  return manage(res);
}

id multi_val::get_tuple_id(isl::dim type) const
{
  auto res = isl_multi_val_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

std::string multi_val::get_tuple_name(isl::dim type) const
{
  auto res = isl_multi_val_get_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  std::string tmp(res);
  return tmp;
}

val multi_val::get_val(int pos) const
{
  auto res = isl_multi_val_get_val(get(), pos);
  return manage(res);
}

boolean multi_val::has_tuple_id(isl::dim type) const
{
  auto res = isl_multi_val_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

multi_val multi_val::insert_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_multi_val_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean multi_val::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_multi_val_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean multi_val::involves_nan() const
{
  auto res = isl_multi_val_involves_nan(get());
  return manage(res);
}

multi_val multi_val::mod_multi_val(multi_val mv) const
{
  auto res = isl_multi_val_mod_multi_val(copy(), mv.release());
  return manage(res);
}

multi_val multi_val::mod_val(val v) const
{
  auto res = isl_multi_val_mod_val(copy(), v.release());
  return manage(res);
}

multi_val multi_val::neg() const
{
  auto res = isl_multi_val_neg(copy());
  return manage(res);
}

boolean multi_val::plain_is_equal(const multi_val &multi2) const
{
  auto res = isl_multi_val_plain_is_equal(get(), multi2.get());
  return manage(res);
}

multi_val multi_val::product(multi_val multi2) const
{
  auto res = isl_multi_val_product(copy(), multi2.release());
  return manage(res);
}

multi_val multi_val::project_domain_on_params() const
{
  auto res = isl_multi_val_project_domain_on_params(copy());
  return manage(res);
}

multi_val multi_val::range_factor_domain() const
{
  auto res = isl_multi_val_range_factor_domain(copy());
  return manage(res);
}

multi_val multi_val::range_factor_range() const
{
  auto res = isl_multi_val_range_factor_range(copy());
  return manage(res);
}

boolean multi_val::range_is_wrapping() const
{
  auto res = isl_multi_val_range_is_wrapping(get());
  return manage(res);
}

multi_val multi_val::range_product(multi_val multi2) const
{
  auto res = isl_multi_val_range_product(copy(), multi2.release());
  return manage(res);
}

multi_val multi_val::range_splice(unsigned int pos, multi_val multi2) const
{
  auto res = isl_multi_val_range_splice(copy(), pos, multi2.release());
  return manage(res);
}

multi_val multi_val::read_from_str(ctx ctx, const std::string &str)
{
  auto res = isl_multi_val_read_from_str(ctx.release(), str.c_str());
  return manage(res);
}

multi_val multi_val::reset_tuple_id(isl::dim type) const
{
  auto res = isl_multi_val_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

multi_val multi_val::reset_user() const
{
  auto res = isl_multi_val_reset_user(copy());
  return manage(res);
}

multi_val multi_val::scale_down_multi_val(multi_val mv) const
{
  auto res = isl_multi_val_scale_down_multi_val(copy(), mv.release());
  return manage(res);
}

multi_val multi_val::scale_down_val(val v) const
{
  auto res = isl_multi_val_scale_down_val(copy(), v.release());
  return manage(res);
}

multi_val multi_val::scale_multi_val(multi_val mv) const
{
  auto res = isl_multi_val_scale_multi_val(copy(), mv.release());
  return manage(res);
}

multi_val multi_val::scale_val(val v) const
{
  auto res = isl_multi_val_scale_val(copy(), v.release());
  return manage(res);
}

multi_val multi_val::set_dim_id(isl::dim type, unsigned int pos, id id) const
{
  auto res = isl_multi_val_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

multi_val multi_val::set_tuple_id(isl::dim type, id id) const
{
  auto res = isl_multi_val_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

multi_val multi_val::set_tuple_name(isl::dim type, const std::string &s) const
{
  auto res = isl_multi_val_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

multi_val multi_val::set_val(int pos, val el) const
{
  auto res = isl_multi_val_set_val(copy(), pos, el.release());
  return manage(res);
}

multi_val multi_val::splice(unsigned int in_pos, unsigned int out_pos, multi_val multi2) const
{
  auto res = isl_multi_val_splice(copy(), in_pos, out_pos, multi2.release());
  return manage(res);
}

multi_val multi_val::sub(multi_val multi2) const
{
  auto res = isl_multi_val_sub(copy(), multi2.release());
  return manage(res);
}

multi_val multi_val::zero(space space)
{
  auto res = isl_multi_val_zero(space.release());
  return manage(res);
}

// implementations for isl::point
point manage(__isl_take isl_point *ptr) {
  return point(ptr);
}
point manage_copy(__isl_keep isl_point *ptr) {
  ptr = isl_point_copy(ptr);
  return point(ptr);
}

point::point()
    : ptr(nullptr) {}

point::point(const point &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
point::point(std::nullptr_t)
    : ptr(nullptr) {}


point::point(__isl_take isl_point *ptr)
    : ptr(ptr) {}

point::point(space dim)
{
  auto res = isl_point_zero(dim.release());
  ptr = res;
}

point &point::operator=(point obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

point::~point() {
  if (ptr)
    isl_point_free(ptr);
}

__isl_give isl_point *point::copy() const & {
  return isl_point_copy(ptr);
}

__isl_keep isl_point *point::get() const {
  return ptr;
}

__isl_give isl_point *point::release() {
  isl_point *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool point::is_null() const {
  return ptr == nullptr;
}
point::operator bool() const {
  return !is_null();
}


ctx point::get_ctx() const {
  return ctx(isl_point_get_ctx(ptr));
}
std::string point::to_str() const {
  char *Tmp = isl_point_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void point::dump() const {
  isl_point_dump(get());
}


point point::add_ui(isl::dim type, int pos, unsigned int val) const
{
  auto res = isl_point_add_ui(copy(), static_cast<enum isl_dim_type>(type), pos, val);
  return manage(res);
}

val point::get_coordinate_val(isl::dim type, int pos) const
{
  auto res = isl_point_get_coordinate_val(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

space point::get_space() const
{
  auto res = isl_point_get_space(get());
  return manage(res);
}

point point::set_coordinate_val(isl::dim type, int pos, val v) const
{
  auto res = isl_point_set_coordinate_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

point point::sub_ui(isl::dim type, int pos, unsigned int val) const
{
  auto res = isl_point_sub_ui(copy(), static_cast<enum isl_dim_type>(type), pos, val);
  return manage(res);
}

// implementations for isl::pw_aff
pw_aff manage(__isl_take isl_pw_aff *ptr) {
  return pw_aff(ptr);
}
pw_aff manage_copy(__isl_keep isl_pw_aff *ptr) {
  ptr = isl_pw_aff_copy(ptr);
  return pw_aff(ptr);
}

pw_aff::pw_aff()
    : ptr(nullptr) {}

pw_aff::pw_aff(const pw_aff &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
pw_aff::pw_aff(std::nullptr_t)
    : ptr(nullptr) {}


pw_aff::pw_aff(__isl_take isl_pw_aff *ptr)
    : ptr(ptr) {}

pw_aff::pw_aff(aff aff)
{
  auto res = isl_pw_aff_from_aff(aff.release());
  ptr = res;
}
pw_aff::pw_aff(local_space ls)
{
  auto res = isl_pw_aff_zero_on_domain(ls.release());
  ptr = res;
}
pw_aff::pw_aff(set domain, val v)
{
  auto res = isl_pw_aff_val_on_domain(domain.release(), v.release());
  ptr = res;
}
pw_aff::pw_aff(ctx ctx, const std::string &str)
{
  auto res = isl_pw_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

pw_aff &pw_aff::operator=(pw_aff obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

pw_aff::~pw_aff() {
  if (ptr)
    isl_pw_aff_free(ptr);
}

__isl_give isl_pw_aff *pw_aff::copy() const & {
  return isl_pw_aff_copy(ptr);
}

__isl_keep isl_pw_aff *pw_aff::get() const {
  return ptr;
}

__isl_give isl_pw_aff *pw_aff::release() {
  isl_pw_aff *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool pw_aff::is_null() const {
  return ptr == nullptr;
}
pw_aff::operator bool() const {
  return !is_null();
}


ctx pw_aff::get_ctx() const {
  return ctx(isl_pw_aff_get_ctx(ptr));
}
std::string pw_aff::to_str() const {
  char *Tmp = isl_pw_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void pw_aff::dump() const {
  isl_pw_aff_dump(get());
}


pw_aff pw_aff::add(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_add(copy(), pwaff2.release());
  return manage(res);
}

pw_aff pw_aff::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_pw_aff_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

pw_aff pw_aff::align_params(space model) const
{
  auto res = isl_pw_aff_align_params(copy(), model.release());
  return manage(res);
}

pw_aff pw_aff::alloc(set set, aff aff)
{
  auto res = isl_pw_aff_alloc(set.release(), aff.release());
  return manage(res);
}

pw_aff pw_aff::ceil() const
{
  auto res = isl_pw_aff_ceil(copy());
  return manage(res);
}

pw_aff pw_aff::coalesce() const
{
  auto res = isl_pw_aff_coalesce(copy());
  return manage(res);
}

pw_aff pw_aff::cond(pw_aff pwaff_true, pw_aff pwaff_false) const
{
  auto res = isl_pw_aff_cond(copy(), pwaff_true.release(), pwaff_false.release());
  return manage(res);
}

unsigned int pw_aff::dim(isl::dim type) const
{
  auto res = isl_pw_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

pw_aff pw_aff::div(pw_aff pa2) const
{
  auto res = isl_pw_aff_div(copy(), pa2.release());
  return manage(res);
}

set pw_aff::domain() const
{
  auto res = isl_pw_aff_domain(copy());
  return manage(res);
}

pw_aff pw_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_pw_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

pw_aff pw_aff::drop_unused_params() const
{
  auto res = isl_pw_aff_drop_unused_params(copy());
  return manage(res);
}

pw_aff pw_aff::empty(space dim)
{
  auto res = isl_pw_aff_empty(dim.release());
  return manage(res);
}

map pw_aff::eq_map(pw_aff pa2) const
{
  auto res = isl_pw_aff_eq_map(copy(), pa2.release());
  return manage(res);
}

set pw_aff::eq_set(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_eq_set(copy(), pwaff2.release());
  return manage(res);
}

val pw_aff::eval(point pnt) const
{
  auto res = isl_pw_aff_eval(copy(), pnt.release());
  return manage(res);
}

int pw_aff::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_pw_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

pw_aff pw_aff::floor() const
{
  auto res = isl_pw_aff_floor(copy());
  return manage(res);
}

stat pw_aff::foreach_piece(const std::function<stat(set, aff)> &fn) const
{
  struct fn_data {
    const std::function<stat(set, aff)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_set *arg_0, isl_aff *arg_1, void *arg_2) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_2);
    stat ret = (*data->func)(manage(arg_0), manage(arg_1));
    return ret.release();
  };
  auto res = isl_pw_aff_foreach_piece(get(), fn_lambda, &fn_data);
  return manage(res);
}

pw_aff pw_aff::from_range() const
{
  auto res = isl_pw_aff_from_range(copy());
  return manage(res);
}

set pw_aff::ge_set(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_ge_set(copy(), pwaff2.release());
  return manage(res);
}

id pw_aff::get_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_pw_aff_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

std::string pw_aff::get_dim_name(isl::dim type, unsigned int pos) const
{
  auto res = isl_pw_aff_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

space pw_aff::get_domain_space() const
{
  auto res = isl_pw_aff_get_domain_space(get());
  return manage(res);
}

uint32_t pw_aff::get_hash() const
{
  auto res = isl_pw_aff_get_hash(get());
  return res;
}

space pw_aff::get_space() const
{
  auto res = isl_pw_aff_get_space(get());
  return manage(res);
}

id pw_aff::get_tuple_id(isl::dim type) const
{
  auto res = isl_pw_aff_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

pw_aff pw_aff::gist(set context) const
{
  auto res = isl_pw_aff_gist(copy(), context.release());
  return manage(res);
}

pw_aff pw_aff::gist_params(set context) const
{
  auto res = isl_pw_aff_gist_params(copy(), context.release());
  return manage(res);
}

map pw_aff::gt_map(pw_aff pa2) const
{
  auto res = isl_pw_aff_gt_map(copy(), pa2.release());
  return manage(res);
}

set pw_aff::gt_set(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_gt_set(copy(), pwaff2.release());
  return manage(res);
}

boolean pw_aff::has_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_pw_aff_has_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

boolean pw_aff::has_tuple_id(isl::dim type) const
{
  auto res = isl_pw_aff_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

pw_aff pw_aff::insert_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_pw_aff_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

pw_aff pw_aff::intersect_domain(set set) const
{
  auto res = isl_pw_aff_intersect_domain(copy(), set.release());
  return manage(res);
}

pw_aff pw_aff::intersect_params(set set) const
{
  auto res = isl_pw_aff_intersect_params(copy(), set.release());
  return manage(res);
}

boolean pw_aff::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_pw_aff_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean pw_aff::involves_nan() const
{
  auto res = isl_pw_aff_involves_nan(get());
  return manage(res);
}

boolean pw_aff::is_cst() const
{
  auto res = isl_pw_aff_is_cst(get());
  return manage(res);
}

boolean pw_aff::is_empty() const
{
  auto res = isl_pw_aff_is_empty(get());
  return manage(res);
}

boolean pw_aff::is_equal(const pw_aff &pa2) const
{
  auto res = isl_pw_aff_is_equal(get(), pa2.get());
  return manage(res);
}

set pw_aff::le_set(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_le_set(copy(), pwaff2.release());
  return manage(res);
}

map pw_aff::lt_map(pw_aff pa2) const
{
  auto res = isl_pw_aff_lt_map(copy(), pa2.release());
  return manage(res);
}

set pw_aff::lt_set(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_lt_set(copy(), pwaff2.release());
  return manage(res);
}

pw_aff pw_aff::max(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_max(copy(), pwaff2.release());
  return manage(res);
}

pw_aff pw_aff::min(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_min(copy(), pwaff2.release());
  return manage(res);
}

pw_aff pw_aff::mod(val mod) const
{
  auto res = isl_pw_aff_mod_val(copy(), mod.release());
  return manage(res);
}

pw_aff pw_aff::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_pw_aff_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

pw_aff pw_aff::mul(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_mul(copy(), pwaff2.release());
  return manage(res);
}

int pw_aff::n_piece() const
{
  auto res = isl_pw_aff_n_piece(get());
  return res;
}

pw_aff pw_aff::nan_on_domain(local_space ls)
{
  auto res = isl_pw_aff_nan_on_domain(ls.release());
  return manage(res);
}

set pw_aff::ne_set(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_ne_set(copy(), pwaff2.release());
  return manage(res);
}

pw_aff pw_aff::neg() const
{
  auto res = isl_pw_aff_neg(copy());
  return manage(res);
}

set pw_aff::non_zero_set() const
{
  auto res = isl_pw_aff_non_zero_set(copy());
  return manage(res);
}

set pw_aff::nonneg_set() const
{
  auto res = isl_pw_aff_nonneg_set(copy());
  return manage(res);
}

set pw_aff::params() const
{
  auto res = isl_pw_aff_params(copy());
  return manage(res);
}

int pw_aff::plain_cmp(const pw_aff &pa2) const
{
  auto res = isl_pw_aff_plain_cmp(get(), pa2.get());
  return res;
}

boolean pw_aff::plain_is_equal(const pw_aff &pwaff2) const
{
  auto res = isl_pw_aff_plain_is_equal(get(), pwaff2.get());
  return manage(res);
}

set pw_aff::pos_set() const
{
  auto res = isl_pw_aff_pos_set(copy());
  return manage(res);
}

pw_aff pw_aff::project_domain_on_params() const
{
  auto res = isl_pw_aff_project_domain_on_params(copy());
  return manage(res);
}

pw_aff pw_aff::pullback(multi_aff ma) const
{
  auto res = isl_pw_aff_pullback_multi_aff(copy(), ma.release());
  return manage(res);
}

pw_aff pw_aff::pullback(pw_multi_aff pma) const
{
  auto res = isl_pw_aff_pullback_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

pw_aff pw_aff::pullback(multi_pw_aff mpa) const
{
  auto res = isl_pw_aff_pullback_multi_pw_aff(copy(), mpa.release());
  return manage(res);
}

pw_aff pw_aff::reset_tuple_id(isl::dim type) const
{
  auto res = isl_pw_aff_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

pw_aff pw_aff::reset_user() const
{
  auto res = isl_pw_aff_reset_user(copy());
  return manage(res);
}

pw_aff pw_aff::scale(val v) const
{
  auto res = isl_pw_aff_scale_val(copy(), v.release());
  return manage(res);
}

pw_aff pw_aff::scale_down(val f) const
{
  auto res = isl_pw_aff_scale_down_val(copy(), f.release());
  return manage(res);
}

pw_aff pw_aff::set_dim_id(isl::dim type, unsigned int pos, id id) const
{
  auto res = isl_pw_aff_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

pw_aff pw_aff::set_tuple_id(isl::dim type, id id) const
{
  auto res = isl_pw_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

pw_aff pw_aff::sub(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_sub(copy(), pwaff2.release());
  return manage(res);
}

pw_aff pw_aff::subtract_domain(set set) const
{
  auto res = isl_pw_aff_subtract_domain(copy(), set.release());
  return manage(res);
}

pw_aff pw_aff::tdiv_q(pw_aff pa2) const
{
  auto res = isl_pw_aff_tdiv_q(copy(), pa2.release());
  return manage(res);
}

pw_aff pw_aff::tdiv_r(pw_aff pa2) const
{
  auto res = isl_pw_aff_tdiv_r(copy(), pa2.release());
  return manage(res);
}

pw_aff pw_aff::union_add(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_union_add(copy(), pwaff2.release());
  return manage(res);
}

pw_aff pw_aff::union_max(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_union_max(copy(), pwaff2.release());
  return manage(res);
}

pw_aff pw_aff::union_min(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_union_min(copy(), pwaff2.release());
  return manage(res);
}

pw_aff pw_aff::var_on_domain(local_space ls, isl::dim type, unsigned int pos)
{
  auto res = isl_pw_aff_var_on_domain(ls.release(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

set pw_aff::zero_set() const
{
  auto res = isl_pw_aff_zero_set(copy());
  return manage(res);
}

// implementations for isl::pw_aff_list
pw_aff_list manage(__isl_take isl_pw_aff_list *ptr) {
  return pw_aff_list(ptr);
}
pw_aff_list manage_copy(__isl_keep isl_pw_aff_list *ptr) {
  ptr = isl_pw_aff_list_copy(ptr);
  return pw_aff_list(ptr);
}

pw_aff_list::pw_aff_list()
    : ptr(nullptr) {}

pw_aff_list::pw_aff_list(const pw_aff_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
pw_aff_list::pw_aff_list(std::nullptr_t)
    : ptr(nullptr) {}


pw_aff_list::pw_aff_list(__isl_take isl_pw_aff_list *ptr)
    : ptr(ptr) {}


pw_aff_list &pw_aff_list::operator=(pw_aff_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

pw_aff_list::~pw_aff_list() {
  if (ptr)
    isl_pw_aff_list_free(ptr);
}

__isl_give isl_pw_aff_list *pw_aff_list::copy() const & {
  return isl_pw_aff_list_copy(ptr);
}

__isl_keep isl_pw_aff_list *pw_aff_list::get() const {
  return ptr;
}

__isl_give isl_pw_aff_list *pw_aff_list::release() {
  isl_pw_aff_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool pw_aff_list::is_null() const {
  return ptr == nullptr;
}
pw_aff_list::operator bool() const {
  return !is_null();
}


ctx pw_aff_list::get_ctx() const {
  return ctx(isl_pw_aff_list_get_ctx(ptr));
}

void pw_aff_list::dump() const {
  isl_pw_aff_list_dump(get());
}


pw_aff_list pw_aff_list::add(pw_aff el) const
{
  auto res = isl_pw_aff_list_add(copy(), el.release());
  return manage(res);
}

pw_aff_list pw_aff_list::alloc(ctx ctx, int n)
{
  auto res = isl_pw_aff_list_alloc(ctx.release(), n);
  return manage(res);
}

pw_aff_list pw_aff_list::concat(pw_aff_list list2) const
{
  auto res = isl_pw_aff_list_concat(copy(), list2.release());
  return manage(res);
}

pw_aff_list pw_aff_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_pw_aff_list_drop(copy(), first, n);
  return manage(res);
}

set pw_aff_list::eq_set(pw_aff_list list2) const
{
  auto res = isl_pw_aff_list_eq_set(copy(), list2.release());
  return manage(res);
}

stat pw_aff_list::foreach(const std::function<stat(pw_aff)> &fn) const
{
  struct fn_data {
    const std::function<stat(pw_aff)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_pw_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_pw_aff_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

pw_aff_list pw_aff_list::from_pw_aff(pw_aff el)
{
  auto res = isl_pw_aff_list_from_pw_aff(el.release());
  return manage(res);
}

set pw_aff_list::ge_set(pw_aff_list list2) const
{
  auto res = isl_pw_aff_list_ge_set(copy(), list2.release());
  return manage(res);
}

pw_aff pw_aff_list::get_at(int index) const
{
  auto res = isl_pw_aff_list_get_at(get(), index);
  return manage(res);
}

pw_aff pw_aff_list::get_pw_aff(int index) const
{
  auto res = isl_pw_aff_list_get_pw_aff(get(), index);
  return manage(res);
}

set pw_aff_list::gt_set(pw_aff_list list2) const
{
  auto res = isl_pw_aff_list_gt_set(copy(), list2.release());
  return manage(res);
}

pw_aff_list pw_aff_list::insert(unsigned int pos, pw_aff el) const
{
  auto res = isl_pw_aff_list_insert(copy(), pos, el.release());
  return manage(res);
}

set pw_aff_list::le_set(pw_aff_list list2) const
{
  auto res = isl_pw_aff_list_le_set(copy(), list2.release());
  return manage(res);
}

set pw_aff_list::lt_set(pw_aff_list list2) const
{
  auto res = isl_pw_aff_list_lt_set(copy(), list2.release());
  return manage(res);
}

pw_aff pw_aff_list::max() const
{
  auto res = isl_pw_aff_list_max(copy());
  return manage(res);
}

pw_aff pw_aff_list::min() const
{
  auto res = isl_pw_aff_list_min(copy());
  return manage(res);
}

int pw_aff_list::n_pw_aff() const
{
  auto res = isl_pw_aff_list_n_pw_aff(get());
  return res;
}

set pw_aff_list::ne_set(pw_aff_list list2) const
{
  auto res = isl_pw_aff_list_ne_set(copy(), list2.release());
  return manage(res);
}

pw_aff_list pw_aff_list::reverse() const
{
  auto res = isl_pw_aff_list_reverse(copy());
  return manage(res);
}

pw_aff_list pw_aff_list::set_pw_aff(int index, pw_aff el) const
{
  auto res = isl_pw_aff_list_set_pw_aff(copy(), index, el.release());
  return manage(res);
}

int pw_aff_list::size() const
{
  auto res = isl_pw_aff_list_size(get());
  return res;
}

pw_aff_list pw_aff_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_pw_aff_list_swap(copy(), pos1, pos2);
  return manage(res);
}

// implementations for isl::pw_multi_aff
pw_multi_aff manage(__isl_take isl_pw_multi_aff *ptr) {
  return pw_multi_aff(ptr);
}
pw_multi_aff manage_copy(__isl_keep isl_pw_multi_aff *ptr) {
  ptr = isl_pw_multi_aff_copy(ptr);
  return pw_multi_aff(ptr);
}

pw_multi_aff::pw_multi_aff()
    : ptr(nullptr) {}

pw_multi_aff::pw_multi_aff(const pw_multi_aff &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
pw_multi_aff::pw_multi_aff(std::nullptr_t)
    : ptr(nullptr) {}


pw_multi_aff::pw_multi_aff(__isl_take isl_pw_multi_aff *ptr)
    : ptr(ptr) {}

pw_multi_aff::pw_multi_aff(multi_aff ma)
{
  auto res = isl_pw_multi_aff_from_multi_aff(ma.release());
  ptr = res;
}
pw_multi_aff::pw_multi_aff(pw_aff pa)
{
  auto res = isl_pw_multi_aff_from_pw_aff(pa.release());
  ptr = res;
}
pw_multi_aff::pw_multi_aff(ctx ctx, const std::string &str)
{
  auto res = isl_pw_multi_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

pw_multi_aff &pw_multi_aff::operator=(pw_multi_aff obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

pw_multi_aff::~pw_multi_aff() {
  if (ptr)
    isl_pw_multi_aff_free(ptr);
}

__isl_give isl_pw_multi_aff *pw_multi_aff::copy() const & {
  return isl_pw_multi_aff_copy(ptr);
}

__isl_keep isl_pw_multi_aff *pw_multi_aff::get() const {
  return ptr;
}

__isl_give isl_pw_multi_aff *pw_multi_aff::release() {
  isl_pw_multi_aff *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool pw_multi_aff::is_null() const {
  return ptr == nullptr;
}
pw_multi_aff::operator bool() const {
  return !is_null();
}


ctx pw_multi_aff::get_ctx() const {
  return ctx(isl_pw_multi_aff_get_ctx(ptr));
}
std::string pw_multi_aff::to_str() const {
  char *Tmp = isl_pw_multi_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void pw_multi_aff::dump() const {
  isl_pw_multi_aff_dump(get());
}


pw_multi_aff pw_multi_aff::add(pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_add(copy(), pma2.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::align_params(space model) const
{
  auto res = isl_pw_multi_aff_align_params(copy(), model.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::alloc(set set, multi_aff maff)
{
  auto res = isl_pw_multi_aff_alloc(set.release(), maff.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::coalesce() const
{
  auto res = isl_pw_multi_aff_coalesce(copy());
  return manage(res);
}

unsigned int pw_multi_aff::dim(isl::dim type) const
{
  auto res = isl_pw_multi_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

set pw_multi_aff::domain() const
{
  auto res = isl_pw_multi_aff_domain(copy());
  return manage(res);
}

pw_multi_aff pw_multi_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_pw_multi_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

pw_multi_aff pw_multi_aff::drop_unused_params() const
{
  auto res = isl_pw_multi_aff_drop_unused_params(copy());
  return manage(res);
}

pw_multi_aff pw_multi_aff::empty(space space)
{
  auto res = isl_pw_multi_aff_empty(space.release());
  return manage(res);
}

int pw_multi_aff::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_pw_multi_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

pw_multi_aff pw_multi_aff::fix_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_pw_multi_aff_fix_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

pw_multi_aff pw_multi_aff::flat_range_product(pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_flat_range_product(copy(), pma2.release());
  return manage(res);
}

stat pw_multi_aff::foreach_piece(const std::function<stat(set, multi_aff)> &fn) const
{
  struct fn_data {
    const std::function<stat(set, multi_aff)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_set *arg_0, isl_multi_aff *arg_1, void *arg_2) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_2);
    stat ret = (*data->func)(manage(arg_0), manage(arg_1));
    return ret.release();
  };
  auto res = isl_pw_multi_aff_foreach_piece(get(), fn_lambda, &fn_data);
  return manage(res);
}

pw_multi_aff pw_multi_aff::from_domain(set set)
{
  auto res = isl_pw_multi_aff_from_domain(set.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::from_map(map map)
{
  auto res = isl_pw_multi_aff_from_map(map.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::from_multi_pw_aff(multi_pw_aff mpa)
{
  auto res = isl_pw_multi_aff_from_multi_pw_aff(mpa.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::from_set(set set)
{
  auto res = isl_pw_multi_aff_from_set(set.release());
  return manage(res);
}

id pw_multi_aff::get_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_pw_multi_aff_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

std::string pw_multi_aff::get_dim_name(isl::dim type, unsigned int pos) const
{
  auto res = isl_pw_multi_aff_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

space pw_multi_aff::get_domain_space() const
{
  auto res = isl_pw_multi_aff_get_domain_space(get());
  return manage(res);
}

pw_aff pw_multi_aff::get_pw_aff(int pos) const
{
  auto res = isl_pw_multi_aff_get_pw_aff(get(), pos);
  return manage(res);
}

space pw_multi_aff::get_space() const
{
  auto res = isl_pw_multi_aff_get_space(get());
  return manage(res);
}

id pw_multi_aff::get_tuple_id(isl::dim type) const
{
  auto res = isl_pw_multi_aff_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

std::string pw_multi_aff::get_tuple_name(isl::dim type) const
{
  auto res = isl_pw_multi_aff_get_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  std::string tmp(res);
  return tmp;
}

pw_multi_aff pw_multi_aff::gist(set set) const
{
  auto res = isl_pw_multi_aff_gist(copy(), set.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::gist_params(set set) const
{
  auto res = isl_pw_multi_aff_gist_params(copy(), set.release());
  return manage(res);
}

boolean pw_multi_aff::has_tuple_id(isl::dim type) const
{
  auto res = isl_pw_multi_aff_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

boolean pw_multi_aff::has_tuple_name(isl::dim type) const
{
  auto res = isl_pw_multi_aff_has_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

pw_multi_aff pw_multi_aff::identity(space space)
{
  auto res = isl_pw_multi_aff_identity(space.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::intersect_domain(set set) const
{
  auto res = isl_pw_multi_aff_intersect_domain(copy(), set.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::intersect_params(set set) const
{
  auto res = isl_pw_multi_aff_intersect_params(copy(), set.release());
  return manage(res);
}

boolean pw_multi_aff::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_pw_multi_aff_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean pw_multi_aff::involves_nan() const
{
  auto res = isl_pw_multi_aff_involves_nan(get());
  return manage(res);
}

boolean pw_multi_aff::is_equal(const pw_multi_aff &pma2) const
{
  auto res = isl_pw_multi_aff_is_equal(get(), pma2.get());
  return manage(res);
}

pw_multi_aff pw_multi_aff::multi_val_on_domain(set domain, multi_val mv)
{
  auto res = isl_pw_multi_aff_multi_val_on_domain(domain.release(), mv.release());
  return manage(res);
}

int pw_multi_aff::n_piece() const
{
  auto res = isl_pw_multi_aff_n_piece(get());
  return res;
}

pw_multi_aff pw_multi_aff::neg() const
{
  auto res = isl_pw_multi_aff_neg(copy());
  return manage(res);
}

boolean pw_multi_aff::plain_is_equal(const pw_multi_aff &pma2) const
{
  auto res = isl_pw_multi_aff_plain_is_equal(get(), pma2.get());
  return manage(res);
}

pw_multi_aff pw_multi_aff::product(pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_product(copy(), pma2.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::project_domain_on_params() const
{
  auto res = isl_pw_multi_aff_project_domain_on_params(copy());
  return manage(res);
}

pw_multi_aff pw_multi_aff::project_out_map(space space, isl::dim type, unsigned int first, unsigned int n)
{
  auto res = isl_pw_multi_aff_project_out_map(space.release(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

pw_multi_aff pw_multi_aff::pullback(multi_aff ma) const
{
  auto res = isl_pw_multi_aff_pullback_multi_aff(copy(), ma.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::pullback(pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_pullback_pw_multi_aff(copy(), pma2.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::range_map(space space)
{
  auto res = isl_pw_multi_aff_range_map(space.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::range_product(pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_range_product(copy(), pma2.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::reset_tuple_id(isl::dim type) const
{
  auto res = isl_pw_multi_aff_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

pw_multi_aff pw_multi_aff::reset_user() const
{
  auto res = isl_pw_multi_aff_reset_user(copy());
  return manage(res);
}

pw_multi_aff pw_multi_aff::scale_down_val(val v) const
{
  auto res = isl_pw_multi_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::scale_multi_val(multi_val mv) const
{
  auto res = isl_pw_multi_aff_scale_multi_val(copy(), mv.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::scale_val(val v) const
{
  auto res = isl_pw_multi_aff_scale_val(copy(), v.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::set_dim_id(isl::dim type, unsigned int pos, id id) const
{
  auto res = isl_pw_multi_aff_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::set_pw_aff(unsigned int pos, pw_aff pa) const
{
  auto res = isl_pw_multi_aff_set_pw_aff(copy(), pos, pa.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::set_tuple_id(isl::dim type, id id) const
{
  auto res = isl_pw_multi_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::sub(pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_sub(copy(), pma2.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::subtract_domain(set set) const
{
  auto res = isl_pw_multi_aff_subtract_domain(copy(), set.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::union_add(pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_union_add(copy(), pma2.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::union_lexmax(pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_union_lexmax(copy(), pma2.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::union_lexmin(pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_union_lexmin(copy(), pma2.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::zero(space space)
{
  auto res = isl_pw_multi_aff_zero(space.release());
  return manage(res);
}

// implementations for isl::pw_multi_aff_list
pw_multi_aff_list manage(__isl_take isl_pw_multi_aff_list *ptr) {
  return pw_multi_aff_list(ptr);
}
pw_multi_aff_list manage_copy(__isl_keep isl_pw_multi_aff_list *ptr) {
  ptr = isl_pw_multi_aff_list_copy(ptr);
  return pw_multi_aff_list(ptr);
}

pw_multi_aff_list::pw_multi_aff_list()
    : ptr(nullptr) {}

pw_multi_aff_list::pw_multi_aff_list(const pw_multi_aff_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
pw_multi_aff_list::pw_multi_aff_list(std::nullptr_t)
    : ptr(nullptr) {}


pw_multi_aff_list::pw_multi_aff_list(__isl_take isl_pw_multi_aff_list *ptr)
    : ptr(ptr) {}


pw_multi_aff_list &pw_multi_aff_list::operator=(pw_multi_aff_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

pw_multi_aff_list::~pw_multi_aff_list() {
  if (ptr)
    isl_pw_multi_aff_list_free(ptr);
}

__isl_give isl_pw_multi_aff_list *pw_multi_aff_list::copy() const & {
  return isl_pw_multi_aff_list_copy(ptr);
}

__isl_keep isl_pw_multi_aff_list *pw_multi_aff_list::get() const {
  return ptr;
}

__isl_give isl_pw_multi_aff_list *pw_multi_aff_list::release() {
  isl_pw_multi_aff_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool pw_multi_aff_list::is_null() const {
  return ptr == nullptr;
}
pw_multi_aff_list::operator bool() const {
  return !is_null();
}


ctx pw_multi_aff_list::get_ctx() const {
  return ctx(isl_pw_multi_aff_list_get_ctx(ptr));
}

void pw_multi_aff_list::dump() const {
  isl_pw_multi_aff_list_dump(get());
}


pw_multi_aff_list pw_multi_aff_list::add(pw_multi_aff el) const
{
  auto res = isl_pw_multi_aff_list_add(copy(), el.release());
  return manage(res);
}

pw_multi_aff_list pw_multi_aff_list::alloc(ctx ctx, int n)
{
  auto res = isl_pw_multi_aff_list_alloc(ctx.release(), n);
  return manage(res);
}

pw_multi_aff_list pw_multi_aff_list::concat(pw_multi_aff_list list2) const
{
  auto res = isl_pw_multi_aff_list_concat(copy(), list2.release());
  return manage(res);
}

pw_multi_aff_list pw_multi_aff_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_pw_multi_aff_list_drop(copy(), first, n);
  return manage(res);
}

stat pw_multi_aff_list::foreach(const std::function<stat(pw_multi_aff)> &fn) const
{
  struct fn_data {
    const std::function<stat(pw_multi_aff)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_pw_multi_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_pw_multi_aff_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

pw_multi_aff_list pw_multi_aff_list::from_pw_multi_aff(pw_multi_aff el)
{
  auto res = isl_pw_multi_aff_list_from_pw_multi_aff(el.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff_list::get_at(int index) const
{
  auto res = isl_pw_multi_aff_list_get_at(get(), index);
  return manage(res);
}

pw_multi_aff pw_multi_aff_list::get_pw_multi_aff(int index) const
{
  auto res = isl_pw_multi_aff_list_get_pw_multi_aff(get(), index);
  return manage(res);
}

pw_multi_aff_list pw_multi_aff_list::insert(unsigned int pos, pw_multi_aff el) const
{
  auto res = isl_pw_multi_aff_list_insert(copy(), pos, el.release());
  return manage(res);
}

int pw_multi_aff_list::n_pw_multi_aff() const
{
  auto res = isl_pw_multi_aff_list_n_pw_multi_aff(get());
  return res;
}

pw_multi_aff_list pw_multi_aff_list::reverse() const
{
  auto res = isl_pw_multi_aff_list_reverse(copy());
  return manage(res);
}

pw_multi_aff_list pw_multi_aff_list::set_pw_multi_aff(int index, pw_multi_aff el) const
{
  auto res = isl_pw_multi_aff_list_set_pw_multi_aff(copy(), index, el.release());
  return manage(res);
}

int pw_multi_aff_list::size() const
{
  auto res = isl_pw_multi_aff_list_size(get());
  return res;
}

pw_multi_aff_list pw_multi_aff_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_pw_multi_aff_list_swap(copy(), pos1, pos2);
  return manage(res);
}

// implementations for isl::pw_qpolynomial
pw_qpolynomial manage(__isl_take isl_pw_qpolynomial *ptr) {
  return pw_qpolynomial(ptr);
}
pw_qpolynomial manage_copy(__isl_keep isl_pw_qpolynomial *ptr) {
  ptr = isl_pw_qpolynomial_copy(ptr);
  return pw_qpolynomial(ptr);
}

pw_qpolynomial::pw_qpolynomial()
    : ptr(nullptr) {}

pw_qpolynomial::pw_qpolynomial(const pw_qpolynomial &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
pw_qpolynomial::pw_qpolynomial(std::nullptr_t)
    : ptr(nullptr) {}


pw_qpolynomial::pw_qpolynomial(__isl_take isl_pw_qpolynomial *ptr)
    : ptr(ptr) {}

pw_qpolynomial::pw_qpolynomial(ctx ctx, const std::string &str)
{
  auto res = isl_pw_qpolynomial_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

pw_qpolynomial &pw_qpolynomial::operator=(pw_qpolynomial obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

pw_qpolynomial::~pw_qpolynomial() {
  if (ptr)
    isl_pw_qpolynomial_free(ptr);
}

__isl_give isl_pw_qpolynomial *pw_qpolynomial::copy() const & {
  return isl_pw_qpolynomial_copy(ptr);
}

__isl_keep isl_pw_qpolynomial *pw_qpolynomial::get() const {
  return ptr;
}

__isl_give isl_pw_qpolynomial *pw_qpolynomial::release() {
  isl_pw_qpolynomial *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool pw_qpolynomial::is_null() const {
  return ptr == nullptr;
}
pw_qpolynomial::operator bool() const {
  return !is_null();
}


ctx pw_qpolynomial::get_ctx() const {
  return ctx(isl_pw_qpolynomial_get_ctx(ptr));
}
std::string pw_qpolynomial::to_str() const {
  char *Tmp = isl_pw_qpolynomial_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void pw_qpolynomial::dump() const {
  isl_pw_qpolynomial_dump(get());
}


pw_qpolynomial pw_qpolynomial::add(pw_qpolynomial pwqp2) const
{
  auto res = isl_pw_qpolynomial_add(copy(), pwqp2.release());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_pw_qpolynomial_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::alloc(set set, qpolynomial qp)
{
  auto res = isl_pw_qpolynomial_alloc(set.release(), qp.release());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::coalesce() const
{
  auto res = isl_pw_qpolynomial_coalesce(copy());
  return manage(res);
}

unsigned int pw_qpolynomial::dim(isl::dim type) const
{
  auto res = isl_pw_qpolynomial_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

set pw_qpolynomial::domain() const
{
  auto res = isl_pw_qpolynomial_domain(copy());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_pw_qpolynomial_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::drop_unused_params() const
{
  auto res = isl_pw_qpolynomial_drop_unused_params(copy());
  return manage(res);
}

val pw_qpolynomial::eval(point pnt) const
{
  auto res = isl_pw_qpolynomial_eval(copy(), pnt.release());
  return manage(res);
}

int pw_qpolynomial::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_pw_qpolynomial_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

pw_qpolynomial pw_qpolynomial::fix_val(isl::dim type, unsigned int n, val v) const
{
  auto res = isl_pw_qpolynomial_fix_val(copy(), static_cast<enum isl_dim_type>(type), n, v.release());
  return manage(res);
}

stat pw_qpolynomial::foreach_piece(const std::function<stat(set, qpolynomial)> &fn) const
{
  struct fn_data {
    const std::function<stat(set, qpolynomial)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_set *arg_0, isl_qpolynomial *arg_1, void *arg_2) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_2);
    stat ret = (*data->func)(manage(arg_0), manage(arg_1));
    return ret.release();
  };
  auto res = isl_pw_qpolynomial_foreach_piece(get(), fn_lambda, &fn_data);
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::from_pw_aff(pw_aff pwaff)
{
  auto res = isl_pw_qpolynomial_from_pw_aff(pwaff.release());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::from_qpolynomial(qpolynomial qp)
{
  auto res = isl_pw_qpolynomial_from_qpolynomial(qp.release());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::from_range() const
{
  auto res = isl_pw_qpolynomial_from_range(copy());
  return manage(res);
}

space pw_qpolynomial::get_domain_space() const
{
  auto res = isl_pw_qpolynomial_get_domain_space(get());
  return manage(res);
}

space pw_qpolynomial::get_space() const
{
  auto res = isl_pw_qpolynomial_get_space(get());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::gist(set context) const
{
  auto res = isl_pw_qpolynomial_gist(copy(), context.release());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::gist_params(set context) const
{
  auto res = isl_pw_qpolynomial_gist_params(copy(), context.release());
  return manage(res);
}

boolean pw_qpolynomial::has_equal_space(const pw_qpolynomial &pwqp2) const
{
  auto res = isl_pw_qpolynomial_has_equal_space(get(), pwqp2.get());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::insert_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_pw_qpolynomial_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::intersect_domain(set set) const
{
  auto res = isl_pw_qpolynomial_intersect_domain(copy(), set.release());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::intersect_params(set set) const
{
  auto res = isl_pw_qpolynomial_intersect_params(copy(), set.release());
  return manage(res);
}

boolean pw_qpolynomial::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_pw_qpolynomial_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean pw_qpolynomial::involves_nan() const
{
  auto res = isl_pw_qpolynomial_involves_nan(get());
  return manage(res);
}

boolean pw_qpolynomial::is_zero() const
{
  auto res = isl_pw_qpolynomial_is_zero(get());
  return manage(res);
}

val pw_qpolynomial::max() const
{
  auto res = isl_pw_qpolynomial_max(copy());
  return manage(res);
}

val pw_qpolynomial::min() const
{
  auto res = isl_pw_qpolynomial_min(copy());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_pw_qpolynomial_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::mul(pw_qpolynomial pwqp2) const
{
  auto res = isl_pw_qpolynomial_mul(copy(), pwqp2.release());
  return manage(res);
}

int pw_qpolynomial::n_piece() const
{
  auto res = isl_pw_qpolynomial_n_piece(get());
  return res;
}

pw_qpolynomial pw_qpolynomial::neg() const
{
  auto res = isl_pw_qpolynomial_neg(copy());
  return manage(res);
}

boolean pw_qpolynomial::plain_is_equal(const pw_qpolynomial &pwqp2) const
{
  auto res = isl_pw_qpolynomial_plain_is_equal(get(), pwqp2.get());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::pow(unsigned int exponent) const
{
  auto res = isl_pw_qpolynomial_pow(copy(), exponent);
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::project_domain_on_params() const
{
  auto res = isl_pw_qpolynomial_project_domain_on_params(copy());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::reset_domain_space(space dim) const
{
  auto res = isl_pw_qpolynomial_reset_domain_space(copy(), dim.release());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::reset_user() const
{
  auto res = isl_pw_qpolynomial_reset_user(copy());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::scale_down_val(val v) const
{
  auto res = isl_pw_qpolynomial_scale_down_val(copy(), v.release());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::scale_val(val v) const
{
  auto res = isl_pw_qpolynomial_scale_val(copy(), v.release());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::split_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_pw_qpolynomial_split_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::split_periods(int max_periods) const
{
  auto res = isl_pw_qpolynomial_split_periods(copy(), max_periods);
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::sub(pw_qpolynomial pwqp2) const
{
  auto res = isl_pw_qpolynomial_sub(copy(), pwqp2.release());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::subtract_domain(set set) const
{
  auto res = isl_pw_qpolynomial_subtract_domain(copy(), set.release());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::to_polynomial(int sign) const
{
  auto res = isl_pw_qpolynomial_to_polynomial(copy(), sign);
  return manage(res);
}

pw_qpolynomial pw_qpolynomial::zero(space dim)
{
  auto res = isl_pw_qpolynomial_zero(dim.release());
  return manage(res);
}

// implementations for isl::pw_qpolynomial_fold_list
pw_qpolynomial_fold_list manage(__isl_take isl_pw_qpolynomial_fold_list *ptr) {
  return pw_qpolynomial_fold_list(ptr);
}
pw_qpolynomial_fold_list manage_copy(__isl_keep isl_pw_qpolynomial_fold_list *ptr) {
  ptr = isl_pw_qpolynomial_fold_list_copy(ptr);
  return pw_qpolynomial_fold_list(ptr);
}

pw_qpolynomial_fold_list::pw_qpolynomial_fold_list()
    : ptr(nullptr) {}

pw_qpolynomial_fold_list::pw_qpolynomial_fold_list(const pw_qpolynomial_fold_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
pw_qpolynomial_fold_list::pw_qpolynomial_fold_list(std::nullptr_t)
    : ptr(nullptr) {}


pw_qpolynomial_fold_list::pw_qpolynomial_fold_list(__isl_take isl_pw_qpolynomial_fold_list *ptr)
    : ptr(ptr) {}


pw_qpolynomial_fold_list &pw_qpolynomial_fold_list::operator=(pw_qpolynomial_fold_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

pw_qpolynomial_fold_list::~pw_qpolynomial_fold_list() {
  if (ptr)
    isl_pw_qpolynomial_fold_list_free(ptr);
}

__isl_give isl_pw_qpolynomial_fold_list *pw_qpolynomial_fold_list::copy() const & {
  return isl_pw_qpolynomial_fold_list_copy(ptr);
}

__isl_keep isl_pw_qpolynomial_fold_list *pw_qpolynomial_fold_list::get() const {
  return ptr;
}

__isl_give isl_pw_qpolynomial_fold_list *pw_qpolynomial_fold_list::release() {
  isl_pw_qpolynomial_fold_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool pw_qpolynomial_fold_list::is_null() const {
  return ptr == nullptr;
}
pw_qpolynomial_fold_list::operator bool() const {
  return !is_null();
}


ctx pw_qpolynomial_fold_list::get_ctx() const {
  return ctx(isl_pw_qpolynomial_fold_list_get_ctx(ptr));
}

void pw_qpolynomial_fold_list::dump() const {
  isl_pw_qpolynomial_fold_list_dump(get());
}



// implementations for isl::pw_qpolynomial_list
pw_qpolynomial_list manage(__isl_take isl_pw_qpolynomial_list *ptr) {
  return pw_qpolynomial_list(ptr);
}
pw_qpolynomial_list manage_copy(__isl_keep isl_pw_qpolynomial_list *ptr) {
  ptr = isl_pw_qpolynomial_list_copy(ptr);
  return pw_qpolynomial_list(ptr);
}

pw_qpolynomial_list::pw_qpolynomial_list()
    : ptr(nullptr) {}

pw_qpolynomial_list::pw_qpolynomial_list(const pw_qpolynomial_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
pw_qpolynomial_list::pw_qpolynomial_list(std::nullptr_t)
    : ptr(nullptr) {}


pw_qpolynomial_list::pw_qpolynomial_list(__isl_take isl_pw_qpolynomial_list *ptr)
    : ptr(ptr) {}


pw_qpolynomial_list &pw_qpolynomial_list::operator=(pw_qpolynomial_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

pw_qpolynomial_list::~pw_qpolynomial_list() {
  if (ptr)
    isl_pw_qpolynomial_list_free(ptr);
}

__isl_give isl_pw_qpolynomial_list *pw_qpolynomial_list::copy() const & {
  return isl_pw_qpolynomial_list_copy(ptr);
}

__isl_keep isl_pw_qpolynomial_list *pw_qpolynomial_list::get() const {
  return ptr;
}

__isl_give isl_pw_qpolynomial_list *pw_qpolynomial_list::release() {
  isl_pw_qpolynomial_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool pw_qpolynomial_list::is_null() const {
  return ptr == nullptr;
}
pw_qpolynomial_list::operator bool() const {
  return !is_null();
}


ctx pw_qpolynomial_list::get_ctx() const {
  return ctx(isl_pw_qpolynomial_list_get_ctx(ptr));
}

void pw_qpolynomial_list::dump() const {
  isl_pw_qpolynomial_list_dump(get());
}


pw_qpolynomial_list pw_qpolynomial_list::add(pw_qpolynomial el) const
{
  auto res = isl_pw_qpolynomial_list_add(copy(), el.release());
  return manage(res);
}

pw_qpolynomial_list pw_qpolynomial_list::alloc(ctx ctx, int n)
{
  auto res = isl_pw_qpolynomial_list_alloc(ctx.release(), n);
  return manage(res);
}

pw_qpolynomial_list pw_qpolynomial_list::concat(pw_qpolynomial_list list2) const
{
  auto res = isl_pw_qpolynomial_list_concat(copy(), list2.release());
  return manage(res);
}

pw_qpolynomial_list pw_qpolynomial_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_pw_qpolynomial_list_drop(copy(), first, n);
  return manage(res);
}

stat pw_qpolynomial_list::foreach(const std::function<stat(pw_qpolynomial)> &fn) const
{
  struct fn_data {
    const std::function<stat(pw_qpolynomial)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_pw_qpolynomial *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_pw_qpolynomial_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

pw_qpolynomial_list pw_qpolynomial_list::from_pw_qpolynomial(pw_qpolynomial el)
{
  auto res = isl_pw_qpolynomial_list_from_pw_qpolynomial(el.release());
  return manage(res);
}

pw_qpolynomial pw_qpolynomial_list::get_at(int index) const
{
  auto res = isl_pw_qpolynomial_list_get_at(get(), index);
  return manage(res);
}

pw_qpolynomial pw_qpolynomial_list::get_pw_qpolynomial(int index) const
{
  auto res = isl_pw_qpolynomial_list_get_pw_qpolynomial(get(), index);
  return manage(res);
}

pw_qpolynomial_list pw_qpolynomial_list::insert(unsigned int pos, pw_qpolynomial el) const
{
  auto res = isl_pw_qpolynomial_list_insert(copy(), pos, el.release());
  return manage(res);
}

int pw_qpolynomial_list::n_pw_qpolynomial() const
{
  auto res = isl_pw_qpolynomial_list_n_pw_qpolynomial(get());
  return res;
}

pw_qpolynomial_list pw_qpolynomial_list::reverse() const
{
  auto res = isl_pw_qpolynomial_list_reverse(copy());
  return manage(res);
}

pw_qpolynomial_list pw_qpolynomial_list::set_pw_qpolynomial(int index, pw_qpolynomial el) const
{
  auto res = isl_pw_qpolynomial_list_set_pw_qpolynomial(copy(), index, el.release());
  return manage(res);
}

int pw_qpolynomial_list::size() const
{
  auto res = isl_pw_qpolynomial_list_size(get());
  return res;
}

pw_qpolynomial_list pw_qpolynomial_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_pw_qpolynomial_list_swap(copy(), pos1, pos2);
  return manage(res);
}

// implementations for isl::qpolynomial
qpolynomial manage(__isl_take isl_qpolynomial *ptr) {
  return qpolynomial(ptr);
}
qpolynomial manage_copy(__isl_keep isl_qpolynomial *ptr) {
  ptr = isl_qpolynomial_copy(ptr);
  return qpolynomial(ptr);
}

qpolynomial::qpolynomial()
    : ptr(nullptr) {}

qpolynomial::qpolynomial(const qpolynomial &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
qpolynomial::qpolynomial(std::nullptr_t)
    : ptr(nullptr) {}


qpolynomial::qpolynomial(__isl_take isl_qpolynomial *ptr)
    : ptr(ptr) {}


qpolynomial &qpolynomial::operator=(qpolynomial obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

qpolynomial::~qpolynomial() {
  if (ptr)
    isl_qpolynomial_free(ptr);
}

__isl_give isl_qpolynomial *qpolynomial::copy() const & {
  return isl_qpolynomial_copy(ptr);
}

__isl_keep isl_qpolynomial *qpolynomial::get() const {
  return ptr;
}

__isl_give isl_qpolynomial *qpolynomial::release() {
  isl_qpolynomial *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool qpolynomial::is_null() const {
  return ptr == nullptr;
}
qpolynomial::operator bool() const {
  return !is_null();
}


ctx qpolynomial::get_ctx() const {
  return ctx(isl_qpolynomial_get_ctx(ptr));
}

void qpolynomial::dump() const {
  isl_qpolynomial_dump(get());
}


qpolynomial qpolynomial::add(qpolynomial qp2) const
{
  auto res = isl_qpolynomial_add(copy(), qp2.release());
  return manage(res);
}

qpolynomial qpolynomial::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_qpolynomial_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

qpolynomial qpolynomial::align_params(space model) const
{
  auto res = isl_qpolynomial_align_params(copy(), model.release());
  return manage(res);
}

stat qpolynomial::as_polynomial_on_domain(const basic_set &bset, const std::function<stat(basic_set, qpolynomial)> &fn) const
{
  struct fn_data {
    const std::function<stat(basic_set, qpolynomial)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_basic_set *arg_0, isl_qpolynomial *arg_1, void *arg_2) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_2);
    stat ret = (*data->func)(manage(arg_0), manage(arg_1));
    return ret.release();
  };
  auto res = isl_qpolynomial_as_polynomial_on_domain(get(), bset.get(), fn_lambda, &fn_data);
  return manage(res);
}

unsigned int qpolynomial::dim(isl::dim type) const
{
  auto res = isl_qpolynomial_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

qpolynomial qpolynomial::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_qpolynomial_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

val qpolynomial::eval(point pnt) const
{
  auto res = isl_qpolynomial_eval(copy(), pnt.release());
  return manage(res);
}

stat qpolynomial::foreach_term(const std::function<stat(term)> &fn) const
{
  struct fn_data {
    const std::function<stat(term)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_term *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_qpolynomial_foreach_term(get(), fn_lambda, &fn_data);
  return manage(res);
}

qpolynomial qpolynomial::from_aff(aff aff)
{
  auto res = isl_qpolynomial_from_aff(aff.release());
  return manage(res);
}

qpolynomial qpolynomial::from_constraint(constraint c, isl::dim type, unsigned int pos)
{
  auto res = isl_qpolynomial_from_constraint(c.release(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

qpolynomial qpolynomial::from_term(term term)
{
  auto res = isl_qpolynomial_from_term(term.release());
  return manage(res);
}

val qpolynomial::get_constant_val() const
{
  auto res = isl_qpolynomial_get_constant_val(get());
  return manage(res);
}

space qpolynomial::get_domain_space() const
{
  auto res = isl_qpolynomial_get_domain_space(get());
  return manage(res);
}

space qpolynomial::get_space() const
{
  auto res = isl_qpolynomial_get_space(get());
  return manage(res);
}

qpolynomial qpolynomial::gist(set context) const
{
  auto res = isl_qpolynomial_gist(copy(), context.release());
  return manage(res);
}

qpolynomial qpolynomial::gist_params(set context) const
{
  auto res = isl_qpolynomial_gist_params(copy(), context.release());
  return manage(res);
}

qpolynomial qpolynomial::homogenize() const
{
  auto res = isl_qpolynomial_homogenize(copy());
  return manage(res);
}

qpolynomial qpolynomial::infty_on_domain(space dim)
{
  auto res = isl_qpolynomial_infty_on_domain(dim.release());
  return manage(res);
}

qpolynomial qpolynomial::insert_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_qpolynomial_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean qpolynomial::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_qpolynomial_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean qpolynomial::is_infty() const
{
  auto res = isl_qpolynomial_is_infty(get());
  return manage(res);
}

boolean qpolynomial::is_nan() const
{
  auto res = isl_qpolynomial_is_nan(get());
  return manage(res);
}

boolean qpolynomial::is_neginfty() const
{
  auto res = isl_qpolynomial_is_neginfty(get());
  return manage(res);
}

boolean qpolynomial::is_zero() const
{
  auto res = isl_qpolynomial_is_zero(get());
  return manage(res);
}

qpolynomial qpolynomial::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_qpolynomial_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

qpolynomial qpolynomial::mul(qpolynomial qp2) const
{
  auto res = isl_qpolynomial_mul(copy(), qp2.release());
  return manage(res);
}

qpolynomial qpolynomial::nan_on_domain(space dim)
{
  auto res = isl_qpolynomial_nan_on_domain(dim.release());
  return manage(res);
}

qpolynomial qpolynomial::neg() const
{
  auto res = isl_qpolynomial_neg(copy());
  return manage(res);
}

qpolynomial qpolynomial::neginfty_on_domain(space dim)
{
  auto res = isl_qpolynomial_neginfty_on_domain(dim.release());
  return manage(res);
}

qpolynomial qpolynomial::one_on_domain(space dim)
{
  auto res = isl_qpolynomial_one_on_domain(dim.release());
  return manage(res);
}

boolean qpolynomial::plain_is_equal(const qpolynomial &qp2) const
{
  auto res = isl_qpolynomial_plain_is_equal(get(), qp2.get());
  return manage(res);
}

qpolynomial qpolynomial::pow(unsigned int power) const
{
  auto res = isl_qpolynomial_pow(copy(), power);
  return manage(res);
}

qpolynomial qpolynomial::project_domain_on_params() const
{
  auto res = isl_qpolynomial_project_domain_on_params(copy());
  return manage(res);
}

qpolynomial qpolynomial::scale_down_val(val v) const
{
  auto res = isl_qpolynomial_scale_down_val(copy(), v.release());
  return manage(res);
}

qpolynomial qpolynomial::scale_val(val v) const
{
  auto res = isl_qpolynomial_scale_val(copy(), v.release());
  return manage(res);
}

int qpolynomial::sgn() const
{
  auto res = isl_qpolynomial_sgn(get());
  return res;
}

qpolynomial qpolynomial::sub(qpolynomial qp2) const
{
  auto res = isl_qpolynomial_sub(copy(), qp2.release());
  return manage(res);
}

qpolynomial qpolynomial::val_on_domain(space space, val val)
{
  auto res = isl_qpolynomial_val_on_domain(space.release(), val.release());
  return manage(res);
}

qpolynomial qpolynomial::var_on_domain(space dim, isl::dim type, unsigned int pos)
{
  auto res = isl_qpolynomial_var_on_domain(dim.release(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

qpolynomial qpolynomial::zero_on_domain(space dim)
{
  auto res = isl_qpolynomial_zero_on_domain(dim.release());
  return manage(res);
}

// implementations for isl::schedule
schedule manage(__isl_take isl_schedule *ptr) {
  return schedule(ptr);
}
schedule manage_copy(__isl_keep isl_schedule *ptr) {
  ptr = isl_schedule_copy(ptr);
  return schedule(ptr);
}

schedule::schedule()
    : ptr(nullptr) {}

schedule::schedule(const schedule &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
schedule::schedule(std::nullptr_t)
    : ptr(nullptr) {}


schedule::schedule(__isl_take isl_schedule *ptr)
    : ptr(ptr) {}

schedule::schedule(ctx ctx, const std::string &str)
{
  auto res = isl_schedule_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

schedule &schedule::operator=(schedule obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

schedule::~schedule() {
  if (ptr)
    isl_schedule_free(ptr);
}

__isl_give isl_schedule *schedule::copy() const & {
  return isl_schedule_copy(ptr);
}

__isl_keep isl_schedule *schedule::get() const {
  return ptr;
}

__isl_give isl_schedule *schedule::release() {
  isl_schedule *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool schedule::is_null() const {
  return ptr == nullptr;
}
schedule::operator bool() const {
  return !is_null();
}


ctx schedule::get_ctx() const {
  return ctx(isl_schedule_get_ctx(ptr));
}
std::string schedule::to_str() const {
  char *Tmp = isl_schedule_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void schedule::dump() const {
  isl_schedule_dump(get());
}


schedule schedule::align_params(space space) const
{
  auto res = isl_schedule_align_params(copy(), space.release());
  return manage(res);
}

schedule schedule::empty(space space)
{
  auto res = isl_schedule_empty(space.release());
  return manage(res);
}

schedule schedule::from_domain(union_set domain)
{
  auto res = isl_schedule_from_domain(domain.release());
  return manage(res);
}

union_set schedule::get_domain() const
{
  auto res = isl_schedule_get_domain(get());
  return manage(res);
}

union_map schedule::get_map() const
{
  auto res = isl_schedule_get_map(get());
  return manage(res);
}

schedule_node schedule::get_root() const
{
  auto res = isl_schedule_get_root(get());
  return manage(res);
}

schedule schedule::gist_domain_params(set context) const
{
  auto res = isl_schedule_gist_domain_params(copy(), context.release());
  return manage(res);
}

schedule schedule::insert_context(set context) const
{
  auto res = isl_schedule_insert_context(copy(), context.release());
  return manage(res);
}

schedule schedule::insert_guard(set guard) const
{
  auto res = isl_schedule_insert_guard(copy(), guard.release());
  return manage(res);
}

schedule schedule::insert_partial_schedule(multi_union_pw_aff partial) const
{
  auto res = isl_schedule_insert_partial_schedule(copy(), partial.release());
  return manage(res);
}

schedule schedule::intersect_domain(union_set domain) const
{
  auto res = isl_schedule_intersect_domain(copy(), domain.release());
  return manage(res);
}

boolean schedule::plain_is_equal(const schedule &schedule2) const
{
  auto res = isl_schedule_plain_is_equal(get(), schedule2.get());
  return manage(res);
}

schedule schedule::pullback(union_pw_multi_aff upma) const
{
  auto res = isl_schedule_pullback_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

schedule schedule::reset_user() const
{
  auto res = isl_schedule_reset_user(copy());
  return manage(res);
}

schedule schedule::sequence(schedule schedule2) const
{
  auto res = isl_schedule_sequence(copy(), schedule2.release());
  return manage(res);
}

// implementations for isl::schedule_constraints
schedule_constraints manage(__isl_take isl_schedule_constraints *ptr) {
  return schedule_constraints(ptr);
}
schedule_constraints manage_copy(__isl_keep isl_schedule_constraints *ptr) {
  ptr = isl_schedule_constraints_copy(ptr);
  return schedule_constraints(ptr);
}

schedule_constraints::schedule_constraints()
    : ptr(nullptr) {}

schedule_constraints::schedule_constraints(const schedule_constraints &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
schedule_constraints::schedule_constraints(std::nullptr_t)
    : ptr(nullptr) {}


schedule_constraints::schedule_constraints(__isl_take isl_schedule_constraints *ptr)
    : ptr(ptr) {}

schedule_constraints::schedule_constraints(ctx ctx, const std::string &str)
{
  auto res = isl_schedule_constraints_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

schedule_constraints &schedule_constraints::operator=(schedule_constraints obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

schedule_constraints::~schedule_constraints() {
  if (ptr)
    isl_schedule_constraints_free(ptr);
}

__isl_give isl_schedule_constraints *schedule_constraints::copy() const & {
  return isl_schedule_constraints_copy(ptr);
}

__isl_keep isl_schedule_constraints *schedule_constraints::get() const {
  return ptr;
}

__isl_give isl_schedule_constraints *schedule_constraints::release() {
  isl_schedule_constraints *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool schedule_constraints::is_null() const {
  return ptr == nullptr;
}
schedule_constraints::operator bool() const {
  return !is_null();
}


ctx schedule_constraints::get_ctx() const {
  return ctx(isl_schedule_constraints_get_ctx(ptr));
}
std::string schedule_constraints::to_str() const {
  char *Tmp = isl_schedule_constraints_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void schedule_constraints::dump() const {
  isl_schedule_constraints_dump(get());
}


schedule_constraints schedule_constraints::apply(union_map umap) const
{
  auto res = isl_schedule_constraints_apply(copy(), umap.release());
  return manage(res);
}

schedule schedule_constraints::compute_schedule() const
{
  auto res = isl_schedule_constraints_compute_schedule(copy());
  return manage(res);
}

union_map schedule_constraints::get_coincidence() const
{
  auto res = isl_schedule_constraints_get_coincidence(get());
  return manage(res);
}

union_map schedule_constraints::get_conditional_validity() const
{
  auto res = isl_schedule_constraints_get_conditional_validity(get());
  return manage(res);
}

union_map schedule_constraints::get_conditional_validity_condition() const
{
  auto res = isl_schedule_constraints_get_conditional_validity_condition(get());
  return manage(res);
}

set schedule_constraints::get_context() const
{
  auto res = isl_schedule_constraints_get_context(get());
  return manage(res);
}

union_set schedule_constraints::get_domain() const
{
  auto res = isl_schedule_constraints_get_domain(get());
  return manage(res);
}

union_map schedule_constraints::get_proximity() const
{
  auto res = isl_schedule_constraints_get_proximity(get());
  return manage(res);
}

union_map schedule_constraints::get_validity() const
{
  auto res = isl_schedule_constraints_get_validity(get());
  return manage(res);
}

schedule_constraints schedule_constraints::on_domain(union_set domain)
{
  auto res = isl_schedule_constraints_on_domain(domain.release());
  return manage(res);
}

schedule_constraints schedule_constraints::set_coincidence(union_map coincidence) const
{
  auto res = isl_schedule_constraints_set_coincidence(copy(), coincidence.release());
  return manage(res);
}

schedule_constraints schedule_constraints::set_conditional_validity(union_map condition, union_map validity) const
{
  auto res = isl_schedule_constraints_set_conditional_validity(copy(), condition.release(), validity.release());
  return manage(res);
}

schedule_constraints schedule_constraints::set_context(set context) const
{
  auto res = isl_schedule_constraints_set_context(copy(), context.release());
  return manage(res);
}

schedule_constraints schedule_constraints::set_proximity(union_map proximity) const
{
  auto res = isl_schedule_constraints_set_proximity(copy(), proximity.release());
  return manage(res);
}

schedule_constraints schedule_constraints::set_validity(union_map validity) const
{
  auto res = isl_schedule_constraints_set_validity(copy(), validity.release());
  return manage(res);
}

// implementations for isl::schedule_node
schedule_node manage(__isl_take isl_schedule_node *ptr) {
  return schedule_node(ptr);
}
schedule_node manage_copy(__isl_keep isl_schedule_node *ptr) {
  ptr = isl_schedule_node_copy(ptr);
  return schedule_node(ptr);
}

schedule_node::schedule_node()
    : ptr(nullptr) {}

schedule_node::schedule_node(const schedule_node &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
schedule_node::schedule_node(std::nullptr_t)
    : ptr(nullptr) {}


schedule_node::schedule_node(__isl_take isl_schedule_node *ptr)
    : ptr(ptr) {}


schedule_node &schedule_node::operator=(schedule_node obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

schedule_node::~schedule_node() {
  if (ptr)
    isl_schedule_node_free(ptr);
}

__isl_give isl_schedule_node *schedule_node::copy() const & {
  return isl_schedule_node_copy(ptr);
}

__isl_keep isl_schedule_node *schedule_node::get() const {
  return ptr;
}

__isl_give isl_schedule_node *schedule_node::release() {
  isl_schedule_node *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool schedule_node::is_null() const {
  return ptr == nullptr;
}
schedule_node::operator bool() const {
  return !is_null();
}


ctx schedule_node::get_ctx() const {
  return ctx(isl_schedule_node_get_ctx(ptr));
}
std::string schedule_node::to_str() const {
  char *Tmp = isl_schedule_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void schedule_node::dump() const {
  isl_schedule_node_dump(get());
}


schedule_node schedule_node::align_params(space space) const
{
  auto res = isl_schedule_node_align_params(copy(), space.release());
  return manage(res);
}

schedule_node schedule_node::ancestor(int generation) const
{
  auto res = isl_schedule_node_ancestor(copy(), generation);
  return manage(res);
}

boolean schedule_node::band_member_get_coincident(int pos) const
{
  auto res = isl_schedule_node_band_member_get_coincident(get(), pos);
  return manage(res);
}

schedule_node schedule_node::band_member_set_coincident(int pos, int coincident) const
{
  auto res = isl_schedule_node_band_member_set_coincident(copy(), pos, coincident);
  return manage(res);
}

schedule_node schedule_node::band_set_ast_build_options(union_set options) const
{
  auto res = isl_schedule_node_band_set_ast_build_options(copy(), options.release());
  return manage(res);
}

schedule_node schedule_node::child(int pos) const
{
  auto res = isl_schedule_node_child(copy(), pos);
  return manage(res);
}

set schedule_node::context_get_context() const
{
  auto res = isl_schedule_node_context_get_context(get());
  return manage(res);
}

schedule_node schedule_node::cut() const
{
  auto res = isl_schedule_node_cut(copy());
  return manage(res);
}

union_set schedule_node::domain_get_domain() const
{
  auto res = isl_schedule_node_domain_get_domain(get());
  return manage(res);
}

union_pw_multi_aff schedule_node::expansion_get_contraction() const
{
  auto res = isl_schedule_node_expansion_get_contraction(get());
  return manage(res);
}

union_map schedule_node::expansion_get_expansion() const
{
  auto res = isl_schedule_node_expansion_get_expansion(get());
  return manage(res);
}

union_map schedule_node::extension_get_extension() const
{
  auto res = isl_schedule_node_extension_get_extension(get());
  return manage(res);
}

union_set schedule_node::filter_get_filter() const
{
  auto res = isl_schedule_node_filter_get_filter(get());
  return manage(res);
}

schedule_node schedule_node::first_child() const
{
  auto res = isl_schedule_node_first_child(copy());
  return manage(res);
}

stat schedule_node::foreach_ancestor_top_down(const std::function<stat(schedule_node)> &fn) const
{
  struct fn_data {
    const std::function<stat(schedule_node)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_schedule_node *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage_copy(arg_0));
    return ret.release();
  };
  auto res = isl_schedule_node_foreach_ancestor_top_down(get(), fn_lambda, &fn_data);
  return manage(res);
}

schedule_node schedule_node::from_domain(union_set domain)
{
  auto res = isl_schedule_node_from_domain(domain.release());
  return manage(res);
}

schedule_node schedule_node::from_extension(union_map extension)
{
  auto res = isl_schedule_node_from_extension(extension.release());
  return manage(res);
}

int schedule_node::get_ancestor_child_position(const schedule_node &ancestor) const
{
  auto res = isl_schedule_node_get_ancestor_child_position(get(), ancestor.get());
  return res;
}

schedule_node schedule_node::get_child(int pos) const
{
  auto res = isl_schedule_node_get_child(get(), pos);
  return manage(res);
}

int schedule_node::get_child_position() const
{
  auto res = isl_schedule_node_get_child_position(get());
  return res;
}

union_set schedule_node::get_domain() const
{
  auto res = isl_schedule_node_get_domain(get());
  return manage(res);
}

multi_union_pw_aff schedule_node::get_prefix_schedule_multi_union_pw_aff() const
{
  auto res = isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(get());
  return manage(res);
}

union_map schedule_node::get_prefix_schedule_relation() const
{
  auto res = isl_schedule_node_get_prefix_schedule_relation(get());
  return manage(res);
}

union_map schedule_node::get_prefix_schedule_union_map() const
{
  auto res = isl_schedule_node_get_prefix_schedule_union_map(get());
  return manage(res);
}

union_pw_multi_aff schedule_node::get_prefix_schedule_union_pw_multi_aff() const
{
  auto res = isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(get());
  return manage(res);
}

schedule schedule_node::get_schedule() const
{
  auto res = isl_schedule_node_get_schedule(get());
  return manage(res);
}

int schedule_node::get_schedule_depth() const
{
  auto res = isl_schedule_node_get_schedule_depth(get());
  return res;
}

schedule_node schedule_node::get_shared_ancestor(const schedule_node &node2) const
{
  auto res = isl_schedule_node_get_shared_ancestor(get(), node2.get());
  return manage(res);
}

union_pw_multi_aff schedule_node::get_subtree_contraction() const
{
  auto res = isl_schedule_node_get_subtree_contraction(get());
  return manage(res);
}

union_map schedule_node::get_subtree_expansion() const
{
  auto res = isl_schedule_node_get_subtree_expansion(get());
  return manage(res);
}

union_map schedule_node::get_subtree_schedule_union_map() const
{
  auto res = isl_schedule_node_get_subtree_schedule_union_map(get());
  return manage(res);
}

int schedule_node::get_tree_depth() const
{
  auto res = isl_schedule_node_get_tree_depth(get());
  return res;
}

union_set schedule_node::get_universe_domain() const
{
  auto res = isl_schedule_node_get_universe_domain(get());
  return manage(res);
}

schedule_node schedule_node::graft_after(schedule_node graft) const
{
  auto res = isl_schedule_node_graft_after(copy(), graft.release());
  return manage(res);
}

schedule_node schedule_node::graft_before(schedule_node graft) const
{
  auto res = isl_schedule_node_graft_before(copy(), graft.release());
  return manage(res);
}

schedule_node schedule_node::group(id group_id) const
{
  auto res = isl_schedule_node_group(copy(), group_id.release());
  return manage(res);
}

set schedule_node::guard_get_guard() const
{
  auto res = isl_schedule_node_guard_get_guard(get());
  return manage(res);
}

boolean schedule_node::has_children() const
{
  auto res = isl_schedule_node_has_children(get());
  return manage(res);
}

boolean schedule_node::has_next_sibling() const
{
  auto res = isl_schedule_node_has_next_sibling(get());
  return manage(res);
}

boolean schedule_node::has_parent() const
{
  auto res = isl_schedule_node_has_parent(get());
  return manage(res);
}

boolean schedule_node::has_previous_sibling() const
{
  auto res = isl_schedule_node_has_previous_sibling(get());
  return manage(res);
}

schedule_node schedule_node::insert_context(set context) const
{
  auto res = isl_schedule_node_insert_context(copy(), context.release());
  return manage(res);
}

schedule_node schedule_node::insert_filter(union_set filter) const
{
  auto res = isl_schedule_node_insert_filter(copy(), filter.release());
  return manage(res);
}

schedule_node schedule_node::insert_guard(set context) const
{
  auto res = isl_schedule_node_insert_guard(copy(), context.release());
  return manage(res);
}

schedule_node schedule_node::insert_mark(id mark) const
{
  auto res = isl_schedule_node_insert_mark(copy(), mark.release());
  return manage(res);
}

schedule_node schedule_node::insert_partial_schedule(multi_union_pw_aff schedule) const
{
  auto res = isl_schedule_node_insert_partial_schedule(copy(), schedule.release());
  return manage(res);
}

schedule_node schedule_node::insert_sequence(union_set_list filters) const
{
  auto res = isl_schedule_node_insert_sequence(copy(), filters.release());
  return manage(res);
}

schedule_node schedule_node::insert_set(union_set_list filters) const
{
  auto res = isl_schedule_node_insert_set(copy(), filters.release());
  return manage(res);
}

boolean schedule_node::is_equal(const schedule_node &node2) const
{
  auto res = isl_schedule_node_is_equal(get(), node2.get());
  return manage(res);
}

boolean schedule_node::is_subtree_anchored() const
{
  auto res = isl_schedule_node_is_subtree_anchored(get());
  return manage(res);
}

id schedule_node::mark_get_id() const
{
  auto res = isl_schedule_node_mark_get_id(get());
  return manage(res);
}

int schedule_node::n_children() const
{
  auto res = isl_schedule_node_n_children(get());
  return res;
}

schedule_node schedule_node::next_sibling() const
{
  auto res = isl_schedule_node_next_sibling(copy());
  return manage(res);
}

schedule_node schedule_node::order_after(union_set filter) const
{
  auto res = isl_schedule_node_order_after(copy(), filter.release());
  return manage(res);
}

schedule_node schedule_node::order_before(union_set filter) const
{
  auto res = isl_schedule_node_order_before(copy(), filter.release());
  return manage(res);
}

schedule_node schedule_node::parent() const
{
  auto res = isl_schedule_node_parent(copy());
  return manage(res);
}

schedule_node schedule_node::previous_sibling() const
{
  auto res = isl_schedule_node_previous_sibling(copy());
  return manage(res);
}

schedule_node schedule_node::reset_user() const
{
  auto res = isl_schedule_node_reset_user(copy());
  return manage(res);
}

schedule_node schedule_node::root() const
{
  auto res = isl_schedule_node_root(copy());
  return manage(res);
}

schedule_node schedule_node::sequence_splice_child(int pos) const
{
  auto res = isl_schedule_node_sequence_splice_child(copy(), pos);
  return manage(res);
}

// implementations for isl::set
set manage(__isl_take isl_set *ptr) {
  return set(ptr);
}
set manage_copy(__isl_keep isl_set *ptr) {
  ptr = isl_set_copy(ptr);
  return set(ptr);
}

set::set()
    : ptr(nullptr) {}

set::set(const set &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
set::set(std::nullptr_t)
    : ptr(nullptr) {}


set::set(__isl_take isl_set *ptr)
    : ptr(ptr) {}

set::set(ctx ctx, const std::string &str)
{
  auto res = isl_set_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}
set::set(basic_set bset)
{
  auto res = isl_set_from_basic_set(bset.release());
  ptr = res;
}
set::set(point pnt)
{
  auto res = isl_set_from_point(pnt.release());
  ptr = res;
}
set::set(union_set uset)
{
  auto res = isl_set_from_union_set(uset.release());
  ptr = res;
}

set &set::operator=(set obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

set::~set() {
  if (ptr)
    isl_set_free(ptr);
}

__isl_give isl_set *set::copy() const & {
  return isl_set_copy(ptr);
}

__isl_keep isl_set *set::get() const {
  return ptr;
}

__isl_give isl_set *set::release() {
  isl_set *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool set::is_null() const {
  return ptr == nullptr;
}
set::operator bool() const {
  return !is_null();
}


ctx set::get_ctx() const {
  return ctx(isl_set_get_ctx(ptr));
}
std::string set::to_str() const {
  char *Tmp = isl_set_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void set::dump() const {
  isl_set_dump(get());
}


set set::add_constraint(constraint constraint) const
{
  auto res = isl_set_add_constraint(copy(), constraint.release());
  return manage(res);
}

set set::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_set_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

basic_set set::affine_hull() const
{
  auto res = isl_set_affine_hull(copy());
  return manage(res);
}

set set::align_params(space model) const
{
  auto res = isl_set_align_params(copy(), model.release());
  return manage(res);
}

set set::apply(map map) const
{
  auto res = isl_set_apply(copy(), map.release());
  return manage(res);
}

basic_set set::bounded_simple_hull() const
{
  auto res = isl_set_bounded_simple_hull(copy());
  return manage(res);
}

set set::box_from_points(point pnt1, point pnt2)
{
  auto res = isl_set_box_from_points(pnt1.release(), pnt2.release());
  return manage(res);
}

set set::coalesce() const
{
  auto res = isl_set_coalesce(copy());
  return manage(res);
}

basic_set set::coefficients() const
{
  auto res = isl_set_coefficients(copy());
  return manage(res);
}

set set::complement() const
{
  auto res = isl_set_complement(copy());
  return manage(res);
}

basic_set set::convex_hull() const
{
  auto res = isl_set_convex_hull(copy());
  return manage(res);
}

val set::count_val() const
{
  auto res = isl_set_count_val(get());
  return manage(res);
}

set set::detect_equalities() const
{
  auto res = isl_set_detect_equalities(copy());
  return manage(res);
}

unsigned int set::dim(isl::dim type) const
{
  auto res = isl_set_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

boolean set::dim_has_any_lower_bound(isl::dim type, unsigned int pos) const
{
  auto res = isl_set_dim_has_any_lower_bound(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

boolean set::dim_has_any_upper_bound(isl::dim type, unsigned int pos) const
{
  auto res = isl_set_dim_has_any_upper_bound(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

boolean set::dim_has_lower_bound(isl::dim type, unsigned int pos) const
{
  auto res = isl_set_dim_has_lower_bound(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

boolean set::dim_has_upper_bound(isl::dim type, unsigned int pos) const
{
  auto res = isl_set_dim_has_upper_bound(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

boolean set::dim_is_bounded(isl::dim type, unsigned int pos) const
{
  auto res = isl_set_dim_is_bounded(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

pw_aff set::dim_max(int pos) const
{
  auto res = isl_set_dim_max(copy(), pos);
  return manage(res);
}

pw_aff set::dim_min(int pos) const
{
  auto res = isl_set_dim_min(copy(), pos);
  return manage(res);
}

set set::drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_drop_constraints_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

set set::drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_drop_constraints_not_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

set set::drop_unused_params() const
{
  auto res = isl_set_drop_unused_params(copy());
  return manage(res);
}

set set::eliminate(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_eliminate(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

set set::empty(space space)
{
  auto res = isl_set_empty(space.release());
  return manage(res);
}

set set::equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_set_equate(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

int set::find_dim_by_id(isl::dim type, const id &id) const
{
  auto res = isl_set_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int set::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_set_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

set set::fix_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_set_fix_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

set set::fix_val(isl::dim type, unsigned int pos, val v) const
{
  auto res = isl_set_fix_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

set set::flat_product(set set2) const
{
  auto res = isl_set_flat_product(copy(), set2.release());
  return manage(res);
}

set set::flatten() const
{
  auto res = isl_set_flatten(copy());
  return manage(res);
}

map set::flatten_map() const
{
  auto res = isl_set_flatten_map(copy());
  return manage(res);
}

int set::follows_at(const set &set2, int pos) const
{
  auto res = isl_set_follows_at(get(), set2.get(), pos);
  return res;
}

stat set::foreach_basic_set(const std::function<stat(basic_set)> &fn) const
{
  struct fn_data {
    const std::function<stat(basic_set)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_basic_set *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_set_foreach_basic_set(get(), fn_lambda, &fn_data);
  return manage(res);
}

stat set::foreach_point(const std::function<stat(point)> &fn) const
{
  struct fn_data {
    const std::function<stat(point)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_point *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_set_foreach_point(get(), fn_lambda, &fn_data);
  return manage(res);
}

set set::from_multi_aff(multi_aff ma)
{
  auto res = isl_set_from_multi_aff(ma.release());
  return manage(res);
}

set set::from_multi_pw_aff(multi_pw_aff mpa)
{
  auto res = isl_set_from_multi_pw_aff(mpa.release());
  return manage(res);
}

set set::from_params() const
{
  auto res = isl_set_from_params(copy());
  return manage(res);
}

set set::from_pw_aff(pw_aff pwaff)
{
  auto res = isl_set_from_pw_aff(pwaff.release());
  return manage(res);
}

set set::from_pw_multi_aff(pw_multi_aff pma)
{
  auto res = isl_set_from_pw_multi_aff(pma.release());
  return manage(res);
}

basic_set_list set::get_basic_set_list() const
{
  auto res = isl_set_get_basic_set_list(get());
  return manage(res);
}

id set::get_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_set_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

std::string set::get_dim_name(isl::dim type, unsigned int pos) const
{
  auto res = isl_set_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

space set::get_space() const
{
  auto res = isl_set_get_space(get());
  return manage(res);
}

val set::get_stride(int pos) const
{
  auto res = isl_set_get_stride(get(), pos);
  return manage(res);
}

id set::get_tuple_id() const
{
  auto res = isl_set_get_tuple_id(get());
  return manage(res);
}

std::string set::get_tuple_name() const
{
  auto res = isl_set_get_tuple_name(get());
  std::string tmp(res);
  return tmp;
}

set set::gist(set context) const
{
  auto res = isl_set_gist(copy(), context.release());
  return manage(res);
}

set set::gist_basic_set(basic_set context) const
{
  auto res = isl_set_gist_basic_set(copy(), context.release());
  return manage(res);
}

set set::gist_params(set context) const
{
  auto res = isl_set_gist_params(copy(), context.release());
  return manage(res);
}

boolean set::has_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_set_has_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

boolean set::has_dim_name(isl::dim type, unsigned int pos) const
{
  auto res = isl_set_has_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

boolean set::has_equal_space(const set &set2) const
{
  auto res = isl_set_has_equal_space(get(), set2.get());
  return manage(res);
}

boolean set::has_tuple_id() const
{
  auto res = isl_set_has_tuple_id(get());
  return manage(res);
}

boolean set::has_tuple_name() const
{
  auto res = isl_set_has_tuple_name(get());
  return manage(res);
}

map set::identity() const
{
  auto res = isl_set_identity(copy());
  return manage(res);
}

pw_aff set::indicator_function() const
{
  auto res = isl_set_indicator_function(copy());
  return manage(res);
}

set set::insert_dims(isl::dim type, unsigned int pos, unsigned int n) const
{
  auto res = isl_set_insert_dims(copy(), static_cast<enum isl_dim_type>(type), pos, n);
  return manage(res);
}

set set::intersect(set set2) const
{
  auto res = isl_set_intersect(copy(), set2.release());
  return manage(res);
}

set set::intersect_params(set params) const
{
  auto res = isl_set_intersect_params(copy(), params.release());
  return manage(res);
}

boolean set::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean set::is_bounded() const
{
  auto res = isl_set_is_bounded(get());
  return manage(res);
}

boolean set::is_box() const
{
  auto res = isl_set_is_box(get());
  return manage(res);
}

boolean set::is_disjoint(const set &set2) const
{
  auto res = isl_set_is_disjoint(get(), set2.get());
  return manage(res);
}

boolean set::is_empty() const
{
  auto res = isl_set_is_empty(get());
  return manage(res);
}

boolean set::is_equal(const set &set2) const
{
  auto res = isl_set_is_equal(get(), set2.get());
  return manage(res);
}

boolean set::is_params() const
{
  auto res = isl_set_is_params(get());
  return manage(res);
}

boolean set::is_singleton() const
{
  auto res = isl_set_is_singleton(get());
  return manage(res);
}

boolean set::is_strict_subset(const set &set2) const
{
  auto res = isl_set_is_strict_subset(get(), set2.get());
  return manage(res);
}

boolean set::is_subset(const set &set2) const
{
  auto res = isl_set_is_subset(get(), set2.get());
  return manage(res);
}

boolean set::is_wrapping() const
{
  auto res = isl_set_is_wrapping(get());
  return manage(res);
}

map set::lex_ge_set(set set2) const
{
  auto res = isl_set_lex_ge_set(copy(), set2.release());
  return manage(res);
}

map set::lex_gt_set(set set2) const
{
  auto res = isl_set_lex_gt_set(copy(), set2.release());
  return manage(res);
}

map set::lex_le_set(set set2) const
{
  auto res = isl_set_lex_le_set(copy(), set2.release());
  return manage(res);
}

map set::lex_lt_set(set set2) const
{
  auto res = isl_set_lex_lt_set(copy(), set2.release());
  return manage(res);
}

set set::lexmax() const
{
  auto res = isl_set_lexmax(copy());
  return manage(res);
}

pw_multi_aff set::lexmax_pw_multi_aff() const
{
  auto res = isl_set_lexmax_pw_multi_aff(copy());
  return manage(res);
}

set set::lexmin() const
{
  auto res = isl_set_lexmin(copy());
  return manage(res);
}

pw_multi_aff set::lexmin_pw_multi_aff() const
{
  auto res = isl_set_lexmin_pw_multi_aff(copy());
  return manage(res);
}

set set::lower_bound_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_set_lower_bound_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

set set::lower_bound_val(isl::dim type, unsigned int pos, val value) const
{
  auto res = isl_set_lower_bound_val(copy(), static_cast<enum isl_dim_type>(type), pos, value.release());
  return manage(res);
}

val set::max_val(const aff &obj) const
{
  auto res = isl_set_max_val(get(), obj.get());
  return manage(res);
}

val set::min_val(const aff &obj) const
{
  auto res = isl_set_min_val(get(), obj.get());
  return manage(res);
}

set set::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_set_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

int set::n_basic_set() const
{
  auto res = isl_set_n_basic_set(get());
  return res;
}

unsigned int set::n_dim() const
{
  auto res = isl_set_n_dim(get());
  return res;
}

set set::nat_universe(space dim)
{
  auto res = isl_set_nat_universe(dim.release());
  return manage(res);
}

set set::neg() const
{
  auto res = isl_set_neg(copy());
  return manage(res);
}

set set::params() const
{
  auto res = isl_set_params(copy());
  return manage(res);
}

int set::plain_cmp(const set &set2) const
{
  auto res = isl_set_plain_cmp(get(), set2.get());
  return res;
}

val set::plain_get_val_if_fixed(isl::dim type, unsigned int pos) const
{
  auto res = isl_set_plain_get_val_if_fixed(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

boolean set::plain_is_disjoint(const set &set2) const
{
  auto res = isl_set_plain_is_disjoint(get(), set2.get());
  return manage(res);
}

boolean set::plain_is_empty() const
{
  auto res = isl_set_plain_is_empty(get());
  return manage(res);
}

boolean set::plain_is_equal(const set &set2) const
{
  auto res = isl_set_plain_is_equal(get(), set2.get());
  return manage(res);
}

boolean set::plain_is_universe() const
{
  auto res = isl_set_plain_is_universe(get());
  return manage(res);
}

basic_set set::plain_unshifted_simple_hull() const
{
  auto res = isl_set_plain_unshifted_simple_hull(copy());
  return manage(res);
}

basic_set set::polyhedral_hull() const
{
  auto res = isl_set_polyhedral_hull(copy());
  return manage(res);
}

set set::preimage_multi_aff(multi_aff ma) const
{
  auto res = isl_set_preimage_multi_aff(copy(), ma.release());
  return manage(res);
}

set set::preimage_multi_pw_aff(multi_pw_aff mpa) const
{
  auto res = isl_set_preimage_multi_pw_aff(copy(), mpa.release());
  return manage(res);
}

set set::preimage_pw_multi_aff(pw_multi_aff pma) const
{
  auto res = isl_set_preimage_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

set set::product(set set2) const
{
  auto res = isl_set_product(copy(), set2.release());
  return manage(res);
}

map set::project_onto_map(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_project_onto_map(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

set set::project_out(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

set set::remove_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_remove_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

set set::remove_divs() const
{
  auto res = isl_set_remove_divs(copy());
  return manage(res);
}

set set::remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_remove_divs_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

set set::remove_redundancies() const
{
  auto res = isl_set_remove_redundancies(copy());
  return manage(res);
}

set set::remove_unknown_divs() const
{
  auto res = isl_set_remove_unknown_divs(copy());
  return manage(res);
}

set set::reset_space(space dim) const
{
  auto res = isl_set_reset_space(copy(), dim.release());
  return manage(res);
}

set set::reset_tuple_id() const
{
  auto res = isl_set_reset_tuple_id(copy());
  return manage(res);
}

set set::reset_user() const
{
  auto res = isl_set_reset_user(copy());
  return manage(res);
}

basic_set set::sample() const
{
  auto res = isl_set_sample(copy());
  return manage(res);
}

point set::sample_point() const
{
  auto res = isl_set_sample_point(copy());
  return manage(res);
}

set set::set_dim_id(isl::dim type, unsigned int pos, id id) const
{
  auto res = isl_set_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

set set::set_tuple_id(id id) const
{
  auto res = isl_set_set_tuple_id(copy(), id.release());
  return manage(res);
}

set set::set_tuple_name(const std::string &s) const
{
  auto res = isl_set_set_tuple_name(copy(), s.c_str());
  return manage(res);
}

basic_set set::simple_hull() const
{
  auto res = isl_set_simple_hull(copy());
  return manage(res);
}

int set::size() const
{
  auto res = isl_set_size(get());
  return res;
}

basic_set set::solutions() const
{
  auto res = isl_set_solutions(copy());
  return manage(res);
}

set set::split_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_split_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

set set::subtract(set set2) const
{
  auto res = isl_set_subtract(copy(), set2.release());
  return manage(res);
}

set set::sum(set set2) const
{
  auto res = isl_set_sum(copy(), set2.release());
  return manage(res);
}

set set::unite(set set2) const
{
  auto res = isl_set_union(copy(), set2.release());
  return manage(res);
}

set set::universe(space space)
{
  auto res = isl_set_universe(space.release());
  return manage(res);
}

basic_set set::unshifted_simple_hull() const
{
  auto res = isl_set_unshifted_simple_hull(copy());
  return manage(res);
}

basic_set set::unshifted_simple_hull_from_set_list(set_list list) const
{
  auto res = isl_set_unshifted_simple_hull_from_set_list(copy(), list.release());
  return manage(res);
}

map set::unwrap() const
{
  auto res = isl_set_unwrap(copy());
  return manage(res);
}

set set::upper_bound_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_set_upper_bound_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

set set::upper_bound_val(isl::dim type, unsigned int pos, val value) const
{
  auto res = isl_set_upper_bound_val(copy(), static_cast<enum isl_dim_type>(type), pos, value.release());
  return manage(res);
}

map set::wrapped_domain_map() const
{
  auto res = isl_set_wrapped_domain_map(copy());
  return manage(res);
}

// implementations for isl::set_list
set_list manage(__isl_take isl_set_list *ptr) {
  return set_list(ptr);
}
set_list manage_copy(__isl_keep isl_set_list *ptr) {
  ptr = isl_set_list_copy(ptr);
  return set_list(ptr);
}

set_list::set_list()
    : ptr(nullptr) {}

set_list::set_list(const set_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
set_list::set_list(std::nullptr_t)
    : ptr(nullptr) {}


set_list::set_list(__isl_take isl_set_list *ptr)
    : ptr(ptr) {}


set_list &set_list::operator=(set_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

set_list::~set_list() {
  if (ptr)
    isl_set_list_free(ptr);
}

__isl_give isl_set_list *set_list::copy() const & {
  return isl_set_list_copy(ptr);
}

__isl_keep isl_set_list *set_list::get() const {
  return ptr;
}

__isl_give isl_set_list *set_list::release() {
  isl_set_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool set_list::is_null() const {
  return ptr == nullptr;
}
set_list::operator bool() const {
  return !is_null();
}


ctx set_list::get_ctx() const {
  return ctx(isl_set_list_get_ctx(ptr));
}

void set_list::dump() const {
  isl_set_list_dump(get());
}


set_list set_list::add(set el) const
{
  auto res = isl_set_list_add(copy(), el.release());
  return manage(res);
}

set_list set_list::alloc(ctx ctx, int n)
{
  auto res = isl_set_list_alloc(ctx.release(), n);
  return manage(res);
}

set_list set_list::concat(set_list list2) const
{
  auto res = isl_set_list_concat(copy(), list2.release());
  return manage(res);
}

set_list set_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_set_list_drop(copy(), first, n);
  return manage(res);
}

stat set_list::foreach(const std::function<stat(set)> &fn) const
{
  struct fn_data {
    const std::function<stat(set)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_set *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_set_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

set_list set_list::from_set(set el)
{
  auto res = isl_set_list_from_set(el.release());
  return manage(res);
}

set set_list::get_at(int index) const
{
  auto res = isl_set_list_get_at(get(), index);
  return manage(res);
}

set set_list::get_set(int index) const
{
  auto res = isl_set_list_get_set(get(), index);
  return manage(res);
}

set_list set_list::insert(unsigned int pos, set el) const
{
  auto res = isl_set_list_insert(copy(), pos, el.release());
  return manage(res);
}

int set_list::n_set() const
{
  auto res = isl_set_list_n_set(get());
  return res;
}

set_list set_list::reverse() const
{
  auto res = isl_set_list_reverse(copy());
  return manage(res);
}

set_list set_list::set_set(int index, set el) const
{
  auto res = isl_set_list_set_set(copy(), index, el.release());
  return manage(res);
}

int set_list::size() const
{
  auto res = isl_set_list_size(get());
  return res;
}

set_list set_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_set_list_swap(copy(), pos1, pos2);
  return manage(res);
}

set set_list::unite() const
{
  auto res = isl_set_list_union(copy());
  return manage(res);
}

// implementations for isl::space
space manage(__isl_take isl_space *ptr) {
  return space(ptr);
}
space manage_copy(__isl_keep isl_space *ptr) {
  ptr = isl_space_copy(ptr);
  return space(ptr);
}

space::space()
    : ptr(nullptr) {}

space::space(const space &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
space::space(std::nullptr_t)
    : ptr(nullptr) {}


space::space(__isl_take isl_space *ptr)
    : ptr(ptr) {}

space::space(ctx ctx, unsigned int nparam, unsigned int n_in, unsigned int n_out)
{
  auto res = isl_space_alloc(ctx.release(), nparam, n_in, n_out);
  ptr = res;
}
space::space(ctx ctx, unsigned int nparam, unsigned int dim)
{
  auto res = isl_space_set_alloc(ctx.release(), nparam, dim);
  ptr = res;
}

space &space::operator=(space obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

space::~space() {
  if (ptr)
    isl_space_free(ptr);
}

__isl_give isl_space *space::copy() const & {
  return isl_space_copy(ptr);
}

__isl_keep isl_space *space::get() const {
  return ptr;
}

__isl_give isl_space *space::release() {
  isl_space *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool space::is_null() const {
  return ptr == nullptr;
}
space::operator bool() const {
  return !is_null();
}


ctx space::get_ctx() const {
  return ctx(isl_space_get_ctx(ptr));
}
std::string space::to_str() const {
  char *Tmp = isl_space_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void space::dump() const {
  isl_space_dump(get());
}


space space::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_space_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

space space::add_param_id(id id) const
{
  auto res = isl_space_add_param_id(copy(), id.release());
  return manage(res);
}

space space::align_params(space dim2) const
{
  auto res = isl_space_align_params(copy(), dim2.release());
  return manage(res);
}

boolean space::can_curry() const
{
  auto res = isl_space_can_curry(get());
  return manage(res);
}

boolean space::can_range_curry() const
{
  auto res = isl_space_can_range_curry(get());
  return manage(res);
}

boolean space::can_uncurry() const
{
  auto res = isl_space_can_uncurry(get());
  return manage(res);
}

boolean space::can_zip() const
{
  auto res = isl_space_can_zip(get());
  return manage(res);
}

space space::curry() const
{
  auto res = isl_space_curry(copy());
  return manage(res);
}

unsigned int space::dim(isl::dim type) const
{
  auto res = isl_space_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

space space::domain() const
{
  auto res = isl_space_domain(copy());
  return manage(res);
}

space space::domain_factor_domain() const
{
  auto res = isl_space_domain_factor_domain(copy());
  return manage(res);
}

space space::domain_factor_range() const
{
  auto res = isl_space_domain_factor_range(copy());
  return manage(res);
}

boolean space::domain_is_wrapping() const
{
  auto res = isl_space_domain_is_wrapping(get());
  return manage(res);
}

space space::domain_map() const
{
  auto res = isl_space_domain_map(copy());
  return manage(res);
}

space space::domain_product(space right) const
{
  auto res = isl_space_domain_product(copy(), right.release());
  return manage(res);
}

space space::drop_dims(isl::dim type, unsigned int first, unsigned int num) const
{
  auto res = isl_space_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, num);
  return manage(res);
}

space space::factor_domain() const
{
  auto res = isl_space_factor_domain(copy());
  return manage(res);
}

space space::factor_range() const
{
  auto res = isl_space_factor_range(copy());
  return manage(res);
}

int space::find_dim_by_id(isl::dim type, const id &id) const
{
  auto res = isl_space_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int space::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_space_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

space space::flatten_domain() const
{
  auto res = isl_space_flatten_domain(copy());
  return manage(res);
}

space space::flatten_range() const
{
  auto res = isl_space_flatten_range(copy());
  return manage(res);
}

space space::from_domain() const
{
  auto res = isl_space_from_domain(copy());
  return manage(res);
}

space space::from_range() const
{
  auto res = isl_space_from_range(copy());
  return manage(res);
}

id space::get_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_space_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

std::string space::get_dim_name(isl::dim type, unsigned int pos) const
{
  auto res = isl_space_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

id space::get_tuple_id(isl::dim type) const
{
  auto res = isl_space_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

std::string space::get_tuple_name(isl::dim type) const
{
  auto res = isl_space_get_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  std::string tmp(res);
  return tmp;
}

boolean space::has_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_space_has_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

boolean space::has_dim_name(isl::dim type, unsigned int pos) const
{
  auto res = isl_space_has_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

boolean space::has_equal_params(const space &space2) const
{
  auto res = isl_space_has_equal_params(get(), space2.get());
  return manage(res);
}

boolean space::has_equal_tuples(const space &space2) const
{
  auto res = isl_space_has_equal_tuples(get(), space2.get());
  return manage(res);
}

boolean space::has_tuple_id(isl::dim type) const
{
  auto res = isl_space_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

boolean space::has_tuple_name(isl::dim type) const
{
  auto res = isl_space_has_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

space space::insert_dims(isl::dim type, unsigned int pos, unsigned int n) const
{
  auto res = isl_space_insert_dims(copy(), static_cast<enum isl_dim_type>(type), pos, n);
  return manage(res);
}

boolean space::is_domain(const space &space2) const
{
  auto res = isl_space_is_domain(get(), space2.get());
  return manage(res);
}

boolean space::is_equal(const space &space2) const
{
  auto res = isl_space_is_equal(get(), space2.get());
  return manage(res);
}

boolean space::is_map() const
{
  auto res = isl_space_is_map(get());
  return manage(res);
}

boolean space::is_params() const
{
  auto res = isl_space_is_params(get());
  return manage(res);
}

boolean space::is_product() const
{
  auto res = isl_space_is_product(get());
  return manage(res);
}

boolean space::is_range(const space &space2) const
{
  auto res = isl_space_is_range(get(), space2.get());
  return manage(res);
}

boolean space::is_set() const
{
  auto res = isl_space_is_set(get());
  return manage(res);
}

boolean space::is_wrapping() const
{
  auto res = isl_space_is_wrapping(get());
  return manage(res);
}

space space::join(space right) const
{
  auto res = isl_space_join(copy(), right.release());
  return manage(res);
}

space space::map_from_domain_and_range(space range) const
{
  auto res = isl_space_map_from_domain_and_range(copy(), range.release());
  return manage(res);
}

space space::map_from_set() const
{
  auto res = isl_space_map_from_set(copy());
  return manage(res);
}

space space::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_space_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

space space::params() const
{
  auto res = isl_space_params(copy());
  return manage(res);
}

space space::params_alloc(ctx ctx, unsigned int nparam)
{
  auto res = isl_space_params_alloc(ctx.release(), nparam);
  return manage(res);
}

space space::product(space right) const
{
  auto res = isl_space_product(copy(), right.release());
  return manage(res);
}

space space::range() const
{
  auto res = isl_space_range(copy());
  return manage(res);
}

space space::range_curry() const
{
  auto res = isl_space_range_curry(copy());
  return manage(res);
}

space space::range_factor_domain() const
{
  auto res = isl_space_range_factor_domain(copy());
  return manage(res);
}

space space::range_factor_range() const
{
  auto res = isl_space_range_factor_range(copy());
  return manage(res);
}

boolean space::range_is_wrapping() const
{
  auto res = isl_space_range_is_wrapping(get());
  return manage(res);
}

space space::range_map() const
{
  auto res = isl_space_range_map(copy());
  return manage(res);
}

space space::range_product(space right) const
{
  auto res = isl_space_range_product(copy(), right.release());
  return manage(res);
}

space space::reset_tuple_id(isl::dim type) const
{
  auto res = isl_space_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

space space::reset_user() const
{
  auto res = isl_space_reset_user(copy());
  return manage(res);
}

space space::reverse() const
{
  auto res = isl_space_reverse(copy());
  return manage(res);
}

space space::set_dim_id(isl::dim type, unsigned int pos, id id) const
{
  auto res = isl_space_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

space space::set_from_params() const
{
  auto res = isl_space_set_from_params(copy());
  return manage(res);
}

space space::set_tuple_id(isl::dim type, id id) const
{
  auto res = isl_space_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

space space::set_tuple_name(isl::dim type, const std::string &s) const
{
  auto res = isl_space_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

boolean space::tuple_is_equal(isl::dim type1, const space &space2, isl::dim type2) const
{
  auto res = isl_space_tuple_is_equal(get(), static_cast<enum isl_dim_type>(type1), space2.get(), static_cast<enum isl_dim_type>(type2));
  return manage(res);
}

space space::uncurry() const
{
  auto res = isl_space_uncurry(copy());
  return manage(res);
}

space space::unwrap() const
{
  auto res = isl_space_unwrap(copy());
  return manage(res);
}

space space::wrap() const
{
  auto res = isl_space_wrap(copy());
  return manage(res);
}

space space::zip() const
{
  auto res = isl_space_zip(copy());
  return manage(res);
}

// implementations for isl::term
term manage(__isl_take isl_term *ptr) {
  return term(ptr);
}
term manage_copy(__isl_keep isl_term *ptr) {
  ptr = isl_term_copy(ptr);
  return term(ptr);
}

term::term()
    : ptr(nullptr) {}

term::term(const term &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
term::term(std::nullptr_t)
    : ptr(nullptr) {}


term::term(__isl_take isl_term *ptr)
    : ptr(ptr) {}


term &term::operator=(term obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

term::~term() {
  if (ptr)
    isl_term_free(ptr);
}

__isl_give isl_term *term::copy() const & {
  return isl_term_copy(ptr);
}

__isl_keep isl_term *term::get() const {
  return ptr;
}

__isl_give isl_term *term::release() {
  isl_term *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool term::is_null() const {
  return ptr == nullptr;
}
term::operator bool() const {
  return !is_null();
}


ctx term::get_ctx() const {
  return ctx(isl_term_get_ctx(ptr));
}


unsigned int term::dim(isl::dim type) const
{
  auto res = isl_term_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

val term::get_coefficient_val() const
{
  auto res = isl_term_get_coefficient_val(get());
  return manage(res);
}

aff term::get_div(unsigned int pos) const
{
  auto res = isl_term_get_div(get(), pos);
  return manage(res);
}

int term::get_exp(isl::dim type, unsigned int pos) const
{
  auto res = isl_term_get_exp(get(), static_cast<enum isl_dim_type>(type), pos);
  return res;
}

// implementations for isl::union_access_info
union_access_info manage(__isl_take isl_union_access_info *ptr) {
  return union_access_info(ptr);
}
union_access_info manage_copy(__isl_keep isl_union_access_info *ptr) {
  ptr = isl_union_access_info_copy(ptr);
  return union_access_info(ptr);
}

union_access_info::union_access_info()
    : ptr(nullptr) {}

union_access_info::union_access_info(const union_access_info &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
union_access_info::union_access_info(std::nullptr_t)
    : ptr(nullptr) {}


union_access_info::union_access_info(__isl_take isl_union_access_info *ptr)
    : ptr(ptr) {}

union_access_info::union_access_info(union_map sink)
{
  auto res = isl_union_access_info_from_sink(sink.release());
  ptr = res;
}

union_access_info &union_access_info::operator=(union_access_info obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_access_info::~union_access_info() {
  if (ptr)
    isl_union_access_info_free(ptr);
}

__isl_give isl_union_access_info *union_access_info::copy() const & {
  return isl_union_access_info_copy(ptr);
}

__isl_keep isl_union_access_info *union_access_info::get() const {
  return ptr;
}

__isl_give isl_union_access_info *union_access_info::release() {
  isl_union_access_info *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_access_info::is_null() const {
  return ptr == nullptr;
}
union_access_info::operator bool() const {
  return !is_null();
}


ctx union_access_info::get_ctx() const {
  return ctx(isl_union_access_info_get_ctx(ptr));
}
std::string union_access_info::to_str() const {
  char *Tmp = isl_union_access_info_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}



union_flow union_access_info::compute_flow() const
{
  auto res = isl_union_access_info_compute_flow(copy());
  return manage(res);
}

union_access_info union_access_info::set_kill(union_map kill) const
{
  auto res = isl_union_access_info_set_kill(copy(), kill.release());
  return manage(res);
}

union_access_info union_access_info::set_may_source(union_map may_source) const
{
  auto res = isl_union_access_info_set_may_source(copy(), may_source.release());
  return manage(res);
}

union_access_info union_access_info::set_must_source(union_map must_source) const
{
  auto res = isl_union_access_info_set_must_source(copy(), must_source.release());
  return manage(res);
}

union_access_info union_access_info::set_schedule(schedule schedule) const
{
  auto res = isl_union_access_info_set_schedule(copy(), schedule.release());
  return manage(res);
}

union_access_info union_access_info::set_schedule_map(union_map schedule_map) const
{
  auto res = isl_union_access_info_set_schedule_map(copy(), schedule_map.release());
  return manage(res);
}

// implementations for isl::union_flow
union_flow manage(__isl_take isl_union_flow *ptr) {
  return union_flow(ptr);
}
union_flow manage_copy(__isl_keep isl_union_flow *ptr) {
  ptr = isl_union_flow_copy(ptr);
  return union_flow(ptr);
}

union_flow::union_flow()
    : ptr(nullptr) {}

union_flow::union_flow(const union_flow &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
union_flow::union_flow(std::nullptr_t)
    : ptr(nullptr) {}


union_flow::union_flow(__isl_take isl_union_flow *ptr)
    : ptr(ptr) {}


union_flow &union_flow::operator=(union_flow obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_flow::~union_flow() {
  if (ptr)
    isl_union_flow_free(ptr);
}

__isl_give isl_union_flow *union_flow::copy() const & {
  return isl_union_flow_copy(ptr);
}

__isl_keep isl_union_flow *union_flow::get() const {
  return ptr;
}

__isl_give isl_union_flow *union_flow::release() {
  isl_union_flow *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_flow::is_null() const {
  return ptr == nullptr;
}
union_flow::operator bool() const {
  return !is_null();
}


ctx union_flow::get_ctx() const {
  return ctx(isl_union_flow_get_ctx(ptr));
}
std::string union_flow::to_str() const {
  char *Tmp = isl_union_flow_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}



union_map union_flow::get_full_may_dependence() const
{
  auto res = isl_union_flow_get_full_may_dependence(get());
  return manage(res);
}

union_map union_flow::get_full_must_dependence() const
{
  auto res = isl_union_flow_get_full_must_dependence(get());
  return manage(res);
}

union_map union_flow::get_may_dependence() const
{
  auto res = isl_union_flow_get_may_dependence(get());
  return manage(res);
}

union_map union_flow::get_may_no_source() const
{
  auto res = isl_union_flow_get_may_no_source(get());
  return manage(res);
}

union_map union_flow::get_must_dependence() const
{
  auto res = isl_union_flow_get_must_dependence(get());
  return manage(res);
}

union_map union_flow::get_must_no_source() const
{
  auto res = isl_union_flow_get_must_no_source(get());
  return manage(res);
}

// implementations for isl::union_map
union_map manage(__isl_take isl_union_map *ptr) {
  return union_map(ptr);
}
union_map manage_copy(__isl_keep isl_union_map *ptr) {
  ptr = isl_union_map_copy(ptr);
  return union_map(ptr);
}

union_map::union_map()
    : ptr(nullptr) {}

union_map::union_map(const union_map &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
union_map::union_map(std::nullptr_t)
    : ptr(nullptr) {}


union_map::union_map(__isl_take isl_union_map *ptr)
    : ptr(ptr) {}

union_map::union_map(union_pw_multi_aff upma)
{
  auto res = isl_union_map_from_union_pw_multi_aff(upma.release());
  ptr = res;
}
union_map::union_map(basic_map bmap)
{
  auto res = isl_union_map_from_basic_map(bmap.release());
  ptr = res;
}
union_map::union_map(map map)
{
  auto res = isl_union_map_from_map(map.release());
  ptr = res;
}
union_map::union_map(ctx ctx, const std::string &str)
{
  auto res = isl_union_map_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

union_map &union_map::operator=(union_map obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_map::~union_map() {
  if (ptr)
    isl_union_map_free(ptr);
}

__isl_give isl_union_map *union_map::copy() const & {
  return isl_union_map_copy(ptr);
}

__isl_keep isl_union_map *union_map::get() const {
  return ptr;
}

__isl_give isl_union_map *union_map::release() {
  isl_union_map *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_map::is_null() const {
  return ptr == nullptr;
}
union_map::operator bool() const {
  return !is_null();
}


ctx union_map::get_ctx() const {
  return ctx(isl_union_map_get_ctx(ptr));
}
std::string union_map::to_str() const {
  char *Tmp = isl_union_map_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void union_map::dump() const {
  isl_union_map_dump(get());
}


union_map union_map::add_map(map map) const
{
  auto res = isl_union_map_add_map(copy(), map.release());
  return manage(res);
}

union_map union_map::affine_hull() const
{
  auto res = isl_union_map_affine_hull(copy());
  return manage(res);
}

union_map union_map::align_params(space model) const
{
  auto res = isl_union_map_align_params(copy(), model.release());
  return manage(res);
}

union_map union_map::apply_domain(union_map umap2) const
{
  auto res = isl_union_map_apply_domain(copy(), umap2.release());
  return manage(res);
}

union_map union_map::apply_range(union_map umap2) const
{
  auto res = isl_union_map_apply_range(copy(), umap2.release());
  return manage(res);
}

union_map union_map::coalesce() const
{
  auto res = isl_union_map_coalesce(copy());
  return manage(res);
}

boolean union_map::contains(const space &space) const
{
  auto res = isl_union_map_contains(get(), space.get());
  return manage(res);
}

union_map union_map::curry() const
{
  auto res = isl_union_map_curry(copy());
  return manage(res);
}

union_set union_map::deltas() const
{
  auto res = isl_union_map_deltas(copy());
  return manage(res);
}

union_map union_map::deltas_map() const
{
  auto res = isl_union_map_deltas_map(copy());
  return manage(res);
}

union_map union_map::detect_equalities() const
{
  auto res = isl_union_map_detect_equalities(copy());
  return manage(res);
}

unsigned int union_map::dim(isl::dim type) const
{
  auto res = isl_union_map_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

union_set union_map::domain() const
{
  auto res = isl_union_map_domain(copy());
  return manage(res);
}

union_map union_map::domain_factor_domain() const
{
  auto res = isl_union_map_domain_factor_domain(copy());
  return manage(res);
}

union_map union_map::domain_factor_range() const
{
  auto res = isl_union_map_domain_factor_range(copy());
  return manage(res);
}

union_map union_map::domain_map() const
{
  auto res = isl_union_map_domain_map(copy());
  return manage(res);
}

union_pw_multi_aff union_map::domain_map_union_pw_multi_aff() const
{
  auto res = isl_union_map_domain_map_union_pw_multi_aff(copy());
  return manage(res);
}

union_map union_map::domain_product(union_map umap2) const
{
  auto res = isl_union_map_domain_product(copy(), umap2.release());
  return manage(res);
}

union_map union_map::empty(space space)
{
  auto res = isl_union_map_empty(space.release());
  return manage(res);
}

union_map union_map::eq_at(multi_union_pw_aff mupa) const
{
  auto res = isl_union_map_eq_at_multi_union_pw_aff(copy(), mupa.release());
  return manage(res);
}

map union_map::extract_map(space dim) const
{
  auto res = isl_union_map_extract_map(get(), dim.release());
  return manage(res);
}

union_map union_map::factor_domain() const
{
  auto res = isl_union_map_factor_domain(copy());
  return manage(res);
}

union_map union_map::factor_range() const
{
  auto res = isl_union_map_factor_range(copy());
  return manage(res);
}

int union_map::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_union_map_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

union_map union_map::fixed_power(val exp) const
{
  auto res = isl_union_map_fixed_power_val(copy(), exp.release());
  return manage(res);
}

union_map union_map::flat_domain_product(union_map umap2) const
{
  auto res = isl_union_map_flat_domain_product(copy(), umap2.release());
  return manage(res);
}

union_map union_map::flat_range_product(union_map umap2) const
{
  auto res = isl_union_map_flat_range_product(copy(), umap2.release());
  return manage(res);
}

stat union_map::foreach_map(const std::function<stat(map)> &fn) const
{
  struct fn_data {
    const std::function<stat(map)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_map *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_union_map_foreach_map(get(), fn_lambda, &fn_data);
  return manage(res);
}

union_map union_map::from(multi_union_pw_aff mupa)
{
  auto res = isl_union_map_from_multi_union_pw_aff(mupa.release());
  return manage(res);
}

union_map union_map::from_domain(union_set uset)
{
  auto res = isl_union_map_from_domain(uset.release());
  return manage(res);
}

union_map union_map::from_domain_and_range(union_set domain, union_set range)
{
  auto res = isl_union_map_from_domain_and_range(domain.release(), range.release());
  return manage(res);
}

union_map union_map::from_range(union_set uset)
{
  auto res = isl_union_map_from_range(uset.release());
  return manage(res);
}

union_map union_map::from_union_pw_aff(union_pw_aff upa)
{
  auto res = isl_union_map_from_union_pw_aff(upa.release());
  return manage(res);
}

id union_map::get_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_union_map_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

uint32_t union_map::get_hash() const
{
  auto res = isl_union_map_get_hash(get());
  return res;
}

map_list union_map::get_map_list() const
{
  auto res = isl_union_map_get_map_list(get());
  return manage(res);
}

space union_map::get_space() const
{
  auto res = isl_union_map_get_space(get());
  return manage(res);
}

union_map union_map::gist(union_map context) const
{
  auto res = isl_union_map_gist(copy(), context.release());
  return manage(res);
}

union_map union_map::gist_domain(union_set uset) const
{
  auto res = isl_union_map_gist_domain(copy(), uset.release());
  return manage(res);
}

union_map union_map::gist_params(set set) const
{
  auto res = isl_union_map_gist_params(copy(), set.release());
  return manage(res);
}

union_map union_map::gist_range(union_set uset) const
{
  auto res = isl_union_map_gist_range(copy(), uset.release());
  return manage(res);
}

union_map union_map::intersect(union_map umap2) const
{
  auto res = isl_union_map_intersect(copy(), umap2.release());
  return manage(res);
}

union_map union_map::intersect_domain(union_set uset) const
{
  auto res = isl_union_map_intersect_domain(copy(), uset.release());
  return manage(res);
}

union_map union_map::intersect_params(set set) const
{
  auto res = isl_union_map_intersect_params(copy(), set.release());
  return manage(res);
}

union_map union_map::intersect_range(union_set uset) const
{
  auto res = isl_union_map_intersect_range(copy(), uset.release());
  return manage(res);
}

union_map union_map::intersect_range_factor_range(union_map factor) const
{
  auto res = isl_union_map_intersect_range_factor_range(copy(), factor.release());
  return manage(res);
}

boolean union_map::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_union_map_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean union_map::is_bijective() const
{
  auto res = isl_union_map_is_bijective(get());
  return manage(res);
}

boolean union_map::is_disjoint(const union_map &umap2) const
{
  auto res = isl_union_map_is_disjoint(get(), umap2.get());
  return manage(res);
}

boolean union_map::is_empty() const
{
  auto res = isl_union_map_is_empty(get());
  return manage(res);
}

boolean union_map::is_equal(const union_map &umap2) const
{
  auto res = isl_union_map_is_equal(get(), umap2.get());
  return manage(res);
}

boolean union_map::is_identity() const
{
  auto res = isl_union_map_is_identity(get());
  return manage(res);
}

boolean union_map::is_injective() const
{
  auto res = isl_union_map_is_injective(get());
  return manage(res);
}

boolean union_map::is_single_valued() const
{
  auto res = isl_union_map_is_single_valued(get());
  return manage(res);
}

boolean union_map::is_strict_subset(const union_map &umap2) const
{
  auto res = isl_union_map_is_strict_subset(get(), umap2.get());
  return manage(res);
}

boolean union_map::is_subset(const union_map &umap2) const
{
  auto res = isl_union_map_is_subset(get(), umap2.get());
  return manage(res);
}

union_map union_map::lex_ge_union_map(union_map umap2) const
{
  auto res = isl_union_map_lex_ge_union_map(copy(), umap2.release());
  return manage(res);
}

union_map union_map::lex_gt_at_multi_union_pw_aff(multi_union_pw_aff mupa) const
{
  auto res = isl_union_map_lex_gt_at_multi_union_pw_aff(copy(), mupa.release());
  return manage(res);
}

union_map union_map::lex_gt_union_map(union_map umap2) const
{
  auto res = isl_union_map_lex_gt_union_map(copy(), umap2.release());
  return manage(res);
}

union_map union_map::lex_le_union_map(union_map umap2) const
{
  auto res = isl_union_map_lex_le_union_map(copy(), umap2.release());
  return manage(res);
}

union_map union_map::lex_lt_at_multi_union_pw_aff(multi_union_pw_aff mupa) const
{
  auto res = isl_union_map_lex_lt_at_multi_union_pw_aff(copy(), mupa.release());
  return manage(res);
}

union_map union_map::lex_lt_union_map(union_map umap2) const
{
  auto res = isl_union_map_lex_lt_union_map(copy(), umap2.release());
  return manage(res);
}

union_map union_map::lexmax() const
{
  auto res = isl_union_map_lexmax(copy());
  return manage(res);
}

union_map union_map::lexmin() const
{
  auto res = isl_union_map_lexmin(copy());
  return manage(res);
}

int union_map::n_map() const
{
  auto res = isl_union_map_n_map(get());
  return res;
}

set union_map::params() const
{
  auto res = isl_union_map_params(copy());
  return manage(res);
}

boolean union_map::plain_is_empty() const
{
  auto res = isl_union_map_plain_is_empty(get());
  return manage(res);
}

boolean union_map::plain_is_injective() const
{
  auto res = isl_union_map_plain_is_injective(get());
  return manage(res);
}

union_map union_map::polyhedral_hull() const
{
  auto res = isl_union_map_polyhedral_hull(copy());
  return manage(res);
}

union_map union_map::preimage_domain_multi_aff(multi_aff ma) const
{
  auto res = isl_union_map_preimage_domain_multi_aff(copy(), ma.release());
  return manage(res);
}

union_map union_map::preimage_domain_multi_pw_aff(multi_pw_aff mpa) const
{
  auto res = isl_union_map_preimage_domain_multi_pw_aff(copy(), mpa.release());
  return manage(res);
}

union_map union_map::preimage_domain_pw_multi_aff(pw_multi_aff pma) const
{
  auto res = isl_union_map_preimage_domain_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

union_map union_map::preimage_domain_union_pw_multi_aff(union_pw_multi_aff upma) const
{
  auto res = isl_union_map_preimage_domain_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

union_map union_map::preimage_range_multi_aff(multi_aff ma) const
{
  auto res = isl_union_map_preimage_range_multi_aff(copy(), ma.release());
  return manage(res);
}

union_map union_map::preimage_range_pw_multi_aff(pw_multi_aff pma) const
{
  auto res = isl_union_map_preimage_range_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

union_map union_map::preimage_range_union_pw_multi_aff(union_pw_multi_aff upma) const
{
  auto res = isl_union_map_preimage_range_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

union_map union_map::product(union_map umap2) const
{
  auto res = isl_union_map_product(copy(), umap2.release());
  return manage(res);
}

union_map union_map::project_out(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_union_map_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

union_map union_map::project_out_all_params() const
{
  auto res = isl_union_map_project_out_all_params(copy());
  return manage(res);
}

union_set union_map::range() const
{
  auto res = isl_union_map_range(copy());
  return manage(res);
}

union_map union_map::range_curry() const
{
  auto res = isl_union_map_range_curry(copy());
  return manage(res);
}

union_map union_map::range_factor_domain() const
{
  auto res = isl_union_map_range_factor_domain(copy());
  return manage(res);
}

union_map union_map::range_factor_range() const
{
  auto res = isl_union_map_range_factor_range(copy());
  return manage(res);
}

union_map union_map::range_map() const
{
  auto res = isl_union_map_range_map(copy());
  return manage(res);
}

union_map union_map::range_product(union_map umap2) const
{
  auto res = isl_union_map_range_product(copy(), umap2.release());
  return manage(res);
}

union_map union_map::remove_divs() const
{
  auto res = isl_union_map_remove_divs(copy());
  return manage(res);
}

union_map union_map::remove_redundancies() const
{
  auto res = isl_union_map_remove_redundancies(copy());
  return manage(res);
}

union_map union_map::reset_user() const
{
  auto res = isl_union_map_reset_user(copy());
  return manage(res);
}

union_map union_map::reverse() const
{
  auto res = isl_union_map_reverse(copy());
  return manage(res);
}

basic_map union_map::sample() const
{
  auto res = isl_union_map_sample(copy());
  return manage(res);
}

union_map union_map::simple_hull() const
{
  auto res = isl_union_map_simple_hull(copy());
  return manage(res);
}

union_map union_map::subtract(union_map umap2) const
{
  auto res = isl_union_map_subtract(copy(), umap2.release());
  return manage(res);
}

union_map union_map::subtract_domain(union_set dom) const
{
  auto res = isl_union_map_subtract_domain(copy(), dom.release());
  return manage(res);
}

union_map union_map::subtract_range(union_set dom) const
{
  auto res = isl_union_map_subtract_range(copy(), dom.release());
  return manage(res);
}

union_map union_map::uncurry() const
{
  auto res = isl_union_map_uncurry(copy());
  return manage(res);
}

union_map union_map::unite(union_map umap2) const
{
  auto res = isl_union_map_union(copy(), umap2.release());
  return manage(res);
}

union_map union_map::universe() const
{
  auto res = isl_union_map_universe(copy());
  return manage(res);
}

union_set union_map::wrap() const
{
  auto res = isl_union_map_wrap(copy());
  return manage(res);
}

union_map union_map::zip() const
{
  auto res = isl_union_map_zip(copy());
  return manage(res);
}

// implementations for isl::union_map_list
union_map_list manage(__isl_take isl_union_map_list *ptr) {
  return union_map_list(ptr);
}
union_map_list manage_copy(__isl_keep isl_union_map_list *ptr) {
  ptr = isl_union_map_list_copy(ptr);
  return union_map_list(ptr);
}

union_map_list::union_map_list()
    : ptr(nullptr) {}

union_map_list::union_map_list(const union_map_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
union_map_list::union_map_list(std::nullptr_t)
    : ptr(nullptr) {}


union_map_list::union_map_list(__isl_take isl_union_map_list *ptr)
    : ptr(ptr) {}


union_map_list &union_map_list::operator=(union_map_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_map_list::~union_map_list() {
  if (ptr)
    isl_union_map_list_free(ptr);
}

__isl_give isl_union_map_list *union_map_list::copy() const & {
  return isl_union_map_list_copy(ptr);
}

__isl_keep isl_union_map_list *union_map_list::get() const {
  return ptr;
}

__isl_give isl_union_map_list *union_map_list::release() {
  isl_union_map_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_map_list::is_null() const {
  return ptr == nullptr;
}
union_map_list::operator bool() const {
  return !is_null();
}


ctx union_map_list::get_ctx() const {
  return ctx(isl_union_map_list_get_ctx(ptr));
}

void union_map_list::dump() const {
  isl_union_map_list_dump(get());
}


union_map_list union_map_list::add(union_map el) const
{
  auto res = isl_union_map_list_add(copy(), el.release());
  return manage(res);
}

union_map_list union_map_list::alloc(ctx ctx, int n)
{
  auto res = isl_union_map_list_alloc(ctx.release(), n);
  return manage(res);
}

union_map_list union_map_list::concat(union_map_list list2) const
{
  auto res = isl_union_map_list_concat(copy(), list2.release());
  return manage(res);
}

union_map_list union_map_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_union_map_list_drop(copy(), first, n);
  return manage(res);
}

stat union_map_list::foreach(const std::function<stat(union_map)> &fn) const
{
  struct fn_data {
    const std::function<stat(union_map)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_union_map *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_union_map_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

union_map_list union_map_list::from_union_map(union_map el)
{
  auto res = isl_union_map_list_from_union_map(el.release());
  return manage(res);
}

union_map union_map_list::get_at(int index) const
{
  auto res = isl_union_map_list_get_at(get(), index);
  return manage(res);
}

union_map union_map_list::get_union_map(int index) const
{
  auto res = isl_union_map_list_get_union_map(get(), index);
  return manage(res);
}

union_map_list union_map_list::insert(unsigned int pos, union_map el) const
{
  auto res = isl_union_map_list_insert(copy(), pos, el.release());
  return manage(res);
}

int union_map_list::n_union_map() const
{
  auto res = isl_union_map_list_n_union_map(get());
  return res;
}

union_map_list union_map_list::reverse() const
{
  auto res = isl_union_map_list_reverse(copy());
  return manage(res);
}

union_map_list union_map_list::set_union_map(int index, union_map el) const
{
  auto res = isl_union_map_list_set_union_map(copy(), index, el.release());
  return manage(res);
}

int union_map_list::size() const
{
  auto res = isl_union_map_list_size(get());
  return res;
}

union_map_list union_map_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_union_map_list_swap(copy(), pos1, pos2);
  return manage(res);
}

// implementations for isl::union_pw_aff
union_pw_aff manage(__isl_take isl_union_pw_aff *ptr) {
  return union_pw_aff(ptr);
}
union_pw_aff manage_copy(__isl_keep isl_union_pw_aff *ptr) {
  ptr = isl_union_pw_aff_copy(ptr);
  return union_pw_aff(ptr);
}

union_pw_aff::union_pw_aff()
    : ptr(nullptr) {}

union_pw_aff::union_pw_aff(const union_pw_aff &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
union_pw_aff::union_pw_aff(std::nullptr_t)
    : ptr(nullptr) {}


union_pw_aff::union_pw_aff(__isl_take isl_union_pw_aff *ptr)
    : ptr(ptr) {}

union_pw_aff::union_pw_aff(pw_aff pa)
{
  auto res = isl_union_pw_aff_from_pw_aff(pa.release());
  ptr = res;
}
union_pw_aff::union_pw_aff(union_set domain, val v)
{
  auto res = isl_union_pw_aff_val_on_domain(domain.release(), v.release());
  ptr = res;
}
union_pw_aff::union_pw_aff(ctx ctx, const std::string &str)
{
  auto res = isl_union_pw_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

union_pw_aff &union_pw_aff::operator=(union_pw_aff obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_pw_aff::~union_pw_aff() {
  if (ptr)
    isl_union_pw_aff_free(ptr);
}

__isl_give isl_union_pw_aff *union_pw_aff::copy() const & {
  return isl_union_pw_aff_copy(ptr);
}

__isl_keep isl_union_pw_aff *union_pw_aff::get() const {
  return ptr;
}

__isl_give isl_union_pw_aff *union_pw_aff::release() {
  isl_union_pw_aff *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_pw_aff::is_null() const {
  return ptr == nullptr;
}
union_pw_aff::operator bool() const {
  return !is_null();
}


ctx union_pw_aff::get_ctx() const {
  return ctx(isl_union_pw_aff_get_ctx(ptr));
}
std::string union_pw_aff::to_str() const {
  char *Tmp = isl_union_pw_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void union_pw_aff::dump() const {
  isl_union_pw_aff_dump(get());
}


union_pw_aff union_pw_aff::add(union_pw_aff upa2) const
{
  auto res = isl_union_pw_aff_add(copy(), upa2.release());
  return manage(res);
}

union_pw_aff union_pw_aff::add_pw_aff(pw_aff pa) const
{
  auto res = isl_union_pw_aff_add_pw_aff(copy(), pa.release());
  return manage(res);
}

union_pw_aff union_pw_aff::aff_on_domain(union_set domain, aff aff)
{
  auto res = isl_union_pw_aff_aff_on_domain(domain.release(), aff.release());
  return manage(res);
}

union_pw_aff union_pw_aff::align_params(space model) const
{
  auto res = isl_union_pw_aff_align_params(copy(), model.release());
  return manage(res);
}

union_pw_aff union_pw_aff::coalesce() const
{
  auto res = isl_union_pw_aff_coalesce(copy());
  return manage(res);
}

unsigned int union_pw_aff::dim(isl::dim type) const
{
  auto res = isl_union_pw_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

union_set union_pw_aff::domain() const
{
  auto res = isl_union_pw_aff_domain(copy());
  return manage(res);
}

union_pw_aff union_pw_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_union_pw_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

union_pw_aff union_pw_aff::empty(space space)
{
  auto res = isl_union_pw_aff_empty(space.release());
  return manage(res);
}

pw_aff union_pw_aff::extract_pw_aff(space space) const
{
  auto res = isl_union_pw_aff_extract_pw_aff(get(), space.release());
  return manage(res);
}

int union_pw_aff::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_union_pw_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

union_pw_aff union_pw_aff::floor() const
{
  auto res = isl_union_pw_aff_floor(copy());
  return manage(res);
}

stat union_pw_aff::foreach_pw_aff(const std::function<stat(pw_aff)> &fn) const
{
  struct fn_data {
    const std::function<stat(pw_aff)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_pw_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_union_pw_aff_foreach_pw_aff(get(), fn_lambda, &fn_data);
  return manage(res);
}

pw_aff_list union_pw_aff::get_pw_aff_list() const
{
  auto res = isl_union_pw_aff_get_pw_aff_list(get());
  return manage(res);
}

space union_pw_aff::get_space() const
{
  auto res = isl_union_pw_aff_get_space(get());
  return manage(res);
}

union_pw_aff union_pw_aff::gist(union_set context) const
{
  auto res = isl_union_pw_aff_gist(copy(), context.release());
  return manage(res);
}

union_pw_aff union_pw_aff::gist_params(set context) const
{
  auto res = isl_union_pw_aff_gist_params(copy(), context.release());
  return manage(res);
}

union_pw_aff union_pw_aff::intersect_domain(union_set uset) const
{
  auto res = isl_union_pw_aff_intersect_domain(copy(), uset.release());
  return manage(res);
}

union_pw_aff union_pw_aff::intersect_params(set set) const
{
  auto res = isl_union_pw_aff_intersect_params(copy(), set.release());
  return manage(res);
}

boolean union_pw_aff::involves_nan() const
{
  auto res = isl_union_pw_aff_involves_nan(get());
  return manage(res);
}

val union_pw_aff::max_val() const
{
  auto res = isl_union_pw_aff_max_val(copy());
  return manage(res);
}

val union_pw_aff::min_val() const
{
  auto res = isl_union_pw_aff_min_val(copy());
  return manage(res);
}

union_pw_aff union_pw_aff::mod_val(val f) const
{
  auto res = isl_union_pw_aff_mod_val(copy(), f.release());
  return manage(res);
}

int union_pw_aff::n_pw_aff() const
{
  auto res = isl_union_pw_aff_n_pw_aff(get());
  return res;
}

union_pw_aff union_pw_aff::neg() const
{
  auto res = isl_union_pw_aff_neg(copy());
  return manage(res);
}

union_pw_aff union_pw_aff::param_on_domain_id(union_set domain, id id)
{
  auto res = isl_union_pw_aff_param_on_domain_id(domain.release(), id.release());
  return manage(res);
}

boolean union_pw_aff::plain_is_equal(const union_pw_aff &upa2) const
{
  auto res = isl_union_pw_aff_plain_is_equal(get(), upa2.get());
  return manage(res);
}

union_pw_aff union_pw_aff::pullback(union_pw_multi_aff upma) const
{
  auto res = isl_union_pw_aff_pullback_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

union_pw_aff union_pw_aff::pw_aff_on_domain(union_set domain, pw_aff pa)
{
  auto res = isl_union_pw_aff_pw_aff_on_domain(domain.release(), pa.release());
  return manage(res);
}

union_pw_aff union_pw_aff::reset_user() const
{
  auto res = isl_union_pw_aff_reset_user(copy());
  return manage(res);
}

union_pw_aff union_pw_aff::scale_down_val(val v) const
{
  auto res = isl_union_pw_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

union_pw_aff union_pw_aff::scale_val(val v) const
{
  auto res = isl_union_pw_aff_scale_val(copy(), v.release());
  return manage(res);
}

union_pw_aff union_pw_aff::sub(union_pw_aff upa2) const
{
  auto res = isl_union_pw_aff_sub(copy(), upa2.release());
  return manage(res);
}

union_pw_aff union_pw_aff::subtract_domain(union_set uset) const
{
  auto res = isl_union_pw_aff_subtract_domain(copy(), uset.release());
  return manage(res);
}

union_pw_aff union_pw_aff::union_add(union_pw_aff upa2) const
{
  auto res = isl_union_pw_aff_union_add(copy(), upa2.release());
  return manage(res);
}

union_set union_pw_aff::zero_union_set() const
{
  auto res = isl_union_pw_aff_zero_union_set(copy());
  return manage(res);
}

// implementations for isl::union_pw_aff_list
union_pw_aff_list manage(__isl_take isl_union_pw_aff_list *ptr) {
  return union_pw_aff_list(ptr);
}
union_pw_aff_list manage_copy(__isl_keep isl_union_pw_aff_list *ptr) {
  ptr = isl_union_pw_aff_list_copy(ptr);
  return union_pw_aff_list(ptr);
}

union_pw_aff_list::union_pw_aff_list()
    : ptr(nullptr) {}

union_pw_aff_list::union_pw_aff_list(const union_pw_aff_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
union_pw_aff_list::union_pw_aff_list(std::nullptr_t)
    : ptr(nullptr) {}


union_pw_aff_list::union_pw_aff_list(__isl_take isl_union_pw_aff_list *ptr)
    : ptr(ptr) {}


union_pw_aff_list &union_pw_aff_list::operator=(union_pw_aff_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_pw_aff_list::~union_pw_aff_list() {
  if (ptr)
    isl_union_pw_aff_list_free(ptr);
}

__isl_give isl_union_pw_aff_list *union_pw_aff_list::copy() const & {
  return isl_union_pw_aff_list_copy(ptr);
}

__isl_keep isl_union_pw_aff_list *union_pw_aff_list::get() const {
  return ptr;
}

__isl_give isl_union_pw_aff_list *union_pw_aff_list::release() {
  isl_union_pw_aff_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_pw_aff_list::is_null() const {
  return ptr == nullptr;
}
union_pw_aff_list::operator bool() const {
  return !is_null();
}


ctx union_pw_aff_list::get_ctx() const {
  return ctx(isl_union_pw_aff_list_get_ctx(ptr));
}

void union_pw_aff_list::dump() const {
  isl_union_pw_aff_list_dump(get());
}


union_pw_aff_list union_pw_aff_list::add(union_pw_aff el) const
{
  auto res = isl_union_pw_aff_list_add(copy(), el.release());
  return manage(res);
}

union_pw_aff_list union_pw_aff_list::alloc(ctx ctx, int n)
{
  auto res = isl_union_pw_aff_list_alloc(ctx.release(), n);
  return manage(res);
}

union_pw_aff_list union_pw_aff_list::concat(union_pw_aff_list list2) const
{
  auto res = isl_union_pw_aff_list_concat(copy(), list2.release());
  return manage(res);
}

union_pw_aff_list union_pw_aff_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_union_pw_aff_list_drop(copy(), first, n);
  return manage(res);
}

stat union_pw_aff_list::foreach(const std::function<stat(union_pw_aff)> &fn) const
{
  struct fn_data {
    const std::function<stat(union_pw_aff)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_union_pw_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_union_pw_aff_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

union_pw_aff_list union_pw_aff_list::from_union_pw_aff(union_pw_aff el)
{
  auto res = isl_union_pw_aff_list_from_union_pw_aff(el.release());
  return manage(res);
}

union_pw_aff union_pw_aff_list::get_at(int index) const
{
  auto res = isl_union_pw_aff_list_get_at(get(), index);
  return manage(res);
}

union_pw_aff union_pw_aff_list::get_union_pw_aff(int index) const
{
  auto res = isl_union_pw_aff_list_get_union_pw_aff(get(), index);
  return manage(res);
}

union_pw_aff_list union_pw_aff_list::insert(unsigned int pos, union_pw_aff el) const
{
  auto res = isl_union_pw_aff_list_insert(copy(), pos, el.release());
  return manage(res);
}

int union_pw_aff_list::n_union_pw_aff() const
{
  auto res = isl_union_pw_aff_list_n_union_pw_aff(get());
  return res;
}

union_pw_aff_list union_pw_aff_list::reverse() const
{
  auto res = isl_union_pw_aff_list_reverse(copy());
  return manage(res);
}

union_pw_aff_list union_pw_aff_list::set_union_pw_aff(int index, union_pw_aff el) const
{
  auto res = isl_union_pw_aff_list_set_union_pw_aff(copy(), index, el.release());
  return manage(res);
}

int union_pw_aff_list::size() const
{
  auto res = isl_union_pw_aff_list_size(get());
  return res;
}

union_pw_aff_list union_pw_aff_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_union_pw_aff_list_swap(copy(), pos1, pos2);
  return manage(res);
}

// implementations for isl::union_pw_multi_aff
union_pw_multi_aff manage(__isl_take isl_union_pw_multi_aff *ptr) {
  return union_pw_multi_aff(ptr);
}
union_pw_multi_aff manage_copy(__isl_keep isl_union_pw_multi_aff *ptr) {
  ptr = isl_union_pw_multi_aff_copy(ptr);
  return union_pw_multi_aff(ptr);
}

union_pw_multi_aff::union_pw_multi_aff()
    : ptr(nullptr) {}

union_pw_multi_aff::union_pw_multi_aff(const union_pw_multi_aff &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
union_pw_multi_aff::union_pw_multi_aff(std::nullptr_t)
    : ptr(nullptr) {}


union_pw_multi_aff::union_pw_multi_aff(__isl_take isl_union_pw_multi_aff *ptr)
    : ptr(ptr) {}

union_pw_multi_aff::union_pw_multi_aff(aff aff)
{
  auto res = isl_union_pw_multi_aff_from_aff(aff.release());
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(pw_multi_aff pma)
{
  auto res = isl_union_pw_multi_aff_from_pw_multi_aff(pma.release());
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(union_set uset)
{
  auto res = isl_union_pw_multi_aff_from_domain(uset.release());
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(union_map umap)
{
  auto res = isl_union_pw_multi_aff_from_union_map(umap.release());
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(ctx ctx, const std::string &str)
{
  auto res = isl_union_pw_multi_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(union_pw_aff upa)
{
  auto res = isl_union_pw_multi_aff_from_union_pw_aff(upa.release());
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(multi_union_pw_aff mupa)
{
  auto res = isl_union_pw_multi_aff_from_multi_union_pw_aff(mupa.release());
  ptr = res;
}

union_pw_multi_aff &union_pw_multi_aff::operator=(union_pw_multi_aff obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_pw_multi_aff::~union_pw_multi_aff() {
  if (ptr)
    isl_union_pw_multi_aff_free(ptr);
}

__isl_give isl_union_pw_multi_aff *union_pw_multi_aff::copy() const & {
  return isl_union_pw_multi_aff_copy(ptr);
}

__isl_keep isl_union_pw_multi_aff *union_pw_multi_aff::get() const {
  return ptr;
}

__isl_give isl_union_pw_multi_aff *union_pw_multi_aff::release() {
  isl_union_pw_multi_aff *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_pw_multi_aff::is_null() const {
  return ptr == nullptr;
}
union_pw_multi_aff::operator bool() const {
  return !is_null();
}


ctx union_pw_multi_aff::get_ctx() const {
  return ctx(isl_union_pw_multi_aff_get_ctx(ptr));
}
std::string union_pw_multi_aff::to_str() const {
  char *Tmp = isl_union_pw_multi_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void union_pw_multi_aff::dump() const {
  isl_union_pw_multi_aff_dump(get());
}


union_pw_multi_aff union_pw_multi_aff::add(union_pw_multi_aff upma2) const
{
  auto res = isl_union_pw_multi_aff_add(copy(), upma2.release());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::add_pw_multi_aff(pw_multi_aff pma) const
{
  auto res = isl_union_pw_multi_aff_add_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::align_params(space model) const
{
  auto res = isl_union_pw_multi_aff_align_params(copy(), model.release());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::coalesce() const
{
  auto res = isl_union_pw_multi_aff_coalesce(copy());
  return manage(res);
}

unsigned int union_pw_multi_aff::dim(isl::dim type) const
{
  auto res = isl_union_pw_multi_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

union_set union_pw_multi_aff::domain() const
{
  auto res = isl_union_pw_multi_aff_domain(copy());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_union_pw_multi_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::empty(space space)
{
  auto res = isl_union_pw_multi_aff_empty(space.release());
  return manage(res);
}

pw_multi_aff union_pw_multi_aff::extract_pw_multi_aff(space space) const
{
  auto res = isl_union_pw_multi_aff_extract_pw_multi_aff(get(), space.release());
  return manage(res);
}

int union_pw_multi_aff::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_union_pw_multi_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

union_pw_multi_aff union_pw_multi_aff::flat_range_product(union_pw_multi_aff upma2) const
{
  auto res = isl_union_pw_multi_aff_flat_range_product(copy(), upma2.release());
  return manage(res);
}

stat union_pw_multi_aff::foreach_pw_multi_aff(const std::function<stat(pw_multi_aff)> &fn) const
{
  struct fn_data {
    const std::function<stat(pw_multi_aff)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_pw_multi_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_union_pw_multi_aff_foreach_pw_multi_aff(get(), fn_lambda, &fn_data);
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::from_union_set(union_set uset)
{
  auto res = isl_union_pw_multi_aff_from_union_set(uset.release());
  return manage(res);
}

pw_multi_aff_list union_pw_multi_aff::get_pw_multi_aff_list() const
{
  auto res = isl_union_pw_multi_aff_get_pw_multi_aff_list(get());
  return manage(res);
}

space union_pw_multi_aff::get_space() const
{
  auto res = isl_union_pw_multi_aff_get_space(get());
  return manage(res);
}

union_pw_aff union_pw_multi_aff::get_union_pw_aff(int pos) const
{
  auto res = isl_union_pw_multi_aff_get_union_pw_aff(get(), pos);
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::gist(union_set context) const
{
  auto res = isl_union_pw_multi_aff_gist(copy(), context.release());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::gist_params(set context) const
{
  auto res = isl_union_pw_multi_aff_gist_params(copy(), context.release());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::intersect_domain(union_set uset) const
{
  auto res = isl_union_pw_multi_aff_intersect_domain(copy(), uset.release());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::intersect_params(set set) const
{
  auto res = isl_union_pw_multi_aff_intersect_params(copy(), set.release());
  return manage(res);
}

boolean union_pw_multi_aff::involves_nan() const
{
  auto res = isl_union_pw_multi_aff_involves_nan(get());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::multi_val_on_domain(union_set domain, multi_val mv)
{
  auto res = isl_union_pw_multi_aff_multi_val_on_domain(domain.release(), mv.release());
  return manage(res);
}

int union_pw_multi_aff::n_pw_multi_aff() const
{
  auto res = isl_union_pw_multi_aff_n_pw_multi_aff(get());
  return res;
}

union_pw_multi_aff union_pw_multi_aff::neg() const
{
  auto res = isl_union_pw_multi_aff_neg(copy());
  return manage(res);
}

boolean union_pw_multi_aff::plain_is_equal(const union_pw_multi_aff &upma2) const
{
  auto res = isl_union_pw_multi_aff_plain_is_equal(get(), upma2.get());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::pullback(union_pw_multi_aff upma2) const
{
  auto res = isl_union_pw_multi_aff_pullback_union_pw_multi_aff(copy(), upma2.release());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::reset_user() const
{
  auto res = isl_union_pw_multi_aff_reset_user(copy());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::scale_down_val(val val) const
{
  auto res = isl_union_pw_multi_aff_scale_down_val(copy(), val.release());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::scale_multi_val(multi_val mv) const
{
  auto res = isl_union_pw_multi_aff_scale_multi_val(copy(), mv.release());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::scale_val(val val) const
{
  auto res = isl_union_pw_multi_aff_scale_val(copy(), val.release());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::sub(union_pw_multi_aff upma2) const
{
  auto res = isl_union_pw_multi_aff_sub(copy(), upma2.release());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::subtract_domain(union_set uset) const
{
  auto res = isl_union_pw_multi_aff_subtract_domain(copy(), uset.release());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::union_add(union_pw_multi_aff upma2) const
{
  auto res = isl_union_pw_multi_aff_union_add(copy(), upma2.release());
  return manage(res);
}

// implementations for isl::union_pw_multi_aff_list
union_pw_multi_aff_list manage(__isl_take isl_union_pw_multi_aff_list *ptr) {
  return union_pw_multi_aff_list(ptr);
}
union_pw_multi_aff_list manage_copy(__isl_keep isl_union_pw_multi_aff_list *ptr) {
  ptr = isl_union_pw_multi_aff_list_copy(ptr);
  return union_pw_multi_aff_list(ptr);
}

union_pw_multi_aff_list::union_pw_multi_aff_list()
    : ptr(nullptr) {}

union_pw_multi_aff_list::union_pw_multi_aff_list(const union_pw_multi_aff_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
union_pw_multi_aff_list::union_pw_multi_aff_list(std::nullptr_t)
    : ptr(nullptr) {}


union_pw_multi_aff_list::union_pw_multi_aff_list(__isl_take isl_union_pw_multi_aff_list *ptr)
    : ptr(ptr) {}


union_pw_multi_aff_list &union_pw_multi_aff_list::operator=(union_pw_multi_aff_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_pw_multi_aff_list::~union_pw_multi_aff_list() {
  if (ptr)
    isl_union_pw_multi_aff_list_free(ptr);
}

__isl_give isl_union_pw_multi_aff_list *union_pw_multi_aff_list::copy() const & {
  return isl_union_pw_multi_aff_list_copy(ptr);
}

__isl_keep isl_union_pw_multi_aff_list *union_pw_multi_aff_list::get() const {
  return ptr;
}

__isl_give isl_union_pw_multi_aff_list *union_pw_multi_aff_list::release() {
  isl_union_pw_multi_aff_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_pw_multi_aff_list::is_null() const {
  return ptr == nullptr;
}
union_pw_multi_aff_list::operator bool() const {
  return !is_null();
}


ctx union_pw_multi_aff_list::get_ctx() const {
  return ctx(isl_union_pw_multi_aff_list_get_ctx(ptr));
}

void union_pw_multi_aff_list::dump() const {
  isl_union_pw_multi_aff_list_dump(get());
}


union_pw_multi_aff_list union_pw_multi_aff_list::add(union_pw_multi_aff el) const
{
  auto res = isl_union_pw_multi_aff_list_add(copy(), el.release());
  return manage(res);
}

union_pw_multi_aff_list union_pw_multi_aff_list::alloc(ctx ctx, int n)
{
  auto res = isl_union_pw_multi_aff_list_alloc(ctx.release(), n);
  return manage(res);
}

union_pw_multi_aff_list union_pw_multi_aff_list::concat(union_pw_multi_aff_list list2) const
{
  auto res = isl_union_pw_multi_aff_list_concat(copy(), list2.release());
  return manage(res);
}

union_pw_multi_aff_list union_pw_multi_aff_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_union_pw_multi_aff_list_drop(copy(), first, n);
  return manage(res);
}

stat union_pw_multi_aff_list::foreach(const std::function<stat(union_pw_multi_aff)> &fn) const
{
  struct fn_data {
    const std::function<stat(union_pw_multi_aff)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_union_pw_multi_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_union_pw_multi_aff_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

union_pw_multi_aff_list union_pw_multi_aff_list::from_union_pw_multi_aff(union_pw_multi_aff el)
{
  auto res = isl_union_pw_multi_aff_list_from_union_pw_multi_aff(el.release());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff_list::get_at(int index) const
{
  auto res = isl_union_pw_multi_aff_list_get_at(get(), index);
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff_list::get_union_pw_multi_aff(int index) const
{
  auto res = isl_union_pw_multi_aff_list_get_union_pw_multi_aff(get(), index);
  return manage(res);
}

union_pw_multi_aff_list union_pw_multi_aff_list::insert(unsigned int pos, union_pw_multi_aff el) const
{
  auto res = isl_union_pw_multi_aff_list_insert(copy(), pos, el.release());
  return manage(res);
}

int union_pw_multi_aff_list::n_union_pw_multi_aff() const
{
  auto res = isl_union_pw_multi_aff_list_n_union_pw_multi_aff(get());
  return res;
}

union_pw_multi_aff_list union_pw_multi_aff_list::reverse() const
{
  auto res = isl_union_pw_multi_aff_list_reverse(copy());
  return manage(res);
}

union_pw_multi_aff_list union_pw_multi_aff_list::set_union_pw_multi_aff(int index, union_pw_multi_aff el) const
{
  auto res = isl_union_pw_multi_aff_list_set_union_pw_multi_aff(copy(), index, el.release());
  return manage(res);
}

int union_pw_multi_aff_list::size() const
{
  auto res = isl_union_pw_multi_aff_list_size(get());
  return res;
}

union_pw_multi_aff_list union_pw_multi_aff_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_union_pw_multi_aff_list_swap(copy(), pos1, pos2);
  return manage(res);
}

// implementations for isl::union_pw_qpolynomial
union_pw_qpolynomial manage(__isl_take isl_union_pw_qpolynomial *ptr) {
  return union_pw_qpolynomial(ptr);
}
union_pw_qpolynomial manage_copy(__isl_keep isl_union_pw_qpolynomial *ptr) {
  ptr = isl_union_pw_qpolynomial_copy(ptr);
  return union_pw_qpolynomial(ptr);
}

union_pw_qpolynomial::union_pw_qpolynomial()
    : ptr(nullptr) {}

union_pw_qpolynomial::union_pw_qpolynomial(const union_pw_qpolynomial &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
union_pw_qpolynomial::union_pw_qpolynomial(std::nullptr_t)
    : ptr(nullptr) {}


union_pw_qpolynomial::union_pw_qpolynomial(__isl_take isl_union_pw_qpolynomial *ptr)
    : ptr(ptr) {}

union_pw_qpolynomial::union_pw_qpolynomial(ctx ctx, const std::string &str)
{
  auto res = isl_union_pw_qpolynomial_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

union_pw_qpolynomial &union_pw_qpolynomial::operator=(union_pw_qpolynomial obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_pw_qpolynomial::~union_pw_qpolynomial() {
  if (ptr)
    isl_union_pw_qpolynomial_free(ptr);
}

__isl_give isl_union_pw_qpolynomial *union_pw_qpolynomial::copy() const & {
  return isl_union_pw_qpolynomial_copy(ptr);
}

__isl_keep isl_union_pw_qpolynomial *union_pw_qpolynomial::get() const {
  return ptr;
}

__isl_give isl_union_pw_qpolynomial *union_pw_qpolynomial::release() {
  isl_union_pw_qpolynomial *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_pw_qpolynomial::is_null() const {
  return ptr == nullptr;
}
union_pw_qpolynomial::operator bool() const {
  return !is_null();
}


ctx union_pw_qpolynomial::get_ctx() const {
  return ctx(isl_union_pw_qpolynomial_get_ctx(ptr));
}
std::string union_pw_qpolynomial::to_str() const {
  char *Tmp = isl_union_pw_qpolynomial_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}



union_pw_qpolynomial union_pw_qpolynomial::add(union_pw_qpolynomial upwqp2) const
{
  auto res = isl_union_pw_qpolynomial_add(copy(), upwqp2.release());
  return manage(res);
}

union_pw_qpolynomial union_pw_qpolynomial::add_pw_qpolynomial(pw_qpolynomial pwqp) const
{
  auto res = isl_union_pw_qpolynomial_add_pw_qpolynomial(copy(), pwqp.release());
  return manage(res);
}

union_pw_qpolynomial union_pw_qpolynomial::align_params(space model) const
{
  auto res = isl_union_pw_qpolynomial_align_params(copy(), model.release());
  return manage(res);
}

union_pw_qpolynomial union_pw_qpolynomial::coalesce() const
{
  auto res = isl_union_pw_qpolynomial_coalesce(copy());
  return manage(res);
}

unsigned int union_pw_qpolynomial::dim(isl::dim type) const
{
  auto res = isl_union_pw_qpolynomial_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

union_set union_pw_qpolynomial::domain() const
{
  auto res = isl_union_pw_qpolynomial_domain(copy());
  return manage(res);
}

union_pw_qpolynomial union_pw_qpolynomial::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_union_pw_qpolynomial_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

val union_pw_qpolynomial::eval(point pnt) const
{
  auto res = isl_union_pw_qpolynomial_eval(copy(), pnt.release());
  return manage(res);
}

pw_qpolynomial union_pw_qpolynomial::extract_pw_qpolynomial(space dim) const
{
  auto res = isl_union_pw_qpolynomial_extract_pw_qpolynomial(get(), dim.release());
  return manage(res);
}

int union_pw_qpolynomial::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_union_pw_qpolynomial_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

stat union_pw_qpolynomial::foreach_pw_qpolynomial(const std::function<stat(pw_qpolynomial)> &fn) const
{
  struct fn_data {
    const std::function<stat(pw_qpolynomial)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_pw_qpolynomial *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_union_pw_qpolynomial_foreach_pw_qpolynomial(get(), fn_lambda, &fn_data);
  return manage(res);
}

union_pw_qpolynomial union_pw_qpolynomial::from_pw_qpolynomial(pw_qpolynomial pwqp)
{
  auto res = isl_union_pw_qpolynomial_from_pw_qpolynomial(pwqp.release());
  return manage(res);
}

pw_qpolynomial_list union_pw_qpolynomial::get_pw_qpolynomial_list() const
{
  auto res = isl_union_pw_qpolynomial_get_pw_qpolynomial_list(get());
  return manage(res);
}

space union_pw_qpolynomial::get_space() const
{
  auto res = isl_union_pw_qpolynomial_get_space(get());
  return manage(res);
}

union_pw_qpolynomial union_pw_qpolynomial::gist(union_set context) const
{
  auto res = isl_union_pw_qpolynomial_gist(copy(), context.release());
  return manage(res);
}

union_pw_qpolynomial union_pw_qpolynomial::gist_params(set context) const
{
  auto res = isl_union_pw_qpolynomial_gist_params(copy(), context.release());
  return manage(res);
}

union_pw_qpolynomial union_pw_qpolynomial::intersect_domain(union_set uset) const
{
  auto res = isl_union_pw_qpolynomial_intersect_domain(copy(), uset.release());
  return manage(res);
}

union_pw_qpolynomial union_pw_qpolynomial::intersect_params(set set) const
{
  auto res = isl_union_pw_qpolynomial_intersect_params(copy(), set.release());
  return manage(res);
}

boolean union_pw_qpolynomial::involves_nan() const
{
  auto res = isl_union_pw_qpolynomial_involves_nan(get());
  return manage(res);
}

union_pw_qpolynomial union_pw_qpolynomial::mul(union_pw_qpolynomial upwqp2) const
{
  auto res = isl_union_pw_qpolynomial_mul(copy(), upwqp2.release());
  return manage(res);
}

int union_pw_qpolynomial::n_pw_qpolynomial() const
{
  auto res = isl_union_pw_qpolynomial_n_pw_qpolynomial(get());
  return res;
}

union_pw_qpolynomial union_pw_qpolynomial::neg() const
{
  auto res = isl_union_pw_qpolynomial_neg(copy());
  return manage(res);
}

boolean union_pw_qpolynomial::plain_is_equal(const union_pw_qpolynomial &upwqp2) const
{
  auto res = isl_union_pw_qpolynomial_plain_is_equal(get(), upwqp2.get());
  return manage(res);
}

union_pw_qpolynomial union_pw_qpolynomial::reset_user() const
{
  auto res = isl_union_pw_qpolynomial_reset_user(copy());
  return manage(res);
}

union_pw_qpolynomial union_pw_qpolynomial::scale_down_val(val v) const
{
  auto res = isl_union_pw_qpolynomial_scale_down_val(copy(), v.release());
  return manage(res);
}

union_pw_qpolynomial union_pw_qpolynomial::scale_val(val v) const
{
  auto res = isl_union_pw_qpolynomial_scale_val(copy(), v.release());
  return manage(res);
}

union_pw_qpolynomial union_pw_qpolynomial::sub(union_pw_qpolynomial upwqp2) const
{
  auto res = isl_union_pw_qpolynomial_sub(copy(), upwqp2.release());
  return manage(res);
}

union_pw_qpolynomial union_pw_qpolynomial::subtract_domain(union_set uset) const
{
  auto res = isl_union_pw_qpolynomial_subtract_domain(copy(), uset.release());
  return manage(res);
}

union_pw_qpolynomial union_pw_qpolynomial::to_polynomial(int sign) const
{
  auto res = isl_union_pw_qpolynomial_to_polynomial(copy(), sign);
  return manage(res);
}

union_pw_qpolynomial union_pw_qpolynomial::zero(space dim)
{
  auto res = isl_union_pw_qpolynomial_zero(dim.release());
  return manage(res);
}

// implementations for isl::union_set
union_set manage(__isl_take isl_union_set *ptr) {
  return union_set(ptr);
}
union_set manage_copy(__isl_keep isl_union_set *ptr) {
  ptr = isl_union_set_copy(ptr);
  return union_set(ptr);
}

union_set::union_set()
    : ptr(nullptr) {}

union_set::union_set(const union_set &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
union_set::union_set(std::nullptr_t)
    : ptr(nullptr) {}


union_set::union_set(__isl_take isl_union_set *ptr)
    : ptr(ptr) {}

union_set::union_set(basic_set bset)
{
  auto res = isl_union_set_from_basic_set(bset.release());
  ptr = res;
}
union_set::union_set(set set)
{
  auto res = isl_union_set_from_set(set.release());
  ptr = res;
}
union_set::union_set(point pnt)
{
  auto res = isl_union_set_from_point(pnt.release());
  ptr = res;
}
union_set::union_set(ctx ctx, const std::string &str)
{
  auto res = isl_union_set_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

union_set &union_set::operator=(union_set obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_set::~union_set() {
  if (ptr)
    isl_union_set_free(ptr);
}

__isl_give isl_union_set *union_set::copy() const & {
  return isl_union_set_copy(ptr);
}

__isl_keep isl_union_set *union_set::get() const {
  return ptr;
}

__isl_give isl_union_set *union_set::release() {
  isl_union_set *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_set::is_null() const {
  return ptr == nullptr;
}
union_set::operator bool() const {
  return !is_null();
}


ctx union_set::get_ctx() const {
  return ctx(isl_union_set_get_ctx(ptr));
}
std::string union_set::to_str() const {
  char *Tmp = isl_union_set_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void union_set::dump() const {
  isl_union_set_dump(get());
}


union_set union_set::add_set(set set) const
{
  auto res = isl_union_set_add_set(copy(), set.release());
  return manage(res);
}

union_set union_set::affine_hull() const
{
  auto res = isl_union_set_affine_hull(copy());
  return manage(res);
}

union_set union_set::align_params(space model) const
{
  auto res = isl_union_set_align_params(copy(), model.release());
  return manage(res);
}

union_set union_set::apply(union_map umap) const
{
  auto res = isl_union_set_apply(copy(), umap.release());
  return manage(res);
}

union_set union_set::coalesce() const
{
  auto res = isl_union_set_coalesce(copy());
  return manage(res);
}

union_set union_set::coefficients() const
{
  auto res = isl_union_set_coefficients(copy());
  return manage(res);
}

schedule union_set::compute_schedule(union_map validity, union_map proximity) const
{
  auto res = isl_union_set_compute_schedule(copy(), validity.release(), proximity.release());
  return manage(res);
}

boolean union_set::contains(const space &space) const
{
  auto res = isl_union_set_contains(get(), space.get());
  return manage(res);
}

union_set union_set::detect_equalities() const
{
  auto res = isl_union_set_detect_equalities(copy());
  return manage(res);
}

unsigned int union_set::dim(isl::dim type) const
{
  auto res = isl_union_set_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

union_set union_set::empty(space space)
{
  auto res = isl_union_set_empty(space.release());
  return manage(res);
}

set union_set::extract_set(space dim) const
{
  auto res = isl_union_set_extract_set(get(), dim.release());
  return manage(res);
}

stat union_set::foreach_point(const std::function<stat(point)> &fn) const
{
  struct fn_data {
    const std::function<stat(point)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_point *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_union_set_foreach_point(get(), fn_lambda, &fn_data);
  return manage(res);
}

stat union_set::foreach_set(const std::function<stat(set)> &fn) const
{
  struct fn_data {
    const std::function<stat(set)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_set *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_union_set_foreach_set(get(), fn_lambda, &fn_data);
  return manage(res);
}

basic_set_list union_set::get_basic_set_list() const
{
  auto res = isl_union_set_get_basic_set_list(get());
  return manage(res);
}

uint32_t union_set::get_hash() const
{
  auto res = isl_union_set_get_hash(get());
  return res;
}

set_list union_set::get_set_list() const
{
  auto res = isl_union_set_get_set_list(get());
  return manage(res);
}

space union_set::get_space() const
{
  auto res = isl_union_set_get_space(get());
  return manage(res);
}

union_set union_set::gist(union_set context) const
{
  auto res = isl_union_set_gist(copy(), context.release());
  return manage(res);
}

union_set union_set::gist_params(set set) const
{
  auto res = isl_union_set_gist_params(copy(), set.release());
  return manage(res);
}

union_map union_set::identity() const
{
  auto res = isl_union_set_identity(copy());
  return manage(res);
}

union_pw_multi_aff union_set::identity_union_pw_multi_aff() const
{
  auto res = isl_union_set_identity_union_pw_multi_aff(copy());
  return manage(res);
}

union_set union_set::intersect(union_set uset2) const
{
  auto res = isl_union_set_intersect(copy(), uset2.release());
  return manage(res);
}

union_set union_set::intersect_params(set set) const
{
  auto res = isl_union_set_intersect_params(copy(), set.release());
  return manage(res);
}

boolean union_set::is_disjoint(const union_set &uset2) const
{
  auto res = isl_union_set_is_disjoint(get(), uset2.get());
  return manage(res);
}

boolean union_set::is_empty() const
{
  auto res = isl_union_set_is_empty(get());
  return manage(res);
}

boolean union_set::is_equal(const union_set &uset2) const
{
  auto res = isl_union_set_is_equal(get(), uset2.get());
  return manage(res);
}

boolean union_set::is_params() const
{
  auto res = isl_union_set_is_params(get());
  return manage(res);
}

boolean union_set::is_strict_subset(const union_set &uset2) const
{
  auto res = isl_union_set_is_strict_subset(get(), uset2.get());
  return manage(res);
}

boolean union_set::is_subset(const union_set &uset2) const
{
  auto res = isl_union_set_is_subset(get(), uset2.get());
  return manage(res);
}

union_map union_set::lex_ge_union_set(union_set uset2) const
{
  auto res = isl_union_set_lex_ge_union_set(copy(), uset2.release());
  return manage(res);
}

union_map union_set::lex_gt_union_set(union_set uset2) const
{
  auto res = isl_union_set_lex_gt_union_set(copy(), uset2.release());
  return manage(res);
}

union_map union_set::lex_le_union_set(union_set uset2) const
{
  auto res = isl_union_set_lex_le_union_set(copy(), uset2.release());
  return manage(res);
}

union_map union_set::lex_lt_union_set(union_set uset2) const
{
  auto res = isl_union_set_lex_lt_union_set(copy(), uset2.release());
  return manage(res);
}

union_set union_set::lexmax() const
{
  auto res = isl_union_set_lexmax(copy());
  return manage(res);
}

union_set union_set::lexmin() const
{
  auto res = isl_union_set_lexmin(copy());
  return manage(res);
}

multi_val union_set::min_multi_union_pw_aff(const multi_union_pw_aff &obj) const
{
  auto res = isl_union_set_min_multi_union_pw_aff(get(), obj.get());
  return manage(res);
}

int union_set::n_set() const
{
  auto res = isl_union_set_n_set(get());
  return res;
}

set union_set::params() const
{
  auto res = isl_union_set_params(copy());
  return manage(res);
}

union_set union_set::polyhedral_hull() const
{
  auto res = isl_union_set_polyhedral_hull(copy());
  return manage(res);
}

union_set union_set::preimage(multi_aff ma) const
{
  auto res = isl_union_set_preimage_multi_aff(copy(), ma.release());
  return manage(res);
}

union_set union_set::preimage(pw_multi_aff pma) const
{
  auto res = isl_union_set_preimage_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

union_set union_set::preimage(union_pw_multi_aff upma) const
{
  auto res = isl_union_set_preimage_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

union_set union_set::product(union_set uset2) const
{
  auto res = isl_union_set_product(copy(), uset2.release());
  return manage(res);
}

union_set union_set::project_out(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_union_set_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

union_set union_set::remove_divs() const
{
  auto res = isl_union_set_remove_divs(copy());
  return manage(res);
}

union_set union_set::remove_redundancies() const
{
  auto res = isl_union_set_remove_redundancies(copy());
  return manage(res);
}

union_set union_set::reset_user() const
{
  auto res = isl_union_set_reset_user(copy());
  return manage(res);
}

basic_set union_set::sample() const
{
  auto res = isl_union_set_sample(copy());
  return manage(res);
}

point union_set::sample_point() const
{
  auto res = isl_union_set_sample_point(copy());
  return manage(res);
}

union_set union_set::simple_hull() const
{
  auto res = isl_union_set_simple_hull(copy());
  return manage(res);
}

union_set union_set::solutions() const
{
  auto res = isl_union_set_solutions(copy());
  return manage(res);
}

union_set union_set::subtract(union_set uset2) const
{
  auto res = isl_union_set_subtract(copy(), uset2.release());
  return manage(res);
}

union_set union_set::unite(union_set uset2) const
{
  auto res = isl_union_set_union(copy(), uset2.release());
  return manage(res);
}

union_set union_set::universe() const
{
  auto res = isl_union_set_universe(copy());
  return manage(res);
}

union_map union_set::unwrap() const
{
  auto res = isl_union_set_unwrap(copy());
  return manage(res);
}

union_map union_set::wrapped_domain_map() const
{
  auto res = isl_union_set_wrapped_domain_map(copy());
  return manage(res);
}

// implementations for isl::union_set_list
union_set_list manage(__isl_take isl_union_set_list *ptr) {
  return union_set_list(ptr);
}
union_set_list manage_copy(__isl_keep isl_union_set_list *ptr) {
  ptr = isl_union_set_list_copy(ptr);
  return union_set_list(ptr);
}

union_set_list::union_set_list()
    : ptr(nullptr) {}

union_set_list::union_set_list(const union_set_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
union_set_list::union_set_list(std::nullptr_t)
    : ptr(nullptr) {}


union_set_list::union_set_list(__isl_take isl_union_set_list *ptr)
    : ptr(ptr) {}


union_set_list &union_set_list::operator=(union_set_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_set_list::~union_set_list() {
  if (ptr)
    isl_union_set_list_free(ptr);
}

__isl_give isl_union_set_list *union_set_list::copy() const & {
  return isl_union_set_list_copy(ptr);
}

__isl_keep isl_union_set_list *union_set_list::get() const {
  return ptr;
}

__isl_give isl_union_set_list *union_set_list::release() {
  isl_union_set_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_set_list::is_null() const {
  return ptr == nullptr;
}
union_set_list::operator bool() const {
  return !is_null();
}


ctx union_set_list::get_ctx() const {
  return ctx(isl_union_set_list_get_ctx(ptr));
}

void union_set_list::dump() const {
  isl_union_set_list_dump(get());
}


union_set_list union_set_list::add(union_set el) const
{
  auto res = isl_union_set_list_add(copy(), el.release());
  return manage(res);
}

union_set_list union_set_list::alloc(ctx ctx, int n)
{
  auto res = isl_union_set_list_alloc(ctx.release(), n);
  return manage(res);
}

union_set_list union_set_list::concat(union_set_list list2) const
{
  auto res = isl_union_set_list_concat(copy(), list2.release());
  return manage(res);
}

union_set_list union_set_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_union_set_list_drop(copy(), first, n);
  return manage(res);
}

stat union_set_list::foreach(const std::function<stat(union_set)> &fn) const
{
  struct fn_data {
    const std::function<stat(union_set)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_union_set *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_union_set_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

union_set_list union_set_list::from_union_set(union_set el)
{
  auto res = isl_union_set_list_from_union_set(el.release());
  return manage(res);
}

union_set union_set_list::get_at(int index) const
{
  auto res = isl_union_set_list_get_at(get(), index);
  return manage(res);
}

union_set union_set_list::get_union_set(int index) const
{
  auto res = isl_union_set_list_get_union_set(get(), index);
  return manage(res);
}

union_set_list union_set_list::insert(unsigned int pos, union_set el) const
{
  auto res = isl_union_set_list_insert(copy(), pos, el.release());
  return manage(res);
}

int union_set_list::n_union_set() const
{
  auto res = isl_union_set_list_n_union_set(get());
  return res;
}

union_set_list union_set_list::reverse() const
{
  auto res = isl_union_set_list_reverse(copy());
  return manage(res);
}

union_set_list union_set_list::set_union_set(int index, union_set el) const
{
  auto res = isl_union_set_list_set_union_set(copy(), index, el.release());
  return manage(res);
}

int union_set_list::size() const
{
  auto res = isl_union_set_list_size(get());
  return res;
}

union_set_list union_set_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_union_set_list_swap(copy(), pos1, pos2);
  return manage(res);
}

union_set union_set_list::unite() const
{
  auto res = isl_union_set_list_union(copy());
  return manage(res);
}

// implementations for isl::val
val manage(__isl_take isl_val *ptr) {
  return val(ptr);
}
val manage_copy(__isl_keep isl_val *ptr) {
  ptr = isl_val_copy(ptr);
  return val(ptr);
}

val::val()
    : ptr(nullptr) {}

val::val(const val &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
val::val(std::nullptr_t)
    : ptr(nullptr) {}


val::val(__isl_take isl_val *ptr)
    : ptr(ptr) {}

val::val(ctx ctx, const std::string &str)
{
  auto res = isl_val_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}
val::val(ctx ctx, long i)
{
  auto res = isl_val_int_from_si(ctx.release(), i);
  ptr = res;
}

val &val::operator=(val obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

val::~val() {
  if (ptr)
    isl_val_free(ptr);
}

__isl_give isl_val *val::copy() const & {
  return isl_val_copy(ptr);
}

__isl_keep isl_val *val::get() const {
  return ptr;
}

__isl_give isl_val *val::release() {
  isl_val *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool val::is_null() const {
  return ptr == nullptr;
}
val::operator bool() const {
  return !is_null();
}


ctx val::get_ctx() const {
  return ctx(isl_val_get_ctx(ptr));
}
std::string val::to_str() const {
  char *Tmp = isl_val_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


void val::dump() const {
  isl_val_dump(get());
}


val val::abs() const
{
  auto res = isl_val_abs(copy());
  return manage(res);
}

boolean val::abs_eq(const val &v2) const
{
  auto res = isl_val_abs_eq(get(), v2.get());
  return manage(res);
}

val val::add(val v2) const
{
  auto res = isl_val_add(copy(), v2.release());
  return manage(res);
}

val val::add_ui(unsigned long v2) const
{
  auto res = isl_val_add_ui(copy(), v2);
  return manage(res);
}

val val::ceil() const
{
  auto res = isl_val_ceil(copy());
  return manage(res);
}

int val::cmp_si(long i) const
{
  auto res = isl_val_cmp_si(get(), i);
  return res;
}

val val::div(val v2) const
{
  auto res = isl_val_div(copy(), v2.release());
  return manage(res);
}

val val::div_ui(unsigned long v2) const
{
  auto res = isl_val_div_ui(copy(), v2);
  return manage(res);
}

boolean val::eq(const val &v2) const
{
  auto res = isl_val_eq(get(), v2.get());
  return manage(res);
}

val val::floor() const
{
  auto res = isl_val_floor(copy());
  return manage(res);
}

val val::gcd(val v2) const
{
  auto res = isl_val_gcd(copy(), v2.release());
  return manage(res);
}

boolean val::ge(const val &v2) const
{
  auto res = isl_val_ge(get(), v2.get());
  return manage(res);
}

uint32_t val::get_hash() const
{
  auto res = isl_val_get_hash(get());
  return res;
}

long val::get_num_si() const
{
  auto res = isl_val_get_num_si(get());
  return res;
}

boolean val::gt(const val &v2) const
{
  auto res = isl_val_gt(get(), v2.get());
  return manage(res);
}

boolean val::gt_si(long i) const
{
  auto res = isl_val_gt_si(get(), i);
  return manage(res);
}

val val::infty(ctx ctx)
{
  auto res = isl_val_infty(ctx.release());
  return manage(res);
}

val val::int_from_ui(ctx ctx, unsigned long u)
{
  auto res = isl_val_int_from_ui(ctx.release(), u);
  return manage(res);
}

val val::inv() const
{
  auto res = isl_val_inv(copy());
  return manage(res);
}

boolean val::is_divisible_by(const val &v2) const
{
  auto res = isl_val_is_divisible_by(get(), v2.get());
  return manage(res);
}

boolean val::is_infty() const
{
  auto res = isl_val_is_infty(get());
  return manage(res);
}

boolean val::is_int() const
{
  auto res = isl_val_is_int(get());
  return manage(res);
}

boolean val::is_nan() const
{
  auto res = isl_val_is_nan(get());
  return manage(res);
}

boolean val::is_neg() const
{
  auto res = isl_val_is_neg(get());
  return manage(res);
}

boolean val::is_neginfty() const
{
  auto res = isl_val_is_neginfty(get());
  return manage(res);
}

boolean val::is_negone() const
{
  auto res = isl_val_is_negone(get());
  return manage(res);
}

boolean val::is_nonneg() const
{
  auto res = isl_val_is_nonneg(get());
  return manage(res);
}

boolean val::is_nonpos() const
{
  auto res = isl_val_is_nonpos(get());
  return manage(res);
}

boolean val::is_one() const
{
  auto res = isl_val_is_one(get());
  return manage(res);
}

boolean val::is_pos() const
{
  auto res = isl_val_is_pos(get());
  return manage(res);
}

boolean val::is_rat() const
{
  auto res = isl_val_is_rat(get());
  return manage(res);
}

boolean val::is_zero() const
{
  auto res = isl_val_is_zero(get());
  return manage(res);
}

boolean val::le(const val &v2) const
{
  auto res = isl_val_le(get(), v2.get());
  return manage(res);
}

boolean val::lt(const val &v2) const
{
  auto res = isl_val_lt(get(), v2.get());
  return manage(res);
}

val val::max(val v2) const
{
  auto res = isl_val_max(copy(), v2.release());
  return manage(res);
}

val val::min(val v2) const
{
  auto res = isl_val_min(copy(), v2.release());
  return manage(res);
}

val val::mod(val v2) const
{
  auto res = isl_val_mod(copy(), v2.release());
  return manage(res);
}

val val::mul(val v2) const
{
  auto res = isl_val_mul(copy(), v2.release());
  return manage(res);
}

val val::mul_ui(unsigned long v2) const
{
  auto res = isl_val_mul_ui(copy(), v2);
  return manage(res);
}

size_t val::n_abs_num_chunks(size_t size) const
{
  auto res = isl_val_n_abs_num_chunks(get(), size);
  return res;
}

val val::nan(ctx ctx)
{
  auto res = isl_val_nan(ctx.release());
  return manage(res);
}

boolean val::ne(const val &v2) const
{
  auto res = isl_val_ne(get(), v2.get());
  return manage(res);
}

val val::neg() const
{
  auto res = isl_val_neg(copy());
  return manage(res);
}

val val::neginfty(ctx ctx)
{
  auto res = isl_val_neginfty(ctx.release());
  return manage(res);
}

val val::negone(ctx ctx)
{
  auto res = isl_val_negone(ctx.release());
  return manage(res);
}

val val::one(ctx ctx)
{
  auto res = isl_val_one(ctx.release());
  return manage(res);
}

val val::pow2() const
{
  auto res = isl_val_pow2(copy());
  return manage(res);
}

val val::set_si(long i) const
{
  auto res = isl_val_set_si(copy(), i);
  return manage(res);
}

int val::sgn() const
{
  auto res = isl_val_sgn(get());
  return res;
}

val val::sub(val v2) const
{
  auto res = isl_val_sub(copy(), v2.release());
  return manage(res);
}

val val::sub_ui(unsigned long v2) const
{
  auto res = isl_val_sub_ui(copy(), v2);
  return manage(res);
}

val val::trunc() const
{
  auto res = isl_val_trunc(copy());
  return manage(res);
}

val val::zero(ctx ctx)
{
  auto res = isl_val_zero(ctx.release());
  return manage(res);
}

// implementations for isl::val_list
val_list manage(__isl_take isl_val_list *ptr) {
  return val_list(ptr);
}
val_list manage_copy(__isl_keep isl_val_list *ptr) {
  ptr = isl_val_list_copy(ptr);
  return val_list(ptr);
}

val_list::val_list()
    : ptr(nullptr) {}

val_list::val_list(const val_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
val_list::val_list(std::nullptr_t)
    : ptr(nullptr) {}


val_list::val_list(__isl_take isl_val_list *ptr)
    : ptr(ptr) {}


val_list &val_list::operator=(val_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

val_list::~val_list() {
  if (ptr)
    isl_val_list_free(ptr);
}

__isl_give isl_val_list *val_list::copy() const & {
  return isl_val_list_copy(ptr);
}

__isl_keep isl_val_list *val_list::get() const {
  return ptr;
}

__isl_give isl_val_list *val_list::release() {
  isl_val_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool val_list::is_null() const {
  return ptr == nullptr;
}
val_list::operator bool() const {
  return !is_null();
}


ctx val_list::get_ctx() const {
  return ctx(isl_val_list_get_ctx(ptr));
}

void val_list::dump() const {
  isl_val_list_dump(get());
}


val_list val_list::add(val el) const
{
  auto res = isl_val_list_add(copy(), el.release());
  return manage(res);
}

val_list val_list::alloc(ctx ctx, int n)
{
  auto res = isl_val_list_alloc(ctx.release(), n);
  return manage(res);
}

val_list val_list::concat(val_list list2) const
{
  auto res = isl_val_list_concat(copy(), list2.release());
  return manage(res);
}

val_list val_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_val_list_drop(copy(), first, n);
  return manage(res);
}

stat val_list::foreach(const std::function<stat(val)> &fn) const
{
  struct fn_data {
    const std::function<stat(val)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_val *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_val_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

val_list val_list::from_val(val el)
{
  auto res = isl_val_list_from_val(el.release());
  return manage(res);
}

val val_list::get_at(int index) const
{
  auto res = isl_val_list_get_at(get(), index);
  return manage(res);
}

val val_list::get_val(int index) const
{
  auto res = isl_val_list_get_val(get(), index);
  return manage(res);
}

val_list val_list::insert(unsigned int pos, val el) const
{
  auto res = isl_val_list_insert(copy(), pos, el.release());
  return manage(res);
}

int val_list::n_val() const
{
  auto res = isl_val_list_n_val(get());
  return res;
}

val_list val_list::reverse() const
{
  auto res = isl_val_list_reverse(copy());
  return manage(res);
}

val_list val_list::set_val(int index, val el) const
{
  auto res = isl_val_list_set_val(copy(), index, el.release());
  return manage(res);
}

int val_list::size() const
{
  auto res = isl_val_list_size(get());
  return res;
}

val_list val_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_val_list_swap(copy(), pos1, pos2);
  return manage(res);
}

// implementations for isl::vec
vec manage(__isl_take isl_vec *ptr) {
  return vec(ptr);
}
vec manage_copy(__isl_keep isl_vec *ptr) {
  ptr = isl_vec_copy(ptr);
  return vec(ptr);
}

vec::vec()
    : ptr(nullptr) {}

vec::vec(const vec &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}
vec::vec(std::nullptr_t)
    : ptr(nullptr) {}


vec::vec(__isl_take isl_vec *ptr)
    : ptr(ptr) {}


vec &vec::operator=(vec obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

vec::~vec() {
  if (ptr)
    isl_vec_free(ptr);
}

__isl_give isl_vec *vec::copy() const & {
  return isl_vec_copy(ptr);
}

__isl_keep isl_vec *vec::get() const {
  return ptr;
}

__isl_give isl_vec *vec::release() {
  isl_vec *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool vec::is_null() const {
  return ptr == nullptr;
}
vec::operator bool() const {
  return !is_null();
}


ctx vec::get_ctx() const {
  return ctx(isl_vec_get_ctx(ptr));
}

void vec::dump() const {
  isl_vec_dump(get());
}


vec vec::add(vec vec2) const
{
  auto res = isl_vec_add(copy(), vec2.release());
  return manage(res);
}

vec vec::add_els(unsigned int n) const
{
  auto res = isl_vec_add_els(copy(), n);
  return manage(res);
}

vec vec::alloc(ctx ctx, unsigned int size)
{
  auto res = isl_vec_alloc(ctx.release(), size);
  return manage(res);
}

vec vec::ceil() const
{
  auto res = isl_vec_ceil(copy());
  return manage(res);
}

vec vec::clr() const
{
  auto res = isl_vec_clr(copy());
  return manage(res);
}

int vec::cmp_element(const vec &vec2, int pos) const
{
  auto res = isl_vec_cmp_element(get(), vec2.get(), pos);
  return res;
}

vec vec::concat(vec vec2) const
{
  auto res = isl_vec_concat(copy(), vec2.release());
  return manage(res);
}

vec vec::drop_els(unsigned int pos, unsigned int n) const
{
  auto res = isl_vec_drop_els(copy(), pos, n);
  return manage(res);
}

vec vec::extend(unsigned int size) const
{
  auto res = isl_vec_extend(copy(), size);
  return manage(res);
}

val vec::get_element_val(int pos) const
{
  auto res = isl_vec_get_element_val(get(), pos);
  return manage(res);
}

vec vec::insert_els(unsigned int pos, unsigned int n) const
{
  auto res = isl_vec_insert_els(copy(), pos, n);
  return manage(res);
}

vec vec::insert_zero_els(unsigned int pos, unsigned int n) const
{
  auto res = isl_vec_insert_zero_els(copy(), pos, n);
  return manage(res);
}

boolean vec::is_equal(const vec &vec2) const
{
  auto res = isl_vec_is_equal(get(), vec2.get());
  return manage(res);
}

vec vec::mat_product(mat mat) const
{
  auto res = isl_vec_mat_product(copy(), mat.release());
  return manage(res);
}

vec vec::move_els(unsigned int dst_col, unsigned int src_col, unsigned int n) const
{
  auto res = isl_vec_move_els(copy(), dst_col, src_col, n);
  return manage(res);
}

vec vec::neg() const
{
  auto res = isl_vec_neg(copy());
  return manage(res);
}

vec vec::set_element_si(int pos, int v) const
{
  auto res = isl_vec_set_element_si(copy(), pos, v);
  return manage(res);
}

vec vec::set_element_val(int pos, val v) const
{
  auto res = isl_vec_set_element_val(copy(), pos, v.release());
  return manage(res);
}

vec vec::set_si(int v) const
{
  auto res = isl_vec_set_si(copy(), v);
  return manage(res);
}

vec vec::set_val(val v) const
{
  auto res = isl_vec_set_val(copy(), v.release());
  return manage(res);
}

int vec::size() const
{
  auto res = isl_vec_size(get());
  return res;
}

vec vec::sort() const
{
  auto res = isl_vec_sort(copy());
  return manage(res);
}

vec vec::zero(ctx ctx, unsigned int size)
{
  auto res = isl_vec_zero(ctx.release(), size);
  return manage(res);
}

vec vec::zero_extend(unsigned int size) const
{
  auto res = isl_vec_zero_extend(copy(), size);
  return manage(res);
}
} // namespace noexceptions 
} // namespace isl

#endif /* ISL_CPP_CHECKED */
