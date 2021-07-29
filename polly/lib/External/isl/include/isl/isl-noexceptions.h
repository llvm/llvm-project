/// These are automatically generated checked C++ bindings for isl.
///
/// isl is a library for computing with integer sets and maps described by
/// Presburger formulas. On top of this, isl provides various tools for
/// polyhedral compilation, ranging from dependence analysis over scheduling
/// to AST generation.

// clang-format off

#ifndef ISL_CPP_CHECKED
#define ISL_CPP_CHECKED

#include <isl/id.h>
#include <isl/space.h>
#include <isl/val.h>
#include <isl/aff.h>
#include <isl/set.h>
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
#include <isl/fixed_box.h>
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
class multi_id;
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
class qpolynomial_list;
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
  inline explicit aff(isl::ctx ctx, const std::string &str);
  inline explicit aff(isl::local_space ls, isl::val val);
  inline explicit aff(isl::local_space ls);
  inline aff &operator=(aff obj);
  inline ~aff();
  inline __isl_give isl_aff *copy() const &;
  inline __isl_give isl_aff *copy() && = delete;
  inline __isl_keep isl_aff *get() const;
  inline __isl_give isl_aff *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::aff add(isl::aff aff2) const;
  inline isl::aff add_coefficient_si(isl::dim type, int pos, int v) const;
  inline isl::aff add_coefficient_val(isl::dim type, int pos, isl::val v) const;
  inline isl::aff add_constant(isl::val v) const;
  inline isl::aff add_constant_num_si(int v) const;
  inline isl::aff add_constant_si(int v) const;
  inline isl::aff add_dims(isl::dim type, unsigned int n) const;
  inline isl::aff align_params(isl::space model) const;
  inline isl::basic_set bind(isl::id id) const;
  inline isl::aff ceil() const;
  inline int coefficient_sgn(isl::dim type, int pos) const;
  inline isl_size dim(isl::dim type) const;
  inline isl::aff div(isl::aff aff2) const;
  inline isl::aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::basic_set eq_basic_set(isl::aff aff2) const;
  inline isl::set eq_set(isl::aff aff2) const;
  inline isl::val eval(isl::point pnt) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::aff floor() const;
  inline isl::aff from_range() const;
  inline isl::basic_set ge_basic_set(isl::aff aff2) const;
  inline isl::set ge_set(isl::aff aff2) const;
  inline isl::val get_coefficient_val(isl::dim type, int pos) const;
  inline isl::val get_constant_val() const;
  inline isl::val get_denominator_val() const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline isl::aff get_div(int pos) const;
  inline isl::local_space get_domain_local_space() const;
  inline isl::space get_domain_space() const;
  inline uint32_t get_hash() const;
  inline isl::local_space get_local_space() const;
  inline isl::space get_space() const;
  inline isl::aff gist(isl::set context) const;
  inline isl::aff gist_params(isl::set context) const;
  inline isl::basic_set gt_basic_set(isl::aff aff2) const;
  inline isl::set gt_set(isl::aff aff2) const;
  inline isl::aff insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_locals() const;
  inline boolean is_cst() const;
  inline boolean is_nan() const;
  inline isl::basic_set le_basic_set(isl::aff aff2) const;
  inline isl::set le_set(isl::aff aff2) const;
  inline isl::basic_set lt_basic_set(isl::aff aff2) const;
  inline isl::set lt_set(isl::aff aff2) const;
  inline isl::aff mod(isl::val mod) const;
  inline isl::aff move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline isl::aff mul(isl::aff aff2) const;
  static inline isl::aff nan_on_domain(isl::local_space ls);
  static inline isl::aff nan_on_domain_space(isl::space space);
  inline isl::set ne_set(isl::aff aff2) const;
  inline isl::aff neg() const;
  inline isl::basic_set neg_basic_set() const;
  static inline isl::aff param_on_domain_space_id(isl::space space, isl::id id);
  inline boolean plain_is_equal(const isl::aff &aff2) const;
  inline boolean plain_is_zero() const;
  inline isl::aff project_domain_on_params() const;
  inline isl::aff pullback(isl::multi_aff ma) const;
  inline isl::aff pullback_aff(isl::aff aff2) const;
  inline isl::aff scale(isl::val v) const;
  inline isl::aff scale_down(isl::val v) const;
  inline isl::aff scale_down_ui(unsigned int f) const;
  inline isl::aff set_coefficient_si(isl::dim type, int pos, int v) const;
  inline isl::aff set_coefficient_val(isl::dim type, int pos, isl::val v) const;
  inline isl::aff set_constant_si(int v) const;
  inline isl::aff set_constant_val(isl::val v) const;
  inline isl::aff set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::aff set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::aff sub(isl::aff aff2) const;
  inline isl::aff unbind_params_insert_domain(isl::multi_id domain) const;
  static inline isl::aff val_on_domain_space(isl::space space, isl::val val);
  static inline isl::aff var_on_domain(isl::local_space ls, isl::dim type, unsigned int pos);
  inline isl::basic_set zero_basic_set() const;
  static inline isl::aff zero_on_domain(isl::space space);
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
  inline aff_list &operator=(aff_list obj);
  inline ~aff_list();
  inline __isl_give isl_aff_list *copy() const &;
  inline __isl_give isl_aff_list *copy() && = delete;
  inline __isl_keep isl_aff_list *get() const;
  inline __isl_give isl_aff_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::aff_list add(isl::aff el) const;
  static inline isl::aff_list alloc(isl::ctx ctx, int n);
  inline isl::aff_list clear() const;
  inline isl::aff_list concat(isl::aff_list list2) const;
  inline isl::aff_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(aff)> &fn) const;
  static inline isl::aff_list from_aff(isl::aff el);
  inline isl::aff get_aff(int index) const;
  inline isl::aff get_at(int index) const;
  inline isl::aff_list insert(unsigned int pos, isl::aff el) const;
  inline isl_size n_aff() const;
  inline isl::aff_list reverse() const;
  inline isl::aff_list set_aff(int index, isl::aff el) const;
  inline isl_size size() const;
  inline isl::aff_list swap(unsigned int pos1, unsigned int pos2) const;
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
  inline explicit ast_build(isl::ctx ctx);
  inline ast_build &operator=(ast_build obj);
  inline ~ast_build();
  inline __isl_give isl_ast_build *copy() const &;
  inline __isl_give isl_ast_build *copy() && = delete;
  inline __isl_keep isl_ast_build *get() const;
  inline __isl_give isl_ast_build *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::ast_expr access_from(isl::multi_pw_aff mpa) const;
  inline isl::ast_expr access_from(isl::pw_multi_aff pma) const;
  inline isl::ast_node ast_from_schedule(isl::union_map schedule) const;
  inline isl::ast_expr call_from(isl::multi_pw_aff mpa) const;
  inline isl::ast_expr call_from(isl::pw_multi_aff pma) const;
  inline isl::ast_expr expr_from(isl::pw_aff pa) const;
  inline isl::ast_expr expr_from(isl::set set) const;
  static inline isl::ast_build from_context(isl::set set);
  inline isl::union_map get_schedule() const;
  inline isl::space get_schedule_space() const;
  inline isl::ast_node node_from(isl::schedule schedule) const;
  inline isl::ast_node node_from_schedule_map(isl::union_map schedule) const;
  inline isl::ast_build restrict(isl::set set) const;
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
  inline ast_expr &operator=(ast_expr obj);
  inline ~ast_expr();
  inline __isl_give isl_ast_expr *copy() const &;
  inline __isl_give isl_ast_expr *copy() && = delete;
  inline __isl_keep isl_ast_expr *get() const;
  inline __isl_give isl_ast_expr *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::ast_expr access(isl::ast_expr_list indices) const;
  inline isl::ast_expr add(isl::ast_expr expr2) const;
  inline isl::ast_expr address_of() const;
  inline isl::ast_expr call(isl::ast_expr_list arguments) const;
  inline isl::ast_expr div(isl::ast_expr expr2) const;
  inline isl::ast_expr eq(isl::ast_expr expr2) const;
  static inline isl::ast_expr from_id(isl::id id);
  static inline isl::ast_expr from_val(isl::val v);
  inline isl::ast_expr ge(isl::ast_expr expr2) const;
  inline isl::id get_id() const;
  inline isl::ast_expr get_op_arg(int pos) const;
  inline isl_size get_op_n_arg() const;
  inline isl::val get_val() const;
  inline isl::ast_expr gt(isl::ast_expr expr2) const;
  inline isl::id id_get_id() const;
  inline isl::val int_get_val() const;
  inline boolean is_equal(const isl::ast_expr &expr2) const;
  inline isl::ast_expr le(isl::ast_expr expr2) const;
  inline isl::ast_expr lt(isl::ast_expr expr2) const;
  inline isl::ast_expr mul(isl::ast_expr expr2) const;
  inline isl::ast_expr neg() const;
  inline isl::ast_expr op_get_arg(int pos) const;
  inline isl_size op_get_n_arg() const;
  inline isl::ast_expr pdiv_q(isl::ast_expr expr2) const;
  inline isl::ast_expr pdiv_r(isl::ast_expr expr2) const;
  inline isl::ast_expr set_op_arg(int pos, isl::ast_expr arg) const;
  inline isl::ast_expr sub(isl::ast_expr expr2) const;
  inline isl::ast_expr substitute_ids(isl::id_to_ast_expr id2expr) const;
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
  inline ast_expr_list &operator=(ast_expr_list obj);
  inline ~ast_expr_list();
  inline __isl_give isl_ast_expr_list *copy() const &;
  inline __isl_give isl_ast_expr_list *copy() && = delete;
  inline __isl_keep isl_ast_expr_list *get() const;
  inline __isl_give isl_ast_expr_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::ast_expr_list add(isl::ast_expr el) const;
  static inline isl::ast_expr_list alloc(isl::ctx ctx, int n);
  inline isl::ast_expr_list clear() const;
  inline isl::ast_expr_list concat(isl::ast_expr_list list2) const;
  inline isl::ast_expr_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(ast_expr)> &fn) const;
  static inline isl::ast_expr_list from_ast_expr(isl::ast_expr el);
  inline isl::ast_expr get_ast_expr(int index) const;
  inline isl::ast_expr get_at(int index) const;
  inline isl::ast_expr_list insert(unsigned int pos, isl::ast_expr el) const;
  inline isl_size n_ast_expr() const;
  inline isl::ast_expr_list reverse() const;
  inline isl::ast_expr_list set_ast_expr(int index, isl::ast_expr el) const;
  inline isl_size size() const;
  inline isl::ast_expr_list swap(unsigned int pos1, unsigned int pos2) const;
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
  inline ast_node &operator=(ast_node obj);
  inline ~ast_node();
  inline __isl_give isl_ast_node *copy() const &;
  inline __isl_give isl_ast_node *copy() && = delete;
  inline __isl_keep isl_ast_node *get() const;
  inline __isl_give isl_ast_node *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  static inline isl::ast_node alloc_user(isl::ast_expr expr);
  inline isl::ast_node_list block_get_children() const;
  inline isl::ast_node for_get_body() const;
  inline isl::ast_expr for_get_cond() const;
  inline isl::ast_expr for_get_inc() const;
  inline isl::ast_expr for_get_init() const;
  inline isl::ast_expr for_get_iterator() const;
  inline boolean for_is_degenerate() const;
  inline isl::id get_annotation() const;
  inline isl::ast_expr if_get_cond() const;
  inline isl::ast_node if_get_else() const;
  inline isl::ast_node if_get_else_node() const;
  inline isl::ast_node if_get_then() const;
  inline isl::ast_node if_get_then_node() const;
  inline boolean if_has_else() const;
  inline boolean if_has_else_node() const;
  inline isl::id mark_get_id() const;
  inline isl::ast_node mark_get_node() const;
  inline isl::ast_node set_annotation(isl::id annotation) const;
  inline std::string to_C_str() const;
  inline isl::ast_expr user_get_expr() const;
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
  inline ast_node_list &operator=(ast_node_list obj);
  inline ~ast_node_list();
  inline __isl_give isl_ast_node_list *copy() const &;
  inline __isl_give isl_ast_node_list *copy() && = delete;
  inline __isl_keep isl_ast_node_list *get() const;
  inline __isl_give isl_ast_node_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::ast_node_list add(isl::ast_node el) const;
  static inline isl::ast_node_list alloc(isl::ctx ctx, int n);
  inline isl::ast_node_list clear() const;
  inline isl::ast_node_list concat(isl::ast_node_list list2) const;
  inline isl::ast_node_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(ast_node)> &fn) const;
  static inline isl::ast_node_list from_ast_node(isl::ast_node el);
  inline isl::ast_node get_ast_node(int index) const;
  inline isl::ast_node get_at(int index) const;
  inline isl::ast_node_list insert(unsigned int pos, isl::ast_node el) const;
  inline isl_size n_ast_node() const;
  inline isl::ast_node_list reverse() const;
  inline isl::ast_node_list set_ast_node(int index, isl::ast_node el) const;
  inline isl_size size() const;
  inline isl::ast_node_list swap(unsigned int pos1, unsigned int pos2) const;
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
  inline explicit basic_map(isl::ctx ctx, const std::string &str);
  inline basic_map &operator=(basic_map obj);
  inline ~basic_map();
  inline __isl_give isl_basic_map *copy() const &;
  inline __isl_give isl_basic_map *copy() && = delete;
  inline __isl_keep isl_basic_map *get() const;
  inline __isl_give isl_basic_map *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::basic_map add_constraint(isl::constraint constraint) const;
  inline isl::basic_map add_dims(isl::dim type, unsigned int n) const;
  inline isl::basic_map affine_hull() const;
  inline isl::basic_map align_params(isl::space model) const;
  inline isl::basic_map apply_domain(isl::basic_map bmap2) const;
  inline isl::basic_map apply_range(isl::basic_map bmap2) const;
  inline boolean can_curry() const;
  inline boolean can_uncurry() const;
  inline boolean can_zip() const;
  inline isl::basic_map curry() const;
  inline isl::basic_set deltas() const;
  inline isl::basic_map deltas_map() const;
  inline isl::basic_map detect_equalities() const;
  inline isl_size dim(isl::dim type) const;
  inline isl::basic_set domain() const;
  inline isl::basic_map domain_map() const;
  inline isl::basic_map domain_product(isl::basic_map bmap2) const;
  inline isl::basic_map drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::basic_map drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::basic_map drop_unused_params() const;
  inline isl::basic_map eliminate(isl::dim type, unsigned int first, unsigned int n) const;
  static inline isl::basic_map empty(isl::space space);
  static inline isl::basic_map equal(isl::space space, unsigned int n_equal);
  inline isl::mat equalities_matrix(isl::dim c1, isl::dim c2, isl::dim c3, isl::dim c4, isl::dim c5) const;
  inline isl::basic_map equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::basic_map fix_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::basic_map fix_val(isl::dim type, unsigned int pos, isl::val v) const;
  inline isl::basic_map flat_product(isl::basic_map bmap2) const;
  inline isl::basic_map flat_range_product(isl::basic_map bmap2) const;
  inline isl::basic_map flatten() const;
  inline isl::basic_map flatten_domain() const;
  inline isl::basic_map flatten_range() const;
  inline stat foreach_constraint(const std::function<stat(constraint)> &fn) const;
  static inline isl::basic_map from_aff(isl::aff aff);
  static inline isl::basic_map from_aff_list(isl::space domain_space, isl::aff_list list);
  static inline isl::basic_map from_constraint(isl::constraint constraint);
  static inline isl::basic_map from_domain(isl::basic_set bset);
  static inline isl::basic_map from_domain_and_range(isl::basic_set domain, isl::basic_set range);
  static inline isl::basic_map from_multi_aff(isl::multi_aff maff);
  static inline isl::basic_map from_qpolynomial(isl::qpolynomial qp);
  static inline isl::basic_map from_range(isl::basic_set bset);
  inline isl::constraint_list get_constraint_list() const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline isl::aff get_div(int pos) const;
  inline isl::local_space get_local_space() const;
  inline isl::space get_space() const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline isl::basic_map gist(isl::basic_map context) const;
  inline isl::basic_map gist_domain(isl::basic_set context) const;
  inline boolean has_dim_id(isl::dim type, unsigned int pos) const;
  static inline isl::basic_map identity(isl::space space);
  inline boolean image_is_bounded() const;
  inline isl::mat inequalities_matrix(isl::dim c1, isl::dim c2, isl::dim c3, isl::dim c4, isl::dim c5) const;
  inline isl::basic_map insert_dims(isl::dim type, unsigned int pos, unsigned int n) const;
  inline isl::basic_map intersect(isl::basic_map bmap2) const;
  inline isl::basic_map intersect_domain(isl::basic_set bset) const;
  inline isl::basic_map intersect_range(isl::basic_set bset) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean is_disjoint(const isl::basic_map &bmap2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const isl::basic_map &bmap2) const;
  inline boolean is_rational() const;
  inline boolean is_single_valued() const;
  inline boolean is_strict_subset(const isl::basic_map &bmap2) const;
  inline boolean is_subset(const isl::basic_map &bmap2) const;
  inline boolean is_universe() const;
  static inline isl::basic_map less_at(isl::space space, unsigned int pos);
  inline isl::map lexmax() const;
  inline isl::map lexmin() const;
  inline isl::pw_multi_aff lexmin_pw_multi_aff() const;
  inline isl::basic_map lower_bound_si(isl::dim type, unsigned int pos, int value) const;
  static inline isl::basic_map more_at(isl::space space, unsigned int pos);
  inline isl::basic_map move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline isl_size n_constraint() const;
  static inline isl::basic_map nat_universe(isl::space space);
  inline isl::basic_map neg() const;
  inline isl::basic_map order_ge(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline isl::basic_map order_gt(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline isl::val plain_get_val_if_fixed(isl::dim type, unsigned int pos) const;
  inline boolean plain_is_empty() const;
  inline boolean plain_is_universe() const;
  inline isl::basic_map preimage_domain_multi_aff(isl::multi_aff ma) const;
  inline isl::basic_map preimage_range_multi_aff(isl::multi_aff ma) const;
  inline isl::basic_map product(isl::basic_map bmap2) const;
  inline isl::basic_map project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::basic_set range() const;
  inline isl::basic_map range_map() const;
  inline isl::basic_map range_product(isl::basic_map bmap2) const;
  inline isl::basic_map remove_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::basic_map remove_divs() const;
  inline isl::basic_map remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::basic_map remove_redundancies() const;
  inline isl::basic_map reverse() const;
  inline isl::basic_map sample() const;
  inline isl::basic_map set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::basic_map set_tuple_name(isl::dim type, const std::string &s) const;
  inline isl::basic_map sum(isl::basic_map bmap2) const;
  inline isl::basic_map uncurry() const;
  inline isl::map unite(isl::basic_map bmap2) const;
  static inline isl::basic_map universe(isl::space space);
  inline isl::basic_map upper_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::basic_set wrap() const;
  inline isl::basic_map zip() const;
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
  inline basic_map_list &operator=(basic_map_list obj);
  inline ~basic_map_list();
  inline __isl_give isl_basic_map_list *copy() const &;
  inline __isl_give isl_basic_map_list *copy() && = delete;
  inline __isl_keep isl_basic_map_list *get() const;
  inline __isl_give isl_basic_map_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::basic_map_list add(isl::basic_map el) const;
  static inline isl::basic_map_list alloc(isl::ctx ctx, int n);
  inline isl::basic_map_list clear() const;
  inline isl::basic_map_list concat(isl::basic_map_list list2) const;
  inline isl::basic_map_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(basic_map)> &fn) const;
  static inline isl::basic_map_list from_basic_map(isl::basic_map el);
  inline isl::basic_map get_at(int index) const;
  inline isl::basic_map get_basic_map(int index) const;
  inline isl::basic_map_list insert(unsigned int pos, isl::basic_map el) const;
  inline isl_size n_basic_map() const;
  inline isl::basic_map_list reverse() const;
  inline isl::basic_map_list set_basic_map(int index, isl::basic_map el) const;
  inline isl_size size() const;
  inline isl::basic_map_list swap(unsigned int pos1, unsigned int pos2) const;
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
  inline /* implicit */ basic_set(isl::point pnt);
  inline explicit basic_set(isl::ctx ctx, const std::string &str);
  inline basic_set &operator=(basic_set obj);
  inline ~basic_set();
  inline __isl_give isl_basic_set *copy() const &;
  inline __isl_give isl_basic_set *copy() && = delete;
  inline __isl_keep isl_basic_set *get() const;
  inline __isl_give isl_basic_set *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::basic_set affine_hull() const;
  inline isl::basic_set align_params(isl::space model) const;
  inline isl::basic_set apply(isl::basic_map bmap) const;
  static inline isl::basic_set box_from_points(isl::point pnt1, isl::point pnt2);
  inline isl::basic_set coefficients() const;
  inline isl::basic_set detect_equalities() const;
  inline isl_size dim(isl::dim type) const;
  inline isl::val dim_max_val(int pos) const;
  inline isl::basic_set drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::basic_set drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::basic_set drop_unused_params() const;
  inline isl::basic_set eliminate(isl::dim type, unsigned int first, unsigned int n) const;
  static inline isl::basic_set empty(isl::space space);
  inline isl::mat equalities_matrix(isl::dim c1, isl::dim c2, isl::dim c3, isl::dim c4) const;
  inline isl::basic_set fix_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::basic_set fix_val(isl::dim type, unsigned int pos, isl::val v) const;
  inline isl::basic_set flat_product(isl::basic_set bset2) const;
  inline isl::basic_set flatten() const;
  inline stat foreach_bound_pair(isl::dim type, unsigned int pos, const std::function<stat(constraint, constraint, basic_set)> &fn) const;
  inline stat foreach_constraint(const std::function<stat(constraint)> &fn) const;
  static inline isl::basic_set from_constraint(isl::constraint constraint);
  static inline isl::basic_set from_multi_aff(isl::multi_aff ma);
  inline isl::basic_set from_params() const;
  inline isl::constraint_list get_constraint_list() const;
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline isl::aff get_div(int pos) const;
  inline isl::local_space get_local_space() const;
  inline isl::space get_space() const;
  inline std::string get_tuple_name() const;
  inline isl::basic_set gist(isl::basic_set context) const;
  inline isl::mat inequalities_matrix(isl::dim c1, isl::dim c2, isl::dim c3, isl::dim c4) const;
  inline isl::basic_set insert_dims(isl::dim type, unsigned int pos, unsigned int n) const;
  inline isl::basic_set intersect(isl::basic_set bset2) const;
  inline isl::basic_set intersect_params(isl::basic_set bset2) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean is_bounded() const;
  inline boolean is_disjoint(const isl::basic_set &bset2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const isl::basic_set &bset2) const;
  inline int is_rational() const;
  inline boolean is_subset(const isl::basic_set &bset2) const;
  inline boolean is_universe() const;
  inline boolean is_wrapping() const;
  inline isl::set lexmax() const;
  inline isl::set lexmin() const;
  inline isl::basic_set lower_bound_val(isl::dim type, unsigned int pos, isl::val value) const;
  inline isl::val max_val(const isl::aff &obj) const;
  inline isl::basic_set move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline isl_size n_constraint() const;
  inline isl_size n_dim() const;
  static inline isl::basic_set nat_universe(isl::space space);
  inline isl::basic_set neg() const;
  inline isl::basic_set params() const;
  inline boolean plain_is_empty() const;
  inline boolean plain_is_equal(const isl::basic_set &bset2) const;
  inline boolean plain_is_universe() const;
  static inline isl::basic_set positive_orthant(isl::space space);
  inline isl::basic_set preimage_multi_aff(isl::multi_aff ma) const;
  inline isl::basic_set project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::mat reduced_basis() const;
  inline isl::basic_set remove_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::basic_set remove_divs() const;
  inline isl::basic_set remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::basic_set remove_redundancies() const;
  inline isl::basic_set remove_unknown_divs() const;
  inline isl::basic_set sample() const;
  inline isl::point sample_point() const;
  inline isl::basic_set set_tuple_id(isl::id id) const;
  inline isl::basic_set set_tuple_name(const std::string &s) const;
  inline isl::basic_set solutions() const;
  inline isl::set unite(isl::basic_set bset2) const;
  static inline isl::basic_set universe(isl::space space);
  inline isl::basic_map unwrap() const;
  inline isl::basic_set upper_bound_val(isl::dim type, unsigned int pos, isl::val value) const;
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
  inline basic_set_list &operator=(basic_set_list obj);
  inline ~basic_set_list();
  inline __isl_give isl_basic_set_list *copy() const &;
  inline __isl_give isl_basic_set_list *copy() && = delete;
  inline __isl_keep isl_basic_set_list *get() const;
  inline __isl_give isl_basic_set_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::basic_set_list add(isl::basic_set el) const;
  static inline isl::basic_set_list alloc(isl::ctx ctx, int n);
  inline isl::basic_set_list clear() const;
  inline isl::basic_set_list coefficients() const;
  inline isl::basic_set_list concat(isl::basic_set_list list2) const;
  inline isl::basic_set_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(basic_set)> &fn) const;
  static inline isl::basic_set_list from_basic_set(isl::basic_set el);
  inline isl::basic_set get_at(int index) const;
  inline isl::basic_set get_basic_set(int index) const;
  inline isl::basic_set_list insert(unsigned int pos, isl::basic_set el) const;
  inline isl_size n_basic_set() const;
  inline isl::basic_set_list reverse() const;
  inline isl::basic_set_list set_basic_set(int index, isl::basic_set el) const;
  inline isl_size size() const;
  inline isl::basic_set_list swap(unsigned int pos1, unsigned int pos2) const;
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
  inline constraint &operator=(constraint obj);
  inline ~constraint();
  inline __isl_give isl_constraint *copy() const &;
  inline __isl_give isl_constraint *copy() && = delete;
  inline __isl_keep isl_constraint *get() const;
  inline __isl_give isl_constraint *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  static inline isl::constraint alloc_equality(isl::local_space ls);
  static inline isl::constraint alloc_inequality(isl::local_space ls);
  inline int cmp_last_non_zero(const isl::constraint &c2) const;
  inline isl::aff get_aff() const;
  inline isl::aff get_bound(isl::dim type, int pos) const;
  inline isl::val get_coefficient_val(isl::dim type, int pos) const;
  inline isl::val get_constant_val() const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline isl::aff get_div(int pos) const;
  inline isl::local_space get_local_space() const;
  inline isl::space get_space() const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean is_div_constraint() const;
  inline boolean is_lower_bound(isl::dim type, unsigned int pos) const;
  inline boolean is_upper_bound(isl::dim type, unsigned int pos) const;
  inline int plain_cmp(const isl::constraint &c2) const;
  inline isl::constraint set_coefficient_si(isl::dim type, int pos, int v) const;
  inline isl::constraint set_coefficient_val(isl::dim type, int pos, isl::val v) const;
  inline isl::constraint set_constant_si(int v) const;
  inline isl::constraint set_constant_val(isl::val v) const;
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
  inline constraint_list &operator=(constraint_list obj);
  inline ~constraint_list();
  inline __isl_give isl_constraint_list *copy() const &;
  inline __isl_give isl_constraint_list *copy() && = delete;
  inline __isl_keep isl_constraint_list *get() const;
  inline __isl_give isl_constraint_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::constraint_list add(isl::constraint el) const;
  static inline isl::constraint_list alloc(isl::ctx ctx, int n);
  inline isl::constraint_list clear() const;
  inline isl::constraint_list concat(isl::constraint_list list2) const;
  inline isl::constraint_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(constraint)> &fn) const;
  static inline isl::constraint_list from_constraint(isl::constraint el);
  inline isl::constraint get_at(int index) const;
  inline isl::constraint get_constraint(int index) const;
  inline isl::constraint_list insert(unsigned int pos, isl::constraint el) const;
  inline isl_size n_constraint() const;
  inline isl::constraint_list reverse() const;
  inline isl::constraint_list set_constraint(int index, isl::constraint el) const;
  inline isl_size size() const;
  inline isl::constraint_list swap(unsigned int pos1, unsigned int pos2) const;
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
  inline fixed_box &operator=(fixed_box obj);
  inline ~fixed_box();
  inline __isl_give isl_fixed_box *copy() const &;
  inline __isl_give isl_fixed_box *copy() && = delete;
  inline __isl_keep isl_fixed_box *get() const;
  inline __isl_give isl_fixed_box *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::multi_aff get_offset() const;
  inline isl::multi_val get_size() const;
  inline isl::space get_space() const;
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
  inline explicit id(isl::ctx ctx, const std::string &str);
  inline id &operator=(id obj);
  inline ~id();
  inline __isl_give isl_id *copy() const &;
  inline __isl_give isl_id *copy() && = delete;
  inline __isl_keep isl_id *get() const;
  inline __isl_give isl_id *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  static inline isl::id alloc(isl::ctx ctx, const std::string &name, void * user);
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
  inline id_list &operator=(id_list obj);
  inline ~id_list();
  inline __isl_give isl_id_list *copy() const &;
  inline __isl_give isl_id_list *copy() && = delete;
  inline __isl_keep isl_id_list *get() const;
  inline __isl_give isl_id_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::id_list add(isl::id el) const;
  static inline isl::id_list alloc(isl::ctx ctx, int n);
  inline isl::id_list clear() const;
  inline isl::id_list concat(isl::id_list list2) const;
  inline isl::id_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(id)> &fn) const;
  static inline isl::id_list from_id(isl::id el);
  inline isl::id get_at(int index) const;
  inline isl::id get_id(int index) const;
  inline isl::id_list insert(unsigned int pos, isl::id el) const;
  inline isl_size n_id() const;
  inline isl::id_list reverse() const;
  inline isl::id_list set_id(int index, isl::id el) const;
  inline isl_size size() const;
  inline isl::id_list swap(unsigned int pos1, unsigned int pos2) const;
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
  inline id_to_ast_expr &operator=(id_to_ast_expr obj);
  inline ~id_to_ast_expr();
  inline __isl_give isl_id_to_ast_expr *copy() const &;
  inline __isl_give isl_id_to_ast_expr *copy() && = delete;
  inline __isl_keep isl_id_to_ast_expr *get() const;
  inline __isl_give isl_id_to_ast_expr *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  static inline isl::id_to_ast_expr alloc(isl::ctx ctx, int min_size);
  inline isl::id_to_ast_expr drop(isl::id key) const;
  inline stat foreach(const std::function<stat(id, ast_expr)> &fn) const;
  inline isl::ast_expr get(isl::id key) const;
  inline boolean has(const isl::id &key) const;
  inline isl::id_to_ast_expr set(isl::id key, isl::ast_expr val) const;
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
  inline explicit local_space(isl::space space);
  inline local_space &operator=(local_space obj);
  inline ~local_space();
  inline __isl_give isl_local_space *copy() const &;
  inline __isl_give isl_local_space *copy() && = delete;
  inline __isl_keep isl_local_space *get() const;
  inline __isl_give isl_local_space *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::local_space add_dims(isl::dim type, unsigned int n) const;
  inline isl_size dim(isl::dim type) const;
  inline isl::local_space domain() const;
  inline isl::local_space drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::local_space flatten_domain() const;
  inline isl::local_space flatten_range() const;
  inline isl::local_space from_domain() const;
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline isl::aff get_div(int pos) const;
  inline isl::space get_space() const;
  inline boolean has_dim_id(isl::dim type, unsigned int pos) const;
  inline boolean has_dim_name(isl::dim type, unsigned int pos) const;
  inline isl::local_space insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::local_space intersect(isl::local_space ls2) const;
  inline boolean is_equal(const isl::local_space &ls2) const;
  inline boolean is_params() const;
  inline boolean is_set() const;
  inline isl::local_space range() const;
  inline isl::local_space set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::local_space set_from_params() const;
  inline isl::local_space set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::local_space wrap() const;
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
  inline /* implicit */ map(isl::basic_map bmap);
  inline explicit map(isl::ctx ctx, const std::string &str);
  inline map &operator=(map obj);
  inline ~map();
  inline __isl_give isl_map *copy() const &;
  inline __isl_give isl_map *copy() && = delete;
  inline __isl_keep isl_map *get() const;
  inline __isl_give isl_map *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::map add_constraint(isl::constraint constraint) const;
  inline isl::map add_dims(isl::dim type, unsigned int n) const;
  inline isl::basic_map affine_hull() const;
  inline isl::map align_params(isl::space model) const;
  inline isl::map apply_domain(isl::map map2) const;
  inline isl::map apply_range(isl::map map2) const;
  inline isl::set bind_domain(isl::multi_id tuple) const;
  inline isl::set bind_range(isl::multi_id tuple) const;
  inline boolean can_curry() const;
  inline boolean can_range_curry() const;
  inline boolean can_uncurry() const;
  inline boolean can_zip() const;
  inline isl::map coalesce() const;
  inline isl::map complement() const;
  inline isl::basic_map convex_hull() const;
  inline isl::map curry() const;
  inline isl::set deltas() const;
  inline isl::map deltas_map() const;
  inline isl::map detect_equalities() const;
  inline isl_size dim(isl::dim type) const;
  inline isl::pw_aff dim_max(int pos) const;
  inline isl::pw_aff dim_min(int pos) const;
  inline isl::set domain() const;
  inline isl::map domain_factor_domain() const;
  inline isl::map domain_factor_range() const;
  inline boolean domain_is_wrapping() const;
  inline isl::map domain_map() const;
  inline isl::map domain_product(isl::map map2) const;
  inline isl_size domain_tuple_dim() const;
  inline isl::map drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::map drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::map drop_unused_params() const;
  inline isl::map eliminate(isl::dim type, unsigned int first, unsigned int n) const;
  static inline isl::map empty(isl::space space);
  inline isl::map eq_at(isl::multi_pw_aff mpa) const;
  inline isl::map equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline isl::map factor_domain() const;
  inline isl::map factor_range() const;
  inline int find_dim_by_id(isl::dim type, const isl::id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::map fix_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::map fix_val(isl::dim type, unsigned int pos, isl::val v) const;
  inline isl::map fixed_power_val(isl::val exp) const;
  inline isl::map flat_domain_product(isl::map map2) const;
  inline isl::map flat_product(isl::map map2) const;
  inline isl::map flat_range_product(isl::map map2) const;
  inline isl::map flatten() const;
  inline isl::map flatten_domain() const;
  inline isl::map flatten_range() const;
  inline isl::map floordiv_val(isl::val d) const;
  inline stat foreach_basic_map(const std::function<stat(basic_map)> &fn) const;
  static inline isl::map from_aff(isl::aff aff);
  static inline isl::map from_domain(isl::set set);
  static inline isl::map from_domain_and_range(isl::set domain, isl::set range);
  static inline isl::map from_multi_aff(isl::multi_aff maff);
  static inline isl::map from_multi_pw_aff(isl::multi_pw_aff mpa);
  static inline isl::map from_pw_aff(isl::pw_aff pwaff);
  static inline isl::map from_pw_multi_aff(isl::pw_multi_aff pma);
  static inline isl::map from_range(isl::set set);
  static inline isl::map from_union_map(isl::union_map umap);
  inline isl::basic_map_list get_basic_map_list() const;
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline uint32_t get_hash() const;
  inline isl::fixed_box get_range_simple_fixed_box_hull() const;
  inline isl::space get_space() const;
  inline isl::id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline isl::map gist(isl::map context) const;
  inline isl::map gist_basic_map(isl::basic_map context) const;
  inline isl::map gist_domain(isl::set context) const;
  inline isl::map gist_params(isl::set context) const;
  inline isl::map gist_range(isl::set context) const;
  inline boolean has_dim_id(isl::dim type, unsigned int pos) const;
  inline boolean has_dim_name(isl::dim type, unsigned int pos) const;
  inline boolean has_equal_space(const isl::map &map2) const;
  inline boolean has_tuple_id(isl::dim type) const;
  inline boolean has_tuple_name(isl::dim type) const;
  static inline isl::map identity(isl::space space);
  inline isl::map insert_dims(isl::dim type, unsigned int pos, unsigned int n) const;
  inline isl::map intersect(isl::map map2) const;
  inline isl::map intersect_domain(isl::set set) const;
  inline isl::map intersect_domain_factor_domain(isl::map factor) const;
  inline isl::map intersect_domain_factor_range(isl::map factor) const;
  inline isl::map intersect_params(isl::set params) const;
  inline isl::map intersect_range(isl::set set) const;
  inline isl::map intersect_range_factor_domain(isl::map factor) const;
  inline isl::map intersect_range_factor_range(isl::map factor) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean is_bijective() const;
  inline boolean is_disjoint(const isl::map &map2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const isl::map &map2) const;
  inline boolean is_identity() const;
  inline boolean is_injective() const;
  inline boolean is_product() const;
  inline boolean is_single_valued() const;
  inline boolean is_strict_subset(const isl::map &map2) const;
  inline boolean is_subset(const isl::map &map2) const;
  inline int is_translation() const;
  static inline isl::map lex_ge(isl::space set_space);
  inline isl::map lex_ge_at(isl::multi_pw_aff mpa) const;
  static inline isl::map lex_ge_first(isl::space space, unsigned int n);
  inline isl::map lex_ge_map(isl::map map2) const;
  static inline isl::map lex_gt(isl::space set_space);
  inline isl::map lex_gt_at(isl::multi_pw_aff mpa) const;
  static inline isl::map lex_gt_first(isl::space space, unsigned int n);
  inline isl::map lex_gt_map(isl::map map2) const;
  static inline isl::map lex_le(isl::space set_space);
  inline isl::map lex_le_at(isl::multi_pw_aff mpa) const;
  static inline isl::map lex_le_first(isl::space space, unsigned int n);
  inline isl::map lex_le_map(isl::map map2) const;
  static inline isl::map lex_lt(isl::space set_space);
  inline isl::map lex_lt_at(isl::multi_pw_aff mpa) const;
  static inline isl::map lex_lt_first(isl::space space, unsigned int n);
  inline isl::map lex_lt_map(isl::map map2) const;
  inline isl::map lexmax() const;
  inline isl::pw_multi_aff lexmax_pw_multi_aff() const;
  inline isl::map lexmin() const;
  inline isl::pw_multi_aff lexmin_pw_multi_aff() const;
  inline isl::map lower_bound(isl::multi_pw_aff lower) const;
  inline isl::map lower_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::map lower_bound_val(isl::dim type, unsigned int pos, isl::val value) const;
  inline isl::multi_pw_aff max_multi_pw_aff() const;
  inline isl::multi_pw_aff min_multi_pw_aff() const;
  inline isl::map move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline isl_size n_basic_map() const;
  static inline isl::map nat_universe(isl::space space);
  inline isl::map neg() const;
  inline isl::map oppose(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline isl::map order_ge(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline isl::map order_gt(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline isl::map order_le(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline isl::map order_lt(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline isl::set params() const;
  inline isl::val plain_get_val_if_fixed(isl::dim type, unsigned int pos) const;
  inline boolean plain_is_empty() const;
  inline boolean plain_is_equal(const isl::map &map2) const;
  inline boolean plain_is_injective() const;
  inline boolean plain_is_single_valued() const;
  inline boolean plain_is_universe() const;
  inline isl::basic_map plain_unshifted_simple_hull() const;
  inline isl::basic_map polyhedral_hull() const;
  inline isl::map preimage_domain(isl::multi_aff ma) const;
  inline isl::map preimage_domain(isl::multi_pw_aff mpa) const;
  inline isl::map preimage_domain(isl::pw_multi_aff pma) const;
  inline isl::map preimage_range(isl::multi_aff ma) const;
  inline isl::map preimage_range(isl::pw_multi_aff pma) const;
  inline isl::map product(isl::map map2) const;
  inline isl::map project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::map project_out_all_params() const;
  inline isl::set range() const;
  inline isl::map range_curry() const;
  inline isl::map range_factor_domain() const;
  inline isl::map range_factor_range() const;
  inline boolean range_is_wrapping() const;
  inline isl::map range_map() const;
  inline isl::map range_product(isl::map map2) const;
  inline isl::map range_reverse() const;
  inline isl_size range_tuple_dim() const;
  inline isl::map remove_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::map remove_divs() const;
  inline isl::map remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::map remove_redundancies() const;
  inline isl::map remove_unknown_divs() const;
  inline isl::map reset_tuple_id(isl::dim type) const;
  inline isl::map reset_user() const;
  inline isl::map reverse() const;
  inline isl::basic_map sample() const;
  inline isl::map set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::map set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::map set_tuple_name(isl::dim type, const std::string &s) const;
  inline isl::basic_map simple_hull() const;
  inline isl::map subtract(isl::map map2) const;
  inline isl::map subtract_domain(isl::set dom) const;
  inline isl::map subtract_range(isl::set dom) const;
  inline isl::map sum(isl::map map2) const;
  inline isl::map uncurry() const;
  inline isl::map unite(isl::map map2) const;
  static inline isl::map universe(isl::space space);
  inline isl::basic_map unshifted_simple_hull() const;
  inline isl::basic_map unshifted_simple_hull_from_map_list(isl::map_list list) const;
  inline isl::map upper_bound(isl::multi_pw_aff upper) const;
  inline isl::map upper_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::map upper_bound_val(isl::dim type, unsigned int pos, isl::val value) const;
  inline isl::set wrap() const;
  inline isl::map zip() const;
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
  inline map_list &operator=(map_list obj);
  inline ~map_list();
  inline __isl_give isl_map_list *copy() const &;
  inline __isl_give isl_map_list *copy() && = delete;
  inline __isl_keep isl_map_list *get() const;
  inline __isl_give isl_map_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::map_list add(isl::map el) const;
  static inline isl::map_list alloc(isl::ctx ctx, int n);
  inline isl::map_list clear() const;
  inline isl::map_list concat(isl::map_list list2) const;
  inline isl::map_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(map)> &fn) const;
  static inline isl::map_list from_map(isl::map el);
  inline isl::map get_at(int index) const;
  inline isl::map get_map(int index) const;
  inline isl::map_list insert(unsigned int pos, isl::map el) const;
  inline isl_size n_map() const;
  inline isl::map_list reverse() const;
  inline isl::map_list set_map(int index, isl::map el) const;
  inline isl_size size() const;
  inline isl::map_list swap(unsigned int pos1, unsigned int pos2) const;
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
  inline mat &operator=(mat obj);
  inline ~mat();
  inline __isl_give isl_mat *copy() const &;
  inline __isl_give isl_mat *copy() && = delete;
  inline __isl_keep isl_mat *get() const;
  inline __isl_give isl_mat *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::mat add_rows(unsigned int n) const;
  inline isl::mat add_zero_cols(unsigned int n) const;
  inline isl::mat add_zero_rows(unsigned int n) const;
  inline isl::mat aff_direct_sum(isl::mat right) const;
  static inline isl::mat alloc(isl::ctx ctx, unsigned int n_row, unsigned int n_col);
  inline isl_size cols() const;
  inline isl::mat concat(isl::mat bot) const;
  inline isl::mat diagonal(isl::mat mat2) const;
  inline isl::mat drop_cols(unsigned int col, unsigned int n) const;
  inline isl::mat drop_rows(unsigned int row, unsigned int n) const;
  static inline isl::mat from_row_vec(isl::vec vec);
  inline isl::val get_element_val(int row, int col) const;
  inline boolean has_linearly_independent_rows(const isl::mat &mat2) const;
  inline int initial_non_zero_cols() const;
  inline isl::mat insert_cols(unsigned int col, unsigned int n) const;
  inline isl::mat insert_rows(unsigned int row, unsigned int n) const;
  inline isl::mat insert_zero_cols(unsigned int first, unsigned int n) const;
  inline isl::mat insert_zero_rows(unsigned int row, unsigned int n) const;
  inline isl::mat inverse_product(isl::mat right) const;
  inline boolean is_equal(const isl::mat &mat2) const;
  inline isl::mat lin_to_aff() const;
  inline isl::mat move_cols(unsigned int dst_col, unsigned int src_col, unsigned int n) const;
  inline isl::mat normalize() const;
  inline isl::mat normalize_row(int row) const;
  inline isl::mat product(isl::mat right) const;
  inline isl_size rank() const;
  inline isl::mat right_inverse() const;
  inline isl::mat right_kernel() const;
  inline isl::mat row_basis() const;
  inline isl::mat row_basis_extension(isl::mat mat2) const;
  inline isl_size rows() const;
  inline isl::mat set_element_si(int row, int col, int v) const;
  inline isl::mat set_element_val(int row, int col, isl::val v) const;
  inline isl::mat swap_cols(unsigned int i, unsigned int j) const;
  inline isl::mat swap_rows(unsigned int i, unsigned int j) const;
  inline isl::mat transpose() const;
  inline isl::mat unimodular_complete(int row) const;
  inline isl::mat vec_concat(isl::vec bot) const;
  inline isl::vec vec_inverse_product(isl::vec vec) const;
  inline isl::vec vec_product(isl::vec vec) const;
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
  inline /* implicit */ multi_aff(isl::aff aff);
  inline explicit multi_aff(isl::space space, isl::aff_list list);
  inline explicit multi_aff(isl::ctx ctx, const std::string &str);
  inline multi_aff &operator=(multi_aff obj);
  inline ~multi_aff();
  inline __isl_give isl_multi_aff *copy() const &;
  inline __isl_give isl_multi_aff *copy() && = delete;
  inline __isl_keep isl_multi_aff *get() const;
  inline __isl_give isl_multi_aff *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::multi_aff add(isl::multi_aff multi2) const;
  inline isl::multi_aff add_constant(isl::multi_val mv) const;
  inline isl::multi_aff add_constant(isl::val v) const;
  inline isl::multi_aff add_dims(isl::dim type, unsigned int n) const;
  inline isl::multi_aff align_params(isl::space model) const;
  inline isl::basic_set bind(isl::multi_id tuple) const;
  inline isl::multi_aff bind_domain(isl::multi_id tuple) const;
  inline isl::multi_aff bind_domain_wrapped_domain(isl::multi_id tuple) const;
  inline isl_size dim(isl::dim type) const;
  static inline isl::multi_aff domain_map(isl::space space);
  inline isl::multi_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::multi_aff factor_range() const;
  inline int find_dim_by_id(isl::dim type, const isl::id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::multi_aff flat_range_product(isl::multi_aff multi2) const;
  inline isl::multi_aff flatten_domain() const;
  inline isl::multi_aff flatten_range() const;
  inline isl::multi_aff floor() const;
  inline isl::multi_aff from_range() const;
  inline isl::aff get_aff(int pos) const;
  inline isl::aff get_at(int pos) const;
  inline isl::multi_val get_constant_multi_val() const;
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline isl::space get_domain_space() const;
  inline isl::aff_list get_list() const;
  inline isl::space get_space() const;
  inline isl::id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline isl::multi_aff gist(isl::set context) const;
  inline isl::multi_aff gist_params(isl::set context) const;
  inline boolean has_tuple_id(isl::dim type) const;
  static inline isl::multi_aff identity(isl::space space);
  inline isl::multi_aff identity() const;
  static inline isl::multi_aff identity_on_domain(isl::space space);
  inline isl::multi_aff insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::multi_aff insert_domain(isl::space domain) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_locals() const;
  inline boolean involves_nan() const;
  inline isl::set lex_ge_set(isl::multi_aff ma2) const;
  inline isl::set lex_gt_set(isl::multi_aff ma2) const;
  inline isl::set lex_le_set(isl::multi_aff ma2) const;
  inline isl::set lex_lt_set(isl::multi_aff ma2) const;
  inline isl::multi_aff mod_multi_val(isl::multi_val mv) const;
  inline isl::multi_aff move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  static inline isl::multi_aff multi_val_on_space(isl::space space, isl::multi_val mv);
  inline isl::multi_aff neg() const;
  inline int plain_cmp(const isl::multi_aff &multi2) const;
  inline boolean plain_is_equal(const isl::multi_aff &multi2) const;
  inline isl::multi_aff product(isl::multi_aff multi2) const;
  inline isl::multi_aff project_domain_on_params() const;
  static inline isl::multi_aff project_out_map(isl::space space, isl::dim type, unsigned int first, unsigned int n);
  inline isl::multi_aff pullback(isl::multi_aff ma2) const;
  inline isl::multi_aff range_factor_domain() const;
  inline isl::multi_aff range_factor_range() const;
  inline boolean range_is_wrapping() const;
  static inline isl::multi_aff range_map(isl::space space);
  inline isl::multi_aff range_product(isl::multi_aff multi2) const;
  inline isl::multi_aff range_splice(unsigned int pos, isl::multi_aff multi2) const;
  inline isl::multi_aff reset_tuple_id(isl::dim type) const;
  inline isl::multi_aff reset_user() const;
  inline isl::multi_aff scale(isl::multi_val mv) const;
  inline isl::multi_aff scale(isl::val v) const;
  inline isl::multi_aff scale_down(isl::multi_val mv) const;
  inline isl::multi_aff scale_down(isl::val v) const;
  inline isl::multi_aff set_aff(int pos, isl::aff el) const;
  inline isl::multi_aff set_at(int pos, isl::aff el) const;
  inline isl::multi_aff set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::multi_aff set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::multi_aff set_tuple_name(isl::dim type, const std::string &s) const;
  inline isl_size size() const;
  inline isl::multi_aff splice(unsigned int in_pos, unsigned int out_pos, isl::multi_aff multi2) const;
  inline isl::multi_aff sub(isl::multi_aff multi2) const;
  inline isl::multi_aff unbind_params_insert_domain(isl::multi_id domain) const;
  static inline isl::multi_aff zero(isl::space space);
};

// declarations for isl::multi_id
inline multi_id manage(__isl_take isl_multi_id *ptr);
inline multi_id manage_copy(__isl_keep isl_multi_id *ptr);

class multi_id {
  friend inline multi_id manage(__isl_take isl_multi_id *ptr);
  friend inline multi_id manage_copy(__isl_keep isl_multi_id *ptr);

  isl_multi_id *ptr = nullptr;

  inline explicit multi_id(__isl_take isl_multi_id *ptr);

public:
  inline /* implicit */ multi_id();
  inline /* implicit */ multi_id(const multi_id &obj);
  inline explicit multi_id(isl::space space, isl::id_list list);
  inline explicit multi_id(isl::ctx ctx, const std::string &str);
  inline multi_id &operator=(multi_id obj);
  inline ~multi_id();
  inline __isl_give isl_multi_id *copy() const &;
  inline __isl_give isl_multi_id *copy() && = delete;
  inline __isl_keep isl_multi_id *get() const;
  inline __isl_give isl_multi_id *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::multi_id align_params(isl::space model) const;
  inline isl::multi_id factor_range() const;
  inline isl::multi_id flat_range_product(isl::multi_id multi2) const;
  inline isl::multi_id flatten_range() const;
  inline isl::multi_id from_range() const;
  inline isl::id get_at(int pos) const;
  inline isl::space get_domain_space() const;
  inline isl::id get_id(int pos) const;
  inline isl::id_list get_list() const;
  inline isl::space get_space() const;
  inline boolean plain_is_equal(const isl::multi_id &multi2) const;
  inline isl::multi_id range_factor_domain() const;
  inline isl::multi_id range_factor_range() const;
  inline boolean range_is_wrapping() const;
  inline isl::multi_id range_product(isl::multi_id multi2) const;
  inline isl::multi_id range_splice(unsigned int pos, isl::multi_id multi2) const;
  inline isl::multi_id reset_user() const;
  inline isl::multi_id set_at(int pos, isl::id el) const;
  inline isl::multi_id set_id(int pos, isl::id el) const;
  inline isl_size size() const;
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
  inline /* implicit */ multi_pw_aff(isl::aff aff);
  inline /* implicit */ multi_pw_aff(isl::multi_aff ma);
  inline /* implicit */ multi_pw_aff(isl::pw_aff pa);
  inline explicit multi_pw_aff(isl::space space, isl::pw_aff_list list);
  inline /* implicit */ multi_pw_aff(isl::pw_multi_aff pma);
  inline explicit multi_pw_aff(isl::ctx ctx, const std::string &str);
  inline multi_pw_aff &operator=(multi_pw_aff obj);
  inline ~multi_pw_aff();
  inline __isl_give isl_multi_pw_aff *copy() const &;
  inline __isl_give isl_multi_pw_aff *copy() && = delete;
  inline __isl_keep isl_multi_pw_aff *get() const;
  inline __isl_give isl_multi_pw_aff *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::multi_pw_aff add(isl::multi_pw_aff multi2) const;
  inline isl::multi_pw_aff add_constant(isl::multi_val mv) const;
  inline isl::multi_pw_aff add_constant(isl::val v) const;
  inline isl::multi_pw_aff add_dims(isl::dim type, unsigned int n) const;
  inline isl::multi_pw_aff align_params(isl::space model) const;
  inline isl::set bind(isl::multi_id tuple) const;
  inline isl::multi_pw_aff bind_domain(isl::multi_id tuple) const;
  inline isl::multi_pw_aff bind_domain_wrapped_domain(isl::multi_id tuple) const;
  inline isl::multi_pw_aff coalesce() const;
  inline isl_size dim(isl::dim type) const;
  inline isl::set domain() const;
  inline isl::multi_pw_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::map eq_map(isl::multi_pw_aff mpa2) const;
  inline isl::multi_pw_aff factor_range() const;
  inline int find_dim_by_id(isl::dim type, const isl::id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::multi_pw_aff flat_range_product(isl::multi_pw_aff multi2) const;
  inline isl::multi_pw_aff flatten_range() const;
  inline isl::multi_pw_aff from_range() const;
  inline isl::pw_aff get_at(int pos) const;
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline isl::space get_domain_space() const;
  inline uint32_t get_hash() const;
  inline isl::pw_aff_list get_list() const;
  inline isl::pw_aff get_pw_aff(int pos) const;
  inline isl::space get_space() const;
  inline isl::id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline isl::multi_pw_aff gist(isl::set set) const;
  inline isl::multi_pw_aff gist_params(isl::set set) const;
  inline boolean has_tuple_id(isl::dim type) const;
  static inline isl::multi_pw_aff identity(isl::space space);
  inline isl::multi_pw_aff identity() const;
  static inline isl::multi_pw_aff identity_on_domain(isl::space space);
  inline isl::multi_pw_aff insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::multi_pw_aff insert_domain(isl::space domain) const;
  inline isl::multi_pw_aff intersect_domain(isl::set domain) const;
  inline isl::multi_pw_aff intersect_params(isl::set set) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_nan() const;
  inline boolean involves_param(const isl::id &id) const;
  inline boolean involves_param(const isl::id_list &list) const;
  inline boolean is_cst() const;
  inline boolean is_equal(const isl::multi_pw_aff &mpa2) const;
  inline isl::map lex_ge_map(isl::multi_pw_aff mpa2) const;
  inline isl::map lex_gt_map(isl::multi_pw_aff mpa2) const;
  inline isl::map lex_le_map(isl::multi_pw_aff mpa2) const;
  inline isl::map lex_lt_map(isl::multi_pw_aff mpa2) const;
  inline isl::multi_pw_aff max(isl::multi_pw_aff multi2) const;
  inline isl::multi_val max_multi_val() const;
  inline isl::multi_pw_aff min(isl::multi_pw_aff multi2) const;
  inline isl::multi_val min_multi_val() const;
  inline isl::multi_pw_aff mod_multi_val(isl::multi_val mv) const;
  inline isl::multi_pw_aff move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline isl::multi_pw_aff neg() const;
  inline boolean plain_is_equal(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_pw_aff product(isl::multi_pw_aff multi2) const;
  inline isl::multi_pw_aff project_domain_on_params() const;
  inline isl::multi_pw_aff pullback(isl::multi_aff ma) const;
  inline isl::multi_pw_aff pullback(isl::multi_pw_aff mpa2) const;
  inline isl::multi_pw_aff pullback(isl::pw_multi_aff pma) const;
  inline isl::multi_pw_aff range_factor_domain() const;
  inline isl::multi_pw_aff range_factor_range() const;
  inline boolean range_is_wrapping() const;
  inline isl::multi_pw_aff range_product(isl::multi_pw_aff multi2) const;
  inline isl::multi_pw_aff range_splice(unsigned int pos, isl::multi_pw_aff multi2) const;
  inline isl::multi_pw_aff reset_tuple_id(isl::dim type) const;
  inline isl::multi_pw_aff reset_user() const;
  inline isl::multi_pw_aff scale(isl::multi_val mv) const;
  inline isl::multi_pw_aff scale(isl::val v) const;
  inline isl::multi_pw_aff scale_down(isl::multi_val mv) const;
  inline isl::multi_pw_aff scale_down(isl::val v) const;
  inline isl::multi_pw_aff set_at(int pos, isl::pw_aff el) const;
  inline isl::multi_pw_aff set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::multi_pw_aff set_pw_aff(int pos, isl::pw_aff el) const;
  inline isl::multi_pw_aff set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::multi_pw_aff set_tuple_name(isl::dim type, const std::string &s) const;
  inline isl_size size() const;
  inline isl::multi_pw_aff splice(unsigned int in_pos, unsigned int out_pos, isl::multi_pw_aff multi2) const;
  inline isl::multi_pw_aff sub(isl::multi_pw_aff multi2) const;
  inline isl::multi_pw_aff unbind_params_insert_domain(isl::multi_id domain) const;
  inline isl::multi_pw_aff union_add(isl::multi_pw_aff mpa2) const;
  static inline isl::multi_pw_aff zero(isl::space space);
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
  inline /* implicit */ multi_union_pw_aff(isl::multi_pw_aff mpa);
  inline /* implicit */ multi_union_pw_aff(isl::union_pw_aff upa);
  inline explicit multi_union_pw_aff(isl::space space, isl::union_pw_aff_list list);
  inline explicit multi_union_pw_aff(isl::union_pw_multi_aff upma);
  inline explicit multi_union_pw_aff(isl::ctx ctx, const std::string &str);
  inline multi_union_pw_aff &operator=(multi_union_pw_aff obj);
  inline ~multi_union_pw_aff();
  inline __isl_give isl_multi_union_pw_aff *copy() const &;
  inline __isl_give isl_multi_union_pw_aff *copy() && = delete;
  inline __isl_keep isl_multi_union_pw_aff *get() const;
  inline __isl_give isl_multi_union_pw_aff *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::multi_union_pw_aff add(isl::multi_union_pw_aff multi2) const;
  inline isl::multi_union_pw_aff align_params(isl::space model) const;
  inline isl::union_pw_aff apply_aff(isl::aff aff) const;
  inline isl::union_pw_aff apply_pw_aff(isl::pw_aff pa) const;
  inline isl::multi_union_pw_aff apply_pw_multi_aff(isl::pw_multi_aff pma) const;
  inline isl::union_set bind(isl::multi_id tuple) const;
  inline isl::multi_union_pw_aff coalesce() const;
  inline isl_size dim(isl::dim type) const;
  inline isl::union_set domain() const;
  inline isl::multi_union_pw_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::multi_pw_aff extract_multi_pw_aff(isl::space space) const;
  inline isl::multi_union_pw_aff factor_range() const;
  inline int find_dim_by_id(isl::dim type, const isl::id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::multi_union_pw_aff flat_range_product(isl::multi_union_pw_aff multi2) const;
  inline isl::multi_union_pw_aff flatten_range() const;
  inline isl::multi_union_pw_aff floor() const;
  static inline isl::multi_union_pw_aff from_multi_aff(isl::multi_aff ma);
  inline isl::multi_union_pw_aff from_range() const;
  static inline isl::multi_union_pw_aff from_union_map(isl::union_map umap);
  inline isl::union_pw_aff get_at(int pos) const;
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline isl::space get_domain_space() const;
  inline isl::union_pw_aff_list get_list() const;
  inline isl::space get_space() const;
  inline isl::id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline isl::union_pw_aff get_union_pw_aff(int pos) const;
  inline isl::multi_union_pw_aff gist(isl::union_set context) const;
  inline isl::multi_union_pw_aff gist_params(isl::set context) const;
  inline boolean has_tuple_id(isl::dim type) const;
  inline isl::multi_union_pw_aff intersect_domain(isl::union_set uset) const;
  inline isl::multi_union_pw_aff intersect_params(isl::set params) const;
  inline isl::multi_union_pw_aff intersect_range(isl::set set) const;
  inline boolean involves_nan() const;
  inline isl::multi_val max_multi_val() const;
  inline isl::multi_val min_multi_val() const;
  inline isl::multi_union_pw_aff mod_multi_val(isl::multi_val mv) const;
  static inline isl::multi_union_pw_aff multi_aff_on_domain(isl::union_set domain, isl::multi_aff ma);
  static inline isl::multi_union_pw_aff multi_val_on_domain(isl::union_set domain, isl::multi_val mv);
  inline isl::multi_union_pw_aff neg() const;
  inline boolean plain_is_equal(const isl::multi_union_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff pullback(isl::union_pw_multi_aff upma) const;
  static inline isl::multi_union_pw_aff pw_multi_aff_on_domain(isl::union_set domain, isl::pw_multi_aff pma);
  inline isl::multi_union_pw_aff range_factor_domain() const;
  inline isl::multi_union_pw_aff range_factor_range() const;
  inline boolean range_is_wrapping() const;
  inline isl::multi_union_pw_aff range_product(isl::multi_union_pw_aff multi2) const;
  inline isl::multi_union_pw_aff range_splice(unsigned int pos, isl::multi_union_pw_aff multi2) const;
  inline isl::multi_union_pw_aff reset_tuple_id(isl::dim type) const;
  inline isl::multi_union_pw_aff reset_user() const;
  inline isl::multi_union_pw_aff scale(isl::multi_val mv) const;
  inline isl::multi_union_pw_aff scale(isl::val v) const;
  inline isl::multi_union_pw_aff scale_down(isl::multi_val mv) const;
  inline isl::multi_union_pw_aff scale_down(isl::val v) const;
  inline isl::multi_union_pw_aff set_at(int pos, isl::union_pw_aff el) const;
  inline isl::multi_union_pw_aff set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::multi_union_pw_aff set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::multi_union_pw_aff set_tuple_name(isl::dim type, const std::string &s) const;
  inline isl::multi_union_pw_aff set_union_pw_aff(int pos, isl::union_pw_aff el) const;
  inline isl_size size() const;
  inline isl::multi_union_pw_aff sub(isl::multi_union_pw_aff multi2) const;
  inline isl::multi_union_pw_aff union_add(isl::multi_union_pw_aff mupa2) const;
  static inline isl::multi_union_pw_aff zero(isl::space space);
  inline isl::union_set zero_union_set() const;
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
  inline explicit multi_val(isl::space space, isl::val_list list);
  inline explicit multi_val(isl::ctx ctx, const std::string &str);
  inline multi_val &operator=(multi_val obj);
  inline ~multi_val();
  inline __isl_give isl_multi_val *copy() const &;
  inline __isl_give isl_multi_val *copy() && = delete;
  inline __isl_keep isl_multi_val *get() const;
  inline __isl_give isl_multi_val *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::multi_val add(isl::multi_val multi2) const;
  inline isl::multi_val add(isl::val v) const;
  inline isl::multi_val add_dims(isl::dim type, unsigned int n) const;
  inline isl::multi_val align_params(isl::space model) const;
  inline isl_size dim(isl::dim type) const;
  inline isl::multi_val drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::multi_val factor_range() const;
  inline int find_dim_by_id(isl::dim type, const isl::id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::multi_val flat_range_product(isl::multi_val multi2) const;
  inline isl::multi_val flatten_range() const;
  inline isl::multi_val from_range() const;
  inline isl::val get_at(int pos) const;
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline isl::space get_domain_space() const;
  inline isl::val_list get_list() const;
  inline isl::space get_space() const;
  inline isl::id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline isl::val get_val(int pos) const;
  inline boolean has_tuple_id(isl::dim type) const;
  inline isl::multi_val insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_nan() const;
  inline boolean is_zero() const;
  inline isl::multi_val max(isl::multi_val multi2) const;
  inline isl::multi_val min(isl::multi_val multi2) const;
  inline isl::multi_val mod_multi_val(isl::multi_val mv) const;
  inline isl::multi_val mod_val(isl::val v) const;
  inline isl::multi_val neg() const;
  inline boolean plain_is_equal(const isl::multi_val &multi2) const;
  inline isl::multi_val product(isl::multi_val multi2) const;
  inline isl::multi_val project_domain_on_params() const;
  inline isl::multi_val range_factor_domain() const;
  inline isl::multi_val range_factor_range() const;
  inline boolean range_is_wrapping() const;
  inline isl::multi_val range_product(isl::multi_val multi2) const;
  inline isl::multi_val range_splice(unsigned int pos, isl::multi_val multi2) const;
  inline isl::multi_val reset_tuple_id(isl::dim type) const;
  inline isl::multi_val reset_user() const;
  inline isl::multi_val scale(isl::multi_val mv) const;
  inline isl::multi_val scale(isl::val v) const;
  inline isl::multi_val scale_down(isl::multi_val mv) const;
  inline isl::multi_val scale_down(isl::val v) const;
  inline isl::multi_val set_at(int pos, isl::val el) const;
  inline isl::multi_val set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::multi_val set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::multi_val set_tuple_name(isl::dim type, const std::string &s) const;
  inline isl::multi_val set_val(int pos, isl::val el) const;
  inline isl_size size() const;
  inline isl::multi_val splice(unsigned int in_pos, unsigned int out_pos, isl::multi_val multi2) const;
  inline isl::multi_val sub(isl::multi_val multi2) const;
  static inline isl::multi_val zero(isl::space space);
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
  inline explicit point(isl::space dim);
  inline point &operator=(point obj);
  inline ~point();
  inline __isl_give isl_point *copy() const &;
  inline __isl_give isl_point *copy() && = delete;
  inline __isl_keep isl_point *get() const;
  inline __isl_give isl_point *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::point add_ui(isl::dim type, int pos, unsigned int val) const;
  inline isl::val get_coordinate_val(isl::dim type, int pos) const;
  inline isl::multi_val get_multi_val() const;
  inline isl::space get_space() const;
  inline isl::point set_coordinate_val(isl::dim type, int pos, isl::val v) const;
  inline isl::point sub_ui(isl::dim type, int pos, unsigned int val) const;
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
  inline /* implicit */ pw_aff(isl::aff aff);
  inline explicit pw_aff(isl::ctx ctx, const std::string &str);
  inline explicit pw_aff(isl::set domain, isl::val v);
  inline explicit pw_aff(isl::local_space ls);
  inline pw_aff &operator=(pw_aff obj);
  inline ~pw_aff();
  inline __isl_give isl_pw_aff *copy() const &;
  inline __isl_give isl_pw_aff *copy() && = delete;
  inline __isl_keep isl_pw_aff *get() const;
  inline __isl_give isl_pw_aff *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::pw_aff add(isl::pw_aff pwaff2) const;
  inline isl::pw_aff add_constant(isl::val v) const;
  inline isl::pw_aff add_dims(isl::dim type, unsigned int n) const;
  inline isl::pw_aff align_params(isl::space model) const;
  static inline isl::pw_aff alloc(isl::set set, isl::aff aff);
  inline isl::aff as_aff() const;
  inline isl::set bind(isl::id id) const;
  inline isl::pw_aff bind_domain(isl::multi_id tuple) const;
  inline isl::pw_aff bind_domain_wrapped_domain(isl::multi_id tuple) const;
  inline isl::pw_aff ceil() const;
  inline isl::pw_aff coalesce() const;
  inline isl::pw_aff cond(isl::pw_aff pwaff_true, isl::pw_aff pwaff_false) const;
  inline isl_size dim(isl::dim type) const;
  inline isl::pw_aff div(isl::pw_aff pa2) const;
  inline isl::set domain() const;
  inline isl::pw_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::pw_aff drop_unused_params() const;
  static inline isl::pw_aff empty(isl::space space);
  inline isl::map eq_map(isl::pw_aff pa2) const;
  inline isl::set eq_set(isl::pw_aff pwaff2) const;
  inline isl::val eval(isl::point pnt) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::pw_aff floor() const;
  inline stat foreach_piece(const std::function<stat(set, aff)> &fn) const;
  inline isl::pw_aff from_range() const;
  inline isl::map ge_map(isl::pw_aff pa2) const;
  inline isl::set ge_set(isl::pw_aff pwaff2) const;
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline isl::space get_domain_space() const;
  inline uint32_t get_hash() const;
  inline isl::space get_space() const;
  inline isl::id get_tuple_id(isl::dim type) const;
  inline isl::pw_aff gist(isl::set context) const;
  inline isl::pw_aff gist_params(isl::set context) const;
  inline isl::map gt_map(isl::pw_aff pa2) const;
  inline isl::set gt_set(isl::pw_aff pwaff2) const;
  inline boolean has_dim_id(isl::dim type, unsigned int pos) const;
  inline boolean has_tuple_id(isl::dim type) const;
  inline isl::pw_aff insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::pw_aff insert_domain(isl::space domain) const;
  inline isl::pw_aff intersect_domain(isl::set set) const;
  inline isl::pw_aff intersect_domain_wrapped_domain(isl::set set) const;
  inline isl::pw_aff intersect_domain_wrapped_range(isl::set set) const;
  inline isl::pw_aff intersect_params(isl::set set) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_nan() const;
  inline boolean involves_param_id(const isl::id &id) const;
  inline boolean is_cst() const;
  inline boolean is_empty() const;
  inline boolean is_equal(const isl::pw_aff &pa2) const;
  inline boolean isa_aff() const;
  inline isl::map le_map(isl::pw_aff pa2) const;
  inline isl::set le_set(isl::pw_aff pwaff2) const;
  inline isl::map lt_map(isl::pw_aff pa2) const;
  inline isl::set lt_set(isl::pw_aff pwaff2) const;
  inline isl::pw_aff max(isl::pw_aff pwaff2) const;
  inline isl::pw_aff min(isl::pw_aff pwaff2) const;
  inline isl::pw_aff mod(isl::val mod) const;
  inline isl::pw_aff move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline isl::pw_aff mul(isl::pw_aff pwaff2) const;
  inline isl_size n_piece() const;
  static inline isl::pw_aff nan_on_domain(isl::local_space ls);
  static inline isl::pw_aff nan_on_domain_space(isl::space space);
  inline isl::set ne_set(isl::pw_aff pwaff2) const;
  inline isl::pw_aff neg() const;
  inline isl::set non_zero_set() const;
  inline isl::set nonneg_set() const;
  static inline isl::pw_aff param_on_domain(isl::set domain, isl::id id);
  inline isl::set params() const;
  inline int plain_cmp(const isl::pw_aff &pa2) const;
  inline boolean plain_is_equal(const isl::pw_aff &pwaff2) const;
  inline isl::set pos_set() const;
  inline isl::pw_aff project_domain_on_params() const;
  inline isl::pw_aff pullback(isl::multi_aff ma) const;
  inline isl::pw_aff pullback(isl::multi_pw_aff mpa) const;
  inline isl::pw_aff pullback(isl::pw_multi_aff pma) const;
  inline isl::pw_aff reset_tuple_id(isl::dim type) const;
  inline isl::pw_aff reset_user() const;
  inline isl::pw_aff scale(isl::val v) const;
  inline isl::pw_aff scale_down(isl::val f) const;
  inline isl::pw_aff set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::pw_aff set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::pw_aff sub(isl::pw_aff pwaff2) const;
  inline isl::pw_aff subtract_domain(isl::set set) const;
  inline isl::pw_aff tdiv_q(isl::pw_aff pa2) const;
  inline isl::pw_aff tdiv_r(isl::pw_aff pa2) const;
  inline isl::pw_aff union_add(isl::pw_aff pwaff2) const;
  inline isl::pw_aff union_max(isl::pw_aff pwaff2) const;
  inline isl::pw_aff union_min(isl::pw_aff pwaff2) const;
  static inline isl::pw_aff var_on_domain(isl::local_space ls, isl::dim type, unsigned int pos);
  inline isl::set zero_set() const;
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
  inline pw_aff_list &operator=(pw_aff_list obj);
  inline ~pw_aff_list();
  inline __isl_give isl_pw_aff_list *copy() const &;
  inline __isl_give isl_pw_aff_list *copy() && = delete;
  inline __isl_keep isl_pw_aff_list *get() const;
  inline __isl_give isl_pw_aff_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::pw_aff_list add(isl::pw_aff el) const;
  static inline isl::pw_aff_list alloc(isl::ctx ctx, int n);
  inline isl::pw_aff_list clear() const;
  inline isl::pw_aff_list concat(isl::pw_aff_list list2) const;
  inline isl::pw_aff_list drop(unsigned int first, unsigned int n) const;
  inline isl::set eq_set(isl::pw_aff_list list2) const;
  inline stat foreach(const std::function<stat(pw_aff)> &fn) const;
  static inline isl::pw_aff_list from_pw_aff(isl::pw_aff el);
  inline isl::set ge_set(isl::pw_aff_list list2) const;
  inline isl::pw_aff get_at(int index) const;
  inline isl::pw_aff get_pw_aff(int index) const;
  inline isl::set gt_set(isl::pw_aff_list list2) const;
  inline isl::pw_aff_list insert(unsigned int pos, isl::pw_aff el) const;
  inline isl::set le_set(isl::pw_aff_list list2) const;
  inline isl::set lt_set(isl::pw_aff_list list2) const;
  inline isl::pw_aff max() const;
  inline isl::pw_aff min() const;
  inline isl_size n_pw_aff() const;
  inline isl::set ne_set(isl::pw_aff_list list2) const;
  inline isl::pw_aff_list reverse() const;
  inline isl::pw_aff_list set_pw_aff(int index, isl::pw_aff el) const;
  inline isl_size size() const;
  inline isl::pw_aff_list swap(unsigned int pos1, unsigned int pos2) const;
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
  inline /* implicit */ pw_multi_aff(isl::multi_aff ma);
  inline /* implicit */ pw_multi_aff(isl::pw_aff pa);
  inline explicit pw_multi_aff(isl::ctx ctx, const std::string &str);
  inline pw_multi_aff &operator=(pw_multi_aff obj);
  inline ~pw_multi_aff();
  inline __isl_give isl_pw_multi_aff *copy() const &;
  inline __isl_give isl_pw_multi_aff *copy() && = delete;
  inline __isl_keep isl_pw_multi_aff *get() const;
  inline __isl_give isl_pw_multi_aff *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::pw_multi_aff add(isl::pw_multi_aff pma2) const;
  inline isl::pw_multi_aff add_constant(isl::multi_val mv) const;
  inline isl::pw_multi_aff add_constant(isl::val v) const;
  inline isl::pw_multi_aff align_params(isl::space model) const;
  static inline isl::pw_multi_aff alloc(isl::set set, isl::multi_aff maff);
  inline isl::multi_aff as_multi_aff() const;
  inline isl::pw_multi_aff bind_domain(isl::multi_id tuple) const;
  inline isl::pw_multi_aff bind_domain_wrapped_domain(isl::multi_id tuple) const;
  inline isl::pw_multi_aff coalesce() const;
  inline isl_size dim(isl::dim type) const;
  inline isl::set domain() const;
  static inline isl::pw_multi_aff domain_map(isl::space space);
  inline isl::pw_multi_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::pw_multi_aff drop_unused_params() const;
  static inline isl::pw_multi_aff empty(isl::space space);
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::pw_multi_aff fix_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::pw_multi_aff flat_range_product(isl::pw_multi_aff pma2) const;
  inline stat foreach_piece(const std::function<stat(set, multi_aff)> &fn) const;
  static inline isl::pw_multi_aff from_domain(isl::set set);
  static inline isl::pw_multi_aff from_map(isl::map map);
  static inline isl::pw_multi_aff from_multi_pw_aff(isl::multi_pw_aff mpa);
  static inline isl::pw_multi_aff from_set(isl::set set);
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline isl::space get_domain_space() const;
  inline isl::pw_aff get_pw_aff(int pos) const;
  inline isl::space get_space() const;
  inline isl::id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline isl::pw_multi_aff gist(isl::set set) const;
  inline isl::pw_multi_aff gist_params(isl::set set) const;
  inline boolean has_tuple_id(isl::dim type) const;
  inline boolean has_tuple_name(isl::dim type) const;
  static inline isl::pw_multi_aff identity(isl::space space);
  static inline isl::pw_multi_aff identity_on_domain(isl::space space);
  inline isl::pw_multi_aff insert_domain(isl::space domain) const;
  inline isl::pw_multi_aff intersect_domain(isl::set set) const;
  inline isl::pw_multi_aff intersect_domain_wrapped_domain(isl::set set) const;
  inline isl::pw_multi_aff intersect_domain_wrapped_range(isl::set set) const;
  inline isl::pw_multi_aff intersect_params(isl::set set) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_locals() const;
  inline boolean involves_nan() const;
  inline boolean involves_param_id(const isl::id &id) const;
  inline boolean is_equal(const isl::pw_multi_aff &pma2) const;
  inline boolean isa_multi_aff() const;
  inline isl::multi_val max_multi_val() const;
  inline isl::multi_val min_multi_val() const;
  static inline isl::pw_multi_aff multi_val_on_domain(isl::set domain, isl::multi_val mv);
  inline isl_size n_piece() const;
  inline isl::pw_multi_aff neg() const;
  inline boolean plain_is_equal(const isl::pw_multi_aff &pma2) const;
  inline isl::pw_multi_aff preimage_domain_wrapped_domain(isl::pw_multi_aff pma2) const;
  inline isl::pw_multi_aff product(isl::pw_multi_aff pma2) const;
  inline isl::pw_multi_aff project_domain_on_params() const;
  static inline isl::pw_multi_aff project_out_map(isl::space space, isl::dim type, unsigned int first, unsigned int n);
  inline isl::pw_multi_aff pullback(isl::multi_aff ma) const;
  inline isl::pw_multi_aff pullback(isl::pw_multi_aff pma2) const;
  inline isl::pw_multi_aff range_factor_domain() const;
  inline isl::pw_multi_aff range_factor_range() const;
  static inline isl::pw_multi_aff range_map(isl::space space);
  inline isl::pw_multi_aff range_product(isl::pw_multi_aff pma2) const;
  inline isl::pw_multi_aff reset_tuple_id(isl::dim type) const;
  inline isl::pw_multi_aff reset_user() const;
  inline isl::pw_multi_aff scale(isl::val v) const;
  inline isl::pw_multi_aff scale_down(isl::val v) const;
  inline isl::pw_multi_aff scale_multi_val(isl::multi_val mv) const;
  inline isl::pw_multi_aff set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::pw_multi_aff set_pw_aff(unsigned int pos, isl::pw_aff pa) const;
  inline isl::pw_multi_aff set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::pw_multi_aff sub(isl::pw_multi_aff pma2) const;
  inline isl::pw_multi_aff subtract_domain(isl::set set) const;
  inline isl::pw_multi_aff union_add(isl::pw_multi_aff pma2) const;
  inline isl::pw_multi_aff union_lexmax(isl::pw_multi_aff pma2) const;
  inline isl::pw_multi_aff union_lexmin(isl::pw_multi_aff pma2) const;
  static inline isl::pw_multi_aff zero(isl::space space);
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
  inline pw_multi_aff_list &operator=(pw_multi_aff_list obj);
  inline ~pw_multi_aff_list();
  inline __isl_give isl_pw_multi_aff_list *copy() const &;
  inline __isl_give isl_pw_multi_aff_list *copy() && = delete;
  inline __isl_keep isl_pw_multi_aff_list *get() const;
  inline __isl_give isl_pw_multi_aff_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::pw_multi_aff_list add(isl::pw_multi_aff el) const;
  static inline isl::pw_multi_aff_list alloc(isl::ctx ctx, int n);
  inline isl::pw_multi_aff_list clear() const;
  inline isl::pw_multi_aff_list concat(isl::pw_multi_aff_list list2) const;
  inline isl::pw_multi_aff_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(pw_multi_aff)> &fn) const;
  static inline isl::pw_multi_aff_list from_pw_multi_aff(isl::pw_multi_aff el);
  inline isl::pw_multi_aff get_at(int index) const;
  inline isl::pw_multi_aff get_pw_multi_aff(int index) const;
  inline isl::pw_multi_aff_list insert(unsigned int pos, isl::pw_multi_aff el) const;
  inline isl_size n_pw_multi_aff() const;
  inline isl::pw_multi_aff_list reverse() const;
  inline isl::pw_multi_aff_list set_pw_multi_aff(int index, isl::pw_multi_aff el) const;
  inline isl_size size() const;
  inline isl::pw_multi_aff_list swap(unsigned int pos1, unsigned int pos2) const;
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
  inline explicit pw_qpolynomial(isl::ctx ctx, const std::string &str);
  inline pw_qpolynomial &operator=(pw_qpolynomial obj);
  inline ~pw_qpolynomial();
  inline __isl_give isl_pw_qpolynomial *copy() const &;
  inline __isl_give isl_pw_qpolynomial *copy() && = delete;
  inline __isl_keep isl_pw_qpolynomial *get() const;
  inline __isl_give isl_pw_qpolynomial *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::pw_qpolynomial add(isl::pw_qpolynomial pwqp2) const;
  inline isl::pw_qpolynomial add_dims(isl::dim type, unsigned int n) const;
  static inline isl::pw_qpolynomial alloc(isl::set set, isl::qpolynomial qp);
  inline isl::qpolynomial as_qpolynomial() const;
  inline isl::pw_qpolynomial coalesce() const;
  inline isl_size dim(isl::dim type) const;
  inline isl::set domain() const;
  inline isl::pw_qpolynomial drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::pw_qpolynomial drop_unused_params() const;
  inline isl::val eval(isl::point pnt) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::pw_qpolynomial fix_val(isl::dim type, unsigned int n, isl::val v) const;
  inline stat foreach_piece(const std::function<stat(set, qpolynomial)> &fn) const;
  static inline isl::pw_qpolynomial from_pw_aff(isl::pw_aff pwaff);
  static inline isl::pw_qpolynomial from_qpolynomial(isl::qpolynomial qp);
  inline isl::pw_qpolynomial from_range() const;
  inline isl::space get_domain_space() const;
  inline isl::space get_space() const;
  inline isl::pw_qpolynomial gist(isl::set context) const;
  inline isl::pw_qpolynomial gist_params(isl::set context) const;
  inline boolean has_equal_space(const isl::pw_qpolynomial &pwqp2) const;
  inline isl::pw_qpolynomial insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::pw_qpolynomial intersect_domain(isl::set set) const;
  inline isl::pw_qpolynomial intersect_domain_wrapped_domain(isl::set set) const;
  inline isl::pw_qpolynomial intersect_domain_wrapped_range(isl::set set) const;
  inline isl::pw_qpolynomial intersect_params(isl::set set) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_nan() const;
  inline boolean involves_param_id(const isl::id &id) const;
  inline boolean is_zero() const;
  inline boolean isa_qpolynomial() const;
  inline isl::val max() const;
  inline isl::val min() const;
  inline isl::pw_qpolynomial move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline isl::pw_qpolynomial mul(isl::pw_qpolynomial pwqp2) const;
  inline isl_size n_piece() const;
  inline isl::pw_qpolynomial neg() const;
  inline boolean plain_is_equal(const isl::pw_qpolynomial &pwqp2) const;
  inline isl::pw_qpolynomial pow(unsigned int exponent) const;
  inline isl::pw_qpolynomial project_domain_on_params() const;
  inline isl::pw_qpolynomial reset_domain_space(isl::space space) const;
  inline isl::pw_qpolynomial reset_user() const;
  inline isl::pw_qpolynomial scale_down_val(isl::val v) const;
  inline isl::pw_qpolynomial scale_val(isl::val v) const;
  inline isl::pw_qpolynomial split_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::pw_qpolynomial split_periods(int max_periods) const;
  inline isl::pw_qpolynomial sub(isl::pw_qpolynomial pwqp2) const;
  inline isl::pw_qpolynomial subtract_domain(isl::set set) const;
  inline isl::pw_qpolynomial to_polynomial(int sign) const;
  static inline isl::pw_qpolynomial zero(isl::space space);
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
  inline pw_qpolynomial_fold_list &operator=(pw_qpolynomial_fold_list obj);
  inline ~pw_qpolynomial_fold_list();
  inline __isl_give isl_pw_qpolynomial_fold_list *copy() const &;
  inline __isl_give isl_pw_qpolynomial_fold_list *copy() && = delete;
  inline __isl_keep isl_pw_qpolynomial_fold_list *get() const;
  inline __isl_give isl_pw_qpolynomial_fold_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
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
  inline pw_qpolynomial_list &operator=(pw_qpolynomial_list obj);
  inline ~pw_qpolynomial_list();
  inline __isl_give isl_pw_qpolynomial_list *copy() const &;
  inline __isl_give isl_pw_qpolynomial_list *copy() && = delete;
  inline __isl_keep isl_pw_qpolynomial_list *get() const;
  inline __isl_give isl_pw_qpolynomial_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::pw_qpolynomial_list add(isl::pw_qpolynomial el) const;
  static inline isl::pw_qpolynomial_list alloc(isl::ctx ctx, int n);
  inline isl::pw_qpolynomial_list clear() const;
  inline isl::pw_qpolynomial_list concat(isl::pw_qpolynomial_list list2) const;
  inline isl::pw_qpolynomial_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(pw_qpolynomial)> &fn) const;
  static inline isl::pw_qpolynomial_list from_pw_qpolynomial(isl::pw_qpolynomial el);
  inline isl::pw_qpolynomial get_at(int index) const;
  inline isl::pw_qpolynomial get_pw_qpolynomial(int index) const;
  inline isl::pw_qpolynomial_list insert(unsigned int pos, isl::pw_qpolynomial el) const;
  inline isl_size n_pw_qpolynomial() const;
  inline isl::pw_qpolynomial_list reverse() const;
  inline isl::pw_qpolynomial_list set_pw_qpolynomial(int index, isl::pw_qpolynomial el) const;
  inline isl_size size() const;
  inline isl::pw_qpolynomial_list swap(unsigned int pos1, unsigned int pos2) const;
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
  inline qpolynomial &operator=(qpolynomial obj);
  inline ~qpolynomial();
  inline __isl_give isl_qpolynomial *copy() const &;
  inline __isl_give isl_qpolynomial *copy() && = delete;
  inline __isl_keep isl_qpolynomial *get() const;
  inline __isl_give isl_qpolynomial *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::qpolynomial add(isl::qpolynomial qp2) const;
  inline isl::qpolynomial add_dims(isl::dim type, unsigned int n) const;
  inline isl::qpolynomial align_params(isl::space model) const;
  inline stat as_polynomial_on_domain(const isl::basic_set &bset, const std::function<stat(basic_set, qpolynomial)> &fn) const;
  inline isl_size dim(isl::dim type) const;
  inline isl::qpolynomial drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::val eval(isl::point pnt) const;
  inline stat foreach_term(const std::function<stat(term)> &fn) const;
  static inline isl::qpolynomial from_aff(isl::aff aff);
  static inline isl::qpolynomial from_constraint(isl::constraint c, isl::dim type, unsigned int pos);
  static inline isl::qpolynomial from_term(isl::term term);
  inline isl::val get_constant_val() const;
  inline isl::space get_domain_space() const;
  inline isl::space get_space() const;
  inline isl::qpolynomial gist(isl::set context) const;
  inline isl::qpolynomial gist_params(isl::set context) const;
  inline isl::qpolynomial homogenize() const;
  static inline isl::qpolynomial infty_on_domain(isl::space domain);
  inline isl::qpolynomial insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean is_infty() const;
  inline boolean is_nan() const;
  inline boolean is_neginfty() const;
  inline boolean is_zero() const;
  inline isl::qpolynomial move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline isl::qpolynomial mul(isl::qpolynomial qp2) const;
  static inline isl::qpolynomial nan_on_domain(isl::space domain);
  inline isl::qpolynomial neg() const;
  static inline isl::qpolynomial neginfty_on_domain(isl::space domain);
  static inline isl::qpolynomial one_on_domain(isl::space domain);
  inline boolean plain_is_equal(const isl::qpolynomial &qp2) const;
  inline isl::qpolynomial pow(unsigned int power) const;
  inline isl::qpolynomial project_domain_on_params() const;
  inline isl::qpolynomial scale_down_val(isl::val v) const;
  inline isl::qpolynomial scale_val(isl::val v) const;
  inline int sgn() const;
  inline isl::qpolynomial sub(isl::qpolynomial qp2) const;
  static inline isl::qpolynomial val_on_domain(isl::space space, isl::val val);
  static inline isl::qpolynomial var_on_domain(isl::space domain, isl::dim type, unsigned int pos);
  static inline isl::qpolynomial zero_on_domain(isl::space domain);
};

// declarations for isl::qpolynomial_list
inline qpolynomial_list manage(__isl_take isl_qpolynomial_list *ptr);
inline qpolynomial_list manage_copy(__isl_keep isl_qpolynomial_list *ptr);

class qpolynomial_list {
  friend inline qpolynomial_list manage(__isl_take isl_qpolynomial_list *ptr);
  friend inline qpolynomial_list manage_copy(__isl_keep isl_qpolynomial_list *ptr);

  isl_qpolynomial_list *ptr = nullptr;

  inline explicit qpolynomial_list(__isl_take isl_qpolynomial_list *ptr);

public:
  inline /* implicit */ qpolynomial_list();
  inline /* implicit */ qpolynomial_list(const qpolynomial_list &obj);
  inline qpolynomial_list &operator=(qpolynomial_list obj);
  inline ~qpolynomial_list();
  inline __isl_give isl_qpolynomial_list *copy() const &;
  inline __isl_give isl_qpolynomial_list *copy() && = delete;
  inline __isl_keep isl_qpolynomial_list *get() const;
  inline __isl_give isl_qpolynomial_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::qpolynomial_list add(isl::qpolynomial el) const;
  static inline isl::qpolynomial_list alloc(isl::ctx ctx, int n);
  inline isl::qpolynomial_list clear() const;
  inline isl::qpolynomial_list concat(isl::qpolynomial_list list2) const;
  inline isl::qpolynomial_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(qpolynomial)> &fn) const;
  static inline isl::qpolynomial_list from_qpolynomial(isl::qpolynomial el);
  inline isl::qpolynomial get_at(int index) const;
  inline isl::qpolynomial get_qpolynomial(int index) const;
  inline isl::qpolynomial_list insert(unsigned int pos, isl::qpolynomial el) const;
  inline isl_size n_qpolynomial() const;
  inline isl::qpolynomial_list reverse() const;
  inline isl::qpolynomial_list set_qpolynomial(int index, isl::qpolynomial el) const;
  inline isl_size size() const;
  inline isl::qpolynomial_list swap(unsigned int pos1, unsigned int pos2) const;
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
  inline explicit schedule(isl::ctx ctx, const std::string &str);
  inline schedule &operator=(schedule obj);
  inline ~schedule();
  inline __isl_give isl_schedule *copy() const &;
  inline __isl_give isl_schedule *copy() && = delete;
  inline __isl_keep isl_schedule *get() const;
  inline __isl_give isl_schedule *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::schedule align_params(isl::space space) const;
  static inline isl::schedule empty(isl::space space);
  static inline isl::schedule from_domain(isl::union_set domain);
  inline isl::union_set get_domain() const;
  inline isl::union_map get_map() const;
  inline isl::schedule_node get_root() const;
  inline isl::schedule gist_domain_params(isl::set context) const;
  inline isl::schedule insert_context(isl::set context) const;
  inline isl::schedule insert_guard(isl::set guard) const;
  inline isl::schedule insert_partial_schedule(isl::multi_union_pw_aff partial) const;
  inline isl::schedule intersect_domain(isl::union_set domain) const;
  inline boolean plain_is_equal(const isl::schedule &schedule2) const;
  inline isl::schedule pullback(isl::union_pw_multi_aff upma) const;
  inline isl::schedule reset_user() const;
  inline isl::schedule sequence(isl::schedule schedule2) const;
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
  inline explicit schedule_constraints(isl::ctx ctx, const std::string &str);
  inline schedule_constraints &operator=(schedule_constraints obj);
  inline ~schedule_constraints();
  inline __isl_give isl_schedule_constraints *copy() const &;
  inline __isl_give isl_schedule_constraints *copy() && = delete;
  inline __isl_keep isl_schedule_constraints *get() const;
  inline __isl_give isl_schedule_constraints *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::schedule_constraints apply(isl::union_map umap) const;
  inline isl::schedule compute_schedule() const;
  inline isl::union_map get_coincidence() const;
  inline isl::union_map get_conditional_validity() const;
  inline isl::union_map get_conditional_validity_condition() const;
  inline isl::set get_context() const;
  inline isl::union_set get_domain() const;
  inline isl::union_map get_proximity() const;
  inline isl::union_map get_validity() const;
  static inline isl::schedule_constraints on_domain(isl::union_set domain);
  inline isl::schedule_constraints set_coincidence(isl::union_map coincidence) const;
  inline isl::schedule_constraints set_conditional_validity(isl::union_map condition, isl::union_map validity) const;
  inline isl::schedule_constraints set_context(isl::set context) const;
  inline isl::schedule_constraints set_proximity(isl::union_map proximity) const;
  inline isl::schedule_constraints set_validity(isl::union_map validity) const;
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
  inline schedule_node &operator=(schedule_node obj);
  inline ~schedule_node();
  inline __isl_give isl_schedule_node *copy() const &;
  inline __isl_give isl_schedule_node *copy() && = delete;
  inline __isl_keep isl_schedule_node *get() const;
  inline __isl_give isl_schedule_node *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::schedule_node align_params(isl::space space) const;
  inline isl::schedule_node ancestor(int generation) const;
  inline boolean band_member_get_coincident(int pos) const;
  inline isl::schedule_node band_member_set_coincident(int pos, int coincident) const;
  inline isl::schedule_node band_set_ast_build_options(isl::union_set options) const;
  inline isl::schedule_node child(int pos) const;
  inline isl::set context_get_context() const;
  inline isl::schedule_node cut() const;
  inline isl::union_set domain_get_domain() const;
  inline isl::union_pw_multi_aff expansion_get_contraction() const;
  inline isl::union_map expansion_get_expansion() const;
  inline isl::union_map extension_get_extension() const;
  inline isl::union_set filter_get_filter() const;
  inline isl::schedule_node first_child() const;
  inline stat foreach_ancestor_top_down(const std::function<stat(schedule_node)> &fn) const;
  static inline isl::schedule_node from_domain(isl::union_set domain);
  static inline isl::schedule_node from_extension(isl::union_map extension);
  inline isl_size get_ancestor_child_position(const isl::schedule_node &ancestor) const;
  inline isl::schedule_node get_child(int pos) const;
  inline isl_size get_child_position() const;
  inline isl::union_set get_domain() const;
  inline isl::multi_union_pw_aff get_prefix_schedule_multi_union_pw_aff() const;
  inline isl::union_map get_prefix_schedule_relation() const;
  inline isl::union_map get_prefix_schedule_union_map() const;
  inline isl::union_pw_multi_aff get_prefix_schedule_union_pw_multi_aff() const;
  inline isl::schedule get_schedule() const;
  inline isl_size get_schedule_depth() const;
  inline isl::schedule_node get_shared_ancestor(const isl::schedule_node &node2) const;
  inline isl::union_pw_multi_aff get_subtree_contraction() const;
  inline isl::union_map get_subtree_expansion() const;
  inline isl::union_map get_subtree_schedule_union_map() const;
  inline isl_size get_tree_depth() const;
  inline isl::union_set get_universe_domain() const;
  inline isl::schedule_node graft_after(isl::schedule_node graft) const;
  inline isl::schedule_node graft_before(isl::schedule_node graft) const;
  inline isl::schedule_node group(isl::id group_id) const;
  inline isl::set guard_get_guard() const;
  inline boolean has_children() const;
  inline boolean has_next_sibling() const;
  inline boolean has_parent() const;
  inline boolean has_previous_sibling() const;
  inline isl::schedule_node insert_context(isl::set context) const;
  inline isl::schedule_node insert_filter(isl::union_set filter) const;
  inline isl::schedule_node insert_guard(isl::set context) const;
  inline isl::schedule_node insert_mark(isl::id mark) const;
  inline isl::schedule_node insert_partial_schedule(isl::multi_union_pw_aff schedule) const;
  inline isl::schedule_node insert_sequence(isl::union_set_list filters) const;
  inline isl::schedule_node insert_set(isl::union_set_list filters) const;
  inline boolean is_equal(const isl::schedule_node &node2) const;
  inline boolean is_subtree_anchored() const;
  inline isl::id mark_get_id() const;
  inline isl_size n_children() const;
  inline isl::schedule_node next_sibling() const;
  inline isl::schedule_node order_after(isl::union_set filter) const;
  inline isl::schedule_node order_before(isl::union_set filter) const;
  inline isl::schedule_node parent() const;
  inline isl::schedule_node previous_sibling() const;
  inline isl::schedule_node reset_user() const;
  inline isl::schedule_node root() const;
  inline isl::schedule_node sequence_splice_child(int pos) const;
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
  inline /* implicit */ set(isl::basic_set bset);
  inline /* implicit */ set(isl::point pnt);
  inline explicit set(isl::union_set uset);
  inline explicit set(isl::ctx ctx, const std::string &str);
  inline set &operator=(set obj);
  inline ~set();
  inline __isl_give isl_set *copy() const &;
  inline __isl_give isl_set *copy() && = delete;
  inline __isl_keep isl_set *get() const;
  inline __isl_give isl_set *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::set add_constraint(isl::constraint constraint) const;
  inline isl::set add_dims(isl::dim type, unsigned int n) const;
  inline isl::basic_set affine_hull() const;
  inline isl::set align_params(isl::space model) const;
  inline isl::set apply(isl::map map) const;
  inline isl::set bind(isl::multi_id tuple) const;
  inline isl::basic_set bounded_simple_hull() const;
  static inline isl::set box_from_points(isl::point pnt1, isl::point pnt2);
  inline isl::set coalesce() const;
  inline isl::basic_set coefficients() const;
  inline isl::set complement() const;
  inline isl::basic_set convex_hull() const;
  inline isl::val count_val() const;
  inline isl::set detect_equalities() const;
  inline isl_size dim(isl::dim type) const;
  inline boolean dim_has_any_lower_bound(isl::dim type, unsigned int pos) const;
  inline boolean dim_has_any_upper_bound(isl::dim type, unsigned int pos) const;
  inline boolean dim_has_lower_bound(isl::dim type, unsigned int pos) const;
  inline boolean dim_has_upper_bound(isl::dim type, unsigned int pos) const;
  inline boolean dim_is_bounded(isl::dim type, unsigned int pos) const;
  inline isl::pw_aff dim_max(int pos) const;
  inline isl::val dim_max_val(int pos) const;
  inline isl::pw_aff dim_min(int pos) const;
  inline isl::val dim_min_val(int pos) const;
  inline isl::set drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set drop_unused_params() const;
  inline isl::set eliminate(isl::dim type, unsigned int first, unsigned int n) const;
  static inline isl::set empty(isl::space space);
  inline isl::set equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline int find_dim_by_id(isl::dim type, const isl::id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::set fix_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::set fix_val(isl::dim type, unsigned int pos, isl::val v) const;
  inline isl::set flat_product(isl::set set2) const;
  inline isl::set flatten() const;
  inline isl::map flatten_map() const;
  inline int follows_at(const isl::set &set2, int pos) const;
  inline stat foreach_basic_set(const std::function<stat(basic_set)> &fn) const;
  inline stat foreach_point(const std::function<stat(point)> &fn) const;
  static inline isl::set from_multi_aff(isl::multi_aff ma);
  static inline isl::set from_multi_pw_aff(isl::multi_pw_aff mpa);
  inline isl::set from_params() const;
  static inline isl::set from_pw_aff(isl::pw_aff pwaff);
  static inline isl::set from_pw_multi_aff(isl::pw_multi_aff pma);
  inline isl::basic_set_list get_basic_set_list() const;
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline isl::multi_val get_plain_multi_val_if_fixed() const;
  inline isl::fixed_box get_simple_fixed_box_hull() const;
  inline isl::space get_space() const;
  inline isl::val get_stride(int pos) const;
  inline isl::id get_tuple_id() const;
  inline std::string get_tuple_name() const;
  inline isl::set gist(isl::set context) const;
  inline isl::set gist_basic_set(isl::basic_set context) const;
  inline isl::set gist_params(isl::set context) const;
  inline boolean has_dim_id(isl::dim type, unsigned int pos) const;
  inline boolean has_dim_name(isl::dim type, unsigned int pos) const;
  inline boolean has_equal_space(const isl::set &set2) const;
  inline boolean has_tuple_id() const;
  inline boolean has_tuple_name() const;
  inline isl::map identity() const;
  inline isl::pw_aff indicator_function() const;
  inline isl::set insert_dims(isl::dim type, unsigned int pos, unsigned int n) const;
  inline isl::map insert_domain(isl::space domain) const;
  inline isl::set intersect(isl::set set2) const;
  inline isl::set intersect_factor_domain(isl::set domain) const;
  inline isl::set intersect_factor_range(isl::set range) const;
  inline isl::set intersect_params(isl::set params) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_locals() const;
  inline boolean is_bounded() const;
  inline boolean is_box() const;
  inline boolean is_disjoint(const isl::set &set2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const isl::set &set2) const;
  inline boolean is_params() const;
  inline boolean is_singleton() const;
  inline boolean is_strict_subset(const isl::set &set2) const;
  inline boolean is_subset(const isl::set &set2) const;
  inline boolean is_wrapping() const;
  inline isl::map lex_ge_set(isl::set set2) const;
  inline isl::map lex_gt_set(isl::set set2) const;
  inline isl::map lex_lt_set(isl::set set2) const;
  inline isl::set lexmax() const;
  inline isl::pw_multi_aff lexmax_pw_multi_aff() const;
  inline isl::set lexmin() const;
  inline isl::pw_multi_aff lexmin_pw_multi_aff() const;
  inline isl::set lower_bound(isl::multi_pw_aff lower) const;
  inline isl::set lower_bound(isl::multi_val lower) const;
  inline isl::set lower_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::set lower_bound_val(isl::dim type, unsigned int pos, isl::val value) const;
  inline isl::multi_pw_aff max_multi_pw_aff() const;
  inline isl::val max_val(const isl::aff &obj) const;
  inline isl::multi_pw_aff min_multi_pw_aff() const;
  inline isl::val min_val(const isl::aff &obj) const;
  inline isl::set move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline isl_size n_basic_set() const;
  inline isl_size n_dim() const;
  static inline isl::set nat_universe(isl::space space);
  inline isl::set neg() const;
  inline isl::set params() const;
  inline int plain_cmp(const isl::set &set2) const;
  inline isl::val plain_get_val_if_fixed(isl::dim type, unsigned int pos) const;
  inline boolean plain_is_disjoint(const isl::set &set2) const;
  inline boolean plain_is_empty() const;
  inline boolean plain_is_equal(const isl::set &set2) const;
  inline boolean plain_is_universe() const;
  inline isl::basic_set plain_unshifted_simple_hull() const;
  inline isl::basic_set polyhedral_hull() const;
  inline isl::set preimage(isl::multi_aff ma) const;
  inline isl::set preimage(isl::multi_pw_aff mpa) const;
  inline isl::set preimage(isl::pw_multi_aff pma) const;
  inline isl::set product(isl::set set2) const;
  inline isl::map project_onto_map(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set project_out_all_params() const;
  inline isl::set project_out_param(isl::id id) const;
  inline isl::set project_out_param(isl::id_list list) const;
  inline isl::set remove_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set remove_divs() const;
  inline isl::set remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set remove_redundancies() const;
  inline isl::set remove_unknown_divs() const;
  inline isl::set reset_space(isl::space space) const;
  inline isl::set reset_tuple_id() const;
  inline isl::set reset_user() const;
  inline isl::basic_set sample() const;
  inline isl::point sample_point() const;
  inline isl::set set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::set set_tuple_id(isl::id id) const;
  inline isl::set set_tuple_name(const std::string &s) const;
  inline isl::basic_set simple_hull() const;
  inline int size() const;
  inline isl::basic_set solutions() const;
  inline isl::set split_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set subtract(isl::set set2) const;
  inline isl::set sum(isl::set set2) const;
  inline isl::map translation() const;
  inline isl_size tuple_dim() const;
  inline isl::set unbind_params(isl::multi_id tuple) const;
  inline isl::map unbind_params_insert_domain(isl::multi_id domain) const;
  inline isl::set unite(isl::set set2) const;
  static inline isl::set universe(isl::space space);
  inline isl::basic_set unshifted_simple_hull() const;
  inline isl::basic_set unshifted_simple_hull_from_set_list(isl::set_list list) const;
  inline isl::map unwrap() const;
  inline isl::set upper_bound(isl::multi_pw_aff upper) const;
  inline isl::set upper_bound(isl::multi_val upper) const;
  inline isl::set upper_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::set upper_bound_val(isl::dim type, unsigned int pos, isl::val value) const;
  inline isl::map wrapped_domain_map() const;
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
  inline set_list &operator=(set_list obj);
  inline ~set_list();
  inline __isl_give isl_set_list *copy() const &;
  inline __isl_give isl_set_list *copy() && = delete;
  inline __isl_keep isl_set_list *get() const;
  inline __isl_give isl_set_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::set_list add(isl::set el) const;
  static inline isl::set_list alloc(isl::ctx ctx, int n);
  inline isl::set_list clear() const;
  inline isl::set_list concat(isl::set_list list2) const;
  inline isl::set_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(set)> &fn) const;
  static inline isl::set_list from_set(isl::set el);
  inline isl::set get_at(int index) const;
  inline isl::set get_set(int index) const;
  inline isl::set_list insert(unsigned int pos, isl::set el) const;
  inline isl_size n_set() const;
  inline isl::set_list reverse() const;
  inline isl::set_list set_set(int index, isl::set el) const;
  inline isl_size size() const;
  inline isl::set_list swap(unsigned int pos1, unsigned int pos2) const;
  inline isl::set unite() const;
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
  inline explicit space(isl::ctx ctx, unsigned int nparam, unsigned int n_in, unsigned int n_out);
  inline explicit space(isl::ctx ctx, unsigned int nparam, unsigned int dim);
  inline space &operator=(space obj);
  inline ~space();
  inline __isl_give isl_space *copy() const &;
  inline __isl_give isl_space *copy() && = delete;
  inline __isl_keep isl_space *get() const;
  inline __isl_give isl_space *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::space add_dims(isl::dim type, unsigned int n) const;
  inline isl::space add_named_tuple(isl::id tuple_id, unsigned int dim) const;
  inline isl::space add_param_id(isl::id id) const;
  inline isl::space add_unnamed_tuple(unsigned int dim) const;
  inline isl::space align_params(isl::space space2) const;
  inline boolean can_curry() const;
  inline boolean can_range_curry() const;
  inline boolean can_uncurry() const;
  inline boolean can_zip() const;
  inline isl::space curry() const;
  inline isl_size dim(isl::dim type) const;
  inline isl::space domain() const;
  inline isl::space domain_factor_domain() const;
  inline isl::space domain_factor_range() const;
  inline boolean domain_is_wrapping() const;
  inline isl::space domain_map() const;
  inline isl::space domain_product(isl::space right) const;
  inline isl::space drop_all_params() const;
  inline isl::space drop_dims(isl::dim type, unsigned int first, unsigned int num) const;
  inline isl::space factor_domain() const;
  inline isl::space factor_range() const;
  inline int find_dim_by_id(isl::dim type, const isl::id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::space flatten_domain() const;
  inline isl::space flatten_range() const;
  inline isl::space from_domain() const;
  inline isl::space from_range() const;
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline isl::id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline boolean has_dim_id(isl::dim type, unsigned int pos) const;
  inline boolean has_dim_name(isl::dim type, unsigned int pos) const;
  inline boolean has_equal_params(const isl::space &space2) const;
  inline boolean has_equal_tuples(const isl::space &space2) const;
  inline boolean has_tuple_id(isl::dim type) const;
  inline boolean has_tuple_name(isl::dim type) const;
  inline isl::space insert_dims(isl::dim type, unsigned int pos, unsigned int n) const;
  inline boolean is_domain(const isl::space &space2) const;
  inline boolean is_equal(const isl::space &space2) const;
  inline boolean is_map() const;
  inline boolean is_params() const;
  inline boolean is_product() const;
  inline boolean is_range(const isl::space &space2) const;
  inline boolean is_set() const;
  inline boolean is_wrapping() const;
  inline isl::space join(isl::space right) const;
  inline isl::space map_from_domain_and_range(isl::space range) const;
  inline isl::space map_from_set() const;
  inline isl::space move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline isl::space params() const;
  static inline isl::space params_alloc(isl::ctx ctx, unsigned int nparam);
  inline isl::space product(isl::space right) const;
  inline isl::space range() const;
  inline isl::space range_curry() const;
  inline isl::space range_factor_domain() const;
  inline isl::space range_factor_range() const;
  inline boolean range_is_wrapping() const;
  inline isl::space range_map() const;
  inline isl::space range_product(isl::space right) const;
  inline isl::space range_reverse() const;
  inline isl::space reset_tuple_id(isl::dim type) const;
  inline isl::space reset_user() const;
  inline isl::space reverse() const;
  inline isl::space set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::space set_from_params() const;
  inline isl::space set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::space set_tuple_name(isl::dim type, const std::string &s) const;
  inline boolean tuple_is_equal(isl::dim type1, const isl::space &space2, isl::dim type2) const;
  inline isl::space uncurry() const;
  static inline isl::space unit(isl::ctx ctx);
  inline isl::space unwrap() const;
  inline isl::space wrap() const;
  inline isl::space zip() const;
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
  inline term &operator=(term obj);
  inline ~term();
  inline __isl_give isl_term *copy() const &;
  inline __isl_give isl_term *copy() && = delete;
  inline __isl_keep isl_term *get() const;
  inline __isl_give isl_term *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl_size dim(isl::dim type) const;
  inline isl::val get_coefficient_val() const;
  inline isl::aff get_div(unsigned int pos) const;
  inline isl_size get_exp(isl::dim type, unsigned int pos) const;
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
  inline explicit union_access_info(isl::union_map sink);
  inline union_access_info &operator=(union_access_info obj);
  inline ~union_access_info();
  inline __isl_give isl_union_access_info *copy() const &;
  inline __isl_give isl_union_access_info *copy() && = delete;
  inline __isl_keep isl_union_access_info *get() const;
  inline __isl_give isl_union_access_info *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::union_flow compute_flow() const;
  inline isl::union_access_info set_kill(isl::union_map kill) const;
  inline isl::union_access_info set_may_source(isl::union_map may_source) const;
  inline isl::union_access_info set_must_source(isl::union_map must_source) const;
  inline isl::union_access_info set_schedule(isl::schedule schedule) const;
  inline isl::union_access_info set_schedule_map(isl::union_map schedule_map) const;
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
  inline union_flow &operator=(union_flow obj);
  inline ~union_flow();
  inline __isl_give isl_union_flow *copy() const &;
  inline __isl_give isl_union_flow *copy() && = delete;
  inline __isl_keep isl_union_flow *get() const;
  inline __isl_give isl_union_flow *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::union_map get_full_may_dependence() const;
  inline isl::union_map get_full_must_dependence() const;
  inline isl::union_map get_may_dependence() const;
  inline isl::union_map get_may_no_source() const;
  inline isl::union_map get_must_dependence() const;
  inline isl::union_map get_must_no_source() const;
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
  inline /* implicit */ union_map(isl::basic_map bmap);
  inline /* implicit */ union_map(isl::map map);
  inline explicit union_map(isl::union_pw_multi_aff upma);
  inline explicit union_map(isl::ctx ctx, const std::string &str);
  inline union_map &operator=(union_map obj);
  inline ~union_map();
  inline __isl_give isl_union_map *copy() const &;
  inline __isl_give isl_union_map *copy() && = delete;
  inline __isl_keep isl_union_map *get() const;
  inline __isl_give isl_union_map *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::union_map affine_hull() const;
  inline isl::union_map align_params(isl::space model) const;
  inline isl::union_map apply_domain(isl::union_map umap2) const;
  inline isl::union_map apply_range(isl::union_map umap2) const;
  inline isl::union_set bind_range(isl::multi_id tuple) const;
  inline isl::union_map coalesce() const;
  inline boolean contains(const isl::space &space) const;
  inline isl::union_map curry() const;
  inline isl::union_set deltas() const;
  inline isl::union_map deltas_map() const;
  inline isl::union_map detect_equalities() const;
  inline isl_size dim(isl::dim type) const;
  inline isl::union_set domain() const;
  inline isl::union_map domain_factor_domain() const;
  inline isl::union_map domain_factor_range() const;
  inline isl::union_map domain_map() const;
  inline isl::union_pw_multi_aff domain_map_union_pw_multi_aff() const;
  inline isl::union_map domain_product(isl::union_map umap2) const;
  static inline isl::union_map empty(isl::ctx ctx);
  inline isl::union_map eq_at(isl::multi_union_pw_aff mupa) const;
  inline isl::map extract_map(isl::space space) const;
  inline isl::union_map factor_domain() const;
  inline isl::union_map factor_range() const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::union_map fixed_power(isl::val exp) const;
  inline isl::union_map flat_domain_product(isl::union_map umap2) const;
  inline isl::union_map flat_range_product(isl::union_map umap2) const;
  inline stat foreach_map(const std::function<stat(map)> &fn) const;
  static inline isl::union_map from(isl::multi_union_pw_aff mupa);
  static inline isl::union_map from_domain(isl::union_set uset);
  static inline isl::union_map from_domain_and_range(isl::union_set domain, isl::union_set range);
  static inline isl::union_map from_range(isl::union_set uset);
  static inline isl::union_map from_union_pw_aff(isl::union_pw_aff upa);
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline uint32_t get_hash() const;
  inline isl::map_list get_map_list() const;
  inline isl::space get_space() const;
  inline isl::union_map gist(isl::union_map context) const;
  inline isl::union_map gist_domain(isl::union_set uset) const;
  inline isl::union_map gist_params(isl::set set) const;
  inline isl::union_map gist_range(isl::union_set uset) const;
  inline isl::union_map intersect(isl::union_map umap2) const;
  inline isl::union_map intersect_domain(isl::space space) const;
  inline isl::union_map intersect_domain(isl::union_set uset) const;
  inline isl::union_map intersect_domain_factor_domain(isl::union_map factor) const;
  inline isl::union_map intersect_domain_factor_range(isl::union_map factor) const;
  inline isl::union_map intersect_params(isl::set set) const;
  inline isl::union_map intersect_range(isl::space space) const;
  inline isl::union_map intersect_range(isl::union_set uset) const;
  inline isl::union_map intersect_range_factor_domain(isl::union_map factor) const;
  inline isl::union_map intersect_range_factor_range(isl::union_map factor) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean is_bijective() const;
  inline boolean is_disjoint(const isl::union_map &umap2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const isl::union_map &umap2) const;
  inline boolean is_identity() const;
  inline boolean is_injective() const;
  inline boolean is_single_valued() const;
  inline boolean is_strict_subset(const isl::union_map &umap2) const;
  inline boolean is_subset(const isl::union_map &umap2) const;
  inline boolean isa_map() const;
  inline isl::union_map lex_ge_at_multi_union_pw_aff(isl::multi_union_pw_aff mupa) const;
  inline isl::union_map lex_ge_union_map(isl::union_map umap2) const;
  inline isl::union_map lex_gt_at_multi_union_pw_aff(isl::multi_union_pw_aff mupa) const;
  inline isl::union_map lex_gt_union_map(isl::union_map umap2) const;
  inline isl::union_map lex_le_at_multi_union_pw_aff(isl::multi_union_pw_aff mupa) const;
  inline isl::union_map lex_le_union_map(isl::union_map umap2) const;
  inline isl::union_map lex_lt_at_multi_union_pw_aff(isl::multi_union_pw_aff mupa) const;
  inline isl::union_map lex_lt_union_map(isl::union_map umap2) const;
  inline isl::union_map lexmax() const;
  inline isl::union_map lexmin() const;
  inline isl_size n_map() const;
  inline isl::set params() const;
  inline boolean plain_is_empty() const;
  inline boolean plain_is_injective() const;
  inline isl::union_map polyhedral_hull() const;
  inline isl::union_map preimage_domain(isl::multi_aff ma) const;
  inline isl::union_map preimage_domain(isl::multi_pw_aff mpa) const;
  inline isl::union_map preimage_domain(isl::pw_multi_aff pma) const;
  inline isl::union_map preimage_domain(isl::union_pw_multi_aff upma) const;
  inline isl::union_map preimage_range(isl::multi_aff ma) const;
  inline isl::union_map preimage_range(isl::pw_multi_aff pma) const;
  inline isl::union_map preimage_range(isl::union_pw_multi_aff upma) const;
  inline isl::union_map product(isl::union_map umap2) const;
  inline isl::union_map project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::union_map project_out_all_params() const;
  inline isl::union_set range() const;
  inline isl::union_map range_curry() const;
  inline isl::union_map range_factor_domain() const;
  inline isl::union_map range_factor_range() const;
  inline isl::union_map range_map() const;
  inline isl::union_map range_product(isl::union_map umap2) const;
  inline isl::union_map range_reverse() const;
  inline isl::union_map remove_divs() const;
  inline isl::union_map remove_redundancies() const;
  inline isl::union_map reset_user() const;
  inline isl::union_map reverse() const;
  inline isl::basic_map sample() const;
  inline isl::union_map simple_hull() const;
  inline isl::union_map subtract(isl::union_map umap2) const;
  inline isl::union_map subtract_domain(isl::union_set dom) const;
  inline isl::union_map subtract_range(isl::union_set dom) const;
  inline isl::union_map uncurry() const;
  inline isl::union_map unite(isl::union_map umap2) const;
  inline isl::union_map universe() const;
  inline isl::union_set wrap() const;
  inline isl::union_map zip() const;
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
  inline union_map_list &operator=(union_map_list obj);
  inline ~union_map_list();
  inline __isl_give isl_union_map_list *copy() const &;
  inline __isl_give isl_union_map_list *copy() && = delete;
  inline __isl_keep isl_union_map_list *get() const;
  inline __isl_give isl_union_map_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::union_map_list add(isl::union_map el) const;
  static inline isl::union_map_list alloc(isl::ctx ctx, int n);
  inline isl::union_map_list clear() const;
  inline isl::union_map_list concat(isl::union_map_list list2) const;
  inline isl::union_map_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(union_map)> &fn) const;
  static inline isl::union_map_list from_union_map(isl::union_map el);
  inline isl::union_map get_at(int index) const;
  inline isl::union_map get_union_map(int index) const;
  inline isl::union_map_list insert(unsigned int pos, isl::union_map el) const;
  inline isl_size n_union_map() const;
  inline isl::union_map_list reverse() const;
  inline isl::union_map_list set_union_map(int index, isl::union_map el) const;
  inline isl_size size() const;
  inline isl::union_map_list swap(unsigned int pos1, unsigned int pos2) const;
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
  inline /* implicit */ union_pw_aff(isl::aff aff);
  inline /* implicit */ union_pw_aff(isl::pw_aff pa);
  inline explicit union_pw_aff(isl::ctx ctx, const std::string &str);
  inline explicit union_pw_aff(isl::union_set domain, isl::val v);
  inline union_pw_aff &operator=(union_pw_aff obj);
  inline ~union_pw_aff();
  inline __isl_give isl_union_pw_aff *copy() const &;
  inline __isl_give isl_union_pw_aff *copy() && = delete;
  inline __isl_keep isl_union_pw_aff *get() const;
  inline __isl_give isl_union_pw_aff *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::union_pw_aff add(isl::union_pw_aff upa2) const;
  inline isl::union_pw_aff add_pw_aff(isl::pw_aff pa) const;
  static inline isl::union_pw_aff aff_on_domain(isl::union_set domain, isl::aff aff);
  inline isl::union_pw_aff align_params(isl::space model) const;
  inline isl::union_set bind(isl::id id) const;
  inline isl::union_pw_aff coalesce() const;
  inline isl_size dim(isl::dim type) const;
  inline isl::union_set domain() const;
  inline isl::union_pw_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  static inline isl::union_pw_aff empty(isl::space space);
  static inline isl::union_pw_aff empty_ctx(isl::ctx ctx);
  static inline isl::union_pw_aff empty_space(isl::space space);
  inline isl::pw_aff extract_pw_aff(isl::space space) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::union_pw_aff floor() const;
  inline stat foreach_pw_aff(const std::function<stat(pw_aff)> &fn) const;
  inline isl::pw_aff_list get_pw_aff_list() const;
  inline isl::space get_space() const;
  inline isl::union_pw_aff gist(isl::union_set context) const;
  inline isl::union_pw_aff gist_params(isl::set context) const;
  inline isl::union_pw_aff intersect_domain(isl::space space) const;
  inline isl::union_pw_aff intersect_domain(isl::union_set uset) const;
  inline isl::union_pw_aff intersect_domain_wrapped_domain(isl::union_set uset) const;
  inline isl::union_pw_aff intersect_domain_wrapped_range(isl::union_set uset) const;
  inline isl::union_pw_aff intersect_params(isl::set set) const;
  inline boolean involves_nan() const;
  inline isl::val max_val() const;
  inline isl::val min_val() const;
  inline isl::union_pw_aff mod_val(isl::val f) const;
  inline isl_size n_pw_aff() const;
  inline isl::union_pw_aff neg() const;
  static inline isl::union_pw_aff param_on_domain_id(isl::union_set domain, isl::id id);
  inline boolean plain_is_equal(const isl::union_pw_aff &upa2) const;
  inline isl::union_pw_aff pullback(isl::union_pw_multi_aff upma) const;
  static inline isl::union_pw_aff pw_aff_on_domain(isl::union_set domain, isl::pw_aff pa);
  inline isl::union_pw_aff reset_user() const;
  inline isl::union_pw_aff scale_down_val(isl::val v) const;
  inline isl::union_pw_aff scale_val(isl::val v) const;
  inline isl::union_pw_aff sub(isl::union_pw_aff upa2) const;
  inline isl::union_pw_aff subtract_domain(isl::space space) const;
  inline isl::union_pw_aff subtract_domain(isl::union_set uset) const;
  inline isl::union_pw_aff union_add(isl::union_pw_aff upa2) const;
  inline isl::union_set zero_union_set() const;
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
  inline union_pw_aff_list &operator=(union_pw_aff_list obj);
  inline ~union_pw_aff_list();
  inline __isl_give isl_union_pw_aff_list *copy() const &;
  inline __isl_give isl_union_pw_aff_list *copy() && = delete;
  inline __isl_keep isl_union_pw_aff_list *get() const;
  inline __isl_give isl_union_pw_aff_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::union_pw_aff_list add(isl::union_pw_aff el) const;
  static inline isl::union_pw_aff_list alloc(isl::ctx ctx, int n);
  inline isl::union_pw_aff_list clear() const;
  inline isl::union_pw_aff_list concat(isl::union_pw_aff_list list2) const;
  inline isl::union_pw_aff_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(union_pw_aff)> &fn) const;
  static inline isl::union_pw_aff_list from_union_pw_aff(isl::union_pw_aff el);
  inline isl::union_pw_aff get_at(int index) const;
  inline isl::union_pw_aff get_union_pw_aff(int index) const;
  inline isl::union_pw_aff_list insert(unsigned int pos, isl::union_pw_aff el) const;
  inline isl_size n_union_pw_aff() const;
  inline isl::union_pw_aff_list reverse() const;
  inline isl::union_pw_aff_list set_union_pw_aff(int index, isl::union_pw_aff el) const;
  inline isl_size size() const;
  inline isl::union_pw_aff_list swap(unsigned int pos1, unsigned int pos2) const;
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
  inline /* implicit */ union_pw_multi_aff(isl::aff aff);
  inline explicit union_pw_multi_aff(isl::union_set uset);
  inline /* implicit */ union_pw_multi_aff(isl::multi_aff ma);
  inline explicit union_pw_multi_aff(isl::multi_union_pw_aff mupa);
  inline /* implicit */ union_pw_multi_aff(isl::pw_multi_aff pma);
  inline explicit union_pw_multi_aff(isl::union_map umap);
  inline /* implicit */ union_pw_multi_aff(isl::union_pw_aff upa);
  inline explicit union_pw_multi_aff(isl::ctx ctx, const std::string &str);
  inline union_pw_multi_aff &operator=(union_pw_multi_aff obj);
  inline ~union_pw_multi_aff();
  inline __isl_give isl_union_pw_multi_aff *copy() const &;
  inline __isl_give isl_union_pw_multi_aff *copy() && = delete;
  inline __isl_keep isl_union_pw_multi_aff *get() const;
  inline __isl_give isl_union_pw_multi_aff *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::union_pw_multi_aff add(isl::union_pw_multi_aff upma2) const;
  inline isl::union_pw_multi_aff add_pw_multi_aff(isl::pw_multi_aff pma) const;
  inline isl::union_pw_multi_aff align_params(isl::space model) const;
  inline isl::union_pw_multi_aff apply(isl::union_pw_multi_aff upma2) const;
  inline isl::pw_multi_aff as_pw_multi_aff() const;
  inline isl::union_pw_multi_aff coalesce() const;
  inline isl_size dim(isl::dim type) const;
  inline isl::union_set domain() const;
  inline isl::union_pw_multi_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  static inline isl::union_pw_multi_aff empty(isl::space space);
  static inline isl::union_pw_multi_aff empty(isl::ctx ctx);
  static inline isl::union_pw_multi_aff empty_space(isl::space space);
  inline isl::pw_multi_aff extract_pw_multi_aff(isl::space space) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::union_pw_multi_aff flat_range_product(isl::union_pw_multi_aff upma2) const;
  inline stat foreach_pw_multi_aff(const std::function<stat(pw_multi_aff)> &fn) const;
  static inline isl::union_pw_multi_aff from_union_set(isl::union_set uset);
  inline isl::pw_multi_aff_list get_pw_multi_aff_list() const;
  inline isl::space get_space() const;
  inline isl::union_pw_aff get_union_pw_aff(int pos) const;
  inline isl::union_pw_multi_aff gist(isl::union_set context) const;
  inline isl::union_pw_multi_aff gist_params(isl::set context) const;
  inline isl::union_pw_multi_aff intersect_domain(isl::space space) const;
  inline isl::union_pw_multi_aff intersect_domain(isl::union_set uset) const;
  inline isl::union_pw_multi_aff intersect_domain_wrapped_domain(isl::union_set uset) const;
  inline isl::union_pw_multi_aff intersect_domain_wrapped_range(isl::union_set uset) const;
  inline isl::union_pw_multi_aff intersect_params(isl::set set) const;
  inline boolean involves_locals() const;
  inline boolean involves_nan() const;
  inline boolean isa_pw_multi_aff() const;
  static inline isl::union_pw_multi_aff multi_val_on_domain(isl::union_set domain, isl::multi_val mv);
  inline isl_size n_pw_multi_aff() const;
  inline isl::union_pw_multi_aff neg() const;
  inline boolean plain_is_empty() const;
  inline boolean plain_is_equal(const isl::union_pw_multi_aff &upma2) const;
  inline isl::union_pw_multi_aff preimage_domain_wrapped_domain(isl::union_pw_multi_aff upma2) const;
  inline isl::union_pw_multi_aff pullback(isl::union_pw_multi_aff upma2) const;
  inline isl::union_pw_multi_aff range_factor_domain() const;
  inline isl::union_pw_multi_aff range_factor_range() const;
  inline isl::union_pw_multi_aff range_product(isl::union_pw_multi_aff upma2) const;
  inline isl::union_pw_multi_aff reset_user() const;
  inline isl::union_pw_multi_aff scale_down_val(isl::val val) const;
  inline isl::union_pw_multi_aff scale_multi_val(isl::multi_val mv) const;
  inline isl::union_pw_multi_aff scale_val(isl::val val) const;
  inline isl::union_pw_multi_aff sub(isl::union_pw_multi_aff upma2) const;
  inline isl::union_pw_multi_aff subtract_domain(isl::space space) const;
  inline isl::union_pw_multi_aff subtract_domain(isl::union_set uset) const;
  inline isl::union_pw_multi_aff union_add(isl::union_pw_multi_aff upma2) const;
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
  inline union_pw_multi_aff_list &operator=(union_pw_multi_aff_list obj);
  inline ~union_pw_multi_aff_list();
  inline __isl_give isl_union_pw_multi_aff_list *copy() const &;
  inline __isl_give isl_union_pw_multi_aff_list *copy() && = delete;
  inline __isl_keep isl_union_pw_multi_aff_list *get() const;
  inline __isl_give isl_union_pw_multi_aff_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::union_pw_multi_aff_list add(isl::union_pw_multi_aff el) const;
  static inline isl::union_pw_multi_aff_list alloc(isl::ctx ctx, int n);
  inline isl::union_pw_multi_aff_list clear() const;
  inline isl::union_pw_multi_aff_list concat(isl::union_pw_multi_aff_list list2) const;
  inline isl::union_pw_multi_aff_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(union_pw_multi_aff)> &fn) const;
  static inline isl::union_pw_multi_aff_list from_union_pw_multi_aff(isl::union_pw_multi_aff el);
  inline isl::union_pw_multi_aff get_at(int index) const;
  inline isl::union_pw_multi_aff get_union_pw_multi_aff(int index) const;
  inline isl::union_pw_multi_aff_list insert(unsigned int pos, isl::union_pw_multi_aff el) const;
  inline isl_size n_union_pw_multi_aff() const;
  inline isl::union_pw_multi_aff_list reverse() const;
  inline isl::union_pw_multi_aff_list set_union_pw_multi_aff(int index, isl::union_pw_multi_aff el) const;
  inline isl_size size() const;
  inline isl::union_pw_multi_aff_list swap(unsigned int pos1, unsigned int pos2) const;
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
  inline explicit union_pw_qpolynomial(isl::ctx ctx, const std::string &str);
  inline union_pw_qpolynomial &operator=(union_pw_qpolynomial obj);
  inline ~union_pw_qpolynomial();
  inline __isl_give isl_union_pw_qpolynomial *copy() const &;
  inline __isl_give isl_union_pw_qpolynomial *copy() && = delete;
  inline __isl_keep isl_union_pw_qpolynomial *get() const;
  inline __isl_give isl_union_pw_qpolynomial *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::union_pw_qpolynomial add(isl::union_pw_qpolynomial upwqp2) const;
  inline isl::union_pw_qpolynomial add_pw_qpolynomial(isl::pw_qpolynomial pwqp) const;
  inline isl::union_pw_qpolynomial align_params(isl::space model) const;
  inline isl::union_pw_qpolynomial coalesce() const;
  inline isl_size dim(isl::dim type) const;
  inline isl::union_set domain() const;
  inline isl::union_pw_qpolynomial drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::val eval(isl::point pnt) const;
  inline isl::pw_qpolynomial extract_pw_qpolynomial(isl::space space) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline stat foreach_pw_qpolynomial(const std::function<stat(pw_qpolynomial)> &fn) const;
  static inline isl::union_pw_qpolynomial from_pw_qpolynomial(isl::pw_qpolynomial pwqp);
  inline isl::pw_qpolynomial_list get_pw_qpolynomial_list() const;
  inline isl::space get_space() const;
  inline isl::union_pw_qpolynomial gist(isl::union_set context) const;
  inline isl::union_pw_qpolynomial gist_params(isl::set context) const;
  inline isl::union_pw_qpolynomial intersect_domain(isl::union_set uset) const;
  inline isl::union_pw_qpolynomial intersect_domain_space(isl::space space) const;
  inline isl::union_pw_qpolynomial intersect_domain_union_set(isl::union_set uset) const;
  inline isl::union_pw_qpolynomial intersect_domain_wrapped_domain(isl::union_set uset) const;
  inline isl::union_pw_qpolynomial intersect_domain_wrapped_range(isl::union_set uset) const;
  inline isl::union_pw_qpolynomial intersect_params(isl::set set) const;
  inline boolean involves_nan() const;
  inline isl::union_pw_qpolynomial mul(isl::union_pw_qpolynomial upwqp2) const;
  inline isl_size n_pw_qpolynomial() const;
  inline isl::union_pw_qpolynomial neg() const;
  inline boolean plain_is_equal(const isl::union_pw_qpolynomial &upwqp2) const;
  inline isl::union_pw_qpolynomial reset_user() const;
  inline isl::union_pw_qpolynomial scale_down_val(isl::val v) const;
  inline isl::union_pw_qpolynomial scale_val(isl::val v) const;
  inline isl::union_pw_qpolynomial sub(isl::union_pw_qpolynomial upwqp2) const;
  inline isl::union_pw_qpolynomial subtract_domain(isl::union_set uset) const;
  inline isl::union_pw_qpolynomial subtract_domain_space(isl::space space) const;
  inline isl::union_pw_qpolynomial subtract_domain_union_set(isl::union_set uset) const;
  inline isl::union_pw_qpolynomial to_polynomial(int sign) const;
  static inline isl::union_pw_qpolynomial zero(isl::space space);
  static inline isl::union_pw_qpolynomial zero_ctx(isl::ctx ctx);
  static inline isl::union_pw_qpolynomial zero_space(isl::space space);
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
  inline /* implicit */ union_set(isl::basic_set bset);
  inline /* implicit */ union_set(isl::point pnt);
  inline /* implicit */ union_set(isl::set set);
  inline explicit union_set(isl::ctx ctx, const std::string &str);
  inline union_set &operator=(union_set obj);
  inline ~union_set();
  inline __isl_give isl_union_set *copy() const &;
  inline __isl_give isl_union_set *copy() && = delete;
  inline __isl_keep isl_union_set *get() const;
  inline __isl_give isl_union_set *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::union_set affine_hull() const;
  inline isl::union_set align_params(isl::space model) const;
  inline isl::union_set apply(isl::union_map umap) const;
  inline isl::union_set coalesce() const;
  inline isl::union_set coefficients() const;
  inline isl::schedule compute_schedule(isl::union_map validity, isl::union_map proximity) const;
  inline boolean contains(const isl::space &space) const;
  inline isl::union_set detect_equalities() const;
  inline isl_size dim(isl::dim type) const;
  static inline isl::union_set empty(isl::ctx ctx);
  inline isl::set extract_set(isl::space space) const;
  inline stat foreach_point(const std::function<stat(point)> &fn) const;
  inline stat foreach_set(const std::function<stat(set)> &fn) const;
  inline isl::basic_set_list get_basic_set_list() const;
  inline uint32_t get_hash() const;
  inline isl::set_list get_set_list() const;
  inline isl::space get_space() const;
  inline isl::union_set gist(isl::union_set context) const;
  inline isl::union_set gist_params(isl::set set) const;
  inline isl::union_map identity() const;
  inline isl::union_pw_multi_aff identity_union_pw_multi_aff() const;
  inline isl::union_set intersect(isl::union_set uset2) const;
  inline isl::union_set intersect_params(isl::set set) const;
  inline boolean is_disjoint(const isl::union_set &uset2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const isl::union_set &uset2) const;
  inline boolean is_params() const;
  inline boolean is_strict_subset(const isl::union_set &uset2) const;
  inline boolean is_subset(const isl::union_set &uset2) const;
  inline boolean isa_set() const;
  inline isl::union_map lex_ge_union_set(isl::union_set uset2) const;
  inline isl::union_map lex_gt_union_set(isl::union_set uset2) const;
  inline isl::union_map lex_le_union_set(isl::union_set uset2) const;
  inline isl::union_map lex_lt_union_set(isl::union_set uset2) const;
  inline isl::union_set lexmax() const;
  inline isl::union_set lexmin() const;
  inline isl::multi_val min_multi_union_pw_aff(const isl::multi_union_pw_aff &obj) const;
  inline isl_size n_set() const;
  inline isl::set params() const;
  inline isl::union_set polyhedral_hull() const;
  inline isl::union_set preimage(isl::multi_aff ma) const;
  inline isl::union_set preimage(isl::pw_multi_aff pma) const;
  inline isl::union_set preimage(isl::union_pw_multi_aff upma) const;
  inline isl::union_set product(isl::union_set uset2) const;
  inline isl::union_set project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::union_set project_out_all_params() const;
  inline isl::union_set remove_divs() const;
  inline isl::union_set remove_redundancies() const;
  inline isl::union_set reset_user() const;
  inline isl::basic_set sample() const;
  inline isl::point sample_point() const;
  inline isl::union_set simple_hull() const;
  inline isl::union_set solutions() const;
  inline isl::union_set subtract(isl::union_set uset2) const;
  inline isl::union_set unite(isl::union_set uset2) const;
  inline isl::union_set universe() const;
  inline isl::union_map unwrap() const;
  inline isl::union_map wrapped_domain_map() const;
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
  inline union_set_list &operator=(union_set_list obj);
  inline ~union_set_list();
  inline __isl_give isl_union_set_list *copy() const &;
  inline __isl_give isl_union_set_list *copy() && = delete;
  inline __isl_keep isl_union_set_list *get() const;
  inline __isl_give isl_union_set_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::union_set_list add(isl::union_set el) const;
  static inline isl::union_set_list alloc(isl::ctx ctx, int n);
  inline isl::union_set_list clear() const;
  inline isl::union_set_list concat(isl::union_set_list list2) const;
  inline isl::union_set_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(union_set)> &fn) const;
  static inline isl::union_set_list from_union_set(isl::union_set el);
  inline isl::union_set get_at(int index) const;
  inline isl::union_set get_union_set(int index) const;
  inline isl::union_set_list insert(unsigned int pos, isl::union_set el) const;
  inline isl_size n_union_set() const;
  inline isl::union_set_list reverse() const;
  inline isl::union_set_list set_union_set(int index, isl::union_set el) const;
  inline isl_size size() const;
  inline isl::union_set_list swap(unsigned int pos1, unsigned int pos2) const;
  inline isl::union_set unite() const;
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
  inline explicit val(isl::ctx ctx, long i);
  inline explicit val(isl::ctx ctx, const std::string &str);
  inline val &operator=(val obj);
  inline ~val();
  inline __isl_give isl_val *copy() const &;
  inline __isl_give isl_val *copy() && = delete;
  inline __isl_keep isl_val *get() const;
  inline __isl_give isl_val *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::val abs() const;
  inline boolean abs_eq(const isl::val &v2) const;
  inline isl::val add(isl::val v2) const;
  inline isl::val add_ui(unsigned long v2) const;
  inline isl::val ceil() const;
  inline int cmp_si(long i) const;
  inline isl::val div(isl::val v2) const;
  inline isl::val div_ui(unsigned long v2) const;
  inline boolean eq(const isl::val &v2) const;
  inline boolean eq_si(long i) const;
  inline isl::val floor() const;
  inline isl::val gcd(isl::val v2) const;
  inline boolean ge(const isl::val &v2) const;
  inline uint32_t get_hash() const;
  inline long get_num_si() const;
  inline boolean gt(const isl::val &v2) const;
  inline boolean gt_si(long i) const;
  static inline isl::val infty(isl::ctx ctx);
  static inline isl::val int_from_ui(isl::ctx ctx, unsigned long u);
  inline isl::val inv() const;
  inline boolean is_divisible_by(const isl::val &v2) const;
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
  inline boolean le(const isl::val &v2) const;
  inline boolean lt(const isl::val &v2) const;
  inline isl::val max(isl::val v2) const;
  inline isl::val min(isl::val v2) const;
  inline isl::val mod(isl::val v2) const;
  inline isl::val mul(isl::val v2) const;
  inline isl::val mul_ui(unsigned long v2) const;
  inline isl_size n_abs_num_chunks(size_t size) const;
  static inline isl::val nan(isl::ctx ctx);
  inline boolean ne(const isl::val &v2) const;
  inline isl::val neg() const;
  static inline isl::val neginfty(isl::ctx ctx);
  static inline isl::val negone(isl::ctx ctx);
  static inline isl::val one(isl::ctx ctx);
  inline isl::val pow2() const;
  inline isl::val set_si(long i) const;
  inline int sgn() const;
  inline isl::val sub(isl::val v2) const;
  inline isl::val sub_ui(unsigned long v2) const;
  inline isl::val trunc() const;
  static inline isl::val zero(isl::ctx ctx);
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
  inline val_list &operator=(val_list obj);
  inline ~val_list();
  inline __isl_give isl_val_list *copy() const &;
  inline __isl_give isl_val_list *copy() && = delete;
  inline __isl_keep isl_val_list *get() const;
  inline __isl_give isl_val_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::val_list add(isl::val el) const;
  static inline isl::val_list alloc(isl::ctx ctx, int n);
  inline isl::val_list clear() const;
  inline isl::val_list concat(isl::val_list list2) const;
  inline isl::val_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(val)> &fn) const;
  static inline isl::val_list from_val(isl::val el);
  inline isl::val get_at(int index) const;
  inline isl::val get_val(int index) const;
  inline isl::val_list insert(unsigned int pos, isl::val el) const;
  inline isl_size n_val() const;
  inline isl::val_list reverse() const;
  inline isl::val_list set_val(int index, isl::val el) const;
  inline isl_size size() const;
  inline isl::val_list swap(unsigned int pos1, unsigned int pos2) const;
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
  inline vec &operator=(vec obj);
  inline ~vec();
  inline __isl_give isl_vec *copy() const &;
  inline __isl_give isl_vec *copy() && = delete;
  inline __isl_keep isl_vec *get() const;
  inline __isl_give isl_vec *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;
  inline void dump() const;

  inline isl::vec add(isl::vec vec2) const;
  inline isl::vec add_els(unsigned int n) const;
  static inline isl::vec alloc(isl::ctx ctx, unsigned int size);
  inline isl::vec ceil() const;
  inline isl::vec clr() const;
  inline int cmp_element(const isl::vec &vec2, int pos) const;
  inline isl::vec concat(isl::vec vec2) const;
  inline isl::vec drop_els(unsigned int pos, unsigned int n) const;
  inline isl::vec extend(unsigned int size) const;
  inline isl::val get_element_val(int pos) const;
  inline isl::vec insert_els(unsigned int pos, unsigned int n) const;
  inline isl::vec insert_zero_els(unsigned int pos, unsigned int n) const;
  inline boolean is_equal(const isl::vec &vec2) const;
  inline isl::vec mat_product(isl::mat mat) const;
  inline isl::vec move_els(unsigned int dst_col, unsigned int src_col, unsigned int n) const;
  inline isl::vec neg() const;
  inline isl::vec set_element_si(int pos, int v) const;
  inline isl::vec set_element_val(int pos, isl::val v) const;
  inline isl::vec set_si(int v) const;
  inline isl::vec set_val(isl::val v) const;
  inline isl_size size() const;
  inline isl::vec sort() const;
  static inline isl::vec zero(isl::ctx ctx, unsigned int size);
  inline isl::vec zero_extend(unsigned int size) const;
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


aff::aff(__isl_take isl_aff *ptr)
    : ptr(ptr) {}

aff::aff(isl::ctx ctx, const std::string &str)
{
  auto res = isl_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}
aff::aff(isl::local_space ls, isl::val val)
{
  auto res = isl_aff_val_on_domain(ls.release(), val.release());
  ptr = res;
}
aff::aff(isl::local_space ls)
{
  auto res = isl_aff_zero_on_domain(ls.release());
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


isl::ctx aff::ctx() const {
  return isl::ctx(isl_aff_get_ctx(ptr));
}

void aff::dump() const {
  isl_aff_dump(get());
}


isl::aff aff::add(isl::aff aff2) const
{
  auto res = isl_aff_add(copy(), aff2.release());
  return manage(res);
}

isl::aff aff::add_coefficient_si(isl::dim type, int pos, int v) const
{
  auto res = isl_aff_add_coefficient_si(copy(), static_cast<enum isl_dim_type>(type), pos, v);
  return manage(res);
}

isl::aff aff::add_coefficient_val(isl::dim type, int pos, isl::val v) const
{
  auto res = isl_aff_add_coefficient_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

isl::aff aff::add_constant(isl::val v) const
{
  auto res = isl_aff_add_constant_val(copy(), v.release());
  return manage(res);
}

isl::aff aff::add_constant_num_si(int v) const
{
  auto res = isl_aff_add_constant_num_si(copy(), v);
  return manage(res);
}

isl::aff aff::add_constant_si(int v) const
{
  auto res = isl_aff_add_constant_si(copy(), v);
  return manage(res);
}

isl::aff aff::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_aff_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::aff aff::align_params(isl::space model) const
{
  auto res = isl_aff_align_params(copy(), model.release());
  return manage(res);
}

isl::basic_set aff::bind(isl::id id) const
{
  auto res = isl_aff_bind_id(copy(), id.release());
  return manage(res);
}

isl::aff aff::ceil() const
{
  auto res = isl_aff_ceil(copy());
  return manage(res);
}

int aff::coefficient_sgn(isl::dim type, int pos) const
{
  auto res = isl_aff_coefficient_sgn(get(), static_cast<enum isl_dim_type>(type), pos);
  return res;
}

isl_size aff::dim(isl::dim type) const
{
  auto res = isl_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::aff aff::div(isl::aff aff2) const
{
  auto res = isl_aff_div(copy(), aff2.release());
  return manage(res);
}

isl::aff aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_set aff::eq_basic_set(isl::aff aff2) const
{
  auto res = isl_aff_eq_basic_set(copy(), aff2.release());
  return manage(res);
}

isl::set aff::eq_set(isl::aff aff2) const
{
  auto res = isl_aff_eq_set(copy(), aff2.release());
  return manage(res);
}

isl::val aff::eval(isl::point pnt) const
{
  auto res = isl_aff_eval(copy(), pnt.release());
  return manage(res);
}

int aff::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::aff aff::floor() const
{
  auto res = isl_aff_floor(copy());
  return manage(res);
}

isl::aff aff::from_range() const
{
  auto res = isl_aff_from_range(copy());
  return manage(res);
}

isl::basic_set aff::ge_basic_set(isl::aff aff2) const
{
  auto res = isl_aff_ge_basic_set(copy(), aff2.release());
  return manage(res);
}

isl::set aff::ge_set(isl::aff aff2) const
{
  auto res = isl_aff_ge_set(copy(), aff2.release());
  return manage(res);
}

isl::val aff::get_coefficient_val(isl::dim type, int pos) const
{
  auto res = isl_aff_get_coefficient_val(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::val aff::get_constant_val() const
{
  auto res = isl_aff_get_constant_val(get());
  return manage(res);
}

isl::val aff::get_denominator_val() const
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

isl::aff aff::get_div(int pos) const
{
  auto res = isl_aff_get_div(get(), pos);
  return manage(res);
}

isl::local_space aff::get_domain_local_space() const
{
  auto res = isl_aff_get_domain_local_space(get());
  return manage(res);
}

isl::space aff::get_domain_space() const
{
  auto res = isl_aff_get_domain_space(get());
  return manage(res);
}

uint32_t aff::get_hash() const
{
  auto res = isl_aff_get_hash(get());
  return res;
}

isl::local_space aff::get_local_space() const
{
  auto res = isl_aff_get_local_space(get());
  return manage(res);
}

isl::space aff::get_space() const
{
  auto res = isl_aff_get_space(get());
  return manage(res);
}

isl::aff aff::gist(isl::set context) const
{
  auto res = isl_aff_gist(copy(), context.release());
  return manage(res);
}

isl::aff aff::gist_params(isl::set context) const
{
  auto res = isl_aff_gist_params(copy(), context.release());
  return manage(res);
}

isl::basic_set aff::gt_basic_set(isl::aff aff2) const
{
  auto res = isl_aff_gt_basic_set(copy(), aff2.release());
  return manage(res);
}

isl::set aff::gt_set(isl::aff aff2) const
{
  auto res = isl_aff_gt_set(copy(), aff2.release());
  return manage(res);
}

isl::aff aff::insert_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_aff_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean aff::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_aff_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean aff::involves_locals() const
{
  auto res = isl_aff_involves_locals(get());
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

isl::basic_set aff::le_basic_set(isl::aff aff2) const
{
  auto res = isl_aff_le_basic_set(copy(), aff2.release());
  return manage(res);
}

isl::set aff::le_set(isl::aff aff2) const
{
  auto res = isl_aff_le_set(copy(), aff2.release());
  return manage(res);
}

isl::basic_set aff::lt_basic_set(isl::aff aff2) const
{
  auto res = isl_aff_lt_basic_set(copy(), aff2.release());
  return manage(res);
}

isl::set aff::lt_set(isl::aff aff2) const
{
  auto res = isl_aff_lt_set(copy(), aff2.release());
  return manage(res);
}

isl::aff aff::mod(isl::val mod) const
{
  auto res = isl_aff_mod_val(copy(), mod.release());
  return manage(res);
}

isl::aff aff::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_aff_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl::aff aff::mul(isl::aff aff2) const
{
  auto res = isl_aff_mul(copy(), aff2.release());
  return manage(res);
}

isl::aff aff::nan_on_domain(isl::local_space ls)
{
  auto res = isl_aff_nan_on_domain(ls.release());
  return manage(res);
}

isl::aff aff::nan_on_domain_space(isl::space space)
{
  auto res = isl_aff_nan_on_domain_space(space.release());
  return manage(res);
}

isl::set aff::ne_set(isl::aff aff2) const
{
  auto res = isl_aff_ne_set(copy(), aff2.release());
  return manage(res);
}

isl::aff aff::neg() const
{
  auto res = isl_aff_neg(copy());
  return manage(res);
}

isl::basic_set aff::neg_basic_set() const
{
  auto res = isl_aff_neg_basic_set(copy());
  return manage(res);
}

isl::aff aff::param_on_domain_space_id(isl::space space, isl::id id)
{
  auto res = isl_aff_param_on_domain_space_id(space.release(), id.release());
  return manage(res);
}

boolean aff::plain_is_equal(const isl::aff &aff2) const
{
  auto res = isl_aff_plain_is_equal(get(), aff2.get());
  return manage(res);
}

boolean aff::plain_is_zero() const
{
  auto res = isl_aff_plain_is_zero(get());
  return manage(res);
}

isl::aff aff::project_domain_on_params() const
{
  auto res = isl_aff_project_domain_on_params(copy());
  return manage(res);
}

isl::aff aff::pullback(isl::multi_aff ma) const
{
  auto res = isl_aff_pullback_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::aff aff::pullback_aff(isl::aff aff2) const
{
  auto res = isl_aff_pullback_aff(copy(), aff2.release());
  return manage(res);
}

isl::aff aff::scale(isl::val v) const
{
  auto res = isl_aff_scale_val(copy(), v.release());
  return manage(res);
}

isl::aff aff::scale_down(isl::val v) const
{
  auto res = isl_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::aff aff::scale_down_ui(unsigned int f) const
{
  auto res = isl_aff_scale_down_ui(copy(), f);
  return manage(res);
}

isl::aff aff::set_coefficient_si(isl::dim type, int pos, int v) const
{
  auto res = isl_aff_set_coefficient_si(copy(), static_cast<enum isl_dim_type>(type), pos, v);
  return manage(res);
}

isl::aff aff::set_coefficient_val(isl::dim type, int pos, isl::val v) const
{
  auto res = isl_aff_set_coefficient_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

isl::aff aff::set_constant_si(int v) const
{
  auto res = isl_aff_set_constant_si(copy(), v);
  return manage(res);
}

isl::aff aff::set_constant_val(isl::val v) const
{
  auto res = isl_aff_set_constant_val(copy(), v.release());
  return manage(res);
}

isl::aff aff::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const
{
  auto res = isl_aff_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::aff aff::set_tuple_id(isl::dim type, isl::id id) const
{
  auto res = isl_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::aff aff::sub(isl::aff aff2) const
{
  auto res = isl_aff_sub(copy(), aff2.release());
  return manage(res);
}

isl::aff aff::unbind_params_insert_domain(isl::multi_id domain) const
{
  auto res = isl_aff_unbind_params_insert_domain(copy(), domain.release());
  return manage(res);
}

isl::aff aff::val_on_domain_space(isl::space space, isl::val val)
{
  auto res = isl_aff_val_on_domain_space(space.release(), val.release());
  return manage(res);
}

isl::aff aff::var_on_domain(isl::local_space ls, isl::dim type, unsigned int pos)
{
  auto res = isl_aff_var_on_domain(ls.release(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::basic_set aff::zero_basic_set() const
{
  auto res = isl_aff_zero_basic_set(copy());
  return manage(res);
}

isl::aff aff::zero_on_domain(isl::space space)
{
  auto res = isl_aff_zero_on_domain_space(space.release());
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


isl::ctx aff_list::ctx() const {
  return isl::ctx(isl_aff_list_get_ctx(ptr));
}

void aff_list::dump() const {
  isl_aff_list_dump(get());
}


isl::aff_list aff_list::add(isl::aff el) const
{
  auto res = isl_aff_list_add(copy(), el.release());
  return manage(res);
}

isl::aff_list aff_list::alloc(isl::ctx ctx, int n)
{
  auto res = isl_aff_list_alloc(ctx.release(), n);
  return manage(res);
}

isl::aff_list aff_list::clear() const
{
  auto res = isl_aff_list_clear(copy());
  return manage(res);
}

isl::aff_list aff_list::concat(isl::aff_list list2) const
{
  auto res = isl_aff_list_concat(copy(), list2.release());
  return manage(res);
}

isl::aff_list aff_list::drop(unsigned int first, unsigned int n) const
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

isl::aff_list aff_list::from_aff(isl::aff el)
{
  auto res = isl_aff_list_from_aff(el.release());
  return manage(res);
}

isl::aff aff_list::get_aff(int index) const
{
  auto res = isl_aff_list_get_aff(get(), index);
  return manage(res);
}

isl::aff aff_list::get_at(int index) const
{
  auto res = isl_aff_list_get_at(get(), index);
  return manage(res);
}

isl::aff_list aff_list::insert(unsigned int pos, isl::aff el) const
{
  auto res = isl_aff_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl_size aff_list::n_aff() const
{
  auto res = isl_aff_list_n_aff(get());
  return res;
}

isl::aff_list aff_list::reverse() const
{
  auto res = isl_aff_list_reverse(copy());
  return manage(res);
}

isl::aff_list aff_list::set_aff(int index, isl::aff el) const
{
  auto res = isl_aff_list_set_aff(copy(), index, el.release());
  return manage(res);
}

isl_size aff_list::size() const
{
  auto res = isl_aff_list_size(get());
  return res;
}

isl::aff_list aff_list::swap(unsigned int pos1, unsigned int pos2) const
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


ast_build::ast_build(__isl_take isl_ast_build *ptr)
    : ptr(ptr) {}

ast_build::ast_build(isl::ctx ctx)
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


isl::ctx ast_build::ctx() const {
  return isl::ctx(isl_ast_build_get_ctx(ptr));
}


isl::ast_expr ast_build::access_from(isl::multi_pw_aff mpa) const
{
  auto res = isl_ast_build_access_from_multi_pw_aff(get(), mpa.release());
  return manage(res);
}

isl::ast_expr ast_build::access_from(isl::pw_multi_aff pma) const
{
  auto res = isl_ast_build_access_from_pw_multi_aff(get(), pma.release());
  return manage(res);
}

isl::ast_node ast_build::ast_from_schedule(isl::union_map schedule) const
{
  auto res = isl_ast_build_ast_from_schedule(get(), schedule.release());
  return manage(res);
}

isl::ast_expr ast_build::call_from(isl::multi_pw_aff mpa) const
{
  auto res = isl_ast_build_call_from_multi_pw_aff(get(), mpa.release());
  return manage(res);
}

isl::ast_expr ast_build::call_from(isl::pw_multi_aff pma) const
{
  auto res = isl_ast_build_call_from_pw_multi_aff(get(), pma.release());
  return manage(res);
}

isl::ast_expr ast_build::expr_from(isl::pw_aff pa) const
{
  auto res = isl_ast_build_expr_from_pw_aff(get(), pa.release());
  return manage(res);
}

isl::ast_expr ast_build::expr_from(isl::set set) const
{
  auto res = isl_ast_build_expr_from_set(get(), set.release());
  return manage(res);
}

isl::ast_build ast_build::from_context(isl::set set)
{
  auto res = isl_ast_build_from_context(set.release());
  return manage(res);
}

isl::union_map ast_build::get_schedule() const
{
  auto res = isl_ast_build_get_schedule(get());
  return manage(res);
}

isl::space ast_build::get_schedule_space() const
{
  auto res = isl_ast_build_get_schedule_space(get());
  return manage(res);
}

isl::ast_node ast_build::node_from(isl::schedule schedule) const
{
  auto res = isl_ast_build_node_from_schedule(get(), schedule.release());
  return manage(res);
}

isl::ast_node ast_build::node_from_schedule_map(isl::union_map schedule) const
{
  auto res = isl_ast_build_node_from_schedule_map(get(), schedule.release());
  return manage(res);
}

isl::ast_build ast_build::restrict(isl::set set) const
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


isl::ctx ast_expr::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

void ast_expr::dump() const {
  isl_ast_expr_dump(get());
}


isl::ast_expr ast_expr::access(isl::ast_expr_list indices) const
{
  auto res = isl_ast_expr_access(copy(), indices.release());
  return manage(res);
}

isl::ast_expr ast_expr::add(isl::ast_expr expr2) const
{
  auto res = isl_ast_expr_add(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::address_of() const
{
  auto res = isl_ast_expr_address_of(copy());
  return manage(res);
}

isl::ast_expr ast_expr::call(isl::ast_expr_list arguments) const
{
  auto res = isl_ast_expr_call(copy(), arguments.release());
  return manage(res);
}

isl::ast_expr ast_expr::div(isl::ast_expr expr2) const
{
  auto res = isl_ast_expr_div(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::eq(isl::ast_expr expr2) const
{
  auto res = isl_ast_expr_eq(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::from_id(isl::id id)
{
  auto res = isl_ast_expr_from_id(id.release());
  return manage(res);
}

isl::ast_expr ast_expr::from_val(isl::val v)
{
  auto res = isl_ast_expr_from_val(v.release());
  return manage(res);
}

isl::ast_expr ast_expr::ge(isl::ast_expr expr2) const
{
  auto res = isl_ast_expr_ge(copy(), expr2.release());
  return manage(res);
}

isl::id ast_expr::get_id() const
{
  auto res = isl_ast_expr_get_id(get());
  return manage(res);
}

isl::ast_expr ast_expr::get_op_arg(int pos) const
{
  auto res = isl_ast_expr_get_op_arg(get(), pos);
  return manage(res);
}

isl_size ast_expr::get_op_n_arg() const
{
  auto res = isl_ast_expr_get_op_n_arg(get());
  return res;
}

isl::val ast_expr::get_val() const
{
  auto res = isl_ast_expr_get_val(get());
  return manage(res);
}

isl::ast_expr ast_expr::gt(isl::ast_expr expr2) const
{
  auto res = isl_ast_expr_gt(copy(), expr2.release());
  return manage(res);
}

isl::id ast_expr::id_get_id() const
{
  auto res = isl_ast_expr_id_get_id(get());
  return manage(res);
}

isl::val ast_expr::int_get_val() const
{
  auto res = isl_ast_expr_int_get_val(get());
  return manage(res);
}

boolean ast_expr::is_equal(const isl::ast_expr &expr2) const
{
  auto res = isl_ast_expr_is_equal(get(), expr2.get());
  return manage(res);
}

isl::ast_expr ast_expr::le(isl::ast_expr expr2) const
{
  auto res = isl_ast_expr_le(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::lt(isl::ast_expr expr2) const
{
  auto res = isl_ast_expr_lt(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::mul(isl::ast_expr expr2) const
{
  auto res = isl_ast_expr_mul(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::neg() const
{
  auto res = isl_ast_expr_neg(copy());
  return manage(res);
}

isl::ast_expr ast_expr::op_get_arg(int pos) const
{
  auto res = isl_ast_expr_op_get_arg(get(), pos);
  return manage(res);
}

isl_size ast_expr::op_get_n_arg() const
{
  auto res = isl_ast_expr_op_get_n_arg(get());
  return res;
}

isl::ast_expr ast_expr::pdiv_q(isl::ast_expr expr2) const
{
  auto res = isl_ast_expr_pdiv_q(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::pdiv_r(isl::ast_expr expr2) const
{
  auto res = isl_ast_expr_pdiv_r(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::set_op_arg(int pos, isl::ast_expr arg) const
{
  auto res = isl_ast_expr_set_op_arg(copy(), pos, arg.release());
  return manage(res);
}

isl::ast_expr ast_expr::sub(isl::ast_expr expr2) const
{
  auto res = isl_ast_expr_sub(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::substitute_ids(isl::id_to_ast_expr id2expr) const
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


isl::ctx ast_expr_list::ctx() const {
  return isl::ctx(isl_ast_expr_list_get_ctx(ptr));
}

void ast_expr_list::dump() const {
  isl_ast_expr_list_dump(get());
}


isl::ast_expr_list ast_expr_list::add(isl::ast_expr el) const
{
  auto res = isl_ast_expr_list_add(copy(), el.release());
  return manage(res);
}

isl::ast_expr_list ast_expr_list::alloc(isl::ctx ctx, int n)
{
  auto res = isl_ast_expr_list_alloc(ctx.release(), n);
  return manage(res);
}

isl::ast_expr_list ast_expr_list::clear() const
{
  auto res = isl_ast_expr_list_clear(copy());
  return manage(res);
}

isl::ast_expr_list ast_expr_list::concat(isl::ast_expr_list list2) const
{
  auto res = isl_ast_expr_list_concat(copy(), list2.release());
  return manage(res);
}

isl::ast_expr_list ast_expr_list::drop(unsigned int first, unsigned int n) const
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

isl::ast_expr_list ast_expr_list::from_ast_expr(isl::ast_expr el)
{
  auto res = isl_ast_expr_list_from_ast_expr(el.release());
  return manage(res);
}

isl::ast_expr ast_expr_list::get_ast_expr(int index) const
{
  auto res = isl_ast_expr_list_get_ast_expr(get(), index);
  return manage(res);
}

isl::ast_expr ast_expr_list::get_at(int index) const
{
  auto res = isl_ast_expr_list_get_at(get(), index);
  return manage(res);
}

isl::ast_expr_list ast_expr_list::insert(unsigned int pos, isl::ast_expr el) const
{
  auto res = isl_ast_expr_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl_size ast_expr_list::n_ast_expr() const
{
  auto res = isl_ast_expr_list_n_ast_expr(get());
  return res;
}

isl::ast_expr_list ast_expr_list::reverse() const
{
  auto res = isl_ast_expr_list_reverse(copy());
  return manage(res);
}

isl::ast_expr_list ast_expr_list::set_ast_expr(int index, isl::ast_expr el) const
{
  auto res = isl_ast_expr_list_set_ast_expr(copy(), index, el.release());
  return manage(res);
}

isl_size ast_expr_list::size() const
{
  auto res = isl_ast_expr_list_size(get());
  return res;
}

isl::ast_expr_list ast_expr_list::swap(unsigned int pos1, unsigned int pos2) const
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


isl::ctx ast_node::ctx() const {
  return isl::ctx(isl_ast_node_get_ctx(ptr));
}

void ast_node::dump() const {
  isl_ast_node_dump(get());
}


isl::ast_node ast_node::alloc_user(isl::ast_expr expr)
{
  auto res = isl_ast_node_alloc_user(expr.release());
  return manage(res);
}

isl::ast_node_list ast_node::block_get_children() const
{
  auto res = isl_ast_node_block_get_children(get());
  return manage(res);
}

isl::ast_node ast_node::for_get_body() const
{
  auto res = isl_ast_node_for_get_body(get());
  return manage(res);
}

isl::ast_expr ast_node::for_get_cond() const
{
  auto res = isl_ast_node_for_get_cond(get());
  return manage(res);
}

isl::ast_expr ast_node::for_get_inc() const
{
  auto res = isl_ast_node_for_get_inc(get());
  return manage(res);
}

isl::ast_expr ast_node::for_get_init() const
{
  auto res = isl_ast_node_for_get_init(get());
  return manage(res);
}

isl::ast_expr ast_node::for_get_iterator() const
{
  auto res = isl_ast_node_for_get_iterator(get());
  return manage(res);
}

boolean ast_node::for_is_degenerate() const
{
  auto res = isl_ast_node_for_is_degenerate(get());
  return manage(res);
}

isl::id ast_node::get_annotation() const
{
  auto res = isl_ast_node_get_annotation(get());
  return manage(res);
}

isl::ast_expr ast_node::if_get_cond() const
{
  auto res = isl_ast_node_if_get_cond(get());
  return manage(res);
}

isl::ast_node ast_node::if_get_else() const
{
  auto res = isl_ast_node_if_get_else(get());
  return manage(res);
}

isl::ast_node ast_node::if_get_else_node() const
{
  auto res = isl_ast_node_if_get_else_node(get());
  return manage(res);
}

isl::ast_node ast_node::if_get_then() const
{
  auto res = isl_ast_node_if_get_then(get());
  return manage(res);
}

isl::ast_node ast_node::if_get_then_node() const
{
  auto res = isl_ast_node_if_get_then_node(get());
  return manage(res);
}

boolean ast_node::if_has_else() const
{
  auto res = isl_ast_node_if_has_else(get());
  return manage(res);
}

boolean ast_node::if_has_else_node() const
{
  auto res = isl_ast_node_if_has_else_node(get());
  return manage(res);
}

isl::id ast_node::mark_get_id() const
{
  auto res = isl_ast_node_mark_get_id(get());
  return manage(res);
}

isl::ast_node ast_node::mark_get_node() const
{
  auto res = isl_ast_node_mark_get_node(get());
  return manage(res);
}

isl::ast_node ast_node::set_annotation(isl::id annotation) const
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

isl::ast_expr ast_node::user_get_expr() const
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


isl::ctx ast_node_list::ctx() const {
  return isl::ctx(isl_ast_node_list_get_ctx(ptr));
}

void ast_node_list::dump() const {
  isl_ast_node_list_dump(get());
}


isl::ast_node_list ast_node_list::add(isl::ast_node el) const
{
  auto res = isl_ast_node_list_add(copy(), el.release());
  return manage(res);
}

isl::ast_node_list ast_node_list::alloc(isl::ctx ctx, int n)
{
  auto res = isl_ast_node_list_alloc(ctx.release(), n);
  return manage(res);
}

isl::ast_node_list ast_node_list::clear() const
{
  auto res = isl_ast_node_list_clear(copy());
  return manage(res);
}

isl::ast_node_list ast_node_list::concat(isl::ast_node_list list2) const
{
  auto res = isl_ast_node_list_concat(copy(), list2.release());
  return manage(res);
}

isl::ast_node_list ast_node_list::drop(unsigned int first, unsigned int n) const
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

isl::ast_node_list ast_node_list::from_ast_node(isl::ast_node el)
{
  auto res = isl_ast_node_list_from_ast_node(el.release());
  return manage(res);
}

isl::ast_node ast_node_list::get_ast_node(int index) const
{
  auto res = isl_ast_node_list_get_ast_node(get(), index);
  return manage(res);
}

isl::ast_node ast_node_list::get_at(int index) const
{
  auto res = isl_ast_node_list_get_at(get(), index);
  return manage(res);
}

isl::ast_node_list ast_node_list::insert(unsigned int pos, isl::ast_node el) const
{
  auto res = isl_ast_node_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl_size ast_node_list::n_ast_node() const
{
  auto res = isl_ast_node_list_n_ast_node(get());
  return res;
}

isl::ast_node_list ast_node_list::reverse() const
{
  auto res = isl_ast_node_list_reverse(copy());
  return manage(res);
}

isl::ast_node_list ast_node_list::set_ast_node(int index, isl::ast_node el) const
{
  auto res = isl_ast_node_list_set_ast_node(copy(), index, el.release());
  return manage(res);
}

isl_size ast_node_list::size() const
{
  auto res = isl_ast_node_list_size(get());
  return res;
}

isl::ast_node_list ast_node_list::swap(unsigned int pos1, unsigned int pos2) const
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


basic_map::basic_map(__isl_take isl_basic_map *ptr)
    : ptr(ptr) {}

basic_map::basic_map(isl::ctx ctx, const std::string &str)
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


isl::ctx basic_map::ctx() const {
  return isl::ctx(isl_basic_map_get_ctx(ptr));
}

void basic_map::dump() const {
  isl_basic_map_dump(get());
}


isl::basic_map basic_map::add_constraint(isl::constraint constraint) const
{
  auto res = isl_basic_map_add_constraint(copy(), constraint.release());
  return manage(res);
}

isl::basic_map basic_map::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_basic_map_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::basic_map basic_map::affine_hull() const
{
  auto res = isl_basic_map_affine_hull(copy());
  return manage(res);
}

isl::basic_map basic_map::align_params(isl::space model) const
{
  auto res = isl_basic_map_align_params(copy(), model.release());
  return manage(res);
}

isl::basic_map basic_map::apply_domain(isl::basic_map bmap2) const
{
  auto res = isl_basic_map_apply_domain(copy(), bmap2.release());
  return manage(res);
}

isl::basic_map basic_map::apply_range(isl::basic_map bmap2) const
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

isl::basic_map basic_map::curry() const
{
  auto res = isl_basic_map_curry(copy());
  return manage(res);
}

isl::basic_set basic_map::deltas() const
{
  auto res = isl_basic_map_deltas(copy());
  return manage(res);
}

isl::basic_map basic_map::deltas_map() const
{
  auto res = isl_basic_map_deltas_map(copy());
  return manage(res);
}

isl::basic_map basic_map::detect_equalities() const
{
  auto res = isl_basic_map_detect_equalities(copy());
  return manage(res);
}

isl_size basic_map::dim(isl::dim type) const
{
  auto res = isl_basic_map_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::basic_set basic_map::domain() const
{
  auto res = isl_basic_map_domain(copy());
  return manage(res);
}

isl::basic_map basic_map::domain_map() const
{
  auto res = isl_basic_map_domain_map(copy());
  return manage(res);
}

isl::basic_map basic_map::domain_product(isl::basic_map bmap2) const
{
  auto res = isl_basic_map_domain_product(copy(), bmap2.release());
  return manage(res);
}

isl::basic_map basic_map::drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_map_drop_constraints_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_map basic_map::drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_map_drop_constraints_not_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_map basic_map::drop_unused_params() const
{
  auto res = isl_basic_map_drop_unused_params(copy());
  return manage(res);
}

isl::basic_map basic_map::eliminate(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_map_eliminate(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_map basic_map::empty(isl::space space)
{
  auto res = isl_basic_map_empty(space.release());
  return manage(res);
}

isl::basic_map basic_map::equal(isl::space space, unsigned int n_equal)
{
  auto res = isl_basic_map_equal(space.release(), n_equal);
  return manage(res);
}

isl::mat basic_map::equalities_matrix(isl::dim c1, isl::dim c2, isl::dim c3, isl::dim c4, isl::dim c5) const
{
  auto res = isl_basic_map_equalities_matrix(get(), static_cast<enum isl_dim_type>(c1), static_cast<enum isl_dim_type>(c2), static_cast<enum isl_dim_type>(c3), static_cast<enum isl_dim_type>(c4), static_cast<enum isl_dim_type>(c5));
  return manage(res);
}

isl::basic_map basic_map::equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_basic_map_equate(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

int basic_map::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_basic_map_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::basic_map basic_map::fix_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_basic_map_fix_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::basic_map basic_map::fix_val(isl::dim type, unsigned int pos, isl::val v) const
{
  auto res = isl_basic_map_fix_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

isl::basic_map basic_map::flat_product(isl::basic_map bmap2) const
{
  auto res = isl_basic_map_flat_product(copy(), bmap2.release());
  return manage(res);
}

isl::basic_map basic_map::flat_range_product(isl::basic_map bmap2) const
{
  auto res = isl_basic_map_flat_range_product(copy(), bmap2.release());
  return manage(res);
}

isl::basic_map basic_map::flatten() const
{
  auto res = isl_basic_map_flatten(copy());
  return manage(res);
}

isl::basic_map basic_map::flatten_domain() const
{
  auto res = isl_basic_map_flatten_domain(copy());
  return manage(res);
}

isl::basic_map basic_map::flatten_range() const
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

isl::basic_map basic_map::from_aff(isl::aff aff)
{
  auto res = isl_basic_map_from_aff(aff.release());
  return manage(res);
}

isl::basic_map basic_map::from_aff_list(isl::space domain_space, isl::aff_list list)
{
  auto res = isl_basic_map_from_aff_list(domain_space.release(), list.release());
  return manage(res);
}

isl::basic_map basic_map::from_constraint(isl::constraint constraint)
{
  auto res = isl_basic_map_from_constraint(constraint.release());
  return manage(res);
}

isl::basic_map basic_map::from_domain(isl::basic_set bset)
{
  auto res = isl_basic_map_from_domain(bset.release());
  return manage(res);
}

isl::basic_map basic_map::from_domain_and_range(isl::basic_set domain, isl::basic_set range)
{
  auto res = isl_basic_map_from_domain_and_range(domain.release(), range.release());
  return manage(res);
}

isl::basic_map basic_map::from_multi_aff(isl::multi_aff maff)
{
  auto res = isl_basic_map_from_multi_aff(maff.release());
  return manage(res);
}

isl::basic_map basic_map::from_qpolynomial(isl::qpolynomial qp)
{
  auto res = isl_basic_map_from_qpolynomial(qp.release());
  return manage(res);
}

isl::basic_map basic_map::from_range(isl::basic_set bset)
{
  auto res = isl_basic_map_from_range(bset.release());
  return manage(res);
}

isl::constraint_list basic_map::get_constraint_list() const
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

isl::aff basic_map::get_div(int pos) const
{
  auto res = isl_basic_map_get_div(get(), pos);
  return manage(res);
}

isl::local_space basic_map::get_local_space() const
{
  auto res = isl_basic_map_get_local_space(get());
  return manage(res);
}

isl::space basic_map::get_space() const
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

isl::basic_map basic_map::gist(isl::basic_map context) const
{
  auto res = isl_basic_map_gist(copy(), context.release());
  return manage(res);
}

isl::basic_map basic_map::gist_domain(isl::basic_set context) const
{
  auto res = isl_basic_map_gist_domain(copy(), context.release());
  return manage(res);
}

boolean basic_map::has_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_basic_map_has_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::basic_map basic_map::identity(isl::space space)
{
  auto res = isl_basic_map_identity(space.release());
  return manage(res);
}

boolean basic_map::image_is_bounded() const
{
  auto res = isl_basic_map_image_is_bounded(get());
  return manage(res);
}

isl::mat basic_map::inequalities_matrix(isl::dim c1, isl::dim c2, isl::dim c3, isl::dim c4, isl::dim c5) const
{
  auto res = isl_basic_map_inequalities_matrix(get(), static_cast<enum isl_dim_type>(c1), static_cast<enum isl_dim_type>(c2), static_cast<enum isl_dim_type>(c3), static_cast<enum isl_dim_type>(c4), static_cast<enum isl_dim_type>(c5));
  return manage(res);
}

isl::basic_map basic_map::insert_dims(isl::dim type, unsigned int pos, unsigned int n) const
{
  auto res = isl_basic_map_insert_dims(copy(), static_cast<enum isl_dim_type>(type), pos, n);
  return manage(res);
}

isl::basic_map basic_map::intersect(isl::basic_map bmap2) const
{
  auto res = isl_basic_map_intersect(copy(), bmap2.release());
  return manage(res);
}

isl::basic_map basic_map::intersect_domain(isl::basic_set bset) const
{
  auto res = isl_basic_map_intersect_domain(copy(), bset.release());
  return manage(res);
}

isl::basic_map basic_map::intersect_range(isl::basic_set bset) const
{
  auto res = isl_basic_map_intersect_range(copy(), bset.release());
  return manage(res);
}

boolean basic_map::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_map_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean basic_map::is_disjoint(const isl::basic_map &bmap2) const
{
  auto res = isl_basic_map_is_disjoint(get(), bmap2.get());
  return manage(res);
}

boolean basic_map::is_empty() const
{
  auto res = isl_basic_map_is_empty(get());
  return manage(res);
}

boolean basic_map::is_equal(const isl::basic_map &bmap2) const
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

boolean basic_map::is_strict_subset(const isl::basic_map &bmap2) const
{
  auto res = isl_basic_map_is_strict_subset(get(), bmap2.get());
  return manage(res);
}

boolean basic_map::is_subset(const isl::basic_map &bmap2) const
{
  auto res = isl_basic_map_is_subset(get(), bmap2.get());
  return manage(res);
}

boolean basic_map::is_universe() const
{
  auto res = isl_basic_map_is_universe(get());
  return manage(res);
}

isl::basic_map basic_map::less_at(isl::space space, unsigned int pos)
{
  auto res = isl_basic_map_less_at(space.release(), pos);
  return manage(res);
}

isl::map basic_map::lexmax() const
{
  auto res = isl_basic_map_lexmax(copy());
  return manage(res);
}

isl::map basic_map::lexmin() const
{
  auto res = isl_basic_map_lexmin(copy());
  return manage(res);
}

isl::pw_multi_aff basic_map::lexmin_pw_multi_aff() const
{
  auto res = isl_basic_map_lexmin_pw_multi_aff(copy());
  return manage(res);
}

isl::basic_map basic_map::lower_bound_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_basic_map_lower_bound_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::basic_map basic_map::more_at(isl::space space, unsigned int pos)
{
  auto res = isl_basic_map_more_at(space.release(), pos);
  return manage(res);
}

isl::basic_map basic_map::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_basic_map_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl_size basic_map::n_constraint() const
{
  auto res = isl_basic_map_n_constraint(get());
  return res;
}

isl::basic_map basic_map::nat_universe(isl::space space)
{
  auto res = isl_basic_map_nat_universe(space.release());
  return manage(res);
}

isl::basic_map basic_map::neg() const
{
  auto res = isl_basic_map_neg(copy());
  return manage(res);
}

isl::basic_map basic_map::order_ge(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_basic_map_order_ge(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

isl::basic_map basic_map::order_gt(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_basic_map_order_gt(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

isl::val basic_map::plain_get_val_if_fixed(isl::dim type, unsigned int pos) const
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

isl::basic_map basic_map::preimage_domain_multi_aff(isl::multi_aff ma) const
{
  auto res = isl_basic_map_preimage_domain_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::basic_map basic_map::preimage_range_multi_aff(isl::multi_aff ma) const
{
  auto res = isl_basic_map_preimage_range_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::basic_map basic_map::product(isl::basic_map bmap2) const
{
  auto res = isl_basic_map_product(copy(), bmap2.release());
  return manage(res);
}

isl::basic_map basic_map::project_out(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_map_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_set basic_map::range() const
{
  auto res = isl_basic_map_range(copy());
  return manage(res);
}

isl::basic_map basic_map::range_map() const
{
  auto res = isl_basic_map_range_map(copy());
  return manage(res);
}

isl::basic_map basic_map::range_product(isl::basic_map bmap2) const
{
  auto res = isl_basic_map_range_product(copy(), bmap2.release());
  return manage(res);
}

isl::basic_map basic_map::remove_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_map_remove_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_map basic_map::remove_divs() const
{
  auto res = isl_basic_map_remove_divs(copy());
  return manage(res);
}

isl::basic_map basic_map::remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_map_remove_divs_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_map basic_map::remove_redundancies() const
{
  auto res = isl_basic_map_remove_redundancies(copy());
  return manage(res);
}

isl::basic_map basic_map::reverse() const
{
  auto res = isl_basic_map_reverse(copy());
  return manage(res);
}

isl::basic_map basic_map::sample() const
{
  auto res = isl_basic_map_sample(copy());
  return manage(res);
}

isl::basic_map basic_map::set_tuple_id(isl::dim type, isl::id id) const
{
  auto res = isl_basic_map_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::basic_map basic_map::set_tuple_name(isl::dim type, const std::string &s) const
{
  auto res = isl_basic_map_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

isl::basic_map basic_map::sum(isl::basic_map bmap2) const
{
  auto res = isl_basic_map_sum(copy(), bmap2.release());
  return manage(res);
}

isl::basic_map basic_map::uncurry() const
{
  auto res = isl_basic_map_uncurry(copy());
  return manage(res);
}

isl::map basic_map::unite(isl::basic_map bmap2) const
{
  auto res = isl_basic_map_union(copy(), bmap2.release());
  return manage(res);
}

isl::basic_map basic_map::universe(isl::space space)
{
  auto res = isl_basic_map_universe(space.release());
  return manage(res);
}

isl::basic_map basic_map::upper_bound_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_basic_map_upper_bound_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::basic_set basic_map::wrap() const
{
  auto res = isl_basic_map_wrap(copy());
  return manage(res);
}

isl::basic_map basic_map::zip() const
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


isl::ctx basic_map_list::ctx() const {
  return isl::ctx(isl_basic_map_list_get_ctx(ptr));
}

void basic_map_list::dump() const {
  isl_basic_map_list_dump(get());
}


isl::basic_map_list basic_map_list::add(isl::basic_map el) const
{
  auto res = isl_basic_map_list_add(copy(), el.release());
  return manage(res);
}

isl::basic_map_list basic_map_list::alloc(isl::ctx ctx, int n)
{
  auto res = isl_basic_map_list_alloc(ctx.release(), n);
  return manage(res);
}

isl::basic_map_list basic_map_list::clear() const
{
  auto res = isl_basic_map_list_clear(copy());
  return manage(res);
}

isl::basic_map_list basic_map_list::concat(isl::basic_map_list list2) const
{
  auto res = isl_basic_map_list_concat(copy(), list2.release());
  return manage(res);
}

isl::basic_map_list basic_map_list::drop(unsigned int first, unsigned int n) const
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

isl::basic_map_list basic_map_list::from_basic_map(isl::basic_map el)
{
  auto res = isl_basic_map_list_from_basic_map(el.release());
  return manage(res);
}

isl::basic_map basic_map_list::get_at(int index) const
{
  auto res = isl_basic_map_list_get_at(get(), index);
  return manage(res);
}

isl::basic_map basic_map_list::get_basic_map(int index) const
{
  auto res = isl_basic_map_list_get_basic_map(get(), index);
  return manage(res);
}

isl::basic_map_list basic_map_list::insert(unsigned int pos, isl::basic_map el) const
{
  auto res = isl_basic_map_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl_size basic_map_list::n_basic_map() const
{
  auto res = isl_basic_map_list_n_basic_map(get());
  return res;
}

isl::basic_map_list basic_map_list::reverse() const
{
  auto res = isl_basic_map_list_reverse(copy());
  return manage(res);
}

isl::basic_map_list basic_map_list::set_basic_map(int index, isl::basic_map el) const
{
  auto res = isl_basic_map_list_set_basic_map(copy(), index, el.release());
  return manage(res);
}

isl_size basic_map_list::size() const
{
  auto res = isl_basic_map_list_size(get());
  return res;
}

isl::basic_map_list basic_map_list::swap(unsigned int pos1, unsigned int pos2) const
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


basic_set::basic_set(__isl_take isl_basic_set *ptr)
    : ptr(ptr) {}

basic_set::basic_set(isl::point pnt)
{
  auto res = isl_basic_set_from_point(pnt.release());
  ptr = res;
}
basic_set::basic_set(isl::ctx ctx, const std::string &str)
{
  auto res = isl_basic_set_read_from_str(ctx.release(), str.c_str());
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


isl::ctx basic_set::ctx() const {
  return isl::ctx(isl_basic_set_get_ctx(ptr));
}

void basic_set::dump() const {
  isl_basic_set_dump(get());
}


isl::basic_set basic_set::affine_hull() const
{
  auto res = isl_basic_set_affine_hull(copy());
  return manage(res);
}

isl::basic_set basic_set::align_params(isl::space model) const
{
  auto res = isl_basic_set_align_params(copy(), model.release());
  return manage(res);
}

isl::basic_set basic_set::apply(isl::basic_map bmap) const
{
  auto res = isl_basic_set_apply(copy(), bmap.release());
  return manage(res);
}

isl::basic_set basic_set::box_from_points(isl::point pnt1, isl::point pnt2)
{
  auto res = isl_basic_set_box_from_points(pnt1.release(), pnt2.release());
  return manage(res);
}

isl::basic_set basic_set::coefficients() const
{
  auto res = isl_basic_set_coefficients(copy());
  return manage(res);
}

isl::basic_set basic_set::detect_equalities() const
{
  auto res = isl_basic_set_detect_equalities(copy());
  return manage(res);
}

isl_size basic_set::dim(isl::dim type) const
{
  auto res = isl_basic_set_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::val basic_set::dim_max_val(int pos) const
{
  auto res = isl_basic_set_dim_max_val(copy(), pos);
  return manage(res);
}

isl::basic_set basic_set::drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_set_drop_constraints_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_set basic_set::drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_set_drop_constraints_not_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_set basic_set::drop_unused_params() const
{
  auto res = isl_basic_set_drop_unused_params(copy());
  return manage(res);
}

isl::basic_set basic_set::eliminate(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_set_eliminate(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_set basic_set::empty(isl::space space)
{
  auto res = isl_basic_set_empty(space.release());
  return manage(res);
}

isl::mat basic_set::equalities_matrix(isl::dim c1, isl::dim c2, isl::dim c3, isl::dim c4) const
{
  auto res = isl_basic_set_equalities_matrix(get(), static_cast<enum isl_dim_type>(c1), static_cast<enum isl_dim_type>(c2), static_cast<enum isl_dim_type>(c3), static_cast<enum isl_dim_type>(c4));
  return manage(res);
}

isl::basic_set basic_set::fix_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_basic_set_fix_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::basic_set basic_set::fix_val(isl::dim type, unsigned int pos, isl::val v) const
{
  auto res = isl_basic_set_fix_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

isl::basic_set basic_set::flat_product(isl::basic_set bset2) const
{
  auto res = isl_basic_set_flat_product(copy(), bset2.release());
  return manage(res);
}

isl::basic_set basic_set::flatten() const
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

isl::basic_set basic_set::from_constraint(isl::constraint constraint)
{
  auto res = isl_basic_set_from_constraint(constraint.release());
  return manage(res);
}

isl::basic_set basic_set::from_multi_aff(isl::multi_aff ma)
{
  auto res = isl_basic_set_from_multi_aff(ma.release());
  return manage(res);
}

isl::basic_set basic_set::from_params() const
{
  auto res = isl_basic_set_from_params(copy());
  return manage(res);
}

isl::constraint_list basic_set::get_constraint_list() const
{
  auto res = isl_basic_set_get_constraint_list(get());
  return manage(res);
}

isl::id basic_set::get_dim_id(isl::dim type, unsigned int pos) const
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

isl::aff basic_set::get_div(int pos) const
{
  auto res = isl_basic_set_get_div(get(), pos);
  return manage(res);
}

isl::local_space basic_set::get_local_space() const
{
  auto res = isl_basic_set_get_local_space(get());
  return manage(res);
}

isl::space basic_set::get_space() const
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

isl::basic_set basic_set::gist(isl::basic_set context) const
{
  auto res = isl_basic_set_gist(copy(), context.release());
  return manage(res);
}

isl::mat basic_set::inequalities_matrix(isl::dim c1, isl::dim c2, isl::dim c3, isl::dim c4) const
{
  auto res = isl_basic_set_inequalities_matrix(get(), static_cast<enum isl_dim_type>(c1), static_cast<enum isl_dim_type>(c2), static_cast<enum isl_dim_type>(c3), static_cast<enum isl_dim_type>(c4));
  return manage(res);
}

isl::basic_set basic_set::insert_dims(isl::dim type, unsigned int pos, unsigned int n) const
{
  auto res = isl_basic_set_insert_dims(copy(), static_cast<enum isl_dim_type>(type), pos, n);
  return manage(res);
}

isl::basic_set basic_set::intersect(isl::basic_set bset2) const
{
  auto res = isl_basic_set_intersect(copy(), bset2.release());
  return manage(res);
}

isl::basic_set basic_set::intersect_params(isl::basic_set bset2) const
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

boolean basic_set::is_disjoint(const isl::basic_set &bset2) const
{
  auto res = isl_basic_set_is_disjoint(get(), bset2.get());
  return manage(res);
}

boolean basic_set::is_empty() const
{
  auto res = isl_basic_set_is_empty(get());
  return manage(res);
}

boolean basic_set::is_equal(const isl::basic_set &bset2) const
{
  auto res = isl_basic_set_is_equal(get(), bset2.get());
  return manage(res);
}

int basic_set::is_rational() const
{
  auto res = isl_basic_set_is_rational(get());
  return res;
}

boolean basic_set::is_subset(const isl::basic_set &bset2) const
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

isl::set basic_set::lexmax() const
{
  auto res = isl_basic_set_lexmax(copy());
  return manage(res);
}

isl::set basic_set::lexmin() const
{
  auto res = isl_basic_set_lexmin(copy());
  return manage(res);
}

isl::basic_set basic_set::lower_bound_val(isl::dim type, unsigned int pos, isl::val value) const
{
  auto res = isl_basic_set_lower_bound_val(copy(), static_cast<enum isl_dim_type>(type), pos, value.release());
  return manage(res);
}

isl::val basic_set::max_val(const isl::aff &obj) const
{
  auto res = isl_basic_set_max_val(get(), obj.get());
  return manage(res);
}

isl::basic_set basic_set::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_basic_set_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl_size basic_set::n_constraint() const
{
  auto res = isl_basic_set_n_constraint(get());
  return res;
}

isl_size basic_set::n_dim() const
{
  auto res = isl_basic_set_n_dim(get());
  return res;
}

isl::basic_set basic_set::nat_universe(isl::space space)
{
  auto res = isl_basic_set_nat_universe(space.release());
  return manage(res);
}

isl::basic_set basic_set::neg() const
{
  auto res = isl_basic_set_neg(copy());
  return manage(res);
}

isl::basic_set basic_set::params() const
{
  auto res = isl_basic_set_params(copy());
  return manage(res);
}

boolean basic_set::plain_is_empty() const
{
  auto res = isl_basic_set_plain_is_empty(get());
  return manage(res);
}

boolean basic_set::plain_is_equal(const isl::basic_set &bset2) const
{
  auto res = isl_basic_set_plain_is_equal(get(), bset2.get());
  return manage(res);
}

boolean basic_set::plain_is_universe() const
{
  auto res = isl_basic_set_plain_is_universe(get());
  return manage(res);
}

isl::basic_set basic_set::positive_orthant(isl::space space)
{
  auto res = isl_basic_set_positive_orthant(space.release());
  return manage(res);
}

isl::basic_set basic_set::preimage_multi_aff(isl::multi_aff ma) const
{
  auto res = isl_basic_set_preimage_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::basic_set basic_set::project_out(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_set_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::mat basic_set::reduced_basis() const
{
  auto res = isl_basic_set_reduced_basis(get());
  return manage(res);
}

isl::basic_set basic_set::remove_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_set_remove_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_set basic_set::remove_divs() const
{
  auto res = isl_basic_set_remove_divs(copy());
  return manage(res);
}

isl::basic_set basic_set::remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_set_remove_divs_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_set basic_set::remove_redundancies() const
{
  auto res = isl_basic_set_remove_redundancies(copy());
  return manage(res);
}

isl::basic_set basic_set::remove_unknown_divs() const
{
  auto res = isl_basic_set_remove_unknown_divs(copy());
  return manage(res);
}

isl::basic_set basic_set::sample() const
{
  auto res = isl_basic_set_sample(copy());
  return manage(res);
}

isl::point basic_set::sample_point() const
{
  auto res = isl_basic_set_sample_point(copy());
  return manage(res);
}

isl::basic_set basic_set::set_tuple_id(isl::id id) const
{
  auto res = isl_basic_set_set_tuple_id(copy(), id.release());
  return manage(res);
}

isl::basic_set basic_set::set_tuple_name(const std::string &s) const
{
  auto res = isl_basic_set_set_tuple_name(copy(), s.c_str());
  return manage(res);
}

isl::basic_set basic_set::solutions() const
{
  auto res = isl_basic_set_solutions(copy());
  return manage(res);
}

isl::set basic_set::unite(isl::basic_set bset2) const
{
  auto res = isl_basic_set_union(copy(), bset2.release());
  return manage(res);
}

isl::basic_set basic_set::universe(isl::space space)
{
  auto res = isl_basic_set_universe(space.release());
  return manage(res);
}

isl::basic_map basic_set::unwrap() const
{
  auto res = isl_basic_set_unwrap(copy());
  return manage(res);
}

isl::basic_set basic_set::upper_bound_val(isl::dim type, unsigned int pos, isl::val value) const
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


isl::ctx basic_set_list::ctx() const {
  return isl::ctx(isl_basic_set_list_get_ctx(ptr));
}

void basic_set_list::dump() const {
  isl_basic_set_list_dump(get());
}


isl::basic_set_list basic_set_list::add(isl::basic_set el) const
{
  auto res = isl_basic_set_list_add(copy(), el.release());
  return manage(res);
}

isl::basic_set_list basic_set_list::alloc(isl::ctx ctx, int n)
{
  auto res = isl_basic_set_list_alloc(ctx.release(), n);
  return manage(res);
}

isl::basic_set_list basic_set_list::clear() const
{
  auto res = isl_basic_set_list_clear(copy());
  return manage(res);
}

isl::basic_set_list basic_set_list::coefficients() const
{
  auto res = isl_basic_set_list_coefficients(copy());
  return manage(res);
}

isl::basic_set_list basic_set_list::concat(isl::basic_set_list list2) const
{
  auto res = isl_basic_set_list_concat(copy(), list2.release());
  return manage(res);
}

isl::basic_set_list basic_set_list::drop(unsigned int first, unsigned int n) const
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

isl::basic_set_list basic_set_list::from_basic_set(isl::basic_set el)
{
  auto res = isl_basic_set_list_from_basic_set(el.release());
  return manage(res);
}

isl::basic_set basic_set_list::get_at(int index) const
{
  auto res = isl_basic_set_list_get_at(get(), index);
  return manage(res);
}

isl::basic_set basic_set_list::get_basic_set(int index) const
{
  auto res = isl_basic_set_list_get_basic_set(get(), index);
  return manage(res);
}

isl::basic_set_list basic_set_list::insert(unsigned int pos, isl::basic_set el) const
{
  auto res = isl_basic_set_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl_size basic_set_list::n_basic_set() const
{
  auto res = isl_basic_set_list_n_basic_set(get());
  return res;
}

isl::basic_set_list basic_set_list::reverse() const
{
  auto res = isl_basic_set_list_reverse(copy());
  return manage(res);
}

isl::basic_set_list basic_set_list::set_basic_set(int index, isl::basic_set el) const
{
  auto res = isl_basic_set_list_set_basic_set(copy(), index, el.release());
  return manage(res);
}

isl_size basic_set_list::size() const
{
  auto res = isl_basic_set_list_size(get());
  return res;
}

isl::basic_set_list basic_set_list::swap(unsigned int pos1, unsigned int pos2) const
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


isl::ctx constraint::ctx() const {
  return isl::ctx(isl_constraint_get_ctx(ptr));
}

void constraint::dump() const {
  isl_constraint_dump(get());
}


isl::constraint constraint::alloc_equality(isl::local_space ls)
{
  auto res = isl_constraint_alloc_equality(ls.release());
  return manage(res);
}

isl::constraint constraint::alloc_inequality(isl::local_space ls)
{
  auto res = isl_constraint_alloc_inequality(ls.release());
  return manage(res);
}

int constraint::cmp_last_non_zero(const isl::constraint &c2) const
{
  auto res = isl_constraint_cmp_last_non_zero(get(), c2.get());
  return res;
}

isl::aff constraint::get_aff() const
{
  auto res = isl_constraint_get_aff(get());
  return manage(res);
}

isl::aff constraint::get_bound(isl::dim type, int pos) const
{
  auto res = isl_constraint_get_bound(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::val constraint::get_coefficient_val(isl::dim type, int pos) const
{
  auto res = isl_constraint_get_coefficient_val(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::val constraint::get_constant_val() const
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

isl::aff constraint::get_div(int pos) const
{
  auto res = isl_constraint_get_div(get(), pos);
  return manage(res);
}

isl::local_space constraint::get_local_space() const
{
  auto res = isl_constraint_get_local_space(get());
  return manage(res);
}

isl::space constraint::get_space() const
{
  auto res = isl_constraint_get_space(get());
  return manage(res);
}

boolean constraint::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_constraint_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean constraint::is_div_constraint() const
{
  auto res = isl_constraint_is_div_constraint(get());
  return manage(res);
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

int constraint::plain_cmp(const isl::constraint &c2) const
{
  auto res = isl_constraint_plain_cmp(get(), c2.get());
  return res;
}

isl::constraint constraint::set_coefficient_si(isl::dim type, int pos, int v) const
{
  auto res = isl_constraint_set_coefficient_si(copy(), static_cast<enum isl_dim_type>(type), pos, v);
  return manage(res);
}

isl::constraint constraint::set_coefficient_val(isl::dim type, int pos, isl::val v) const
{
  auto res = isl_constraint_set_coefficient_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

isl::constraint constraint::set_constant_si(int v) const
{
  auto res = isl_constraint_set_constant_si(copy(), v);
  return manage(res);
}

isl::constraint constraint::set_constant_val(isl::val v) const
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


isl::ctx constraint_list::ctx() const {
  return isl::ctx(isl_constraint_list_get_ctx(ptr));
}

void constraint_list::dump() const {
  isl_constraint_list_dump(get());
}


isl::constraint_list constraint_list::add(isl::constraint el) const
{
  auto res = isl_constraint_list_add(copy(), el.release());
  return manage(res);
}

isl::constraint_list constraint_list::alloc(isl::ctx ctx, int n)
{
  auto res = isl_constraint_list_alloc(ctx.release(), n);
  return manage(res);
}

isl::constraint_list constraint_list::clear() const
{
  auto res = isl_constraint_list_clear(copy());
  return manage(res);
}

isl::constraint_list constraint_list::concat(isl::constraint_list list2) const
{
  auto res = isl_constraint_list_concat(copy(), list2.release());
  return manage(res);
}

isl::constraint_list constraint_list::drop(unsigned int first, unsigned int n) const
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

isl::constraint_list constraint_list::from_constraint(isl::constraint el)
{
  auto res = isl_constraint_list_from_constraint(el.release());
  return manage(res);
}

isl::constraint constraint_list::get_at(int index) const
{
  auto res = isl_constraint_list_get_at(get(), index);
  return manage(res);
}

isl::constraint constraint_list::get_constraint(int index) const
{
  auto res = isl_constraint_list_get_constraint(get(), index);
  return manage(res);
}

isl::constraint_list constraint_list::insert(unsigned int pos, isl::constraint el) const
{
  auto res = isl_constraint_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl_size constraint_list::n_constraint() const
{
  auto res = isl_constraint_list_n_constraint(get());
  return res;
}

isl::constraint_list constraint_list::reverse() const
{
  auto res = isl_constraint_list_reverse(copy());
  return manage(res);
}

isl::constraint_list constraint_list::set_constraint(int index, isl::constraint el) const
{
  auto res = isl_constraint_list_set_constraint(copy(), index, el.release());
  return manage(res);
}

isl_size constraint_list::size() const
{
  auto res = isl_constraint_list_size(get());
  return res;
}

isl::constraint_list constraint_list::swap(unsigned int pos1, unsigned int pos2) const
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


isl::ctx fixed_box::ctx() const {
  return isl::ctx(isl_fixed_box_get_ctx(ptr));
}

void fixed_box::dump() const {
  isl_fixed_box_dump(get());
}


isl::multi_aff fixed_box::get_offset() const
{
  auto res = isl_fixed_box_get_offset(get());
  return manage(res);
}

isl::multi_val fixed_box::get_size() const
{
  auto res = isl_fixed_box_get_size(get());
  return manage(res);
}

isl::space fixed_box::get_space() const
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


id::id(__isl_take isl_id *ptr)
    : ptr(ptr) {}

id::id(isl::ctx ctx, const std::string &str)
{
  auto res = isl_id_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

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


isl::ctx id::ctx() const {
  return isl::ctx(isl_id_get_ctx(ptr));
}

void id::dump() const {
  isl_id_dump(get());
}


isl::id id::alloc(isl::ctx ctx, const std::string &name, void * user)
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


isl::ctx id_list::ctx() const {
  return isl::ctx(isl_id_list_get_ctx(ptr));
}

void id_list::dump() const {
  isl_id_list_dump(get());
}


isl::id_list id_list::add(isl::id el) const
{
  auto res = isl_id_list_add(copy(), el.release());
  return manage(res);
}

isl::id_list id_list::alloc(isl::ctx ctx, int n)
{
  auto res = isl_id_list_alloc(ctx.release(), n);
  return manage(res);
}

isl::id_list id_list::clear() const
{
  auto res = isl_id_list_clear(copy());
  return manage(res);
}

isl::id_list id_list::concat(isl::id_list list2) const
{
  auto res = isl_id_list_concat(copy(), list2.release());
  return manage(res);
}

isl::id_list id_list::drop(unsigned int first, unsigned int n) const
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

isl::id_list id_list::from_id(isl::id el)
{
  auto res = isl_id_list_from_id(el.release());
  return manage(res);
}

isl::id id_list::get_at(int index) const
{
  auto res = isl_id_list_get_at(get(), index);
  return manage(res);
}

isl::id id_list::get_id(int index) const
{
  auto res = isl_id_list_get_id(get(), index);
  return manage(res);
}

isl::id_list id_list::insert(unsigned int pos, isl::id el) const
{
  auto res = isl_id_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl_size id_list::n_id() const
{
  auto res = isl_id_list_n_id(get());
  return res;
}

isl::id_list id_list::reverse() const
{
  auto res = isl_id_list_reverse(copy());
  return manage(res);
}

isl::id_list id_list::set_id(int index, isl::id el) const
{
  auto res = isl_id_list_set_id(copy(), index, el.release());
  return manage(res);
}

isl_size id_list::size() const
{
  auto res = isl_id_list_size(get());
  return res;
}

isl::id_list id_list::swap(unsigned int pos1, unsigned int pos2) const
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


isl::ctx id_to_ast_expr::ctx() const {
  return isl::ctx(isl_id_to_ast_expr_get_ctx(ptr));
}

void id_to_ast_expr::dump() const {
  isl_id_to_ast_expr_dump(get());
}


isl::id_to_ast_expr id_to_ast_expr::alloc(isl::ctx ctx, int min_size)
{
  auto res = isl_id_to_ast_expr_alloc(ctx.release(), min_size);
  return manage(res);
}

isl::id_to_ast_expr id_to_ast_expr::drop(isl::id key) const
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

isl::ast_expr id_to_ast_expr::get(isl::id key) const
{
  auto res = isl_id_to_ast_expr_get(get(), key.release());
  return manage(res);
}

boolean id_to_ast_expr::has(const isl::id &key) const
{
  auto res = isl_id_to_ast_expr_has(get(), key.get());
  return manage(res);
}

isl::id_to_ast_expr id_to_ast_expr::set(isl::id key, isl::ast_expr val) const
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


local_space::local_space(__isl_take isl_local_space *ptr)
    : ptr(ptr) {}

local_space::local_space(isl::space space)
{
  auto res = isl_local_space_from_space(space.release());
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


isl::ctx local_space::ctx() const {
  return isl::ctx(isl_local_space_get_ctx(ptr));
}

void local_space::dump() const {
  isl_local_space_dump(get());
}


isl::local_space local_space::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_local_space_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl_size local_space::dim(isl::dim type) const
{
  auto res = isl_local_space_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::local_space local_space::domain() const
{
  auto res = isl_local_space_domain(copy());
  return manage(res);
}

isl::local_space local_space::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_local_space_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

int local_space::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_local_space_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::local_space local_space::flatten_domain() const
{
  auto res = isl_local_space_flatten_domain(copy());
  return manage(res);
}

isl::local_space local_space::flatten_range() const
{
  auto res = isl_local_space_flatten_range(copy());
  return manage(res);
}

isl::local_space local_space::from_domain() const
{
  auto res = isl_local_space_from_domain(copy());
  return manage(res);
}

isl::id local_space::get_dim_id(isl::dim type, unsigned int pos) const
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

isl::aff local_space::get_div(int pos) const
{
  auto res = isl_local_space_get_div(get(), pos);
  return manage(res);
}

isl::space local_space::get_space() const
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

isl::local_space local_space::insert_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_local_space_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::local_space local_space::intersect(isl::local_space ls2) const
{
  auto res = isl_local_space_intersect(copy(), ls2.release());
  return manage(res);
}

boolean local_space::is_equal(const isl::local_space &ls2) const
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

isl::local_space local_space::range() const
{
  auto res = isl_local_space_range(copy());
  return manage(res);
}

isl::local_space local_space::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const
{
  auto res = isl_local_space_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::local_space local_space::set_from_params() const
{
  auto res = isl_local_space_set_from_params(copy());
  return manage(res);
}

isl::local_space local_space::set_tuple_id(isl::dim type, isl::id id) const
{
  auto res = isl_local_space_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::local_space local_space::wrap() const
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


map::map(__isl_take isl_map *ptr)
    : ptr(ptr) {}

map::map(isl::basic_map bmap)
{
  auto res = isl_map_from_basic_map(bmap.release());
  ptr = res;
}
map::map(isl::ctx ctx, const std::string &str)
{
  auto res = isl_map_read_from_str(ctx.release(), str.c_str());
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


isl::ctx map::ctx() const {
  return isl::ctx(isl_map_get_ctx(ptr));
}

void map::dump() const {
  isl_map_dump(get());
}


isl::map map::add_constraint(isl::constraint constraint) const
{
  auto res = isl_map_add_constraint(copy(), constraint.release());
  return manage(res);
}

isl::map map::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_map_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::basic_map map::affine_hull() const
{
  auto res = isl_map_affine_hull(copy());
  return manage(res);
}

isl::map map::align_params(isl::space model) const
{
  auto res = isl_map_align_params(copy(), model.release());
  return manage(res);
}

isl::map map::apply_domain(isl::map map2) const
{
  auto res = isl_map_apply_domain(copy(), map2.release());
  return manage(res);
}

isl::map map::apply_range(isl::map map2) const
{
  auto res = isl_map_apply_range(copy(), map2.release());
  return manage(res);
}

isl::set map::bind_domain(isl::multi_id tuple) const
{
  auto res = isl_map_bind_domain(copy(), tuple.release());
  return manage(res);
}

isl::set map::bind_range(isl::multi_id tuple) const
{
  auto res = isl_map_bind_range(copy(), tuple.release());
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

isl::map map::coalesce() const
{
  auto res = isl_map_coalesce(copy());
  return manage(res);
}

isl::map map::complement() const
{
  auto res = isl_map_complement(copy());
  return manage(res);
}

isl::basic_map map::convex_hull() const
{
  auto res = isl_map_convex_hull(copy());
  return manage(res);
}

isl::map map::curry() const
{
  auto res = isl_map_curry(copy());
  return manage(res);
}

isl::set map::deltas() const
{
  auto res = isl_map_deltas(copy());
  return manage(res);
}

isl::map map::deltas_map() const
{
  auto res = isl_map_deltas_map(copy());
  return manage(res);
}

isl::map map::detect_equalities() const
{
  auto res = isl_map_detect_equalities(copy());
  return manage(res);
}

isl_size map::dim(isl::dim type) const
{
  auto res = isl_map_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::pw_aff map::dim_max(int pos) const
{
  auto res = isl_map_dim_max(copy(), pos);
  return manage(res);
}

isl::pw_aff map::dim_min(int pos) const
{
  auto res = isl_map_dim_min(copy(), pos);
  return manage(res);
}

isl::set map::domain() const
{
  auto res = isl_map_domain(copy());
  return manage(res);
}

isl::map map::domain_factor_domain() const
{
  auto res = isl_map_domain_factor_domain(copy());
  return manage(res);
}

isl::map map::domain_factor_range() const
{
  auto res = isl_map_domain_factor_range(copy());
  return manage(res);
}

boolean map::domain_is_wrapping() const
{
  auto res = isl_map_domain_is_wrapping(get());
  return manage(res);
}

isl::map map::domain_map() const
{
  auto res = isl_map_domain_map(copy());
  return manage(res);
}

isl::map map::domain_product(isl::map map2) const
{
  auto res = isl_map_domain_product(copy(), map2.release());
  return manage(res);
}

isl_size map::domain_tuple_dim() const
{
  auto res = isl_map_domain_tuple_dim(get());
  return res;
}

isl::map map::drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_map_drop_constraints_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::map map::drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_map_drop_constraints_not_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::map map::drop_unused_params() const
{
  auto res = isl_map_drop_unused_params(copy());
  return manage(res);
}

isl::map map::eliminate(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_map_eliminate(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::map map::empty(isl::space space)
{
  auto res = isl_map_empty(space.release());
  return manage(res);
}

isl::map map::eq_at(isl::multi_pw_aff mpa) const
{
  auto res = isl_map_eq_at_multi_pw_aff(copy(), mpa.release());
  return manage(res);
}

isl::map map::equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_map_equate(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

isl::map map::factor_domain() const
{
  auto res = isl_map_factor_domain(copy());
  return manage(res);
}

isl::map map::factor_range() const
{
  auto res = isl_map_factor_range(copy());
  return manage(res);
}

int map::find_dim_by_id(isl::dim type, const isl::id &id) const
{
  auto res = isl_map_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int map::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_map_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::map map::fix_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_map_fix_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::map map::fix_val(isl::dim type, unsigned int pos, isl::val v) const
{
  auto res = isl_map_fix_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

isl::map map::fixed_power_val(isl::val exp) const
{
  auto res = isl_map_fixed_power_val(copy(), exp.release());
  return manage(res);
}

isl::map map::flat_domain_product(isl::map map2) const
{
  auto res = isl_map_flat_domain_product(copy(), map2.release());
  return manage(res);
}

isl::map map::flat_product(isl::map map2) const
{
  auto res = isl_map_flat_product(copy(), map2.release());
  return manage(res);
}

isl::map map::flat_range_product(isl::map map2) const
{
  auto res = isl_map_flat_range_product(copy(), map2.release());
  return manage(res);
}

isl::map map::flatten() const
{
  auto res = isl_map_flatten(copy());
  return manage(res);
}

isl::map map::flatten_domain() const
{
  auto res = isl_map_flatten_domain(copy());
  return manage(res);
}

isl::map map::flatten_range() const
{
  auto res = isl_map_flatten_range(copy());
  return manage(res);
}

isl::map map::floordiv_val(isl::val d) const
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

isl::map map::from_aff(isl::aff aff)
{
  auto res = isl_map_from_aff(aff.release());
  return manage(res);
}

isl::map map::from_domain(isl::set set)
{
  auto res = isl_map_from_domain(set.release());
  return manage(res);
}

isl::map map::from_domain_and_range(isl::set domain, isl::set range)
{
  auto res = isl_map_from_domain_and_range(domain.release(), range.release());
  return manage(res);
}

isl::map map::from_multi_aff(isl::multi_aff maff)
{
  auto res = isl_map_from_multi_aff(maff.release());
  return manage(res);
}

isl::map map::from_multi_pw_aff(isl::multi_pw_aff mpa)
{
  auto res = isl_map_from_multi_pw_aff(mpa.release());
  return manage(res);
}

isl::map map::from_pw_aff(isl::pw_aff pwaff)
{
  auto res = isl_map_from_pw_aff(pwaff.release());
  return manage(res);
}

isl::map map::from_pw_multi_aff(isl::pw_multi_aff pma)
{
  auto res = isl_map_from_pw_multi_aff(pma.release());
  return manage(res);
}

isl::map map::from_range(isl::set set)
{
  auto res = isl_map_from_range(set.release());
  return manage(res);
}

isl::map map::from_union_map(isl::union_map umap)
{
  auto res = isl_map_from_union_map(umap.release());
  return manage(res);
}

isl::basic_map_list map::get_basic_map_list() const
{
  auto res = isl_map_get_basic_map_list(get());
  return manage(res);
}

isl::id map::get_dim_id(isl::dim type, unsigned int pos) const
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

isl::fixed_box map::get_range_simple_fixed_box_hull() const
{
  auto res = isl_map_get_range_simple_fixed_box_hull(get());
  return manage(res);
}

isl::space map::get_space() const
{
  auto res = isl_map_get_space(get());
  return manage(res);
}

isl::id map::get_tuple_id(isl::dim type) const
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

isl::map map::gist(isl::map context) const
{
  auto res = isl_map_gist(copy(), context.release());
  return manage(res);
}

isl::map map::gist_basic_map(isl::basic_map context) const
{
  auto res = isl_map_gist_basic_map(copy(), context.release());
  return manage(res);
}

isl::map map::gist_domain(isl::set context) const
{
  auto res = isl_map_gist_domain(copy(), context.release());
  return manage(res);
}

isl::map map::gist_params(isl::set context) const
{
  auto res = isl_map_gist_params(copy(), context.release());
  return manage(res);
}

isl::map map::gist_range(isl::set context) const
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

boolean map::has_equal_space(const isl::map &map2) const
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

isl::map map::identity(isl::space space)
{
  auto res = isl_map_identity(space.release());
  return manage(res);
}

isl::map map::insert_dims(isl::dim type, unsigned int pos, unsigned int n) const
{
  auto res = isl_map_insert_dims(copy(), static_cast<enum isl_dim_type>(type), pos, n);
  return manage(res);
}

isl::map map::intersect(isl::map map2) const
{
  auto res = isl_map_intersect(copy(), map2.release());
  return manage(res);
}

isl::map map::intersect_domain(isl::set set) const
{
  auto res = isl_map_intersect_domain(copy(), set.release());
  return manage(res);
}

isl::map map::intersect_domain_factor_domain(isl::map factor) const
{
  auto res = isl_map_intersect_domain_factor_domain(copy(), factor.release());
  return manage(res);
}

isl::map map::intersect_domain_factor_range(isl::map factor) const
{
  auto res = isl_map_intersect_domain_factor_range(copy(), factor.release());
  return manage(res);
}

isl::map map::intersect_params(isl::set params) const
{
  auto res = isl_map_intersect_params(copy(), params.release());
  return manage(res);
}

isl::map map::intersect_range(isl::set set) const
{
  auto res = isl_map_intersect_range(copy(), set.release());
  return manage(res);
}

isl::map map::intersect_range_factor_domain(isl::map factor) const
{
  auto res = isl_map_intersect_range_factor_domain(copy(), factor.release());
  return manage(res);
}

isl::map map::intersect_range_factor_range(isl::map factor) const
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

boolean map::is_disjoint(const isl::map &map2) const
{
  auto res = isl_map_is_disjoint(get(), map2.get());
  return manage(res);
}

boolean map::is_empty() const
{
  auto res = isl_map_is_empty(get());
  return manage(res);
}

boolean map::is_equal(const isl::map &map2) const
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

boolean map::is_strict_subset(const isl::map &map2) const
{
  auto res = isl_map_is_strict_subset(get(), map2.get());
  return manage(res);
}

boolean map::is_subset(const isl::map &map2) const
{
  auto res = isl_map_is_subset(get(), map2.get());
  return manage(res);
}

int map::is_translation() const
{
  auto res = isl_map_is_translation(get());
  return res;
}

isl::map map::lex_ge(isl::space set_space)
{
  auto res = isl_map_lex_ge(set_space.release());
  return manage(res);
}

isl::map map::lex_ge_at(isl::multi_pw_aff mpa) const
{
  auto res = isl_map_lex_ge_at_multi_pw_aff(copy(), mpa.release());
  return manage(res);
}

isl::map map::lex_ge_first(isl::space space, unsigned int n)
{
  auto res = isl_map_lex_ge_first(space.release(), n);
  return manage(res);
}

isl::map map::lex_ge_map(isl::map map2) const
{
  auto res = isl_map_lex_ge_map(copy(), map2.release());
  return manage(res);
}

isl::map map::lex_gt(isl::space set_space)
{
  auto res = isl_map_lex_gt(set_space.release());
  return manage(res);
}

isl::map map::lex_gt_at(isl::multi_pw_aff mpa) const
{
  auto res = isl_map_lex_gt_at_multi_pw_aff(copy(), mpa.release());
  return manage(res);
}

isl::map map::lex_gt_first(isl::space space, unsigned int n)
{
  auto res = isl_map_lex_gt_first(space.release(), n);
  return manage(res);
}

isl::map map::lex_gt_map(isl::map map2) const
{
  auto res = isl_map_lex_gt_map(copy(), map2.release());
  return manage(res);
}

isl::map map::lex_le(isl::space set_space)
{
  auto res = isl_map_lex_le(set_space.release());
  return manage(res);
}

isl::map map::lex_le_at(isl::multi_pw_aff mpa) const
{
  auto res = isl_map_lex_le_at_multi_pw_aff(copy(), mpa.release());
  return manage(res);
}

isl::map map::lex_le_first(isl::space space, unsigned int n)
{
  auto res = isl_map_lex_le_first(space.release(), n);
  return manage(res);
}

isl::map map::lex_le_map(isl::map map2) const
{
  auto res = isl_map_lex_le_map(copy(), map2.release());
  return manage(res);
}

isl::map map::lex_lt(isl::space set_space)
{
  auto res = isl_map_lex_lt(set_space.release());
  return manage(res);
}

isl::map map::lex_lt_at(isl::multi_pw_aff mpa) const
{
  auto res = isl_map_lex_lt_at_multi_pw_aff(copy(), mpa.release());
  return manage(res);
}

isl::map map::lex_lt_first(isl::space space, unsigned int n)
{
  auto res = isl_map_lex_lt_first(space.release(), n);
  return manage(res);
}

isl::map map::lex_lt_map(isl::map map2) const
{
  auto res = isl_map_lex_lt_map(copy(), map2.release());
  return manage(res);
}

isl::map map::lexmax() const
{
  auto res = isl_map_lexmax(copy());
  return manage(res);
}

isl::pw_multi_aff map::lexmax_pw_multi_aff() const
{
  auto res = isl_map_lexmax_pw_multi_aff(copy());
  return manage(res);
}

isl::map map::lexmin() const
{
  auto res = isl_map_lexmin(copy());
  return manage(res);
}

isl::pw_multi_aff map::lexmin_pw_multi_aff() const
{
  auto res = isl_map_lexmin_pw_multi_aff(copy());
  return manage(res);
}

isl::map map::lower_bound(isl::multi_pw_aff lower) const
{
  auto res = isl_map_lower_bound_multi_pw_aff(copy(), lower.release());
  return manage(res);
}

isl::map map::lower_bound_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_map_lower_bound_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::map map::lower_bound_val(isl::dim type, unsigned int pos, isl::val value) const
{
  auto res = isl_map_lower_bound_val(copy(), static_cast<enum isl_dim_type>(type), pos, value.release());
  return manage(res);
}

isl::multi_pw_aff map::max_multi_pw_aff() const
{
  auto res = isl_map_max_multi_pw_aff(copy());
  return manage(res);
}

isl::multi_pw_aff map::min_multi_pw_aff() const
{
  auto res = isl_map_min_multi_pw_aff(copy());
  return manage(res);
}

isl::map map::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_map_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl_size map::n_basic_map() const
{
  auto res = isl_map_n_basic_map(get());
  return res;
}

isl::map map::nat_universe(isl::space space)
{
  auto res = isl_map_nat_universe(space.release());
  return manage(res);
}

isl::map map::neg() const
{
  auto res = isl_map_neg(copy());
  return manage(res);
}

isl::map map::oppose(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_map_oppose(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

isl::map map::order_ge(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_map_order_ge(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

isl::map map::order_gt(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_map_order_gt(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

isl::map map::order_le(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_map_order_le(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

isl::map map::order_lt(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_map_order_lt(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

isl::set map::params() const
{
  auto res = isl_map_params(copy());
  return manage(res);
}

isl::val map::plain_get_val_if_fixed(isl::dim type, unsigned int pos) const
{
  auto res = isl_map_plain_get_val_if_fixed(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

boolean map::plain_is_empty() const
{
  auto res = isl_map_plain_is_empty(get());
  return manage(res);
}

boolean map::plain_is_equal(const isl::map &map2) const
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

isl::basic_map map::plain_unshifted_simple_hull() const
{
  auto res = isl_map_plain_unshifted_simple_hull(copy());
  return manage(res);
}

isl::basic_map map::polyhedral_hull() const
{
  auto res = isl_map_polyhedral_hull(copy());
  return manage(res);
}

isl::map map::preimage_domain(isl::multi_aff ma) const
{
  auto res = isl_map_preimage_domain_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::map map::preimage_domain(isl::multi_pw_aff mpa) const
{
  auto res = isl_map_preimage_domain_multi_pw_aff(copy(), mpa.release());
  return manage(res);
}

isl::map map::preimage_domain(isl::pw_multi_aff pma) const
{
  auto res = isl_map_preimage_domain_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::map map::preimage_range(isl::multi_aff ma) const
{
  auto res = isl_map_preimage_range_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::map map::preimage_range(isl::pw_multi_aff pma) const
{
  auto res = isl_map_preimage_range_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::map map::product(isl::map map2) const
{
  auto res = isl_map_product(copy(), map2.release());
  return manage(res);
}

isl::map map::project_out(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_map_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::map map::project_out_all_params() const
{
  auto res = isl_map_project_out_all_params(copy());
  return manage(res);
}

isl::set map::range() const
{
  auto res = isl_map_range(copy());
  return manage(res);
}

isl::map map::range_curry() const
{
  auto res = isl_map_range_curry(copy());
  return manage(res);
}

isl::map map::range_factor_domain() const
{
  auto res = isl_map_range_factor_domain(copy());
  return manage(res);
}

isl::map map::range_factor_range() const
{
  auto res = isl_map_range_factor_range(copy());
  return manage(res);
}

boolean map::range_is_wrapping() const
{
  auto res = isl_map_range_is_wrapping(get());
  return manage(res);
}

isl::map map::range_map() const
{
  auto res = isl_map_range_map(copy());
  return manage(res);
}

isl::map map::range_product(isl::map map2) const
{
  auto res = isl_map_range_product(copy(), map2.release());
  return manage(res);
}

isl::map map::range_reverse() const
{
  auto res = isl_map_range_reverse(copy());
  return manage(res);
}

isl_size map::range_tuple_dim() const
{
  auto res = isl_map_range_tuple_dim(get());
  return res;
}

isl::map map::remove_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_map_remove_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::map map::remove_divs() const
{
  auto res = isl_map_remove_divs(copy());
  return manage(res);
}

isl::map map::remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_map_remove_divs_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::map map::remove_redundancies() const
{
  auto res = isl_map_remove_redundancies(copy());
  return manage(res);
}

isl::map map::remove_unknown_divs() const
{
  auto res = isl_map_remove_unknown_divs(copy());
  return manage(res);
}

isl::map map::reset_tuple_id(isl::dim type) const
{
  auto res = isl_map_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::map map::reset_user() const
{
  auto res = isl_map_reset_user(copy());
  return manage(res);
}

isl::map map::reverse() const
{
  auto res = isl_map_reverse(copy());
  return manage(res);
}

isl::basic_map map::sample() const
{
  auto res = isl_map_sample(copy());
  return manage(res);
}

isl::map map::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const
{
  auto res = isl_map_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::map map::set_tuple_id(isl::dim type, isl::id id) const
{
  auto res = isl_map_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::map map::set_tuple_name(isl::dim type, const std::string &s) const
{
  auto res = isl_map_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

isl::basic_map map::simple_hull() const
{
  auto res = isl_map_simple_hull(copy());
  return manage(res);
}

isl::map map::subtract(isl::map map2) const
{
  auto res = isl_map_subtract(copy(), map2.release());
  return manage(res);
}

isl::map map::subtract_domain(isl::set dom) const
{
  auto res = isl_map_subtract_domain(copy(), dom.release());
  return manage(res);
}

isl::map map::subtract_range(isl::set dom) const
{
  auto res = isl_map_subtract_range(copy(), dom.release());
  return manage(res);
}

isl::map map::sum(isl::map map2) const
{
  auto res = isl_map_sum(copy(), map2.release());
  return manage(res);
}

isl::map map::uncurry() const
{
  auto res = isl_map_uncurry(copy());
  return manage(res);
}

isl::map map::unite(isl::map map2) const
{
  auto res = isl_map_union(copy(), map2.release());
  return manage(res);
}

isl::map map::universe(isl::space space)
{
  auto res = isl_map_universe(space.release());
  return manage(res);
}

isl::basic_map map::unshifted_simple_hull() const
{
  auto res = isl_map_unshifted_simple_hull(copy());
  return manage(res);
}

isl::basic_map map::unshifted_simple_hull_from_map_list(isl::map_list list) const
{
  auto res = isl_map_unshifted_simple_hull_from_map_list(copy(), list.release());
  return manage(res);
}

isl::map map::upper_bound(isl::multi_pw_aff upper) const
{
  auto res = isl_map_upper_bound_multi_pw_aff(copy(), upper.release());
  return manage(res);
}

isl::map map::upper_bound_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_map_upper_bound_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::map map::upper_bound_val(isl::dim type, unsigned int pos, isl::val value) const
{
  auto res = isl_map_upper_bound_val(copy(), static_cast<enum isl_dim_type>(type), pos, value.release());
  return manage(res);
}

isl::set map::wrap() const
{
  auto res = isl_map_wrap(copy());
  return manage(res);
}

isl::map map::zip() const
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


isl::ctx map_list::ctx() const {
  return isl::ctx(isl_map_list_get_ctx(ptr));
}

void map_list::dump() const {
  isl_map_list_dump(get());
}


isl::map_list map_list::add(isl::map el) const
{
  auto res = isl_map_list_add(copy(), el.release());
  return manage(res);
}

isl::map_list map_list::alloc(isl::ctx ctx, int n)
{
  auto res = isl_map_list_alloc(ctx.release(), n);
  return manage(res);
}

isl::map_list map_list::clear() const
{
  auto res = isl_map_list_clear(copy());
  return manage(res);
}

isl::map_list map_list::concat(isl::map_list list2) const
{
  auto res = isl_map_list_concat(copy(), list2.release());
  return manage(res);
}

isl::map_list map_list::drop(unsigned int first, unsigned int n) const
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

isl::map_list map_list::from_map(isl::map el)
{
  auto res = isl_map_list_from_map(el.release());
  return manage(res);
}

isl::map map_list::get_at(int index) const
{
  auto res = isl_map_list_get_at(get(), index);
  return manage(res);
}

isl::map map_list::get_map(int index) const
{
  auto res = isl_map_list_get_map(get(), index);
  return manage(res);
}

isl::map_list map_list::insert(unsigned int pos, isl::map el) const
{
  auto res = isl_map_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl_size map_list::n_map() const
{
  auto res = isl_map_list_n_map(get());
  return res;
}

isl::map_list map_list::reverse() const
{
  auto res = isl_map_list_reverse(copy());
  return manage(res);
}

isl::map_list map_list::set_map(int index, isl::map el) const
{
  auto res = isl_map_list_set_map(copy(), index, el.release());
  return manage(res);
}

isl_size map_list::size() const
{
  auto res = isl_map_list_size(get());
  return res;
}

isl::map_list map_list::swap(unsigned int pos1, unsigned int pos2) const
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


isl::ctx mat::ctx() const {
  return isl::ctx(isl_mat_get_ctx(ptr));
}

void mat::dump() const {
  isl_mat_dump(get());
}


isl::mat mat::add_rows(unsigned int n) const
{
  auto res = isl_mat_add_rows(copy(), n);
  return manage(res);
}

isl::mat mat::add_zero_cols(unsigned int n) const
{
  auto res = isl_mat_add_zero_cols(copy(), n);
  return manage(res);
}

isl::mat mat::add_zero_rows(unsigned int n) const
{
  auto res = isl_mat_add_zero_rows(copy(), n);
  return manage(res);
}

isl::mat mat::aff_direct_sum(isl::mat right) const
{
  auto res = isl_mat_aff_direct_sum(copy(), right.release());
  return manage(res);
}

isl::mat mat::alloc(isl::ctx ctx, unsigned int n_row, unsigned int n_col)
{
  auto res = isl_mat_alloc(ctx.release(), n_row, n_col);
  return manage(res);
}

isl_size mat::cols() const
{
  auto res = isl_mat_cols(get());
  return res;
}

isl::mat mat::concat(isl::mat bot) const
{
  auto res = isl_mat_concat(copy(), bot.release());
  return manage(res);
}

isl::mat mat::diagonal(isl::mat mat2) const
{
  auto res = isl_mat_diagonal(copy(), mat2.release());
  return manage(res);
}

isl::mat mat::drop_cols(unsigned int col, unsigned int n) const
{
  auto res = isl_mat_drop_cols(copy(), col, n);
  return manage(res);
}

isl::mat mat::drop_rows(unsigned int row, unsigned int n) const
{
  auto res = isl_mat_drop_rows(copy(), row, n);
  return manage(res);
}

isl::mat mat::from_row_vec(isl::vec vec)
{
  auto res = isl_mat_from_row_vec(vec.release());
  return manage(res);
}

isl::val mat::get_element_val(int row, int col) const
{
  auto res = isl_mat_get_element_val(get(), row, col);
  return manage(res);
}

boolean mat::has_linearly_independent_rows(const isl::mat &mat2) const
{
  auto res = isl_mat_has_linearly_independent_rows(get(), mat2.get());
  return manage(res);
}

int mat::initial_non_zero_cols() const
{
  auto res = isl_mat_initial_non_zero_cols(get());
  return res;
}

isl::mat mat::insert_cols(unsigned int col, unsigned int n) const
{
  auto res = isl_mat_insert_cols(copy(), col, n);
  return manage(res);
}

isl::mat mat::insert_rows(unsigned int row, unsigned int n) const
{
  auto res = isl_mat_insert_rows(copy(), row, n);
  return manage(res);
}

isl::mat mat::insert_zero_cols(unsigned int first, unsigned int n) const
{
  auto res = isl_mat_insert_zero_cols(copy(), first, n);
  return manage(res);
}

isl::mat mat::insert_zero_rows(unsigned int row, unsigned int n) const
{
  auto res = isl_mat_insert_zero_rows(copy(), row, n);
  return manage(res);
}

isl::mat mat::inverse_product(isl::mat right) const
{
  auto res = isl_mat_inverse_product(copy(), right.release());
  return manage(res);
}

boolean mat::is_equal(const isl::mat &mat2) const
{
  auto res = isl_mat_is_equal(get(), mat2.get());
  return manage(res);
}

isl::mat mat::lin_to_aff() const
{
  auto res = isl_mat_lin_to_aff(copy());
  return manage(res);
}

isl::mat mat::move_cols(unsigned int dst_col, unsigned int src_col, unsigned int n) const
{
  auto res = isl_mat_move_cols(copy(), dst_col, src_col, n);
  return manage(res);
}

isl::mat mat::normalize() const
{
  auto res = isl_mat_normalize(copy());
  return manage(res);
}

isl::mat mat::normalize_row(int row) const
{
  auto res = isl_mat_normalize_row(copy(), row);
  return manage(res);
}

isl::mat mat::product(isl::mat right) const
{
  auto res = isl_mat_product(copy(), right.release());
  return manage(res);
}

isl_size mat::rank() const
{
  auto res = isl_mat_rank(get());
  return res;
}

isl::mat mat::right_inverse() const
{
  auto res = isl_mat_right_inverse(copy());
  return manage(res);
}

isl::mat mat::right_kernel() const
{
  auto res = isl_mat_right_kernel(copy());
  return manage(res);
}

isl::mat mat::row_basis() const
{
  auto res = isl_mat_row_basis(copy());
  return manage(res);
}

isl::mat mat::row_basis_extension(isl::mat mat2) const
{
  auto res = isl_mat_row_basis_extension(copy(), mat2.release());
  return manage(res);
}

isl_size mat::rows() const
{
  auto res = isl_mat_rows(get());
  return res;
}

isl::mat mat::set_element_si(int row, int col, int v) const
{
  auto res = isl_mat_set_element_si(copy(), row, col, v);
  return manage(res);
}

isl::mat mat::set_element_val(int row, int col, isl::val v) const
{
  auto res = isl_mat_set_element_val(copy(), row, col, v.release());
  return manage(res);
}

isl::mat mat::swap_cols(unsigned int i, unsigned int j) const
{
  auto res = isl_mat_swap_cols(copy(), i, j);
  return manage(res);
}

isl::mat mat::swap_rows(unsigned int i, unsigned int j) const
{
  auto res = isl_mat_swap_rows(copy(), i, j);
  return manage(res);
}

isl::mat mat::transpose() const
{
  auto res = isl_mat_transpose(copy());
  return manage(res);
}

isl::mat mat::unimodular_complete(int row) const
{
  auto res = isl_mat_unimodular_complete(copy(), row);
  return manage(res);
}

isl::mat mat::vec_concat(isl::vec bot) const
{
  auto res = isl_mat_vec_concat(copy(), bot.release());
  return manage(res);
}

isl::vec mat::vec_inverse_product(isl::vec vec) const
{
  auto res = isl_mat_vec_inverse_product(copy(), vec.release());
  return manage(res);
}

isl::vec mat::vec_product(isl::vec vec) const
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


multi_aff::multi_aff(__isl_take isl_multi_aff *ptr)
    : ptr(ptr) {}

multi_aff::multi_aff(isl::aff aff)
{
  auto res = isl_multi_aff_from_aff(aff.release());
  ptr = res;
}
multi_aff::multi_aff(isl::space space, isl::aff_list list)
{
  auto res = isl_multi_aff_from_aff_list(space.release(), list.release());
  ptr = res;
}
multi_aff::multi_aff(isl::ctx ctx, const std::string &str)
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


isl::ctx multi_aff::ctx() const {
  return isl::ctx(isl_multi_aff_get_ctx(ptr));
}

void multi_aff::dump() const {
  isl_multi_aff_dump(get());
}


isl::multi_aff multi_aff::add(isl::multi_aff multi2) const
{
  auto res = isl_multi_aff_add(copy(), multi2.release());
  return manage(res);
}

isl::multi_aff multi_aff::add_constant(isl::multi_val mv) const
{
  auto res = isl_multi_aff_add_constant_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_aff multi_aff::add_constant(isl::val v) const
{
  auto res = isl_multi_aff_add_constant_val(copy(), v.release());
  return manage(res);
}

isl::multi_aff multi_aff::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_multi_aff_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::multi_aff multi_aff::align_params(isl::space model) const
{
  auto res = isl_multi_aff_align_params(copy(), model.release());
  return manage(res);
}

isl::basic_set multi_aff::bind(isl::multi_id tuple) const
{
  auto res = isl_multi_aff_bind(copy(), tuple.release());
  return manage(res);
}

isl::multi_aff multi_aff::bind_domain(isl::multi_id tuple) const
{
  auto res = isl_multi_aff_bind_domain(copy(), tuple.release());
  return manage(res);
}

isl::multi_aff multi_aff::bind_domain_wrapped_domain(isl::multi_id tuple) const
{
  auto res = isl_multi_aff_bind_domain_wrapped_domain(copy(), tuple.release());
  return manage(res);
}

isl_size multi_aff::dim(isl::dim type) const
{
  auto res = isl_multi_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::multi_aff multi_aff::domain_map(isl::space space)
{
  auto res = isl_multi_aff_domain_map(space.release());
  return manage(res);
}

isl::multi_aff multi_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_multi_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::multi_aff multi_aff::factor_range() const
{
  auto res = isl_multi_aff_factor_range(copy());
  return manage(res);
}

int multi_aff::find_dim_by_id(isl::dim type, const isl::id &id) const
{
  auto res = isl_multi_aff_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int multi_aff::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_multi_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::multi_aff multi_aff::flat_range_product(isl::multi_aff multi2) const
{
  auto res = isl_multi_aff_flat_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_aff multi_aff::flatten_domain() const
{
  auto res = isl_multi_aff_flatten_domain(copy());
  return manage(res);
}

isl::multi_aff multi_aff::flatten_range() const
{
  auto res = isl_multi_aff_flatten_range(copy());
  return manage(res);
}

isl::multi_aff multi_aff::floor() const
{
  auto res = isl_multi_aff_floor(copy());
  return manage(res);
}

isl::multi_aff multi_aff::from_range() const
{
  auto res = isl_multi_aff_from_range(copy());
  return manage(res);
}

isl::aff multi_aff::get_aff(int pos) const
{
  auto res = isl_multi_aff_get_aff(get(), pos);
  return manage(res);
}

isl::aff multi_aff::get_at(int pos) const
{
  auto res = isl_multi_aff_get_at(get(), pos);
  return manage(res);
}

isl::multi_val multi_aff::get_constant_multi_val() const
{
  auto res = isl_multi_aff_get_constant_multi_val(get());
  return manage(res);
}

isl::id multi_aff::get_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_multi_aff_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::space multi_aff::get_domain_space() const
{
  auto res = isl_multi_aff_get_domain_space(get());
  return manage(res);
}

isl::aff_list multi_aff::get_list() const
{
  auto res = isl_multi_aff_get_list(get());
  return manage(res);
}

isl::space multi_aff::get_space() const
{
  auto res = isl_multi_aff_get_space(get());
  return manage(res);
}

isl::id multi_aff::get_tuple_id(isl::dim type) const
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

isl::multi_aff multi_aff::gist(isl::set context) const
{
  auto res = isl_multi_aff_gist(copy(), context.release());
  return manage(res);
}

isl::multi_aff multi_aff::gist_params(isl::set context) const
{
  auto res = isl_multi_aff_gist_params(copy(), context.release());
  return manage(res);
}

boolean multi_aff::has_tuple_id(isl::dim type) const
{
  auto res = isl_multi_aff_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::multi_aff multi_aff::identity(isl::space space)
{
  auto res = isl_multi_aff_identity(space.release());
  return manage(res);
}

isl::multi_aff multi_aff::identity() const
{
  auto res = isl_multi_aff_identity_multi_aff(copy());
  return manage(res);
}

isl::multi_aff multi_aff::identity_on_domain(isl::space space)
{
  auto res = isl_multi_aff_identity_on_domain_space(space.release());
  return manage(res);
}

isl::multi_aff multi_aff::insert_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_multi_aff_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::multi_aff multi_aff::insert_domain(isl::space domain) const
{
  auto res = isl_multi_aff_insert_domain(copy(), domain.release());
  return manage(res);
}

boolean multi_aff::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_multi_aff_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean multi_aff::involves_locals() const
{
  auto res = isl_multi_aff_involves_locals(get());
  return manage(res);
}

boolean multi_aff::involves_nan() const
{
  auto res = isl_multi_aff_involves_nan(get());
  return manage(res);
}

isl::set multi_aff::lex_ge_set(isl::multi_aff ma2) const
{
  auto res = isl_multi_aff_lex_ge_set(copy(), ma2.release());
  return manage(res);
}

isl::set multi_aff::lex_gt_set(isl::multi_aff ma2) const
{
  auto res = isl_multi_aff_lex_gt_set(copy(), ma2.release());
  return manage(res);
}

isl::set multi_aff::lex_le_set(isl::multi_aff ma2) const
{
  auto res = isl_multi_aff_lex_le_set(copy(), ma2.release());
  return manage(res);
}

isl::set multi_aff::lex_lt_set(isl::multi_aff ma2) const
{
  auto res = isl_multi_aff_lex_lt_set(copy(), ma2.release());
  return manage(res);
}

isl::multi_aff multi_aff::mod_multi_val(isl::multi_val mv) const
{
  auto res = isl_multi_aff_mod_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_aff multi_aff::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_multi_aff_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl::multi_aff multi_aff::multi_val_on_space(isl::space space, isl::multi_val mv)
{
  auto res = isl_multi_aff_multi_val_on_space(space.release(), mv.release());
  return manage(res);
}

isl::multi_aff multi_aff::neg() const
{
  auto res = isl_multi_aff_neg(copy());
  return manage(res);
}

int multi_aff::plain_cmp(const isl::multi_aff &multi2) const
{
  auto res = isl_multi_aff_plain_cmp(get(), multi2.get());
  return res;
}

boolean multi_aff::plain_is_equal(const isl::multi_aff &multi2) const
{
  auto res = isl_multi_aff_plain_is_equal(get(), multi2.get());
  return manage(res);
}

isl::multi_aff multi_aff::product(isl::multi_aff multi2) const
{
  auto res = isl_multi_aff_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_aff multi_aff::project_domain_on_params() const
{
  auto res = isl_multi_aff_project_domain_on_params(copy());
  return manage(res);
}

isl::multi_aff multi_aff::project_out_map(isl::space space, isl::dim type, unsigned int first, unsigned int n)
{
  auto res = isl_multi_aff_project_out_map(space.release(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::multi_aff multi_aff::pullback(isl::multi_aff ma2) const
{
  auto res = isl_multi_aff_pullback_multi_aff(copy(), ma2.release());
  return manage(res);
}

isl::multi_aff multi_aff::range_factor_domain() const
{
  auto res = isl_multi_aff_range_factor_domain(copy());
  return manage(res);
}

isl::multi_aff multi_aff::range_factor_range() const
{
  auto res = isl_multi_aff_range_factor_range(copy());
  return manage(res);
}

boolean multi_aff::range_is_wrapping() const
{
  auto res = isl_multi_aff_range_is_wrapping(get());
  return manage(res);
}

isl::multi_aff multi_aff::range_map(isl::space space)
{
  auto res = isl_multi_aff_range_map(space.release());
  return manage(res);
}

isl::multi_aff multi_aff::range_product(isl::multi_aff multi2) const
{
  auto res = isl_multi_aff_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_aff multi_aff::range_splice(unsigned int pos, isl::multi_aff multi2) const
{
  auto res = isl_multi_aff_range_splice(copy(), pos, multi2.release());
  return manage(res);
}

isl::multi_aff multi_aff::reset_tuple_id(isl::dim type) const
{
  auto res = isl_multi_aff_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::multi_aff multi_aff::reset_user() const
{
  auto res = isl_multi_aff_reset_user(copy());
  return manage(res);
}

isl::multi_aff multi_aff::scale(isl::multi_val mv) const
{
  auto res = isl_multi_aff_scale_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_aff multi_aff::scale(isl::val v) const
{
  auto res = isl_multi_aff_scale_val(copy(), v.release());
  return manage(res);
}

isl::multi_aff multi_aff::scale_down(isl::multi_val mv) const
{
  auto res = isl_multi_aff_scale_down_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_aff multi_aff::scale_down(isl::val v) const
{
  auto res = isl_multi_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::multi_aff multi_aff::set_aff(int pos, isl::aff el) const
{
  auto res = isl_multi_aff_set_aff(copy(), pos, el.release());
  return manage(res);
}

isl::multi_aff multi_aff::set_at(int pos, isl::aff el) const
{
  auto res = isl_multi_aff_set_at(copy(), pos, el.release());
  return manage(res);
}

isl::multi_aff multi_aff::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const
{
  auto res = isl_multi_aff_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::multi_aff multi_aff::set_tuple_id(isl::dim type, isl::id id) const
{
  auto res = isl_multi_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::multi_aff multi_aff::set_tuple_name(isl::dim type, const std::string &s) const
{
  auto res = isl_multi_aff_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

isl_size multi_aff::size() const
{
  auto res = isl_multi_aff_size(get());
  return res;
}

isl::multi_aff multi_aff::splice(unsigned int in_pos, unsigned int out_pos, isl::multi_aff multi2) const
{
  auto res = isl_multi_aff_splice(copy(), in_pos, out_pos, multi2.release());
  return manage(res);
}

isl::multi_aff multi_aff::sub(isl::multi_aff multi2) const
{
  auto res = isl_multi_aff_sub(copy(), multi2.release());
  return manage(res);
}

isl::multi_aff multi_aff::unbind_params_insert_domain(isl::multi_id domain) const
{
  auto res = isl_multi_aff_unbind_params_insert_domain(copy(), domain.release());
  return manage(res);
}

isl::multi_aff multi_aff::zero(isl::space space)
{
  auto res = isl_multi_aff_zero(space.release());
  return manage(res);
}

// implementations for isl::multi_id
multi_id manage(__isl_take isl_multi_id *ptr) {
  return multi_id(ptr);
}
multi_id manage_copy(__isl_keep isl_multi_id *ptr) {
  ptr = isl_multi_id_copy(ptr);
  return multi_id(ptr);
}

multi_id::multi_id()
    : ptr(nullptr) {}

multi_id::multi_id(const multi_id &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}


multi_id::multi_id(__isl_take isl_multi_id *ptr)
    : ptr(ptr) {}

multi_id::multi_id(isl::space space, isl::id_list list)
{
  auto res = isl_multi_id_from_id_list(space.release(), list.release());
  ptr = res;
}
multi_id::multi_id(isl::ctx ctx, const std::string &str)
{
  auto res = isl_multi_id_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

multi_id &multi_id::operator=(multi_id obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

multi_id::~multi_id() {
  if (ptr)
    isl_multi_id_free(ptr);
}

__isl_give isl_multi_id *multi_id::copy() const & {
  return isl_multi_id_copy(ptr);
}

__isl_keep isl_multi_id *multi_id::get() const {
  return ptr;
}

__isl_give isl_multi_id *multi_id::release() {
  isl_multi_id *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool multi_id::is_null() const {
  return ptr == nullptr;
}


isl::ctx multi_id::ctx() const {
  return isl::ctx(isl_multi_id_get_ctx(ptr));
}

void multi_id::dump() const {
  isl_multi_id_dump(get());
}


isl::multi_id multi_id::align_params(isl::space model) const
{
  auto res = isl_multi_id_align_params(copy(), model.release());
  return manage(res);
}

isl::multi_id multi_id::factor_range() const
{
  auto res = isl_multi_id_factor_range(copy());
  return manage(res);
}

isl::multi_id multi_id::flat_range_product(isl::multi_id multi2) const
{
  auto res = isl_multi_id_flat_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_id multi_id::flatten_range() const
{
  auto res = isl_multi_id_flatten_range(copy());
  return manage(res);
}

isl::multi_id multi_id::from_range() const
{
  auto res = isl_multi_id_from_range(copy());
  return manage(res);
}

isl::id multi_id::get_at(int pos) const
{
  auto res = isl_multi_id_get_at(get(), pos);
  return manage(res);
}

isl::space multi_id::get_domain_space() const
{
  auto res = isl_multi_id_get_domain_space(get());
  return manage(res);
}

isl::id multi_id::get_id(int pos) const
{
  auto res = isl_multi_id_get_id(get(), pos);
  return manage(res);
}

isl::id_list multi_id::get_list() const
{
  auto res = isl_multi_id_get_list(get());
  return manage(res);
}

isl::space multi_id::get_space() const
{
  auto res = isl_multi_id_get_space(get());
  return manage(res);
}

boolean multi_id::plain_is_equal(const isl::multi_id &multi2) const
{
  auto res = isl_multi_id_plain_is_equal(get(), multi2.get());
  return manage(res);
}

isl::multi_id multi_id::range_factor_domain() const
{
  auto res = isl_multi_id_range_factor_domain(copy());
  return manage(res);
}

isl::multi_id multi_id::range_factor_range() const
{
  auto res = isl_multi_id_range_factor_range(copy());
  return manage(res);
}

boolean multi_id::range_is_wrapping() const
{
  auto res = isl_multi_id_range_is_wrapping(get());
  return manage(res);
}

isl::multi_id multi_id::range_product(isl::multi_id multi2) const
{
  auto res = isl_multi_id_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_id multi_id::range_splice(unsigned int pos, isl::multi_id multi2) const
{
  auto res = isl_multi_id_range_splice(copy(), pos, multi2.release());
  return manage(res);
}

isl::multi_id multi_id::reset_user() const
{
  auto res = isl_multi_id_reset_user(copy());
  return manage(res);
}

isl::multi_id multi_id::set_at(int pos, isl::id el) const
{
  auto res = isl_multi_id_set_at(copy(), pos, el.release());
  return manage(res);
}

isl::multi_id multi_id::set_id(int pos, isl::id el) const
{
  auto res = isl_multi_id_set_id(copy(), pos, el.release());
  return manage(res);
}

isl_size multi_id::size() const
{
  auto res = isl_multi_id_size(get());
  return res;
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


multi_pw_aff::multi_pw_aff(__isl_take isl_multi_pw_aff *ptr)
    : ptr(ptr) {}

multi_pw_aff::multi_pw_aff(isl::aff aff)
{
  auto res = isl_multi_pw_aff_from_aff(aff.release());
  ptr = res;
}
multi_pw_aff::multi_pw_aff(isl::multi_aff ma)
{
  auto res = isl_multi_pw_aff_from_multi_aff(ma.release());
  ptr = res;
}
multi_pw_aff::multi_pw_aff(isl::pw_aff pa)
{
  auto res = isl_multi_pw_aff_from_pw_aff(pa.release());
  ptr = res;
}
multi_pw_aff::multi_pw_aff(isl::space space, isl::pw_aff_list list)
{
  auto res = isl_multi_pw_aff_from_pw_aff_list(space.release(), list.release());
  ptr = res;
}
multi_pw_aff::multi_pw_aff(isl::pw_multi_aff pma)
{
  auto res = isl_multi_pw_aff_from_pw_multi_aff(pma.release());
  ptr = res;
}
multi_pw_aff::multi_pw_aff(isl::ctx ctx, const std::string &str)
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


isl::ctx multi_pw_aff::ctx() const {
  return isl::ctx(isl_multi_pw_aff_get_ctx(ptr));
}

void multi_pw_aff::dump() const {
  isl_multi_pw_aff_dump(get());
}


isl::multi_pw_aff multi_pw_aff::add(isl::multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_add(copy(), multi2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::add_constant(isl::multi_val mv) const
{
  auto res = isl_multi_pw_aff_add_constant_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::add_constant(isl::val v) const
{
  auto res = isl_multi_pw_aff_add_constant_val(copy(), v.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_multi_pw_aff_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::align_params(isl::space model) const
{
  auto res = isl_multi_pw_aff_align_params(copy(), model.release());
  return manage(res);
}

isl::set multi_pw_aff::bind(isl::multi_id tuple) const
{
  auto res = isl_multi_pw_aff_bind(copy(), tuple.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::bind_domain(isl::multi_id tuple) const
{
  auto res = isl_multi_pw_aff_bind_domain(copy(), tuple.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::bind_domain_wrapped_domain(isl::multi_id tuple) const
{
  auto res = isl_multi_pw_aff_bind_domain_wrapped_domain(copy(), tuple.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::coalesce() const
{
  auto res = isl_multi_pw_aff_coalesce(copy());
  return manage(res);
}

isl_size multi_pw_aff::dim(isl::dim type) const
{
  auto res = isl_multi_pw_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::set multi_pw_aff::domain() const
{
  auto res = isl_multi_pw_aff_domain(copy());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_multi_pw_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::map multi_pw_aff::eq_map(isl::multi_pw_aff mpa2) const
{
  auto res = isl_multi_pw_aff_eq_map(copy(), mpa2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::factor_range() const
{
  auto res = isl_multi_pw_aff_factor_range(copy());
  return manage(res);
}

int multi_pw_aff::find_dim_by_id(isl::dim type, const isl::id &id) const
{
  auto res = isl_multi_pw_aff_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int multi_pw_aff::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_multi_pw_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::multi_pw_aff multi_pw_aff::flat_range_product(isl::multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_flat_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::flatten_range() const
{
  auto res = isl_multi_pw_aff_flatten_range(copy());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::from_range() const
{
  auto res = isl_multi_pw_aff_from_range(copy());
  return manage(res);
}

isl::pw_aff multi_pw_aff::get_at(int pos) const
{
  auto res = isl_multi_pw_aff_get_at(get(), pos);
  return manage(res);
}

isl::id multi_pw_aff::get_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_multi_pw_aff_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::space multi_pw_aff::get_domain_space() const
{
  auto res = isl_multi_pw_aff_get_domain_space(get());
  return manage(res);
}

uint32_t multi_pw_aff::get_hash() const
{
  auto res = isl_multi_pw_aff_get_hash(get());
  return res;
}

isl::pw_aff_list multi_pw_aff::get_list() const
{
  auto res = isl_multi_pw_aff_get_list(get());
  return manage(res);
}

isl::pw_aff multi_pw_aff::get_pw_aff(int pos) const
{
  auto res = isl_multi_pw_aff_get_pw_aff(get(), pos);
  return manage(res);
}

isl::space multi_pw_aff::get_space() const
{
  auto res = isl_multi_pw_aff_get_space(get());
  return manage(res);
}

isl::id multi_pw_aff::get_tuple_id(isl::dim type) const
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

isl::multi_pw_aff multi_pw_aff::gist(isl::set set) const
{
  auto res = isl_multi_pw_aff_gist(copy(), set.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::gist_params(isl::set set) const
{
  auto res = isl_multi_pw_aff_gist_params(copy(), set.release());
  return manage(res);
}

boolean multi_pw_aff::has_tuple_id(isl::dim type) const
{
  auto res = isl_multi_pw_aff_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::identity(isl::space space)
{
  auto res = isl_multi_pw_aff_identity(space.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::identity() const
{
  auto res = isl_multi_pw_aff_identity_multi_pw_aff(copy());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::identity_on_domain(isl::space space)
{
  auto res = isl_multi_pw_aff_identity_on_domain_space(space.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::insert_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_multi_pw_aff_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::insert_domain(isl::space domain) const
{
  auto res = isl_multi_pw_aff_insert_domain(copy(), domain.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::intersect_domain(isl::set domain) const
{
  auto res = isl_multi_pw_aff_intersect_domain(copy(), domain.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::intersect_params(isl::set set) const
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

boolean multi_pw_aff::involves_param(const isl::id &id) const
{
  auto res = isl_multi_pw_aff_involves_param_id(get(), id.get());
  return manage(res);
}

boolean multi_pw_aff::involves_param(const isl::id_list &list) const
{
  auto res = isl_multi_pw_aff_involves_param_id_list(get(), list.get());
  return manage(res);
}

boolean multi_pw_aff::is_cst() const
{
  auto res = isl_multi_pw_aff_is_cst(get());
  return manage(res);
}

boolean multi_pw_aff::is_equal(const isl::multi_pw_aff &mpa2) const
{
  auto res = isl_multi_pw_aff_is_equal(get(), mpa2.get());
  return manage(res);
}

isl::map multi_pw_aff::lex_ge_map(isl::multi_pw_aff mpa2) const
{
  auto res = isl_multi_pw_aff_lex_ge_map(copy(), mpa2.release());
  return manage(res);
}

isl::map multi_pw_aff::lex_gt_map(isl::multi_pw_aff mpa2) const
{
  auto res = isl_multi_pw_aff_lex_gt_map(copy(), mpa2.release());
  return manage(res);
}

isl::map multi_pw_aff::lex_le_map(isl::multi_pw_aff mpa2) const
{
  auto res = isl_multi_pw_aff_lex_le_map(copy(), mpa2.release());
  return manage(res);
}

isl::map multi_pw_aff::lex_lt_map(isl::multi_pw_aff mpa2) const
{
  auto res = isl_multi_pw_aff_lex_lt_map(copy(), mpa2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::max(isl::multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_max(copy(), multi2.release());
  return manage(res);
}

isl::multi_val multi_pw_aff::max_multi_val() const
{
  auto res = isl_multi_pw_aff_max_multi_val(copy());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::min(isl::multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_min(copy(), multi2.release());
  return manage(res);
}

isl::multi_val multi_pw_aff::min_multi_val() const
{
  auto res = isl_multi_pw_aff_min_multi_val(copy());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::mod_multi_val(isl::multi_val mv) const
{
  auto res = isl_multi_pw_aff_mod_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_multi_pw_aff_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::neg() const
{
  auto res = isl_multi_pw_aff_neg(copy());
  return manage(res);
}

boolean multi_pw_aff::plain_is_equal(const isl::multi_pw_aff &multi2) const
{
  auto res = isl_multi_pw_aff_plain_is_equal(get(), multi2.get());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::product(isl::multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::project_domain_on_params() const
{
  auto res = isl_multi_pw_aff_project_domain_on_params(copy());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::pullback(isl::multi_aff ma) const
{
  auto res = isl_multi_pw_aff_pullback_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::pullback(isl::multi_pw_aff mpa2) const
{
  auto res = isl_multi_pw_aff_pullback_multi_pw_aff(copy(), mpa2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::pullback(isl::pw_multi_aff pma) const
{
  auto res = isl_multi_pw_aff_pullback_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::range_factor_domain() const
{
  auto res = isl_multi_pw_aff_range_factor_domain(copy());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::range_factor_range() const
{
  auto res = isl_multi_pw_aff_range_factor_range(copy());
  return manage(res);
}

boolean multi_pw_aff::range_is_wrapping() const
{
  auto res = isl_multi_pw_aff_range_is_wrapping(get());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::range_product(isl::multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::range_splice(unsigned int pos, isl::multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_range_splice(copy(), pos, multi2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::reset_tuple_id(isl::dim type) const
{
  auto res = isl_multi_pw_aff_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::reset_user() const
{
  auto res = isl_multi_pw_aff_reset_user(copy());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::scale(isl::multi_val mv) const
{
  auto res = isl_multi_pw_aff_scale_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::scale(isl::val v) const
{
  auto res = isl_multi_pw_aff_scale_val(copy(), v.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::scale_down(isl::multi_val mv) const
{
  auto res = isl_multi_pw_aff_scale_down_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::scale_down(isl::val v) const
{
  auto res = isl_multi_pw_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::set_at(int pos, isl::pw_aff el) const
{
  auto res = isl_multi_pw_aff_set_at(copy(), pos, el.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const
{
  auto res = isl_multi_pw_aff_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::set_pw_aff(int pos, isl::pw_aff el) const
{
  auto res = isl_multi_pw_aff_set_pw_aff(copy(), pos, el.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::set_tuple_id(isl::dim type, isl::id id) const
{
  auto res = isl_multi_pw_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::set_tuple_name(isl::dim type, const std::string &s) const
{
  auto res = isl_multi_pw_aff_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

isl_size multi_pw_aff::size() const
{
  auto res = isl_multi_pw_aff_size(get());
  return res;
}

isl::multi_pw_aff multi_pw_aff::splice(unsigned int in_pos, unsigned int out_pos, isl::multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_splice(copy(), in_pos, out_pos, multi2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::sub(isl::multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_sub(copy(), multi2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::unbind_params_insert_domain(isl::multi_id domain) const
{
  auto res = isl_multi_pw_aff_unbind_params_insert_domain(copy(), domain.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::union_add(isl::multi_pw_aff mpa2) const
{
  auto res = isl_multi_pw_aff_union_add(copy(), mpa2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::zero(isl::space space)
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


multi_union_pw_aff::multi_union_pw_aff(__isl_take isl_multi_union_pw_aff *ptr)
    : ptr(ptr) {}

multi_union_pw_aff::multi_union_pw_aff(isl::multi_pw_aff mpa)
{
  auto res = isl_multi_union_pw_aff_from_multi_pw_aff(mpa.release());
  ptr = res;
}
multi_union_pw_aff::multi_union_pw_aff(isl::union_pw_aff upa)
{
  auto res = isl_multi_union_pw_aff_from_union_pw_aff(upa.release());
  ptr = res;
}
multi_union_pw_aff::multi_union_pw_aff(isl::space space, isl::union_pw_aff_list list)
{
  auto res = isl_multi_union_pw_aff_from_union_pw_aff_list(space.release(), list.release());
  ptr = res;
}
multi_union_pw_aff::multi_union_pw_aff(isl::union_pw_multi_aff upma)
{
  auto res = isl_multi_union_pw_aff_from_union_pw_multi_aff(upma.release());
  ptr = res;
}
multi_union_pw_aff::multi_union_pw_aff(isl::ctx ctx, const std::string &str)
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


isl::ctx multi_union_pw_aff::ctx() const {
  return isl::ctx(isl_multi_union_pw_aff_get_ctx(ptr));
}

void multi_union_pw_aff::dump() const {
  isl_multi_union_pw_aff_dump(get());
}


isl::multi_union_pw_aff multi_union_pw_aff::add(isl::multi_union_pw_aff multi2) const
{
  auto res = isl_multi_union_pw_aff_add(copy(), multi2.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::align_params(isl::space model) const
{
  auto res = isl_multi_union_pw_aff_align_params(copy(), model.release());
  return manage(res);
}

isl::union_pw_aff multi_union_pw_aff::apply_aff(isl::aff aff) const
{
  auto res = isl_multi_union_pw_aff_apply_aff(copy(), aff.release());
  return manage(res);
}

isl::union_pw_aff multi_union_pw_aff::apply_pw_aff(isl::pw_aff pa) const
{
  auto res = isl_multi_union_pw_aff_apply_pw_aff(copy(), pa.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::apply_pw_multi_aff(isl::pw_multi_aff pma) const
{
  auto res = isl_multi_union_pw_aff_apply_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::union_set multi_union_pw_aff::bind(isl::multi_id tuple) const
{
  auto res = isl_multi_union_pw_aff_bind(copy(), tuple.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::coalesce() const
{
  auto res = isl_multi_union_pw_aff_coalesce(copy());
  return manage(res);
}

isl_size multi_union_pw_aff::dim(isl::dim type) const
{
  auto res = isl_multi_union_pw_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::union_set multi_union_pw_aff::domain() const
{
  auto res = isl_multi_union_pw_aff_domain(copy());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_multi_union_pw_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::multi_pw_aff multi_union_pw_aff::extract_multi_pw_aff(isl::space space) const
{
  auto res = isl_multi_union_pw_aff_extract_multi_pw_aff(get(), space.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::factor_range() const
{
  auto res = isl_multi_union_pw_aff_factor_range(copy());
  return manage(res);
}

int multi_union_pw_aff::find_dim_by_id(isl::dim type, const isl::id &id) const
{
  auto res = isl_multi_union_pw_aff_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int multi_union_pw_aff::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_multi_union_pw_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::multi_union_pw_aff multi_union_pw_aff::flat_range_product(isl::multi_union_pw_aff multi2) const
{
  auto res = isl_multi_union_pw_aff_flat_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::flatten_range() const
{
  auto res = isl_multi_union_pw_aff_flatten_range(copy());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::floor() const
{
  auto res = isl_multi_union_pw_aff_floor(copy());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::from_multi_aff(isl::multi_aff ma)
{
  auto res = isl_multi_union_pw_aff_from_multi_aff(ma.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::from_range() const
{
  auto res = isl_multi_union_pw_aff_from_range(copy());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::from_union_map(isl::union_map umap)
{
  auto res = isl_multi_union_pw_aff_from_union_map(umap.release());
  return manage(res);
}

isl::union_pw_aff multi_union_pw_aff::get_at(int pos) const
{
  auto res = isl_multi_union_pw_aff_get_at(get(), pos);
  return manage(res);
}

isl::id multi_union_pw_aff::get_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_multi_union_pw_aff_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::space multi_union_pw_aff::get_domain_space() const
{
  auto res = isl_multi_union_pw_aff_get_domain_space(get());
  return manage(res);
}

isl::union_pw_aff_list multi_union_pw_aff::get_list() const
{
  auto res = isl_multi_union_pw_aff_get_list(get());
  return manage(res);
}

isl::space multi_union_pw_aff::get_space() const
{
  auto res = isl_multi_union_pw_aff_get_space(get());
  return manage(res);
}

isl::id multi_union_pw_aff::get_tuple_id(isl::dim type) const
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

isl::union_pw_aff multi_union_pw_aff::get_union_pw_aff(int pos) const
{
  auto res = isl_multi_union_pw_aff_get_union_pw_aff(get(), pos);
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::gist(isl::union_set context) const
{
  auto res = isl_multi_union_pw_aff_gist(copy(), context.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::gist_params(isl::set context) const
{
  auto res = isl_multi_union_pw_aff_gist_params(copy(), context.release());
  return manage(res);
}

boolean multi_union_pw_aff::has_tuple_id(isl::dim type) const
{
  auto res = isl_multi_union_pw_aff_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::intersect_domain(isl::union_set uset) const
{
  auto res = isl_multi_union_pw_aff_intersect_domain(copy(), uset.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::intersect_params(isl::set params) const
{
  auto res = isl_multi_union_pw_aff_intersect_params(copy(), params.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::intersect_range(isl::set set) const
{
  auto res = isl_multi_union_pw_aff_intersect_range(copy(), set.release());
  return manage(res);
}

boolean multi_union_pw_aff::involves_nan() const
{
  auto res = isl_multi_union_pw_aff_involves_nan(get());
  return manage(res);
}

isl::multi_val multi_union_pw_aff::max_multi_val() const
{
  auto res = isl_multi_union_pw_aff_max_multi_val(copy());
  return manage(res);
}

isl::multi_val multi_union_pw_aff::min_multi_val() const
{
  auto res = isl_multi_union_pw_aff_min_multi_val(copy());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::mod_multi_val(isl::multi_val mv) const
{
  auto res = isl_multi_union_pw_aff_mod_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::multi_aff_on_domain(isl::union_set domain, isl::multi_aff ma)
{
  auto res = isl_multi_union_pw_aff_multi_aff_on_domain(domain.release(), ma.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::multi_val_on_domain(isl::union_set domain, isl::multi_val mv)
{
  auto res = isl_multi_union_pw_aff_multi_val_on_domain(domain.release(), mv.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::neg() const
{
  auto res = isl_multi_union_pw_aff_neg(copy());
  return manage(res);
}

boolean multi_union_pw_aff::plain_is_equal(const isl::multi_union_pw_aff &multi2) const
{
  auto res = isl_multi_union_pw_aff_plain_is_equal(get(), multi2.get());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::pullback(isl::union_pw_multi_aff upma) const
{
  auto res = isl_multi_union_pw_aff_pullback_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::pw_multi_aff_on_domain(isl::union_set domain, isl::pw_multi_aff pma)
{
  auto res = isl_multi_union_pw_aff_pw_multi_aff_on_domain(domain.release(), pma.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::range_factor_domain() const
{
  auto res = isl_multi_union_pw_aff_range_factor_domain(copy());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::range_factor_range() const
{
  auto res = isl_multi_union_pw_aff_range_factor_range(copy());
  return manage(res);
}

boolean multi_union_pw_aff::range_is_wrapping() const
{
  auto res = isl_multi_union_pw_aff_range_is_wrapping(get());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::range_product(isl::multi_union_pw_aff multi2) const
{
  auto res = isl_multi_union_pw_aff_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::range_splice(unsigned int pos, isl::multi_union_pw_aff multi2) const
{
  auto res = isl_multi_union_pw_aff_range_splice(copy(), pos, multi2.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::reset_tuple_id(isl::dim type) const
{
  auto res = isl_multi_union_pw_aff_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::reset_user() const
{
  auto res = isl_multi_union_pw_aff_reset_user(copy());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::scale(isl::multi_val mv) const
{
  auto res = isl_multi_union_pw_aff_scale_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::scale(isl::val v) const
{
  auto res = isl_multi_union_pw_aff_scale_val(copy(), v.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::scale_down(isl::multi_val mv) const
{
  auto res = isl_multi_union_pw_aff_scale_down_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::scale_down(isl::val v) const
{
  auto res = isl_multi_union_pw_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::set_at(int pos, isl::union_pw_aff el) const
{
  auto res = isl_multi_union_pw_aff_set_at(copy(), pos, el.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const
{
  auto res = isl_multi_union_pw_aff_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::set_tuple_id(isl::dim type, isl::id id) const
{
  auto res = isl_multi_union_pw_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::set_tuple_name(isl::dim type, const std::string &s) const
{
  auto res = isl_multi_union_pw_aff_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::set_union_pw_aff(int pos, isl::union_pw_aff el) const
{
  auto res = isl_multi_union_pw_aff_set_union_pw_aff(copy(), pos, el.release());
  return manage(res);
}

isl_size multi_union_pw_aff::size() const
{
  auto res = isl_multi_union_pw_aff_size(get());
  return res;
}

isl::multi_union_pw_aff multi_union_pw_aff::sub(isl::multi_union_pw_aff multi2) const
{
  auto res = isl_multi_union_pw_aff_sub(copy(), multi2.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::union_add(isl::multi_union_pw_aff mupa2) const
{
  auto res = isl_multi_union_pw_aff_union_add(copy(), mupa2.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::zero(isl::space space)
{
  auto res = isl_multi_union_pw_aff_zero(space.release());
  return manage(res);
}

isl::union_set multi_union_pw_aff::zero_union_set() const
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


multi_val::multi_val(__isl_take isl_multi_val *ptr)
    : ptr(ptr) {}

multi_val::multi_val(isl::space space, isl::val_list list)
{
  auto res = isl_multi_val_from_val_list(space.release(), list.release());
  ptr = res;
}
multi_val::multi_val(isl::ctx ctx, const std::string &str)
{
  auto res = isl_multi_val_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

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


isl::ctx multi_val::ctx() const {
  return isl::ctx(isl_multi_val_get_ctx(ptr));
}

void multi_val::dump() const {
  isl_multi_val_dump(get());
}


isl::multi_val multi_val::add(isl::multi_val multi2) const
{
  auto res = isl_multi_val_add(copy(), multi2.release());
  return manage(res);
}

isl::multi_val multi_val::add(isl::val v) const
{
  auto res = isl_multi_val_add_val(copy(), v.release());
  return manage(res);
}

isl::multi_val multi_val::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_multi_val_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::multi_val multi_val::align_params(isl::space model) const
{
  auto res = isl_multi_val_align_params(copy(), model.release());
  return manage(res);
}

isl_size multi_val::dim(isl::dim type) const
{
  auto res = isl_multi_val_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::multi_val multi_val::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_multi_val_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::multi_val multi_val::factor_range() const
{
  auto res = isl_multi_val_factor_range(copy());
  return manage(res);
}

int multi_val::find_dim_by_id(isl::dim type, const isl::id &id) const
{
  auto res = isl_multi_val_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int multi_val::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_multi_val_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::multi_val multi_val::flat_range_product(isl::multi_val multi2) const
{
  auto res = isl_multi_val_flat_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_val multi_val::flatten_range() const
{
  auto res = isl_multi_val_flatten_range(copy());
  return manage(res);
}

isl::multi_val multi_val::from_range() const
{
  auto res = isl_multi_val_from_range(copy());
  return manage(res);
}

isl::val multi_val::get_at(int pos) const
{
  auto res = isl_multi_val_get_at(get(), pos);
  return manage(res);
}

isl::id multi_val::get_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_multi_val_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::space multi_val::get_domain_space() const
{
  auto res = isl_multi_val_get_domain_space(get());
  return manage(res);
}

isl::val_list multi_val::get_list() const
{
  auto res = isl_multi_val_get_list(get());
  return manage(res);
}

isl::space multi_val::get_space() const
{
  auto res = isl_multi_val_get_space(get());
  return manage(res);
}

isl::id multi_val::get_tuple_id(isl::dim type) const
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

isl::val multi_val::get_val(int pos) const
{
  auto res = isl_multi_val_get_val(get(), pos);
  return manage(res);
}

boolean multi_val::has_tuple_id(isl::dim type) const
{
  auto res = isl_multi_val_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::multi_val multi_val::insert_dims(isl::dim type, unsigned int first, unsigned int n) const
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

boolean multi_val::is_zero() const
{
  auto res = isl_multi_val_is_zero(get());
  return manage(res);
}

isl::multi_val multi_val::max(isl::multi_val multi2) const
{
  auto res = isl_multi_val_max(copy(), multi2.release());
  return manage(res);
}

isl::multi_val multi_val::min(isl::multi_val multi2) const
{
  auto res = isl_multi_val_min(copy(), multi2.release());
  return manage(res);
}

isl::multi_val multi_val::mod_multi_val(isl::multi_val mv) const
{
  auto res = isl_multi_val_mod_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_val multi_val::mod_val(isl::val v) const
{
  auto res = isl_multi_val_mod_val(copy(), v.release());
  return manage(res);
}

isl::multi_val multi_val::neg() const
{
  auto res = isl_multi_val_neg(copy());
  return manage(res);
}

boolean multi_val::plain_is_equal(const isl::multi_val &multi2) const
{
  auto res = isl_multi_val_plain_is_equal(get(), multi2.get());
  return manage(res);
}

isl::multi_val multi_val::product(isl::multi_val multi2) const
{
  auto res = isl_multi_val_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_val multi_val::project_domain_on_params() const
{
  auto res = isl_multi_val_project_domain_on_params(copy());
  return manage(res);
}

isl::multi_val multi_val::range_factor_domain() const
{
  auto res = isl_multi_val_range_factor_domain(copy());
  return manage(res);
}

isl::multi_val multi_val::range_factor_range() const
{
  auto res = isl_multi_val_range_factor_range(copy());
  return manage(res);
}

boolean multi_val::range_is_wrapping() const
{
  auto res = isl_multi_val_range_is_wrapping(get());
  return manage(res);
}

isl::multi_val multi_val::range_product(isl::multi_val multi2) const
{
  auto res = isl_multi_val_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_val multi_val::range_splice(unsigned int pos, isl::multi_val multi2) const
{
  auto res = isl_multi_val_range_splice(copy(), pos, multi2.release());
  return manage(res);
}

isl::multi_val multi_val::reset_tuple_id(isl::dim type) const
{
  auto res = isl_multi_val_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::multi_val multi_val::reset_user() const
{
  auto res = isl_multi_val_reset_user(copy());
  return manage(res);
}

isl::multi_val multi_val::scale(isl::multi_val mv) const
{
  auto res = isl_multi_val_scale_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_val multi_val::scale(isl::val v) const
{
  auto res = isl_multi_val_scale_val(copy(), v.release());
  return manage(res);
}

isl::multi_val multi_val::scale_down(isl::multi_val mv) const
{
  auto res = isl_multi_val_scale_down_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_val multi_val::scale_down(isl::val v) const
{
  auto res = isl_multi_val_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::multi_val multi_val::set_at(int pos, isl::val el) const
{
  auto res = isl_multi_val_set_at(copy(), pos, el.release());
  return manage(res);
}

isl::multi_val multi_val::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const
{
  auto res = isl_multi_val_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::multi_val multi_val::set_tuple_id(isl::dim type, isl::id id) const
{
  auto res = isl_multi_val_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::multi_val multi_val::set_tuple_name(isl::dim type, const std::string &s) const
{
  auto res = isl_multi_val_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

isl::multi_val multi_val::set_val(int pos, isl::val el) const
{
  auto res = isl_multi_val_set_val(copy(), pos, el.release());
  return manage(res);
}

isl_size multi_val::size() const
{
  auto res = isl_multi_val_size(get());
  return res;
}

isl::multi_val multi_val::splice(unsigned int in_pos, unsigned int out_pos, isl::multi_val multi2) const
{
  auto res = isl_multi_val_splice(copy(), in_pos, out_pos, multi2.release());
  return manage(res);
}

isl::multi_val multi_val::sub(isl::multi_val multi2) const
{
  auto res = isl_multi_val_sub(copy(), multi2.release());
  return manage(res);
}

isl::multi_val multi_val::zero(isl::space space)
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


point::point(__isl_take isl_point *ptr)
    : ptr(ptr) {}

point::point(isl::space dim)
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


isl::ctx point::ctx() const {
  return isl::ctx(isl_point_get_ctx(ptr));
}

void point::dump() const {
  isl_point_dump(get());
}


isl::point point::add_ui(isl::dim type, int pos, unsigned int val) const
{
  auto res = isl_point_add_ui(copy(), static_cast<enum isl_dim_type>(type), pos, val);
  return manage(res);
}

isl::val point::get_coordinate_val(isl::dim type, int pos) const
{
  auto res = isl_point_get_coordinate_val(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::multi_val point::get_multi_val() const
{
  auto res = isl_point_get_multi_val(get());
  return manage(res);
}

isl::space point::get_space() const
{
  auto res = isl_point_get_space(get());
  return manage(res);
}

isl::point point::set_coordinate_val(isl::dim type, int pos, isl::val v) const
{
  auto res = isl_point_set_coordinate_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

isl::point point::sub_ui(isl::dim type, int pos, unsigned int val) const
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


pw_aff::pw_aff(__isl_take isl_pw_aff *ptr)
    : ptr(ptr) {}

pw_aff::pw_aff(isl::aff aff)
{
  auto res = isl_pw_aff_from_aff(aff.release());
  ptr = res;
}
pw_aff::pw_aff(isl::ctx ctx, const std::string &str)
{
  auto res = isl_pw_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}
pw_aff::pw_aff(isl::set domain, isl::val v)
{
  auto res = isl_pw_aff_val_on_domain(domain.release(), v.release());
  ptr = res;
}
pw_aff::pw_aff(isl::local_space ls)
{
  auto res = isl_pw_aff_zero_on_domain(ls.release());
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


isl::ctx pw_aff::ctx() const {
  return isl::ctx(isl_pw_aff_get_ctx(ptr));
}

void pw_aff::dump() const {
  isl_pw_aff_dump(get());
}


isl::pw_aff pw_aff::add(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_add(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::add_constant(isl::val v) const
{
  auto res = isl_pw_aff_add_constant_val(copy(), v.release());
  return manage(res);
}

isl::pw_aff pw_aff::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_pw_aff_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::pw_aff pw_aff::align_params(isl::space model) const
{
  auto res = isl_pw_aff_align_params(copy(), model.release());
  return manage(res);
}

isl::pw_aff pw_aff::alloc(isl::set set, isl::aff aff)
{
  auto res = isl_pw_aff_alloc(set.release(), aff.release());
  return manage(res);
}

isl::aff pw_aff::as_aff() const
{
  auto res = isl_pw_aff_as_aff(copy());
  return manage(res);
}

isl::set pw_aff::bind(isl::id id) const
{
  auto res = isl_pw_aff_bind_id(copy(), id.release());
  return manage(res);
}

isl::pw_aff pw_aff::bind_domain(isl::multi_id tuple) const
{
  auto res = isl_pw_aff_bind_domain(copy(), tuple.release());
  return manage(res);
}

isl::pw_aff pw_aff::bind_domain_wrapped_domain(isl::multi_id tuple) const
{
  auto res = isl_pw_aff_bind_domain_wrapped_domain(copy(), tuple.release());
  return manage(res);
}

isl::pw_aff pw_aff::ceil() const
{
  auto res = isl_pw_aff_ceil(copy());
  return manage(res);
}

isl::pw_aff pw_aff::coalesce() const
{
  auto res = isl_pw_aff_coalesce(copy());
  return manage(res);
}

isl::pw_aff pw_aff::cond(isl::pw_aff pwaff_true, isl::pw_aff pwaff_false) const
{
  auto res = isl_pw_aff_cond(copy(), pwaff_true.release(), pwaff_false.release());
  return manage(res);
}

isl_size pw_aff::dim(isl::dim type) const
{
  auto res = isl_pw_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::pw_aff pw_aff::div(isl::pw_aff pa2) const
{
  auto res = isl_pw_aff_div(copy(), pa2.release());
  return manage(res);
}

isl::set pw_aff::domain() const
{
  auto res = isl_pw_aff_domain(copy());
  return manage(res);
}

isl::pw_aff pw_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_pw_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::pw_aff pw_aff::drop_unused_params() const
{
  auto res = isl_pw_aff_drop_unused_params(copy());
  return manage(res);
}

isl::pw_aff pw_aff::empty(isl::space space)
{
  auto res = isl_pw_aff_empty(space.release());
  return manage(res);
}

isl::map pw_aff::eq_map(isl::pw_aff pa2) const
{
  auto res = isl_pw_aff_eq_map(copy(), pa2.release());
  return manage(res);
}

isl::set pw_aff::eq_set(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_eq_set(copy(), pwaff2.release());
  return manage(res);
}

isl::val pw_aff::eval(isl::point pnt) const
{
  auto res = isl_pw_aff_eval(copy(), pnt.release());
  return manage(res);
}

int pw_aff::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_pw_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::pw_aff pw_aff::floor() const
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

isl::pw_aff pw_aff::from_range() const
{
  auto res = isl_pw_aff_from_range(copy());
  return manage(res);
}

isl::map pw_aff::ge_map(isl::pw_aff pa2) const
{
  auto res = isl_pw_aff_ge_map(copy(), pa2.release());
  return manage(res);
}

isl::set pw_aff::ge_set(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_ge_set(copy(), pwaff2.release());
  return manage(res);
}

isl::id pw_aff::get_dim_id(isl::dim type, unsigned int pos) const
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

isl::space pw_aff::get_domain_space() const
{
  auto res = isl_pw_aff_get_domain_space(get());
  return manage(res);
}

uint32_t pw_aff::get_hash() const
{
  auto res = isl_pw_aff_get_hash(get());
  return res;
}

isl::space pw_aff::get_space() const
{
  auto res = isl_pw_aff_get_space(get());
  return manage(res);
}

isl::id pw_aff::get_tuple_id(isl::dim type) const
{
  auto res = isl_pw_aff_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::pw_aff pw_aff::gist(isl::set context) const
{
  auto res = isl_pw_aff_gist(copy(), context.release());
  return manage(res);
}

isl::pw_aff pw_aff::gist_params(isl::set context) const
{
  auto res = isl_pw_aff_gist_params(copy(), context.release());
  return manage(res);
}

isl::map pw_aff::gt_map(isl::pw_aff pa2) const
{
  auto res = isl_pw_aff_gt_map(copy(), pa2.release());
  return manage(res);
}

isl::set pw_aff::gt_set(isl::pw_aff pwaff2) const
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

isl::pw_aff pw_aff::insert_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_pw_aff_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::pw_aff pw_aff::insert_domain(isl::space domain) const
{
  auto res = isl_pw_aff_insert_domain(copy(), domain.release());
  return manage(res);
}

isl::pw_aff pw_aff::intersect_domain(isl::set set) const
{
  auto res = isl_pw_aff_intersect_domain(copy(), set.release());
  return manage(res);
}

isl::pw_aff pw_aff::intersect_domain_wrapped_domain(isl::set set) const
{
  auto res = isl_pw_aff_intersect_domain_wrapped_domain(copy(), set.release());
  return manage(res);
}

isl::pw_aff pw_aff::intersect_domain_wrapped_range(isl::set set) const
{
  auto res = isl_pw_aff_intersect_domain_wrapped_range(copy(), set.release());
  return manage(res);
}

isl::pw_aff pw_aff::intersect_params(isl::set set) const
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

boolean pw_aff::involves_param_id(const isl::id &id) const
{
  auto res = isl_pw_aff_involves_param_id(get(), id.get());
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

boolean pw_aff::is_equal(const isl::pw_aff &pa2) const
{
  auto res = isl_pw_aff_is_equal(get(), pa2.get());
  return manage(res);
}

boolean pw_aff::isa_aff() const
{
  auto res = isl_pw_aff_isa_aff(get());
  return manage(res);
}

isl::map pw_aff::le_map(isl::pw_aff pa2) const
{
  auto res = isl_pw_aff_le_map(copy(), pa2.release());
  return manage(res);
}

isl::set pw_aff::le_set(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_le_set(copy(), pwaff2.release());
  return manage(res);
}

isl::map pw_aff::lt_map(isl::pw_aff pa2) const
{
  auto res = isl_pw_aff_lt_map(copy(), pa2.release());
  return manage(res);
}

isl::set pw_aff::lt_set(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_lt_set(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::max(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_max(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::min(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_min(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::mod(isl::val mod) const
{
  auto res = isl_pw_aff_mod_val(copy(), mod.release());
  return manage(res);
}

isl::pw_aff pw_aff::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_pw_aff_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl::pw_aff pw_aff::mul(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_mul(copy(), pwaff2.release());
  return manage(res);
}

isl_size pw_aff::n_piece() const
{
  auto res = isl_pw_aff_n_piece(get());
  return res;
}

isl::pw_aff pw_aff::nan_on_domain(isl::local_space ls)
{
  auto res = isl_pw_aff_nan_on_domain(ls.release());
  return manage(res);
}

isl::pw_aff pw_aff::nan_on_domain_space(isl::space space)
{
  auto res = isl_pw_aff_nan_on_domain_space(space.release());
  return manage(res);
}

isl::set pw_aff::ne_set(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_ne_set(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::neg() const
{
  auto res = isl_pw_aff_neg(copy());
  return manage(res);
}

isl::set pw_aff::non_zero_set() const
{
  auto res = isl_pw_aff_non_zero_set(copy());
  return manage(res);
}

isl::set pw_aff::nonneg_set() const
{
  auto res = isl_pw_aff_nonneg_set(copy());
  return manage(res);
}

isl::pw_aff pw_aff::param_on_domain(isl::set domain, isl::id id)
{
  auto res = isl_pw_aff_param_on_domain_id(domain.release(), id.release());
  return manage(res);
}

isl::set pw_aff::params() const
{
  auto res = isl_pw_aff_params(copy());
  return manage(res);
}

int pw_aff::plain_cmp(const isl::pw_aff &pa2) const
{
  auto res = isl_pw_aff_plain_cmp(get(), pa2.get());
  return res;
}

boolean pw_aff::plain_is_equal(const isl::pw_aff &pwaff2) const
{
  auto res = isl_pw_aff_plain_is_equal(get(), pwaff2.get());
  return manage(res);
}

isl::set pw_aff::pos_set() const
{
  auto res = isl_pw_aff_pos_set(copy());
  return manage(res);
}

isl::pw_aff pw_aff::project_domain_on_params() const
{
  auto res = isl_pw_aff_project_domain_on_params(copy());
  return manage(res);
}

isl::pw_aff pw_aff::pullback(isl::multi_aff ma) const
{
  auto res = isl_pw_aff_pullback_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::pw_aff pw_aff::pullback(isl::multi_pw_aff mpa) const
{
  auto res = isl_pw_aff_pullback_multi_pw_aff(copy(), mpa.release());
  return manage(res);
}

isl::pw_aff pw_aff::pullback(isl::pw_multi_aff pma) const
{
  auto res = isl_pw_aff_pullback_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::pw_aff pw_aff::reset_tuple_id(isl::dim type) const
{
  auto res = isl_pw_aff_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::pw_aff pw_aff::reset_user() const
{
  auto res = isl_pw_aff_reset_user(copy());
  return manage(res);
}

isl::pw_aff pw_aff::scale(isl::val v) const
{
  auto res = isl_pw_aff_scale_val(copy(), v.release());
  return manage(res);
}

isl::pw_aff pw_aff::scale_down(isl::val f) const
{
  auto res = isl_pw_aff_scale_down_val(copy(), f.release());
  return manage(res);
}

isl::pw_aff pw_aff::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const
{
  auto res = isl_pw_aff_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::pw_aff pw_aff::set_tuple_id(isl::dim type, isl::id id) const
{
  auto res = isl_pw_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::pw_aff pw_aff::sub(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_sub(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::subtract_domain(isl::set set) const
{
  auto res = isl_pw_aff_subtract_domain(copy(), set.release());
  return manage(res);
}

isl::pw_aff pw_aff::tdiv_q(isl::pw_aff pa2) const
{
  auto res = isl_pw_aff_tdiv_q(copy(), pa2.release());
  return manage(res);
}

isl::pw_aff pw_aff::tdiv_r(isl::pw_aff pa2) const
{
  auto res = isl_pw_aff_tdiv_r(copy(), pa2.release());
  return manage(res);
}

isl::pw_aff pw_aff::union_add(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_union_add(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::union_max(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_union_max(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::union_min(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_union_min(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::var_on_domain(isl::local_space ls, isl::dim type, unsigned int pos)
{
  auto res = isl_pw_aff_var_on_domain(ls.release(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::set pw_aff::zero_set() const
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


isl::ctx pw_aff_list::ctx() const {
  return isl::ctx(isl_pw_aff_list_get_ctx(ptr));
}

void pw_aff_list::dump() const {
  isl_pw_aff_list_dump(get());
}


isl::pw_aff_list pw_aff_list::add(isl::pw_aff el) const
{
  auto res = isl_pw_aff_list_add(copy(), el.release());
  return manage(res);
}

isl::pw_aff_list pw_aff_list::alloc(isl::ctx ctx, int n)
{
  auto res = isl_pw_aff_list_alloc(ctx.release(), n);
  return manage(res);
}

isl::pw_aff_list pw_aff_list::clear() const
{
  auto res = isl_pw_aff_list_clear(copy());
  return manage(res);
}

isl::pw_aff_list pw_aff_list::concat(isl::pw_aff_list list2) const
{
  auto res = isl_pw_aff_list_concat(copy(), list2.release());
  return manage(res);
}

isl::pw_aff_list pw_aff_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_pw_aff_list_drop(copy(), first, n);
  return manage(res);
}

isl::set pw_aff_list::eq_set(isl::pw_aff_list list2) const
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

isl::pw_aff_list pw_aff_list::from_pw_aff(isl::pw_aff el)
{
  auto res = isl_pw_aff_list_from_pw_aff(el.release());
  return manage(res);
}

isl::set pw_aff_list::ge_set(isl::pw_aff_list list2) const
{
  auto res = isl_pw_aff_list_ge_set(copy(), list2.release());
  return manage(res);
}

isl::pw_aff pw_aff_list::get_at(int index) const
{
  auto res = isl_pw_aff_list_get_at(get(), index);
  return manage(res);
}

isl::pw_aff pw_aff_list::get_pw_aff(int index) const
{
  auto res = isl_pw_aff_list_get_pw_aff(get(), index);
  return manage(res);
}

isl::set pw_aff_list::gt_set(isl::pw_aff_list list2) const
{
  auto res = isl_pw_aff_list_gt_set(copy(), list2.release());
  return manage(res);
}

isl::pw_aff_list pw_aff_list::insert(unsigned int pos, isl::pw_aff el) const
{
  auto res = isl_pw_aff_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl::set pw_aff_list::le_set(isl::pw_aff_list list2) const
{
  auto res = isl_pw_aff_list_le_set(copy(), list2.release());
  return manage(res);
}

isl::set pw_aff_list::lt_set(isl::pw_aff_list list2) const
{
  auto res = isl_pw_aff_list_lt_set(copy(), list2.release());
  return manage(res);
}

isl::pw_aff pw_aff_list::max() const
{
  auto res = isl_pw_aff_list_max(copy());
  return manage(res);
}

isl::pw_aff pw_aff_list::min() const
{
  auto res = isl_pw_aff_list_min(copy());
  return manage(res);
}

isl_size pw_aff_list::n_pw_aff() const
{
  auto res = isl_pw_aff_list_n_pw_aff(get());
  return res;
}

isl::set pw_aff_list::ne_set(isl::pw_aff_list list2) const
{
  auto res = isl_pw_aff_list_ne_set(copy(), list2.release());
  return manage(res);
}

isl::pw_aff_list pw_aff_list::reverse() const
{
  auto res = isl_pw_aff_list_reverse(copy());
  return manage(res);
}

isl::pw_aff_list pw_aff_list::set_pw_aff(int index, isl::pw_aff el) const
{
  auto res = isl_pw_aff_list_set_pw_aff(copy(), index, el.release());
  return manage(res);
}

isl_size pw_aff_list::size() const
{
  auto res = isl_pw_aff_list_size(get());
  return res;
}

isl::pw_aff_list pw_aff_list::swap(unsigned int pos1, unsigned int pos2) const
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


pw_multi_aff::pw_multi_aff(__isl_take isl_pw_multi_aff *ptr)
    : ptr(ptr) {}

pw_multi_aff::pw_multi_aff(isl::multi_aff ma)
{
  auto res = isl_pw_multi_aff_from_multi_aff(ma.release());
  ptr = res;
}
pw_multi_aff::pw_multi_aff(isl::pw_aff pa)
{
  auto res = isl_pw_multi_aff_from_pw_aff(pa.release());
  ptr = res;
}
pw_multi_aff::pw_multi_aff(isl::ctx ctx, const std::string &str)
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


isl::ctx pw_multi_aff::ctx() const {
  return isl::ctx(isl_pw_multi_aff_get_ctx(ptr));
}

void pw_multi_aff::dump() const {
  isl_pw_multi_aff_dump(get());
}


isl::pw_multi_aff pw_multi_aff::add(isl::pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_add(copy(), pma2.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::add_constant(isl::multi_val mv) const
{
  auto res = isl_pw_multi_aff_add_constant_multi_val(copy(), mv.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::add_constant(isl::val v) const
{
  auto res = isl_pw_multi_aff_add_constant_val(copy(), v.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::align_params(isl::space model) const
{
  auto res = isl_pw_multi_aff_align_params(copy(), model.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::alloc(isl::set set, isl::multi_aff maff)
{
  auto res = isl_pw_multi_aff_alloc(set.release(), maff.release());
  return manage(res);
}

isl::multi_aff pw_multi_aff::as_multi_aff() const
{
  auto res = isl_pw_multi_aff_as_multi_aff(copy());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::bind_domain(isl::multi_id tuple) const
{
  auto res = isl_pw_multi_aff_bind_domain(copy(), tuple.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::bind_domain_wrapped_domain(isl::multi_id tuple) const
{
  auto res = isl_pw_multi_aff_bind_domain_wrapped_domain(copy(), tuple.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::coalesce() const
{
  auto res = isl_pw_multi_aff_coalesce(copy());
  return manage(res);
}

isl_size pw_multi_aff::dim(isl::dim type) const
{
  auto res = isl_pw_multi_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::set pw_multi_aff::domain() const
{
  auto res = isl_pw_multi_aff_domain(copy());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::domain_map(isl::space space)
{
  auto res = isl_pw_multi_aff_domain_map(space.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_pw_multi_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::drop_unused_params() const
{
  auto res = isl_pw_multi_aff_drop_unused_params(copy());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::empty(isl::space space)
{
  auto res = isl_pw_multi_aff_empty(space.release());
  return manage(res);
}

int pw_multi_aff::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_pw_multi_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::pw_multi_aff pw_multi_aff::fix_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_pw_multi_aff_fix_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::flat_range_product(isl::pw_multi_aff pma2) const
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

isl::pw_multi_aff pw_multi_aff::from_domain(isl::set set)
{
  auto res = isl_pw_multi_aff_from_domain(set.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::from_map(isl::map map)
{
  auto res = isl_pw_multi_aff_from_map(map.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::from_multi_pw_aff(isl::multi_pw_aff mpa)
{
  auto res = isl_pw_multi_aff_from_multi_pw_aff(mpa.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::from_set(isl::set set)
{
  auto res = isl_pw_multi_aff_from_set(set.release());
  return manage(res);
}

isl::id pw_multi_aff::get_dim_id(isl::dim type, unsigned int pos) const
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

isl::space pw_multi_aff::get_domain_space() const
{
  auto res = isl_pw_multi_aff_get_domain_space(get());
  return manage(res);
}

isl::pw_aff pw_multi_aff::get_pw_aff(int pos) const
{
  auto res = isl_pw_multi_aff_get_pw_aff(get(), pos);
  return manage(res);
}

isl::space pw_multi_aff::get_space() const
{
  auto res = isl_pw_multi_aff_get_space(get());
  return manage(res);
}

isl::id pw_multi_aff::get_tuple_id(isl::dim type) const
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

isl::pw_multi_aff pw_multi_aff::gist(isl::set set) const
{
  auto res = isl_pw_multi_aff_gist(copy(), set.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::gist_params(isl::set set) const
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

isl::pw_multi_aff pw_multi_aff::identity(isl::space space)
{
  auto res = isl_pw_multi_aff_identity(space.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::identity_on_domain(isl::space space)
{
  auto res = isl_pw_multi_aff_identity_on_domain_space(space.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::insert_domain(isl::space domain) const
{
  auto res = isl_pw_multi_aff_insert_domain(copy(), domain.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::intersect_domain(isl::set set) const
{
  auto res = isl_pw_multi_aff_intersect_domain(copy(), set.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::intersect_domain_wrapped_domain(isl::set set) const
{
  auto res = isl_pw_multi_aff_intersect_domain_wrapped_domain(copy(), set.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::intersect_domain_wrapped_range(isl::set set) const
{
  auto res = isl_pw_multi_aff_intersect_domain_wrapped_range(copy(), set.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::intersect_params(isl::set set) const
{
  auto res = isl_pw_multi_aff_intersect_params(copy(), set.release());
  return manage(res);
}

boolean pw_multi_aff::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_pw_multi_aff_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean pw_multi_aff::involves_locals() const
{
  auto res = isl_pw_multi_aff_involves_locals(get());
  return manage(res);
}

boolean pw_multi_aff::involves_nan() const
{
  auto res = isl_pw_multi_aff_involves_nan(get());
  return manage(res);
}

boolean pw_multi_aff::involves_param_id(const isl::id &id) const
{
  auto res = isl_pw_multi_aff_involves_param_id(get(), id.get());
  return manage(res);
}

boolean pw_multi_aff::is_equal(const isl::pw_multi_aff &pma2) const
{
  auto res = isl_pw_multi_aff_is_equal(get(), pma2.get());
  return manage(res);
}

boolean pw_multi_aff::isa_multi_aff() const
{
  auto res = isl_pw_multi_aff_isa_multi_aff(get());
  return manage(res);
}

isl::multi_val pw_multi_aff::max_multi_val() const
{
  auto res = isl_pw_multi_aff_max_multi_val(copy());
  return manage(res);
}

isl::multi_val pw_multi_aff::min_multi_val() const
{
  auto res = isl_pw_multi_aff_min_multi_val(copy());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::multi_val_on_domain(isl::set domain, isl::multi_val mv)
{
  auto res = isl_pw_multi_aff_multi_val_on_domain(domain.release(), mv.release());
  return manage(res);
}

isl_size pw_multi_aff::n_piece() const
{
  auto res = isl_pw_multi_aff_n_piece(get());
  return res;
}

isl::pw_multi_aff pw_multi_aff::neg() const
{
  auto res = isl_pw_multi_aff_neg(copy());
  return manage(res);
}

boolean pw_multi_aff::plain_is_equal(const isl::pw_multi_aff &pma2) const
{
  auto res = isl_pw_multi_aff_plain_is_equal(get(), pma2.get());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::preimage_domain_wrapped_domain(isl::pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_preimage_domain_wrapped_domain_pw_multi_aff(copy(), pma2.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::product(isl::pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_product(copy(), pma2.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::project_domain_on_params() const
{
  auto res = isl_pw_multi_aff_project_domain_on_params(copy());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::project_out_map(isl::space space, isl::dim type, unsigned int first, unsigned int n)
{
  auto res = isl_pw_multi_aff_project_out_map(space.release(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::pullback(isl::multi_aff ma) const
{
  auto res = isl_pw_multi_aff_pullback_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::pullback(isl::pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_pullback_pw_multi_aff(copy(), pma2.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::range_factor_domain() const
{
  auto res = isl_pw_multi_aff_range_factor_domain(copy());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::range_factor_range() const
{
  auto res = isl_pw_multi_aff_range_factor_range(copy());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::range_map(isl::space space)
{
  auto res = isl_pw_multi_aff_range_map(space.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::range_product(isl::pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_range_product(copy(), pma2.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::reset_tuple_id(isl::dim type) const
{
  auto res = isl_pw_multi_aff_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::reset_user() const
{
  auto res = isl_pw_multi_aff_reset_user(copy());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::scale(isl::val v) const
{
  auto res = isl_pw_multi_aff_scale_val(copy(), v.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::scale_down(isl::val v) const
{
  auto res = isl_pw_multi_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::scale_multi_val(isl::multi_val mv) const
{
  auto res = isl_pw_multi_aff_scale_multi_val(copy(), mv.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const
{
  auto res = isl_pw_multi_aff_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::set_pw_aff(unsigned int pos, isl::pw_aff pa) const
{
  auto res = isl_pw_multi_aff_set_pw_aff(copy(), pos, pa.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::set_tuple_id(isl::dim type, isl::id id) const
{
  auto res = isl_pw_multi_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::sub(isl::pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_sub(copy(), pma2.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::subtract_domain(isl::set set) const
{
  auto res = isl_pw_multi_aff_subtract_domain(copy(), set.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::union_add(isl::pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_union_add(copy(), pma2.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::union_lexmax(isl::pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_union_lexmax(copy(), pma2.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::union_lexmin(isl::pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_union_lexmin(copy(), pma2.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::zero(isl::space space)
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


isl::ctx pw_multi_aff_list::ctx() const {
  return isl::ctx(isl_pw_multi_aff_list_get_ctx(ptr));
}

void pw_multi_aff_list::dump() const {
  isl_pw_multi_aff_list_dump(get());
}


isl::pw_multi_aff_list pw_multi_aff_list::add(isl::pw_multi_aff el) const
{
  auto res = isl_pw_multi_aff_list_add(copy(), el.release());
  return manage(res);
}

isl::pw_multi_aff_list pw_multi_aff_list::alloc(isl::ctx ctx, int n)
{
  auto res = isl_pw_multi_aff_list_alloc(ctx.release(), n);
  return manage(res);
}

isl::pw_multi_aff_list pw_multi_aff_list::clear() const
{
  auto res = isl_pw_multi_aff_list_clear(copy());
  return manage(res);
}

isl::pw_multi_aff_list pw_multi_aff_list::concat(isl::pw_multi_aff_list list2) const
{
  auto res = isl_pw_multi_aff_list_concat(copy(), list2.release());
  return manage(res);
}

isl::pw_multi_aff_list pw_multi_aff_list::drop(unsigned int first, unsigned int n) const
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

isl::pw_multi_aff_list pw_multi_aff_list::from_pw_multi_aff(isl::pw_multi_aff el)
{
  auto res = isl_pw_multi_aff_list_from_pw_multi_aff(el.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff_list::get_at(int index) const
{
  auto res = isl_pw_multi_aff_list_get_at(get(), index);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff_list::get_pw_multi_aff(int index) const
{
  auto res = isl_pw_multi_aff_list_get_pw_multi_aff(get(), index);
  return manage(res);
}

isl::pw_multi_aff_list pw_multi_aff_list::insert(unsigned int pos, isl::pw_multi_aff el) const
{
  auto res = isl_pw_multi_aff_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl_size pw_multi_aff_list::n_pw_multi_aff() const
{
  auto res = isl_pw_multi_aff_list_n_pw_multi_aff(get());
  return res;
}

isl::pw_multi_aff_list pw_multi_aff_list::reverse() const
{
  auto res = isl_pw_multi_aff_list_reverse(copy());
  return manage(res);
}

isl::pw_multi_aff_list pw_multi_aff_list::set_pw_multi_aff(int index, isl::pw_multi_aff el) const
{
  auto res = isl_pw_multi_aff_list_set_pw_multi_aff(copy(), index, el.release());
  return manage(res);
}

isl_size pw_multi_aff_list::size() const
{
  auto res = isl_pw_multi_aff_list_size(get());
  return res;
}

isl::pw_multi_aff_list pw_multi_aff_list::swap(unsigned int pos1, unsigned int pos2) const
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


pw_qpolynomial::pw_qpolynomial(__isl_take isl_pw_qpolynomial *ptr)
    : ptr(ptr) {}

pw_qpolynomial::pw_qpolynomial(isl::ctx ctx, const std::string &str)
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


isl::ctx pw_qpolynomial::ctx() const {
  return isl::ctx(isl_pw_qpolynomial_get_ctx(ptr));
}

void pw_qpolynomial::dump() const {
  isl_pw_qpolynomial_dump(get());
}


isl::pw_qpolynomial pw_qpolynomial::add(isl::pw_qpolynomial pwqp2) const
{
  auto res = isl_pw_qpolynomial_add(copy(), pwqp2.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_pw_qpolynomial_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::alloc(isl::set set, isl::qpolynomial qp)
{
  auto res = isl_pw_qpolynomial_alloc(set.release(), qp.release());
  return manage(res);
}

isl::qpolynomial pw_qpolynomial::as_qpolynomial() const
{
  auto res = isl_pw_qpolynomial_as_qpolynomial(copy());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::coalesce() const
{
  auto res = isl_pw_qpolynomial_coalesce(copy());
  return manage(res);
}

isl_size pw_qpolynomial::dim(isl::dim type) const
{
  auto res = isl_pw_qpolynomial_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::set pw_qpolynomial::domain() const
{
  auto res = isl_pw_qpolynomial_domain(copy());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_pw_qpolynomial_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::drop_unused_params() const
{
  auto res = isl_pw_qpolynomial_drop_unused_params(copy());
  return manage(res);
}

isl::val pw_qpolynomial::eval(isl::point pnt) const
{
  auto res = isl_pw_qpolynomial_eval(copy(), pnt.release());
  return manage(res);
}

int pw_qpolynomial::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_pw_qpolynomial_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::pw_qpolynomial pw_qpolynomial::fix_val(isl::dim type, unsigned int n, isl::val v) const
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

isl::pw_qpolynomial pw_qpolynomial::from_pw_aff(isl::pw_aff pwaff)
{
  auto res = isl_pw_qpolynomial_from_pw_aff(pwaff.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::from_qpolynomial(isl::qpolynomial qp)
{
  auto res = isl_pw_qpolynomial_from_qpolynomial(qp.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::from_range() const
{
  auto res = isl_pw_qpolynomial_from_range(copy());
  return manage(res);
}

isl::space pw_qpolynomial::get_domain_space() const
{
  auto res = isl_pw_qpolynomial_get_domain_space(get());
  return manage(res);
}

isl::space pw_qpolynomial::get_space() const
{
  auto res = isl_pw_qpolynomial_get_space(get());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::gist(isl::set context) const
{
  auto res = isl_pw_qpolynomial_gist(copy(), context.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::gist_params(isl::set context) const
{
  auto res = isl_pw_qpolynomial_gist_params(copy(), context.release());
  return manage(res);
}

boolean pw_qpolynomial::has_equal_space(const isl::pw_qpolynomial &pwqp2) const
{
  auto res = isl_pw_qpolynomial_has_equal_space(get(), pwqp2.get());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::insert_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_pw_qpolynomial_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::intersect_domain(isl::set set) const
{
  auto res = isl_pw_qpolynomial_intersect_domain(copy(), set.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::intersect_domain_wrapped_domain(isl::set set) const
{
  auto res = isl_pw_qpolynomial_intersect_domain_wrapped_domain(copy(), set.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::intersect_domain_wrapped_range(isl::set set) const
{
  auto res = isl_pw_qpolynomial_intersect_domain_wrapped_range(copy(), set.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::intersect_params(isl::set set) const
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

boolean pw_qpolynomial::involves_param_id(const isl::id &id) const
{
  auto res = isl_pw_qpolynomial_involves_param_id(get(), id.get());
  return manage(res);
}

boolean pw_qpolynomial::is_zero() const
{
  auto res = isl_pw_qpolynomial_is_zero(get());
  return manage(res);
}

boolean pw_qpolynomial::isa_qpolynomial() const
{
  auto res = isl_pw_qpolynomial_isa_qpolynomial(get());
  return manage(res);
}

isl::val pw_qpolynomial::max() const
{
  auto res = isl_pw_qpolynomial_max(copy());
  return manage(res);
}

isl::val pw_qpolynomial::min() const
{
  auto res = isl_pw_qpolynomial_min(copy());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_pw_qpolynomial_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::mul(isl::pw_qpolynomial pwqp2) const
{
  auto res = isl_pw_qpolynomial_mul(copy(), pwqp2.release());
  return manage(res);
}

isl_size pw_qpolynomial::n_piece() const
{
  auto res = isl_pw_qpolynomial_n_piece(get());
  return res;
}

isl::pw_qpolynomial pw_qpolynomial::neg() const
{
  auto res = isl_pw_qpolynomial_neg(copy());
  return manage(res);
}

boolean pw_qpolynomial::plain_is_equal(const isl::pw_qpolynomial &pwqp2) const
{
  auto res = isl_pw_qpolynomial_plain_is_equal(get(), pwqp2.get());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::pow(unsigned int exponent) const
{
  auto res = isl_pw_qpolynomial_pow(copy(), exponent);
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::project_domain_on_params() const
{
  auto res = isl_pw_qpolynomial_project_domain_on_params(copy());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::reset_domain_space(isl::space space) const
{
  auto res = isl_pw_qpolynomial_reset_domain_space(copy(), space.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::reset_user() const
{
  auto res = isl_pw_qpolynomial_reset_user(copy());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::scale_down_val(isl::val v) const
{
  auto res = isl_pw_qpolynomial_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::scale_val(isl::val v) const
{
  auto res = isl_pw_qpolynomial_scale_val(copy(), v.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::split_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_pw_qpolynomial_split_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::split_periods(int max_periods) const
{
  auto res = isl_pw_qpolynomial_split_periods(copy(), max_periods);
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::sub(isl::pw_qpolynomial pwqp2) const
{
  auto res = isl_pw_qpolynomial_sub(copy(), pwqp2.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::subtract_domain(isl::set set) const
{
  auto res = isl_pw_qpolynomial_subtract_domain(copy(), set.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::to_polynomial(int sign) const
{
  auto res = isl_pw_qpolynomial_to_polynomial(copy(), sign);
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::zero(isl::space space)
{
  auto res = isl_pw_qpolynomial_zero(space.release());
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


isl::ctx pw_qpolynomial_fold_list::ctx() const {
  return isl::ctx(isl_pw_qpolynomial_fold_list_get_ctx(ptr));
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


isl::ctx pw_qpolynomial_list::ctx() const {
  return isl::ctx(isl_pw_qpolynomial_list_get_ctx(ptr));
}

void pw_qpolynomial_list::dump() const {
  isl_pw_qpolynomial_list_dump(get());
}


isl::pw_qpolynomial_list pw_qpolynomial_list::add(isl::pw_qpolynomial el) const
{
  auto res = isl_pw_qpolynomial_list_add(copy(), el.release());
  return manage(res);
}

isl::pw_qpolynomial_list pw_qpolynomial_list::alloc(isl::ctx ctx, int n)
{
  auto res = isl_pw_qpolynomial_list_alloc(ctx.release(), n);
  return manage(res);
}

isl::pw_qpolynomial_list pw_qpolynomial_list::clear() const
{
  auto res = isl_pw_qpolynomial_list_clear(copy());
  return manage(res);
}

isl::pw_qpolynomial_list pw_qpolynomial_list::concat(isl::pw_qpolynomial_list list2) const
{
  auto res = isl_pw_qpolynomial_list_concat(copy(), list2.release());
  return manage(res);
}

isl::pw_qpolynomial_list pw_qpolynomial_list::drop(unsigned int first, unsigned int n) const
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

isl::pw_qpolynomial_list pw_qpolynomial_list::from_pw_qpolynomial(isl::pw_qpolynomial el)
{
  auto res = isl_pw_qpolynomial_list_from_pw_qpolynomial(el.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial_list::get_at(int index) const
{
  auto res = isl_pw_qpolynomial_list_get_at(get(), index);
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial_list::get_pw_qpolynomial(int index) const
{
  auto res = isl_pw_qpolynomial_list_get_pw_qpolynomial(get(), index);
  return manage(res);
}

isl::pw_qpolynomial_list pw_qpolynomial_list::insert(unsigned int pos, isl::pw_qpolynomial el) const
{
  auto res = isl_pw_qpolynomial_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl_size pw_qpolynomial_list::n_pw_qpolynomial() const
{
  auto res = isl_pw_qpolynomial_list_n_pw_qpolynomial(get());
  return res;
}

isl::pw_qpolynomial_list pw_qpolynomial_list::reverse() const
{
  auto res = isl_pw_qpolynomial_list_reverse(copy());
  return manage(res);
}

isl::pw_qpolynomial_list pw_qpolynomial_list::set_pw_qpolynomial(int index, isl::pw_qpolynomial el) const
{
  auto res = isl_pw_qpolynomial_list_set_pw_qpolynomial(copy(), index, el.release());
  return manage(res);
}

isl_size pw_qpolynomial_list::size() const
{
  auto res = isl_pw_qpolynomial_list_size(get());
  return res;
}

isl::pw_qpolynomial_list pw_qpolynomial_list::swap(unsigned int pos1, unsigned int pos2) const
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


isl::ctx qpolynomial::ctx() const {
  return isl::ctx(isl_qpolynomial_get_ctx(ptr));
}

void qpolynomial::dump() const {
  isl_qpolynomial_dump(get());
}


isl::qpolynomial qpolynomial::add(isl::qpolynomial qp2) const
{
  auto res = isl_qpolynomial_add(copy(), qp2.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_qpolynomial_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::qpolynomial qpolynomial::align_params(isl::space model) const
{
  auto res = isl_qpolynomial_align_params(copy(), model.release());
  return manage(res);
}

stat qpolynomial::as_polynomial_on_domain(const isl::basic_set &bset, const std::function<stat(basic_set, qpolynomial)> &fn) const
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

isl_size qpolynomial::dim(isl::dim type) const
{
  auto res = isl_qpolynomial_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::qpolynomial qpolynomial::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_qpolynomial_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::val qpolynomial::eval(isl::point pnt) const
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

isl::qpolynomial qpolynomial::from_aff(isl::aff aff)
{
  auto res = isl_qpolynomial_from_aff(aff.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::from_constraint(isl::constraint c, isl::dim type, unsigned int pos)
{
  auto res = isl_qpolynomial_from_constraint(c.release(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::qpolynomial qpolynomial::from_term(isl::term term)
{
  auto res = isl_qpolynomial_from_term(term.release());
  return manage(res);
}

isl::val qpolynomial::get_constant_val() const
{
  auto res = isl_qpolynomial_get_constant_val(get());
  return manage(res);
}

isl::space qpolynomial::get_domain_space() const
{
  auto res = isl_qpolynomial_get_domain_space(get());
  return manage(res);
}

isl::space qpolynomial::get_space() const
{
  auto res = isl_qpolynomial_get_space(get());
  return manage(res);
}

isl::qpolynomial qpolynomial::gist(isl::set context) const
{
  auto res = isl_qpolynomial_gist(copy(), context.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::gist_params(isl::set context) const
{
  auto res = isl_qpolynomial_gist_params(copy(), context.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::homogenize() const
{
  auto res = isl_qpolynomial_homogenize(copy());
  return manage(res);
}

isl::qpolynomial qpolynomial::infty_on_domain(isl::space domain)
{
  auto res = isl_qpolynomial_infty_on_domain(domain.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::insert_dims(isl::dim type, unsigned int first, unsigned int n) const
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

isl::qpolynomial qpolynomial::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_qpolynomial_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl::qpolynomial qpolynomial::mul(isl::qpolynomial qp2) const
{
  auto res = isl_qpolynomial_mul(copy(), qp2.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::nan_on_domain(isl::space domain)
{
  auto res = isl_qpolynomial_nan_on_domain(domain.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::neg() const
{
  auto res = isl_qpolynomial_neg(copy());
  return manage(res);
}

isl::qpolynomial qpolynomial::neginfty_on_domain(isl::space domain)
{
  auto res = isl_qpolynomial_neginfty_on_domain(domain.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::one_on_domain(isl::space domain)
{
  auto res = isl_qpolynomial_one_on_domain(domain.release());
  return manage(res);
}

boolean qpolynomial::plain_is_equal(const isl::qpolynomial &qp2) const
{
  auto res = isl_qpolynomial_plain_is_equal(get(), qp2.get());
  return manage(res);
}

isl::qpolynomial qpolynomial::pow(unsigned int power) const
{
  auto res = isl_qpolynomial_pow(copy(), power);
  return manage(res);
}

isl::qpolynomial qpolynomial::project_domain_on_params() const
{
  auto res = isl_qpolynomial_project_domain_on_params(copy());
  return manage(res);
}

isl::qpolynomial qpolynomial::scale_down_val(isl::val v) const
{
  auto res = isl_qpolynomial_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::scale_val(isl::val v) const
{
  auto res = isl_qpolynomial_scale_val(copy(), v.release());
  return manage(res);
}

int qpolynomial::sgn() const
{
  auto res = isl_qpolynomial_sgn(get());
  return res;
}

isl::qpolynomial qpolynomial::sub(isl::qpolynomial qp2) const
{
  auto res = isl_qpolynomial_sub(copy(), qp2.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::val_on_domain(isl::space space, isl::val val)
{
  auto res = isl_qpolynomial_val_on_domain(space.release(), val.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::var_on_domain(isl::space domain, isl::dim type, unsigned int pos)
{
  auto res = isl_qpolynomial_var_on_domain(domain.release(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::qpolynomial qpolynomial::zero_on_domain(isl::space domain)
{
  auto res = isl_qpolynomial_zero_on_domain(domain.release());
  return manage(res);
}

// implementations for isl::qpolynomial_list
qpolynomial_list manage(__isl_take isl_qpolynomial_list *ptr) {
  return qpolynomial_list(ptr);
}
qpolynomial_list manage_copy(__isl_keep isl_qpolynomial_list *ptr) {
  ptr = isl_qpolynomial_list_copy(ptr);
  return qpolynomial_list(ptr);
}

qpolynomial_list::qpolynomial_list()
    : ptr(nullptr) {}

qpolynomial_list::qpolynomial_list(const qpolynomial_list &obj)
    : ptr(nullptr)
{
  ptr = obj.copy();
}


qpolynomial_list::qpolynomial_list(__isl_take isl_qpolynomial_list *ptr)
    : ptr(ptr) {}


qpolynomial_list &qpolynomial_list::operator=(qpolynomial_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

qpolynomial_list::~qpolynomial_list() {
  if (ptr)
    isl_qpolynomial_list_free(ptr);
}

__isl_give isl_qpolynomial_list *qpolynomial_list::copy() const & {
  return isl_qpolynomial_list_copy(ptr);
}

__isl_keep isl_qpolynomial_list *qpolynomial_list::get() const {
  return ptr;
}

__isl_give isl_qpolynomial_list *qpolynomial_list::release() {
  isl_qpolynomial_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool qpolynomial_list::is_null() const {
  return ptr == nullptr;
}


isl::ctx qpolynomial_list::ctx() const {
  return isl::ctx(isl_qpolynomial_list_get_ctx(ptr));
}

void qpolynomial_list::dump() const {
  isl_qpolynomial_list_dump(get());
}


isl::qpolynomial_list qpolynomial_list::add(isl::qpolynomial el) const
{
  auto res = isl_qpolynomial_list_add(copy(), el.release());
  return manage(res);
}

isl::qpolynomial_list qpolynomial_list::alloc(isl::ctx ctx, int n)
{
  auto res = isl_qpolynomial_list_alloc(ctx.release(), n);
  return manage(res);
}

isl::qpolynomial_list qpolynomial_list::clear() const
{
  auto res = isl_qpolynomial_list_clear(copy());
  return manage(res);
}

isl::qpolynomial_list qpolynomial_list::concat(isl::qpolynomial_list list2) const
{
  auto res = isl_qpolynomial_list_concat(copy(), list2.release());
  return manage(res);
}

isl::qpolynomial_list qpolynomial_list::drop(unsigned int first, unsigned int n) const
{
  auto res = isl_qpolynomial_list_drop(copy(), first, n);
  return manage(res);
}

stat qpolynomial_list::foreach(const std::function<stat(qpolynomial)> &fn) const
{
  struct fn_data {
    const std::function<stat(qpolynomial)> *func;
  } fn_data = { &fn };
  auto fn_lambda = [](isl_qpolynomial *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    stat ret = (*data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_qpolynomial_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

isl::qpolynomial_list qpolynomial_list::from_qpolynomial(isl::qpolynomial el)
{
  auto res = isl_qpolynomial_list_from_qpolynomial(el.release());
  return manage(res);
}

isl::qpolynomial qpolynomial_list::get_at(int index) const
{
  auto res = isl_qpolynomial_list_get_at(get(), index);
  return manage(res);
}

isl::qpolynomial qpolynomial_list::get_qpolynomial(int index) const
{
  auto res = isl_qpolynomial_list_get_qpolynomial(get(), index);
  return manage(res);
}

isl::qpolynomial_list qpolynomial_list::insert(unsigned int pos, isl::qpolynomial el) const
{
  auto res = isl_qpolynomial_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl_size qpolynomial_list::n_qpolynomial() const
{
  auto res = isl_qpolynomial_list_n_qpolynomial(get());
  return res;
}

isl::qpolynomial_list qpolynomial_list::reverse() const
{
  auto res = isl_qpolynomial_list_reverse(copy());
  return manage(res);
}

isl::qpolynomial_list qpolynomial_list::set_qpolynomial(int index, isl::qpolynomial el) const
{
  auto res = isl_qpolynomial_list_set_qpolynomial(copy(), index, el.release());
  return manage(res);
}

isl_size qpolynomial_list::size() const
{
  auto res = isl_qpolynomial_list_size(get());
  return res;
}

isl::qpolynomial_list qpolynomial_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_qpolynomial_list_swap(copy(), pos1, pos2);
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


schedule::schedule(__isl_take isl_schedule *ptr)
    : ptr(ptr) {}

schedule::schedule(isl::ctx ctx, const std::string &str)
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


isl::ctx schedule::ctx() const {
  return isl::ctx(isl_schedule_get_ctx(ptr));
}

void schedule::dump() const {
  isl_schedule_dump(get());
}


isl::schedule schedule::align_params(isl::space space) const
{
  auto res = isl_schedule_align_params(copy(), space.release());
  return manage(res);
}

isl::schedule schedule::empty(isl::space space)
{
  auto res = isl_schedule_empty(space.release());
  return manage(res);
}

isl::schedule schedule::from_domain(isl::union_set domain)
{
  auto res = isl_schedule_from_domain(domain.release());
  return manage(res);
}

isl::union_set schedule::get_domain() const
{
  auto res = isl_schedule_get_domain(get());
  return manage(res);
}

isl::union_map schedule::get_map() const
{
  auto res = isl_schedule_get_map(get());
  return manage(res);
}

isl::schedule_node schedule::get_root() const
{
  auto res = isl_schedule_get_root(get());
  return manage(res);
}

isl::schedule schedule::gist_domain_params(isl::set context) const
{
  auto res = isl_schedule_gist_domain_params(copy(), context.release());
  return manage(res);
}

isl::schedule schedule::insert_context(isl::set context) const
{
  auto res = isl_schedule_insert_context(copy(), context.release());
  return manage(res);
}

isl::schedule schedule::insert_guard(isl::set guard) const
{
  auto res = isl_schedule_insert_guard(copy(), guard.release());
  return manage(res);
}

isl::schedule schedule::insert_partial_schedule(isl::multi_union_pw_aff partial) const
{
  auto res = isl_schedule_insert_partial_schedule(copy(), partial.release());
  return manage(res);
}

isl::schedule schedule::intersect_domain(isl::union_set domain) const
{
  auto res = isl_schedule_intersect_domain(copy(), domain.release());
  return manage(res);
}

boolean schedule::plain_is_equal(const isl::schedule &schedule2) const
{
  auto res = isl_schedule_plain_is_equal(get(), schedule2.get());
  return manage(res);
}

isl::schedule schedule::pullback(isl::union_pw_multi_aff upma) const
{
  auto res = isl_schedule_pullback_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

isl::schedule schedule::reset_user() const
{
  auto res = isl_schedule_reset_user(copy());
  return manage(res);
}

isl::schedule schedule::sequence(isl::schedule schedule2) const
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


schedule_constraints::schedule_constraints(__isl_take isl_schedule_constraints *ptr)
    : ptr(ptr) {}

schedule_constraints::schedule_constraints(isl::ctx ctx, const std::string &str)
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


isl::ctx schedule_constraints::ctx() const {
  return isl::ctx(isl_schedule_constraints_get_ctx(ptr));
}

void schedule_constraints::dump() const {
  isl_schedule_constraints_dump(get());
}


isl::schedule_constraints schedule_constraints::apply(isl::union_map umap) const
{
  auto res = isl_schedule_constraints_apply(copy(), umap.release());
  return manage(res);
}

isl::schedule schedule_constraints::compute_schedule() const
{
  auto res = isl_schedule_constraints_compute_schedule(copy());
  return manage(res);
}

isl::union_map schedule_constraints::get_coincidence() const
{
  auto res = isl_schedule_constraints_get_coincidence(get());
  return manage(res);
}

isl::union_map schedule_constraints::get_conditional_validity() const
{
  auto res = isl_schedule_constraints_get_conditional_validity(get());
  return manage(res);
}

isl::union_map schedule_constraints::get_conditional_validity_condition() const
{
  auto res = isl_schedule_constraints_get_conditional_validity_condition(get());
  return manage(res);
}

isl::set schedule_constraints::get_context() const
{
  auto res = isl_schedule_constraints_get_context(get());
  return manage(res);
}

isl::union_set schedule_constraints::get_domain() const
{
  auto res = isl_schedule_constraints_get_domain(get());
  return manage(res);
}

isl::union_map schedule_constraints::get_proximity() const
{
  auto res = isl_schedule_constraints_get_proximity(get());
  return manage(res);
}

isl::union_map schedule_constraints::get_validity() const
{
  auto res = isl_schedule_constraints_get_validity(get());
  return manage(res);
}

isl::schedule_constraints schedule_constraints::on_domain(isl::union_set domain)
{
  auto res = isl_schedule_constraints_on_domain(domain.release());
  return manage(res);
}

isl::schedule_constraints schedule_constraints::set_coincidence(isl::union_map coincidence) const
{
  auto res = isl_schedule_constraints_set_coincidence(copy(), coincidence.release());
  return manage(res);
}

isl::schedule_constraints schedule_constraints::set_conditional_validity(isl::union_map condition, isl::union_map validity) const
{
  auto res = isl_schedule_constraints_set_conditional_validity(copy(), condition.release(), validity.release());
  return manage(res);
}

isl::schedule_constraints schedule_constraints::set_context(isl::set context) const
{
  auto res = isl_schedule_constraints_set_context(copy(), context.release());
  return manage(res);
}

isl::schedule_constraints schedule_constraints::set_proximity(isl::union_map proximity) const
{
  auto res = isl_schedule_constraints_set_proximity(copy(), proximity.release());
  return manage(res);
}

isl::schedule_constraints schedule_constraints::set_validity(isl::union_map validity) const
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


isl::ctx schedule_node::ctx() const {
  return isl::ctx(isl_schedule_node_get_ctx(ptr));
}

void schedule_node::dump() const {
  isl_schedule_node_dump(get());
}


isl::schedule_node schedule_node::align_params(isl::space space) const
{
  auto res = isl_schedule_node_align_params(copy(), space.release());
  return manage(res);
}

isl::schedule_node schedule_node::ancestor(int generation) const
{
  auto res = isl_schedule_node_ancestor(copy(), generation);
  return manage(res);
}

boolean schedule_node::band_member_get_coincident(int pos) const
{
  auto res = isl_schedule_node_band_member_get_coincident(get(), pos);
  return manage(res);
}

isl::schedule_node schedule_node::band_member_set_coincident(int pos, int coincident) const
{
  auto res = isl_schedule_node_band_member_set_coincident(copy(), pos, coincident);
  return manage(res);
}

isl::schedule_node schedule_node::band_set_ast_build_options(isl::union_set options) const
{
  auto res = isl_schedule_node_band_set_ast_build_options(copy(), options.release());
  return manage(res);
}

isl::schedule_node schedule_node::child(int pos) const
{
  auto res = isl_schedule_node_child(copy(), pos);
  return manage(res);
}

isl::set schedule_node::context_get_context() const
{
  auto res = isl_schedule_node_context_get_context(get());
  return manage(res);
}

isl::schedule_node schedule_node::cut() const
{
  auto res = isl_schedule_node_cut(copy());
  return manage(res);
}

isl::union_set schedule_node::domain_get_domain() const
{
  auto res = isl_schedule_node_domain_get_domain(get());
  return manage(res);
}

isl::union_pw_multi_aff schedule_node::expansion_get_contraction() const
{
  auto res = isl_schedule_node_expansion_get_contraction(get());
  return manage(res);
}

isl::union_map schedule_node::expansion_get_expansion() const
{
  auto res = isl_schedule_node_expansion_get_expansion(get());
  return manage(res);
}

isl::union_map schedule_node::extension_get_extension() const
{
  auto res = isl_schedule_node_extension_get_extension(get());
  return manage(res);
}

isl::union_set schedule_node::filter_get_filter() const
{
  auto res = isl_schedule_node_filter_get_filter(get());
  return manage(res);
}

isl::schedule_node schedule_node::first_child() const
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

isl::schedule_node schedule_node::from_domain(isl::union_set domain)
{
  auto res = isl_schedule_node_from_domain(domain.release());
  return manage(res);
}

isl::schedule_node schedule_node::from_extension(isl::union_map extension)
{
  auto res = isl_schedule_node_from_extension(extension.release());
  return manage(res);
}

isl_size schedule_node::get_ancestor_child_position(const isl::schedule_node &ancestor) const
{
  auto res = isl_schedule_node_get_ancestor_child_position(get(), ancestor.get());
  return res;
}

isl::schedule_node schedule_node::get_child(int pos) const
{
  auto res = isl_schedule_node_get_child(get(), pos);
  return manage(res);
}

isl_size schedule_node::get_child_position() const
{
  auto res = isl_schedule_node_get_child_position(get());
  return res;
}

isl::union_set schedule_node::get_domain() const
{
  auto res = isl_schedule_node_get_domain(get());
  return manage(res);
}

isl::multi_union_pw_aff schedule_node::get_prefix_schedule_multi_union_pw_aff() const
{
  auto res = isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(get());
  return manage(res);
}

isl::union_map schedule_node::get_prefix_schedule_relation() const
{
  auto res = isl_schedule_node_get_prefix_schedule_relation(get());
  return manage(res);
}

isl::union_map schedule_node::get_prefix_schedule_union_map() const
{
  auto res = isl_schedule_node_get_prefix_schedule_union_map(get());
  return manage(res);
}

isl::union_pw_multi_aff schedule_node::get_prefix_schedule_union_pw_multi_aff() const
{
  auto res = isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(get());
  return manage(res);
}

isl::schedule schedule_node::get_schedule() const
{
  auto res = isl_schedule_node_get_schedule(get());
  return manage(res);
}

isl_size schedule_node::get_schedule_depth() const
{
  auto res = isl_schedule_node_get_schedule_depth(get());
  return res;
}

isl::schedule_node schedule_node::get_shared_ancestor(const isl::schedule_node &node2) const
{
  auto res = isl_schedule_node_get_shared_ancestor(get(), node2.get());
  return manage(res);
}

isl::union_pw_multi_aff schedule_node::get_subtree_contraction() const
{
  auto res = isl_schedule_node_get_subtree_contraction(get());
  return manage(res);
}

isl::union_map schedule_node::get_subtree_expansion() const
{
  auto res = isl_schedule_node_get_subtree_expansion(get());
  return manage(res);
}

isl::union_map schedule_node::get_subtree_schedule_union_map() const
{
  auto res = isl_schedule_node_get_subtree_schedule_union_map(get());
  return manage(res);
}

isl_size schedule_node::get_tree_depth() const
{
  auto res = isl_schedule_node_get_tree_depth(get());
  return res;
}

isl::union_set schedule_node::get_universe_domain() const
{
  auto res = isl_schedule_node_get_universe_domain(get());
  return manage(res);
}

isl::schedule_node schedule_node::graft_after(isl::schedule_node graft) const
{
  auto res = isl_schedule_node_graft_after(copy(), graft.release());
  return manage(res);
}

isl::schedule_node schedule_node::graft_before(isl::schedule_node graft) const
{
  auto res = isl_schedule_node_graft_before(copy(), graft.release());
  return manage(res);
}

isl::schedule_node schedule_node::group(isl::id group_id) const
{
  auto res = isl_schedule_node_group(copy(), group_id.release());
  return manage(res);
}

isl::set schedule_node::guard_get_guard() const
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

isl::schedule_node schedule_node::insert_context(isl::set context) const
{
  auto res = isl_schedule_node_insert_context(copy(), context.release());
  return manage(res);
}

isl::schedule_node schedule_node::insert_filter(isl::union_set filter) const
{
  auto res = isl_schedule_node_insert_filter(copy(), filter.release());
  return manage(res);
}

isl::schedule_node schedule_node::insert_guard(isl::set context) const
{
  auto res = isl_schedule_node_insert_guard(copy(), context.release());
  return manage(res);
}

isl::schedule_node schedule_node::insert_mark(isl::id mark) const
{
  auto res = isl_schedule_node_insert_mark(copy(), mark.release());
  return manage(res);
}

isl::schedule_node schedule_node::insert_partial_schedule(isl::multi_union_pw_aff schedule) const
{
  auto res = isl_schedule_node_insert_partial_schedule(copy(), schedule.release());
  return manage(res);
}

isl::schedule_node schedule_node::insert_sequence(isl::union_set_list filters) const
{
  auto res = isl_schedule_node_insert_sequence(copy(), filters.release());
  return manage(res);
}

isl::schedule_node schedule_node::insert_set(isl::union_set_list filters) const
{
  auto res = isl_schedule_node_insert_set(copy(), filters.release());
  return manage(res);
}

boolean schedule_node::is_equal(const isl::schedule_node &node2) const
{
  auto res = isl_schedule_node_is_equal(get(), node2.get());
  return manage(res);
}

boolean schedule_node::is_subtree_anchored() const
{
  auto res = isl_schedule_node_is_subtree_anchored(get());
  return manage(res);
}

isl::id schedule_node::mark_get_id() const
{
  auto res = isl_schedule_node_mark_get_id(get());
  return manage(res);
}

isl_size schedule_node::n_children() const
{
  auto res = isl_schedule_node_n_children(get());
  return res;
}

isl::schedule_node schedule_node::next_sibling() const
{
  auto res = isl_schedule_node_next_sibling(copy());
  return manage(res);
}

isl::schedule_node schedule_node::order_after(isl::union_set filter) const
{
  auto res = isl_schedule_node_order_after(copy(), filter.release());
  return manage(res);
}

isl::schedule_node schedule_node::order_before(isl::union_set filter) const
{
  auto res = isl_schedule_node_order_before(copy(), filter.release());
  return manage(res);
}

isl::schedule_node schedule_node::parent() const
{
  auto res = isl_schedule_node_parent(copy());
  return manage(res);
}

isl::schedule_node schedule_node::previous_sibling() const
{
  auto res = isl_schedule_node_previous_sibling(copy());
  return manage(res);
}

isl::schedule_node schedule_node::reset_user() const
{
  auto res = isl_schedule_node_reset_user(copy());
  return manage(res);
}

isl::schedule_node schedule_node::root() const
{
  auto res = isl_schedule_node_root(copy());
  return manage(res);
}

isl::schedule_node schedule_node::sequence_splice_child(int pos) const
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


set::set(__isl_take isl_set *ptr)
    : ptr(ptr) {}

set::set(isl::basic_set bset)
{
  auto res = isl_set_from_basic_set(bset.release());
  ptr = res;
}
set::set(isl::point pnt)
{
  auto res = isl_set_from_point(pnt.release());
  ptr = res;
}
set::set(isl::union_set uset)
{
  auto res = isl_set_from_union_set(uset.release());
  ptr = res;
}
set::set(isl::ctx ctx, const std::string &str)
{
  auto res = isl_set_read_from_str(ctx.release(), str.c_str());
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


isl::ctx set::ctx() const {
  return isl::ctx(isl_set_get_ctx(ptr));
}

void set::dump() const {
  isl_set_dump(get());
}


isl::set set::add_constraint(isl::constraint constraint) const
{
  auto res = isl_set_add_constraint(copy(), constraint.release());
  return manage(res);
}

isl::set set::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_set_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::basic_set set::affine_hull() const
{
  auto res = isl_set_affine_hull(copy());
  return manage(res);
}

isl::set set::align_params(isl::space model) const
{
  auto res = isl_set_align_params(copy(), model.release());
  return manage(res);
}

isl::set set::apply(isl::map map) const
{
  auto res = isl_set_apply(copy(), map.release());
  return manage(res);
}

isl::set set::bind(isl::multi_id tuple) const
{
  auto res = isl_set_bind(copy(), tuple.release());
  return manage(res);
}

isl::basic_set set::bounded_simple_hull() const
{
  auto res = isl_set_bounded_simple_hull(copy());
  return manage(res);
}

isl::set set::box_from_points(isl::point pnt1, isl::point pnt2)
{
  auto res = isl_set_box_from_points(pnt1.release(), pnt2.release());
  return manage(res);
}

isl::set set::coalesce() const
{
  auto res = isl_set_coalesce(copy());
  return manage(res);
}

isl::basic_set set::coefficients() const
{
  auto res = isl_set_coefficients(copy());
  return manage(res);
}

isl::set set::complement() const
{
  auto res = isl_set_complement(copy());
  return manage(res);
}

isl::basic_set set::convex_hull() const
{
  auto res = isl_set_convex_hull(copy());
  return manage(res);
}

isl::val set::count_val() const
{
  auto res = isl_set_count_val(get());
  return manage(res);
}

isl::set set::detect_equalities() const
{
  auto res = isl_set_detect_equalities(copy());
  return manage(res);
}

isl_size set::dim(isl::dim type) const
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

isl::pw_aff set::dim_max(int pos) const
{
  auto res = isl_set_dim_max(copy(), pos);
  return manage(res);
}

isl::val set::dim_max_val(int pos) const
{
  auto res = isl_set_dim_max_val(copy(), pos);
  return manage(res);
}

isl::pw_aff set::dim_min(int pos) const
{
  auto res = isl_set_dim_min(copy(), pos);
  return manage(res);
}

isl::val set::dim_min_val(int pos) const
{
  auto res = isl_set_dim_min_val(copy(), pos);
  return manage(res);
}

isl::set set::drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_drop_constraints_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::set set::drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_drop_constraints_not_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::set set::drop_unused_params() const
{
  auto res = isl_set_drop_unused_params(copy());
  return manage(res);
}

isl::set set::eliminate(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_eliminate(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::set set::empty(isl::space space)
{
  auto res = isl_set_empty(space.release());
  return manage(res);
}

isl::set set::equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_set_equate(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

int set::find_dim_by_id(isl::dim type, const isl::id &id) const
{
  auto res = isl_set_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int set::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_set_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::set set::fix_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_set_fix_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::set set::fix_val(isl::dim type, unsigned int pos, isl::val v) const
{
  auto res = isl_set_fix_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

isl::set set::flat_product(isl::set set2) const
{
  auto res = isl_set_flat_product(copy(), set2.release());
  return manage(res);
}

isl::set set::flatten() const
{
  auto res = isl_set_flatten(copy());
  return manage(res);
}

isl::map set::flatten_map() const
{
  auto res = isl_set_flatten_map(copy());
  return manage(res);
}

int set::follows_at(const isl::set &set2, int pos) const
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

isl::set set::from_multi_aff(isl::multi_aff ma)
{
  auto res = isl_set_from_multi_aff(ma.release());
  return manage(res);
}

isl::set set::from_multi_pw_aff(isl::multi_pw_aff mpa)
{
  auto res = isl_set_from_multi_pw_aff(mpa.release());
  return manage(res);
}

isl::set set::from_params() const
{
  auto res = isl_set_from_params(copy());
  return manage(res);
}

isl::set set::from_pw_aff(isl::pw_aff pwaff)
{
  auto res = isl_set_from_pw_aff(pwaff.release());
  return manage(res);
}

isl::set set::from_pw_multi_aff(isl::pw_multi_aff pma)
{
  auto res = isl_set_from_pw_multi_aff(pma.release());
  return manage(res);
}

isl::basic_set_list set::get_basic_set_list() const
{
  auto res = isl_set_get_basic_set_list(get());
  return manage(res);
}

isl::id set::get_dim_id(isl::dim type, unsigned int pos) const
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

isl::multi_val set::get_plain_multi_val_if_fixed() const
{
  auto res = isl_set_get_plain_multi_val_if_fixed(get());
  return manage(res);
}

isl::fixed_box set::get_simple_fixed_box_hull() const
{
  auto res = isl_set_get_simple_fixed_box_hull(get());
  return manage(res);
}

isl::space set::get_space() const
{
  auto res = isl_set_get_space(get());
  return manage(res);
}

isl::val set::get_stride(int pos) const
{
  auto res = isl_set_get_stride(get(), pos);
  return manage(res);
}

isl::id set::get_tuple_id() const
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

isl::set set::gist(isl::set context) const
{
  auto res = isl_set_gist(copy(), context.release());
  return manage(res);
}

isl::set set::gist_basic_set(isl::basic_set context) const
{
  auto res = isl_set_gist_basic_set(copy(), context.release());
  return manage(res);
}

isl::set set::gist_params(isl::set context) const
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

boolean set::has_equal_space(const isl::set &set2) const
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

isl::map set::identity() const
{
  auto res = isl_set_identity(copy());
  return manage(res);
}

isl::pw_aff set::indicator_function() const
{
  auto res = isl_set_indicator_function(copy());
  return manage(res);
}

isl::set set::insert_dims(isl::dim type, unsigned int pos, unsigned int n) const
{
  auto res = isl_set_insert_dims(copy(), static_cast<enum isl_dim_type>(type), pos, n);
  return manage(res);
}

isl::map set::insert_domain(isl::space domain) const
{
  auto res = isl_set_insert_domain(copy(), domain.release());
  return manage(res);
}

isl::set set::intersect(isl::set set2) const
{
  auto res = isl_set_intersect(copy(), set2.release());
  return manage(res);
}

isl::set set::intersect_factor_domain(isl::set domain) const
{
  auto res = isl_set_intersect_factor_domain(copy(), domain.release());
  return manage(res);
}

isl::set set::intersect_factor_range(isl::set range) const
{
  auto res = isl_set_intersect_factor_range(copy(), range.release());
  return manage(res);
}

isl::set set::intersect_params(isl::set params) const
{
  auto res = isl_set_intersect_params(copy(), params.release());
  return manage(res);
}

boolean set::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

boolean set::involves_locals() const
{
  auto res = isl_set_involves_locals(get());
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

boolean set::is_disjoint(const isl::set &set2) const
{
  auto res = isl_set_is_disjoint(get(), set2.get());
  return manage(res);
}

boolean set::is_empty() const
{
  auto res = isl_set_is_empty(get());
  return manage(res);
}

boolean set::is_equal(const isl::set &set2) const
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

boolean set::is_strict_subset(const isl::set &set2) const
{
  auto res = isl_set_is_strict_subset(get(), set2.get());
  return manage(res);
}

boolean set::is_subset(const isl::set &set2) const
{
  auto res = isl_set_is_subset(get(), set2.get());
  return manage(res);
}

boolean set::is_wrapping() const
{
  auto res = isl_set_is_wrapping(get());
  return manage(res);
}

isl::map set::lex_ge_set(isl::set set2) const
{
  auto res = isl_set_lex_ge_set(copy(), set2.release());
  return manage(res);
}

isl::map set::lex_gt_set(isl::set set2) const
{
  auto res = isl_set_lex_gt_set(copy(), set2.release());
  return manage(res);
}

isl::map set::lex_lt_set(isl::set set2) const
{
  auto res = isl_set_lex_lt_set(copy(), set2.release());
  return manage(res);
}

isl::set set::lexmax() const
{
  auto res = isl_set_lexmax(copy());
  return manage(res);
}

isl::pw_multi_aff set::lexmax_pw_multi_aff() const
{
  auto res = isl_set_lexmax_pw_multi_aff(copy());
  return manage(res);
}

isl::set set::lexmin() const
{
  auto res = isl_set_lexmin(copy());
  return manage(res);
}

isl::pw_multi_aff set::lexmin_pw_multi_aff() const
{
  auto res = isl_set_lexmin_pw_multi_aff(copy());
  return manage(res);
}

isl::set set::lower_bound(isl::multi_pw_aff lower) const
{
  auto res = isl_set_lower_bound_multi_pw_aff(copy(), lower.release());
  return manage(res);
}

isl::set set::lower_bound(isl::multi_val lower) const
{
  auto res = isl_set_lower_bound_multi_val(copy(), lower.release());
  return manage(res);
}

isl::set set::lower_bound_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_set_lower_bound_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::set set::lower_bound_val(isl::dim type, unsigned int pos, isl::val value) const
{
  auto res = isl_set_lower_bound_val(copy(), static_cast<enum isl_dim_type>(type), pos, value.release());
  return manage(res);
}

isl::multi_pw_aff set::max_multi_pw_aff() const
{
  auto res = isl_set_max_multi_pw_aff(copy());
  return manage(res);
}

isl::val set::max_val(const isl::aff &obj) const
{
  auto res = isl_set_max_val(get(), obj.get());
  return manage(res);
}

isl::multi_pw_aff set::min_multi_pw_aff() const
{
  auto res = isl_set_min_multi_pw_aff(copy());
  return manage(res);
}

isl::val set::min_val(const isl::aff &obj) const
{
  auto res = isl_set_min_val(get(), obj.get());
  return manage(res);
}

isl::set set::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_set_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl_size set::n_basic_set() const
{
  auto res = isl_set_n_basic_set(get());
  return res;
}

isl_size set::n_dim() const
{
  auto res = isl_set_n_dim(get());
  return res;
}

isl::set set::nat_universe(isl::space space)
{
  auto res = isl_set_nat_universe(space.release());
  return manage(res);
}

isl::set set::neg() const
{
  auto res = isl_set_neg(copy());
  return manage(res);
}

isl::set set::params() const
{
  auto res = isl_set_params(copy());
  return manage(res);
}

int set::plain_cmp(const isl::set &set2) const
{
  auto res = isl_set_plain_cmp(get(), set2.get());
  return res;
}

isl::val set::plain_get_val_if_fixed(isl::dim type, unsigned int pos) const
{
  auto res = isl_set_plain_get_val_if_fixed(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

boolean set::plain_is_disjoint(const isl::set &set2) const
{
  auto res = isl_set_plain_is_disjoint(get(), set2.get());
  return manage(res);
}

boolean set::plain_is_empty() const
{
  auto res = isl_set_plain_is_empty(get());
  return manage(res);
}

boolean set::plain_is_equal(const isl::set &set2) const
{
  auto res = isl_set_plain_is_equal(get(), set2.get());
  return manage(res);
}

boolean set::plain_is_universe() const
{
  auto res = isl_set_plain_is_universe(get());
  return manage(res);
}

isl::basic_set set::plain_unshifted_simple_hull() const
{
  auto res = isl_set_plain_unshifted_simple_hull(copy());
  return manage(res);
}

isl::basic_set set::polyhedral_hull() const
{
  auto res = isl_set_polyhedral_hull(copy());
  return manage(res);
}

isl::set set::preimage(isl::multi_aff ma) const
{
  auto res = isl_set_preimage_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::set set::preimage(isl::multi_pw_aff mpa) const
{
  auto res = isl_set_preimage_multi_pw_aff(copy(), mpa.release());
  return manage(res);
}

isl::set set::preimage(isl::pw_multi_aff pma) const
{
  auto res = isl_set_preimage_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::set set::product(isl::set set2) const
{
  auto res = isl_set_product(copy(), set2.release());
  return manage(res);
}

isl::map set::project_onto_map(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_project_onto_map(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::set set::project_out(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::set set::project_out_all_params() const
{
  auto res = isl_set_project_out_all_params(copy());
  return manage(res);
}

isl::set set::project_out_param(isl::id id) const
{
  auto res = isl_set_project_out_param_id(copy(), id.release());
  return manage(res);
}

isl::set set::project_out_param(isl::id_list list) const
{
  auto res = isl_set_project_out_param_id_list(copy(), list.release());
  return manage(res);
}

isl::set set::remove_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_remove_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::set set::remove_divs() const
{
  auto res = isl_set_remove_divs(copy());
  return manage(res);
}

isl::set set::remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_remove_divs_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::set set::remove_redundancies() const
{
  auto res = isl_set_remove_redundancies(copy());
  return manage(res);
}

isl::set set::remove_unknown_divs() const
{
  auto res = isl_set_remove_unknown_divs(copy());
  return manage(res);
}

isl::set set::reset_space(isl::space space) const
{
  auto res = isl_set_reset_space(copy(), space.release());
  return manage(res);
}

isl::set set::reset_tuple_id() const
{
  auto res = isl_set_reset_tuple_id(copy());
  return manage(res);
}

isl::set set::reset_user() const
{
  auto res = isl_set_reset_user(copy());
  return manage(res);
}

isl::basic_set set::sample() const
{
  auto res = isl_set_sample(copy());
  return manage(res);
}

isl::point set::sample_point() const
{
  auto res = isl_set_sample_point(copy());
  return manage(res);
}

isl::set set::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const
{
  auto res = isl_set_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::set set::set_tuple_id(isl::id id) const
{
  auto res = isl_set_set_tuple_id(copy(), id.release());
  return manage(res);
}

isl::set set::set_tuple_name(const std::string &s) const
{
  auto res = isl_set_set_tuple_name(copy(), s.c_str());
  return manage(res);
}

isl::basic_set set::simple_hull() const
{
  auto res = isl_set_simple_hull(copy());
  return manage(res);
}

int set::size() const
{
  auto res = isl_set_size(get());
  return res;
}

isl::basic_set set::solutions() const
{
  auto res = isl_set_solutions(copy());
  return manage(res);
}

isl::set set::split_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_split_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::set set::subtract(isl::set set2) const
{
  auto res = isl_set_subtract(copy(), set2.release());
  return manage(res);
}

isl::set set::sum(isl::set set2) const
{
  auto res = isl_set_sum(copy(), set2.release());
  return manage(res);
}

isl::map set::translation() const
{
  auto res = isl_set_translation(copy());
  return manage(res);
}

isl_size set::tuple_dim() const
{
  auto res = isl_set_tuple_dim(get());
  return res;
}

isl::set set::unbind_params(isl::multi_id tuple) const
{
  auto res = isl_set_unbind_params(copy(), tuple.release());
  return manage(res);
}

isl::map set::unbind_params_insert_domain(isl::multi_id domain) const
{
  auto res = isl_set_unbind_params_insert_domain(copy(), domain.release());
  return manage(res);
}

isl::set set::unite(isl::set set2) const
{
  auto res = isl_set_union(copy(), set2.release());
  return manage(res);
}

isl::set set::universe(isl::space space)
{
  auto res = isl_set_universe(space.release());
  return manage(res);
}

isl::basic_set set::unshifted_simple_hull() const
{
  auto res = isl_set_unshifted_simple_hull(copy());
  return manage(res);
}

isl::basic_set set::unshifted_simple_hull_from_set_list(isl::set_list list) const
{
  auto res = isl_set_unshifted_simple_hull_from_set_list(copy(), list.release());
  return manage(res);
}

isl::map set::unwrap() const
{
  auto res = isl_set_unwrap(copy());
  return manage(res);
}

isl::set set::upper_bound(isl::multi_pw_aff upper) const
{
  auto res = isl_set_upper_bound_multi_pw_aff(copy(), upper.release());
  return manage(res);
}

isl::set set::upper_bound(isl::multi_val upper) const
{
  auto res = isl_set_upper_bound_multi_val(copy(), upper.release());
  return manage(res);
}

isl::set set::upper_bound_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_set_upper_bound_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::set set::upper_bound_val(isl::dim type, unsigned int pos, isl::val value) const
{
  auto res = isl_set_upper_bound_val(copy(), static_cast<enum isl_dim_type>(type), pos, value.release());
  return manage(res);
}

isl::map set::wrapped_domain_map() const
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


isl::ctx set_list::ctx() const {
  return isl::ctx(isl_set_list_get_ctx(ptr));
}

void set_list::dump() const {
  isl_set_list_dump(get());
}


isl::set_list set_list::add(isl::set el) const
{
  auto res = isl_set_list_add(copy(), el.release());
  return manage(res);
}

isl::set_list set_list::alloc(isl::ctx ctx, int n)
{
  auto res = isl_set_list_alloc(ctx.release(), n);
  return manage(res);
}

isl::set_list set_list::clear() const
{
  auto res = isl_set_list_clear(copy());
  return manage(res);
}

isl::set_list set_list::concat(isl::set_list list2) const
{
  auto res = isl_set_list_concat(copy(), list2.release());
  return manage(res);
}

isl::set_list set_list::drop(unsigned int first, unsigned int n) const
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

isl::set_list set_list::from_set(isl::set el)
{
  auto res = isl_set_list_from_set(el.release());
  return manage(res);
}

isl::set set_list::get_at(int index) const
{
  auto res = isl_set_list_get_at(get(), index);
  return manage(res);
}

isl::set set_list::get_set(int index) const
{
  auto res = isl_set_list_get_set(get(), index);
  return manage(res);
}

isl::set_list set_list::insert(unsigned int pos, isl::set el) const
{
  auto res = isl_set_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl_size set_list::n_set() const
{
  auto res = isl_set_list_n_set(get());
  return res;
}

isl::set_list set_list::reverse() const
{
  auto res = isl_set_list_reverse(copy());
  return manage(res);
}

isl::set_list set_list::set_set(int index, isl::set el) const
{
  auto res = isl_set_list_set_set(copy(), index, el.release());
  return manage(res);
}

isl_size set_list::size() const
{
  auto res = isl_set_list_size(get());
  return res;
}

isl::set_list set_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_set_list_swap(copy(), pos1, pos2);
  return manage(res);
}

isl::set set_list::unite() const
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


space::space(__isl_take isl_space *ptr)
    : ptr(ptr) {}

space::space(isl::ctx ctx, unsigned int nparam, unsigned int n_in, unsigned int n_out)
{
  auto res = isl_space_alloc(ctx.release(), nparam, n_in, n_out);
  ptr = res;
}
space::space(isl::ctx ctx, unsigned int nparam, unsigned int dim)
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


isl::ctx space::ctx() const {
  return isl::ctx(isl_space_get_ctx(ptr));
}

void space::dump() const {
  isl_space_dump(get());
}


isl::space space::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_space_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::space space::add_named_tuple(isl::id tuple_id, unsigned int dim) const
{
  auto res = isl_space_add_named_tuple_id_ui(copy(), tuple_id.release(), dim);
  return manage(res);
}

isl::space space::add_param_id(isl::id id) const
{
  auto res = isl_space_add_param_id(copy(), id.release());
  return manage(res);
}

isl::space space::add_unnamed_tuple(unsigned int dim) const
{
  auto res = isl_space_add_unnamed_tuple_ui(copy(), dim);
  return manage(res);
}

isl::space space::align_params(isl::space space2) const
{
  auto res = isl_space_align_params(copy(), space2.release());
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

isl::space space::curry() const
{
  auto res = isl_space_curry(copy());
  return manage(res);
}

isl_size space::dim(isl::dim type) const
{
  auto res = isl_space_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::space space::domain() const
{
  auto res = isl_space_domain(copy());
  return manage(res);
}

isl::space space::domain_factor_domain() const
{
  auto res = isl_space_domain_factor_domain(copy());
  return manage(res);
}

isl::space space::domain_factor_range() const
{
  auto res = isl_space_domain_factor_range(copy());
  return manage(res);
}

boolean space::domain_is_wrapping() const
{
  auto res = isl_space_domain_is_wrapping(get());
  return manage(res);
}

isl::space space::domain_map() const
{
  auto res = isl_space_domain_map(copy());
  return manage(res);
}

isl::space space::domain_product(isl::space right) const
{
  auto res = isl_space_domain_product(copy(), right.release());
  return manage(res);
}

isl::space space::drop_all_params() const
{
  auto res = isl_space_drop_all_params(copy());
  return manage(res);
}

isl::space space::drop_dims(isl::dim type, unsigned int first, unsigned int num) const
{
  auto res = isl_space_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, num);
  return manage(res);
}

isl::space space::factor_domain() const
{
  auto res = isl_space_factor_domain(copy());
  return manage(res);
}

isl::space space::factor_range() const
{
  auto res = isl_space_factor_range(copy());
  return manage(res);
}

int space::find_dim_by_id(isl::dim type, const isl::id &id) const
{
  auto res = isl_space_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int space::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_space_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::space space::flatten_domain() const
{
  auto res = isl_space_flatten_domain(copy());
  return manage(res);
}

isl::space space::flatten_range() const
{
  auto res = isl_space_flatten_range(copy());
  return manage(res);
}

isl::space space::from_domain() const
{
  auto res = isl_space_from_domain(copy());
  return manage(res);
}

isl::space space::from_range() const
{
  auto res = isl_space_from_range(copy());
  return manage(res);
}

isl::id space::get_dim_id(isl::dim type, unsigned int pos) const
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

isl::id space::get_tuple_id(isl::dim type) const
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

boolean space::has_equal_params(const isl::space &space2) const
{
  auto res = isl_space_has_equal_params(get(), space2.get());
  return manage(res);
}

boolean space::has_equal_tuples(const isl::space &space2) const
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

isl::space space::insert_dims(isl::dim type, unsigned int pos, unsigned int n) const
{
  auto res = isl_space_insert_dims(copy(), static_cast<enum isl_dim_type>(type), pos, n);
  return manage(res);
}

boolean space::is_domain(const isl::space &space2) const
{
  auto res = isl_space_is_domain(get(), space2.get());
  return manage(res);
}

boolean space::is_equal(const isl::space &space2) const
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

boolean space::is_range(const isl::space &space2) const
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

isl::space space::join(isl::space right) const
{
  auto res = isl_space_join(copy(), right.release());
  return manage(res);
}

isl::space space::map_from_domain_and_range(isl::space range) const
{
  auto res = isl_space_map_from_domain_and_range(copy(), range.release());
  return manage(res);
}

isl::space space::map_from_set() const
{
  auto res = isl_space_map_from_set(copy());
  return manage(res);
}

isl::space space::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  auto res = isl_space_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl::space space::params() const
{
  auto res = isl_space_params(copy());
  return manage(res);
}

isl::space space::params_alloc(isl::ctx ctx, unsigned int nparam)
{
  auto res = isl_space_params_alloc(ctx.release(), nparam);
  return manage(res);
}

isl::space space::product(isl::space right) const
{
  auto res = isl_space_product(copy(), right.release());
  return manage(res);
}

isl::space space::range() const
{
  auto res = isl_space_range(copy());
  return manage(res);
}

isl::space space::range_curry() const
{
  auto res = isl_space_range_curry(copy());
  return manage(res);
}

isl::space space::range_factor_domain() const
{
  auto res = isl_space_range_factor_domain(copy());
  return manage(res);
}

isl::space space::range_factor_range() const
{
  auto res = isl_space_range_factor_range(copy());
  return manage(res);
}

boolean space::range_is_wrapping() const
{
  auto res = isl_space_range_is_wrapping(get());
  return manage(res);
}

isl::space space::range_map() const
{
  auto res = isl_space_range_map(copy());
  return manage(res);
}

isl::space space::range_product(isl::space right) const
{
  auto res = isl_space_range_product(copy(), right.release());
  return manage(res);
}

isl::space space::range_reverse() const
{
  auto res = isl_space_range_reverse(copy());
  return manage(res);
}

isl::space space::reset_tuple_id(isl::dim type) const
{
  auto res = isl_space_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::space space::reset_user() const
{
  auto res = isl_space_reset_user(copy());
  return manage(res);
}

isl::space space::reverse() const
{
  auto res = isl_space_reverse(copy());
  return manage(res);
}

isl::space space::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const
{
  auto res = isl_space_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::space space::set_from_params() const
{
  auto res = isl_space_set_from_params(copy());
  return manage(res);
}

isl::space space::set_tuple_id(isl::dim type, isl::id id) const
{
  auto res = isl_space_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::space space::set_tuple_name(isl::dim type, const std::string &s) const
{
  auto res = isl_space_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

boolean space::tuple_is_equal(isl::dim type1, const isl::space &space2, isl::dim type2) const
{
  auto res = isl_space_tuple_is_equal(get(), static_cast<enum isl_dim_type>(type1), space2.get(), static_cast<enum isl_dim_type>(type2));
  return manage(res);
}

isl::space space::uncurry() const
{
  auto res = isl_space_uncurry(copy());
  return manage(res);
}

isl::space space::unit(isl::ctx ctx)
{
  auto res = isl_space_unit(ctx.release());
  return manage(res);
}

isl::space space::unwrap() const
{
  auto res = isl_space_unwrap(copy());
  return manage(res);
}

isl::space space::wrap() const
{
  auto res = isl_space_wrap(copy());
  return manage(res);
}

isl::space space::zip() const
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


isl::ctx term::ctx() const {
  return isl::ctx(isl_term_get_ctx(ptr));
}


isl_size term::dim(isl::dim type) const
{
  auto res = isl_term_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::val term::get_coefficient_val() const
{
  auto res = isl_term_get_coefficient_val(get());
  return manage(res);
}

isl::aff term::get_div(unsigned int pos) const
{
  auto res = isl_term_get_div(get(), pos);
  return manage(res);
}

isl_size term::get_exp(isl::dim type, unsigned int pos) const
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


union_access_info::union_access_info(__isl_take isl_union_access_info *ptr)
    : ptr(ptr) {}

union_access_info::union_access_info(isl::union_map sink)
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


isl::ctx union_access_info::ctx() const {
  return isl::ctx(isl_union_access_info_get_ctx(ptr));
}


isl::union_flow union_access_info::compute_flow() const
{
  auto res = isl_union_access_info_compute_flow(copy());
  return manage(res);
}

isl::union_access_info union_access_info::set_kill(isl::union_map kill) const
{
  auto res = isl_union_access_info_set_kill(copy(), kill.release());
  return manage(res);
}

isl::union_access_info union_access_info::set_may_source(isl::union_map may_source) const
{
  auto res = isl_union_access_info_set_may_source(copy(), may_source.release());
  return manage(res);
}

isl::union_access_info union_access_info::set_must_source(isl::union_map must_source) const
{
  auto res = isl_union_access_info_set_must_source(copy(), must_source.release());
  return manage(res);
}

isl::union_access_info union_access_info::set_schedule(isl::schedule schedule) const
{
  auto res = isl_union_access_info_set_schedule(copy(), schedule.release());
  return manage(res);
}

isl::union_access_info union_access_info::set_schedule_map(isl::union_map schedule_map) const
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


isl::ctx union_flow::ctx() const {
  return isl::ctx(isl_union_flow_get_ctx(ptr));
}


isl::union_map union_flow::get_full_may_dependence() const
{
  auto res = isl_union_flow_get_full_may_dependence(get());
  return manage(res);
}

isl::union_map union_flow::get_full_must_dependence() const
{
  auto res = isl_union_flow_get_full_must_dependence(get());
  return manage(res);
}

isl::union_map union_flow::get_may_dependence() const
{
  auto res = isl_union_flow_get_may_dependence(get());
  return manage(res);
}

isl::union_map union_flow::get_may_no_source() const
{
  auto res = isl_union_flow_get_may_no_source(get());
  return manage(res);
}

isl::union_map union_flow::get_must_dependence() const
{
  auto res = isl_union_flow_get_must_dependence(get());
  return manage(res);
}

isl::union_map union_flow::get_must_no_source() const
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


union_map::union_map(__isl_take isl_union_map *ptr)
    : ptr(ptr) {}

union_map::union_map(isl::basic_map bmap)
{
  auto res = isl_union_map_from_basic_map(bmap.release());
  ptr = res;
}
union_map::union_map(isl::map map)
{
  auto res = isl_union_map_from_map(map.release());
  ptr = res;
}
union_map::union_map(isl::union_pw_multi_aff upma)
{
  auto res = isl_union_map_from_union_pw_multi_aff(upma.release());
  ptr = res;
}
union_map::union_map(isl::ctx ctx, const std::string &str)
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


isl::ctx union_map::ctx() const {
  return isl::ctx(isl_union_map_get_ctx(ptr));
}

void union_map::dump() const {
  isl_union_map_dump(get());
}


isl::union_map union_map::affine_hull() const
{
  auto res = isl_union_map_affine_hull(copy());
  return manage(res);
}

isl::union_map union_map::align_params(isl::space model) const
{
  auto res = isl_union_map_align_params(copy(), model.release());
  return manage(res);
}

isl::union_map union_map::apply_domain(isl::union_map umap2) const
{
  auto res = isl_union_map_apply_domain(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::apply_range(isl::union_map umap2) const
{
  auto res = isl_union_map_apply_range(copy(), umap2.release());
  return manage(res);
}

isl::union_set union_map::bind_range(isl::multi_id tuple) const
{
  auto res = isl_union_map_bind_range(copy(), tuple.release());
  return manage(res);
}

isl::union_map union_map::coalesce() const
{
  auto res = isl_union_map_coalesce(copy());
  return manage(res);
}

boolean union_map::contains(const isl::space &space) const
{
  auto res = isl_union_map_contains(get(), space.get());
  return manage(res);
}

isl::union_map union_map::curry() const
{
  auto res = isl_union_map_curry(copy());
  return manage(res);
}

isl::union_set union_map::deltas() const
{
  auto res = isl_union_map_deltas(copy());
  return manage(res);
}

isl::union_map union_map::deltas_map() const
{
  auto res = isl_union_map_deltas_map(copy());
  return manage(res);
}

isl::union_map union_map::detect_equalities() const
{
  auto res = isl_union_map_detect_equalities(copy());
  return manage(res);
}

isl_size union_map::dim(isl::dim type) const
{
  auto res = isl_union_map_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::union_set union_map::domain() const
{
  auto res = isl_union_map_domain(copy());
  return manage(res);
}

isl::union_map union_map::domain_factor_domain() const
{
  auto res = isl_union_map_domain_factor_domain(copy());
  return manage(res);
}

isl::union_map union_map::domain_factor_range() const
{
  auto res = isl_union_map_domain_factor_range(copy());
  return manage(res);
}

isl::union_map union_map::domain_map() const
{
  auto res = isl_union_map_domain_map(copy());
  return manage(res);
}

isl::union_pw_multi_aff union_map::domain_map_union_pw_multi_aff() const
{
  auto res = isl_union_map_domain_map_union_pw_multi_aff(copy());
  return manage(res);
}

isl::union_map union_map::domain_product(isl::union_map umap2) const
{
  auto res = isl_union_map_domain_product(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::empty(isl::ctx ctx)
{
  auto res = isl_union_map_empty_ctx(ctx.release());
  return manage(res);
}

isl::union_map union_map::eq_at(isl::multi_union_pw_aff mupa) const
{
  auto res = isl_union_map_eq_at_multi_union_pw_aff(copy(), mupa.release());
  return manage(res);
}

isl::map union_map::extract_map(isl::space space) const
{
  auto res = isl_union_map_extract_map(get(), space.release());
  return manage(res);
}

isl::union_map union_map::factor_domain() const
{
  auto res = isl_union_map_factor_domain(copy());
  return manage(res);
}

isl::union_map union_map::factor_range() const
{
  auto res = isl_union_map_factor_range(copy());
  return manage(res);
}

int union_map::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_union_map_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::union_map union_map::fixed_power(isl::val exp) const
{
  auto res = isl_union_map_fixed_power_val(copy(), exp.release());
  return manage(res);
}

isl::union_map union_map::flat_domain_product(isl::union_map umap2) const
{
  auto res = isl_union_map_flat_domain_product(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::flat_range_product(isl::union_map umap2) const
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

isl::union_map union_map::from(isl::multi_union_pw_aff mupa)
{
  auto res = isl_union_map_from_multi_union_pw_aff(mupa.release());
  return manage(res);
}

isl::union_map union_map::from_domain(isl::union_set uset)
{
  auto res = isl_union_map_from_domain(uset.release());
  return manage(res);
}

isl::union_map union_map::from_domain_and_range(isl::union_set domain, isl::union_set range)
{
  auto res = isl_union_map_from_domain_and_range(domain.release(), range.release());
  return manage(res);
}

isl::union_map union_map::from_range(isl::union_set uset)
{
  auto res = isl_union_map_from_range(uset.release());
  return manage(res);
}

isl::union_map union_map::from_union_pw_aff(isl::union_pw_aff upa)
{
  auto res = isl_union_map_from_union_pw_aff(upa.release());
  return manage(res);
}

isl::id union_map::get_dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_union_map_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

uint32_t union_map::get_hash() const
{
  auto res = isl_union_map_get_hash(get());
  return res;
}

isl::map_list union_map::get_map_list() const
{
  auto res = isl_union_map_get_map_list(get());
  return manage(res);
}

isl::space union_map::get_space() const
{
  auto res = isl_union_map_get_space(get());
  return manage(res);
}

isl::union_map union_map::gist(isl::union_map context) const
{
  auto res = isl_union_map_gist(copy(), context.release());
  return manage(res);
}

isl::union_map union_map::gist_domain(isl::union_set uset) const
{
  auto res = isl_union_map_gist_domain(copy(), uset.release());
  return manage(res);
}

isl::union_map union_map::gist_params(isl::set set) const
{
  auto res = isl_union_map_gist_params(copy(), set.release());
  return manage(res);
}

isl::union_map union_map::gist_range(isl::union_set uset) const
{
  auto res = isl_union_map_gist_range(copy(), uset.release());
  return manage(res);
}

isl::union_map union_map::intersect(isl::union_map umap2) const
{
  auto res = isl_union_map_intersect(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::intersect_domain(isl::space space) const
{
  auto res = isl_union_map_intersect_domain_space(copy(), space.release());
  return manage(res);
}

isl::union_map union_map::intersect_domain(isl::union_set uset) const
{
  auto res = isl_union_map_intersect_domain_union_set(copy(), uset.release());
  return manage(res);
}

isl::union_map union_map::intersect_domain_factor_domain(isl::union_map factor) const
{
  auto res = isl_union_map_intersect_domain_factor_domain(copy(), factor.release());
  return manage(res);
}

isl::union_map union_map::intersect_domain_factor_range(isl::union_map factor) const
{
  auto res = isl_union_map_intersect_domain_factor_range(copy(), factor.release());
  return manage(res);
}

isl::union_map union_map::intersect_params(isl::set set) const
{
  auto res = isl_union_map_intersect_params(copy(), set.release());
  return manage(res);
}

isl::union_map union_map::intersect_range(isl::space space) const
{
  auto res = isl_union_map_intersect_range_space(copy(), space.release());
  return manage(res);
}

isl::union_map union_map::intersect_range(isl::union_set uset) const
{
  auto res = isl_union_map_intersect_range_union_set(copy(), uset.release());
  return manage(res);
}

isl::union_map union_map::intersect_range_factor_domain(isl::union_map factor) const
{
  auto res = isl_union_map_intersect_range_factor_domain(copy(), factor.release());
  return manage(res);
}

isl::union_map union_map::intersect_range_factor_range(isl::union_map factor) const
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

boolean union_map::is_disjoint(const isl::union_map &umap2) const
{
  auto res = isl_union_map_is_disjoint(get(), umap2.get());
  return manage(res);
}

boolean union_map::is_empty() const
{
  auto res = isl_union_map_is_empty(get());
  return manage(res);
}

boolean union_map::is_equal(const isl::union_map &umap2) const
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

boolean union_map::is_strict_subset(const isl::union_map &umap2) const
{
  auto res = isl_union_map_is_strict_subset(get(), umap2.get());
  return manage(res);
}

boolean union_map::is_subset(const isl::union_map &umap2) const
{
  auto res = isl_union_map_is_subset(get(), umap2.get());
  return manage(res);
}

boolean union_map::isa_map() const
{
  auto res = isl_union_map_isa_map(get());
  return manage(res);
}

isl::union_map union_map::lex_ge_at_multi_union_pw_aff(isl::multi_union_pw_aff mupa) const
{
  auto res = isl_union_map_lex_ge_at_multi_union_pw_aff(copy(), mupa.release());
  return manage(res);
}

isl::union_map union_map::lex_ge_union_map(isl::union_map umap2) const
{
  auto res = isl_union_map_lex_ge_union_map(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::lex_gt_at_multi_union_pw_aff(isl::multi_union_pw_aff mupa) const
{
  auto res = isl_union_map_lex_gt_at_multi_union_pw_aff(copy(), mupa.release());
  return manage(res);
}

isl::union_map union_map::lex_gt_union_map(isl::union_map umap2) const
{
  auto res = isl_union_map_lex_gt_union_map(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::lex_le_at_multi_union_pw_aff(isl::multi_union_pw_aff mupa) const
{
  auto res = isl_union_map_lex_le_at_multi_union_pw_aff(copy(), mupa.release());
  return manage(res);
}

isl::union_map union_map::lex_le_union_map(isl::union_map umap2) const
{
  auto res = isl_union_map_lex_le_union_map(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::lex_lt_at_multi_union_pw_aff(isl::multi_union_pw_aff mupa) const
{
  auto res = isl_union_map_lex_lt_at_multi_union_pw_aff(copy(), mupa.release());
  return manage(res);
}

isl::union_map union_map::lex_lt_union_map(isl::union_map umap2) const
{
  auto res = isl_union_map_lex_lt_union_map(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::lexmax() const
{
  auto res = isl_union_map_lexmax(copy());
  return manage(res);
}

isl::union_map union_map::lexmin() const
{
  auto res = isl_union_map_lexmin(copy());
  return manage(res);
}

isl_size union_map::n_map() const
{
  auto res = isl_union_map_n_map(get());
  return res;
}

isl::set union_map::params() const
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

isl::union_map union_map::polyhedral_hull() const
{
  auto res = isl_union_map_polyhedral_hull(copy());
  return manage(res);
}

isl::union_map union_map::preimage_domain(isl::multi_aff ma) const
{
  auto res = isl_union_map_preimage_domain_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::union_map union_map::preimage_domain(isl::multi_pw_aff mpa) const
{
  auto res = isl_union_map_preimage_domain_multi_pw_aff(copy(), mpa.release());
  return manage(res);
}

isl::union_map union_map::preimage_domain(isl::pw_multi_aff pma) const
{
  auto res = isl_union_map_preimage_domain_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::union_map union_map::preimage_domain(isl::union_pw_multi_aff upma) const
{
  auto res = isl_union_map_preimage_domain_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

isl::union_map union_map::preimage_range(isl::multi_aff ma) const
{
  auto res = isl_union_map_preimage_range_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::union_map union_map::preimage_range(isl::pw_multi_aff pma) const
{
  auto res = isl_union_map_preimage_range_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::union_map union_map::preimage_range(isl::union_pw_multi_aff upma) const
{
  auto res = isl_union_map_preimage_range_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

isl::union_map union_map::product(isl::union_map umap2) const
{
  auto res = isl_union_map_product(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::project_out(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_union_map_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::union_map union_map::project_out_all_params() const
{
  auto res = isl_union_map_project_out_all_params(copy());
  return manage(res);
}

isl::union_set union_map::range() const
{
  auto res = isl_union_map_range(copy());
  return manage(res);
}

isl::union_map union_map::range_curry() const
{
  auto res = isl_union_map_range_curry(copy());
  return manage(res);
}

isl::union_map union_map::range_factor_domain() const
{
  auto res = isl_union_map_range_factor_domain(copy());
  return manage(res);
}

isl::union_map union_map::range_factor_range() const
{
  auto res = isl_union_map_range_factor_range(copy());
  return manage(res);
}

isl::union_map union_map::range_map() const
{
  auto res = isl_union_map_range_map(copy());
  return manage(res);
}

isl::union_map union_map::range_product(isl::union_map umap2) const
{
  auto res = isl_union_map_range_product(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::range_reverse() const
{
  auto res = isl_union_map_range_reverse(copy());
  return manage(res);
}

isl::union_map union_map::remove_divs() const
{
  auto res = isl_union_map_remove_divs(copy());
  return manage(res);
}

isl::union_map union_map::remove_redundancies() const
{
  auto res = isl_union_map_remove_redundancies(copy());
  return manage(res);
}

isl::union_map union_map::reset_user() const
{
  auto res = isl_union_map_reset_user(copy());
  return manage(res);
}

isl::union_map union_map::reverse() const
{
  auto res = isl_union_map_reverse(copy());
  return manage(res);
}

isl::basic_map union_map::sample() const
{
  auto res = isl_union_map_sample(copy());
  return manage(res);
}

isl::union_map union_map::simple_hull() const
{
  auto res = isl_union_map_simple_hull(copy());
  return manage(res);
}

isl::union_map union_map::subtract(isl::union_map umap2) const
{
  auto res = isl_union_map_subtract(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::subtract_domain(isl::union_set dom) const
{
  auto res = isl_union_map_subtract_domain(copy(), dom.release());
  return manage(res);
}

isl::union_map union_map::subtract_range(isl::union_set dom) const
{
  auto res = isl_union_map_subtract_range(copy(), dom.release());
  return manage(res);
}

isl::union_map union_map::uncurry() const
{
  auto res = isl_union_map_uncurry(copy());
  return manage(res);
}

isl::union_map union_map::unite(isl::union_map umap2) const
{
  auto res = isl_union_map_union(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::universe() const
{
  auto res = isl_union_map_universe(copy());
  return manage(res);
}

isl::union_set union_map::wrap() const
{
  auto res = isl_union_map_wrap(copy());
  return manage(res);
}

isl::union_map union_map::zip() const
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


isl::ctx union_map_list::ctx() const {
  return isl::ctx(isl_union_map_list_get_ctx(ptr));
}

void union_map_list::dump() const {
  isl_union_map_list_dump(get());
}


isl::union_map_list union_map_list::add(isl::union_map el) const
{
  auto res = isl_union_map_list_add(copy(), el.release());
  return manage(res);
}

isl::union_map_list union_map_list::alloc(isl::ctx ctx, int n)
{
  auto res = isl_union_map_list_alloc(ctx.release(), n);
  return manage(res);
}

isl::union_map_list union_map_list::clear() const
{
  auto res = isl_union_map_list_clear(copy());
  return manage(res);
}

isl::union_map_list union_map_list::concat(isl::union_map_list list2) const
{
  auto res = isl_union_map_list_concat(copy(), list2.release());
  return manage(res);
}

isl::union_map_list union_map_list::drop(unsigned int first, unsigned int n) const
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

isl::union_map_list union_map_list::from_union_map(isl::union_map el)
{
  auto res = isl_union_map_list_from_union_map(el.release());
  return manage(res);
}

isl::union_map union_map_list::get_at(int index) const
{
  auto res = isl_union_map_list_get_at(get(), index);
  return manage(res);
}

isl::union_map union_map_list::get_union_map(int index) const
{
  auto res = isl_union_map_list_get_union_map(get(), index);
  return manage(res);
}

isl::union_map_list union_map_list::insert(unsigned int pos, isl::union_map el) const
{
  auto res = isl_union_map_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl_size union_map_list::n_union_map() const
{
  auto res = isl_union_map_list_n_union_map(get());
  return res;
}

isl::union_map_list union_map_list::reverse() const
{
  auto res = isl_union_map_list_reverse(copy());
  return manage(res);
}

isl::union_map_list union_map_list::set_union_map(int index, isl::union_map el) const
{
  auto res = isl_union_map_list_set_union_map(copy(), index, el.release());
  return manage(res);
}

isl_size union_map_list::size() const
{
  auto res = isl_union_map_list_size(get());
  return res;
}

isl::union_map_list union_map_list::swap(unsigned int pos1, unsigned int pos2) const
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


union_pw_aff::union_pw_aff(__isl_take isl_union_pw_aff *ptr)
    : ptr(ptr) {}

union_pw_aff::union_pw_aff(isl::aff aff)
{
  auto res = isl_union_pw_aff_from_aff(aff.release());
  ptr = res;
}
union_pw_aff::union_pw_aff(isl::pw_aff pa)
{
  auto res = isl_union_pw_aff_from_pw_aff(pa.release());
  ptr = res;
}
union_pw_aff::union_pw_aff(isl::ctx ctx, const std::string &str)
{
  auto res = isl_union_pw_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}
union_pw_aff::union_pw_aff(isl::union_set domain, isl::val v)
{
  auto res = isl_union_pw_aff_val_on_domain(domain.release(), v.release());
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


isl::ctx union_pw_aff::ctx() const {
  return isl::ctx(isl_union_pw_aff_get_ctx(ptr));
}

void union_pw_aff::dump() const {
  isl_union_pw_aff_dump(get());
}


isl::union_pw_aff union_pw_aff::add(isl::union_pw_aff upa2) const
{
  auto res = isl_union_pw_aff_add(copy(), upa2.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::add_pw_aff(isl::pw_aff pa) const
{
  auto res = isl_union_pw_aff_add_pw_aff(copy(), pa.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::aff_on_domain(isl::union_set domain, isl::aff aff)
{
  auto res = isl_union_pw_aff_aff_on_domain(domain.release(), aff.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::align_params(isl::space model) const
{
  auto res = isl_union_pw_aff_align_params(copy(), model.release());
  return manage(res);
}

isl::union_set union_pw_aff::bind(isl::id id) const
{
  auto res = isl_union_pw_aff_bind_id(copy(), id.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::coalesce() const
{
  auto res = isl_union_pw_aff_coalesce(copy());
  return manage(res);
}

isl_size union_pw_aff::dim(isl::dim type) const
{
  auto res = isl_union_pw_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::union_set union_pw_aff::domain() const
{
  auto res = isl_union_pw_aff_domain(copy());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_union_pw_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::union_pw_aff union_pw_aff::empty(isl::space space)
{
  auto res = isl_union_pw_aff_empty(space.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::empty_ctx(isl::ctx ctx)
{
  auto res = isl_union_pw_aff_empty_ctx(ctx.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::empty_space(isl::space space)
{
  auto res = isl_union_pw_aff_empty_space(space.release());
  return manage(res);
}

isl::pw_aff union_pw_aff::extract_pw_aff(isl::space space) const
{
  auto res = isl_union_pw_aff_extract_pw_aff(get(), space.release());
  return manage(res);
}

int union_pw_aff::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_union_pw_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::union_pw_aff union_pw_aff::floor() const
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

isl::pw_aff_list union_pw_aff::get_pw_aff_list() const
{
  auto res = isl_union_pw_aff_get_pw_aff_list(get());
  return manage(res);
}

isl::space union_pw_aff::get_space() const
{
  auto res = isl_union_pw_aff_get_space(get());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::gist(isl::union_set context) const
{
  auto res = isl_union_pw_aff_gist(copy(), context.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::gist_params(isl::set context) const
{
  auto res = isl_union_pw_aff_gist_params(copy(), context.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::intersect_domain(isl::space space) const
{
  auto res = isl_union_pw_aff_intersect_domain_space(copy(), space.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::intersect_domain(isl::union_set uset) const
{
  auto res = isl_union_pw_aff_intersect_domain_union_set(copy(), uset.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::intersect_domain_wrapped_domain(isl::union_set uset) const
{
  auto res = isl_union_pw_aff_intersect_domain_wrapped_domain(copy(), uset.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::intersect_domain_wrapped_range(isl::union_set uset) const
{
  auto res = isl_union_pw_aff_intersect_domain_wrapped_range(copy(), uset.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::intersect_params(isl::set set) const
{
  auto res = isl_union_pw_aff_intersect_params(copy(), set.release());
  return manage(res);
}

boolean union_pw_aff::involves_nan() const
{
  auto res = isl_union_pw_aff_involves_nan(get());
  return manage(res);
}

isl::val union_pw_aff::max_val() const
{
  auto res = isl_union_pw_aff_max_val(copy());
  return manage(res);
}

isl::val union_pw_aff::min_val() const
{
  auto res = isl_union_pw_aff_min_val(copy());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::mod_val(isl::val f) const
{
  auto res = isl_union_pw_aff_mod_val(copy(), f.release());
  return manage(res);
}

isl_size union_pw_aff::n_pw_aff() const
{
  auto res = isl_union_pw_aff_n_pw_aff(get());
  return res;
}

isl::union_pw_aff union_pw_aff::neg() const
{
  auto res = isl_union_pw_aff_neg(copy());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::param_on_domain_id(isl::union_set domain, isl::id id)
{
  auto res = isl_union_pw_aff_param_on_domain_id(domain.release(), id.release());
  return manage(res);
}

boolean union_pw_aff::plain_is_equal(const isl::union_pw_aff &upa2) const
{
  auto res = isl_union_pw_aff_plain_is_equal(get(), upa2.get());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::pullback(isl::union_pw_multi_aff upma) const
{
  auto res = isl_union_pw_aff_pullback_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::pw_aff_on_domain(isl::union_set domain, isl::pw_aff pa)
{
  auto res = isl_union_pw_aff_pw_aff_on_domain(domain.release(), pa.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::reset_user() const
{
  auto res = isl_union_pw_aff_reset_user(copy());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::scale_down_val(isl::val v) const
{
  auto res = isl_union_pw_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::scale_val(isl::val v) const
{
  auto res = isl_union_pw_aff_scale_val(copy(), v.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::sub(isl::union_pw_aff upa2) const
{
  auto res = isl_union_pw_aff_sub(copy(), upa2.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::subtract_domain(isl::space space) const
{
  auto res = isl_union_pw_aff_subtract_domain_space(copy(), space.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::subtract_domain(isl::union_set uset) const
{
  auto res = isl_union_pw_aff_subtract_domain_union_set(copy(), uset.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::union_add(isl::union_pw_aff upa2) const
{
  auto res = isl_union_pw_aff_union_add(copy(), upa2.release());
  return manage(res);
}

isl::union_set union_pw_aff::zero_union_set() const
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


isl::ctx union_pw_aff_list::ctx() const {
  return isl::ctx(isl_union_pw_aff_list_get_ctx(ptr));
}

void union_pw_aff_list::dump() const {
  isl_union_pw_aff_list_dump(get());
}


isl::union_pw_aff_list union_pw_aff_list::add(isl::union_pw_aff el) const
{
  auto res = isl_union_pw_aff_list_add(copy(), el.release());
  return manage(res);
}

isl::union_pw_aff_list union_pw_aff_list::alloc(isl::ctx ctx, int n)
{
  auto res = isl_union_pw_aff_list_alloc(ctx.release(), n);
  return manage(res);
}

isl::union_pw_aff_list union_pw_aff_list::clear() const
{
  auto res = isl_union_pw_aff_list_clear(copy());
  return manage(res);
}

isl::union_pw_aff_list union_pw_aff_list::concat(isl::union_pw_aff_list list2) const
{
  auto res = isl_union_pw_aff_list_concat(copy(), list2.release());
  return manage(res);
}

isl::union_pw_aff_list union_pw_aff_list::drop(unsigned int first, unsigned int n) const
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

isl::union_pw_aff_list union_pw_aff_list::from_union_pw_aff(isl::union_pw_aff el)
{
  auto res = isl_union_pw_aff_list_from_union_pw_aff(el.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff_list::get_at(int index) const
{
  auto res = isl_union_pw_aff_list_get_at(get(), index);
  return manage(res);
}

isl::union_pw_aff union_pw_aff_list::get_union_pw_aff(int index) const
{
  auto res = isl_union_pw_aff_list_get_union_pw_aff(get(), index);
  return manage(res);
}

isl::union_pw_aff_list union_pw_aff_list::insert(unsigned int pos, isl::union_pw_aff el) const
{
  auto res = isl_union_pw_aff_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl_size union_pw_aff_list::n_union_pw_aff() const
{
  auto res = isl_union_pw_aff_list_n_union_pw_aff(get());
  return res;
}

isl::union_pw_aff_list union_pw_aff_list::reverse() const
{
  auto res = isl_union_pw_aff_list_reverse(copy());
  return manage(res);
}

isl::union_pw_aff_list union_pw_aff_list::set_union_pw_aff(int index, isl::union_pw_aff el) const
{
  auto res = isl_union_pw_aff_list_set_union_pw_aff(copy(), index, el.release());
  return manage(res);
}

isl_size union_pw_aff_list::size() const
{
  auto res = isl_union_pw_aff_list_size(get());
  return res;
}

isl::union_pw_aff_list union_pw_aff_list::swap(unsigned int pos1, unsigned int pos2) const
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


union_pw_multi_aff::union_pw_multi_aff(__isl_take isl_union_pw_multi_aff *ptr)
    : ptr(ptr) {}

union_pw_multi_aff::union_pw_multi_aff(isl::aff aff)
{
  auto res = isl_union_pw_multi_aff_from_aff(aff.release());
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(isl::union_set uset)
{
  auto res = isl_union_pw_multi_aff_from_domain(uset.release());
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(isl::multi_aff ma)
{
  auto res = isl_union_pw_multi_aff_from_multi_aff(ma.release());
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(isl::multi_union_pw_aff mupa)
{
  auto res = isl_union_pw_multi_aff_from_multi_union_pw_aff(mupa.release());
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(isl::pw_multi_aff pma)
{
  auto res = isl_union_pw_multi_aff_from_pw_multi_aff(pma.release());
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(isl::union_map umap)
{
  auto res = isl_union_pw_multi_aff_from_union_map(umap.release());
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(isl::union_pw_aff upa)
{
  auto res = isl_union_pw_multi_aff_from_union_pw_aff(upa.release());
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(isl::ctx ctx, const std::string &str)
{
  auto res = isl_union_pw_multi_aff_read_from_str(ctx.release(), str.c_str());
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


isl::ctx union_pw_multi_aff::ctx() const {
  return isl::ctx(isl_union_pw_multi_aff_get_ctx(ptr));
}

void union_pw_multi_aff::dump() const {
  isl_union_pw_multi_aff_dump(get());
}


isl::union_pw_multi_aff union_pw_multi_aff::add(isl::union_pw_multi_aff upma2) const
{
  auto res = isl_union_pw_multi_aff_add(copy(), upma2.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::add_pw_multi_aff(isl::pw_multi_aff pma) const
{
  auto res = isl_union_pw_multi_aff_add_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::align_params(isl::space model) const
{
  auto res = isl_union_pw_multi_aff_align_params(copy(), model.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::apply(isl::union_pw_multi_aff upma2) const
{
  auto res = isl_union_pw_multi_aff_apply_union_pw_multi_aff(copy(), upma2.release());
  return manage(res);
}

isl::pw_multi_aff union_pw_multi_aff::as_pw_multi_aff() const
{
  auto res = isl_union_pw_multi_aff_as_pw_multi_aff(copy());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::coalesce() const
{
  auto res = isl_union_pw_multi_aff_coalesce(copy());
  return manage(res);
}

isl_size union_pw_multi_aff::dim(isl::dim type) const
{
  auto res = isl_union_pw_multi_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::union_set union_pw_multi_aff::domain() const
{
  auto res = isl_union_pw_multi_aff_domain(copy());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_union_pw_multi_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::empty(isl::space space)
{
  auto res = isl_union_pw_multi_aff_empty(space.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::empty(isl::ctx ctx)
{
  auto res = isl_union_pw_multi_aff_empty_ctx(ctx.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::empty_space(isl::space space)
{
  auto res = isl_union_pw_multi_aff_empty_space(space.release());
  return manage(res);
}

isl::pw_multi_aff union_pw_multi_aff::extract_pw_multi_aff(isl::space space) const
{
  auto res = isl_union_pw_multi_aff_extract_pw_multi_aff(get(), space.release());
  return manage(res);
}

int union_pw_multi_aff::find_dim_by_name(isl::dim type, const std::string &name) const
{
  auto res = isl_union_pw_multi_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::union_pw_multi_aff union_pw_multi_aff::flat_range_product(isl::union_pw_multi_aff upma2) const
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

isl::union_pw_multi_aff union_pw_multi_aff::from_union_set(isl::union_set uset)
{
  auto res = isl_union_pw_multi_aff_from_union_set(uset.release());
  return manage(res);
}

isl::pw_multi_aff_list union_pw_multi_aff::get_pw_multi_aff_list() const
{
  auto res = isl_union_pw_multi_aff_get_pw_multi_aff_list(get());
  return manage(res);
}

isl::space union_pw_multi_aff::get_space() const
{
  auto res = isl_union_pw_multi_aff_get_space(get());
  return manage(res);
}

isl::union_pw_aff union_pw_multi_aff::get_union_pw_aff(int pos) const
{
  auto res = isl_union_pw_multi_aff_get_union_pw_aff(get(), pos);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::gist(isl::union_set context) const
{
  auto res = isl_union_pw_multi_aff_gist(copy(), context.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::gist_params(isl::set context) const
{
  auto res = isl_union_pw_multi_aff_gist_params(copy(), context.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::intersect_domain(isl::space space) const
{
  auto res = isl_union_pw_multi_aff_intersect_domain_space(copy(), space.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::intersect_domain(isl::union_set uset) const
{
  auto res = isl_union_pw_multi_aff_intersect_domain_union_set(copy(), uset.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::intersect_domain_wrapped_domain(isl::union_set uset) const
{
  auto res = isl_union_pw_multi_aff_intersect_domain_wrapped_domain(copy(), uset.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::intersect_domain_wrapped_range(isl::union_set uset) const
{
  auto res = isl_union_pw_multi_aff_intersect_domain_wrapped_range(copy(), uset.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::intersect_params(isl::set set) const
{
  auto res = isl_union_pw_multi_aff_intersect_params(copy(), set.release());
  return manage(res);
}

boolean union_pw_multi_aff::involves_locals() const
{
  auto res = isl_union_pw_multi_aff_involves_locals(get());
  return manage(res);
}

boolean union_pw_multi_aff::involves_nan() const
{
  auto res = isl_union_pw_multi_aff_involves_nan(get());
  return manage(res);
}

boolean union_pw_multi_aff::isa_pw_multi_aff() const
{
  auto res = isl_union_pw_multi_aff_isa_pw_multi_aff(get());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::multi_val_on_domain(isl::union_set domain, isl::multi_val mv)
{
  auto res = isl_union_pw_multi_aff_multi_val_on_domain(domain.release(), mv.release());
  return manage(res);
}

isl_size union_pw_multi_aff::n_pw_multi_aff() const
{
  auto res = isl_union_pw_multi_aff_n_pw_multi_aff(get());
  return res;
}

isl::union_pw_multi_aff union_pw_multi_aff::neg() const
{
  auto res = isl_union_pw_multi_aff_neg(copy());
  return manage(res);
}

boolean union_pw_multi_aff::plain_is_empty() const
{
  auto res = isl_union_pw_multi_aff_plain_is_empty(get());
  return manage(res);
}

boolean union_pw_multi_aff::plain_is_equal(const isl::union_pw_multi_aff &upma2) const
{
  auto res = isl_union_pw_multi_aff_plain_is_equal(get(), upma2.get());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::preimage_domain_wrapped_domain(isl::union_pw_multi_aff upma2) const
{
  auto res = isl_union_pw_multi_aff_preimage_domain_wrapped_domain_union_pw_multi_aff(copy(), upma2.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::pullback(isl::union_pw_multi_aff upma2) const
{
  auto res = isl_union_pw_multi_aff_pullback_union_pw_multi_aff(copy(), upma2.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::range_factor_domain() const
{
  auto res = isl_union_pw_multi_aff_range_factor_domain(copy());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::range_factor_range() const
{
  auto res = isl_union_pw_multi_aff_range_factor_range(copy());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::range_product(isl::union_pw_multi_aff upma2) const
{
  auto res = isl_union_pw_multi_aff_range_product(copy(), upma2.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::reset_user() const
{
  auto res = isl_union_pw_multi_aff_reset_user(copy());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::scale_down_val(isl::val val) const
{
  auto res = isl_union_pw_multi_aff_scale_down_val(copy(), val.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::scale_multi_val(isl::multi_val mv) const
{
  auto res = isl_union_pw_multi_aff_scale_multi_val(copy(), mv.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::scale_val(isl::val val) const
{
  auto res = isl_union_pw_multi_aff_scale_val(copy(), val.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::sub(isl::union_pw_multi_aff upma2) const
{
  auto res = isl_union_pw_multi_aff_sub(copy(), upma2.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::subtract_domain(isl::space space) const
{
  auto res = isl_union_pw_multi_aff_subtract_domain_space(copy(), space.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::subtract_domain(isl::union_set uset) const
{
  auto res = isl_union_pw_multi_aff_subtract_domain_union_set(copy(), uset.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::union_add(isl::union_pw_multi_aff upma2) const
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


isl::ctx union_pw_multi_aff_list::ctx() const {
  return isl::ctx(isl_union_pw_multi_aff_list_get_ctx(ptr));
}

void union_pw_multi_aff_list::dump() const {
  isl_union_pw_multi_aff_list_dump(get());
}


isl::union_pw_multi_aff_list union_pw_multi_aff_list::add(isl::union_pw_multi_aff el) const
{
  auto res = isl_union_pw_multi_aff_list_add(copy(), el.release());
  return manage(res);
}

isl::union_pw_multi_aff_list union_pw_multi_aff_list::alloc(isl::ctx ctx, int n)
{
  auto res = isl_union_pw_multi_aff_list_alloc(ctx.release(), n);
  return manage(res);
}

isl::union_pw_multi_aff_list union_pw_multi_aff_list::clear() const
{
  auto res = isl_union_pw_multi_aff_list_clear(copy());
  return manage(res);
}

isl::union_pw_multi_aff_list union_pw_multi_aff_list::concat(isl::union_pw_multi_aff_list list2) const
{
  auto res = isl_union_pw_multi_aff_list_concat(copy(), list2.release());
  return manage(res);
}

isl::union_pw_multi_aff_list union_pw_multi_aff_list::drop(unsigned int first, unsigned int n) const
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

isl::union_pw_multi_aff_list union_pw_multi_aff_list::from_union_pw_multi_aff(isl::union_pw_multi_aff el)
{
  auto res = isl_union_pw_multi_aff_list_from_union_pw_multi_aff(el.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff_list::get_at(int index) const
{
  auto res = isl_union_pw_multi_aff_list_get_at(get(), index);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff_list::get_union_pw_multi_aff(int index) const
{
  auto res = isl_union_pw_multi_aff_list_get_union_pw_multi_aff(get(), index);
  return manage(res);
}

isl::union_pw_multi_aff_list union_pw_multi_aff_list::insert(unsigned int pos, isl::union_pw_multi_aff el) const
{
  auto res = isl_union_pw_multi_aff_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl_size union_pw_multi_aff_list::n_union_pw_multi_aff() const
{
  auto res = isl_union_pw_multi_aff_list_n_union_pw_multi_aff(get());
  return res;
}

isl::union_pw_multi_aff_list union_pw_multi_aff_list::reverse() const
{
  auto res = isl_union_pw_multi_aff_list_reverse(copy());
  return manage(res);
}

isl::union_pw_multi_aff_list union_pw_multi_aff_list::set_union_pw_multi_aff(int index, isl::union_pw_multi_aff el) const
{
  auto res = isl_union_pw_multi_aff_list_set_union_pw_multi_aff(copy(), index, el.release());
  return manage(res);
}

isl_size union_pw_multi_aff_list::size() const
{
  auto res = isl_union_pw_multi_aff_list_size(get());
  return res;
}

isl::union_pw_multi_aff_list union_pw_multi_aff_list::swap(unsigned int pos1, unsigned int pos2) const
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


union_pw_qpolynomial::union_pw_qpolynomial(__isl_take isl_union_pw_qpolynomial *ptr)
    : ptr(ptr) {}

union_pw_qpolynomial::union_pw_qpolynomial(isl::ctx ctx, const std::string &str)
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


isl::ctx union_pw_qpolynomial::ctx() const {
  return isl::ctx(isl_union_pw_qpolynomial_get_ctx(ptr));
}


isl::union_pw_qpolynomial union_pw_qpolynomial::add(isl::union_pw_qpolynomial upwqp2) const
{
  auto res = isl_union_pw_qpolynomial_add(copy(), upwqp2.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::add_pw_qpolynomial(isl::pw_qpolynomial pwqp) const
{
  auto res = isl_union_pw_qpolynomial_add_pw_qpolynomial(copy(), pwqp.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::align_params(isl::space model) const
{
  auto res = isl_union_pw_qpolynomial_align_params(copy(), model.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::coalesce() const
{
  auto res = isl_union_pw_qpolynomial_coalesce(copy());
  return manage(res);
}

isl_size union_pw_qpolynomial::dim(isl::dim type) const
{
  auto res = isl_union_pw_qpolynomial_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::union_set union_pw_qpolynomial::domain() const
{
  auto res = isl_union_pw_qpolynomial_domain(copy());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_union_pw_qpolynomial_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::val union_pw_qpolynomial::eval(isl::point pnt) const
{
  auto res = isl_union_pw_qpolynomial_eval(copy(), pnt.release());
  return manage(res);
}

isl::pw_qpolynomial union_pw_qpolynomial::extract_pw_qpolynomial(isl::space space) const
{
  auto res = isl_union_pw_qpolynomial_extract_pw_qpolynomial(get(), space.release());
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

isl::union_pw_qpolynomial union_pw_qpolynomial::from_pw_qpolynomial(isl::pw_qpolynomial pwqp)
{
  auto res = isl_union_pw_qpolynomial_from_pw_qpolynomial(pwqp.release());
  return manage(res);
}

isl::pw_qpolynomial_list union_pw_qpolynomial::get_pw_qpolynomial_list() const
{
  auto res = isl_union_pw_qpolynomial_get_pw_qpolynomial_list(get());
  return manage(res);
}

isl::space union_pw_qpolynomial::get_space() const
{
  auto res = isl_union_pw_qpolynomial_get_space(get());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::gist(isl::union_set context) const
{
  auto res = isl_union_pw_qpolynomial_gist(copy(), context.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::gist_params(isl::set context) const
{
  auto res = isl_union_pw_qpolynomial_gist_params(copy(), context.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::intersect_domain(isl::union_set uset) const
{
  auto res = isl_union_pw_qpolynomial_intersect_domain(copy(), uset.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::intersect_domain_space(isl::space space) const
{
  auto res = isl_union_pw_qpolynomial_intersect_domain_space(copy(), space.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::intersect_domain_union_set(isl::union_set uset) const
{
  auto res = isl_union_pw_qpolynomial_intersect_domain_union_set(copy(), uset.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::intersect_domain_wrapped_domain(isl::union_set uset) const
{
  auto res = isl_union_pw_qpolynomial_intersect_domain_wrapped_domain(copy(), uset.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::intersect_domain_wrapped_range(isl::union_set uset) const
{
  auto res = isl_union_pw_qpolynomial_intersect_domain_wrapped_range(copy(), uset.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::intersect_params(isl::set set) const
{
  auto res = isl_union_pw_qpolynomial_intersect_params(copy(), set.release());
  return manage(res);
}

boolean union_pw_qpolynomial::involves_nan() const
{
  auto res = isl_union_pw_qpolynomial_involves_nan(get());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::mul(isl::union_pw_qpolynomial upwqp2) const
{
  auto res = isl_union_pw_qpolynomial_mul(copy(), upwqp2.release());
  return manage(res);
}

isl_size union_pw_qpolynomial::n_pw_qpolynomial() const
{
  auto res = isl_union_pw_qpolynomial_n_pw_qpolynomial(get());
  return res;
}

isl::union_pw_qpolynomial union_pw_qpolynomial::neg() const
{
  auto res = isl_union_pw_qpolynomial_neg(copy());
  return manage(res);
}

boolean union_pw_qpolynomial::plain_is_equal(const isl::union_pw_qpolynomial &upwqp2) const
{
  auto res = isl_union_pw_qpolynomial_plain_is_equal(get(), upwqp2.get());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::reset_user() const
{
  auto res = isl_union_pw_qpolynomial_reset_user(copy());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::scale_down_val(isl::val v) const
{
  auto res = isl_union_pw_qpolynomial_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::scale_val(isl::val v) const
{
  auto res = isl_union_pw_qpolynomial_scale_val(copy(), v.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::sub(isl::union_pw_qpolynomial upwqp2) const
{
  auto res = isl_union_pw_qpolynomial_sub(copy(), upwqp2.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::subtract_domain(isl::union_set uset) const
{
  auto res = isl_union_pw_qpolynomial_subtract_domain(copy(), uset.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::subtract_domain_space(isl::space space) const
{
  auto res = isl_union_pw_qpolynomial_subtract_domain_space(copy(), space.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::subtract_domain_union_set(isl::union_set uset) const
{
  auto res = isl_union_pw_qpolynomial_subtract_domain_union_set(copy(), uset.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::to_polynomial(int sign) const
{
  auto res = isl_union_pw_qpolynomial_to_polynomial(copy(), sign);
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::zero(isl::space space)
{
  auto res = isl_union_pw_qpolynomial_zero(space.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::zero_ctx(isl::ctx ctx)
{
  auto res = isl_union_pw_qpolynomial_zero_ctx(ctx.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::zero_space(isl::space space)
{
  auto res = isl_union_pw_qpolynomial_zero_space(space.release());
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


union_set::union_set(__isl_take isl_union_set *ptr)
    : ptr(ptr) {}

union_set::union_set(isl::basic_set bset)
{
  auto res = isl_union_set_from_basic_set(bset.release());
  ptr = res;
}
union_set::union_set(isl::point pnt)
{
  auto res = isl_union_set_from_point(pnt.release());
  ptr = res;
}
union_set::union_set(isl::set set)
{
  auto res = isl_union_set_from_set(set.release());
  ptr = res;
}
union_set::union_set(isl::ctx ctx, const std::string &str)
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


isl::ctx union_set::ctx() const {
  return isl::ctx(isl_union_set_get_ctx(ptr));
}

void union_set::dump() const {
  isl_union_set_dump(get());
}


isl::union_set union_set::affine_hull() const
{
  auto res = isl_union_set_affine_hull(copy());
  return manage(res);
}

isl::union_set union_set::align_params(isl::space model) const
{
  auto res = isl_union_set_align_params(copy(), model.release());
  return manage(res);
}

isl::union_set union_set::apply(isl::union_map umap) const
{
  auto res = isl_union_set_apply(copy(), umap.release());
  return manage(res);
}

isl::union_set union_set::coalesce() const
{
  auto res = isl_union_set_coalesce(copy());
  return manage(res);
}

isl::union_set union_set::coefficients() const
{
  auto res = isl_union_set_coefficients(copy());
  return manage(res);
}

isl::schedule union_set::compute_schedule(isl::union_map validity, isl::union_map proximity) const
{
  auto res = isl_union_set_compute_schedule(copy(), validity.release(), proximity.release());
  return manage(res);
}

boolean union_set::contains(const isl::space &space) const
{
  auto res = isl_union_set_contains(get(), space.get());
  return manage(res);
}

isl::union_set union_set::detect_equalities() const
{
  auto res = isl_union_set_detect_equalities(copy());
  return manage(res);
}

isl_size union_set::dim(isl::dim type) const
{
  auto res = isl_union_set_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::union_set union_set::empty(isl::ctx ctx)
{
  auto res = isl_union_set_empty_ctx(ctx.release());
  return manage(res);
}

isl::set union_set::extract_set(isl::space space) const
{
  auto res = isl_union_set_extract_set(get(), space.release());
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

isl::basic_set_list union_set::get_basic_set_list() const
{
  auto res = isl_union_set_get_basic_set_list(get());
  return manage(res);
}

uint32_t union_set::get_hash() const
{
  auto res = isl_union_set_get_hash(get());
  return res;
}

isl::set_list union_set::get_set_list() const
{
  auto res = isl_union_set_get_set_list(get());
  return manage(res);
}

isl::space union_set::get_space() const
{
  auto res = isl_union_set_get_space(get());
  return manage(res);
}

isl::union_set union_set::gist(isl::union_set context) const
{
  auto res = isl_union_set_gist(copy(), context.release());
  return manage(res);
}

isl::union_set union_set::gist_params(isl::set set) const
{
  auto res = isl_union_set_gist_params(copy(), set.release());
  return manage(res);
}

isl::union_map union_set::identity() const
{
  auto res = isl_union_set_identity(copy());
  return manage(res);
}

isl::union_pw_multi_aff union_set::identity_union_pw_multi_aff() const
{
  auto res = isl_union_set_identity_union_pw_multi_aff(copy());
  return manage(res);
}

isl::union_set union_set::intersect(isl::union_set uset2) const
{
  auto res = isl_union_set_intersect(copy(), uset2.release());
  return manage(res);
}

isl::union_set union_set::intersect_params(isl::set set) const
{
  auto res = isl_union_set_intersect_params(copy(), set.release());
  return manage(res);
}

boolean union_set::is_disjoint(const isl::union_set &uset2) const
{
  auto res = isl_union_set_is_disjoint(get(), uset2.get());
  return manage(res);
}

boolean union_set::is_empty() const
{
  auto res = isl_union_set_is_empty(get());
  return manage(res);
}

boolean union_set::is_equal(const isl::union_set &uset2) const
{
  auto res = isl_union_set_is_equal(get(), uset2.get());
  return manage(res);
}

boolean union_set::is_params() const
{
  auto res = isl_union_set_is_params(get());
  return manage(res);
}

boolean union_set::is_strict_subset(const isl::union_set &uset2) const
{
  auto res = isl_union_set_is_strict_subset(get(), uset2.get());
  return manage(res);
}

boolean union_set::is_subset(const isl::union_set &uset2) const
{
  auto res = isl_union_set_is_subset(get(), uset2.get());
  return manage(res);
}

boolean union_set::isa_set() const
{
  auto res = isl_union_set_isa_set(get());
  return manage(res);
}

isl::union_map union_set::lex_ge_union_set(isl::union_set uset2) const
{
  auto res = isl_union_set_lex_ge_union_set(copy(), uset2.release());
  return manage(res);
}

isl::union_map union_set::lex_gt_union_set(isl::union_set uset2) const
{
  auto res = isl_union_set_lex_gt_union_set(copy(), uset2.release());
  return manage(res);
}

isl::union_map union_set::lex_le_union_set(isl::union_set uset2) const
{
  auto res = isl_union_set_lex_le_union_set(copy(), uset2.release());
  return manage(res);
}

isl::union_map union_set::lex_lt_union_set(isl::union_set uset2) const
{
  auto res = isl_union_set_lex_lt_union_set(copy(), uset2.release());
  return manage(res);
}

isl::union_set union_set::lexmax() const
{
  auto res = isl_union_set_lexmax(copy());
  return manage(res);
}

isl::union_set union_set::lexmin() const
{
  auto res = isl_union_set_lexmin(copy());
  return manage(res);
}

isl::multi_val union_set::min_multi_union_pw_aff(const isl::multi_union_pw_aff &obj) const
{
  auto res = isl_union_set_min_multi_union_pw_aff(get(), obj.get());
  return manage(res);
}

isl_size union_set::n_set() const
{
  auto res = isl_union_set_n_set(get());
  return res;
}

isl::set union_set::params() const
{
  auto res = isl_union_set_params(copy());
  return manage(res);
}

isl::union_set union_set::polyhedral_hull() const
{
  auto res = isl_union_set_polyhedral_hull(copy());
  return manage(res);
}

isl::union_set union_set::preimage(isl::multi_aff ma) const
{
  auto res = isl_union_set_preimage_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::union_set union_set::preimage(isl::pw_multi_aff pma) const
{
  auto res = isl_union_set_preimage_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::union_set union_set::preimage(isl::union_pw_multi_aff upma) const
{
  auto res = isl_union_set_preimage_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

isl::union_set union_set::product(isl::union_set uset2) const
{
  auto res = isl_union_set_product(copy(), uset2.release());
  return manage(res);
}

isl::union_set union_set::project_out(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_union_set_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::union_set union_set::project_out_all_params() const
{
  auto res = isl_union_set_project_out_all_params(copy());
  return manage(res);
}

isl::union_set union_set::remove_divs() const
{
  auto res = isl_union_set_remove_divs(copy());
  return manage(res);
}

isl::union_set union_set::remove_redundancies() const
{
  auto res = isl_union_set_remove_redundancies(copy());
  return manage(res);
}

isl::union_set union_set::reset_user() const
{
  auto res = isl_union_set_reset_user(copy());
  return manage(res);
}

isl::basic_set union_set::sample() const
{
  auto res = isl_union_set_sample(copy());
  return manage(res);
}

isl::point union_set::sample_point() const
{
  auto res = isl_union_set_sample_point(copy());
  return manage(res);
}

isl::union_set union_set::simple_hull() const
{
  auto res = isl_union_set_simple_hull(copy());
  return manage(res);
}

isl::union_set union_set::solutions() const
{
  auto res = isl_union_set_solutions(copy());
  return manage(res);
}

isl::union_set union_set::subtract(isl::union_set uset2) const
{
  auto res = isl_union_set_subtract(copy(), uset2.release());
  return manage(res);
}

isl::union_set union_set::unite(isl::union_set uset2) const
{
  auto res = isl_union_set_union(copy(), uset2.release());
  return manage(res);
}

isl::union_set union_set::universe() const
{
  auto res = isl_union_set_universe(copy());
  return manage(res);
}

isl::union_map union_set::unwrap() const
{
  auto res = isl_union_set_unwrap(copy());
  return manage(res);
}

isl::union_map union_set::wrapped_domain_map() const
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


isl::ctx union_set_list::ctx() const {
  return isl::ctx(isl_union_set_list_get_ctx(ptr));
}

void union_set_list::dump() const {
  isl_union_set_list_dump(get());
}


isl::union_set_list union_set_list::add(isl::union_set el) const
{
  auto res = isl_union_set_list_add(copy(), el.release());
  return manage(res);
}

isl::union_set_list union_set_list::alloc(isl::ctx ctx, int n)
{
  auto res = isl_union_set_list_alloc(ctx.release(), n);
  return manage(res);
}

isl::union_set_list union_set_list::clear() const
{
  auto res = isl_union_set_list_clear(copy());
  return manage(res);
}

isl::union_set_list union_set_list::concat(isl::union_set_list list2) const
{
  auto res = isl_union_set_list_concat(copy(), list2.release());
  return manage(res);
}

isl::union_set_list union_set_list::drop(unsigned int first, unsigned int n) const
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

isl::union_set_list union_set_list::from_union_set(isl::union_set el)
{
  auto res = isl_union_set_list_from_union_set(el.release());
  return manage(res);
}

isl::union_set union_set_list::get_at(int index) const
{
  auto res = isl_union_set_list_get_at(get(), index);
  return manage(res);
}

isl::union_set union_set_list::get_union_set(int index) const
{
  auto res = isl_union_set_list_get_union_set(get(), index);
  return manage(res);
}

isl::union_set_list union_set_list::insert(unsigned int pos, isl::union_set el) const
{
  auto res = isl_union_set_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl_size union_set_list::n_union_set() const
{
  auto res = isl_union_set_list_n_union_set(get());
  return res;
}

isl::union_set_list union_set_list::reverse() const
{
  auto res = isl_union_set_list_reverse(copy());
  return manage(res);
}

isl::union_set_list union_set_list::set_union_set(int index, isl::union_set el) const
{
  auto res = isl_union_set_list_set_union_set(copy(), index, el.release());
  return manage(res);
}

isl_size union_set_list::size() const
{
  auto res = isl_union_set_list_size(get());
  return res;
}

isl::union_set_list union_set_list::swap(unsigned int pos1, unsigned int pos2) const
{
  auto res = isl_union_set_list_swap(copy(), pos1, pos2);
  return manage(res);
}

isl::union_set union_set_list::unite() const
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


val::val(__isl_take isl_val *ptr)
    : ptr(ptr) {}

val::val(isl::ctx ctx, long i)
{
  auto res = isl_val_int_from_si(ctx.release(), i);
  ptr = res;
}
val::val(isl::ctx ctx, const std::string &str)
{
  auto res = isl_val_read_from_str(ctx.release(), str.c_str());
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


isl::ctx val::ctx() const {
  return isl::ctx(isl_val_get_ctx(ptr));
}

void val::dump() const {
  isl_val_dump(get());
}


isl::val val::abs() const
{
  auto res = isl_val_abs(copy());
  return manage(res);
}

boolean val::abs_eq(const isl::val &v2) const
{
  auto res = isl_val_abs_eq(get(), v2.get());
  return manage(res);
}

isl::val val::add(isl::val v2) const
{
  auto res = isl_val_add(copy(), v2.release());
  return manage(res);
}

isl::val val::add_ui(unsigned long v2) const
{
  auto res = isl_val_add_ui(copy(), v2);
  return manage(res);
}

isl::val val::ceil() const
{
  auto res = isl_val_ceil(copy());
  return manage(res);
}

int val::cmp_si(long i) const
{
  auto res = isl_val_cmp_si(get(), i);
  return res;
}

isl::val val::div(isl::val v2) const
{
  auto res = isl_val_div(copy(), v2.release());
  return manage(res);
}

isl::val val::div_ui(unsigned long v2) const
{
  auto res = isl_val_div_ui(copy(), v2);
  return manage(res);
}

boolean val::eq(const isl::val &v2) const
{
  auto res = isl_val_eq(get(), v2.get());
  return manage(res);
}

boolean val::eq_si(long i) const
{
  auto res = isl_val_eq_si(get(), i);
  return manage(res);
}

isl::val val::floor() const
{
  auto res = isl_val_floor(copy());
  return manage(res);
}

isl::val val::gcd(isl::val v2) const
{
  auto res = isl_val_gcd(copy(), v2.release());
  return manage(res);
}

boolean val::ge(const isl::val &v2) const
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

boolean val::gt(const isl::val &v2) const
{
  auto res = isl_val_gt(get(), v2.get());
  return manage(res);
}

boolean val::gt_si(long i) const
{
  auto res = isl_val_gt_si(get(), i);
  return manage(res);
}

isl::val val::infty(isl::ctx ctx)
{
  auto res = isl_val_infty(ctx.release());
  return manage(res);
}

isl::val val::int_from_ui(isl::ctx ctx, unsigned long u)
{
  auto res = isl_val_int_from_ui(ctx.release(), u);
  return manage(res);
}

isl::val val::inv() const
{
  auto res = isl_val_inv(copy());
  return manage(res);
}

boolean val::is_divisible_by(const isl::val &v2) const
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

boolean val::le(const isl::val &v2) const
{
  auto res = isl_val_le(get(), v2.get());
  return manage(res);
}

boolean val::lt(const isl::val &v2) const
{
  auto res = isl_val_lt(get(), v2.get());
  return manage(res);
}

isl::val val::max(isl::val v2) const
{
  auto res = isl_val_max(copy(), v2.release());
  return manage(res);
}

isl::val val::min(isl::val v2) const
{
  auto res = isl_val_min(copy(), v2.release());
  return manage(res);
}

isl::val val::mod(isl::val v2) const
{
  auto res = isl_val_mod(copy(), v2.release());
  return manage(res);
}

isl::val val::mul(isl::val v2) const
{
  auto res = isl_val_mul(copy(), v2.release());
  return manage(res);
}

isl::val val::mul_ui(unsigned long v2) const
{
  auto res = isl_val_mul_ui(copy(), v2);
  return manage(res);
}

isl_size val::n_abs_num_chunks(size_t size) const
{
  auto res = isl_val_n_abs_num_chunks(get(), size);
  return res;
}

isl::val val::nan(isl::ctx ctx)
{
  auto res = isl_val_nan(ctx.release());
  return manage(res);
}

boolean val::ne(const isl::val &v2) const
{
  auto res = isl_val_ne(get(), v2.get());
  return manage(res);
}

isl::val val::neg() const
{
  auto res = isl_val_neg(copy());
  return manage(res);
}

isl::val val::neginfty(isl::ctx ctx)
{
  auto res = isl_val_neginfty(ctx.release());
  return manage(res);
}

isl::val val::negone(isl::ctx ctx)
{
  auto res = isl_val_negone(ctx.release());
  return manage(res);
}

isl::val val::one(isl::ctx ctx)
{
  auto res = isl_val_one(ctx.release());
  return manage(res);
}

isl::val val::pow2() const
{
  auto res = isl_val_pow2(copy());
  return manage(res);
}

isl::val val::set_si(long i) const
{
  auto res = isl_val_set_si(copy(), i);
  return manage(res);
}

int val::sgn() const
{
  auto res = isl_val_sgn(get());
  return res;
}

isl::val val::sub(isl::val v2) const
{
  auto res = isl_val_sub(copy(), v2.release());
  return manage(res);
}

isl::val val::sub_ui(unsigned long v2) const
{
  auto res = isl_val_sub_ui(copy(), v2);
  return manage(res);
}

isl::val val::trunc() const
{
  auto res = isl_val_trunc(copy());
  return manage(res);
}

isl::val val::zero(isl::ctx ctx)
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


isl::ctx val_list::ctx() const {
  return isl::ctx(isl_val_list_get_ctx(ptr));
}

void val_list::dump() const {
  isl_val_list_dump(get());
}


isl::val_list val_list::add(isl::val el) const
{
  auto res = isl_val_list_add(copy(), el.release());
  return manage(res);
}

isl::val_list val_list::alloc(isl::ctx ctx, int n)
{
  auto res = isl_val_list_alloc(ctx.release(), n);
  return manage(res);
}

isl::val_list val_list::clear() const
{
  auto res = isl_val_list_clear(copy());
  return manage(res);
}

isl::val_list val_list::concat(isl::val_list list2) const
{
  auto res = isl_val_list_concat(copy(), list2.release());
  return manage(res);
}

isl::val_list val_list::drop(unsigned int first, unsigned int n) const
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

isl::val_list val_list::from_val(isl::val el)
{
  auto res = isl_val_list_from_val(el.release());
  return manage(res);
}

isl::val val_list::get_at(int index) const
{
  auto res = isl_val_list_get_at(get(), index);
  return manage(res);
}

isl::val val_list::get_val(int index) const
{
  auto res = isl_val_list_get_val(get(), index);
  return manage(res);
}

isl::val_list val_list::insert(unsigned int pos, isl::val el) const
{
  auto res = isl_val_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl_size val_list::n_val() const
{
  auto res = isl_val_list_n_val(get());
  return res;
}

isl::val_list val_list::reverse() const
{
  auto res = isl_val_list_reverse(copy());
  return manage(res);
}

isl::val_list val_list::set_val(int index, isl::val el) const
{
  auto res = isl_val_list_set_val(copy(), index, el.release());
  return manage(res);
}

isl_size val_list::size() const
{
  auto res = isl_val_list_size(get());
  return res;
}

isl::val_list val_list::swap(unsigned int pos1, unsigned int pos2) const
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


isl::ctx vec::ctx() const {
  return isl::ctx(isl_vec_get_ctx(ptr));
}

void vec::dump() const {
  isl_vec_dump(get());
}


isl::vec vec::add(isl::vec vec2) const
{
  auto res = isl_vec_add(copy(), vec2.release());
  return manage(res);
}

isl::vec vec::add_els(unsigned int n) const
{
  auto res = isl_vec_add_els(copy(), n);
  return manage(res);
}

isl::vec vec::alloc(isl::ctx ctx, unsigned int size)
{
  auto res = isl_vec_alloc(ctx.release(), size);
  return manage(res);
}

isl::vec vec::ceil() const
{
  auto res = isl_vec_ceil(copy());
  return manage(res);
}

isl::vec vec::clr() const
{
  auto res = isl_vec_clr(copy());
  return manage(res);
}

int vec::cmp_element(const isl::vec &vec2, int pos) const
{
  auto res = isl_vec_cmp_element(get(), vec2.get(), pos);
  return res;
}

isl::vec vec::concat(isl::vec vec2) const
{
  auto res = isl_vec_concat(copy(), vec2.release());
  return manage(res);
}

isl::vec vec::drop_els(unsigned int pos, unsigned int n) const
{
  auto res = isl_vec_drop_els(copy(), pos, n);
  return manage(res);
}

isl::vec vec::extend(unsigned int size) const
{
  auto res = isl_vec_extend(copy(), size);
  return manage(res);
}

isl::val vec::get_element_val(int pos) const
{
  auto res = isl_vec_get_element_val(get(), pos);
  return manage(res);
}

isl::vec vec::insert_els(unsigned int pos, unsigned int n) const
{
  auto res = isl_vec_insert_els(copy(), pos, n);
  return manage(res);
}

isl::vec vec::insert_zero_els(unsigned int pos, unsigned int n) const
{
  auto res = isl_vec_insert_zero_els(copy(), pos, n);
  return manage(res);
}

boolean vec::is_equal(const isl::vec &vec2) const
{
  auto res = isl_vec_is_equal(get(), vec2.get());
  return manage(res);
}

isl::vec vec::mat_product(isl::mat mat) const
{
  auto res = isl_vec_mat_product(copy(), mat.release());
  return manage(res);
}

isl::vec vec::move_els(unsigned int dst_col, unsigned int src_col, unsigned int n) const
{
  auto res = isl_vec_move_els(copy(), dst_col, src_col, n);
  return manage(res);
}

isl::vec vec::neg() const
{
  auto res = isl_vec_neg(copy());
  return manage(res);
}

isl::vec vec::set_element_si(int pos, int v) const
{
  auto res = isl_vec_set_element_si(copy(), pos, v);
  return manage(res);
}

isl::vec vec::set_element_val(int pos, isl::val v) const
{
  auto res = isl_vec_set_element_val(copy(), pos, v.release());
  return manage(res);
}

isl::vec vec::set_si(int v) const
{
  auto res = isl_vec_set_si(copy(), v);
  return manage(res);
}

isl::vec vec::set_val(isl::val v) const
{
  auto res = isl_vec_set_val(copy(), v.release());
  return manage(res);
}

isl_size vec::size() const
{
  auto res = isl_vec_size(get());
  return res;
}

isl::vec vec::sort() const
{
  auto res = isl_vec_sort(copy());
  return manage(res);
}

isl::vec vec::zero(isl::ctx ctx, unsigned int size)
{
  auto res = isl_vec_zero(ctx.release(), size);
  return manage(res);
}

isl::vec vec::zero_extend(unsigned int size) const
{
  auto res = isl_vec_zero_extend(copy(), size);
  return manage(res);
}
} // namespace noexceptions 
} // namespace isl

#endif /* ISL_CPP_CHECKED */
