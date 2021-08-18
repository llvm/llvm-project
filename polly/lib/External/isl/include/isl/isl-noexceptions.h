/// These are automatically generated checked C++ bindings for isl.
///
/// isl is a library for computing with integer sets and maps described by
/// Presburger formulas. On top of this, isl provides various tools for
/// polyhedral compilation, ranging from dependence analysis over scheduling
/// to AST generation.

// clang-format off

#ifndef ISL_CPP_CHECKED
#define ISL_CPP_CHECKED

#include <stdio.h>
#include <stdlib.h>
#include <isl/set.h>

#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>

namespace isl {

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

/* Class used to check that isl::checked::boolean,
 * isl::checked::stat and isl::checked::size values are checked for errors.
 */
struct checker {
	bool checked = false;
	~checker() {
		//ISLPP_ASSERT(checked, "IMPLEMENTATION ERROR: Unchecked state");
	}
};

class boolean {
private:
  mutable std::shared_ptr<checker> check = std::make_shared<checker>();
  isl_bool val;

  friend boolean manage(isl_bool val);
  boolean(isl_bool val): val(val) {}
public:
  static boolean error() {
    return boolean(isl_bool_error);
  }
  boolean()
      : val(isl_bool_error) {}

  /* implicit */ boolean(bool val)
      : val(val ? isl_bool_true : isl_bool_false) {}

  isl_bool release() {
    auto tmp = val;
    val = isl_bool_error;
    check->checked = true;
    return tmp;
  }

  bool is_error() const { check->checked = true; return val == isl_bool_error; }
  bool is_false() const { check->checked = true; return val == isl_bool_false; }
  bool is_true() const { check->checked = true; return val == isl_bool_true; }

  operator bool() const {
    //ISLPP_ASSERT(check->checked, "IMPLEMENTATION ERROR: Unchecked error state");
    ISLPP_ASSERT(!is_error(), "IMPLEMENTATION ERROR: Unhandled error state");
    return is_true();
  }

  boolean negate() {
    if (val == isl_bool_true)
      val = isl_bool_false;
    else if (val == isl_bool_false)
      val = isl_bool_true;
    return *this;
  }

  boolean operator!() const {
    return boolean(*this).negate();
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
	mutable std::shared_ptr<checker> check = std::make_shared<checker>();
	isl_stat val;

	friend stat manage(isl_stat val);
	stat(isl_stat val) : val(val) {}
public:
	static stat ok() {
		return stat(isl_stat_ok);
	}
	static stat error() {
		return stat(isl_stat_error);
	}
	stat() : val(isl_stat_error) {}

	isl_stat release() {
		check->checked = true;
		return val;
	}

	bool is_error() const {
		check->checked = true;
		return val == isl_stat_error;
	}
	bool is_ok() const {
		check->checked = true;
		return val == isl_stat_ok;
	}
};

inline stat manage(isl_stat val)
{
	return stat(val);
}

/* Class encapsulating an isl_size value.
 */
class size {
private:
	mutable std::shared_ptr<checker> check = std::make_shared<checker>();
	isl_size val;

	friend size manage(isl_size val);
	size(isl_size val) : val(val) {}
public:
	size() : val(isl_size_error) {}

	isl_size release() {
		auto tmp = val;
		val = isl_size_error;
		check->checked = true;
		return tmp;
	}

	bool is_error() const {
		check->checked = true;
		return val == isl_size_error;
	}

	explicit operator unsigned() const {
		ISLPP_ASSERT(check->checked,
			    "IMPLEMENTATION ERROR: Unchecked error state");
		ISLPP_ASSERT(!is_error(),
			    "IMPLEMENTATION ERROR: Unhandled error state");
		return val;
	}
};

inline size manage(isl_size val)
{
	return size(val);
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

} // namespace isl

#include <isl/id.h>
#include <isl/space.h>
#include <isl/val.h>
#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/ilp.h>
#include <isl/constraint.h>
#include <isl/union_set.h>
#include <isl/union_map.h>
#include <isl/flow.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/ast_build.h>
#include <isl/fixed_box.h>

namespace isl {

// forward declarations
class aff;
class aff_list;
class ast_build;
class ast_expr;
class ast_expr_id;
class ast_expr_int;
class ast_expr_op;
class ast_expr_op_access;
class ast_expr_op_add;
class ast_expr_op_address_of;
class ast_expr_op_and;
class ast_expr_op_and_then;
class ast_expr_op_call;
class ast_expr_op_cond;
class ast_expr_op_div;
class ast_expr_op_eq;
class ast_expr_op_fdiv_q;
class ast_expr_op_ge;
class ast_expr_op_gt;
class ast_expr_op_le;
class ast_expr_op_lt;
class ast_expr_op_max;
class ast_expr_op_member;
class ast_expr_op_min;
class ast_expr_op_minus;
class ast_expr_op_mul;
class ast_expr_op_or;
class ast_expr_op_or_else;
class ast_expr_op_pdiv_q;
class ast_expr_op_pdiv_r;
class ast_expr_op_select;
class ast_expr_op_sub;
class ast_expr_op_zdiv_r;
class ast_node;
class ast_node_block;
class ast_node_for;
class ast_node_if;
class ast_node_list;
class ast_node_mark;
class ast_node_user;
class basic_map;
class basic_map_list;
class basic_set;
class basic_set_list;
class constraint;
class fixed_box;
class id;
class id_list;
class id_to_ast_expr;
class local_space;
class map;
class map_list;
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
class schedule;
class schedule_constraints;
class schedule_node;
class schedule_node_band;
class schedule_node_context;
class schedule_node_domain;
class schedule_node_expansion;
class schedule_node_extension;
class schedule_node_filter;
class schedule_node_guard;
class schedule_node_leaf;
class schedule_node_mark;
class schedule_node_sequence;
class schedule_node_set;
class set;
class set_list;
class space;
class union_access_info;
class union_flow;
class union_map;
class union_pw_aff;
class union_pw_aff_list;
class union_pw_multi_aff;
class union_set;
class union_set_list;
class val;
class val_list;

// declarations for isl::aff
inline aff manage(__isl_take isl_aff *ptr);
inline aff manage_copy(__isl_keep isl_aff *ptr);

class aff {
  friend inline aff manage(__isl_take isl_aff *ptr);
  friend inline aff manage_copy(__isl_keep isl_aff *ptr);

protected:
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

  inline isl::aff add(isl::aff aff2) const;
  inline isl::multi_aff add(const isl::multi_aff &multi2) const;
  inline isl::multi_pw_aff add(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff add(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_aff add(const isl::pw_aff &pwaff2) const;
  inline isl::pw_multi_aff add(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_aff add(const isl::union_pw_aff &upa2) const;
  inline isl::union_pw_multi_aff add(const isl::union_pw_multi_aff &upma2) const;
  inline isl::aff add_constant(isl::val v) const;
  inline isl::aff add_constant(long v) const;
  inline isl::multi_aff add_constant(const isl::multi_val &mv) const;
  inline isl::aff add_constant_si(int v) const;
  inline isl::pw_aff add_dims(isl::dim type, unsigned int n) const;
  inline isl::union_pw_multi_aff add_pw_multi_aff(const isl::pw_multi_aff &pma) const;
  inline isl::union_pw_multi_aff apply(const isl::union_pw_multi_aff &upma2) const;
  inline isl::aff as_aff() const;
  inline isl::map as_map() const;
  inline isl::multi_aff as_multi_aff() const;
  inline isl::multi_union_pw_aff as_multi_union_pw_aff() const;
  inline isl::pw_multi_aff as_pw_multi_aff() const;
  inline isl::set as_set() const;
  inline isl::union_map as_union_map() const;
  inline isl::aff at(int pos) const;
  inline isl::basic_set bind(isl::id id) const;
  inline isl::basic_set bind(const std::string &id) const;
  inline isl::basic_set bind(const isl::multi_id &tuple) const;
  inline isl::pw_aff bind_domain(const isl::multi_id &tuple) const;
  inline isl::pw_aff bind_domain_wrapped_domain(const isl::multi_id &tuple) const;
  inline isl::aff ceil() const;
  inline isl::pw_aff coalesce() const;
  inline isl::pw_aff cond(const isl::pw_aff &pwaff_true, const isl::pw_aff &pwaff_false) const;
  inline isl::multi_val constant_multi_val() const;
  inline isl::val constant_val() const;
  inline isl::val get_constant_val() const;
  inline isl::val denominator_val() const;
  inline isl::val get_denominator_val() const;
  inline class size dim(isl::dim type) const;
  inline isl::id dim_id(isl::dim type, unsigned int pos) const;
  inline isl::aff div(isl::aff aff2) const;
  inline isl::pw_aff div(const isl::pw_aff &pa2) const;
  inline isl::set domain() const;
  inline isl::space domain_space() const;
  inline isl::pw_multi_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set eq_set(isl::aff aff2) const;
  inline isl::set eq_set(const isl::pw_aff &pwaff2) const;
  inline isl::val eval(isl::point pnt) const;
  inline isl::pw_multi_aff extract_pw_multi_aff(const isl::space &space) const;
  inline isl::multi_aff flat_range_product(const isl::multi_aff &multi2) const;
  inline isl::multi_pw_aff flat_range_product(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff flat_range_product(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_multi_aff flat_range_product(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_multi_aff flat_range_product(const isl::union_pw_multi_aff &upma2) const;
  inline isl::aff floor() const;
  inline stat foreach_piece(const std::function<stat(isl::set, isl::aff)> &fn) const;
  inline stat foreach_piece(const std::function<stat(isl::set, isl::multi_aff)> &fn) const;
  inline stat foreach_pw_aff(const std::function<stat(isl::pw_aff)> &fn) const;
  inline isl::set ge_set(isl::aff aff2) const;
  inline isl::set ge_set(const isl::pw_aff &pwaff2) const;
  inline isl::aff gist(isl::set context) const;
  inline isl::union_pw_aff gist(const isl::union_set &context) const;
  inline isl::aff gist(const isl::basic_set &context) const;
  inline isl::aff gist(const isl::point &context) const;
  inline isl::set gt_set(isl::aff aff2) const;
  inline isl::set gt_set(const isl::pw_aff &pwaff2) const;
  inline boolean has_range_tuple_id() const;
  inline isl::multi_aff identity() const;
  inline isl::pw_aff insert_domain(const isl::space &domain) const;
  inline isl::pw_aff intersect_domain(const isl::set &set) const;
  inline isl::union_pw_aff intersect_domain(const isl::space &space) const;
  inline isl::union_pw_aff intersect_domain(const isl::union_set &uset) const;
  inline isl::union_pw_aff intersect_domain_wrapped_domain(const isl::union_set &uset) const;
  inline isl::union_pw_aff intersect_domain_wrapped_range(const isl::union_set &uset) const;
  inline isl::pw_aff intersect_params(const isl::set &set) const;
  inline boolean involves_locals() const;
  inline boolean involves_nan() const;
  inline boolean involves_param(const isl::id &id) const;
  inline boolean involves_param(const std::string &id) const;
  inline boolean involves_param(const isl::id_list &list) const;
  inline boolean is_cst() const;
  inline boolean is_equal(const isl::pw_aff &pa2) const;
  inline boolean isa_aff() const;
  inline boolean isa_multi_aff() const;
  inline boolean isa_pw_multi_aff() const;
  inline isl::set le_set(isl::aff aff2) const;
  inline isl::set le_set(const isl::pw_aff &pwaff2) const;
  inline isl::aff_list list() const;
  inline isl::set lt_set(isl::aff aff2) const;
  inline isl::set lt_set(const isl::pw_aff &pwaff2) const;
  inline isl::multi_pw_aff max(const isl::multi_pw_aff &multi2) const;
  inline isl::pw_aff max(const isl::pw_aff &pwaff2) const;
  inline isl::multi_val max_multi_val() const;
  inline isl::multi_pw_aff min(const isl::multi_pw_aff &multi2) const;
  inline isl::pw_aff min(const isl::pw_aff &pwaff2) const;
  inline isl::multi_val min_multi_val() const;
  inline isl::aff mod(isl::val mod) const;
  inline isl::aff mod(long mod) const;
  inline isl::aff mul(isl::aff aff2) const;
  inline isl::pw_aff mul(const isl::pw_aff &pwaff2) const;
  inline class size n_piece() const;
  inline isl::set ne_set(isl::aff aff2) const;
  inline isl::set ne_set(const isl::pw_aff &pwaff2) const;
  inline isl::aff neg() const;
  inline boolean plain_is_empty() const;
  inline boolean plain_is_equal(const isl::multi_aff &multi2) const;
  inline boolean plain_is_equal(const isl::multi_pw_aff &multi2) const;
  inline boolean plain_is_equal(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_multi_aff preimage_domain_wrapped_domain(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_multi_aff preimage_domain_wrapped_domain(const isl::union_pw_multi_aff &upma2) const;
  inline isl::multi_aff product(const isl::multi_aff &multi2) const;
  inline isl::multi_pw_aff product(const isl::multi_pw_aff &multi2) const;
  inline isl::pw_multi_aff product(const isl::pw_multi_aff &pma2) const;
  inline isl::aff pullback(isl::multi_aff ma) const;
  inline isl::pw_aff pullback(const isl::multi_pw_aff &mpa) const;
  inline isl::pw_aff pullback(const isl::pw_multi_aff &pma) const;
  inline isl::union_pw_aff pullback(const isl::union_pw_multi_aff &upma) const;
  inline isl::aff pullback(const isl::aff &ma) const;
  inline isl::pw_multi_aff_list pw_multi_aff_list() const;
  inline isl::pw_multi_aff range_factor_domain() const;
  inline isl::pw_multi_aff range_factor_range() const;
  inline isl::multi_aff range_product(const isl::multi_aff &multi2) const;
  inline isl::multi_pw_aff range_product(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff range_product(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_multi_aff range_product(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_multi_aff range_product(const isl::union_pw_multi_aff &upma2) const;
  inline isl::id range_tuple_id() const;
  inline isl::multi_aff reset_range_tuple_id() const;
  inline isl::multi_aff reset_tuple_id(isl::dim type) const;
  inline isl::aff scale(isl::val v) const;
  inline isl::aff scale(long v) const;
  inline isl::multi_aff scale(const isl::multi_val &mv) const;
  inline isl::aff scale_down(isl::val v) const;
  inline isl::aff scale_down(long v) const;
  inline isl::multi_aff scale_down(const isl::multi_val &mv) const;
  inline isl::multi_aff set_aff(int pos, const isl::aff &el) const;
  inline isl::multi_aff set_at(int pos, const isl::aff &el) const;
  inline isl::multi_pw_aff set_at(int pos, const isl::pw_aff &el) const;
  inline isl::multi_union_pw_aff set_at(int pos, const isl::union_pw_aff &el) const;
  inline isl::aff set_constant_si(int v) const;
  inline isl::multi_pw_aff set_pw_aff(int pos, const isl::pw_aff &el) const;
  inline isl::pw_multi_aff set_pw_aff(unsigned int pos, const isl::pw_aff &pa) const;
  inline isl::multi_aff set_range_tuple(const isl::id &id) const;
  inline isl::multi_aff set_range_tuple(const std::string &id) const;
  inline isl::pw_aff set_tuple_id(isl::dim type, const isl::id &id) const;
  inline isl::pw_aff set_tuple_id(isl::dim type, const std::string &id) const;
  inline isl::multi_union_pw_aff set_union_pw_aff(int pos, const isl::union_pw_aff &el) const;
  inline class size size() const;
  inline isl::space space() const;
  inline isl::aff sub(isl::aff aff2) const;
  inline isl::multi_aff sub(const isl::multi_aff &multi2) const;
  inline isl::multi_pw_aff sub(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff sub(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_aff sub(const isl::pw_aff &pwaff2) const;
  inline isl::pw_multi_aff sub(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_aff sub(const isl::union_pw_aff &upa2) const;
  inline isl::union_pw_multi_aff sub(const isl::union_pw_multi_aff &upma2) const;
  inline isl::pw_aff subtract_domain(const isl::set &set) const;
  inline isl::union_pw_aff subtract_domain(const isl::space &space) const;
  inline isl::union_pw_aff subtract_domain(const isl::union_set &uset) const;
  inline isl::pw_aff tdiv_q(const isl::pw_aff &pa2) const;
  inline isl::pw_aff tdiv_r(const isl::pw_aff &pa2) const;
  inline isl::aff_list to_list() const;
  inline isl::multi_pw_aff to_multi_pw_aff() const;
  inline isl::multi_union_pw_aff to_multi_union_pw_aff() const;
  inline isl::pw_multi_aff to_pw_multi_aff() const;
  inline isl::union_pw_aff to_union_pw_aff() const;
  inline isl::union_pw_multi_aff to_union_pw_multi_aff() const;
  inline isl::id tuple_id(isl::dim type) const;
  inline isl::aff unbind_params_insert_domain(isl::multi_id domain) const;
  inline isl::multi_pw_aff union_add(const isl::multi_pw_aff &mpa2) const;
  inline isl::multi_union_pw_aff union_add(const isl::multi_union_pw_aff &mupa2) const;
  inline isl::pw_aff union_add(const isl::pw_aff &pwaff2) const;
  inline isl::pw_multi_aff union_add(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_aff union_add(const isl::union_pw_aff &upa2) const;
  inline isl::union_pw_multi_aff union_add(const isl::union_pw_multi_aff &upma2) const;
  static inline isl::aff var_on_domain(isl::local_space ls, isl::dim type, unsigned int pos);
  static inline isl::aff zero_on_domain(isl::space space);
};

// declarations for isl::aff_list
inline aff_list manage(__isl_take isl_aff_list *ptr);
inline aff_list manage_copy(__isl_keep isl_aff_list *ptr);

class aff_list {
  friend inline aff_list manage(__isl_take isl_aff_list *ptr);
  friend inline aff_list manage_copy(__isl_keep isl_aff_list *ptr);

protected:
  isl_aff_list *ptr = nullptr;

  inline explicit aff_list(__isl_take isl_aff_list *ptr);

public:
  inline /* implicit */ aff_list();
  inline /* implicit */ aff_list(const aff_list &obj);
  inline explicit aff_list(isl::ctx ctx, int n);
  inline explicit aff_list(isl::aff el);
  inline explicit aff_list(isl::ctx ctx, const std::string &str);
  inline aff_list &operator=(aff_list obj);
  inline ~aff_list();
  inline __isl_give isl_aff_list *copy() const &;
  inline __isl_give isl_aff_list *copy() && = delete;
  inline __isl_keep isl_aff_list *get() const;
  inline __isl_give isl_aff_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::aff_list add(isl::aff el) const;
  inline isl::aff at(int index) const;
  inline isl::aff get_at(int index) const;
  inline isl::aff_list clear() const;
  inline isl::aff_list concat(isl::aff_list list2) const;
  inline isl::aff_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(isl::aff)> &fn) const;
  inline isl::aff_list insert(unsigned int pos, isl::aff el) const;
  inline class size size() const;
};

// declarations for isl::ast_build
inline ast_build manage(__isl_take isl_ast_build *ptr);
inline ast_build manage_copy(__isl_keep isl_ast_build *ptr);

class ast_build {
  friend inline ast_build manage(__isl_take isl_ast_build *ptr);
  friend inline ast_build manage_copy(__isl_keep isl_ast_build *ptr);

protected:
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

private:
  inline ast_build &copy_callbacks(const ast_build &obj);
  struct at_each_domain_data {
    std::function<isl::ast_node(isl::ast_node, isl::ast_build)> func;
  };
  std::shared_ptr<at_each_domain_data> at_each_domain_data;
  static inline isl_ast_node *at_each_domain(isl_ast_node *arg_0, isl_ast_build *arg_1, void *arg_2);
  inline void set_at_each_domain_data(const std::function<isl::ast_node(isl::ast_node, isl::ast_build)> &fn);
public:
  inline isl::ast_build set_at_each_domain(const std::function<isl::ast_node(isl::ast_node, isl::ast_build)> &fn) const;
  inline isl::ast_expr access_from(isl::multi_pw_aff mpa) const;
  inline isl::ast_expr access_from(isl::pw_multi_aff pma) const;
  inline isl::ast_expr call_from(isl::multi_pw_aff mpa) const;
  inline isl::ast_expr call_from(isl::pw_multi_aff pma) const;
  inline isl::ast_expr expr_from(isl::pw_aff pa) const;
  inline isl::ast_expr expr_from(isl::set set) const;
  static inline isl::ast_build from_context(isl::set set);
  inline isl::ast_node node_from(isl::schedule schedule) const;
  inline isl::ast_node node_from_schedule_map(isl::union_map schedule) const;
  inline isl::ast_build restrict(isl::set set) const;
  inline isl::union_map schedule() const;
  inline isl::union_map get_schedule() const;
};

// declarations for isl::ast_expr
inline ast_expr manage(__isl_take isl_ast_expr *ptr);
inline ast_expr manage_copy(__isl_keep isl_ast_expr *ptr);

class ast_expr {
  friend inline ast_expr manage(__isl_take isl_ast_expr *ptr);
  friend inline ast_expr manage_copy(__isl_keep isl_ast_expr *ptr);

protected:
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
private:
  template <typename T,
          typename = typename std::enable_if<std::is_same<
                  const decltype(isl_ast_expr_get_type(NULL)),
                  const T>::value>::type>
  inline boolean isa_type(T subtype) const;
public:
  template <class T> inline boolean isa() const;
  template <class T> inline T as() const;
  inline isl::ctx ctx() const;

  inline isl::ast_expr add(isl::ast_expr expr2) const;
  inline isl::ast_expr address_of() const;
  inline isl::ast_expr eq(isl::ast_expr expr2) const;
  static inline isl::ast_expr from_val(isl::val v);
  inline isl::id id() const;
  inline isl::id get_id() const;
  inline isl::ast_expr le(isl::ast_expr expr2) const;
  inline isl::ast_expr mul(isl::ast_expr expr2) const;
  inline isl::ast_expr op_arg(int pos) const;
  inline isl::ast_expr get_op_arg(int pos) const;
  inline std::string to_C_str() const;
  inline isl::val val() const;
  inline isl::val get_val() const;
};

// declarations for isl::ast_expr_id

class ast_expr_id : public ast_expr {
  template <class T>
  friend boolean ast_expr::isa() const;
  friend ast_expr_id ast_expr::as<ast_expr_id>() const;
  static const auto type = isl_ast_expr_id;

protected:
  inline explicit ast_expr_id(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_id();
  inline /* implicit */ ast_expr_id(const ast_expr_id &obj);
  inline ast_expr_id &operator=(ast_expr_id obj);
  inline isl::ctx ctx() const;

  inline isl::id id() const;
  inline isl::id get_id() const;
};

// declarations for isl::ast_expr_int

class ast_expr_int : public ast_expr {
  template <class T>
  friend boolean ast_expr::isa() const;
  friend ast_expr_int ast_expr::as<ast_expr_int>() const;
  static const auto type = isl_ast_expr_int;

protected:
  inline explicit ast_expr_int(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_int();
  inline /* implicit */ ast_expr_int(const ast_expr_int &obj);
  inline ast_expr_int &operator=(ast_expr_int obj);
  inline isl::ctx ctx() const;

  inline isl::val val() const;
  inline isl::val get_val() const;
};

// declarations for isl::ast_expr_op

class ast_expr_op : public ast_expr {
  template <class T>
  friend boolean ast_expr::isa() const;
  friend ast_expr_op ast_expr::as<ast_expr_op>() const;
  static const auto type = isl_ast_expr_op;

protected:
  inline explicit ast_expr_op(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op();
  inline /* implicit */ ast_expr_op(const ast_expr_op &obj);
  inline ast_expr_op &operator=(ast_expr_op obj);
private:
  template <typename T,
          typename = typename std::enable_if<std::is_same<
                  const decltype(isl_ast_expr_op_get_type(NULL)),
                  const T>::value>::type>
  inline boolean isa_type(T subtype) const;
public:
  template <class T> inline boolean isa() const;
  template <class T> inline T as() const;
  inline isl::ctx ctx() const;

  inline isl::ast_expr arg(int pos) const;
  inline isl::ast_expr get_arg(int pos) const;
  inline class size n_arg() const;
  inline class size get_n_arg() const;
};

// declarations for isl::ast_expr_op_access

class ast_expr_op_access : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_access ast_expr_op::as<ast_expr_op_access>() const;
  static const auto type = isl_ast_expr_op_access;

protected:
  inline explicit ast_expr_op_access(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_access();
  inline /* implicit */ ast_expr_op_access(const ast_expr_op_access &obj);
  inline ast_expr_op_access &operator=(ast_expr_op_access obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_add

class ast_expr_op_add : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_add ast_expr_op::as<ast_expr_op_add>() const;
  static const auto type = isl_ast_expr_op_add;

protected:
  inline explicit ast_expr_op_add(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_add();
  inline /* implicit */ ast_expr_op_add(const ast_expr_op_add &obj);
  inline ast_expr_op_add &operator=(ast_expr_op_add obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_address_of

class ast_expr_op_address_of : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_address_of ast_expr_op::as<ast_expr_op_address_of>() const;
  static const auto type = isl_ast_expr_op_address_of;

protected:
  inline explicit ast_expr_op_address_of(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_address_of();
  inline /* implicit */ ast_expr_op_address_of(const ast_expr_op_address_of &obj);
  inline ast_expr_op_address_of &operator=(ast_expr_op_address_of obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_and

class ast_expr_op_and : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_and ast_expr_op::as<ast_expr_op_and>() const;
  static const auto type = isl_ast_expr_op_and;

protected:
  inline explicit ast_expr_op_and(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_and();
  inline /* implicit */ ast_expr_op_and(const ast_expr_op_and &obj);
  inline ast_expr_op_and &operator=(ast_expr_op_and obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_and_then

class ast_expr_op_and_then : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_and_then ast_expr_op::as<ast_expr_op_and_then>() const;
  static const auto type = isl_ast_expr_op_and_then;

protected:
  inline explicit ast_expr_op_and_then(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_and_then();
  inline /* implicit */ ast_expr_op_and_then(const ast_expr_op_and_then &obj);
  inline ast_expr_op_and_then &operator=(ast_expr_op_and_then obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_call

class ast_expr_op_call : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_call ast_expr_op::as<ast_expr_op_call>() const;
  static const auto type = isl_ast_expr_op_call;

protected:
  inline explicit ast_expr_op_call(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_call();
  inline /* implicit */ ast_expr_op_call(const ast_expr_op_call &obj);
  inline ast_expr_op_call &operator=(ast_expr_op_call obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_cond

class ast_expr_op_cond : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_cond ast_expr_op::as<ast_expr_op_cond>() const;
  static const auto type = isl_ast_expr_op_cond;

protected:
  inline explicit ast_expr_op_cond(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_cond();
  inline /* implicit */ ast_expr_op_cond(const ast_expr_op_cond &obj);
  inline ast_expr_op_cond &operator=(ast_expr_op_cond obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_div

class ast_expr_op_div : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_div ast_expr_op::as<ast_expr_op_div>() const;
  static const auto type = isl_ast_expr_op_div;

protected:
  inline explicit ast_expr_op_div(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_div();
  inline /* implicit */ ast_expr_op_div(const ast_expr_op_div &obj);
  inline ast_expr_op_div &operator=(ast_expr_op_div obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_eq

class ast_expr_op_eq : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_eq ast_expr_op::as<ast_expr_op_eq>() const;
  static const auto type = isl_ast_expr_op_eq;

protected:
  inline explicit ast_expr_op_eq(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_eq();
  inline /* implicit */ ast_expr_op_eq(const ast_expr_op_eq &obj);
  inline ast_expr_op_eq &operator=(ast_expr_op_eq obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_fdiv_q

class ast_expr_op_fdiv_q : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_fdiv_q ast_expr_op::as<ast_expr_op_fdiv_q>() const;
  static const auto type = isl_ast_expr_op_fdiv_q;

protected:
  inline explicit ast_expr_op_fdiv_q(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_fdiv_q();
  inline /* implicit */ ast_expr_op_fdiv_q(const ast_expr_op_fdiv_q &obj);
  inline ast_expr_op_fdiv_q &operator=(ast_expr_op_fdiv_q obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_ge

class ast_expr_op_ge : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_ge ast_expr_op::as<ast_expr_op_ge>() const;
  static const auto type = isl_ast_expr_op_ge;

protected:
  inline explicit ast_expr_op_ge(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_ge();
  inline /* implicit */ ast_expr_op_ge(const ast_expr_op_ge &obj);
  inline ast_expr_op_ge &operator=(ast_expr_op_ge obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_gt

class ast_expr_op_gt : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_gt ast_expr_op::as<ast_expr_op_gt>() const;
  static const auto type = isl_ast_expr_op_gt;

protected:
  inline explicit ast_expr_op_gt(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_gt();
  inline /* implicit */ ast_expr_op_gt(const ast_expr_op_gt &obj);
  inline ast_expr_op_gt &operator=(ast_expr_op_gt obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_le

class ast_expr_op_le : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_le ast_expr_op::as<ast_expr_op_le>() const;
  static const auto type = isl_ast_expr_op_le;

protected:
  inline explicit ast_expr_op_le(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_le();
  inline /* implicit */ ast_expr_op_le(const ast_expr_op_le &obj);
  inline ast_expr_op_le &operator=(ast_expr_op_le obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_lt

class ast_expr_op_lt : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_lt ast_expr_op::as<ast_expr_op_lt>() const;
  static const auto type = isl_ast_expr_op_lt;

protected:
  inline explicit ast_expr_op_lt(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_lt();
  inline /* implicit */ ast_expr_op_lt(const ast_expr_op_lt &obj);
  inline ast_expr_op_lt &operator=(ast_expr_op_lt obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_max

class ast_expr_op_max : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_max ast_expr_op::as<ast_expr_op_max>() const;
  static const auto type = isl_ast_expr_op_max;

protected:
  inline explicit ast_expr_op_max(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_max();
  inline /* implicit */ ast_expr_op_max(const ast_expr_op_max &obj);
  inline ast_expr_op_max &operator=(ast_expr_op_max obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_member

class ast_expr_op_member : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_member ast_expr_op::as<ast_expr_op_member>() const;
  static const auto type = isl_ast_expr_op_member;

protected:
  inline explicit ast_expr_op_member(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_member();
  inline /* implicit */ ast_expr_op_member(const ast_expr_op_member &obj);
  inline ast_expr_op_member &operator=(ast_expr_op_member obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_min

class ast_expr_op_min : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_min ast_expr_op::as<ast_expr_op_min>() const;
  static const auto type = isl_ast_expr_op_min;

protected:
  inline explicit ast_expr_op_min(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_min();
  inline /* implicit */ ast_expr_op_min(const ast_expr_op_min &obj);
  inline ast_expr_op_min &operator=(ast_expr_op_min obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_minus

class ast_expr_op_minus : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_minus ast_expr_op::as<ast_expr_op_minus>() const;
  static const auto type = isl_ast_expr_op_minus;

protected:
  inline explicit ast_expr_op_minus(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_minus();
  inline /* implicit */ ast_expr_op_minus(const ast_expr_op_minus &obj);
  inline ast_expr_op_minus &operator=(ast_expr_op_minus obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_mul

class ast_expr_op_mul : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_mul ast_expr_op::as<ast_expr_op_mul>() const;
  static const auto type = isl_ast_expr_op_mul;

protected:
  inline explicit ast_expr_op_mul(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_mul();
  inline /* implicit */ ast_expr_op_mul(const ast_expr_op_mul &obj);
  inline ast_expr_op_mul &operator=(ast_expr_op_mul obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_or

class ast_expr_op_or : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_or ast_expr_op::as<ast_expr_op_or>() const;
  static const auto type = isl_ast_expr_op_or;

protected:
  inline explicit ast_expr_op_or(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_or();
  inline /* implicit */ ast_expr_op_or(const ast_expr_op_or &obj);
  inline ast_expr_op_or &operator=(ast_expr_op_or obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_or_else

class ast_expr_op_or_else : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_or_else ast_expr_op::as<ast_expr_op_or_else>() const;
  static const auto type = isl_ast_expr_op_or_else;

protected:
  inline explicit ast_expr_op_or_else(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_or_else();
  inline /* implicit */ ast_expr_op_or_else(const ast_expr_op_or_else &obj);
  inline ast_expr_op_or_else &operator=(ast_expr_op_or_else obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_pdiv_q

class ast_expr_op_pdiv_q : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_pdiv_q ast_expr_op::as<ast_expr_op_pdiv_q>() const;
  static const auto type = isl_ast_expr_op_pdiv_q;

protected:
  inline explicit ast_expr_op_pdiv_q(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_pdiv_q();
  inline /* implicit */ ast_expr_op_pdiv_q(const ast_expr_op_pdiv_q &obj);
  inline ast_expr_op_pdiv_q &operator=(ast_expr_op_pdiv_q obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_pdiv_r

class ast_expr_op_pdiv_r : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_pdiv_r ast_expr_op::as<ast_expr_op_pdiv_r>() const;
  static const auto type = isl_ast_expr_op_pdiv_r;

protected:
  inline explicit ast_expr_op_pdiv_r(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_pdiv_r();
  inline /* implicit */ ast_expr_op_pdiv_r(const ast_expr_op_pdiv_r &obj);
  inline ast_expr_op_pdiv_r &operator=(ast_expr_op_pdiv_r obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_select

class ast_expr_op_select : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_select ast_expr_op::as<ast_expr_op_select>() const;
  static const auto type = isl_ast_expr_op_select;

protected:
  inline explicit ast_expr_op_select(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_select();
  inline /* implicit */ ast_expr_op_select(const ast_expr_op_select &obj);
  inline ast_expr_op_select &operator=(ast_expr_op_select obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_sub

class ast_expr_op_sub : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_sub ast_expr_op::as<ast_expr_op_sub>() const;
  static const auto type = isl_ast_expr_op_sub;

protected:
  inline explicit ast_expr_op_sub(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_sub();
  inline /* implicit */ ast_expr_op_sub(const ast_expr_op_sub &obj);
  inline ast_expr_op_sub &operator=(ast_expr_op_sub obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_expr_op_zdiv_r

class ast_expr_op_zdiv_r : public ast_expr_op {
  template <class T>
  friend boolean ast_expr_op::isa() const;
  friend ast_expr_op_zdiv_r ast_expr_op::as<ast_expr_op_zdiv_r>() const;
  static const auto type = isl_ast_expr_op_zdiv_r;

protected:
  inline explicit ast_expr_op_zdiv_r(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op_zdiv_r();
  inline /* implicit */ ast_expr_op_zdiv_r(const ast_expr_op_zdiv_r &obj);
  inline ast_expr_op_zdiv_r &operator=(ast_expr_op_zdiv_r obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::ast_node
inline ast_node manage(__isl_take isl_ast_node *ptr);
inline ast_node manage_copy(__isl_keep isl_ast_node *ptr);

class ast_node {
  friend inline ast_node manage(__isl_take isl_ast_node *ptr);
  friend inline ast_node manage_copy(__isl_keep isl_ast_node *ptr);

protected:
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
private:
  template <typename T,
          typename = typename std::enable_if<std::is_same<
                  const decltype(isl_ast_node_get_type(NULL)),
                  const T>::value>::type>
  inline boolean isa_type(T subtype) const;
public:
  template <class T> inline boolean isa() const;
  template <class T> inline T as() const;
  inline isl::ctx ctx() const;

  inline isl::id annotation() const;
  inline isl::id get_annotation() const;
  inline std::string to_C_str() const;
  inline isl::ast_node_list to_list() const;
};

// declarations for isl::ast_node_block

class ast_node_block : public ast_node {
  template <class T>
  friend boolean ast_node::isa() const;
  friend ast_node_block ast_node::as<ast_node_block>() const;
  static const auto type = isl_ast_node_block;

protected:
  inline explicit ast_node_block(__isl_take isl_ast_node *ptr);

public:
  inline /* implicit */ ast_node_block();
  inline /* implicit */ ast_node_block(const ast_node_block &obj);
  inline ast_node_block &operator=(ast_node_block obj);
  inline isl::ctx ctx() const;

  inline isl::ast_node_list children() const;
  inline isl::ast_node_list get_children() const;
};

// declarations for isl::ast_node_for

class ast_node_for : public ast_node {
  template <class T>
  friend boolean ast_node::isa() const;
  friend ast_node_for ast_node::as<ast_node_for>() const;
  static const auto type = isl_ast_node_for;

protected:
  inline explicit ast_node_for(__isl_take isl_ast_node *ptr);

public:
  inline /* implicit */ ast_node_for();
  inline /* implicit */ ast_node_for(const ast_node_for &obj);
  inline ast_node_for &operator=(ast_node_for obj);
  inline isl::ctx ctx() const;

  inline isl::ast_node body() const;
  inline isl::ast_node get_body() const;
  inline isl::ast_expr cond() const;
  inline isl::ast_expr get_cond() const;
  inline isl::ast_expr inc() const;
  inline isl::ast_expr get_inc() const;
  inline isl::ast_expr init() const;
  inline isl::ast_expr get_init() const;
  inline boolean is_degenerate() const;
  inline isl::ast_expr iterator() const;
  inline isl::ast_expr get_iterator() const;
};

// declarations for isl::ast_node_if

class ast_node_if : public ast_node {
  template <class T>
  friend boolean ast_node::isa() const;
  friend ast_node_if ast_node::as<ast_node_if>() const;
  static const auto type = isl_ast_node_if;

protected:
  inline explicit ast_node_if(__isl_take isl_ast_node *ptr);

public:
  inline /* implicit */ ast_node_if();
  inline /* implicit */ ast_node_if(const ast_node_if &obj);
  inline ast_node_if &operator=(ast_node_if obj);
  inline isl::ctx ctx() const;

  inline isl::ast_expr cond() const;
  inline isl::ast_expr get_cond() const;
  inline isl::ast_node else_node() const;
  inline isl::ast_node get_else_node() const;
  inline boolean has_else_node() const;
  inline isl::ast_node then_node() const;
  inline isl::ast_node get_then_node() const;
};

// declarations for isl::ast_node_list
inline ast_node_list manage(__isl_take isl_ast_node_list *ptr);
inline ast_node_list manage_copy(__isl_keep isl_ast_node_list *ptr);

class ast_node_list {
  friend inline ast_node_list manage(__isl_take isl_ast_node_list *ptr);
  friend inline ast_node_list manage_copy(__isl_keep isl_ast_node_list *ptr);

protected:
  isl_ast_node_list *ptr = nullptr;

  inline explicit ast_node_list(__isl_take isl_ast_node_list *ptr);

public:
  inline /* implicit */ ast_node_list();
  inline /* implicit */ ast_node_list(const ast_node_list &obj);
  inline explicit ast_node_list(isl::ctx ctx, int n);
  inline explicit ast_node_list(isl::ast_node el);
  inline ast_node_list &operator=(ast_node_list obj);
  inline ~ast_node_list();
  inline __isl_give isl_ast_node_list *copy() const &;
  inline __isl_give isl_ast_node_list *copy() && = delete;
  inline __isl_keep isl_ast_node_list *get() const;
  inline __isl_give isl_ast_node_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::ast_node_list add(isl::ast_node el) const;
  inline isl::ast_node at(int index) const;
  inline isl::ast_node get_at(int index) const;
  inline isl::ast_node_list clear() const;
  inline isl::ast_node_list concat(isl::ast_node_list list2) const;
  inline isl::ast_node_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(isl::ast_node)> &fn) const;
  inline isl::ast_node_list insert(unsigned int pos, isl::ast_node el) const;
  inline class size size() const;
};

// declarations for isl::ast_node_mark

class ast_node_mark : public ast_node {
  template <class T>
  friend boolean ast_node::isa() const;
  friend ast_node_mark ast_node::as<ast_node_mark>() const;
  static const auto type = isl_ast_node_mark;

protected:
  inline explicit ast_node_mark(__isl_take isl_ast_node *ptr);

public:
  inline /* implicit */ ast_node_mark();
  inline /* implicit */ ast_node_mark(const ast_node_mark &obj);
  inline ast_node_mark &operator=(ast_node_mark obj);
  inline isl::ctx ctx() const;

  inline isl::id id() const;
  inline isl::id get_id() const;
  inline isl::ast_node node() const;
  inline isl::ast_node get_node() const;
};

// declarations for isl::ast_node_user

class ast_node_user : public ast_node {
  template <class T>
  friend boolean ast_node::isa() const;
  friend ast_node_user ast_node::as<ast_node_user>() const;
  static const auto type = isl_ast_node_user;

protected:
  inline explicit ast_node_user(__isl_take isl_ast_node *ptr);

public:
  inline /* implicit */ ast_node_user();
  inline /* implicit */ ast_node_user(const ast_node_user &obj);
  inline ast_node_user &operator=(ast_node_user obj);
  inline isl::ctx ctx() const;

  inline isl::ast_expr expr() const;
  inline isl::ast_expr get_expr() const;
};

// declarations for isl::basic_map
inline basic_map manage(__isl_take isl_basic_map *ptr);
inline basic_map manage_copy(__isl_keep isl_basic_map *ptr);

class basic_map {
  friend inline basic_map manage(__isl_take isl_basic_map *ptr);
  friend inline basic_map manage_copy(__isl_keep isl_basic_map *ptr);

protected:
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

  inline isl::map add_constraint(const isl::constraint &constraint) const;
  inline isl::map add_dims(isl::dim type, unsigned int n) const;
  inline isl::basic_map affine_hull() const;
  inline isl::map align_params(const isl::space &model) const;
  inline isl::basic_map apply_domain(isl::basic_map bmap2) const;
  inline isl::map apply_domain(const isl::map &map2) const;
  inline isl::union_map apply_domain(const isl::union_map &umap2) const;
  inline isl::basic_map apply_range(isl::basic_map bmap2) const;
  inline isl::map apply_range(const isl::map &map2) const;
  inline isl::union_map apply_range(const isl::union_map &umap2) const;
  inline isl::map as_map() const;
  inline isl::multi_union_pw_aff as_multi_union_pw_aff() const;
  inline isl::pw_multi_aff as_pw_multi_aff() const;
  inline isl::union_pw_multi_aff as_union_pw_multi_aff() const;
  inline isl::basic_map_list basic_map_list() const;
  inline isl::set bind_domain(const isl::multi_id &tuple) const;
  inline isl::set bind_range(const isl::multi_id &tuple) const;
  inline boolean can_curry() const;
  inline isl::map coalesce() const;
  inline isl::map complement() const;
  inline isl::union_map compute_divs() const;
  inline isl::map curry() const;
  inline isl::basic_set deltas() const;
  inline isl::basic_map detect_equalities() const;
  inline class size dim(isl::dim type) const;
  inline isl::pw_aff dim_max(int pos) const;
  inline isl::pw_aff dim_min(int pos) const;
  inline isl::basic_set domain() const;
  inline isl::map domain_factor_domain() const;
  inline isl::map domain_factor_range() const;
  inline isl::map domain_map() const;
  inline isl::union_pw_multi_aff domain_map_union_pw_multi_aff() const;
  inline isl::map domain_product(const isl::map &map2) const;
  inline isl::union_map domain_product(const isl::union_map &umap2) const;
  inline class size domain_tuple_dim() const;
  inline isl::id domain_tuple_id() const;
  inline isl::map eq_at(const isl::multi_pw_aff &mpa) const;
  inline isl::union_map eq_at(const isl::multi_union_pw_aff &mupa) const;
  static inline isl::basic_map equal(isl::space space, unsigned int n_equal);
  inline isl::basic_map equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline boolean every_map(const std::function<boolean(isl::map)> &test) const;
  inline isl::map extract_map(const isl::space &space) const;
  inline isl::map factor_domain() const;
  inline isl::map factor_range() const;
  inline isl::basic_map fix_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::basic_map fix_val(isl::dim type, unsigned int pos, isl::val v) const;
  inline isl::basic_map fix_val(isl::dim type, unsigned int pos, long v) const;
  inline isl::union_map fixed_power(const isl::val &exp) const;
  inline isl::union_map fixed_power(long exp) const;
  inline isl::map flat_range_product(const isl::map &map2) const;
  inline isl::union_map flat_range_product(const isl::union_map &umap2) const;
  inline isl::basic_map flatten() const;
  inline isl::basic_map flatten_domain() const;
  inline isl::basic_map flatten_range() const;
  inline isl::map floordiv_val(const isl::val &d) const;
  inline isl::map floordiv_val(long d) const;
  inline stat foreach_basic_map(const std::function<stat(isl::basic_map)> &fn) const;
  inline stat foreach_map(const std::function<stat(isl::map)> &fn) const;
  static inline isl::basic_map from_aff(isl::aff aff);
  static inline isl::basic_map from_domain_and_range(isl::basic_set domain, isl::basic_set range);
  inline isl::basic_map gist(isl::basic_map context) const;
  inline isl::map gist(const isl::map &context) const;
  inline isl::union_map gist(const isl::union_map &context) const;
  inline isl::map gist_domain(const isl::set &context) const;
  inline isl::union_map gist_domain(const isl::union_set &uset) const;
  inline isl::map gist_params(const isl::set &context) const;
  inline isl::union_map gist_range(const isl::union_set &uset) const;
  inline boolean has_domain_tuple_id() const;
  inline boolean has_equal_space(const isl::map &map2) const;
  inline boolean has_range_tuple_id() const;
  inline boolean has_tuple_id(isl::dim type) const;
  inline boolean has_tuple_name(isl::dim type) const;
  inline isl::basic_map intersect(isl::basic_map bmap2) const;
  inline isl::map intersect(const isl::map &map2) const;
  inline isl::union_map intersect(const isl::union_map &umap2) const;
  inline isl::basic_map intersect_domain(isl::basic_set bset) const;
  inline isl::map intersect_domain(const isl::set &set) const;
  inline isl::union_map intersect_domain(const isl::space &space) const;
  inline isl::union_map intersect_domain(const isl::union_set &uset) const;
  inline isl::basic_map intersect_domain(const isl::point &bset) const;
  inline isl::map intersect_domain_factor_domain(const isl::map &factor) const;
  inline isl::union_map intersect_domain_factor_domain(const isl::union_map &factor) const;
  inline isl::map intersect_domain_factor_range(const isl::map &factor) const;
  inline isl::union_map intersect_domain_factor_range(const isl::union_map &factor) const;
  inline isl::map intersect_params(const isl::set &params) const;
  inline isl::basic_map intersect_range(isl::basic_set bset) const;
  inline isl::map intersect_range(const isl::set &set) const;
  inline isl::union_map intersect_range(const isl::space &space) const;
  inline isl::union_map intersect_range(const isl::union_set &uset) const;
  inline isl::basic_map intersect_range(const isl::point &bset) const;
  inline isl::map intersect_range_factor_domain(const isl::map &factor) const;
  inline isl::union_map intersect_range_factor_domain(const isl::union_map &factor) const;
  inline isl::map intersect_range_factor_range(const isl::map &factor) const;
  inline isl::union_map intersect_range_factor_range(const isl::union_map &factor) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean is_bijective() const;
  inline boolean is_disjoint(const isl::map &map2) const;
  inline boolean is_disjoint(const isl::union_map &umap2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const isl::basic_map &bmap2) const;
  inline boolean is_equal(const isl::map &map2) const;
  inline boolean is_equal(const isl::union_map &umap2) const;
  inline boolean is_injective() const;
  inline boolean is_single_valued() const;
  inline boolean is_strict_subset(const isl::map &map2) const;
  inline boolean is_strict_subset(const isl::union_map &umap2) const;
  inline boolean is_subset(const isl::basic_map &bmap2) const;
  inline boolean is_subset(const isl::map &map2) const;
  inline boolean is_subset(const isl::union_map &umap2) const;
  inline boolean isa_map() const;
  inline isl::map lex_ge_at(const isl::multi_pw_aff &mpa) const;
  inline isl::map lex_gt_at(const isl::multi_pw_aff &mpa) const;
  inline isl::map lex_le_at(const isl::multi_pw_aff &mpa) const;
  inline isl::map lex_lt_at(const isl::multi_pw_aff &mpa) const;
  inline isl::map lexmax() const;
  inline isl::pw_multi_aff lexmax_pw_multi_aff() const;
  inline isl::map lexmin() const;
  inline isl::pw_multi_aff lexmin_pw_multi_aff() const;
  inline isl::map lower_bound(const isl::multi_pw_aff &lower) const;
  inline isl::map lower_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::map_list map_list() const;
  inline isl::multi_pw_aff max_multi_pw_aff() const;
  inline isl::multi_pw_aff min_multi_pw_aff() const;
  inline isl::map move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline class size n_basic_map() const;
  inline isl::map order_lt(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline isl::set params() const;
  inline isl::val plain_get_val_if_fixed(isl::dim type, unsigned int pos) const;
  inline isl::basic_map polyhedral_hull() const;
  inline isl::map preimage_domain(const isl::multi_aff &ma) const;
  inline isl::map preimage_domain(const isl::multi_pw_aff &mpa) const;
  inline isl::map preimage_domain(const isl::pw_multi_aff &pma) const;
  inline isl::union_map preimage_domain(const isl::union_pw_multi_aff &upma) const;
  inline isl::map preimage_range(const isl::multi_aff &ma) const;
  inline isl::map preimage_range(const isl::pw_multi_aff &pma) const;
  inline isl::union_map preimage_range(const isl::union_pw_multi_aff &upma) const;
  inline isl::map product(const isl::map &map2) const;
  inline isl::union_map product(const isl::union_map &umap2) const;
  inline isl::map project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::map project_out_all_params() const;
  inline isl::set range() const;
  inline isl::map range_factor_domain() const;
  inline isl::map range_factor_range() const;
  inline isl::fixed_box range_lattice_tile() const;
  inline isl::map range_map() const;
  inline isl::map range_product(const isl::map &map2) const;
  inline isl::union_map range_product(const isl::union_map &umap2) const;
  inline isl::map range_reverse() const;
  inline isl::fixed_box range_simple_fixed_box_hull() const;
  inline class size range_tuple_dim() const;
  inline isl::id range_tuple_id() const;
  inline isl::basic_map reverse() const;
  inline isl::basic_map sample() const;
  inline isl::map set_domain_tuple(const isl::id &id) const;
  inline isl::map set_domain_tuple(const std::string &id) const;
  inline isl::map set_range_tuple(const isl::id &id) const;
  inline isl::map set_range_tuple(const std::string &id) const;
  inline isl::map set_tuple_id(isl::dim type, const isl::id &id) const;
  inline isl::map set_tuple_id(isl::dim type, const std::string &id) const;
  inline isl::space space() const;
  inline isl::map subtract(const isl::map &map2) const;
  inline isl::union_map subtract(const isl::union_map &umap2) const;
  inline isl::union_map subtract_domain(const isl::union_set &dom) const;
  inline isl::union_map subtract_range(const isl::union_set &dom) const;
  inline isl::map sum(const isl::map &map2) const;
  inline isl::basic_map_list to_list() const;
  inline isl::union_map to_union_map() const;
  inline isl::id tuple_id(isl::dim type) const;
  inline isl::map uncurry() const;
  inline isl::map unite(isl::basic_map bmap2) const;
  inline isl::map unite(const isl::map &map2) const;
  inline isl::union_map unite(const isl::union_map &umap2) const;
  static inline isl::basic_map universe(isl::space space);
  inline isl::basic_map unshifted_simple_hull() const;
  inline isl::map upper_bound(const isl::multi_pw_aff &upper) const;
  inline isl::map upper_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::set wrap() const;
  inline isl::map zip() const;
};

// declarations for isl::basic_map_list
inline basic_map_list manage(__isl_take isl_basic_map_list *ptr);
inline basic_map_list manage_copy(__isl_keep isl_basic_map_list *ptr);

class basic_map_list {
  friend inline basic_map_list manage(__isl_take isl_basic_map_list *ptr);
  friend inline basic_map_list manage_copy(__isl_keep isl_basic_map_list *ptr);

protected:
  isl_basic_map_list *ptr = nullptr;

  inline explicit basic_map_list(__isl_take isl_basic_map_list *ptr);

public:
  inline /* implicit */ basic_map_list();
  inline /* implicit */ basic_map_list(const basic_map_list &obj);
  inline explicit basic_map_list(isl::ctx ctx, int n);
  inline explicit basic_map_list(isl::basic_map el);
  inline basic_map_list &operator=(basic_map_list obj);
  inline ~basic_map_list();
  inline __isl_give isl_basic_map_list *copy() const &;
  inline __isl_give isl_basic_map_list *copy() && = delete;
  inline __isl_keep isl_basic_map_list *get() const;
  inline __isl_give isl_basic_map_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::basic_map_list add(isl::basic_map el) const;
  inline isl::basic_map at(int index) const;
  inline isl::basic_map get_at(int index) const;
  inline isl::basic_map_list clear() const;
  inline isl::basic_map_list concat(isl::basic_map_list list2) const;
  inline isl::basic_map_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(isl::basic_map)> &fn) const;
  inline isl::basic_map_list insert(unsigned int pos, isl::basic_map el) const;
  inline class size size() const;
};

// declarations for isl::basic_set
inline basic_set manage(__isl_take isl_basic_set *ptr);
inline basic_set manage_copy(__isl_keep isl_basic_set *ptr);

class basic_set {
  friend inline basic_set manage(__isl_take isl_basic_set *ptr);
  friend inline basic_set manage_copy(__isl_keep isl_basic_set *ptr);

protected:
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

  inline isl::set add_constraint(const isl::constraint &constraint) const;
  inline isl::set add_dims(isl::dim type, unsigned int n) const;
  inline isl::basic_set affine_hull() const;
  inline isl::set align_params(const isl::space &model) const;
  inline isl::basic_set apply(isl::basic_map bmap) const;
  inline isl::set apply(const isl::map &map) const;
  inline isl::union_set apply(const isl::union_map &umap) const;
  inline isl::pw_multi_aff as_pw_multi_aff() const;
  inline isl::set as_set() const;
  inline isl::basic_set_list basic_set_list() const;
  inline isl::set bind(const isl::multi_id &tuple) const;
  inline isl::set coalesce() const;
  inline isl::set complement() const;
  inline isl::union_set compute_divs() const;
  inline boolean contains(const isl::space &space) const;
  inline isl::basic_set convex_hull() const;
  inline isl::basic_set detect_equalities() const;
  inline class size dim(isl::dim type) const;
  inline boolean dim_has_any_lower_bound(isl::dim type, unsigned int pos) const;
  inline isl::id dim_id(isl::dim type, unsigned int pos) const;
  inline isl::pw_aff dim_max(int pos) const;
  inline isl::val dim_max_val(int pos) const;
  inline isl::pw_aff dim_min(int pos) const;
  inline isl::val dim_min_val(int pos) const;
  inline std::string dim_name(isl::dim type, unsigned int pos) const;
  inline isl::aff div(int pos) const;
  inline isl::aff get_div(int pos) const;
  inline isl::set drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set eliminate(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean every_set(const std::function<boolean(isl::set)> &test) const;
  inline isl::set extract_set(const isl::space &space) const;
  inline int find_dim_by_id(isl::dim type, const isl::id &id) const;
  inline int find_dim_by_id(isl::dim type, const std::string &id) const;
  inline isl::basic_set fix_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::basic_set fix_val(isl::dim type, unsigned int pos, isl::val v) const;
  inline isl::basic_set fix_val(isl::dim type, unsigned int pos, long v) const;
  inline isl::basic_set flatten() const;
  inline stat foreach_basic_set(const std::function<stat(isl::basic_set)> &fn) const;
  inline stat foreach_point(const std::function<stat(isl::point)> &fn) const;
  inline stat foreach_set(const std::function<stat(isl::set)> &fn) const;
  inline isl::basic_set gist(isl::basic_set context) const;
  inline isl::set gist(const isl::set &context) const;
  inline isl::union_set gist(const isl::union_set &context) const;
  inline isl::basic_set gist(const isl::point &context) const;
  inline isl::set gist_params(const isl::set &context) const;
  inline boolean has_equal_space(const isl::set &set2) const;
  inline isl::map identity() const;
  inline isl::union_pw_multi_aff identity_union_pw_multi_aff() const;
  inline isl::pw_aff indicator_function() const;
  inline isl::set insert_dims(isl::dim type, unsigned int pos, unsigned int n) const;
  inline isl::map insert_domain(const isl::space &domain) const;
  inline isl::basic_set intersect(isl::basic_set bset2) const;
  inline isl::set intersect(const isl::set &set2) const;
  inline isl::union_set intersect(const isl::union_set &uset2) const;
  inline isl::basic_set intersect(const isl::point &bset2) const;
  inline isl::basic_set intersect_params(isl::basic_set bset2) const;
  inline isl::set intersect_params(const isl::set &params) const;
  inline isl::basic_set intersect_params(const isl::point &bset2) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_locals() const;
  inline boolean is_bounded() const;
  inline boolean is_disjoint(const isl::set &set2) const;
  inline boolean is_disjoint(const isl::union_set &uset2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const isl::basic_set &bset2) const;
  inline boolean is_equal(const isl::set &set2) const;
  inline boolean is_equal(const isl::union_set &uset2) const;
  inline boolean is_equal(const isl::point &bset2) const;
  inline boolean is_params() const;
  inline boolean is_singleton() const;
  inline boolean is_strict_subset(const isl::set &set2) const;
  inline boolean is_strict_subset(const isl::union_set &uset2) const;
  inline boolean is_subset(const isl::basic_set &bset2) const;
  inline boolean is_subset(const isl::set &set2) const;
  inline boolean is_subset(const isl::union_set &uset2) const;
  inline boolean is_subset(const isl::point &bset2) const;
  inline boolean is_wrapping() const;
  inline boolean isa_set() const;
  inline isl::set lexmax() const;
  inline isl::pw_multi_aff lexmax_pw_multi_aff() const;
  inline isl::set lexmin() const;
  inline isl::pw_multi_aff lexmin_pw_multi_aff() const;
  inline isl::set lower_bound(const isl::multi_pw_aff &lower) const;
  inline isl::set lower_bound(const isl::multi_val &lower) const;
  inline isl::set lower_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::set lower_bound_val(isl::dim type, unsigned int pos, const isl::val &value) const;
  inline isl::set lower_bound_val(isl::dim type, unsigned int pos, long value) const;
  inline isl::multi_pw_aff max_multi_pw_aff() const;
  inline isl::val max_val(const isl::aff &obj) const;
  inline isl::multi_pw_aff min_multi_pw_aff() const;
  inline isl::val min_val(const isl::aff &obj) const;
  inline class size n_basic_set() const;
  inline isl::basic_set params() const;
  inline isl::val plain_get_val_if_fixed(isl::dim type, unsigned int pos) const;
  inline isl::multi_val plain_multi_val_if_fixed() const;
  inline isl::basic_set polyhedral_hull() const;
  inline isl::set preimage(const isl::multi_aff &ma) const;
  inline isl::set preimage(const isl::multi_pw_aff &mpa) const;
  inline isl::set preimage(const isl::pw_multi_aff &pma) const;
  inline isl::union_set preimage(const isl::union_pw_multi_aff &upma) const;
  inline isl::set product(const isl::set &set2) const;
  inline isl::basic_set project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set project_out_all_params() const;
  inline isl::set project_out_param(const isl::id &id) const;
  inline isl::set project_out_param(const std::string &id) const;
  inline isl::set project_out_param(const isl::id_list &list) const;
  inline isl::pw_multi_aff pw_multi_aff_on_domain(const isl::multi_val &mv) const;
  inline isl::set remove_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set remove_divs() const;
  inline isl::set remove_redundancies() const;
  inline isl::set reset_tuple_id() const;
  inline isl::basic_set sample() const;
  inline isl::point sample_point() const;
  inline isl::set set_dim_id(isl::dim type, unsigned int pos, const isl::id &id) const;
  inline isl::set set_dim_id(isl::dim type, unsigned int pos, const std::string &id) const;
  inline isl::set_list set_list() const;
  inline isl::set set_tuple_id(const isl::id &id) const;
  inline isl::set set_tuple_id(const std::string &id) const;
  inline isl::fixed_box simple_fixed_box_hull() const;
  inline isl::basic_set simple_hull() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::val stride(int pos) const;
  inline isl::set subtract(const isl::set &set2) const;
  inline isl::union_set subtract(const isl::union_set &uset2) const;
  inline isl::basic_set_list to_list() const;
  inline isl::set to_set() const;
  inline isl::union_set to_union_set() const;
  inline isl::map translation() const;
  inline class size tuple_dim() const;
  inline isl::id tuple_id() const;
  inline std::string tuple_name() const;
  inline isl::set unbind_params(const isl::multi_id &tuple) const;
  inline isl::map unbind_params_insert_domain(const isl::multi_id &domain) const;
  inline isl::set unite(isl::basic_set bset2) const;
  inline isl::set unite(const isl::set &set2) const;
  inline isl::union_set unite(const isl::union_set &uset2) const;
  inline isl::set unite(const isl::point &bset2) const;
  static inline isl::basic_set universe(isl::space space);
  inline isl::basic_set unshifted_simple_hull() const;
  inline isl::map unwrap() const;
  inline isl::set upper_bound(const isl::multi_pw_aff &upper) const;
  inline isl::set upper_bound(const isl::multi_val &upper) const;
  inline isl::set upper_bound_val(isl::dim type, unsigned int pos, const isl::val &value) const;
  inline isl::set upper_bound_val(isl::dim type, unsigned int pos, long value) const;
};

// declarations for isl::basic_set_list
inline basic_set_list manage(__isl_take isl_basic_set_list *ptr);
inline basic_set_list manage_copy(__isl_keep isl_basic_set_list *ptr);

class basic_set_list {
  friend inline basic_set_list manage(__isl_take isl_basic_set_list *ptr);
  friend inline basic_set_list manage_copy(__isl_keep isl_basic_set_list *ptr);

protected:
  isl_basic_set_list *ptr = nullptr;

  inline explicit basic_set_list(__isl_take isl_basic_set_list *ptr);

public:
  inline /* implicit */ basic_set_list();
  inline /* implicit */ basic_set_list(const basic_set_list &obj);
  inline explicit basic_set_list(isl::ctx ctx, int n);
  inline explicit basic_set_list(isl::basic_set el);
  inline basic_set_list &operator=(basic_set_list obj);
  inline ~basic_set_list();
  inline __isl_give isl_basic_set_list *copy() const &;
  inline __isl_give isl_basic_set_list *copy() && = delete;
  inline __isl_keep isl_basic_set_list *get() const;
  inline __isl_give isl_basic_set_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::basic_set_list add(isl::basic_set el) const;
  inline isl::basic_set at(int index) const;
  inline isl::basic_set get_at(int index) const;
  inline isl::basic_set_list clear() const;
  inline isl::basic_set_list concat(isl::basic_set_list list2) const;
  inline isl::basic_set_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(isl::basic_set)> &fn) const;
  inline isl::basic_set_list insert(unsigned int pos, isl::basic_set el) const;
  inline class size size() const;
};

// declarations for isl::constraint
inline constraint manage(__isl_take isl_constraint *ptr);
inline constraint manage_copy(__isl_keep isl_constraint *ptr);

class constraint {
  friend inline constraint manage(__isl_take isl_constraint *ptr);
  friend inline constraint manage_copy(__isl_keep isl_constraint *ptr);

protected:
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

  static inline isl::constraint alloc_equality(isl::local_space ls);
  static inline isl::constraint alloc_inequality(isl::local_space ls);
  inline isl::constraint set_coefficient_si(isl::dim type, int pos, int v) const;
  inline isl::constraint set_constant_si(int v) const;
  inline isl::constraint set_constant_val(isl::val v) const;
  inline isl::constraint set_constant_val(long v) const;
};

// declarations for isl::fixed_box
inline fixed_box manage(__isl_take isl_fixed_box *ptr);
inline fixed_box manage_copy(__isl_keep isl_fixed_box *ptr);

class fixed_box {
  friend inline fixed_box manage(__isl_take isl_fixed_box *ptr);
  friend inline fixed_box manage_copy(__isl_keep isl_fixed_box *ptr);

protected:
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

  inline boolean is_valid() const;
  inline isl::multi_aff offset() const;
  inline isl::multi_aff get_offset() const;
  inline isl::multi_val size() const;
  inline isl::multi_val get_size() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
};

// declarations for isl::id
inline id manage(__isl_take isl_id *ptr);
inline id manage_copy(__isl_keep isl_id *ptr);

class id {
  friend inline id manage(__isl_take isl_id *ptr);
  friend inline id manage_copy(__isl_keep isl_id *ptr);

protected:
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

  static inline isl::id alloc(isl::ctx ctx, const std::string &name, void * user);
  inline std::string name() const;
  inline std::string get_name() const;
  inline isl::id_list to_list() const;
  inline void * user() const;
  inline void * get_user() const;
};

// declarations for isl::id_list
inline id_list manage(__isl_take isl_id_list *ptr);
inline id_list manage_copy(__isl_keep isl_id_list *ptr);

class id_list {
  friend inline id_list manage(__isl_take isl_id_list *ptr);
  friend inline id_list manage_copy(__isl_keep isl_id_list *ptr);

protected:
  isl_id_list *ptr = nullptr;

  inline explicit id_list(__isl_take isl_id_list *ptr);

public:
  inline /* implicit */ id_list();
  inline /* implicit */ id_list(const id_list &obj);
  inline explicit id_list(isl::ctx ctx, int n);
  inline explicit id_list(isl::id el);
  inline explicit id_list(isl::ctx ctx, const std::string &str);
  inline id_list &operator=(id_list obj);
  inline ~id_list();
  inline __isl_give isl_id_list *copy() const &;
  inline __isl_give isl_id_list *copy() && = delete;
  inline __isl_keep isl_id_list *get() const;
  inline __isl_give isl_id_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::id_list add(isl::id el) const;
  inline isl::id_list add(const std::string &el) const;
  inline isl::id at(int index) const;
  inline isl::id get_at(int index) const;
  inline isl::id_list clear() const;
  inline isl::id_list concat(isl::id_list list2) const;
  inline isl::id_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(isl::id)> &fn) const;
  inline isl::id_list insert(unsigned int pos, isl::id el) const;
  inline isl::id_list insert(unsigned int pos, const std::string &el) const;
  inline class size size() const;
};

// declarations for isl::id_to_ast_expr
inline id_to_ast_expr manage(__isl_take isl_id_to_ast_expr *ptr);
inline id_to_ast_expr manage_copy(__isl_keep isl_id_to_ast_expr *ptr);

class id_to_ast_expr {
  friend inline id_to_ast_expr manage(__isl_take isl_id_to_ast_expr *ptr);
  friend inline id_to_ast_expr manage_copy(__isl_keep isl_id_to_ast_expr *ptr);

protected:
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

  static inline isl::id_to_ast_expr alloc(isl::ctx ctx, int min_size);
  inline isl::id_to_ast_expr set(isl::id key, isl::ast_expr val) const;
  inline isl::id_to_ast_expr set(const std::string &key, const isl::ast_expr &val) const;
};

// declarations for isl::local_space
inline local_space manage(__isl_take isl_local_space *ptr);
inline local_space manage_copy(__isl_keep isl_local_space *ptr);

class local_space {
  friend inline local_space manage(__isl_take isl_local_space *ptr);
  friend inline local_space manage_copy(__isl_keep isl_local_space *ptr);

protected:
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

};

// declarations for isl::map
inline map manage(__isl_take isl_map *ptr);
inline map manage_copy(__isl_keep isl_map *ptr);

class map {
  friend inline map manage(__isl_take isl_map *ptr);
  friend inline map manage_copy(__isl_keep isl_map *ptr);

protected:
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

  inline isl::map add_constraint(isl::constraint constraint) const;
  inline isl::map add_dims(isl::dim type, unsigned int n) const;
  inline isl::basic_map affine_hull() const;
  inline isl::map align_params(isl::space model) const;
  inline isl::map apply_domain(isl::map map2) const;
  inline isl::union_map apply_domain(const isl::union_map &umap2) const;
  inline isl::map apply_domain(const isl::basic_map &map2) const;
  inline isl::map apply_range(isl::map map2) const;
  inline isl::union_map apply_range(const isl::union_map &umap2) const;
  inline isl::map apply_range(const isl::basic_map &map2) const;
  inline isl::map as_map() const;
  inline isl::multi_union_pw_aff as_multi_union_pw_aff() const;
  inline isl::pw_multi_aff as_pw_multi_aff() const;
  inline isl::union_pw_multi_aff as_union_pw_multi_aff() const;
  inline isl::basic_map_list basic_map_list() const;
  inline isl::basic_map_list get_basic_map_list() const;
  inline isl::set bind_domain(isl::multi_id tuple) const;
  inline isl::set bind_range(isl::multi_id tuple) const;
  inline boolean can_curry() const;
  inline isl::map coalesce() const;
  inline isl::map complement() const;
  inline isl::union_map compute_divs() const;
  inline isl::map curry() const;
  inline isl::set deltas() const;
  inline isl::map detect_equalities() const;
  inline class size dim(isl::dim type) const;
  inline isl::pw_aff dim_max(int pos) const;
  inline isl::pw_aff dim_min(int pos) const;
  inline isl::set domain() const;
  inline isl::map domain_factor_domain() const;
  inline isl::map domain_factor_range() const;
  inline isl::map domain_map() const;
  inline isl::union_pw_multi_aff domain_map_union_pw_multi_aff() const;
  inline isl::map domain_product(isl::map map2) const;
  inline isl::union_map domain_product(const isl::union_map &umap2) const;
  inline isl::map domain_product(const isl::basic_map &map2) const;
  inline class size domain_tuple_dim() const;
  inline isl::id domain_tuple_id() const;
  inline isl::id get_domain_tuple_id() const;
  static inline isl::map empty(isl::space space);
  inline isl::map eq_at(isl::multi_pw_aff mpa) const;
  inline isl::union_map eq_at(const isl::multi_union_pw_aff &mupa) const;
  inline isl::map eq_at(const isl::aff &mpa) const;
  inline isl::map eq_at(const isl::multi_aff &mpa) const;
  inline isl::map eq_at(const isl::pw_aff &mpa) const;
  inline isl::map eq_at(const isl::pw_multi_aff &mpa) const;
  inline isl::map equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline boolean every_map(const std::function<boolean(isl::map)> &test) const;
  inline isl::map extract_map(const isl::space &space) const;
  inline isl::map factor_domain() const;
  inline isl::map factor_range() const;
  inline isl::map fix_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::union_map fixed_power(const isl::val &exp) const;
  inline isl::union_map fixed_power(long exp) const;
  inline isl::map flat_range_product(isl::map map2) const;
  inline isl::union_map flat_range_product(const isl::union_map &umap2) const;
  inline isl::map flat_range_product(const isl::basic_map &map2) const;
  inline isl::map flatten() const;
  inline isl::map flatten_domain() const;
  inline isl::map flatten_range() const;
  inline isl::map floordiv_val(isl::val d) const;
  inline isl::map floordiv_val(long d) const;
  inline stat foreach_basic_map(const std::function<stat(isl::basic_map)> &fn) const;
  inline stat foreach_map(const std::function<stat(isl::map)> &fn) const;
  static inline isl::map from_aff(isl::aff aff);
  static inline isl::map from_domain(isl::set set);
  static inline isl::map from_domain_and_range(isl::set domain, isl::set range);
  static inline isl::map from_multi_aff(isl::multi_aff maff);
  static inline isl::map from_pw_aff(isl::pw_aff pwaff);
  static inline isl::map from_range(isl::set set);
  static inline isl::map from_union_map(isl::union_map umap);
  inline isl::map gist(isl::map context) const;
  inline isl::union_map gist(const isl::union_map &context) const;
  inline isl::map gist(const isl::basic_map &context) const;
  inline isl::map gist_domain(isl::set context) const;
  inline isl::union_map gist_domain(const isl::union_set &uset) const;
  inline isl::map gist_domain(const isl::basic_set &context) const;
  inline isl::map gist_domain(const isl::point &context) const;
  inline isl::map gist_params(isl::set context) const;
  inline isl::union_map gist_range(const isl::union_set &uset) const;
  inline boolean has_domain_tuple_id() const;
  inline boolean has_equal_space(const isl::map &map2) const;
  inline boolean has_range_tuple_id() const;
  inline boolean has_tuple_id(isl::dim type) const;
  inline boolean has_tuple_name(isl::dim type) const;
  static inline isl::map identity(isl::space space);
  inline isl::map intersect(isl::map map2) const;
  inline isl::union_map intersect(const isl::union_map &umap2) const;
  inline isl::map intersect(const isl::basic_map &map2) const;
  inline isl::map intersect_domain(isl::set set) const;
  inline isl::union_map intersect_domain(const isl::space &space) const;
  inline isl::union_map intersect_domain(const isl::union_set &uset) const;
  inline isl::map intersect_domain(const isl::basic_set &set) const;
  inline isl::map intersect_domain(const isl::point &set) const;
  inline isl::map intersect_domain_factor_domain(isl::map factor) const;
  inline isl::union_map intersect_domain_factor_domain(const isl::union_map &factor) const;
  inline isl::map intersect_domain_factor_domain(const isl::basic_map &factor) const;
  inline isl::map intersect_domain_factor_range(isl::map factor) const;
  inline isl::union_map intersect_domain_factor_range(const isl::union_map &factor) const;
  inline isl::map intersect_domain_factor_range(const isl::basic_map &factor) const;
  inline isl::map intersect_params(isl::set params) const;
  inline isl::map intersect_range(isl::set set) const;
  inline isl::union_map intersect_range(const isl::space &space) const;
  inline isl::union_map intersect_range(const isl::union_set &uset) const;
  inline isl::map intersect_range(const isl::basic_set &set) const;
  inline isl::map intersect_range(const isl::point &set) const;
  inline isl::map intersect_range_factor_domain(isl::map factor) const;
  inline isl::union_map intersect_range_factor_domain(const isl::union_map &factor) const;
  inline isl::map intersect_range_factor_domain(const isl::basic_map &factor) const;
  inline isl::map intersect_range_factor_range(isl::map factor) const;
  inline isl::union_map intersect_range_factor_range(const isl::union_map &factor) const;
  inline isl::map intersect_range_factor_range(const isl::basic_map &factor) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean is_bijective() const;
  inline boolean is_disjoint(const isl::map &map2) const;
  inline boolean is_disjoint(const isl::union_map &umap2) const;
  inline boolean is_disjoint(const isl::basic_map &map2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const isl::map &map2) const;
  inline boolean is_equal(const isl::union_map &umap2) const;
  inline boolean is_equal(const isl::basic_map &map2) const;
  inline boolean is_injective() const;
  inline boolean is_single_valued() const;
  inline boolean is_strict_subset(const isl::map &map2) const;
  inline boolean is_strict_subset(const isl::union_map &umap2) const;
  inline boolean is_strict_subset(const isl::basic_map &map2) const;
  inline boolean is_subset(const isl::map &map2) const;
  inline boolean is_subset(const isl::union_map &umap2) const;
  inline boolean is_subset(const isl::basic_map &map2) const;
  inline boolean isa_map() const;
  static inline isl::map lex_ge(isl::space set_space);
  inline isl::map lex_ge_at(isl::multi_pw_aff mpa) const;
  static inline isl::map lex_gt(isl::space set_space);
  inline isl::map lex_gt_at(isl::multi_pw_aff mpa) const;
  static inline isl::map lex_le(isl::space set_space);
  inline isl::map lex_le_at(isl::multi_pw_aff mpa) const;
  static inline isl::map lex_lt(isl::space set_space);
  inline isl::map lex_lt_at(isl::multi_pw_aff mpa) const;
  inline isl::map lexmax() const;
  inline isl::pw_multi_aff lexmax_pw_multi_aff() const;
  inline isl::map lexmin() const;
  inline isl::pw_multi_aff lexmin_pw_multi_aff() const;
  inline isl::map lower_bound(isl::multi_pw_aff lower) const;
  inline isl::map lower_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::map_list map_list() const;
  inline isl::multi_pw_aff max_multi_pw_aff() const;
  inline isl::multi_pw_aff min_multi_pw_aff() const;
  inline isl::map move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline class size n_basic_map() const;
  inline isl::map order_lt(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline isl::set params() const;
  inline isl::basic_map polyhedral_hull() const;
  inline isl::map preimage_domain(isl::multi_aff ma) const;
  inline isl::map preimage_domain(isl::multi_pw_aff mpa) const;
  inline isl::map preimage_domain(isl::pw_multi_aff pma) const;
  inline isl::union_map preimage_domain(const isl::union_pw_multi_aff &upma) const;
  inline isl::map preimage_range(isl::multi_aff ma) const;
  inline isl::map preimage_range(isl::pw_multi_aff pma) const;
  inline isl::union_map preimage_range(const isl::union_pw_multi_aff &upma) const;
  inline isl::map product(isl::map map2) const;
  inline isl::union_map product(const isl::union_map &umap2) const;
  inline isl::map product(const isl::basic_map &map2) const;
  inline isl::map project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::map project_out_all_params() const;
  inline isl::set range() const;
  inline isl::map range_factor_domain() const;
  inline isl::map range_factor_range() const;
  inline isl::fixed_box range_lattice_tile() const;
  inline isl::fixed_box get_range_lattice_tile() const;
  inline isl::map range_map() const;
  inline isl::map range_product(isl::map map2) const;
  inline isl::union_map range_product(const isl::union_map &umap2) const;
  inline isl::map range_product(const isl::basic_map &map2) const;
  inline isl::map range_reverse() const;
  inline isl::fixed_box range_simple_fixed_box_hull() const;
  inline isl::fixed_box get_range_simple_fixed_box_hull() const;
  inline class size range_tuple_dim() const;
  inline isl::id range_tuple_id() const;
  inline isl::id get_range_tuple_id() const;
  inline isl::map reverse() const;
  inline isl::basic_map sample() const;
  inline isl::map set_domain_tuple(isl::id id) const;
  inline isl::map set_domain_tuple(const std::string &id) const;
  inline isl::map set_range_tuple(isl::id id) const;
  inline isl::map set_range_tuple(const std::string &id) const;
  inline isl::map set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::map set_tuple_id(isl::dim type, const std::string &id) const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::map subtract(isl::map map2) const;
  inline isl::union_map subtract(const isl::union_map &umap2) const;
  inline isl::map subtract(const isl::basic_map &map2) const;
  inline isl::union_map subtract_domain(const isl::union_set &dom) const;
  inline isl::union_map subtract_range(const isl::union_set &dom) const;
  inline isl::map sum(isl::map map2) const;
  inline isl::map_list to_list() const;
  inline isl::union_map to_union_map() const;
  inline isl::id tuple_id(isl::dim type) const;
  inline isl::id get_tuple_id(isl::dim type) const;
  inline isl::map uncurry() const;
  inline isl::map unite(isl::map map2) const;
  inline isl::union_map unite(const isl::union_map &umap2) const;
  inline isl::map unite(const isl::basic_map &map2) const;
  static inline isl::map universe(isl::space space);
  inline isl::basic_map unshifted_simple_hull() const;
  inline isl::map upper_bound(isl::multi_pw_aff upper) const;
  inline isl::map upper_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::set wrap() const;
  inline isl::map zip() const;
};

// declarations for isl::map_list
inline map_list manage(__isl_take isl_map_list *ptr);
inline map_list manage_copy(__isl_keep isl_map_list *ptr);

class map_list {
  friend inline map_list manage(__isl_take isl_map_list *ptr);
  friend inline map_list manage_copy(__isl_keep isl_map_list *ptr);

protected:
  isl_map_list *ptr = nullptr;

  inline explicit map_list(__isl_take isl_map_list *ptr);

public:
  inline /* implicit */ map_list();
  inline /* implicit */ map_list(const map_list &obj);
  inline explicit map_list(isl::ctx ctx, int n);
  inline explicit map_list(isl::map el);
  inline explicit map_list(isl::ctx ctx, const std::string &str);
  inline map_list &operator=(map_list obj);
  inline ~map_list();
  inline __isl_give isl_map_list *copy() const &;
  inline __isl_give isl_map_list *copy() && = delete;
  inline __isl_keep isl_map_list *get() const;
  inline __isl_give isl_map_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::map_list add(isl::map el) const;
  inline isl::map at(int index) const;
  inline isl::map get_at(int index) const;
  inline isl::map_list clear() const;
  inline isl::map_list concat(isl::map_list list2) const;
  inline isl::map_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(isl::map)> &fn) const;
  inline isl::map_list insert(unsigned int pos, isl::map el) const;
  inline class size size() const;
};

// declarations for isl::multi_aff
inline multi_aff manage(__isl_take isl_multi_aff *ptr);
inline multi_aff manage_copy(__isl_keep isl_multi_aff *ptr);

class multi_aff {
  friend inline multi_aff manage(__isl_take isl_multi_aff *ptr);
  friend inline multi_aff manage_copy(__isl_keep isl_multi_aff *ptr);

protected:
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

  inline isl::multi_aff add(isl::multi_aff multi2) const;
  inline isl::multi_pw_aff add(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff add(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_multi_aff add(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_multi_aff add(const isl::union_pw_multi_aff &upma2) const;
  inline isl::multi_aff add(const isl::aff &multi2) const;
  inline isl::multi_aff add_constant(isl::multi_val mv) const;
  inline isl::multi_aff add_constant(isl::val v) const;
  inline isl::multi_aff add_constant(long v) const;
  inline isl::union_pw_multi_aff add_pw_multi_aff(const isl::pw_multi_aff &pma) const;
  inline isl::union_pw_multi_aff apply(const isl::union_pw_multi_aff &upma2) const;
  inline isl::map as_map() const;
  inline isl::multi_aff as_multi_aff() const;
  inline isl::multi_union_pw_aff as_multi_union_pw_aff() const;
  inline isl::pw_multi_aff as_pw_multi_aff() const;
  inline isl::set as_set() const;
  inline isl::union_map as_union_map() const;
  inline isl::aff at(int pos) const;
  inline isl::aff get_at(int pos) const;
  inline isl::basic_set bind(isl::multi_id tuple) const;
  inline isl::multi_aff bind_domain(isl::multi_id tuple) const;
  inline isl::multi_aff bind_domain_wrapped_domain(isl::multi_id tuple) const;
  inline isl::pw_multi_aff coalesce() const;
  inline isl::multi_val constant_multi_val() const;
  inline isl::multi_val get_constant_multi_val() const;
  inline class size dim(isl::dim type) const;
  inline isl::set domain() const;
  static inline isl::multi_aff domain_map(isl::space space);
  inline isl::pw_multi_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::pw_multi_aff extract_pw_multi_aff(const isl::space &space) const;
  inline isl::multi_aff flat_range_product(isl::multi_aff multi2) const;
  inline isl::multi_pw_aff flat_range_product(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff flat_range_product(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_multi_aff flat_range_product(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_multi_aff flat_range_product(const isl::union_pw_multi_aff &upma2) const;
  inline isl::multi_aff flat_range_product(const isl::aff &multi2) const;
  inline isl::multi_aff floor() const;
  inline stat foreach_piece(const std::function<stat(isl::set, isl::multi_aff)> &fn) const;
  inline isl::multi_aff gist(isl::set context) const;
  inline isl::union_pw_multi_aff gist(const isl::union_set &context) const;
  inline isl::multi_aff gist(const isl::basic_set &context) const;
  inline isl::multi_aff gist(const isl::point &context) const;
  inline boolean has_range_tuple_id() const;
  static inline isl::multi_aff identity(isl::space space);
  inline isl::multi_aff identity() const;
  static inline isl::multi_aff identity_on_domain(isl::space space);
  inline isl::multi_aff insert_domain(isl::space domain) const;
  inline isl::pw_multi_aff intersect_domain(const isl::set &set) const;
  inline isl::union_pw_multi_aff intersect_domain(const isl::space &space) const;
  inline isl::union_pw_multi_aff intersect_domain(const isl::union_set &uset) const;
  inline isl::union_pw_multi_aff intersect_domain_wrapped_domain(const isl::union_set &uset) const;
  inline isl::union_pw_multi_aff intersect_domain_wrapped_range(const isl::union_set &uset) const;
  inline isl::pw_multi_aff intersect_params(const isl::set &set) const;
  inline boolean involves_locals() const;
  inline boolean involves_nan() const;
  inline boolean involves_param(const isl::id &id) const;
  inline boolean involves_param(const std::string &id) const;
  inline boolean involves_param(const isl::id_list &list) const;
  inline boolean isa_multi_aff() const;
  inline boolean isa_pw_multi_aff() const;
  inline isl::aff_list list() const;
  inline isl::aff_list get_list() const;
  inline isl::multi_pw_aff max(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_val max_multi_val() const;
  inline isl::multi_pw_aff min(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_val min_multi_val() const;
  static inline isl::multi_aff multi_val_on_domain(isl::space space, isl::multi_val mv);
  inline class size n_piece() const;
  inline isl::multi_aff neg() const;
  inline boolean plain_is_empty() const;
  inline boolean plain_is_equal(const isl::multi_aff &multi2) const;
  inline boolean plain_is_equal(const isl::multi_pw_aff &multi2) const;
  inline boolean plain_is_equal(const isl::multi_union_pw_aff &multi2) const;
  inline boolean plain_is_equal(const isl::aff &multi2) const;
  inline isl::pw_multi_aff preimage_domain_wrapped_domain(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_multi_aff preimage_domain_wrapped_domain(const isl::union_pw_multi_aff &upma2) const;
  inline isl::multi_aff product(isl::multi_aff multi2) const;
  inline isl::multi_pw_aff product(const isl::multi_pw_aff &multi2) const;
  inline isl::pw_multi_aff product(const isl::pw_multi_aff &pma2) const;
  inline isl::multi_aff product(const isl::aff &multi2) const;
  inline isl::multi_aff pullback(isl::multi_aff ma2) const;
  inline isl::multi_pw_aff pullback(const isl::multi_pw_aff &mpa2) const;
  inline isl::pw_multi_aff pullback(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_multi_aff pullback(const isl::union_pw_multi_aff &upma2) const;
  inline isl::multi_aff pullback(const isl::aff &ma2) const;
  inline isl::pw_multi_aff_list pw_multi_aff_list() const;
  inline isl::pw_multi_aff range_factor_domain() const;
  inline isl::pw_multi_aff range_factor_range() const;
  static inline isl::multi_aff range_map(isl::space space);
  inline isl::multi_aff range_product(isl::multi_aff multi2) const;
  inline isl::multi_pw_aff range_product(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff range_product(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_multi_aff range_product(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_multi_aff range_product(const isl::union_pw_multi_aff &upma2) const;
  inline isl::multi_aff range_product(const isl::aff &multi2) const;
  inline isl::id range_tuple_id() const;
  inline isl::id get_range_tuple_id() const;
  inline isl::multi_aff reset_range_tuple_id() const;
  inline isl::multi_aff reset_tuple_id(isl::dim type) const;
  inline isl::multi_aff scale(isl::multi_val mv) const;
  inline isl::multi_aff scale(isl::val v) const;
  inline isl::multi_aff scale(long v) const;
  inline isl::multi_aff scale_down(isl::multi_val mv) const;
  inline isl::multi_aff scale_down(isl::val v) const;
  inline isl::multi_aff scale_down(long v) const;
  inline isl::multi_aff set_aff(int pos, isl::aff el) const;
  inline isl::multi_aff set_at(int pos, isl::aff el) const;
  inline isl::multi_pw_aff set_at(int pos, const isl::pw_aff &el) const;
  inline isl::multi_union_pw_aff set_at(int pos, const isl::union_pw_aff &el) const;
  inline isl::multi_pw_aff set_pw_aff(int pos, const isl::pw_aff &el) const;
  inline isl::pw_multi_aff set_pw_aff(unsigned int pos, const isl::pw_aff &pa) const;
  inline isl::multi_aff set_range_tuple(isl::id id) const;
  inline isl::multi_aff set_range_tuple(const std::string &id) const;
  inline isl::multi_union_pw_aff set_union_pw_aff(int pos, const isl::union_pw_aff &el) const;
  inline class size size() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::multi_aff sub(isl::multi_aff multi2) const;
  inline isl::multi_pw_aff sub(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff sub(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_multi_aff sub(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_multi_aff sub(const isl::union_pw_multi_aff &upma2) const;
  inline isl::multi_aff sub(const isl::aff &multi2) const;
  inline isl::pw_multi_aff subtract_domain(const isl::set &set) const;
  inline isl::union_pw_multi_aff subtract_domain(const isl::space &space) const;
  inline isl::union_pw_multi_aff subtract_domain(const isl::union_set &uset) const;
  inline isl::pw_multi_aff_list to_list() const;
  inline isl::multi_pw_aff to_multi_pw_aff() const;
  inline isl::multi_union_pw_aff to_multi_union_pw_aff() const;
  inline isl::pw_multi_aff to_pw_multi_aff() const;
  inline isl::union_pw_multi_aff to_union_pw_multi_aff() const;
  inline isl::id tuple_id(isl::dim type) const;
  inline isl::multi_aff unbind_params_insert_domain(isl::multi_id domain) const;
  inline isl::multi_pw_aff union_add(const isl::multi_pw_aff &mpa2) const;
  inline isl::multi_union_pw_aff union_add(const isl::multi_union_pw_aff &mupa2) const;
  inline isl::pw_multi_aff union_add(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_multi_aff union_add(const isl::union_pw_multi_aff &upma2) const;
  static inline isl::multi_aff zero(isl::space space);
};

// declarations for isl::multi_id
inline multi_id manage(__isl_take isl_multi_id *ptr);
inline multi_id manage_copy(__isl_keep isl_multi_id *ptr);

class multi_id {
  friend inline multi_id manage(__isl_take isl_multi_id *ptr);
  friend inline multi_id manage_copy(__isl_keep isl_multi_id *ptr);

protected:
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

  inline isl::id at(int pos) const;
  inline isl::id get_at(int pos) const;
  inline isl::multi_id flat_range_product(isl::multi_id multi2) const;
  inline isl::id_list list() const;
  inline isl::id_list get_list() const;
  inline boolean plain_is_equal(const isl::multi_id &multi2) const;
  inline isl::multi_id range_product(isl::multi_id multi2) const;
  inline isl::multi_id set_at(int pos, isl::id el) const;
  inline isl::multi_id set_at(int pos, const std::string &el) const;
  inline isl::multi_id set_id(int pos, isl::id el) const;
  inline isl::multi_id set_id(int pos, const std::string &el) const;
  inline class size size() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
};

// declarations for isl::multi_pw_aff
inline multi_pw_aff manage(__isl_take isl_multi_pw_aff *ptr);
inline multi_pw_aff manage_copy(__isl_keep isl_multi_pw_aff *ptr);

class multi_pw_aff {
  friend inline multi_pw_aff manage(__isl_take isl_multi_pw_aff *ptr);
  friend inline multi_pw_aff manage_copy(__isl_keep isl_multi_pw_aff *ptr);

protected:
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

  inline isl::multi_pw_aff add(isl::multi_pw_aff multi2) const;
  inline isl::multi_union_pw_aff add(const isl::multi_union_pw_aff &multi2) const;
  inline isl::multi_pw_aff add(const isl::aff &multi2) const;
  inline isl::multi_pw_aff add(const isl::multi_aff &multi2) const;
  inline isl::multi_pw_aff add(const isl::pw_aff &multi2) const;
  inline isl::multi_pw_aff add(const isl::pw_multi_aff &multi2) const;
  inline isl::multi_pw_aff add_constant(isl::multi_val mv) const;
  inline isl::multi_pw_aff add_constant(isl::val v) const;
  inline isl::multi_pw_aff add_constant(long v) const;
  inline isl::map as_map() const;
  inline isl::multi_aff as_multi_aff() const;
  inline isl::set as_set() const;
  inline isl::pw_aff at(int pos) const;
  inline isl::pw_aff get_at(int pos) const;
  inline isl::set bind(isl::multi_id tuple) const;
  inline isl::multi_pw_aff bind_domain(isl::multi_id tuple) const;
  inline isl::multi_pw_aff bind_domain_wrapped_domain(isl::multi_id tuple) const;
  inline isl::multi_pw_aff coalesce() const;
  inline class size dim(isl::dim type) const;
  inline isl::set domain() const;
  inline isl::multi_pw_aff flat_range_product(isl::multi_pw_aff multi2) const;
  inline isl::multi_union_pw_aff flat_range_product(const isl::multi_union_pw_aff &multi2) const;
  inline isl::multi_pw_aff flat_range_product(const isl::aff &multi2) const;
  inline isl::multi_pw_aff flat_range_product(const isl::multi_aff &multi2) const;
  inline isl::multi_pw_aff flat_range_product(const isl::pw_aff &multi2) const;
  inline isl::multi_pw_aff flat_range_product(const isl::pw_multi_aff &multi2) const;
  inline isl::multi_pw_aff gist(isl::set set) const;
  inline isl::multi_union_pw_aff gist(const isl::union_set &context) const;
  inline isl::multi_pw_aff gist(const isl::basic_set &set) const;
  inline isl::multi_pw_aff gist(const isl::point &set) const;
  inline boolean has_range_tuple_id() const;
  static inline isl::multi_pw_aff identity(isl::space space);
  inline isl::multi_pw_aff identity() const;
  static inline isl::multi_pw_aff identity_on_domain(isl::space space);
  inline isl::multi_pw_aff insert_domain(isl::space domain) const;
  inline isl::multi_pw_aff intersect_domain(isl::set domain) const;
  inline isl::multi_union_pw_aff intersect_domain(const isl::union_set &uset) const;
  inline isl::multi_pw_aff intersect_domain(const isl::basic_set &domain) const;
  inline isl::multi_pw_aff intersect_domain(const isl::point &domain) const;
  inline isl::multi_pw_aff intersect_params(isl::set set) const;
  inline boolean involves_nan() const;
  inline boolean involves_param(const isl::id &id) const;
  inline boolean involves_param(const std::string &id) const;
  inline boolean involves_param(const isl::id_list &list) const;
  inline boolean isa_multi_aff() const;
  inline isl::pw_aff_list list() const;
  inline isl::pw_aff_list get_list() const;
  inline isl::multi_pw_aff max(isl::multi_pw_aff multi2) const;
  inline isl::multi_val max_multi_val() const;
  inline isl::multi_pw_aff min(isl::multi_pw_aff multi2) const;
  inline isl::multi_val min_multi_val() const;
  inline isl::multi_pw_aff neg() const;
  inline boolean plain_is_equal(const isl::multi_pw_aff &multi2) const;
  inline boolean plain_is_equal(const isl::multi_union_pw_aff &multi2) const;
  inline boolean plain_is_equal(const isl::aff &multi2) const;
  inline boolean plain_is_equal(const isl::multi_aff &multi2) const;
  inline boolean plain_is_equal(const isl::pw_aff &multi2) const;
  inline boolean plain_is_equal(const isl::pw_multi_aff &multi2) const;
  inline isl::multi_pw_aff product(isl::multi_pw_aff multi2) const;
  inline isl::multi_pw_aff pullback(isl::multi_aff ma) const;
  inline isl::multi_pw_aff pullback(isl::multi_pw_aff mpa2) const;
  inline isl::multi_pw_aff pullback(isl::pw_multi_aff pma) const;
  inline isl::multi_union_pw_aff pullback(const isl::union_pw_multi_aff &upma) const;
  inline isl::multi_pw_aff range_product(isl::multi_pw_aff multi2) const;
  inline isl::multi_union_pw_aff range_product(const isl::multi_union_pw_aff &multi2) const;
  inline isl::multi_pw_aff range_product(const isl::aff &multi2) const;
  inline isl::multi_pw_aff range_product(const isl::multi_aff &multi2) const;
  inline isl::multi_pw_aff range_product(const isl::pw_aff &multi2) const;
  inline isl::multi_pw_aff range_product(const isl::pw_multi_aff &multi2) const;
  inline isl::id range_tuple_id() const;
  inline isl::id get_range_tuple_id() const;
  inline isl::multi_pw_aff reset_range_tuple_id() const;
  inline isl::multi_pw_aff reset_tuple_id(isl::dim type) const;
  inline isl::multi_pw_aff scale(isl::multi_val mv) const;
  inline isl::multi_pw_aff scale(isl::val v) const;
  inline isl::multi_pw_aff scale(long v) const;
  inline isl::multi_pw_aff scale_down(isl::multi_val mv) const;
  inline isl::multi_pw_aff scale_down(isl::val v) const;
  inline isl::multi_pw_aff scale_down(long v) const;
  inline isl::multi_pw_aff set_at(int pos, isl::pw_aff el) const;
  inline isl::multi_union_pw_aff set_at(int pos, const isl::union_pw_aff &el) const;
  inline isl::multi_pw_aff set_pw_aff(int pos, isl::pw_aff el) const;
  inline isl::multi_pw_aff set_range_tuple(isl::id id) const;
  inline isl::multi_pw_aff set_range_tuple(const std::string &id) const;
  inline isl::multi_union_pw_aff set_union_pw_aff(int pos, const isl::union_pw_aff &el) const;
  inline class size size() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::multi_pw_aff sub(isl::multi_pw_aff multi2) const;
  inline isl::multi_union_pw_aff sub(const isl::multi_union_pw_aff &multi2) const;
  inline isl::multi_pw_aff sub(const isl::aff &multi2) const;
  inline isl::multi_pw_aff sub(const isl::multi_aff &multi2) const;
  inline isl::multi_pw_aff sub(const isl::pw_aff &multi2) const;
  inline isl::multi_pw_aff sub(const isl::pw_multi_aff &multi2) const;
  inline isl::multi_pw_aff unbind_params_insert_domain(isl::multi_id domain) const;
  inline isl::multi_pw_aff union_add(isl::multi_pw_aff mpa2) const;
  inline isl::multi_union_pw_aff union_add(const isl::multi_union_pw_aff &mupa2) const;
  inline isl::multi_pw_aff union_add(const isl::aff &mpa2) const;
  inline isl::multi_pw_aff union_add(const isl::multi_aff &mpa2) const;
  inline isl::multi_pw_aff union_add(const isl::pw_aff &mpa2) const;
  inline isl::multi_pw_aff union_add(const isl::pw_multi_aff &mpa2) const;
  static inline isl::multi_pw_aff zero(isl::space space);
};

// declarations for isl::multi_union_pw_aff
inline multi_union_pw_aff manage(__isl_take isl_multi_union_pw_aff *ptr);
inline multi_union_pw_aff manage_copy(__isl_keep isl_multi_union_pw_aff *ptr);

class multi_union_pw_aff {
  friend inline multi_union_pw_aff manage(__isl_take isl_multi_union_pw_aff *ptr);
  friend inline multi_union_pw_aff manage_copy(__isl_keep isl_multi_union_pw_aff *ptr);

protected:
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

  inline isl::multi_union_pw_aff add(isl::multi_union_pw_aff multi2) const;
  inline isl::union_pw_aff at(int pos) const;
  inline isl::union_pw_aff get_at(int pos) const;
  inline isl::union_set bind(isl::multi_id tuple) const;
  inline isl::multi_union_pw_aff coalesce() const;
  inline class size dim(isl::dim type) const;
  inline isl::union_set domain() const;
  inline isl::multi_union_pw_aff flat_range_product(isl::multi_union_pw_aff multi2) const;
  static inline isl::multi_union_pw_aff from_union_map(isl::union_map umap);
  inline isl::multi_union_pw_aff gist(isl::union_set context) const;
  inline boolean has_range_tuple_id() const;
  inline isl::multi_union_pw_aff intersect_domain(isl::union_set uset) const;
  inline isl::multi_union_pw_aff intersect_params(isl::set params) const;
  inline boolean involves_nan() const;
  inline isl::union_pw_aff_list list() const;
  inline isl::union_pw_aff_list get_list() const;
  inline isl::multi_union_pw_aff neg() const;
  inline boolean plain_is_equal(const isl::multi_union_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff pullback(isl::union_pw_multi_aff upma) const;
  inline isl::multi_union_pw_aff range_product(isl::multi_union_pw_aff multi2) const;
  inline isl::id range_tuple_id() const;
  inline isl::id get_range_tuple_id() const;
  inline isl::multi_union_pw_aff reset_range_tuple_id() const;
  inline isl::multi_union_pw_aff reset_tuple_id(isl::dim type) const;
  inline isl::multi_union_pw_aff scale(isl::multi_val mv) const;
  inline isl::multi_union_pw_aff scale(isl::val v) const;
  inline isl::multi_union_pw_aff scale(long v) const;
  inline isl::multi_union_pw_aff scale_down(isl::multi_val mv) const;
  inline isl::multi_union_pw_aff scale_down(isl::val v) const;
  inline isl::multi_union_pw_aff scale_down(long v) const;
  inline isl::multi_union_pw_aff set_at(int pos, isl::union_pw_aff el) const;
  inline isl::multi_union_pw_aff set_range_tuple(isl::id id) const;
  inline isl::multi_union_pw_aff set_range_tuple(const std::string &id) const;
  inline isl::multi_union_pw_aff set_union_pw_aff(int pos, isl::union_pw_aff el) const;
  inline class size size() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::multi_union_pw_aff sub(isl::multi_union_pw_aff multi2) const;
  inline isl::multi_union_pw_aff union_add(isl::multi_union_pw_aff mupa2) const;
  static inline isl::multi_union_pw_aff zero(isl::space space);
};

// declarations for isl::multi_val
inline multi_val manage(__isl_take isl_multi_val *ptr);
inline multi_val manage_copy(__isl_keep isl_multi_val *ptr);

class multi_val {
  friend inline multi_val manage(__isl_take isl_multi_val *ptr);
  friend inline multi_val manage_copy(__isl_keep isl_multi_val *ptr);

protected:
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

  inline isl::multi_val add(isl::multi_val multi2) const;
  inline isl::multi_val add(isl::val v) const;
  inline isl::multi_val add(long v) const;
  inline isl::val at(int pos) const;
  inline isl::val get_at(int pos) const;
  inline class size dim(isl::dim type) const;
  inline isl::multi_val flat_range_product(isl::multi_val multi2) const;
  inline boolean has_range_tuple_id() const;
  inline boolean involves_nan() const;
  inline isl::val_list list() const;
  inline isl::val_list get_list() const;
  inline isl::multi_val max(isl::multi_val multi2) const;
  inline isl::multi_val min(isl::multi_val multi2) const;
  inline isl::multi_val neg() const;
  inline boolean plain_is_equal(const isl::multi_val &multi2) const;
  inline isl::multi_val product(isl::multi_val multi2) const;
  inline isl::multi_val range_product(isl::multi_val multi2) const;
  inline isl::id range_tuple_id() const;
  inline isl::id get_range_tuple_id() const;
  inline isl::multi_val reset_range_tuple_id() const;
  inline isl::multi_val reset_tuple_id(isl::dim type) const;
  inline isl::multi_val scale(isl::multi_val mv) const;
  inline isl::multi_val scale(isl::val v) const;
  inline isl::multi_val scale(long v) const;
  inline isl::multi_val scale_down(isl::multi_val mv) const;
  inline isl::multi_val scale_down(isl::val v) const;
  inline isl::multi_val scale_down(long v) const;
  inline isl::multi_val set_at(int pos, isl::val el) const;
  inline isl::multi_val set_at(int pos, long el) const;
  inline isl::multi_val set_range_tuple(isl::id id) const;
  inline isl::multi_val set_range_tuple(const std::string &id) const;
  inline isl::multi_val set_val(int pos, isl::val el) const;
  inline isl::multi_val set_val(int pos, long el) const;
  inline class size size() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::multi_val sub(isl::multi_val multi2) const;
  static inline isl::multi_val zero(isl::space space);
};

// declarations for isl::point
inline point manage(__isl_take isl_point *ptr);
inline point manage_copy(__isl_keep isl_point *ptr);

class point {
  friend inline point manage(__isl_take isl_point *ptr);
  friend inline point manage_copy(__isl_keep isl_point *ptr);

protected:
  isl_point *ptr = nullptr;

  inline explicit point(__isl_take isl_point *ptr);

public:
  inline /* implicit */ point();
  inline /* implicit */ point(const point &obj);
  inline explicit point(isl::space space);
  inline point &operator=(point obj);
  inline ~point();
  inline __isl_give isl_point *copy() const &;
  inline __isl_give isl_point *copy() && = delete;
  inline __isl_keep isl_point *get() const;
  inline __isl_give isl_point *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::set add_constraint(const isl::constraint &constraint) const;
  inline isl::set add_dims(isl::dim type, unsigned int n) const;
  inline isl::basic_set affine_hull() const;
  inline isl::set align_params(const isl::space &model) const;
  inline isl::basic_set apply(const isl::basic_map &bmap) const;
  inline isl::set apply(const isl::map &map) const;
  inline isl::union_set apply(const isl::union_map &umap) const;
  inline isl::pw_multi_aff as_pw_multi_aff() const;
  inline isl::set as_set() const;
  inline isl::basic_set_list basic_set_list() const;
  inline isl::set bind(const isl::multi_id &tuple) const;
  inline isl::set coalesce() const;
  inline isl::set complement() const;
  inline isl::union_set compute_divs() const;
  inline boolean contains(const isl::space &space) const;
  inline isl::basic_set convex_hull() const;
  inline isl::val coordinate_val(isl::dim type, int pos) const;
  inline isl::val get_coordinate_val(isl::dim type, int pos) const;
  inline isl::basic_set detect_equalities() const;
  inline class size dim(isl::dim type) const;
  inline boolean dim_has_any_lower_bound(isl::dim type, unsigned int pos) const;
  inline isl::id dim_id(isl::dim type, unsigned int pos) const;
  inline isl::pw_aff dim_max(int pos) const;
  inline isl::val dim_max_val(int pos) const;
  inline isl::pw_aff dim_min(int pos) const;
  inline isl::val dim_min_val(int pos) const;
  inline std::string dim_name(isl::dim type, unsigned int pos) const;
  inline isl::aff div(int pos) const;
  inline isl::set drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set eliminate(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean every_set(const std::function<boolean(isl::set)> &test) const;
  inline isl::set extract_set(const isl::space &space) const;
  inline int find_dim_by_id(isl::dim type, const isl::id &id) const;
  inline int find_dim_by_id(isl::dim type, const std::string &id) const;
  inline isl::basic_set fix_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::basic_set fix_val(isl::dim type, unsigned int pos, const isl::val &v) const;
  inline isl::basic_set fix_val(isl::dim type, unsigned int pos, long v) const;
  inline isl::basic_set flatten() const;
  inline stat foreach_basic_set(const std::function<stat(isl::basic_set)> &fn) const;
  inline stat foreach_point(const std::function<stat(isl::point)> &fn) const;
  inline stat foreach_set(const std::function<stat(isl::set)> &fn) const;
  inline isl::basic_set gist(const isl::basic_set &context) const;
  inline isl::set gist(const isl::set &context) const;
  inline isl::union_set gist(const isl::union_set &context) const;
  inline isl::set gist_params(const isl::set &context) const;
  inline boolean has_equal_space(const isl::set &set2) const;
  inline isl::map identity() const;
  inline isl::union_pw_multi_aff identity_union_pw_multi_aff() const;
  inline isl::pw_aff indicator_function() const;
  inline isl::set insert_dims(isl::dim type, unsigned int pos, unsigned int n) const;
  inline isl::map insert_domain(const isl::space &domain) const;
  inline isl::basic_set intersect(const isl::basic_set &bset2) const;
  inline isl::set intersect(const isl::set &set2) const;
  inline isl::union_set intersect(const isl::union_set &uset2) const;
  inline isl::basic_set intersect_params(const isl::basic_set &bset2) const;
  inline isl::set intersect_params(const isl::set &params) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_locals() const;
  inline boolean is_bounded() const;
  inline boolean is_disjoint(const isl::set &set2) const;
  inline boolean is_disjoint(const isl::union_set &uset2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const isl::basic_set &bset2) const;
  inline boolean is_equal(const isl::set &set2) const;
  inline boolean is_equal(const isl::union_set &uset2) const;
  inline boolean is_params() const;
  inline boolean is_singleton() const;
  inline boolean is_strict_subset(const isl::set &set2) const;
  inline boolean is_strict_subset(const isl::union_set &uset2) const;
  inline boolean is_subset(const isl::basic_set &bset2) const;
  inline boolean is_subset(const isl::set &set2) const;
  inline boolean is_subset(const isl::union_set &uset2) const;
  inline boolean is_wrapping() const;
  inline boolean isa_set() const;
  inline isl::set lexmax() const;
  inline isl::pw_multi_aff lexmax_pw_multi_aff() const;
  inline isl::set lexmin() const;
  inline isl::pw_multi_aff lexmin_pw_multi_aff() const;
  inline isl::set lower_bound(const isl::multi_pw_aff &lower) const;
  inline isl::set lower_bound(const isl::multi_val &lower) const;
  inline isl::set lower_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::set lower_bound_val(isl::dim type, unsigned int pos, const isl::val &value) const;
  inline isl::set lower_bound_val(isl::dim type, unsigned int pos, long value) const;
  inline isl::multi_pw_aff max_multi_pw_aff() const;
  inline isl::val max_val(const isl::aff &obj) const;
  inline isl::multi_pw_aff min_multi_pw_aff() const;
  inline isl::val min_val(const isl::aff &obj) const;
  inline isl::multi_val multi_val() const;
  inline isl::multi_val get_multi_val() const;
  inline class size n_basic_set() const;
  inline isl::basic_set params() const;
  inline isl::val plain_get_val_if_fixed(isl::dim type, unsigned int pos) const;
  inline isl::multi_val plain_multi_val_if_fixed() const;
  inline isl::basic_set polyhedral_hull() const;
  inline isl::set preimage(const isl::multi_aff &ma) const;
  inline isl::set preimage(const isl::multi_pw_aff &mpa) const;
  inline isl::set preimage(const isl::pw_multi_aff &pma) const;
  inline isl::union_set preimage(const isl::union_pw_multi_aff &upma) const;
  inline isl::set product(const isl::set &set2) const;
  inline isl::basic_set project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set project_out_all_params() const;
  inline isl::set project_out_param(const isl::id &id) const;
  inline isl::set project_out_param(const std::string &id) const;
  inline isl::set project_out_param(const isl::id_list &list) const;
  inline isl::pw_multi_aff pw_multi_aff_on_domain(const isl::multi_val &mv) const;
  inline isl::set remove_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set remove_divs() const;
  inline isl::set remove_redundancies() const;
  inline isl::set reset_tuple_id() const;
  inline isl::basic_set sample() const;
  inline isl::point sample_point() const;
  inline isl::set set_dim_id(isl::dim type, unsigned int pos, const isl::id &id) const;
  inline isl::set set_dim_id(isl::dim type, unsigned int pos, const std::string &id) const;
  inline isl::set_list set_list() const;
  inline isl::set set_tuple_id(const isl::id &id) const;
  inline isl::set set_tuple_id(const std::string &id) const;
  inline isl::fixed_box simple_fixed_box_hull() const;
  inline isl::basic_set simple_hull() const;
  inline isl::space space() const;
  inline isl::val stride(int pos) const;
  inline isl::set subtract(const isl::set &set2) const;
  inline isl::union_set subtract(const isl::union_set &uset2) const;
  inline isl::basic_set_list to_list() const;
  inline isl::set to_set() const;
  inline isl::union_set to_union_set() const;
  inline isl::map translation() const;
  inline class size tuple_dim() const;
  inline isl::id tuple_id() const;
  inline std::string tuple_name() const;
  inline isl::set unbind_params(const isl::multi_id &tuple) const;
  inline isl::map unbind_params_insert_domain(const isl::multi_id &domain) const;
  inline isl::set unite(const isl::basic_set &bset2) const;
  inline isl::set unite(const isl::set &set2) const;
  inline isl::union_set unite(const isl::union_set &uset2) const;
  inline isl::basic_set unshifted_simple_hull() const;
  inline isl::map unwrap() const;
  inline isl::set upper_bound(const isl::multi_pw_aff &upper) const;
  inline isl::set upper_bound(const isl::multi_val &upper) const;
  inline isl::set upper_bound_val(isl::dim type, unsigned int pos, const isl::val &value) const;
  inline isl::set upper_bound_val(isl::dim type, unsigned int pos, long value) const;
};

// declarations for isl::pw_aff
inline pw_aff manage(__isl_take isl_pw_aff *ptr);
inline pw_aff manage_copy(__isl_keep isl_pw_aff *ptr);

class pw_aff {
  friend inline pw_aff manage(__isl_take isl_pw_aff *ptr);
  friend inline pw_aff manage_copy(__isl_keep isl_pw_aff *ptr);

protected:
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

  inline isl::multi_pw_aff add(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff add(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_aff add(isl::pw_aff pwaff2) const;
  inline isl::pw_multi_aff add(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_aff add(const isl::union_pw_aff &upa2) const;
  inline isl::union_pw_multi_aff add(const isl::union_pw_multi_aff &upma2) const;
  inline isl::pw_aff add(const isl::aff &pwaff2) const;
  inline isl::pw_aff add_constant(isl::val v) const;
  inline isl::pw_aff add_constant(long v) const;
  inline isl::pw_multi_aff add_constant(const isl::multi_val &mv) const;
  inline isl::pw_aff add_dims(isl::dim type, unsigned int n) const;
  inline isl::union_pw_multi_aff add_pw_multi_aff(const isl::pw_multi_aff &pma) const;
  inline isl::union_pw_multi_aff apply(const isl::union_pw_multi_aff &upma2) const;
  inline isl::aff as_aff() const;
  inline isl::map as_map() const;
  inline isl::multi_aff as_multi_aff() const;
  inline isl::multi_union_pw_aff as_multi_union_pw_aff() const;
  inline isl::pw_multi_aff as_pw_multi_aff() const;
  inline isl::set as_set() const;
  inline isl::union_map as_union_map() const;
  inline isl::pw_aff at(int pos) const;
  inline isl::set bind(const isl::multi_id &tuple) const;
  inline isl::set bind(isl::id id) const;
  inline isl::set bind(const std::string &id) const;
  inline isl::pw_aff bind_domain(isl::multi_id tuple) const;
  inline isl::pw_aff bind_domain_wrapped_domain(isl::multi_id tuple) const;
  inline isl::pw_aff ceil() const;
  inline isl::pw_aff coalesce() const;
  inline isl::pw_aff cond(isl::pw_aff pwaff_true, isl::pw_aff pwaff_false) const;
  inline class size dim(isl::dim type) const;
  inline isl::id dim_id(isl::dim type, unsigned int pos) const;
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline isl::pw_aff div(isl::pw_aff pa2) const;
  inline isl::set domain() const;
  inline isl::space domain_space() const;
  inline isl::space get_domain_space() const;
  inline isl::pw_multi_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set eq_set(isl::pw_aff pwaff2) const;
  inline isl::val eval(isl::point pnt) const;
  inline isl::pw_multi_aff extract_pw_multi_aff(const isl::space &space) const;
  inline isl::multi_pw_aff flat_range_product(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff flat_range_product(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_multi_aff flat_range_product(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_multi_aff flat_range_product(const isl::union_pw_multi_aff &upma2) const;
  inline isl::pw_aff floor() const;
  inline stat foreach_piece(const std::function<stat(isl::set, isl::aff)> &fn) const;
  inline stat foreach_piece(const std::function<stat(isl::set, isl::multi_aff)> &fn) const;
  inline stat foreach_pw_aff(const std::function<stat(isl::pw_aff)> &fn) const;
  inline isl::set ge_set(isl::pw_aff pwaff2) const;
  inline isl::pw_aff gist(isl::set context) const;
  inline isl::union_pw_aff gist(const isl::union_set &context) const;
  inline isl::pw_aff gist(const isl::basic_set &context) const;
  inline isl::pw_aff gist(const isl::point &context) const;
  inline isl::set gt_set(isl::pw_aff pwaff2) const;
  inline boolean has_range_tuple_id() const;
  inline isl::multi_pw_aff identity() const;
  inline isl::pw_aff insert_domain(isl::space domain) const;
  inline isl::pw_aff intersect_domain(isl::set set) const;
  inline isl::union_pw_aff intersect_domain(const isl::space &space) const;
  inline isl::union_pw_aff intersect_domain(const isl::union_set &uset) const;
  inline isl::pw_aff intersect_domain(const isl::basic_set &set) const;
  inline isl::pw_aff intersect_domain(const isl::point &set) const;
  inline isl::union_pw_aff intersect_domain_wrapped_domain(const isl::union_set &uset) const;
  inline isl::union_pw_aff intersect_domain_wrapped_range(const isl::union_set &uset) const;
  inline isl::pw_aff intersect_params(isl::set set) const;
  inline boolean involves_locals() const;
  inline boolean involves_nan() const;
  inline boolean involves_param(const isl::id &id) const;
  inline boolean involves_param(const std::string &id) const;
  inline boolean involves_param(const isl::id_list &list) const;
  inline boolean is_cst() const;
  inline boolean is_equal(const isl::pw_aff &pa2) const;
  inline boolean isa_aff() const;
  inline boolean isa_multi_aff() const;
  inline boolean isa_pw_multi_aff() const;
  inline isl::set le_set(isl::pw_aff pwaff2) const;
  inline isl::pw_aff_list list() const;
  inline isl::set lt_set(isl::pw_aff pwaff2) const;
  inline isl::multi_pw_aff max(const isl::multi_pw_aff &multi2) const;
  inline isl::pw_aff max(isl::pw_aff pwaff2) const;
  inline isl::pw_aff max(const isl::aff &pwaff2) const;
  inline isl::multi_val max_multi_val() const;
  inline isl::multi_pw_aff min(const isl::multi_pw_aff &multi2) const;
  inline isl::pw_aff min(isl::pw_aff pwaff2) const;
  inline isl::pw_aff min(const isl::aff &pwaff2) const;
  inline isl::multi_val min_multi_val() const;
  inline isl::pw_aff mod(isl::val mod) const;
  inline isl::pw_aff mod(long mod) const;
  inline isl::pw_aff mul(isl::pw_aff pwaff2) const;
  inline class size n_piece() const;
  inline isl::set ne_set(isl::pw_aff pwaff2) const;
  inline isl::pw_aff neg() const;
  static inline isl::pw_aff param_on_domain(isl::set domain, isl::id id);
  inline boolean plain_is_empty() const;
  inline boolean plain_is_equal(const isl::multi_pw_aff &multi2) const;
  inline boolean plain_is_equal(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_multi_aff preimage_domain_wrapped_domain(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_multi_aff preimage_domain_wrapped_domain(const isl::union_pw_multi_aff &upma2) const;
  inline isl::multi_pw_aff product(const isl::multi_pw_aff &multi2) const;
  inline isl::pw_multi_aff product(const isl::pw_multi_aff &pma2) const;
  inline isl::pw_aff pullback(isl::multi_aff ma) const;
  inline isl::pw_aff pullback(isl::multi_pw_aff mpa) const;
  inline isl::pw_aff pullback(isl::pw_multi_aff pma) const;
  inline isl::union_pw_aff pullback(const isl::union_pw_multi_aff &upma) const;
  inline isl::pw_multi_aff_list pw_multi_aff_list() const;
  inline isl::pw_multi_aff range_factor_domain() const;
  inline isl::pw_multi_aff range_factor_range() const;
  inline isl::multi_pw_aff range_product(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff range_product(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_multi_aff range_product(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_multi_aff range_product(const isl::union_pw_multi_aff &upma2) const;
  inline isl::id range_tuple_id() const;
  inline isl::multi_pw_aff reset_range_tuple_id() const;
  inline isl::multi_pw_aff reset_tuple_id(isl::dim type) const;
  inline isl::multi_pw_aff scale(const isl::multi_val &mv) const;
  inline isl::pw_aff scale(isl::val v) const;
  inline isl::pw_aff scale(long v) const;
  inline isl::multi_pw_aff scale_down(const isl::multi_val &mv) const;
  inline isl::pw_aff scale_down(isl::val f) const;
  inline isl::pw_aff scale_down(long f) const;
  inline isl::multi_pw_aff set_at(int pos, const isl::pw_aff &el) const;
  inline isl::multi_union_pw_aff set_at(int pos, const isl::union_pw_aff &el) const;
  inline isl::multi_pw_aff set_pw_aff(int pos, const isl::pw_aff &el) const;
  inline isl::pw_multi_aff set_pw_aff(unsigned int pos, const isl::pw_aff &pa) const;
  inline isl::pw_multi_aff set_range_tuple(const isl::id &id) const;
  inline isl::pw_multi_aff set_range_tuple(const std::string &id) const;
  inline isl::pw_aff set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::pw_aff set_tuple_id(isl::dim type, const std::string &id) const;
  inline isl::multi_union_pw_aff set_union_pw_aff(int pos, const isl::union_pw_aff &el) const;
  inline class size size() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::multi_pw_aff sub(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff sub(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_aff sub(isl::pw_aff pwaff2) const;
  inline isl::pw_multi_aff sub(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_aff sub(const isl::union_pw_aff &upa2) const;
  inline isl::union_pw_multi_aff sub(const isl::union_pw_multi_aff &upma2) const;
  inline isl::pw_aff sub(const isl::aff &pwaff2) const;
  inline isl::pw_aff subtract_domain(isl::set set) const;
  inline isl::union_pw_aff subtract_domain(const isl::space &space) const;
  inline isl::union_pw_aff subtract_domain(const isl::union_set &uset) const;
  inline isl::pw_aff subtract_domain(const isl::basic_set &set) const;
  inline isl::pw_aff subtract_domain(const isl::point &set) const;
  inline isl::pw_aff tdiv_q(isl::pw_aff pa2) const;
  inline isl::pw_aff tdiv_r(isl::pw_aff pa2) const;
  inline isl::pw_aff_list to_list() const;
  inline isl::multi_pw_aff to_multi_pw_aff() const;
  inline isl::union_pw_aff to_union_pw_aff() const;
  inline isl::union_pw_multi_aff to_union_pw_multi_aff() const;
  inline isl::id tuple_id(isl::dim type) const;
  inline isl::id get_tuple_id(isl::dim type) const;
  inline isl::multi_pw_aff unbind_params_insert_domain(const isl::multi_id &domain) const;
  inline isl::multi_pw_aff union_add(const isl::multi_pw_aff &mpa2) const;
  inline isl::multi_union_pw_aff union_add(const isl::multi_union_pw_aff &mupa2) const;
  inline isl::pw_aff union_add(isl::pw_aff pwaff2) const;
  inline isl::pw_multi_aff union_add(const isl::pw_multi_aff &pma2) const;
  inline isl::union_pw_aff union_add(const isl::union_pw_aff &upa2) const;
  inline isl::union_pw_multi_aff union_add(const isl::union_pw_multi_aff &upma2) const;
  inline isl::pw_aff union_add(const isl::aff &pwaff2) const;
  static inline isl::pw_aff var_on_domain(isl::local_space ls, isl::dim type, unsigned int pos);
};

// declarations for isl::pw_aff_list
inline pw_aff_list manage(__isl_take isl_pw_aff_list *ptr);
inline pw_aff_list manage_copy(__isl_keep isl_pw_aff_list *ptr);

class pw_aff_list {
  friend inline pw_aff_list manage(__isl_take isl_pw_aff_list *ptr);
  friend inline pw_aff_list manage_copy(__isl_keep isl_pw_aff_list *ptr);

protected:
  isl_pw_aff_list *ptr = nullptr;

  inline explicit pw_aff_list(__isl_take isl_pw_aff_list *ptr);

public:
  inline /* implicit */ pw_aff_list();
  inline /* implicit */ pw_aff_list(const pw_aff_list &obj);
  inline explicit pw_aff_list(isl::ctx ctx, int n);
  inline explicit pw_aff_list(isl::pw_aff el);
  inline explicit pw_aff_list(isl::ctx ctx, const std::string &str);
  inline pw_aff_list &operator=(pw_aff_list obj);
  inline ~pw_aff_list();
  inline __isl_give isl_pw_aff_list *copy() const &;
  inline __isl_give isl_pw_aff_list *copy() && = delete;
  inline __isl_keep isl_pw_aff_list *get() const;
  inline __isl_give isl_pw_aff_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::pw_aff_list add(isl::pw_aff el) const;
  inline isl::pw_aff at(int index) const;
  inline isl::pw_aff get_at(int index) const;
  inline isl::pw_aff_list clear() const;
  inline isl::pw_aff_list concat(isl::pw_aff_list list2) const;
  inline isl::pw_aff_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(isl::pw_aff)> &fn) const;
  inline isl::pw_aff_list insert(unsigned int pos, isl::pw_aff el) const;
  inline class size size() const;
};

// declarations for isl::pw_multi_aff
inline pw_multi_aff manage(__isl_take isl_pw_multi_aff *ptr);
inline pw_multi_aff manage_copy(__isl_keep isl_pw_multi_aff *ptr);

class pw_multi_aff {
  friend inline pw_multi_aff manage(__isl_take isl_pw_multi_aff *ptr);
  friend inline pw_multi_aff manage_copy(__isl_keep isl_pw_multi_aff *ptr);

protected:
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

  inline isl::multi_pw_aff add(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff add(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_multi_aff add(isl::pw_multi_aff pma2) const;
  inline isl::union_pw_multi_aff add(const isl::union_pw_multi_aff &upma2) const;
  inline isl::pw_multi_aff add(const isl::multi_aff &pma2) const;
  inline isl::pw_multi_aff add(const isl::pw_aff &pma2) const;
  inline isl::pw_multi_aff add_constant(isl::multi_val mv) const;
  inline isl::pw_multi_aff add_constant(isl::val v) const;
  inline isl::pw_multi_aff add_constant(long v) const;
  inline isl::union_pw_multi_aff add_pw_multi_aff(const isl::pw_multi_aff &pma) const;
  inline isl::union_pw_multi_aff apply(const isl::union_pw_multi_aff &upma2) const;
  inline isl::map as_map() const;
  inline isl::multi_aff as_multi_aff() const;
  inline isl::multi_union_pw_aff as_multi_union_pw_aff() const;
  inline isl::pw_multi_aff as_pw_multi_aff() const;
  inline isl::set as_set() const;
  inline isl::union_map as_union_map() const;
  inline isl::pw_aff at(int pos) const;
  inline isl::pw_aff get_at(int pos) const;
  inline isl::set bind(const isl::multi_id &tuple) const;
  inline isl::pw_multi_aff bind_domain(isl::multi_id tuple) const;
  inline isl::pw_multi_aff bind_domain_wrapped_domain(isl::multi_id tuple) const;
  inline isl::pw_multi_aff coalesce() const;
  inline class size dim(isl::dim type) const;
  inline isl::set domain() const;
  static inline isl::pw_multi_aff domain_map(isl::space space);
  inline isl::pw_multi_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::pw_multi_aff extract_pw_multi_aff(const isl::space &space) const;
  inline isl::multi_pw_aff flat_range_product(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff flat_range_product(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_multi_aff flat_range_product(isl::pw_multi_aff pma2) const;
  inline isl::union_pw_multi_aff flat_range_product(const isl::union_pw_multi_aff &upma2) const;
  inline isl::pw_multi_aff flat_range_product(const isl::multi_aff &pma2) const;
  inline isl::pw_multi_aff flat_range_product(const isl::pw_aff &pma2) const;
  inline stat foreach_piece(const std::function<stat(isl::set, isl::multi_aff)> &fn) const;
  static inline isl::pw_multi_aff from_map(isl::map map);
  inline isl::pw_multi_aff gist(isl::set set) const;
  inline isl::union_pw_multi_aff gist(const isl::union_set &context) const;
  inline isl::pw_multi_aff gist(const isl::basic_set &set) const;
  inline isl::pw_multi_aff gist(const isl::point &set) const;
  inline boolean has_range_tuple_id() const;
  inline isl::multi_pw_aff identity() const;
  static inline isl::pw_multi_aff identity_on_domain(isl::space space);
  inline isl::pw_multi_aff insert_domain(isl::space domain) const;
  inline isl::pw_multi_aff intersect_domain(isl::set set) const;
  inline isl::union_pw_multi_aff intersect_domain(const isl::space &space) const;
  inline isl::union_pw_multi_aff intersect_domain(const isl::union_set &uset) const;
  inline isl::pw_multi_aff intersect_domain(const isl::basic_set &set) const;
  inline isl::pw_multi_aff intersect_domain(const isl::point &set) const;
  inline isl::union_pw_multi_aff intersect_domain_wrapped_domain(const isl::union_set &uset) const;
  inline isl::union_pw_multi_aff intersect_domain_wrapped_range(const isl::union_set &uset) const;
  inline isl::pw_multi_aff intersect_params(isl::set set) const;
  inline boolean involves_locals() const;
  inline boolean involves_nan() const;
  inline boolean involves_param(const isl::id &id) const;
  inline boolean involves_param(const std::string &id) const;
  inline boolean involves_param(const isl::id_list &list) const;
  inline boolean isa_multi_aff() const;
  inline boolean isa_pw_multi_aff() const;
  inline isl::pw_aff_list list() const;
  inline isl::multi_pw_aff max(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_val max_multi_val() const;
  inline isl::multi_pw_aff min(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_val min_multi_val() const;
  static inline isl::pw_multi_aff multi_val_on_domain(isl::set domain, isl::multi_val mv);
  inline class size n_piece() const;
  inline isl::multi_pw_aff neg() const;
  inline boolean plain_is_empty() const;
  inline boolean plain_is_equal(const isl::multi_pw_aff &multi2) const;
  inline boolean plain_is_equal(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_multi_aff preimage_domain_wrapped_domain(isl::pw_multi_aff pma2) const;
  inline isl::union_pw_multi_aff preimage_domain_wrapped_domain(const isl::union_pw_multi_aff &upma2) const;
  inline isl::pw_multi_aff preimage_domain_wrapped_domain(const isl::multi_aff &pma2) const;
  inline isl::pw_multi_aff preimage_domain_wrapped_domain(const isl::pw_aff &pma2) const;
  inline isl::multi_pw_aff product(const isl::multi_pw_aff &multi2) const;
  inline isl::pw_multi_aff product(isl::pw_multi_aff pma2) const;
  inline isl::pw_multi_aff product(const isl::multi_aff &pma2) const;
  inline isl::pw_multi_aff product(const isl::pw_aff &pma2) const;
  static inline isl::pw_multi_aff project_out_map(isl::space space, isl::dim type, unsigned int first, unsigned int n);
  inline isl::multi_pw_aff pullback(const isl::multi_pw_aff &mpa2) const;
  inline isl::pw_multi_aff pullback(isl::multi_aff ma) const;
  inline isl::pw_multi_aff pullback(isl::pw_multi_aff pma2) const;
  inline isl::union_pw_multi_aff pullback(const isl::union_pw_multi_aff &upma2) const;
  inline isl::pw_multi_aff_list pw_multi_aff_list() const;
  inline isl::pw_multi_aff range_factor_domain() const;
  inline isl::pw_multi_aff range_factor_range() const;
  static inline isl::pw_multi_aff range_map(isl::space space);
  inline isl::multi_pw_aff range_product(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff range_product(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_multi_aff range_product(isl::pw_multi_aff pma2) const;
  inline isl::union_pw_multi_aff range_product(const isl::union_pw_multi_aff &upma2) const;
  inline isl::pw_multi_aff range_product(const isl::multi_aff &pma2) const;
  inline isl::pw_multi_aff range_product(const isl::pw_aff &pma2) const;
  inline isl::id range_tuple_id() const;
  inline isl::id get_range_tuple_id() const;
  inline isl::multi_pw_aff reset_range_tuple_id() const;
  inline isl::multi_pw_aff reset_tuple_id(isl::dim type) const;
  inline isl::multi_pw_aff scale(const isl::multi_val &mv) const;
  inline isl::pw_multi_aff scale(isl::val v) const;
  inline isl::pw_multi_aff scale(long v) const;
  inline isl::multi_pw_aff scale_down(const isl::multi_val &mv) const;
  inline isl::pw_multi_aff scale_down(isl::val v) const;
  inline isl::pw_multi_aff scale_down(long v) const;
  inline isl::multi_pw_aff set_at(int pos, const isl::pw_aff &el) const;
  inline isl::multi_union_pw_aff set_at(int pos, const isl::union_pw_aff &el) const;
  inline isl::multi_pw_aff set_pw_aff(int pos, const isl::pw_aff &el) const;
  inline isl::pw_multi_aff set_pw_aff(unsigned int pos, isl::pw_aff pa) const;
  inline isl::pw_multi_aff set_range_tuple(isl::id id) const;
  inline isl::pw_multi_aff set_range_tuple(const std::string &id) const;
  inline isl::multi_union_pw_aff set_union_pw_aff(int pos, const isl::union_pw_aff &el) const;
  inline class size size() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::multi_pw_aff sub(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff sub(const isl::multi_union_pw_aff &multi2) const;
  inline isl::pw_multi_aff sub(isl::pw_multi_aff pma2) const;
  inline isl::union_pw_multi_aff sub(const isl::union_pw_multi_aff &upma2) const;
  inline isl::pw_multi_aff sub(const isl::multi_aff &pma2) const;
  inline isl::pw_multi_aff sub(const isl::pw_aff &pma2) const;
  inline isl::pw_multi_aff subtract_domain(isl::set set) const;
  inline isl::union_pw_multi_aff subtract_domain(const isl::space &space) const;
  inline isl::union_pw_multi_aff subtract_domain(const isl::union_set &uset) const;
  inline isl::pw_multi_aff subtract_domain(const isl::basic_set &set) const;
  inline isl::pw_multi_aff subtract_domain(const isl::point &set) const;
  inline isl::pw_multi_aff_list to_list() const;
  inline isl::multi_pw_aff to_multi_pw_aff() const;
  inline isl::union_pw_multi_aff to_union_pw_multi_aff() const;
  inline isl::id tuple_id(isl::dim type) const;
  inline isl::id get_tuple_id(isl::dim type) const;
  inline isl::multi_pw_aff unbind_params_insert_domain(const isl::multi_id &domain) const;
  inline isl::multi_pw_aff union_add(const isl::multi_pw_aff &mpa2) const;
  inline isl::multi_union_pw_aff union_add(const isl::multi_union_pw_aff &mupa2) const;
  inline isl::pw_multi_aff union_add(isl::pw_multi_aff pma2) const;
  inline isl::union_pw_multi_aff union_add(const isl::union_pw_multi_aff &upma2) const;
  inline isl::pw_multi_aff union_add(const isl::multi_aff &pma2) const;
  inline isl::pw_multi_aff union_add(const isl::pw_aff &pma2) const;
  static inline isl::pw_multi_aff zero(isl::space space);
};

// declarations for isl::pw_multi_aff_list
inline pw_multi_aff_list manage(__isl_take isl_pw_multi_aff_list *ptr);
inline pw_multi_aff_list manage_copy(__isl_keep isl_pw_multi_aff_list *ptr);

class pw_multi_aff_list {
  friend inline pw_multi_aff_list manage(__isl_take isl_pw_multi_aff_list *ptr);
  friend inline pw_multi_aff_list manage_copy(__isl_keep isl_pw_multi_aff_list *ptr);

protected:
  isl_pw_multi_aff_list *ptr = nullptr;

  inline explicit pw_multi_aff_list(__isl_take isl_pw_multi_aff_list *ptr);

public:
  inline /* implicit */ pw_multi_aff_list();
  inline /* implicit */ pw_multi_aff_list(const pw_multi_aff_list &obj);
  inline explicit pw_multi_aff_list(isl::ctx ctx, int n);
  inline explicit pw_multi_aff_list(isl::pw_multi_aff el);
  inline explicit pw_multi_aff_list(isl::ctx ctx, const std::string &str);
  inline pw_multi_aff_list &operator=(pw_multi_aff_list obj);
  inline ~pw_multi_aff_list();
  inline __isl_give isl_pw_multi_aff_list *copy() const &;
  inline __isl_give isl_pw_multi_aff_list *copy() && = delete;
  inline __isl_keep isl_pw_multi_aff_list *get() const;
  inline __isl_give isl_pw_multi_aff_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::pw_multi_aff_list add(isl::pw_multi_aff el) const;
  inline isl::pw_multi_aff at(int index) const;
  inline isl::pw_multi_aff get_at(int index) const;
  inline isl::pw_multi_aff_list clear() const;
  inline isl::pw_multi_aff_list concat(isl::pw_multi_aff_list list2) const;
  inline isl::pw_multi_aff_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(isl::pw_multi_aff)> &fn) const;
  inline isl::pw_multi_aff_list insert(unsigned int pos, isl::pw_multi_aff el) const;
  inline class size size() const;
};

// declarations for isl::schedule
inline schedule manage(__isl_take isl_schedule *ptr);
inline schedule manage_copy(__isl_keep isl_schedule *ptr);

class schedule {
  friend inline schedule manage(__isl_take isl_schedule *ptr);
  friend inline schedule manage_copy(__isl_keep isl_schedule *ptr);

protected:
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

  inline isl::schedule align_params(isl::space space) const;
  inline isl::union_set domain() const;
  inline isl::union_set get_domain() const;
  static inline isl::schedule from_domain(isl::union_set domain);
  inline isl::schedule gist_domain_params(isl::set context) const;
  inline isl::schedule insert_partial_schedule(isl::multi_union_pw_aff partial) const;
  inline isl::schedule intersect_domain(isl::union_set domain) const;
  inline isl::union_map map() const;
  inline isl::union_map get_map() const;
  inline isl::schedule pullback(isl::union_pw_multi_aff upma) const;
  inline isl::schedule_node root() const;
  inline isl::schedule_node get_root() const;
  inline isl::schedule sequence(isl::schedule schedule2) const;
};

// declarations for isl::schedule_constraints
inline schedule_constraints manage(__isl_take isl_schedule_constraints *ptr);
inline schedule_constraints manage_copy(__isl_keep isl_schedule_constraints *ptr);

class schedule_constraints {
  friend inline schedule_constraints manage(__isl_take isl_schedule_constraints *ptr);
  friend inline schedule_constraints manage_copy(__isl_keep isl_schedule_constraints *ptr);

protected:
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

  inline isl::union_map coincidence() const;
  inline isl::union_map get_coincidence() const;
  inline isl::schedule compute_schedule() const;
  inline isl::union_map conditional_validity() const;
  inline isl::union_map get_conditional_validity() const;
  inline isl::union_map conditional_validity_condition() const;
  inline isl::union_map get_conditional_validity_condition() const;
  inline isl::set context() const;
  inline isl::set get_context() const;
  inline isl::union_set domain() const;
  inline isl::union_set get_domain() const;
  static inline isl::schedule_constraints on_domain(isl::union_set domain);
  inline isl::union_map proximity() const;
  inline isl::union_map get_proximity() const;
  inline isl::schedule_constraints set_coincidence(isl::union_map coincidence) const;
  inline isl::schedule_constraints set_conditional_validity(isl::union_map condition, isl::union_map validity) const;
  inline isl::schedule_constraints set_context(isl::set context) const;
  inline isl::schedule_constraints set_proximity(isl::union_map proximity) const;
  inline isl::schedule_constraints set_validity(isl::union_map validity) const;
  inline isl::union_map validity() const;
  inline isl::union_map get_validity() const;
};

// declarations for isl::schedule_node
inline schedule_node manage(__isl_take isl_schedule_node *ptr);
inline schedule_node manage_copy(__isl_keep isl_schedule_node *ptr);

class schedule_node {
  friend inline schedule_node manage(__isl_take isl_schedule_node *ptr);
  friend inline schedule_node manage_copy(__isl_keep isl_schedule_node *ptr);

protected:
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
private:
  template <typename T,
          typename = typename std::enable_if<std::is_same<
                  const decltype(isl_schedule_node_get_type(NULL)),
                  const T>::value>::type>
  inline boolean isa_type(T subtype) const;
public:
  template <class T> inline boolean isa() const;
  template <class T> inline T as() const;
  inline isl::ctx ctx() const;

  inline isl::schedule_node ancestor(int generation) const;
  inline class size ancestor_child_position(const isl::schedule_node &ancestor) const;
  inline class size get_ancestor_child_position(const isl::schedule_node &ancestor) const;
  inline isl::schedule_node child(int pos) const;
  inline class size child_position() const;
  inline class size get_child_position() const;
  inline isl::union_set domain() const;
  inline isl::union_set get_domain() const;
  inline boolean every_descendant(const std::function<boolean(isl::schedule_node)> &test) const;
  inline isl::schedule_node first_child() const;
  inline stat foreach_ancestor_top_down(const std::function<stat(isl::schedule_node)> &fn) const;
  inline stat foreach_descendant_top_down(const std::function<boolean(isl::schedule_node)> &fn) const;
  static inline isl::schedule_node from_domain(isl::union_set domain);
  static inline isl::schedule_node from_extension(isl::union_map extension);
  inline isl::schedule_node graft_after(isl::schedule_node graft) const;
  inline isl::schedule_node graft_before(isl::schedule_node graft) const;
  inline boolean has_children() const;
  inline boolean has_next_sibling() const;
  inline boolean has_parent() const;
  inline boolean has_previous_sibling() const;
  inline isl::schedule_node insert_context(isl::set context) const;
  inline isl::schedule_node insert_filter(isl::union_set filter) const;
  inline isl::schedule_node insert_guard(isl::set context) const;
  inline isl::schedule_node insert_mark(isl::id mark) const;
  inline isl::schedule_node insert_mark(const std::string &mark) const;
  inline isl::schedule_node insert_partial_schedule(isl::multi_union_pw_aff schedule) const;
  inline isl::schedule_node insert_sequence(isl::union_set_list filters) const;
  inline isl::schedule_node insert_set(isl::union_set_list filters) const;
  inline boolean is_equal(const isl::schedule_node &node2) const;
  inline boolean is_subtree_anchored() const;
  inline isl::schedule_node map_descendant_bottom_up(const std::function<isl::schedule_node(isl::schedule_node)> &fn) const;
  inline class size n_children() const;
  inline isl::schedule_node next_sibling() const;
  inline isl::schedule_node order_after(isl::union_set filter) const;
  inline isl::schedule_node order_before(isl::union_set filter) const;
  inline isl::schedule_node parent() const;
  inline isl::multi_union_pw_aff prefix_schedule_multi_union_pw_aff() const;
  inline isl::multi_union_pw_aff get_prefix_schedule_multi_union_pw_aff() const;
  inline isl::union_map prefix_schedule_relation() const;
  inline isl::union_map get_prefix_schedule_relation() const;
  inline isl::union_map prefix_schedule_union_map() const;
  inline isl::union_map get_prefix_schedule_union_map() const;
  inline isl::union_pw_multi_aff prefix_schedule_union_pw_multi_aff() const;
  inline isl::union_pw_multi_aff get_prefix_schedule_union_pw_multi_aff() const;
  inline isl::schedule_node previous_sibling() const;
  inline isl::schedule_node root() const;
  inline isl::schedule schedule() const;
  inline isl::schedule get_schedule() const;
  inline class size schedule_depth() const;
  inline class size get_schedule_depth() const;
  inline isl::schedule_node shared_ancestor(const isl::schedule_node &node2) const;
  inline isl::schedule_node get_shared_ancestor(const isl::schedule_node &node2) const;
  inline class size tree_depth() const;
  inline class size get_tree_depth() const;
  inline isl::union_set universe_domain() const;
  inline isl::union_set get_universe_domain() const;
};

// declarations for isl::schedule_node_band

class schedule_node_band : public schedule_node {
  template <class T>
  friend boolean schedule_node::isa() const;
  friend schedule_node_band schedule_node::as<schedule_node_band>() const;
  static const auto type = isl_schedule_node_band;

protected:
  inline explicit schedule_node_band(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_band();
  inline /* implicit */ schedule_node_band(const schedule_node_band &obj);
  inline schedule_node_band &operator=(schedule_node_band obj);
  inline isl::ctx ctx() const;

  inline isl::union_set ast_build_options() const;
  inline isl::union_set get_ast_build_options() const;
  inline isl::set ast_isolate_option() const;
  inline isl::set get_ast_isolate_option() const;
  inline boolean member_get_coincident(int pos) const;
  inline schedule_node_band member_set_coincident(int pos, int coincident) const;
  inline schedule_node_band mod(isl::multi_val mv) const;
  inline class size n_member() const;
  inline isl::multi_union_pw_aff partial_schedule() const;
  inline isl::multi_union_pw_aff get_partial_schedule() const;
  inline boolean permutable() const;
  inline boolean get_permutable() const;
  inline schedule_node_band scale(isl::multi_val mv) const;
  inline schedule_node_band scale_down(isl::multi_val mv) const;
  inline schedule_node_band set_ast_build_options(isl::union_set options) const;
  inline schedule_node_band set_permutable(int permutable) const;
  inline schedule_node_band shift(isl::multi_union_pw_aff shift) const;
  inline schedule_node_band split(int pos) const;
  inline schedule_node_band tile(isl::multi_val sizes) const;
  inline schedule_node_band member_set_ast_loop_default(int pos) const;
  inline schedule_node_band member_set_ast_loop_atomic(int pos) const;
  inline schedule_node_band member_set_ast_loop_unroll(int pos) const;
  inline schedule_node_band member_set_ast_loop_separate(int pos) const;
};

// declarations for isl::schedule_node_context

class schedule_node_context : public schedule_node {
  template <class T>
  friend boolean schedule_node::isa() const;
  friend schedule_node_context schedule_node::as<schedule_node_context>() const;
  static const auto type = isl_schedule_node_context;

protected:
  inline explicit schedule_node_context(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_context();
  inline /* implicit */ schedule_node_context(const schedule_node_context &obj);
  inline schedule_node_context &operator=(schedule_node_context obj);
  inline isl::ctx ctx() const;

  inline isl::set context() const;
  inline isl::set get_context() const;
};

// declarations for isl::schedule_node_domain

class schedule_node_domain : public schedule_node {
  template <class T>
  friend boolean schedule_node::isa() const;
  friend schedule_node_domain schedule_node::as<schedule_node_domain>() const;
  static const auto type = isl_schedule_node_domain;

protected:
  inline explicit schedule_node_domain(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_domain();
  inline /* implicit */ schedule_node_domain(const schedule_node_domain &obj);
  inline schedule_node_domain &operator=(schedule_node_domain obj);
  inline isl::ctx ctx() const;

  inline isl::union_set domain() const;
  inline isl::union_set get_domain() const;
};

// declarations for isl::schedule_node_expansion

class schedule_node_expansion : public schedule_node {
  template <class T>
  friend boolean schedule_node::isa() const;
  friend schedule_node_expansion schedule_node::as<schedule_node_expansion>() const;
  static const auto type = isl_schedule_node_expansion;

protected:
  inline explicit schedule_node_expansion(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_expansion();
  inline /* implicit */ schedule_node_expansion(const schedule_node_expansion &obj);
  inline schedule_node_expansion &operator=(schedule_node_expansion obj);
  inline isl::ctx ctx() const;

  inline isl::union_pw_multi_aff contraction() const;
  inline isl::union_pw_multi_aff get_contraction() const;
  inline isl::union_map expansion() const;
  inline isl::union_map get_expansion() const;
};

// declarations for isl::schedule_node_extension

class schedule_node_extension : public schedule_node {
  template <class T>
  friend boolean schedule_node::isa() const;
  friend schedule_node_extension schedule_node::as<schedule_node_extension>() const;
  static const auto type = isl_schedule_node_extension;

protected:
  inline explicit schedule_node_extension(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_extension();
  inline /* implicit */ schedule_node_extension(const schedule_node_extension &obj);
  inline schedule_node_extension &operator=(schedule_node_extension obj);
  inline isl::ctx ctx() const;

  inline isl::union_map extension() const;
  inline isl::union_map get_extension() const;
};

// declarations for isl::schedule_node_filter

class schedule_node_filter : public schedule_node {
  template <class T>
  friend boolean schedule_node::isa() const;
  friend schedule_node_filter schedule_node::as<schedule_node_filter>() const;
  static const auto type = isl_schedule_node_filter;

protected:
  inline explicit schedule_node_filter(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_filter();
  inline /* implicit */ schedule_node_filter(const schedule_node_filter &obj);
  inline schedule_node_filter &operator=(schedule_node_filter obj);
  inline isl::ctx ctx() const;

  inline isl::union_set filter() const;
  inline isl::union_set get_filter() const;
};

// declarations for isl::schedule_node_guard

class schedule_node_guard : public schedule_node {
  template <class T>
  friend boolean schedule_node::isa() const;
  friend schedule_node_guard schedule_node::as<schedule_node_guard>() const;
  static const auto type = isl_schedule_node_guard;

protected:
  inline explicit schedule_node_guard(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_guard();
  inline /* implicit */ schedule_node_guard(const schedule_node_guard &obj);
  inline schedule_node_guard &operator=(schedule_node_guard obj);
  inline isl::ctx ctx() const;

  inline isl::set guard() const;
  inline isl::set get_guard() const;
};

// declarations for isl::schedule_node_leaf

class schedule_node_leaf : public schedule_node {
  template <class T>
  friend boolean schedule_node::isa() const;
  friend schedule_node_leaf schedule_node::as<schedule_node_leaf>() const;
  static const auto type = isl_schedule_node_leaf;

protected:
  inline explicit schedule_node_leaf(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_leaf();
  inline /* implicit */ schedule_node_leaf(const schedule_node_leaf &obj);
  inline schedule_node_leaf &operator=(schedule_node_leaf obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::schedule_node_mark

class schedule_node_mark : public schedule_node {
  template <class T>
  friend boolean schedule_node::isa() const;
  friend schedule_node_mark schedule_node::as<schedule_node_mark>() const;
  static const auto type = isl_schedule_node_mark;

protected:
  inline explicit schedule_node_mark(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_mark();
  inline /* implicit */ schedule_node_mark(const schedule_node_mark &obj);
  inline schedule_node_mark &operator=(schedule_node_mark obj);
  inline isl::ctx ctx() const;

  inline isl::id id() const;
  inline isl::id get_id() const;
};

// declarations for isl::schedule_node_sequence

class schedule_node_sequence : public schedule_node {
  template <class T>
  friend boolean schedule_node::isa() const;
  friend schedule_node_sequence schedule_node::as<schedule_node_sequence>() const;
  static const auto type = isl_schedule_node_sequence;

protected:
  inline explicit schedule_node_sequence(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_sequence();
  inline /* implicit */ schedule_node_sequence(const schedule_node_sequence &obj);
  inline schedule_node_sequence &operator=(schedule_node_sequence obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::schedule_node_set

class schedule_node_set : public schedule_node {
  template <class T>
  friend boolean schedule_node::isa() const;
  friend schedule_node_set schedule_node::as<schedule_node_set>() const;
  static const auto type = isl_schedule_node_set;

protected:
  inline explicit schedule_node_set(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_set();
  inline /* implicit */ schedule_node_set(const schedule_node_set &obj);
  inline schedule_node_set &operator=(schedule_node_set obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::set
inline set manage(__isl_take isl_set *ptr);
inline set manage_copy(__isl_keep isl_set *ptr);

class set {
  friend inline set manage(__isl_take isl_set *ptr);
  friend inline set manage_copy(__isl_keep isl_set *ptr);

protected:
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

  inline isl::set add_constraint(isl::constraint constraint) const;
  inline isl::set add_dims(isl::dim type, unsigned int n) const;
  inline isl::basic_set affine_hull() const;
  inline isl::set align_params(isl::space model) const;
  inline isl::set apply(isl::map map) const;
  inline isl::union_set apply(const isl::union_map &umap) const;
  inline isl::set apply(const isl::basic_map &map) const;
  inline isl::pw_multi_aff as_pw_multi_aff() const;
  inline isl::set as_set() const;
  inline isl::basic_set_list basic_set_list() const;
  inline isl::basic_set_list get_basic_set_list() const;
  inline isl::set bind(isl::multi_id tuple) const;
  inline isl::set coalesce() const;
  inline isl::set complement() const;
  inline isl::union_set compute_divs() const;
  inline boolean contains(const isl::space &space) const;
  inline isl::basic_set convex_hull() const;
  inline isl::set detect_equalities() const;
  inline class size dim(isl::dim type) const;
  inline boolean dim_has_any_lower_bound(isl::dim type, unsigned int pos) const;
  inline isl::id dim_id(isl::dim type, unsigned int pos) const;
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline isl::pw_aff dim_max(int pos) const;
  inline isl::val dim_max_val(int pos) const;
  inline isl::pw_aff dim_min(int pos) const;
  inline isl::val dim_min_val(int pos) const;
  inline std::string dim_name(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline isl::set drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set eliminate(isl::dim type, unsigned int first, unsigned int n) const;
  static inline isl::set empty(isl::space space);
  inline boolean every_set(const std::function<boolean(isl::set)> &test) const;
  inline isl::set extract_set(const isl::space &space) const;
  inline int find_dim_by_id(isl::dim type, const isl::id &id) const;
  inline int find_dim_by_id(isl::dim type, const std::string &id) const;
  inline isl::set fix_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::set flatten() const;
  inline stat foreach_basic_set(const std::function<stat(isl::basic_set)> &fn) const;
  inline stat foreach_point(const std::function<stat(isl::point)> &fn) const;
  inline stat foreach_set(const std::function<stat(isl::set)> &fn) const;
  inline isl::set gist(isl::set context) const;
  inline isl::union_set gist(const isl::union_set &context) const;
  inline isl::set gist(const isl::basic_set &context) const;
  inline isl::set gist(const isl::point &context) const;
  inline isl::set gist_params(isl::set context) const;
  inline boolean has_equal_space(const isl::set &set2) const;
  inline isl::map identity() const;
  inline isl::union_pw_multi_aff identity_union_pw_multi_aff() const;
  inline isl::pw_aff indicator_function() const;
  inline isl::set insert_dims(isl::dim type, unsigned int pos, unsigned int n) const;
  inline isl::map insert_domain(isl::space domain) const;
  inline isl::set intersect(isl::set set2) const;
  inline isl::union_set intersect(const isl::union_set &uset2) const;
  inline isl::set intersect(const isl::basic_set &set2) const;
  inline isl::set intersect(const isl::point &set2) const;
  inline isl::set intersect_params(isl::set params) const;
  inline boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline boolean involves_locals() const;
  inline boolean is_bounded() const;
  inline boolean is_disjoint(const isl::set &set2) const;
  inline boolean is_disjoint(const isl::union_set &uset2) const;
  inline boolean is_disjoint(const isl::basic_set &set2) const;
  inline boolean is_disjoint(const isl::point &set2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const isl::set &set2) const;
  inline boolean is_equal(const isl::union_set &uset2) const;
  inline boolean is_equal(const isl::basic_set &set2) const;
  inline boolean is_equal(const isl::point &set2) const;
  inline boolean is_params() const;
  inline boolean is_singleton() const;
  inline boolean is_strict_subset(const isl::set &set2) const;
  inline boolean is_strict_subset(const isl::union_set &uset2) const;
  inline boolean is_strict_subset(const isl::basic_set &set2) const;
  inline boolean is_strict_subset(const isl::point &set2) const;
  inline boolean is_subset(const isl::set &set2) const;
  inline boolean is_subset(const isl::union_set &uset2) const;
  inline boolean is_subset(const isl::basic_set &set2) const;
  inline boolean is_subset(const isl::point &set2) const;
  inline boolean is_wrapping() const;
  inline boolean isa_set() const;
  inline isl::set lexmax() const;
  inline isl::pw_multi_aff lexmax_pw_multi_aff() const;
  inline isl::set lexmin() const;
  inline isl::pw_multi_aff lexmin_pw_multi_aff() const;
  inline isl::set lower_bound(isl::multi_pw_aff lower) const;
  inline isl::set lower_bound(isl::multi_val lower) const;
  inline isl::set lower_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::set lower_bound_val(isl::dim type, unsigned int pos, isl::val value) const;
  inline isl::set lower_bound_val(isl::dim type, unsigned int pos, long value) const;
  inline isl::multi_pw_aff max_multi_pw_aff() const;
  inline isl::val max_val(const isl::aff &obj) const;
  inline isl::multi_pw_aff min_multi_pw_aff() const;
  inline isl::val min_val(const isl::aff &obj) const;
  inline class size n_basic_set() const;
  inline isl::set params() const;
  inline isl::val plain_get_val_if_fixed(isl::dim type, unsigned int pos) const;
  inline isl::multi_val plain_multi_val_if_fixed() const;
  inline isl::multi_val get_plain_multi_val_if_fixed() const;
  inline isl::basic_set polyhedral_hull() const;
  inline isl::set preimage(isl::multi_aff ma) const;
  inline isl::set preimage(isl::multi_pw_aff mpa) const;
  inline isl::set preimage(isl::pw_multi_aff pma) const;
  inline isl::union_set preimage(const isl::union_pw_multi_aff &upma) const;
  inline isl::set product(isl::set set2) const;
  inline isl::set project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set project_out_all_params() const;
  inline isl::set project_out_param(isl::id id) const;
  inline isl::set project_out_param(const std::string &id) const;
  inline isl::set project_out_param(isl::id_list list) const;
  inline isl::pw_multi_aff pw_multi_aff_on_domain(isl::multi_val mv) const;
  inline isl::set remove_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set remove_divs() const;
  inline isl::set remove_redundancies() const;
  inline isl::set reset_tuple_id() const;
  inline isl::basic_set sample() const;
  inline isl::point sample_point() const;
  inline isl::set set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::set set_dim_id(isl::dim type, unsigned int pos, const std::string &id) const;
  inline isl::set_list set_list() const;
  inline isl::set set_tuple_id(isl::id id) const;
  inline isl::set set_tuple_id(const std::string &id) const;
  inline isl::fixed_box simple_fixed_box_hull() const;
  inline isl::fixed_box get_simple_fixed_box_hull() const;
  inline isl::basic_set simple_hull() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::val stride(int pos) const;
  inline isl::val get_stride(int pos) const;
  inline isl::set subtract(isl::set set2) const;
  inline isl::union_set subtract(const isl::union_set &uset2) const;
  inline isl::set subtract(const isl::basic_set &set2) const;
  inline isl::set subtract(const isl::point &set2) const;
  inline isl::set_list to_list() const;
  inline isl::union_set to_union_set() const;
  inline isl::map translation() const;
  inline class size tuple_dim() const;
  inline isl::id tuple_id() const;
  inline isl::id get_tuple_id() const;
  inline std::string tuple_name() const;
  inline std::string get_tuple_name() const;
  inline isl::set unbind_params(isl::multi_id tuple) const;
  inline isl::map unbind_params_insert_domain(isl::multi_id domain) const;
  inline isl::set unite(isl::set set2) const;
  inline isl::union_set unite(const isl::union_set &uset2) const;
  inline isl::set unite(const isl::basic_set &set2) const;
  inline isl::set unite(const isl::point &set2) const;
  static inline isl::set universe(isl::space space);
  inline isl::basic_set unshifted_simple_hull() const;
  inline isl::map unwrap() const;
  inline isl::set upper_bound(isl::multi_pw_aff upper) const;
  inline isl::set upper_bound(isl::multi_val upper) const;
  inline isl::set upper_bound_val(isl::dim type, unsigned int pos, isl::val value) const;
  inline isl::set upper_bound_val(isl::dim type, unsigned int pos, long value) const;
};

// declarations for isl::set_list
inline set_list manage(__isl_take isl_set_list *ptr);
inline set_list manage_copy(__isl_keep isl_set_list *ptr);

class set_list {
  friend inline set_list manage(__isl_take isl_set_list *ptr);
  friend inline set_list manage_copy(__isl_keep isl_set_list *ptr);

protected:
  isl_set_list *ptr = nullptr;

  inline explicit set_list(__isl_take isl_set_list *ptr);

public:
  inline /* implicit */ set_list();
  inline /* implicit */ set_list(const set_list &obj);
  inline explicit set_list(isl::ctx ctx, int n);
  inline explicit set_list(isl::set el);
  inline explicit set_list(isl::ctx ctx, const std::string &str);
  inline set_list &operator=(set_list obj);
  inline ~set_list();
  inline __isl_give isl_set_list *copy() const &;
  inline __isl_give isl_set_list *copy() && = delete;
  inline __isl_keep isl_set_list *get() const;
  inline __isl_give isl_set_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::set_list add(isl::set el) const;
  inline isl::set at(int index) const;
  inline isl::set get_at(int index) const;
  inline isl::set_list clear() const;
  inline isl::set_list concat(isl::set_list list2) const;
  inline isl::set_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(isl::set)> &fn) const;
  inline isl::set_list insert(unsigned int pos, isl::set el) const;
  inline class size size() const;
};

// declarations for isl::space
inline space manage(__isl_take isl_space *ptr);
inline space manage_copy(__isl_keep isl_space *ptr);

class space {
  friend inline space manage(__isl_take isl_space *ptr);
  friend inline space manage_copy(__isl_keep isl_space *ptr);

protected:
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

  inline isl::space add_dims(isl::dim type, unsigned int n) const;
  inline isl::space add_named_tuple(isl::id tuple_id, unsigned int dim) const;
  inline isl::space add_named_tuple(const std::string &tuple_id, unsigned int dim) const;
  inline isl::space add_param(isl::id id) const;
  inline isl::space add_param(const std::string &id) const;
  inline isl::space add_unnamed_tuple(unsigned int dim) const;
  inline isl::space align_params(isl::space space2) const;
  inline isl::space curry() const;
  inline class size dim(isl::dim type) const;
  inline isl::id dim_id(isl::dim type, unsigned int pos) const;
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline isl::space domain() const;
  inline isl::multi_aff domain_map_multi_aff() const;
  inline isl::pw_multi_aff domain_map_pw_multi_aff() const;
  inline isl::id domain_tuple_id() const;
  inline isl::id get_domain_tuple_id() const;
  inline isl::space drop_dims(isl::dim type, unsigned int first, unsigned int num) const;
  inline int find_dim_by_id(isl::dim type, const isl::id &id) const;
  inline int find_dim_by_id(isl::dim type, const std::string &id) const;
  inline isl::space flatten_domain() const;
  inline isl::space flatten_range() const;
  inline boolean has_domain_tuple_id() const;
  inline boolean has_equal_tuples(const isl::space &space2) const;
  inline boolean has_range_tuple_id() const;
  inline boolean has_tuple_id(isl::dim type) const;
  inline boolean has_tuple_name(isl::dim type) const;
  inline isl::multi_aff identity_multi_aff_on_domain() const;
  inline isl::multi_pw_aff identity_multi_pw_aff_on_domain() const;
  inline isl::pw_multi_aff identity_pw_multi_aff_on_domain() const;
  inline boolean is_equal(const isl::space &space2) const;
  inline boolean is_params() const;
  inline boolean is_set() const;
  inline boolean is_wrapping() const;
  inline isl::space map_from_domain_and_range(isl::space range) const;
  inline isl::space map_from_set() const;
  inline isl::multi_aff multi_aff(isl::aff_list list) const;
  inline isl::multi_aff multi_aff_on_domain(isl::multi_val mv) const;
  inline isl::multi_id multi_id(isl::id_list list) const;
  inline isl::multi_pw_aff multi_pw_aff(isl::pw_aff_list list) const;
  inline isl::multi_union_pw_aff multi_union_pw_aff(isl::union_pw_aff_list list) const;
  inline isl::multi_val multi_val(isl::val_list list) const;
  inline isl::aff param_aff_on_domain(isl::id id) const;
  inline isl::aff param_aff_on_domain(const std::string &id) const;
  inline isl::space params() const;
  static inline isl::space params_alloc(isl::ctx ctx, unsigned int nparam);
  inline isl::space product(isl::space right) const;
  inline isl::space range() const;
  inline isl::multi_aff range_map_multi_aff() const;
  inline isl::pw_multi_aff range_map_pw_multi_aff() const;
  inline isl::space range_reverse() const;
  inline isl::id range_tuple_id() const;
  inline isl::id get_range_tuple_id() const;
  inline isl::space reverse() const;
  inline isl::space set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::space set_dim_id(isl::dim type, unsigned int pos, const std::string &id) const;
  inline isl::space set_domain_tuple(isl::id id) const;
  inline isl::space set_domain_tuple(const std::string &id) const;
  inline isl::space set_from_params() const;
  inline isl::space set_range_tuple(isl::id id) const;
  inline isl::space set_range_tuple(const std::string &id) const;
  inline isl::space set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::space set_tuple_id(isl::dim type, const std::string &id) const;
  inline isl::id tuple_id(isl::dim type) const;
  inline isl::id get_tuple_id(isl::dim type) const;
  inline std::string tuple_name(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline isl::space uncurry() const;
  static inline isl::space unit(isl::ctx ctx);
  inline isl::map universe_map() const;
  inline isl::set universe_set() const;
  inline isl::space unwrap() const;
  inline isl::space wrap() const;
  inline isl::aff zero_aff_on_domain() const;
  inline isl::multi_aff zero_multi_aff() const;
  inline isl::multi_pw_aff zero_multi_pw_aff() const;
  inline isl::multi_union_pw_aff zero_multi_union_pw_aff() const;
  inline isl::multi_val zero_multi_val() const;
};

// declarations for isl::union_access_info
inline union_access_info manage(__isl_take isl_union_access_info *ptr);
inline union_access_info manage_copy(__isl_keep isl_union_access_info *ptr);

class union_access_info {
  friend inline union_access_info manage(__isl_take isl_union_access_info *ptr);
  friend inline union_access_info manage_copy(__isl_keep isl_union_access_info *ptr);

protected:
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

protected:
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

  inline isl::union_map full_may_dependence() const;
  inline isl::union_map get_full_may_dependence() const;
  inline isl::union_map full_must_dependence() const;
  inline isl::union_map get_full_must_dependence() const;
  inline isl::union_map may_dependence() const;
  inline isl::union_map get_may_dependence() const;
  inline isl::union_map may_no_source() const;
  inline isl::union_map get_may_no_source() const;
  inline isl::union_map must_dependence() const;
  inline isl::union_map get_must_dependence() const;
  inline isl::union_map must_no_source() const;
  inline isl::union_map get_must_no_source() const;
};

// declarations for isl::union_map
inline union_map manage(__isl_take isl_union_map *ptr);
inline union_map manage_copy(__isl_keep isl_union_map *ptr);

class union_map {
  friend inline union_map manage(__isl_take isl_union_map *ptr);
  friend inline union_map manage_copy(__isl_keep isl_union_map *ptr);

protected:
  isl_union_map *ptr = nullptr;

  inline explicit union_map(__isl_take isl_union_map *ptr);

public:
  inline /* implicit */ union_map();
  inline /* implicit */ union_map(const union_map &obj);
  inline /* implicit */ union_map(isl::basic_map bmap);
  inline /* implicit */ union_map(isl::map map);
  inline explicit union_map(isl::ctx ctx, const std::string &str);
  inline union_map &operator=(union_map obj);
  inline ~union_map();
  inline __isl_give isl_union_map *copy() const &;
  inline __isl_give isl_union_map *copy() && = delete;
  inline __isl_keep isl_union_map *get() const;
  inline __isl_give isl_union_map *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::union_map affine_hull() const;
  inline isl::union_map apply_domain(isl::union_map umap2) const;
  inline isl::union_map apply_range(isl::union_map umap2) const;
  inline isl::map as_map() const;
  inline isl::multi_union_pw_aff as_multi_union_pw_aff() const;
  inline isl::union_pw_multi_aff as_union_pw_multi_aff() const;
  inline isl::union_set bind_range(isl::multi_id tuple) const;
  inline isl::union_map coalesce() const;
  inline isl::union_map compute_divs() const;
  inline isl::union_map curry() const;
  inline isl::union_set deltas() const;
  inline isl::union_map detect_equalities() const;
  inline isl::union_set domain() const;
  inline isl::union_map domain_factor_domain() const;
  inline isl::union_map domain_factor_range() const;
  inline isl::union_map domain_map() const;
  inline isl::union_pw_multi_aff domain_map_union_pw_multi_aff() const;
  inline isl::union_map domain_product(isl::union_map umap2) const;
  static inline isl::union_map empty(isl::ctx ctx);
  inline isl::union_map eq_at(isl::multi_union_pw_aff mupa) const;
  inline boolean every_map(const std::function<boolean(isl::map)> &test) const;
  inline isl::map extract_map(isl::space space) const;
  inline isl::union_map factor_domain() const;
  inline isl::union_map factor_range() const;
  inline isl::union_map fixed_power(isl::val exp) const;
  inline isl::union_map fixed_power(long exp) const;
  inline isl::union_map flat_range_product(isl::union_map umap2) const;
  inline stat foreach_map(const std::function<stat(isl::map)> &fn) const;
  static inline isl::union_map from(isl::multi_union_pw_aff mupa);
  static inline isl::union_map from(isl::union_pw_multi_aff upma);
  static inline isl::union_map from_domain(isl::union_set uset);
  static inline isl::union_map from_domain_and_range(isl::union_set domain, isl::union_set range);
  static inline isl::union_map from_range(isl::union_set uset);
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
  inline boolean is_bijective() const;
  inline boolean is_disjoint(const isl::union_map &umap2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const isl::union_map &umap2) const;
  inline boolean is_injective() const;
  inline boolean is_single_valued() const;
  inline boolean is_strict_subset(const isl::union_map &umap2) const;
  inline boolean is_subset(const isl::union_map &umap2) const;
  inline boolean isa_map() const;
  inline isl::union_map lexmax() const;
  inline isl::union_map lexmin() const;
  inline isl::map_list map_list() const;
  inline isl::map_list get_map_list() const;
  inline isl::set params() const;
  inline isl::union_map polyhedral_hull() const;
  inline isl::union_map preimage_domain(isl::multi_aff ma) const;
  inline isl::union_map preimage_domain(isl::multi_pw_aff mpa) const;
  inline isl::union_map preimage_domain(isl::pw_multi_aff pma) const;
  inline isl::union_map preimage_domain(isl::union_pw_multi_aff upma) const;
  inline isl::union_map preimage_range(isl::multi_aff ma) const;
  inline isl::union_map preimage_range(isl::pw_multi_aff pma) const;
  inline isl::union_map preimage_range(isl::union_pw_multi_aff upma) const;
  inline isl::union_map product(isl::union_map umap2) const;
  inline isl::union_map project_out_all_params() const;
  inline isl::union_set range() const;
  inline isl::union_map range_factor_domain() const;
  inline isl::union_map range_factor_range() const;
  inline isl::union_map range_map() const;
  inline isl::union_map range_product(isl::union_map umap2) const;
  inline isl::union_map range_reverse() const;
  inline isl::union_map reverse() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::union_map subtract(isl::union_map umap2) const;
  inline isl::union_map subtract_domain(isl::union_set dom) const;
  inline isl::union_map subtract_range(isl::union_set dom) const;
  inline isl::union_map uncurry() const;
  inline isl::union_map unite(isl::union_map umap2) const;
  inline isl::union_map universe() const;
  inline isl::union_set wrap() const;
  inline isl::union_map zip() const;
};

// declarations for isl::union_pw_aff
inline union_pw_aff manage(__isl_take isl_union_pw_aff *ptr);
inline union_pw_aff manage_copy(__isl_keep isl_union_pw_aff *ptr);

class union_pw_aff {
  friend inline union_pw_aff manage(__isl_take isl_union_pw_aff *ptr);
  friend inline union_pw_aff manage_copy(__isl_keep isl_union_pw_aff *ptr);

protected:
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

  inline isl::multi_union_pw_aff add(const isl::multi_union_pw_aff &multi2) const;
  inline isl::union_pw_aff add(isl::union_pw_aff upa2) const;
  inline isl::union_pw_multi_aff add(const isl::union_pw_multi_aff &upma2) const;
  inline isl::union_pw_aff add(const isl::aff &upa2) const;
  inline isl::union_pw_aff add(const isl::pw_aff &upa2) const;
  inline isl::union_pw_multi_aff add_pw_multi_aff(const isl::pw_multi_aff &pma) const;
  inline isl::union_pw_multi_aff apply(const isl::union_pw_multi_aff &upma2) const;
  inline isl::multi_union_pw_aff as_multi_union_pw_aff() const;
  inline isl::pw_multi_aff as_pw_multi_aff() const;
  inline isl::union_map as_union_map() const;
  inline isl::union_pw_aff at(int pos) const;
  inline isl::union_set bind(const isl::multi_id &tuple) const;
  inline isl::union_set bind(isl::id id) const;
  inline isl::union_set bind(const std::string &id) const;
  inline isl::union_pw_aff coalesce() const;
  inline class size dim(isl::dim type) const;
  inline isl::union_set domain() const;
  static inline isl::union_pw_aff empty(isl::space space);
  inline isl::pw_multi_aff extract_pw_multi_aff(const isl::space &space) const;
  inline isl::multi_union_pw_aff flat_range_product(const isl::multi_union_pw_aff &multi2) const;
  inline isl::union_pw_multi_aff flat_range_product(const isl::union_pw_multi_aff &upma2) const;
  inline stat foreach_pw_aff(const std::function<stat(isl::pw_aff)> &fn) const;
  inline isl::union_pw_aff gist(isl::union_set context) const;
  inline boolean has_range_tuple_id() const;
  inline isl::union_pw_aff intersect_domain(isl::space space) const;
  inline isl::union_pw_aff intersect_domain(isl::union_set uset) const;
  inline isl::union_pw_aff intersect_domain_wrapped_domain(isl::union_set uset) const;
  inline isl::union_pw_aff intersect_domain_wrapped_range(isl::union_set uset) const;
  inline isl::union_pw_aff intersect_params(isl::set set) const;
  inline boolean involves_locals() const;
  inline boolean involves_nan() const;
  inline boolean isa_pw_multi_aff() const;
  inline isl::union_pw_aff_list list() const;
  inline isl::multi_union_pw_aff neg() const;
  inline boolean plain_is_empty() const;
  inline boolean plain_is_equal(const isl::multi_union_pw_aff &multi2) const;
  inline isl::union_pw_multi_aff preimage_domain_wrapped_domain(const isl::union_pw_multi_aff &upma2) const;
  inline isl::union_pw_aff pullback(isl::union_pw_multi_aff upma) const;
  inline isl::pw_multi_aff_list pw_multi_aff_list() const;
  inline isl::union_pw_multi_aff range_factor_domain() const;
  inline isl::union_pw_multi_aff range_factor_range() const;
  inline isl::multi_union_pw_aff range_product(const isl::multi_union_pw_aff &multi2) const;
  inline isl::union_pw_multi_aff range_product(const isl::union_pw_multi_aff &upma2) const;
  inline isl::id range_tuple_id() const;
  inline isl::multi_union_pw_aff reset_range_tuple_id() const;
  inline isl::multi_union_pw_aff reset_tuple_id(isl::dim type) const;
  inline isl::multi_union_pw_aff scale(const isl::multi_val &mv) const;
  inline isl::multi_union_pw_aff scale(const isl::val &v) const;
  inline isl::multi_union_pw_aff scale(long v) const;
  inline isl::multi_union_pw_aff scale_down(const isl::multi_val &mv) const;
  inline isl::multi_union_pw_aff scale_down(const isl::val &v) const;
  inline isl::multi_union_pw_aff scale_down(long v) const;
  inline isl::multi_union_pw_aff set_at(int pos, const isl::union_pw_aff &el) const;
  inline isl::multi_union_pw_aff set_range_tuple(const isl::id &id) const;
  inline isl::multi_union_pw_aff set_range_tuple(const std::string &id) const;
  inline isl::multi_union_pw_aff set_union_pw_aff(int pos, const isl::union_pw_aff &el) const;
  inline class size size() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::multi_union_pw_aff sub(const isl::multi_union_pw_aff &multi2) const;
  inline isl::union_pw_aff sub(isl::union_pw_aff upa2) const;
  inline isl::union_pw_multi_aff sub(const isl::union_pw_multi_aff &upma2) const;
  inline isl::union_pw_aff sub(const isl::aff &upa2) const;
  inline isl::union_pw_aff sub(const isl::pw_aff &upa2) const;
  inline isl::union_pw_aff subtract_domain(isl::space space) const;
  inline isl::union_pw_aff subtract_domain(isl::union_set uset) const;
  inline isl::union_pw_aff_list to_list() const;
  inline isl::multi_union_pw_aff union_add(const isl::multi_union_pw_aff &mupa2) const;
  inline isl::union_pw_aff union_add(isl::union_pw_aff upa2) const;
  inline isl::union_pw_multi_aff union_add(const isl::union_pw_multi_aff &upma2) const;
  inline isl::union_pw_aff union_add(const isl::aff &upa2) const;
  inline isl::union_pw_aff union_add(const isl::pw_aff &upa2) const;
};

// declarations for isl::union_pw_aff_list
inline union_pw_aff_list manage(__isl_take isl_union_pw_aff_list *ptr);
inline union_pw_aff_list manage_copy(__isl_keep isl_union_pw_aff_list *ptr);

class union_pw_aff_list {
  friend inline union_pw_aff_list manage(__isl_take isl_union_pw_aff_list *ptr);
  friend inline union_pw_aff_list manage_copy(__isl_keep isl_union_pw_aff_list *ptr);

protected:
  isl_union_pw_aff_list *ptr = nullptr;

  inline explicit union_pw_aff_list(__isl_take isl_union_pw_aff_list *ptr);

public:
  inline /* implicit */ union_pw_aff_list();
  inline /* implicit */ union_pw_aff_list(const union_pw_aff_list &obj);
  inline explicit union_pw_aff_list(isl::ctx ctx, int n);
  inline explicit union_pw_aff_list(isl::union_pw_aff el);
  inline explicit union_pw_aff_list(isl::ctx ctx, const std::string &str);
  inline union_pw_aff_list &operator=(union_pw_aff_list obj);
  inline ~union_pw_aff_list();
  inline __isl_give isl_union_pw_aff_list *copy() const &;
  inline __isl_give isl_union_pw_aff_list *copy() && = delete;
  inline __isl_keep isl_union_pw_aff_list *get() const;
  inline __isl_give isl_union_pw_aff_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::union_pw_aff_list add(isl::union_pw_aff el) const;
  inline isl::union_pw_aff at(int index) const;
  inline isl::union_pw_aff get_at(int index) const;
  inline isl::union_pw_aff_list clear() const;
  inline isl::union_pw_aff_list concat(isl::union_pw_aff_list list2) const;
  inline isl::union_pw_aff_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(isl::union_pw_aff)> &fn) const;
  inline isl::union_pw_aff_list insert(unsigned int pos, isl::union_pw_aff el) const;
  inline class size size() const;
};

// declarations for isl::union_pw_multi_aff
inline union_pw_multi_aff manage(__isl_take isl_union_pw_multi_aff *ptr);
inline union_pw_multi_aff manage_copy(__isl_keep isl_union_pw_multi_aff *ptr);

class union_pw_multi_aff {
  friend inline union_pw_multi_aff manage(__isl_take isl_union_pw_multi_aff *ptr);
  friend inline union_pw_multi_aff manage_copy(__isl_keep isl_union_pw_multi_aff *ptr);

protected:
  isl_union_pw_multi_aff *ptr = nullptr;

  inline explicit union_pw_multi_aff(__isl_take isl_union_pw_multi_aff *ptr);

public:
  inline /* implicit */ union_pw_multi_aff();
  inline /* implicit */ union_pw_multi_aff(const union_pw_multi_aff &obj);
  inline explicit union_pw_multi_aff(isl::union_set uset);
  inline /* implicit */ union_pw_multi_aff(isl::multi_aff ma);
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

  inline isl::union_pw_multi_aff add(isl::union_pw_multi_aff upma2) const;
  inline isl::union_pw_multi_aff add_pw_multi_aff(isl::pw_multi_aff pma) const;
  inline isl::union_pw_multi_aff apply(isl::union_pw_multi_aff upma2) const;
  inline isl::multi_union_pw_aff as_multi_union_pw_aff() const;
  inline isl::pw_multi_aff as_pw_multi_aff() const;
  inline isl::union_map as_union_map() const;
  inline isl::union_pw_multi_aff coalesce() const;
  inline isl::union_set domain() const;
  static inline isl::union_pw_multi_aff empty(isl::space space);
  static inline isl::union_pw_multi_aff empty(isl::ctx ctx);
  inline isl::pw_multi_aff extract_pw_multi_aff(isl::space space) const;
  inline isl::union_pw_multi_aff flat_range_product(isl::union_pw_multi_aff upma2) const;
  inline isl::union_pw_multi_aff gist(isl::union_set context) const;
  inline isl::union_pw_multi_aff intersect_domain(isl::space space) const;
  inline isl::union_pw_multi_aff intersect_domain(isl::union_set uset) const;
  inline isl::union_pw_multi_aff intersect_domain_wrapped_domain(isl::union_set uset) const;
  inline isl::union_pw_multi_aff intersect_domain_wrapped_range(isl::union_set uset) const;
  inline isl::union_pw_multi_aff intersect_params(isl::set set) const;
  inline boolean involves_locals() const;
  inline boolean isa_pw_multi_aff() const;
  inline boolean plain_is_empty() const;
  inline isl::union_pw_multi_aff preimage_domain_wrapped_domain(isl::union_pw_multi_aff upma2) const;
  inline isl::union_pw_multi_aff pullback(isl::union_pw_multi_aff upma2) const;
  inline isl::pw_multi_aff_list pw_multi_aff_list() const;
  inline isl::pw_multi_aff_list get_pw_multi_aff_list() const;
  inline isl::union_pw_multi_aff range_factor_domain() const;
  inline isl::union_pw_multi_aff range_factor_range() const;
  inline isl::union_pw_multi_aff range_product(isl::union_pw_multi_aff upma2) const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::union_pw_multi_aff sub(isl::union_pw_multi_aff upma2) const;
  inline isl::union_pw_multi_aff subtract_domain(isl::space space) const;
  inline isl::union_pw_multi_aff subtract_domain(isl::union_set uset) const;
  inline isl::union_pw_multi_aff union_add(isl::union_pw_multi_aff upma2) const;
};

// declarations for isl::union_set
inline union_set manage(__isl_take isl_union_set *ptr);
inline union_set manage_copy(__isl_keep isl_union_set *ptr);

class union_set {
  friend inline union_set manage(__isl_take isl_union_set *ptr);
  friend inline union_set manage_copy(__isl_keep isl_union_set *ptr);

protected:
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

  inline isl::union_set affine_hull() const;
  inline isl::union_set apply(isl::union_map umap) const;
  inline isl::set as_set() const;
  inline isl::union_set coalesce() const;
  inline isl::union_set compute_divs() const;
  inline boolean contains(const isl::space &space) const;
  inline isl::union_set detect_equalities() const;
  static inline isl::union_set empty(isl::ctx ctx);
  inline boolean every_set(const std::function<boolean(isl::set)> &test) const;
  inline isl::set extract_set(isl::space space) const;
  inline stat foreach_point(const std::function<stat(isl::point)> &fn) const;
  inline stat foreach_set(const std::function<stat(isl::set)> &fn) const;
  inline isl::union_set gist(isl::union_set context) const;
  inline isl::union_set gist_params(isl::set set) const;
  inline isl::union_map identity() const;
  inline isl::union_pw_multi_aff identity_union_pw_multi_aff() const;
  inline isl::union_set intersect(isl::union_set uset2) const;
  inline isl::union_set intersect_params(isl::set set) const;
  inline boolean is_disjoint(const isl::union_set &uset2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const isl::union_set &uset2) const;
  inline boolean is_strict_subset(const isl::union_set &uset2) const;
  inline boolean is_subset(const isl::union_set &uset2) const;
  inline boolean isa_set() const;
  inline isl::union_set lexmax() const;
  inline isl::union_set lexmin() const;
  inline isl::set params() const;
  inline isl::union_set polyhedral_hull() const;
  inline isl::union_set preimage(isl::multi_aff ma) const;
  inline isl::union_set preimage(isl::pw_multi_aff pma) const;
  inline isl::union_set preimage(isl::union_pw_multi_aff upma) const;
  inline isl::point sample_point() const;
  inline isl::set_list set_list() const;
  inline isl::set_list get_set_list() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::union_set subtract(isl::union_set uset2) const;
  inline isl::union_set_list to_list() const;
  inline isl::union_set unite(isl::union_set uset2) const;
  inline isl::union_set universe() const;
  inline isl::union_map unwrap() const;
};

// declarations for isl::union_set_list
inline union_set_list manage(__isl_take isl_union_set_list *ptr);
inline union_set_list manage_copy(__isl_keep isl_union_set_list *ptr);

class union_set_list {
  friend inline union_set_list manage(__isl_take isl_union_set_list *ptr);
  friend inline union_set_list manage_copy(__isl_keep isl_union_set_list *ptr);

protected:
  isl_union_set_list *ptr = nullptr;

  inline explicit union_set_list(__isl_take isl_union_set_list *ptr);

public:
  inline /* implicit */ union_set_list();
  inline /* implicit */ union_set_list(const union_set_list &obj);
  inline explicit union_set_list(isl::ctx ctx, int n);
  inline explicit union_set_list(isl::union_set el);
  inline explicit union_set_list(isl::ctx ctx, const std::string &str);
  inline union_set_list &operator=(union_set_list obj);
  inline ~union_set_list();
  inline __isl_give isl_union_set_list *copy() const &;
  inline __isl_give isl_union_set_list *copy() && = delete;
  inline __isl_keep isl_union_set_list *get() const;
  inline __isl_give isl_union_set_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::union_set_list add(isl::union_set el) const;
  inline isl::union_set at(int index) const;
  inline isl::union_set get_at(int index) const;
  inline isl::union_set_list clear() const;
  inline isl::union_set_list concat(isl::union_set_list list2) const;
  inline isl::union_set_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(isl::union_set)> &fn) const;
  inline isl::union_set_list insert(unsigned int pos, isl::union_set el) const;
  inline class size size() const;
};

// declarations for isl::val
inline val manage(__isl_take isl_val *ptr);
inline val manage_copy(__isl_keep isl_val *ptr);

class val {
  friend inline val manage(__isl_take isl_val *ptr);
  friend inline val manage_copy(__isl_keep isl_val *ptr);

protected:
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

  inline isl::val abs() const;
  inline boolean abs_eq(const isl::val &v2) const;
  inline boolean abs_eq(long v2) const;
  inline isl::val add(isl::val v2) const;
  inline isl::val add(long v2) const;
  inline isl::val ceil() const;
  inline int cmp_si(long i) const;
  inline long den_si() const;
  inline long get_den_si() const;
  inline isl::val div(isl::val v2) const;
  inline isl::val div(long v2) const;
  inline boolean eq(const isl::val &v2) const;
  inline boolean eq(long v2) const;
  inline isl::val floor() const;
  inline isl::val gcd(isl::val v2) const;
  inline isl::val gcd(long v2) const;
  inline boolean ge(const isl::val &v2) const;
  inline boolean ge(long v2) const;
  inline boolean gt(const isl::val &v2) const;
  inline boolean gt(long v2) const;
  static inline isl::val infty(isl::ctx ctx);
  static inline isl::val int_from_ui(isl::ctx ctx, unsigned long u);
  inline isl::val inv() const;
  inline boolean is_divisible_by(const isl::val &v2) const;
  inline boolean is_divisible_by(long v2) const;
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
  inline boolean le(long v2) const;
  inline boolean lt(const isl::val &v2) const;
  inline boolean lt(long v2) const;
  inline isl::val max(isl::val v2) const;
  inline isl::val max(long v2) const;
  inline isl::val min(isl::val v2) const;
  inline isl::val min(long v2) const;
  inline isl::val mod(isl::val v2) const;
  inline isl::val mod(long v2) const;
  inline isl::val mul(isl::val v2) const;
  inline isl::val mul(long v2) const;
  static inline isl::val nan(isl::ctx ctx);
  inline boolean ne(const isl::val &v2) const;
  inline boolean ne(long v2) const;
  inline isl::val neg() const;
  static inline isl::val neginfty(isl::ctx ctx);
  static inline isl::val negone(isl::ctx ctx);
  inline long num_si() const;
  inline long get_num_si() const;
  static inline isl::val one(isl::ctx ctx);
  inline isl::val pow2() const;
  inline int sgn() const;
  inline isl::val sub(isl::val v2) const;
  inline isl::val sub(long v2) const;
  inline isl::val_list to_list() const;
  inline isl::val trunc() const;
  static inline isl::val zero(isl::ctx ctx);
};

// declarations for isl::val_list
inline val_list manage(__isl_take isl_val_list *ptr);
inline val_list manage_copy(__isl_keep isl_val_list *ptr);

class val_list {
  friend inline val_list manage(__isl_take isl_val_list *ptr);
  friend inline val_list manage_copy(__isl_keep isl_val_list *ptr);

protected:
  isl_val_list *ptr = nullptr;

  inline explicit val_list(__isl_take isl_val_list *ptr);

public:
  inline /* implicit */ val_list();
  inline /* implicit */ val_list(const val_list &obj);
  inline explicit val_list(isl::ctx ctx, int n);
  inline explicit val_list(isl::val el);
  inline explicit val_list(isl::ctx ctx, const std::string &str);
  inline val_list &operator=(val_list obj);
  inline ~val_list();
  inline __isl_give isl_val_list *copy() const &;
  inline __isl_give isl_val_list *copy() && = delete;
  inline __isl_keep isl_val_list *get() const;
  inline __isl_give isl_val_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::val_list add(isl::val el) const;
  inline isl::val_list add(long el) const;
  inline isl::val at(int index) const;
  inline isl::val get_at(int index) const;
  inline isl::val_list clear() const;
  inline isl::val_list concat(isl::val_list list2) const;
  inline isl::val_list drop(unsigned int first, unsigned int n) const;
  inline stat foreach(const std::function<stat(isl::val)> &fn) const;
  inline isl::val_list insert(unsigned int pos, isl::val el) const;
  inline isl::val_list insert(unsigned int pos, long el) const;
  inline class size size() const;
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

isl::aff aff::add(isl::aff aff2) const
{
  auto res = isl_aff_add(copy(), aff2.release());
  return manage(res);
}

isl::multi_aff aff::add(const isl::multi_aff &multi2) const
{
  return isl::multi_aff(*this).add(multi2);
}

isl::multi_pw_aff aff::add(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_aff(*this).add(multi2);
}

isl::multi_union_pw_aff aff::add(const isl::multi_union_pw_aff &multi2) const
{
  return isl::pw_aff(*this).add(multi2);
}

isl::pw_aff aff::add(const isl::pw_aff &pwaff2) const
{
  return isl::pw_aff(*this).add(pwaff2);
}

isl::pw_multi_aff aff::add(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_aff(*this).add(pma2);
}

isl::union_pw_aff aff::add(const isl::union_pw_aff &upa2) const
{
  return isl::pw_aff(*this).add(upa2);
}

isl::union_pw_multi_aff aff::add(const isl::union_pw_multi_aff &upma2) const
{
  return isl::pw_aff(*this).add(upma2);
}

isl::aff aff::add_constant(isl::val v) const
{
  auto res = isl_aff_add_constant_val(copy(), v.release());
  return manage(res);
}

isl::aff aff::add_constant(long v) const
{
  return this->add_constant(isl::val(ctx(), v));
}

isl::multi_aff aff::add_constant(const isl::multi_val &mv) const
{
  return isl::multi_aff(*this).add_constant(mv);
}

isl::aff aff::add_constant_si(int v) const
{
  auto res = isl_aff_add_constant_si(copy(), v);
  return manage(res);
}

isl::pw_aff aff::add_dims(isl::dim type, unsigned int n) const
{
  return isl::pw_aff(*this).add_dims(type, n);
}

isl::union_pw_multi_aff aff::add_pw_multi_aff(const isl::pw_multi_aff &pma) const
{
  return isl::pw_aff(*this).add_pw_multi_aff(pma);
}

isl::union_pw_multi_aff aff::apply(const isl::union_pw_multi_aff &upma2) const
{
  return isl::pw_aff(*this).apply(upma2);
}

isl::aff aff::as_aff() const
{
  return isl::pw_aff(*this).as_aff();
}

isl::map aff::as_map() const
{
  return isl::pw_aff(*this).as_map();
}

isl::multi_aff aff::as_multi_aff() const
{
  return isl::pw_aff(*this).as_multi_aff();
}

isl::multi_union_pw_aff aff::as_multi_union_pw_aff() const
{
  return isl::pw_aff(*this).as_multi_union_pw_aff();
}

isl::pw_multi_aff aff::as_pw_multi_aff() const
{
  return isl::pw_aff(*this).as_pw_multi_aff();
}

isl::set aff::as_set() const
{
  return isl::multi_aff(*this).as_set();
}

isl::union_map aff::as_union_map() const
{
  return isl::pw_aff(*this).as_union_map();
}

isl::aff aff::at(int pos) const
{
  return isl::multi_aff(*this).at(pos);
}

isl::basic_set aff::bind(isl::id id) const
{
  auto res = isl_aff_bind_id(copy(), id.release());
  return manage(res);
}

isl::basic_set aff::bind(const std::string &id) const
{
  return this->bind(isl::id(ctx(), id));
}

isl::basic_set aff::bind(const isl::multi_id &tuple) const
{
  return isl::multi_aff(*this).bind(tuple);
}

isl::pw_aff aff::bind_domain(const isl::multi_id &tuple) const
{
  return isl::pw_aff(*this).bind_domain(tuple);
}

isl::pw_aff aff::bind_domain_wrapped_domain(const isl::multi_id &tuple) const
{
  return isl::pw_aff(*this).bind_domain_wrapped_domain(tuple);
}

isl::aff aff::ceil() const
{
  auto res = isl_aff_ceil(copy());
  return manage(res);
}

isl::pw_aff aff::coalesce() const
{
  return isl::pw_aff(*this).coalesce();
}

isl::pw_aff aff::cond(const isl::pw_aff &pwaff_true, const isl::pw_aff &pwaff_false) const
{
  return isl::pw_aff(*this).cond(pwaff_true, pwaff_false);
}

isl::multi_val aff::constant_multi_val() const
{
  return isl::multi_aff(*this).constant_multi_val();
}

isl::val aff::constant_val() const
{
  auto res = isl_aff_get_constant_val(get());
  return manage(res);
}

isl::val aff::get_constant_val() const
{
  return constant_val();
}

isl::val aff::denominator_val() const
{
  auto res = isl_aff_get_denominator_val(get());
  return manage(res);
}

isl::val aff::get_denominator_val() const
{
  return denominator_val();
}

class size aff::dim(isl::dim type) const
{
  return isl::multi_aff(*this).dim(type);
}

isl::id aff::dim_id(isl::dim type, unsigned int pos) const
{
  return isl::pw_aff(*this).dim_id(type, pos);
}

isl::aff aff::div(isl::aff aff2) const
{
  auto res = isl_aff_div(copy(), aff2.release());
  return manage(res);
}

isl::pw_aff aff::div(const isl::pw_aff &pa2) const
{
  return isl::pw_aff(*this).div(pa2);
}

isl::set aff::domain() const
{
  return isl::pw_aff(*this).domain();
}

isl::space aff::domain_space() const
{
  return isl::pw_aff(*this).domain_space();
}

isl::pw_multi_aff aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  return isl::pw_aff(*this).drop_dims(type, first, n);
}

isl::set aff::eq_set(isl::aff aff2) const
{
  auto res = isl_aff_eq_set(copy(), aff2.release());
  return manage(res);
}

isl::set aff::eq_set(const isl::pw_aff &pwaff2) const
{
  return isl::pw_aff(*this).eq_set(pwaff2);
}

isl::val aff::eval(isl::point pnt) const
{
  auto res = isl_aff_eval(copy(), pnt.release());
  return manage(res);
}

isl::pw_multi_aff aff::extract_pw_multi_aff(const isl::space &space) const
{
  return isl::pw_aff(*this).extract_pw_multi_aff(space);
}

isl::multi_aff aff::flat_range_product(const isl::multi_aff &multi2) const
{
  return isl::multi_aff(*this).flat_range_product(multi2);
}

isl::multi_pw_aff aff::flat_range_product(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_aff(*this).flat_range_product(multi2);
}

isl::multi_union_pw_aff aff::flat_range_product(const isl::multi_union_pw_aff &multi2) const
{
  return isl::pw_aff(*this).flat_range_product(multi2);
}

isl::pw_multi_aff aff::flat_range_product(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_aff(*this).flat_range_product(pma2);
}

isl::union_pw_multi_aff aff::flat_range_product(const isl::union_pw_multi_aff &upma2) const
{
  return isl::pw_aff(*this).flat_range_product(upma2);
}

isl::aff aff::floor() const
{
  auto res = isl_aff_floor(copy());
  return manage(res);
}

stat aff::foreach_piece(const std::function<stat(isl::set, isl::aff)> &fn) const
{
  return isl::pw_aff(*this).foreach_piece(fn);
}

stat aff::foreach_piece(const std::function<stat(isl::set, isl::multi_aff)> &fn) const
{
  return isl::pw_aff(*this).foreach_piece(fn);
}

stat aff::foreach_pw_aff(const std::function<stat(isl::pw_aff)> &fn) const
{
  return isl::pw_aff(*this).foreach_pw_aff(fn);
}

isl::set aff::ge_set(isl::aff aff2) const
{
  auto res = isl_aff_ge_set(copy(), aff2.release());
  return manage(res);
}

isl::set aff::ge_set(const isl::pw_aff &pwaff2) const
{
  return isl::pw_aff(*this).ge_set(pwaff2);
}

isl::aff aff::gist(isl::set context) const
{
  auto res = isl_aff_gist(copy(), context.release());
  return manage(res);
}

isl::union_pw_aff aff::gist(const isl::union_set &context) const
{
  return isl::pw_aff(*this).gist(context);
}

isl::aff aff::gist(const isl::basic_set &context) const
{
  return this->gist(isl::set(context));
}

isl::aff aff::gist(const isl::point &context) const
{
  return this->gist(isl::set(context));
}

isl::set aff::gt_set(isl::aff aff2) const
{
  auto res = isl_aff_gt_set(copy(), aff2.release());
  return manage(res);
}

isl::set aff::gt_set(const isl::pw_aff &pwaff2) const
{
  return isl::pw_aff(*this).gt_set(pwaff2);
}

boolean aff::has_range_tuple_id() const
{
  return isl::multi_aff(*this).has_range_tuple_id();
}

isl::multi_aff aff::identity() const
{
  return isl::multi_aff(*this).identity();
}

isl::pw_aff aff::insert_domain(const isl::space &domain) const
{
  return isl::pw_aff(*this).insert_domain(domain);
}

isl::pw_aff aff::intersect_domain(const isl::set &set) const
{
  return isl::pw_aff(*this).intersect_domain(set);
}

isl::union_pw_aff aff::intersect_domain(const isl::space &space) const
{
  return isl::pw_aff(*this).intersect_domain(space);
}

isl::union_pw_aff aff::intersect_domain(const isl::union_set &uset) const
{
  return isl::pw_aff(*this).intersect_domain(uset);
}

isl::union_pw_aff aff::intersect_domain_wrapped_domain(const isl::union_set &uset) const
{
  return isl::pw_aff(*this).intersect_domain_wrapped_domain(uset);
}

isl::union_pw_aff aff::intersect_domain_wrapped_range(const isl::union_set &uset) const
{
  return isl::pw_aff(*this).intersect_domain_wrapped_range(uset);
}

isl::pw_aff aff::intersect_params(const isl::set &set) const
{
  return isl::pw_aff(*this).intersect_params(set);
}

boolean aff::involves_locals() const
{
  return isl::multi_aff(*this).involves_locals();
}

boolean aff::involves_nan() const
{
  return isl::multi_aff(*this).involves_nan();
}

boolean aff::involves_param(const isl::id &id) const
{
  return isl::pw_aff(*this).involves_param(id);
}

boolean aff::involves_param(const std::string &id) const
{
  return this->involves_param(isl::id(ctx(), id));
}

boolean aff::involves_param(const isl::id_list &list) const
{
  return isl::pw_aff(*this).involves_param(list);
}

boolean aff::is_cst() const
{
  auto res = isl_aff_is_cst(get());
  return manage(res);
}

boolean aff::is_equal(const isl::pw_aff &pa2) const
{
  return isl::pw_aff(*this).is_equal(pa2);
}

boolean aff::isa_aff() const
{
  return isl::pw_aff(*this).isa_aff();
}

boolean aff::isa_multi_aff() const
{
  return isl::pw_aff(*this).isa_multi_aff();
}

boolean aff::isa_pw_multi_aff() const
{
  return isl::pw_aff(*this).isa_pw_multi_aff();
}

isl::set aff::le_set(isl::aff aff2) const
{
  auto res = isl_aff_le_set(copy(), aff2.release());
  return manage(res);
}

isl::set aff::le_set(const isl::pw_aff &pwaff2) const
{
  return isl::pw_aff(*this).le_set(pwaff2);
}

isl::aff_list aff::list() const
{
  return isl::multi_aff(*this).list();
}

isl::set aff::lt_set(isl::aff aff2) const
{
  auto res = isl_aff_lt_set(copy(), aff2.release());
  return manage(res);
}

isl::set aff::lt_set(const isl::pw_aff &pwaff2) const
{
  return isl::pw_aff(*this).lt_set(pwaff2);
}

isl::multi_pw_aff aff::max(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_aff(*this).max(multi2);
}

isl::pw_aff aff::max(const isl::pw_aff &pwaff2) const
{
  return isl::pw_aff(*this).max(pwaff2);
}

isl::multi_val aff::max_multi_val() const
{
  return isl::pw_aff(*this).max_multi_val();
}

isl::multi_pw_aff aff::min(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_aff(*this).min(multi2);
}

isl::pw_aff aff::min(const isl::pw_aff &pwaff2) const
{
  return isl::pw_aff(*this).min(pwaff2);
}

isl::multi_val aff::min_multi_val() const
{
  return isl::pw_aff(*this).min_multi_val();
}

isl::aff aff::mod(isl::val mod) const
{
  auto res = isl_aff_mod_val(copy(), mod.release());
  return manage(res);
}

isl::aff aff::mod(long mod) const
{
  return this->mod(isl::val(ctx(), mod));
}

isl::aff aff::mul(isl::aff aff2) const
{
  auto res = isl_aff_mul(copy(), aff2.release());
  return manage(res);
}

isl::pw_aff aff::mul(const isl::pw_aff &pwaff2) const
{
  return isl::pw_aff(*this).mul(pwaff2);
}

class size aff::n_piece() const
{
  return isl::pw_aff(*this).n_piece();
}

isl::set aff::ne_set(isl::aff aff2) const
{
  auto res = isl_aff_ne_set(copy(), aff2.release());
  return manage(res);
}

isl::set aff::ne_set(const isl::pw_aff &pwaff2) const
{
  return isl::pw_aff(*this).ne_set(pwaff2);
}

isl::aff aff::neg() const
{
  auto res = isl_aff_neg(copy());
  return manage(res);
}

boolean aff::plain_is_empty() const
{
  return isl::pw_aff(*this).plain_is_empty();
}

boolean aff::plain_is_equal(const isl::multi_aff &multi2) const
{
  return isl::multi_aff(*this).plain_is_equal(multi2);
}

boolean aff::plain_is_equal(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_aff(*this).plain_is_equal(multi2);
}

boolean aff::plain_is_equal(const isl::multi_union_pw_aff &multi2) const
{
  return isl::pw_aff(*this).plain_is_equal(multi2);
}

isl::pw_multi_aff aff::preimage_domain_wrapped_domain(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_aff(*this).preimage_domain_wrapped_domain(pma2);
}

isl::union_pw_multi_aff aff::preimage_domain_wrapped_domain(const isl::union_pw_multi_aff &upma2) const
{
  return isl::pw_aff(*this).preimage_domain_wrapped_domain(upma2);
}

isl::multi_aff aff::product(const isl::multi_aff &multi2) const
{
  return isl::multi_aff(*this).product(multi2);
}

isl::multi_pw_aff aff::product(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_aff(*this).product(multi2);
}

isl::pw_multi_aff aff::product(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_aff(*this).product(pma2);
}

isl::aff aff::pullback(isl::multi_aff ma) const
{
  auto res = isl_aff_pullback_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::pw_aff aff::pullback(const isl::multi_pw_aff &mpa) const
{
  return isl::pw_aff(*this).pullback(mpa);
}

isl::pw_aff aff::pullback(const isl::pw_multi_aff &pma) const
{
  return isl::pw_aff(*this).pullback(pma);
}

isl::union_pw_aff aff::pullback(const isl::union_pw_multi_aff &upma) const
{
  return isl::pw_aff(*this).pullback(upma);
}

isl::aff aff::pullback(const isl::aff &ma) const
{
  return this->pullback(isl::multi_aff(ma));
}

isl::pw_multi_aff_list aff::pw_multi_aff_list() const
{
  return isl::pw_aff(*this).pw_multi_aff_list();
}

isl::pw_multi_aff aff::range_factor_domain() const
{
  return isl::pw_aff(*this).range_factor_domain();
}

isl::pw_multi_aff aff::range_factor_range() const
{
  return isl::pw_aff(*this).range_factor_range();
}

isl::multi_aff aff::range_product(const isl::multi_aff &multi2) const
{
  return isl::multi_aff(*this).range_product(multi2);
}

isl::multi_pw_aff aff::range_product(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_aff(*this).range_product(multi2);
}

isl::multi_union_pw_aff aff::range_product(const isl::multi_union_pw_aff &multi2) const
{
  return isl::pw_aff(*this).range_product(multi2);
}

isl::pw_multi_aff aff::range_product(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_aff(*this).range_product(pma2);
}

isl::union_pw_multi_aff aff::range_product(const isl::union_pw_multi_aff &upma2) const
{
  return isl::pw_aff(*this).range_product(upma2);
}

isl::id aff::range_tuple_id() const
{
  return isl::multi_aff(*this).range_tuple_id();
}

isl::multi_aff aff::reset_range_tuple_id() const
{
  return isl::multi_aff(*this).reset_range_tuple_id();
}

isl::multi_aff aff::reset_tuple_id(isl::dim type) const
{
  return isl::multi_aff(*this).reset_tuple_id(type);
}

isl::aff aff::scale(isl::val v) const
{
  auto res = isl_aff_scale_val(copy(), v.release());
  return manage(res);
}

isl::aff aff::scale(long v) const
{
  return this->scale(isl::val(ctx(), v));
}

isl::multi_aff aff::scale(const isl::multi_val &mv) const
{
  return isl::multi_aff(*this).scale(mv);
}

isl::aff aff::scale_down(isl::val v) const
{
  auto res = isl_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::aff aff::scale_down(long v) const
{
  return this->scale_down(isl::val(ctx(), v));
}

isl::multi_aff aff::scale_down(const isl::multi_val &mv) const
{
  return isl::multi_aff(*this).scale_down(mv);
}

isl::multi_aff aff::set_aff(int pos, const isl::aff &el) const
{
  return isl::multi_aff(*this).set_aff(pos, el);
}

isl::multi_aff aff::set_at(int pos, const isl::aff &el) const
{
  return isl::multi_aff(*this).set_at(pos, el);
}

isl::multi_pw_aff aff::set_at(int pos, const isl::pw_aff &el) const
{
  return isl::pw_aff(*this).set_at(pos, el);
}

isl::multi_union_pw_aff aff::set_at(int pos, const isl::union_pw_aff &el) const
{
  return isl::pw_aff(*this).set_at(pos, el);
}

isl::aff aff::set_constant_si(int v) const
{
  auto res = isl_aff_set_constant_si(copy(), v);
  return manage(res);
}

isl::multi_pw_aff aff::set_pw_aff(int pos, const isl::pw_aff &el) const
{
  return isl::pw_aff(*this).set_pw_aff(pos, el);
}

isl::pw_multi_aff aff::set_pw_aff(unsigned int pos, const isl::pw_aff &pa) const
{
  return isl::pw_aff(*this).set_pw_aff(pos, pa);
}

isl::multi_aff aff::set_range_tuple(const isl::id &id) const
{
  return isl::multi_aff(*this).set_range_tuple(id);
}

isl::multi_aff aff::set_range_tuple(const std::string &id) const
{
  return this->set_range_tuple(isl::id(ctx(), id));
}

isl::pw_aff aff::set_tuple_id(isl::dim type, const isl::id &id) const
{
  return isl::pw_aff(*this).set_tuple_id(type, id);
}

isl::pw_aff aff::set_tuple_id(isl::dim type, const std::string &id) const
{
  return this->set_tuple_id(type, isl::id(ctx(), id));
}

isl::multi_union_pw_aff aff::set_union_pw_aff(int pos, const isl::union_pw_aff &el) const
{
  return isl::pw_aff(*this).set_union_pw_aff(pos, el);
}

class size aff::size() const
{
  return isl::multi_aff(*this).size();
}

isl::space aff::space() const
{
  return isl::pw_aff(*this).space();
}

isl::aff aff::sub(isl::aff aff2) const
{
  auto res = isl_aff_sub(copy(), aff2.release());
  return manage(res);
}

isl::multi_aff aff::sub(const isl::multi_aff &multi2) const
{
  return isl::multi_aff(*this).sub(multi2);
}

isl::multi_pw_aff aff::sub(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_aff(*this).sub(multi2);
}

isl::multi_union_pw_aff aff::sub(const isl::multi_union_pw_aff &multi2) const
{
  return isl::pw_aff(*this).sub(multi2);
}

isl::pw_aff aff::sub(const isl::pw_aff &pwaff2) const
{
  return isl::pw_aff(*this).sub(pwaff2);
}

isl::pw_multi_aff aff::sub(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_aff(*this).sub(pma2);
}

isl::union_pw_aff aff::sub(const isl::union_pw_aff &upa2) const
{
  return isl::pw_aff(*this).sub(upa2);
}

isl::union_pw_multi_aff aff::sub(const isl::union_pw_multi_aff &upma2) const
{
  return isl::pw_aff(*this).sub(upma2);
}

isl::pw_aff aff::subtract_domain(const isl::set &set) const
{
  return isl::pw_aff(*this).subtract_domain(set);
}

isl::union_pw_aff aff::subtract_domain(const isl::space &space) const
{
  return isl::pw_aff(*this).subtract_domain(space);
}

isl::union_pw_aff aff::subtract_domain(const isl::union_set &uset) const
{
  return isl::pw_aff(*this).subtract_domain(uset);
}

isl::pw_aff aff::tdiv_q(const isl::pw_aff &pa2) const
{
  return isl::pw_aff(*this).tdiv_q(pa2);
}

isl::pw_aff aff::tdiv_r(const isl::pw_aff &pa2) const
{
  return isl::pw_aff(*this).tdiv_r(pa2);
}

isl::aff_list aff::to_list() const
{
  auto res = isl_aff_to_list(copy());
  return manage(res);
}

isl::multi_pw_aff aff::to_multi_pw_aff() const
{
  return isl::multi_aff(*this).to_multi_pw_aff();
}

isl::multi_union_pw_aff aff::to_multi_union_pw_aff() const
{
  return isl::multi_aff(*this).to_multi_union_pw_aff();
}

isl::pw_multi_aff aff::to_pw_multi_aff() const
{
  return isl::multi_aff(*this).to_pw_multi_aff();
}

isl::union_pw_aff aff::to_union_pw_aff() const
{
  return isl::pw_aff(*this).to_union_pw_aff();
}

isl::union_pw_multi_aff aff::to_union_pw_multi_aff() const
{
  return isl::pw_aff(*this).to_union_pw_multi_aff();
}

isl::id aff::tuple_id(isl::dim type) const
{
  return isl::pw_aff(*this).tuple_id(type);
}

isl::aff aff::unbind_params_insert_domain(isl::multi_id domain) const
{
  auto res = isl_aff_unbind_params_insert_domain(copy(), domain.release());
  return manage(res);
}

isl::multi_pw_aff aff::union_add(const isl::multi_pw_aff &mpa2) const
{
  return isl::pw_aff(*this).union_add(mpa2);
}

isl::multi_union_pw_aff aff::union_add(const isl::multi_union_pw_aff &mupa2) const
{
  return isl::pw_aff(*this).union_add(mupa2);
}

isl::pw_aff aff::union_add(const isl::pw_aff &pwaff2) const
{
  return isl::pw_aff(*this).union_add(pwaff2);
}

isl::pw_multi_aff aff::union_add(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_aff(*this).union_add(pma2);
}

isl::union_pw_aff aff::union_add(const isl::union_pw_aff &upa2) const
{
  return isl::pw_aff(*this).union_add(upa2);
}

isl::union_pw_multi_aff aff::union_add(const isl::union_pw_multi_aff &upma2) const
{
  return isl::pw_aff(*this).union_add(upma2);
}

isl::aff aff::var_on_domain(isl::local_space ls, isl::dim type, unsigned int pos)
{
  auto res = isl_aff_var_on_domain(ls.release(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::aff aff::zero_on_domain(isl::space space)
{
  auto res = isl_aff_zero_on_domain_space(space.release());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const aff &obj)
{
  char *str = isl_aff_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

aff_list::aff_list(isl::ctx ctx, int n)
{
  auto res = isl_aff_list_alloc(ctx.release(), n);
  ptr = res;
}

aff_list::aff_list(isl::aff el)
{
  auto res = isl_aff_list_from_aff(el.release());
  ptr = res;
}

aff_list::aff_list(isl::ctx ctx, const std::string &str)
{
  auto res = isl_aff_list_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

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

isl::aff_list aff_list::add(isl::aff el) const
{
  auto res = isl_aff_list_add(copy(), el.release());
  return manage(res);
}

isl::aff aff_list::at(int index) const
{
  auto res = isl_aff_list_get_at(get(), index);
  return manage(res);
}

isl::aff aff_list::get_at(int index) const
{
  return at(index);
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

stat aff_list::foreach(const std::function<stat(isl::aff)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::aff)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_aff_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

isl::aff_list aff_list::insert(unsigned int pos, isl::aff el) const
{
  auto res = isl_aff_list_insert(copy(), pos, el.release());
  return manage(res);
}

class size aff_list::size() const
{
  auto res = isl_aff_list_size(get());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const aff_list &obj)
{
  char *str = isl_aff_list_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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
  copy_callbacks(obj);
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
  copy_callbacks(obj);
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
  if (at_each_domain_data)
    isl_die(ctx().get(), isl_error_invalid, "cannot release object with persistent callbacks", return nullptr);
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

ast_build &ast_build::copy_callbacks(const ast_build &obj)
{
  at_each_domain_data = obj.at_each_domain_data;
  return *this;
}

isl_ast_node *ast_build::at_each_domain(isl_ast_node *arg_0, isl_ast_build *arg_1, void *arg_2)
{
  auto *data = static_cast<struct at_each_domain_data *>(arg_2);
  auto ret = (data->func)(manage(arg_0), manage_copy(arg_1));
  return ret.release();
}

void ast_build::set_at_each_domain_data(const std::function<isl::ast_node(isl::ast_node, isl::ast_build)> &fn)
{
  at_each_domain_data = std::make_shared<struct at_each_domain_data>();
  at_each_domain_data->func = fn;
  ptr = isl_ast_build_set_at_each_domain(ptr, &at_each_domain, at_each_domain_data.get());
}

isl::ast_build ast_build::set_at_each_domain(const std::function<isl::ast_node(isl::ast_node, isl::ast_build)> &fn) const
{
  auto copy = *this;
  copy.set_at_each_domain_data(fn);
  return copy;
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
  return manage(res).copy_callbacks(*this);
}

isl::union_map ast_build::schedule() const
{
  auto res = isl_ast_build_get_schedule(get());
  return manage(res);
}

isl::union_map ast_build::get_schedule() const
{
  return schedule();
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

template <typename T, typename>
boolean ast_expr::isa_type(T subtype) const
{
  if (is_null())
    return boolean();
  return isl_ast_expr_get_type(get()) == subtype;
}
template <class T>
boolean ast_expr::isa() const
{
  return isa_type<decltype(T::type)>(T::type);
}
template <class T>
T ast_expr::as() const
{
 if (isa<T>().is_false())
    isl_die(ctx().get(), isl_error_invalid, "not an object of the requested subtype", return T());
  return T(copy());
}

isl::ctx ast_expr::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
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

isl::ast_expr ast_expr::eq(isl::ast_expr expr2) const
{
  auto res = isl_ast_expr_eq(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::from_val(isl::val v)
{
  auto res = isl_ast_expr_from_val(v.release());
  return manage(res);
}

isl::id ast_expr::id() const
{
  auto res = isl_ast_expr_get_id(get());
  return manage(res);
}

isl::id ast_expr::get_id() const
{
  return id();
}

isl::ast_expr ast_expr::le(isl::ast_expr expr2) const
{
  auto res = isl_ast_expr_le(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::mul(isl::ast_expr expr2) const
{
  auto res = isl_ast_expr_mul(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::op_arg(int pos) const
{
  auto res = isl_ast_expr_get_op_arg(get(), pos);
  return manage(res);
}

isl::ast_expr ast_expr::get_op_arg(int pos) const
{
  return op_arg(pos);
}

std::string ast_expr::to_C_str() const
{
  auto res = isl_ast_expr_to_C_str(get());
  std::string tmp(res);
  free(res);
  return tmp;
}

isl::val ast_expr::val() const
{
  auto res = isl_ast_expr_get_val(get());
  return manage(res);
}

isl::val ast_expr::get_val() const
{
  return val();
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_id
ast_expr_id::ast_expr_id()
    : ast_expr() {}

ast_expr_id::ast_expr_id(const ast_expr_id &obj)
    : ast_expr(obj)
{
}

ast_expr_id::ast_expr_id(__isl_take isl_ast_expr *ptr)
    : ast_expr(ptr) {}

ast_expr_id &ast_expr_id::operator=(ast_expr_id obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_id::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

isl::id ast_expr_id::id() const
{
  auto res = isl_ast_expr_id_get_id(get());
  return manage(res);
}

isl::id ast_expr_id::get_id() const
{
  return id();
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_id &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_int
ast_expr_int::ast_expr_int()
    : ast_expr() {}

ast_expr_int::ast_expr_int(const ast_expr_int &obj)
    : ast_expr(obj)
{
}

ast_expr_int::ast_expr_int(__isl_take isl_ast_expr *ptr)
    : ast_expr(ptr) {}

ast_expr_int &ast_expr_int::operator=(ast_expr_int obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_int::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

isl::val ast_expr_int::val() const
{
  auto res = isl_ast_expr_int_get_val(get());
  return manage(res);
}

isl::val ast_expr_int::get_val() const
{
  return val();
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_int &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op
ast_expr_op::ast_expr_op()
    : ast_expr() {}

ast_expr_op::ast_expr_op(const ast_expr_op &obj)
    : ast_expr(obj)
{
}

ast_expr_op::ast_expr_op(__isl_take isl_ast_expr *ptr)
    : ast_expr(ptr) {}

ast_expr_op &ast_expr_op::operator=(ast_expr_op obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

template <typename T, typename>
boolean ast_expr_op::isa_type(T subtype) const
{
  if (is_null())
    return boolean();
  return isl_ast_expr_op_get_type(get()) == subtype;
}
template <class T>
boolean ast_expr_op::isa() const
{
  return isa_type<decltype(T::type)>(T::type);
}
template <class T>
T ast_expr_op::as() const
{
 if (isa<T>().is_false())
    isl_die(ctx().get(), isl_error_invalid, "not an object of the requested subtype", return T());
  return T(copy());
}

isl::ctx ast_expr_op::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

isl::ast_expr ast_expr_op::arg(int pos) const
{
  auto res = isl_ast_expr_op_get_arg(get(), pos);
  return manage(res);
}

isl::ast_expr ast_expr_op::get_arg(int pos) const
{
  return arg(pos);
}

class size ast_expr_op::n_arg() const
{
  auto res = isl_ast_expr_op_get_n_arg(get());
  return manage(res);
}

class size ast_expr_op::get_n_arg() const
{
  return n_arg();
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_access
ast_expr_op_access::ast_expr_op_access()
    : ast_expr_op() {}

ast_expr_op_access::ast_expr_op_access(const ast_expr_op_access &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_access::ast_expr_op_access(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_access &ast_expr_op_access::operator=(ast_expr_op_access obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_access::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_access &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_add
ast_expr_op_add::ast_expr_op_add()
    : ast_expr_op() {}

ast_expr_op_add::ast_expr_op_add(const ast_expr_op_add &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_add::ast_expr_op_add(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_add &ast_expr_op_add::operator=(ast_expr_op_add obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_add::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_add &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_address_of
ast_expr_op_address_of::ast_expr_op_address_of()
    : ast_expr_op() {}

ast_expr_op_address_of::ast_expr_op_address_of(const ast_expr_op_address_of &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_address_of::ast_expr_op_address_of(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_address_of &ast_expr_op_address_of::operator=(ast_expr_op_address_of obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_address_of::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_address_of &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_and
ast_expr_op_and::ast_expr_op_and()
    : ast_expr_op() {}

ast_expr_op_and::ast_expr_op_and(const ast_expr_op_and &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_and::ast_expr_op_and(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_and &ast_expr_op_and::operator=(ast_expr_op_and obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_and::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_and &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_and_then
ast_expr_op_and_then::ast_expr_op_and_then()
    : ast_expr_op() {}

ast_expr_op_and_then::ast_expr_op_and_then(const ast_expr_op_and_then &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_and_then::ast_expr_op_and_then(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_and_then &ast_expr_op_and_then::operator=(ast_expr_op_and_then obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_and_then::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_and_then &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_call
ast_expr_op_call::ast_expr_op_call()
    : ast_expr_op() {}

ast_expr_op_call::ast_expr_op_call(const ast_expr_op_call &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_call::ast_expr_op_call(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_call &ast_expr_op_call::operator=(ast_expr_op_call obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_call::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_call &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_cond
ast_expr_op_cond::ast_expr_op_cond()
    : ast_expr_op() {}

ast_expr_op_cond::ast_expr_op_cond(const ast_expr_op_cond &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_cond::ast_expr_op_cond(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_cond &ast_expr_op_cond::operator=(ast_expr_op_cond obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_cond::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_cond &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_div
ast_expr_op_div::ast_expr_op_div()
    : ast_expr_op() {}

ast_expr_op_div::ast_expr_op_div(const ast_expr_op_div &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_div::ast_expr_op_div(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_div &ast_expr_op_div::operator=(ast_expr_op_div obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_div::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_div &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_eq
ast_expr_op_eq::ast_expr_op_eq()
    : ast_expr_op() {}

ast_expr_op_eq::ast_expr_op_eq(const ast_expr_op_eq &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_eq::ast_expr_op_eq(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_eq &ast_expr_op_eq::operator=(ast_expr_op_eq obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_eq::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_eq &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_fdiv_q
ast_expr_op_fdiv_q::ast_expr_op_fdiv_q()
    : ast_expr_op() {}

ast_expr_op_fdiv_q::ast_expr_op_fdiv_q(const ast_expr_op_fdiv_q &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_fdiv_q::ast_expr_op_fdiv_q(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_fdiv_q &ast_expr_op_fdiv_q::operator=(ast_expr_op_fdiv_q obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_fdiv_q::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_fdiv_q &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_ge
ast_expr_op_ge::ast_expr_op_ge()
    : ast_expr_op() {}

ast_expr_op_ge::ast_expr_op_ge(const ast_expr_op_ge &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_ge::ast_expr_op_ge(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_ge &ast_expr_op_ge::operator=(ast_expr_op_ge obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_ge::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_ge &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_gt
ast_expr_op_gt::ast_expr_op_gt()
    : ast_expr_op() {}

ast_expr_op_gt::ast_expr_op_gt(const ast_expr_op_gt &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_gt::ast_expr_op_gt(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_gt &ast_expr_op_gt::operator=(ast_expr_op_gt obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_gt::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_gt &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_le
ast_expr_op_le::ast_expr_op_le()
    : ast_expr_op() {}

ast_expr_op_le::ast_expr_op_le(const ast_expr_op_le &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_le::ast_expr_op_le(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_le &ast_expr_op_le::operator=(ast_expr_op_le obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_le::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_le &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_lt
ast_expr_op_lt::ast_expr_op_lt()
    : ast_expr_op() {}

ast_expr_op_lt::ast_expr_op_lt(const ast_expr_op_lt &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_lt::ast_expr_op_lt(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_lt &ast_expr_op_lt::operator=(ast_expr_op_lt obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_lt::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_lt &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_max
ast_expr_op_max::ast_expr_op_max()
    : ast_expr_op() {}

ast_expr_op_max::ast_expr_op_max(const ast_expr_op_max &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_max::ast_expr_op_max(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_max &ast_expr_op_max::operator=(ast_expr_op_max obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_max::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_max &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_member
ast_expr_op_member::ast_expr_op_member()
    : ast_expr_op() {}

ast_expr_op_member::ast_expr_op_member(const ast_expr_op_member &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_member::ast_expr_op_member(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_member &ast_expr_op_member::operator=(ast_expr_op_member obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_member::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_member &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_min
ast_expr_op_min::ast_expr_op_min()
    : ast_expr_op() {}

ast_expr_op_min::ast_expr_op_min(const ast_expr_op_min &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_min::ast_expr_op_min(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_min &ast_expr_op_min::operator=(ast_expr_op_min obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_min::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_min &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_minus
ast_expr_op_minus::ast_expr_op_minus()
    : ast_expr_op() {}

ast_expr_op_minus::ast_expr_op_minus(const ast_expr_op_minus &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_minus::ast_expr_op_minus(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_minus &ast_expr_op_minus::operator=(ast_expr_op_minus obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_minus::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_minus &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_mul
ast_expr_op_mul::ast_expr_op_mul()
    : ast_expr_op() {}

ast_expr_op_mul::ast_expr_op_mul(const ast_expr_op_mul &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_mul::ast_expr_op_mul(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_mul &ast_expr_op_mul::operator=(ast_expr_op_mul obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_mul::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_mul &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_or
ast_expr_op_or::ast_expr_op_or()
    : ast_expr_op() {}

ast_expr_op_or::ast_expr_op_or(const ast_expr_op_or &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_or::ast_expr_op_or(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_or &ast_expr_op_or::operator=(ast_expr_op_or obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_or::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_or &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_or_else
ast_expr_op_or_else::ast_expr_op_or_else()
    : ast_expr_op() {}

ast_expr_op_or_else::ast_expr_op_or_else(const ast_expr_op_or_else &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_or_else::ast_expr_op_or_else(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_or_else &ast_expr_op_or_else::operator=(ast_expr_op_or_else obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_or_else::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_or_else &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_pdiv_q
ast_expr_op_pdiv_q::ast_expr_op_pdiv_q()
    : ast_expr_op() {}

ast_expr_op_pdiv_q::ast_expr_op_pdiv_q(const ast_expr_op_pdiv_q &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_pdiv_q::ast_expr_op_pdiv_q(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_pdiv_q &ast_expr_op_pdiv_q::operator=(ast_expr_op_pdiv_q obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_pdiv_q::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_pdiv_q &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_pdiv_r
ast_expr_op_pdiv_r::ast_expr_op_pdiv_r()
    : ast_expr_op() {}

ast_expr_op_pdiv_r::ast_expr_op_pdiv_r(const ast_expr_op_pdiv_r &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_pdiv_r::ast_expr_op_pdiv_r(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_pdiv_r &ast_expr_op_pdiv_r::operator=(ast_expr_op_pdiv_r obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_pdiv_r::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_pdiv_r &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_select
ast_expr_op_select::ast_expr_op_select()
    : ast_expr_op() {}

ast_expr_op_select::ast_expr_op_select(const ast_expr_op_select &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_select::ast_expr_op_select(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_select &ast_expr_op_select::operator=(ast_expr_op_select obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_select::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_select &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_sub
ast_expr_op_sub::ast_expr_op_sub()
    : ast_expr_op() {}

ast_expr_op_sub::ast_expr_op_sub(const ast_expr_op_sub &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_sub::ast_expr_op_sub(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_sub &ast_expr_op_sub::operator=(ast_expr_op_sub obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_sub::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_sub &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_expr_op_zdiv_r
ast_expr_op_zdiv_r::ast_expr_op_zdiv_r()
    : ast_expr_op() {}

ast_expr_op_zdiv_r::ast_expr_op_zdiv_r(const ast_expr_op_zdiv_r &obj)
    : ast_expr_op(obj)
{
}

ast_expr_op_zdiv_r::ast_expr_op_zdiv_r(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}

ast_expr_op_zdiv_r &ast_expr_op_zdiv_r::operator=(ast_expr_op_zdiv_r obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_expr_op_zdiv_r::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op_zdiv_r &obj)
{
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

template <typename T, typename>
boolean ast_node::isa_type(T subtype) const
{
  if (is_null())
    return boolean();
  return isl_ast_node_get_type(get()) == subtype;
}
template <class T>
boolean ast_node::isa() const
{
  return isa_type<decltype(T::type)>(T::type);
}
template <class T>
T ast_node::as() const
{
 if (isa<T>().is_false())
    isl_die(ctx().get(), isl_error_invalid, "not an object of the requested subtype", return T());
  return T(copy());
}

isl::ctx ast_node::ctx() const {
  return isl::ctx(isl_ast_node_get_ctx(ptr));
}

isl::id ast_node::annotation() const
{
  auto res = isl_ast_node_get_annotation(get());
  return manage(res);
}

isl::id ast_node::get_annotation() const
{
  return annotation();
}

std::string ast_node::to_C_str() const
{
  auto res = isl_ast_node_to_C_str(get());
  std::string tmp(res);
  free(res);
  return tmp;
}

isl::ast_node_list ast_node::to_list() const
{
  auto res = isl_ast_node_to_list(copy());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const ast_node &obj)
{
  char *str = isl_ast_node_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_node_block
ast_node_block::ast_node_block()
    : ast_node() {}

ast_node_block::ast_node_block(const ast_node_block &obj)
    : ast_node(obj)
{
}

ast_node_block::ast_node_block(__isl_take isl_ast_node *ptr)
    : ast_node(ptr) {}

ast_node_block &ast_node_block::operator=(ast_node_block obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_node_block::ctx() const {
  return isl::ctx(isl_ast_node_get_ctx(ptr));
}

isl::ast_node_list ast_node_block::children() const
{
  auto res = isl_ast_node_block_get_children(get());
  return manage(res);
}

isl::ast_node_list ast_node_block::get_children() const
{
  return children();
}

inline std::ostream &operator<<(std::ostream &os, const ast_node_block &obj)
{
  char *str = isl_ast_node_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_node_for
ast_node_for::ast_node_for()
    : ast_node() {}

ast_node_for::ast_node_for(const ast_node_for &obj)
    : ast_node(obj)
{
}

ast_node_for::ast_node_for(__isl_take isl_ast_node *ptr)
    : ast_node(ptr) {}

ast_node_for &ast_node_for::operator=(ast_node_for obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_node_for::ctx() const {
  return isl::ctx(isl_ast_node_get_ctx(ptr));
}

isl::ast_node ast_node_for::body() const
{
  auto res = isl_ast_node_for_get_body(get());
  return manage(res);
}

isl::ast_node ast_node_for::get_body() const
{
  return body();
}

isl::ast_expr ast_node_for::cond() const
{
  auto res = isl_ast_node_for_get_cond(get());
  return manage(res);
}

isl::ast_expr ast_node_for::get_cond() const
{
  return cond();
}

isl::ast_expr ast_node_for::inc() const
{
  auto res = isl_ast_node_for_get_inc(get());
  return manage(res);
}

isl::ast_expr ast_node_for::get_inc() const
{
  return inc();
}

isl::ast_expr ast_node_for::init() const
{
  auto res = isl_ast_node_for_get_init(get());
  return manage(res);
}

isl::ast_expr ast_node_for::get_init() const
{
  return init();
}

boolean ast_node_for::is_degenerate() const
{
  auto res = isl_ast_node_for_is_degenerate(get());
  return manage(res);
}

isl::ast_expr ast_node_for::iterator() const
{
  auto res = isl_ast_node_for_get_iterator(get());
  return manage(res);
}

isl::ast_expr ast_node_for::get_iterator() const
{
  return iterator();
}

inline std::ostream &operator<<(std::ostream &os, const ast_node_for &obj)
{
  char *str = isl_ast_node_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_node_if
ast_node_if::ast_node_if()
    : ast_node() {}

ast_node_if::ast_node_if(const ast_node_if &obj)
    : ast_node(obj)
{
}

ast_node_if::ast_node_if(__isl_take isl_ast_node *ptr)
    : ast_node(ptr) {}

ast_node_if &ast_node_if::operator=(ast_node_if obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_node_if::ctx() const {
  return isl::ctx(isl_ast_node_get_ctx(ptr));
}

isl::ast_expr ast_node_if::cond() const
{
  auto res = isl_ast_node_if_get_cond(get());
  return manage(res);
}

isl::ast_expr ast_node_if::get_cond() const
{
  return cond();
}

isl::ast_node ast_node_if::else_node() const
{
  auto res = isl_ast_node_if_get_else_node(get());
  return manage(res);
}

isl::ast_node ast_node_if::get_else_node() const
{
  return else_node();
}

boolean ast_node_if::has_else_node() const
{
  auto res = isl_ast_node_if_has_else_node(get());
  return manage(res);
}

isl::ast_node ast_node_if::then_node() const
{
  auto res = isl_ast_node_if_get_then_node(get());
  return manage(res);
}

isl::ast_node ast_node_if::get_then_node() const
{
  return then_node();
}

inline std::ostream &operator<<(std::ostream &os, const ast_node_if &obj)
{
  char *str = isl_ast_node_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

ast_node_list::ast_node_list(isl::ctx ctx, int n)
{
  auto res = isl_ast_node_list_alloc(ctx.release(), n);
  ptr = res;
}

ast_node_list::ast_node_list(isl::ast_node el)
{
  auto res = isl_ast_node_list_from_ast_node(el.release());
  ptr = res;
}

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

isl::ast_node_list ast_node_list::add(isl::ast_node el) const
{
  auto res = isl_ast_node_list_add(copy(), el.release());
  return manage(res);
}

isl::ast_node ast_node_list::at(int index) const
{
  auto res = isl_ast_node_list_get_at(get(), index);
  return manage(res);
}

isl::ast_node ast_node_list::get_at(int index) const
{
  return at(index);
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

stat ast_node_list::foreach(const std::function<stat(isl::ast_node)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::ast_node)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_ast_node *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_ast_node_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

isl::ast_node_list ast_node_list::insert(unsigned int pos, isl::ast_node el) const
{
  auto res = isl_ast_node_list_insert(copy(), pos, el.release());
  return manage(res);
}

class size ast_node_list::size() const
{
  auto res = isl_ast_node_list_size(get());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const ast_node_list &obj)
{
  char *str = isl_ast_node_list_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_node_mark
ast_node_mark::ast_node_mark()
    : ast_node() {}

ast_node_mark::ast_node_mark(const ast_node_mark &obj)
    : ast_node(obj)
{
}

ast_node_mark::ast_node_mark(__isl_take isl_ast_node *ptr)
    : ast_node(ptr) {}

ast_node_mark &ast_node_mark::operator=(ast_node_mark obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_node_mark::ctx() const {
  return isl::ctx(isl_ast_node_get_ctx(ptr));
}

isl::id ast_node_mark::id() const
{
  auto res = isl_ast_node_mark_get_id(get());
  return manage(res);
}

isl::id ast_node_mark::get_id() const
{
  return id();
}

isl::ast_node ast_node_mark::node() const
{
  auto res = isl_ast_node_mark_get_node(get());
  return manage(res);
}

isl::ast_node ast_node_mark::get_node() const
{
  return node();
}

inline std::ostream &operator<<(std::ostream &os, const ast_node_mark &obj)
{
  char *str = isl_ast_node_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_node_user
ast_node_user::ast_node_user()
    : ast_node() {}

ast_node_user::ast_node_user(const ast_node_user &obj)
    : ast_node(obj)
{
}

ast_node_user::ast_node_user(__isl_take isl_ast_node *ptr)
    : ast_node(ptr) {}

ast_node_user &ast_node_user::operator=(ast_node_user obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx ast_node_user::ctx() const {
  return isl::ctx(isl_ast_node_get_ctx(ptr));
}

isl::ast_expr ast_node_user::expr() const
{
  auto res = isl_ast_node_user_get_expr(get());
  return manage(res);
}

isl::ast_expr ast_node_user::get_expr() const
{
  return expr();
}

inline std::ostream &operator<<(std::ostream &os, const ast_node_user &obj)
{
  char *str = isl_ast_node_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::map basic_map::add_constraint(const isl::constraint &constraint) const
{
  return isl::map(*this).add_constraint(constraint);
}

isl::map basic_map::add_dims(isl::dim type, unsigned int n) const
{
  return isl::map(*this).add_dims(type, n);
}

isl::basic_map basic_map::affine_hull() const
{
  auto res = isl_basic_map_affine_hull(copy());
  return manage(res);
}

isl::map basic_map::align_params(const isl::space &model) const
{
  return isl::map(*this).align_params(model);
}

isl::basic_map basic_map::apply_domain(isl::basic_map bmap2) const
{
  auto res = isl_basic_map_apply_domain(copy(), bmap2.release());
  return manage(res);
}

isl::map basic_map::apply_domain(const isl::map &map2) const
{
  return isl::map(*this).apply_domain(map2);
}

isl::union_map basic_map::apply_domain(const isl::union_map &umap2) const
{
  return isl::map(*this).apply_domain(umap2);
}

isl::basic_map basic_map::apply_range(isl::basic_map bmap2) const
{
  auto res = isl_basic_map_apply_range(copy(), bmap2.release());
  return manage(res);
}

isl::map basic_map::apply_range(const isl::map &map2) const
{
  return isl::map(*this).apply_range(map2);
}

isl::union_map basic_map::apply_range(const isl::union_map &umap2) const
{
  return isl::map(*this).apply_range(umap2);
}

isl::map basic_map::as_map() const
{
  return isl::map(*this).as_map();
}

isl::multi_union_pw_aff basic_map::as_multi_union_pw_aff() const
{
  return isl::map(*this).as_multi_union_pw_aff();
}

isl::pw_multi_aff basic_map::as_pw_multi_aff() const
{
  return isl::map(*this).as_pw_multi_aff();
}

isl::union_pw_multi_aff basic_map::as_union_pw_multi_aff() const
{
  return isl::map(*this).as_union_pw_multi_aff();
}

isl::basic_map_list basic_map::basic_map_list() const
{
  return isl::map(*this).basic_map_list();
}

isl::set basic_map::bind_domain(const isl::multi_id &tuple) const
{
  return isl::map(*this).bind_domain(tuple);
}

isl::set basic_map::bind_range(const isl::multi_id &tuple) const
{
  return isl::map(*this).bind_range(tuple);
}

boolean basic_map::can_curry() const
{
  return isl::map(*this).can_curry();
}

isl::map basic_map::coalesce() const
{
  return isl::map(*this).coalesce();
}

isl::map basic_map::complement() const
{
  return isl::map(*this).complement();
}

isl::union_map basic_map::compute_divs() const
{
  return isl::map(*this).compute_divs();
}

isl::map basic_map::curry() const
{
  return isl::map(*this).curry();
}

isl::basic_set basic_map::deltas() const
{
  auto res = isl_basic_map_deltas(copy());
  return manage(res);
}

isl::basic_map basic_map::detect_equalities() const
{
  auto res = isl_basic_map_detect_equalities(copy());
  return manage(res);
}

class size basic_map::dim(isl::dim type) const
{
  return isl::map(*this).dim(type);
}

isl::pw_aff basic_map::dim_max(int pos) const
{
  return isl::map(*this).dim_max(pos);
}

isl::pw_aff basic_map::dim_min(int pos) const
{
  return isl::map(*this).dim_min(pos);
}

isl::basic_set basic_map::domain() const
{
  auto res = isl_basic_map_domain(copy());
  return manage(res);
}

isl::map basic_map::domain_factor_domain() const
{
  return isl::map(*this).domain_factor_domain();
}

isl::map basic_map::domain_factor_range() const
{
  return isl::map(*this).domain_factor_range();
}

isl::map basic_map::domain_map() const
{
  return isl::map(*this).domain_map();
}

isl::union_pw_multi_aff basic_map::domain_map_union_pw_multi_aff() const
{
  return isl::map(*this).domain_map_union_pw_multi_aff();
}

isl::map basic_map::domain_product(const isl::map &map2) const
{
  return isl::map(*this).domain_product(map2);
}

isl::union_map basic_map::domain_product(const isl::union_map &umap2) const
{
  return isl::map(*this).domain_product(umap2);
}

class size basic_map::domain_tuple_dim() const
{
  return isl::map(*this).domain_tuple_dim();
}

isl::id basic_map::domain_tuple_id() const
{
  return isl::map(*this).domain_tuple_id();
}

isl::map basic_map::eq_at(const isl::multi_pw_aff &mpa) const
{
  return isl::map(*this).eq_at(mpa);
}

isl::union_map basic_map::eq_at(const isl::multi_union_pw_aff &mupa) const
{
  return isl::map(*this).eq_at(mupa);
}

isl::basic_map basic_map::equal(isl::space space, unsigned int n_equal)
{
  auto res = isl_basic_map_equal(space.release(), n_equal);
  return manage(res);
}

isl::basic_map basic_map::equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_basic_map_equate(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

boolean basic_map::every_map(const std::function<boolean(isl::map)> &test) const
{
  return isl::map(*this).every_map(test);
}

isl::map basic_map::extract_map(const isl::space &space) const
{
  return isl::map(*this).extract_map(space);
}

isl::map basic_map::factor_domain() const
{
  return isl::map(*this).factor_domain();
}

isl::map basic_map::factor_range() const
{
  return isl::map(*this).factor_range();
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

isl::basic_map basic_map::fix_val(isl::dim type, unsigned int pos, long v) const
{
  return this->fix_val(type, pos, isl::val(ctx(), v));
}

isl::union_map basic_map::fixed_power(const isl::val &exp) const
{
  return isl::map(*this).fixed_power(exp);
}

isl::union_map basic_map::fixed_power(long exp) const
{
  return this->fixed_power(isl::val(ctx(), exp));
}

isl::map basic_map::flat_range_product(const isl::map &map2) const
{
  return isl::map(*this).flat_range_product(map2);
}

isl::union_map basic_map::flat_range_product(const isl::union_map &umap2) const
{
  return isl::map(*this).flat_range_product(umap2);
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

isl::map basic_map::floordiv_val(const isl::val &d) const
{
  return isl::map(*this).floordiv_val(d);
}

isl::map basic_map::floordiv_val(long d) const
{
  return this->floordiv_val(isl::val(ctx(), d));
}

stat basic_map::foreach_basic_map(const std::function<stat(isl::basic_map)> &fn) const
{
  return isl::map(*this).foreach_basic_map(fn);
}

stat basic_map::foreach_map(const std::function<stat(isl::map)> &fn) const
{
  return isl::map(*this).foreach_map(fn);
}

isl::basic_map basic_map::from_aff(isl::aff aff)
{
  auto res = isl_basic_map_from_aff(aff.release());
  return manage(res);
}

isl::basic_map basic_map::from_domain_and_range(isl::basic_set domain, isl::basic_set range)
{
  auto res = isl_basic_map_from_domain_and_range(domain.release(), range.release());
  return manage(res);
}

isl::basic_map basic_map::gist(isl::basic_map context) const
{
  auto res = isl_basic_map_gist(copy(), context.release());
  return manage(res);
}

isl::map basic_map::gist(const isl::map &context) const
{
  return isl::map(*this).gist(context);
}

isl::union_map basic_map::gist(const isl::union_map &context) const
{
  return isl::map(*this).gist(context);
}

isl::map basic_map::gist_domain(const isl::set &context) const
{
  return isl::map(*this).gist_domain(context);
}

isl::union_map basic_map::gist_domain(const isl::union_set &uset) const
{
  return isl::map(*this).gist_domain(uset);
}

isl::map basic_map::gist_params(const isl::set &context) const
{
  return isl::map(*this).gist_params(context);
}

isl::union_map basic_map::gist_range(const isl::union_set &uset) const
{
  return isl::map(*this).gist_range(uset);
}

boolean basic_map::has_domain_tuple_id() const
{
  return isl::map(*this).has_domain_tuple_id();
}

boolean basic_map::has_equal_space(const isl::map &map2) const
{
  return isl::map(*this).has_equal_space(map2);
}

boolean basic_map::has_range_tuple_id() const
{
  return isl::map(*this).has_range_tuple_id();
}

boolean basic_map::has_tuple_id(isl::dim type) const
{
  return isl::map(*this).has_tuple_id(type);
}

boolean basic_map::has_tuple_name(isl::dim type) const
{
  return isl::map(*this).has_tuple_name(type);
}

isl::basic_map basic_map::intersect(isl::basic_map bmap2) const
{
  auto res = isl_basic_map_intersect(copy(), bmap2.release());
  return manage(res);
}

isl::map basic_map::intersect(const isl::map &map2) const
{
  return isl::map(*this).intersect(map2);
}

isl::union_map basic_map::intersect(const isl::union_map &umap2) const
{
  return isl::map(*this).intersect(umap2);
}

isl::basic_map basic_map::intersect_domain(isl::basic_set bset) const
{
  auto res = isl_basic_map_intersect_domain(copy(), bset.release());
  return manage(res);
}

isl::map basic_map::intersect_domain(const isl::set &set) const
{
  return isl::map(*this).intersect_domain(set);
}

isl::union_map basic_map::intersect_domain(const isl::space &space) const
{
  return isl::map(*this).intersect_domain(space);
}

isl::union_map basic_map::intersect_domain(const isl::union_set &uset) const
{
  return isl::map(*this).intersect_domain(uset);
}

isl::basic_map basic_map::intersect_domain(const isl::point &bset) const
{
  return this->intersect_domain(isl::basic_set(bset));
}

isl::map basic_map::intersect_domain_factor_domain(const isl::map &factor) const
{
  return isl::map(*this).intersect_domain_factor_domain(factor);
}

isl::union_map basic_map::intersect_domain_factor_domain(const isl::union_map &factor) const
{
  return isl::map(*this).intersect_domain_factor_domain(factor);
}

isl::map basic_map::intersect_domain_factor_range(const isl::map &factor) const
{
  return isl::map(*this).intersect_domain_factor_range(factor);
}

isl::union_map basic_map::intersect_domain_factor_range(const isl::union_map &factor) const
{
  return isl::map(*this).intersect_domain_factor_range(factor);
}

isl::map basic_map::intersect_params(const isl::set &params) const
{
  return isl::map(*this).intersect_params(params);
}

isl::basic_map basic_map::intersect_range(isl::basic_set bset) const
{
  auto res = isl_basic_map_intersect_range(copy(), bset.release());
  return manage(res);
}

isl::map basic_map::intersect_range(const isl::set &set) const
{
  return isl::map(*this).intersect_range(set);
}

isl::union_map basic_map::intersect_range(const isl::space &space) const
{
  return isl::map(*this).intersect_range(space);
}

isl::union_map basic_map::intersect_range(const isl::union_set &uset) const
{
  return isl::map(*this).intersect_range(uset);
}

isl::basic_map basic_map::intersect_range(const isl::point &bset) const
{
  return this->intersect_range(isl::basic_set(bset));
}

isl::map basic_map::intersect_range_factor_domain(const isl::map &factor) const
{
  return isl::map(*this).intersect_range_factor_domain(factor);
}

isl::union_map basic_map::intersect_range_factor_domain(const isl::union_map &factor) const
{
  return isl::map(*this).intersect_range_factor_domain(factor);
}

isl::map basic_map::intersect_range_factor_range(const isl::map &factor) const
{
  return isl::map(*this).intersect_range_factor_range(factor);
}

isl::union_map basic_map::intersect_range_factor_range(const isl::union_map &factor) const
{
  return isl::map(*this).intersect_range_factor_range(factor);
}

boolean basic_map::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  return isl::map(*this).involves_dims(type, first, n);
}

boolean basic_map::is_bijective() const
{
  return isl::map(*this).is_bijective();
}

boolean basic_map::is_disjoint(const isl::map &map2) const
{
  return isl::map(*this).is_disjoint(map2);
}

boolean basic_map::is_disjoint(const isl::union_map &umap2) const
{
  return isl::map(*this).is_disjoint(umap2);
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

boolean basic_map::is_equal(const isl::map &map2) const
{
  return isl::map(*this).is_equal(map2);
}

boolean basic_map::is_equal(const isl::union_map &umap2) const
{
  return isl::map(*this).is_equal(umap2);
}

boolean basic_map::is_injective() const
{
  return isl::map(*this).is_injective();
}

boolean basic_map::is_single_valued() const
{
  return isl::map(*this).is_single_valued();
}

boolean basic_map::is_strict_subset(const isl::map &map2) const
{
  return isl::map(*this).is_strict_subset(map2);
}

boolean basic_map::is_strict_subset(const isl::union_map &umap2) const
{
  return isl::map(*this).is_strict_subset(umap2);
}

boolean basic_map::is_subset(const isl::basic_map &bmap2) const
{
  auto res = isl_basic_map_is_subset(get(), bmap2.get());
  return manage(res);
}

boolean basic_map::is_subset(const isl::map &map2) const
{
  return isl::map(*this).is_subset(map2);
}

boolean basic_map::is_subset(const isl::union_map &umap2) const
{
  return isl::map(*this).is_subset(umap2);
}

boolean basic_map::isa_map() const
{
  return isl::map(*this).isa_map();
}

isl::map basic_map::lex_ge_at(const isl::multi_pw_aff &mpa) const
{
  return isl::map(*this).lex_ge_at(mpa);
}

isl::map basic_map::lex_gt_at(const isl::multi_pw_aff &mpa) const
{
  return isl::map(*this).lex_gt_at(mpa);
}

isl::map basic_map::lex_le_at(const isl::multi_pw_aff &mpa) const
{
  return isl::map(*this).lex_le_at(mpa);
}

isl::map basic_map::lex_lt_at(const isl::multi_pw_aff &mpa) const
{
  return isl::map(*this).lex_lt_at(mpa);
}

isl::map basic_map::lexmax() const
{
  auto res = isl_basic_map_lexmax(copy());
  return manage(res);
}

isl::pw_multi_aff basic_map::lexmax_pw_multi_aff() const
{
  return isl::map(*this).lexmax_pw_multi_aff();
}

isl::map basic_map::lexmin() const
{
  auto res = isl_basic_map_lexmin(copy());
  return manage(res);
}

isl::pw_multi_aff basic_map::lexmin_pw_multi_aff() const
{
  return isl::map(*this).lexmin_pw_multi_aff();
}

isl::map basic_map::lower_bound(const isl::multi_pw_aff &lower) const
{
  return isl::map(*this).lower_bound(lower);
}

isl::map basic_map::lower_bound_si(isl::dim type, unsigned int pos, int value) const
{
  return isl::map(*this).lower_bound_si(type, pos, value);
}

isl::map_list basic_map::map_list() const
{
  return isl::map(*this).map_list();
}

isl::multi_pw_aff basic_map::max_multi_pw_aff() const
{
  return isl::map(*this).max_multi_pw_aff();
}

isl::multi_pw_aff basic_map::min_multi_pw_aff() const
{
  return isl::map(*this).min_multi_pw_aff();
}

isl::map basic_map::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const
{
  return isl::map(*this).move_dims(dst_type, dst_pos, src_type, src_pos, n);
}

class size basic_map::n_basic_map() const
{
  return isl::map(*this).n_basic_map();
}

isl::map basic_map::order_lt(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  return isl::map(*this).order_lt(type1, pos1, type2, pos2);
}

isl::set basic_map::params() const
{
  return isl::map(*this).params();
}

isl::val basic_map::plain_get_val_if_fixed(isl::dim type, unsigned int pos) const
{
  auto res = isl_basic_map_plain_get_val_if_fixed(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::basic_map basic_map::polyhedral_hull() const
{
  return isl::map(*this).polyhedral_hull();
}

isl::map basic_map::preimage_domain(const isl::multi_aff &ma) const
{
  return isl::map(*this).preimage_domain(ma);
}

isl::map basic_map::preimage_domain(const isl::multi_pw_aff &mpa) const
{
  return isl::map(*this).preimage_domain(mpa);
}

isl::map basic_map::preimage_domain(const isl::pw_multi_aff &pma) const
{
  return isl::map(*this).preimage_domain(pma);
}

isl::union_map basic_map::preimage_domain(const isl::union_pw_multi_aff &upma) const
{
  return isl::map(*this).preimage_domain(upma);
}

isl::map basic_map::preimage_range(const isl::multi_aff &ma) const
{
  return isl::map(*this).preimage_range(ma);
}

isl::map basic_map::preimage_range(const isl::pw_multi_aff &pma) const
{
  return isl::map(*this).preimage_range(pma);
}

isl::union_map basic_map::preimage_range(const isl::union_pw_multi_aff &upma) const
{
  return isl::map(*this).preimage_range(upma);
}

isl::map basic_map::product(const isl::map &map2) const
{
  return isl::map(*this).product(map2);
}

isl::union_map basic_map::product(const isl::union_map &umap2) const
{
  return isl::map(*this).product(umap2);
}

isl::map basic_map::project_out(isl::dim type, unsigned int first, unsigned int n) const
{
  return isl::map(*this).project_out(type, first, n);
}

isl::map basic_map::project_out_all_params() const
{
  return isl::map(*this).project_out_all_params();
}

isl::set basic_map::range() const
{
  return isl::map(*this).range();
}

isl::map basic_map::range_factor_domain() const
{
  return isl::map(*this).range_factor_domain();
}

isl::map basic_map::range_factor_range() const
{
  return isl::map(*this).range_factor_range();
}

isl::fixed_box basic_map::range_lattice_tile() const
{
  return isl::map(*this).range_lattice_tile();
}

isl::map basic_map::range_map() const
{
  return isl::map(*this).range_map();
}

isl::map basic_map::range_product(const isl::map &map2) const
{
  return isl::map(*this).range_product(map2);
}

isl::union_map basic_map::range_product(const isl::union_map &umap2) const
{
  return isl::map(*this).range_product(umap2);
}

isl::map basic_map::range_reverse() const
{
  return isl::map(*this).range_reverse();
}

isl::fixed_box basic_map::range_simple_fixed_box_hull() const
{
  return isl::map(*this).range_simple_fixed_box_hull();
}

class size basic_map::range_tuple_dim() const
{
  return isl::map(*this).range_tuple_dim();
}

isl::id basic_map::range_tuple_id() const
{
  return isl::map(*this).range_tuple_id();
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

isl::map basic_map::set_domain_tuple(const isl::id &id) const
{
  return isl::map(*this).set_domain_tuple(id);
}

isl::map basic_map::set_domain_tuple(const std::string &id) const
{
  return this->set_domain_tuple(isl::id(ctx(), id));
}

isl::map basic_map::set_range_tuple(const isl::id &id) const
{
  return isl::map(*this).set_range_tuple(id);
}

isl::map basic_map::set_range_tuple(const std::string &id) const
{
  return this->set_range_tuple(isl::id(ctx(), id));
}

isl::map basic_map::set_tuple_id(isl::dim type, const isl::id &id) const
{
  return isl::map(*this).set_tuple_id(type, id);
}

isl::map basic_map::set_tuple_id(isl::dim type, const std::string &id) const
{
  return this->set_tuple_id(type, isl::id(ctx(), id));
}

isl::space basic_map::space() const
{
  return isl::map(*this).space();
}

isl::map basic_map::subtract(const isl::map &map2) const
{
  return isl::map(*this).subtract(map2);
}

isl::union_map basic_map::subtract(const isl::union_map &umap2) const
{
  return isl::map(*this).subtract(umap2);
}

isl::union_map basic_map::subtract_domain(const isl::union_set &dom) const
{
  return isl::map(*this).subtract_domain(dom);
}

isl::union_map basic_map::subtract_range(const isl::union_set &dom) const
{
  return isl::map(*this).subtract_range(dom);
}

isl::map basic_map::sum(const isl::map &map2) const
{
  return isl::map(*this).sum(map2);
}

isl::basic_map_list basic_map::to_list() const
{
  auto res = isl_basic_map_to_list(copy());
  return manage(res);
}

isl::union_map basic_map::to_union_map() const
{
  return isl::map(*this).to_union_map();
}

isl::id basic_map::tuple_id(isl::dim type) const
{
  return isl::map(*this).tuple_id(type);
}

isl::map basic_map::uncurry() const
{
  return isl::map(*this).uncurry();
}

isl::map basic_map::unite(isl::basic_map bmap2) const
{
  auto res = isl_basic_map_union(copy(), bmap2.release());
  return manage(res);
}

isl::map basic_map::unite(const isl::map &map2) const
{
  return isl::map(*this).unite(map2);
}

isl::union_map basic_map::unite(const isl::union_map &umap2) const
{
  return isl::map(*this).unite(umap2);
}

isl::basic_map basic_map::universe(isl::space space)
{
  auto res = isl_basic_map_universe(space.release());
  return manage(res);
}

isl::basic_map basic_map::unshifted_simple_hull() const
{
  return isl::map(*this).unshifted_simple_hull();
}

isl::map basic_map::upper_bound(const isl::multi_pw_aff &upper) const
{
  return isl::map(*this).upper_bound(upper);
}

isl::map basic_map::upper_bound_si(isl::dim type, unsigned int pos, int value) const
{
  return isl::map(*this).upper_bound_si(type, pos, value);
}

isl::set basic_map::wrap() const
{
  return isl::map(*this).wrap();
}

isl::map basic_map::zip() const
{
  return isl::map(*this).zip();
}

inline std::ostream &operator<<(std::ostream &os, const basic_map &obj)
{
  char *str = isl_basic_map_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

basic_map_list::basic_map_list(isl::ctx ctx, int n)
{
  auto res = isl_basic_map_list_alloc(ctx.release(), n);
  ptr = res;
}

basic_map_list::basic_map_list(isl::basic_map el)
{
  auto res = isl_basic_map_list_from_basic_map(el.release());
  ptr = res;
}

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

isl::basic_map_list basic_map_list::add(isl::basic_map el) const
{
  auto res = isl_basic_map_list_add(copy(), el.release());
  return manage(res);
}

isl::basic_map basic_map_list::at(int index) const
{
  auto res = isl_basic_map_list_get_at(get(), index);
  return manage(res);
}

isl::basic_map basic_map_list::get_at(int index) const
{
  return at(index);
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

stat basic_map_list::foreach(const std::function<stat(isl::basic_map)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::basic_map)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_basic_map *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_basic_map_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

isl::basic_map_list basic_map_list::insert(unsigned int pos, isl::basic_map el) const
{
  auto res = isl_basic_map_list_insert(copy(), pos, el.release());
  return manage(res);
}

class size basic_map_list::size() const
{
  auto res = isl_basic_map_list_size(get());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const basic_map_list &obj)
{
  char *str = isl_basic_map_list_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::set basic_set::add_constraint(const isl::constraint &constraint) const
{
  return isl::set(*this).add_constraint(constraint);
}

isl::set basic_set::add_dims(isl::dim type, unsigned int n) const
{
  return isl::set(*this).add_dims(type, n);
}

isl::basic_set basic_set::affine_hull() const
{
  auto res = isl_basic_set_affine_hull(copy());
  return manage(res);
}

isl::set basic_set::align_params(const isl::space &model) const
{
  return isl::set(*this).align_params(model);
}

isl::basic_set basic_set::apply(isl::basic_map bmap) const
{
  auto res = isl_basic_set_apply(copy(), bmap.release());
  return manage(res);
}

isl::set basic_set::apply(const isl::map &map) const
{
  return isl::set(*this).apply(map);
}

isl::union_set basic_set::apply(const isl::union_map &umap) const
{
  return isl::set(*this).apply(umap);
}

isl::pw_multi_aff basic_set::as_pw_multi_aff() const
{
  return isl::set(*this).as_pw_multi_aff();
}

isl::set basic_set::as_set() const
{
  return isl::set(*this).as_set();
}

isl::basic_set_list basic_set::basic_set_list() const
{
  return isl::set(*this).basic_set_list();
}

isl::set basic_set::bind(const isl::multi_id &tuple) const
{
  return isl::set(*this).bind(tuple);
}

isl::set basic_set::coalesce() const
{
  return isl::set(*this).coalesce();
}

isl::set basic_set::complement() const
{
  return isl::set(*this).complement();
}

isl::union_set basic_set::compute_divs() const
{
  return isl::set(*this).compute_divs();
}

boolean basic_set::contains(const isl::space &space) const
{
  return isl::set(*this).contains(space);
}

isl::basic_set basic_set::convex_hull() const
{
  return isl::set(*this).convex_hull();
}

isl::basic_set basic_set::detect_equalities() const
{
  auto res = isl_basic_set_detect_equalities(copy());
  return manage(res);
}

class size basic_set::dim(isl::dim type) const
{
  auto res = isl_basic_set_dim(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

boolean basic_set::dim_has_any_lower_bound(isl::dim type, unsigned int pos) const
{
  return isl::set(*this).dim_has_any_lower_bound(type, pos);
}

isl::id basic_set::dim_id(isl::dim type, unsigned int pos) const
{
  return isl::set(*this).dim_id(type, pos);
}

isl::pw_aff basic_set::dim_max(int pos) const
{
  return isl::set(*this).dim_max(pos);
}

isl::val basic_set::dim_max_val(int pos) const
{
  auto res = isl_basic_set_dim_max_val(copy(), pos);
  return manage(res);
}

isl::pw_aff basic_set::dim_min(int pos) const
{
  return isl::set(*this).dim_min(pos);
}

isl::val basic_set::dim_min_val(int pos) const
{
  return isl::set(*this).dim_min_val(pos);
}

std::string basic_set::dim_name(isl::dim type, unsigned int pos) const
{
  return isl::set(*this).dim_name(type, pos);
}

isl::aff basic_set::div(int pos) const
{
  auto res = isl_basic_set_get_div(get(), pos);
  return manage(res);
}

isl::aff basic_set::get_div(int pos) const
{
  return div(pos);
}

isl::set basic_set::drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  return isl::set(*this).drop_constraints_involving_dims(type, first, n);
}

isl::set basic_set::eliminate(isl::dim type, unsigned int first, unsigned int n) const
{
  return isl::set(*this).eliminate(type, first, n);
}

boolean basic_set::every_set(const std::function<boolean(isl::set)> &test) const
{
  return isl::set(*this).every_set(test);
}

isl::set basic_set::extract_set(const isl::space &space) const
{
  return isl::set(*this).extract_set(space);
}

int basic_set::find_dim_by_id(isl::dim type, const isl::id &id) const
{
  return isl::set(*this).find_dim_by_id(type, id);
}

int basic_set::find_dim_by_id(isl::dim type, const std::string &id) const
{
  return this->find_dim_by_id(type, isl::id(ctx(), id));
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

isl::basic_set basic_set::fix_val(isl::dim type, unsigned int pos, long v) const
{
  return this->fix_val(type, pos, isl::val(ctx(), v));
}

isl::basic_set basic_set::flatten() const
{
  auto res = isl_basic_set_flatten(copy());
  return manage(res);
}

stat basic_set::foreach_basic_set(const std::function<stat(isl::basic_set)> &fn) const
{
  return isl::set(*this).foreach_basic_set(fn);
}

stat basic_set::foreach_point(const std::function<stat(isl::point)> &fn) const
{
  return isl::set(*this).foreach_point(fn);
}

stat basic_set::foreach_set(const std::function<stat(isl::set)> &fn) const
{
  return isl::set(*this).foreach_set(fn);
}

isl::basic_set basic_set::gist(isl::basic_set context) const
{
  auto res = isl_basic_set_gist(copy(), context.release());
  return manage(res);
}

isl::set basic_set::gist(const isl::set &context) const
{
  return isl::set(*this).gist(context);
}

isl::union_set basic_set::gist(const isl::union_set &context) const
{
  return isl::set(*this).gist(context);
}

isl::basic_set basic_set::gist(const isl::point &context) const
{
  return this->gist(isl::basic_set(context));
}

isl::set basic_set::gist_params(const isl::set &context) const
{
  return isl::set(*this).gist_params(context);
}

boolean basic_set::has_equal_space(const isl::set &set2) const
{
  return isl::set(*this).has_equal_space(set2);
}

isl::map basic_set::identity() const
{
  return isl::set(*this).identity();
}

isl::union_pw_multi_aff basic_set::identity_union_pw_multi_aff() const
{
  return isl::set(*this).identity_union_pw_multi_aff();
}

isl::pw_aff basic_set::indicator_function() const
{
  return isl::set(*this).indicator_function();
}

isl::set basic_set::insert_dims(isl::dim type, unsigned int pos, unsigned int n) const
{
  return isl::set(*this).insert_dims(type, pos, n);
}

isl::map basic_set::insert_domain(const isl::space &domain) const
{
  return isl::set(*this).insert_domain(domain);
}

isl::basic_set basic_set::intersect(isl::basic_set bset2) const
{
  auto res = isl_basic_set_intersect(copy(), bset2.release());
  return manage(res);
}

isl::set basic_set::intersect(const isl::set &set2) const
{
  return isl::set(*this).intersect(set2);
}

isl::union_set basic_set::intersect(const isl::union_set &uset2) const
{
  return isl::set(*this).intersect(uset2);
}

isl::basic_set basic_set::intersect(const isl::point &bset2) const
{
  return this->intersect(isl::basic_set(bset2));
}

isl::basic_set basic_set::intersect_params(isl::basic_set bset2) const
{
  auto res = isl_basic_set_intersect_params(copy(), bset2.release());
  return manage(res);
}

isl::set basic_set::intersect_params(const isl::set &params) const
{
  return isl::set(*this).intersect_params(params);
}

isl::basic_set basic_set::intersect_params(const isl::point &bset2) const
{
  return this->intersect_params(isl::basic_set(bset2));
}

boolean basic_set::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  return isl::set(*this).involves_dims(type, first, n);
}

boolean basic_set::involves_locals() const
{
  return isl::set(*this).involves_locals();
}

boolean basic_set::is_bounded() const
{
  auto res = isl_basic_set_is_bounded(get());
  return manage(res);
}

boolean basic_set::is_disjoint(const isl::set &set2) const
{
  return isl::set(*this).is_disjoint(set2);
}

boolean basic_set::is_disjoint(const isl::union_set &uset2) const
{
  return isl::set(*this).is_disjoint(uset2);
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

boolean basic_set::is_equal(const isl::set &set2) const
{
  return isl::set(*this).is_equal(set2);
}

boolean basic_set::is_equal(const isl::union_set &uset2) const
{
  return isl::set(*this).is_equal(uset2);
}

boolean basic_set::is_equal(const isl::point &bset2) const
{
  return this->is_equal(isl::basic_set(bset2));
}

boolean basic_set::is_params() const
{
  return isl::set(*this).is_params();
}

boolean basic_set::is_singleton() const
{
  return isl::set(*this).is_singleton();
}

boolean basic_set::is_strict_subset(const isl::set &set2) const
{
  return isl::set(*this).is_strict_subset(set2);
}

boolean basic_set::is_strict_subset(const isl::union_set &uset2) const
{
  return isl::set(*this).is_strict_subset(uset2);
}

boolean basic_set::is_subset(const isl::basic_set &bset2) const
{
  auto res = isl_basic_set_is_subset(get(), bset2.get());
  return manage(res);
}

boolean basic_set::is_subset(const isl::set &set2) const
{
  return isl::set(*this).is_subset(set2);
}

boolean basic_set::is_subset(const isl::union_set &uset2) const
{
  return isl::set(*this).is_subset(uset2);
}

boolean basic_set::is_subset(const isl::point &bset2) const
{
  return this->is_subset(isl::basic_set(bset2));
}

boolean basic_set::is_wrapping() const
{
  auto res = isl_basic_set_is_wrapping(get());
  return manage(res);
}

boolean basic_set::isa_set() const
{
  return isl::set(*this).isa_set();
}

isl::set basic_set::lexmax() const
{
  auto res = isl_basic_set_lexmax(copy());
  return manage(res);
}

isl::pw_multi_aff basic_set::lexmax_pw_multi_aff() const
{
  return isl::set(*this).lexmax_pw_multi_aff();
}

isl::set basic_set::lexmin() const
{
  auto res = isl_basic_set_lexmin(copy());
  return manage(res);
}

isl::pw_multi_aff basic_set::lexmin_pw_multi_aff() const
{
  return isl::set(*this).lexmin_pw_multi_aff();
}

isl::set basic_set::lower_bound(const isl::multi_pw_aff &lower) const
{
  return isl::set(*this).lower_bound(lower);
}

isl::set basic_set::lower_bound(const isl::multi_val &lower) const
{
  return isl::set(*this).lower_bound(lower);
}

isl::set basic_set::lower_bound_si(isl::dim type, unsigned int pos, int value) const
{
  return isl::set(*this).lower_bound_si(type, pos, value);
}

isl::set basic_set::lower_bound_val(isl::dim type, unsigned int pos, const isl::val &value) const
{
  return isl::set(*this).lower_bound_val(type, pos, value);
}

isl::set basic_set::lower_bound_val(isl::dim type, unsigned int pos, long value) const
{
  return this->lower_bound_val(type, pos, isl::val(ctx(), value));
}

isl::multi_pw_aff basic_set::max_multi_pw_aff() const
{
  return isl::set(*this).max_multi_pw_aff();
}

isl::val basic_set::max_val(const isl::aff &obj) const
{
  return isl::set(*this).max_val(obj);
}

isl::multi_pw_aff basic_set::min_multi_pw_aff() const
{
  return isl::set(*this).min_multi_pw_aff();
}

isl::val basic_set::min_val(const isl::aff &obj) const
{
  return isl::set(*this).min_val(obj);
}

class size basic_set::n_basic_set() const
{
  return isl::set(*this).n_basic_set();
}

isl::basic_set basic_set::params() const
{
  auto res = isl_basic_set_params(copy());
  return manage(res);
}

isl::val basic_set::plain_get_val_if_fixed(isl::dim type, unsigned int pos) const
{
  return isl::set(*this).plain_get_val_if_fixed(type, pos);
}

isl::multi_val basic_set::plain_multi_val_if_fixed() const
{
  return isl::set(*this).plain_multi_val_if_fixed();
}

isl::basic_set basic_set::polyhedral_hull() const
{
  return isl::set(*this).polyhedral_hull();
}

isl::set basic_set::preimage(const isl::multi_aff &ma) const
{
  return isl::set(*this).preimage(ma);
}

isl::set basic_set::preimage(const isl::multi_pw_aff &mpa) const
{
  return isl::set(*this).preimage(mpa);
}

isl::set basic_set::preimage(const isl::pw_multi_aff &pma) const
{
  return isl::set(*this).preimage(pma);
}

isl::union_set basic_set::preimage(const isl::union_pw_multi_aff &upma) const
{
  return isl::set(*this).preimage(upma);
}

isl::set basic_set::product(const isl::set &set2) const
{
  return isl::set(*this).product(set2);
}

isl::basic_set basic_set::project_out(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_basic_set_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::set basic_set::project_out_all_params() const
{
  return isl::set(*this).project_out_all_params();
}

isl::set basic_set::project_out_param(const isl::id &id) const
{
  return isl::set(*this).project_out_param(id);
}

isl::set basic_set::project_out_param(const std::string &id) const
{
  return this->project_out_param(isl::id(ctx(), id));
}

isl::set basic_set::project_out_param(const isl::id_list &list) const
{
  return isl::set(*this).project_out_param(list);
}

isl::pw_multi_aff basic_set::pw_multi_aff_on_domain(const isl::multi_val &mv) const
{
  return isl::set(*this).pw_multi_aff_on_domain(mv);
}

isl::set basic_set::remove_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  return isl::set(*this).remove_dims(type, first, n);
}

isl::set basic_set::remove_divs() const
{
  return isl::set(*this).remove_divs();
}

isl::set basic_set::remove_redundancies() const
{
  return isl::set(*this).remove_redundancies();
}

isl::set basic_set::reset_tuple_id() const
{
  return isl::set(*this).reset_tuple_id();
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

isl::set basic_set::set_dim_id(isl::dim type, unsigned int pos, const isl::id &id) const
{
  return isl::set(*this).set_dim_id(type, pos, id);
}

isl::set basic_set::set_dim_id(isl::dim type, unsigned int pos, const std::string &id) const
{
  return this->set_dim_id(type, pos, isl::id(ctx(), id));
}

isl::set_list basic_set::set_list() const
{
  return isl::set(*this).set_list();
}

isl::set basic_set::set_tuple_id(const isl::id &id) const
{
  return isl::set(*this).set_tuple_id(id);
}

isl::set basic_set::set_tuple_id(const std::string &id) const
{
  return this->set_tuple_id(isl::id(ctx(), id));
}

isl::fixed_box basic_set::simple_fixed_box_hull() const
{
  return isl::set(*this).simple_fixed_box_hull();
}

isl::basic_set basic_set::simple_hull() const
{
  return isl::set(*this).simple_hull();
}

isl::space basic_set::space() const
{
  auto res = isl_basic_set_get_space(get());
  return manage(res);
}

isl::space basic_set::get_space() const
{
  return space();
}

isl::val basic_set::stride(int pos) const
{
  return isl::set(*this).stride(pos);
}

isl::set basic_set::subtract(const isl::set &set2) const
{
  return isl::set(*this).subtract(set2);
}

isl::union_set basic_set::subtract(const isl::union_set &uset2) const
{
  return isl::set(*this).subtract(uset2);
}

isl::basic_set_list basic_set::to_list() const
{
  auto res = isl_basic_set_to_list(copy());
  return manage(res);
}

isl::set basic_set::to_set() const
{
  auto res = isl_basic_set_to_set(copy());
  return manage(res);
}

isl::union_set basic_set::to_union_set() const
{
  return isl::set(*this).to_union_set();
}

isl::map basic_set::translation() const
{
  return isl::set(*this).translation();
}

class size basic_set::tuple_dim() const
{
  return isl::set(*this).tuple_dim();
}

isl::id basic_set::tuple_id() const
{
  return isl::set(*this).tuple_id();
}

std::string basic_set::tuple_name() const
{
  return isl::set(*this).tuple_name();
}

isl::set basic_set::unbind_params(const isl::multi_id &tuple) const
{
  return isl::set(*this).unbind_params(tuple);
}

isl::map basic_set::unbind_params_insert_domain(const isl::multi_id &domain) const
{
  return isl::set(*this).unbind_params_insert_domain(domain);
}

isl::set basic_set::unite(isl::basic_set bset2) const
{
  auto res = isl_basic_set_union(copy(), bset2.release());
  return manage(res);
}

isl::set basic_set::unite(const isl::set &set2) const
{
  return isl::set(*this).unite(set2);
}

isl::union_set basic_set::unite(const isl::union_set &uset2) const
{
  return isl::set(*this).unite(uset2);
}

isl::set basic_set::unite(const isl::point &bset2) const
{
  return this->unite(isl::basic_set(bset2));
}

isl::basic_set basic_set::universe(isl::space space)
{
  auto res = isl_basic_set_universe(space.release());
  return manage(res);
}

isl::basic_set basic_set::unshifted_simple_hull() const
{
  return isl::set(*this).unshifted_simple_hull();
}

isl::map basic_set::unwrap() const
{
  return isl::set(*this).unwrap();
}

isl::set basic_set::upper_bound(const isl::multi_pw_aff &upper) const
{
  return isl::set(*this).upper_bound(upper);
}

isl::set basic_set::upper_bound(const isl::multi_val &upper) const
{
  return isl::set(*this).upper_bound(upper);
}

isl::set basic_set::upper_bound_val(isl::dim type, unsigned int pos, const isl::val &value) const
{
  return isl::set(*this).upper_bound_val(type, pos, value);
}

isl::set basic_set::upper_bound_val(isl::dim type, unsigned int pos, long value) const
{
  return this->upper_bound_val(type, pos, isl::val(ctx(), value));
}

inline std::ostream &operator<<(std::ostream &os, const basic_set &obj)
{
  char *str = isl_basic_set_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

basic_set_list::basic_set_list(isl::ctx ctx, int n)
{
  auto res = isl_basic_set_list_alloc(ctx.release(), n);
  ptr = res;
}

basic_set_list::basic_set_list(isl::basic_set el)
{
  auto res = isl_basic_set_list_from_basic_set(el.release());
  ptr = res;
}

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

isl::basic_set_list basic_set_list::add(isl::basic_set el) const
{
  auto res = isl_basic_set_list_add(copy(), el.release());
  return manage(res);
}

isl::basic_set basic_set_list::at(int index) const
{
  auto res = isl_basic_set_list_get_at(get(), index);
  return manage(res);
}

isl::basic_set basic_set_list::get_at(int index) const
{
  return at(index);
}

isl::basic_set_list basic_set_list::clear() const
{
  auto res = isl_basic_set_list_clear(copy());
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

stat basic_set_list::foreach(const std::function<stat(isl::basic_set)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::basic_set)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_basic_set *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_basic_set_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

isl::basic_set_list basic_set_list::insert(unsigned int pos, isl::basic_set el) const
{
  auto res = isl_basic_set_list_insert(copy(), pos, el.release());
  return manage(res);
}

class size basic_set_list::size() const
{
  auto res = isl_basic_set_list_size(get());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const basic_set_list &obj)
{
  char *str = isl_basic_set_list_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::constraint constraint::set_coefficient_si(isl::dim type, int pos, int v) const
{
  auto res = isl_constraint_set_coefficient_si(copy(), static_cast<enum isl_dim_type>(type), pos, v);
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

isl::constraint constraint::set_constant_val(long v) const
{
  return this->set_constant_val(isl::val(ctx(), v));
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

boolean fixed_box::is_valid() const
{
  auto res = isl_fixed_box_is_valid(get());
  return manage(res);
}

isl::multi_aff fixed_box::offset() const
{
  auto res = isl_fixed_box_get_offset(get());
  return manage(res);
}

isl::multi_aff fixed_box::get_offset() const
{
  return offset();
}

isl::multi_val fixed_box::size() const
{
  auto res = isl_fixed_box_get_size(get());
  return manage(res);
}

isl::multi_val fixed_box::get_size() const
{
  return size();
}

isl::space fixed_box::space() const
{
  auto res = isl_fixed_box_get_space(get());
  return manage(res);
}

isl::space fixed_box::get_space() const
{
  return space();
}

inline std::ostream &operator<<(std::ostream &os, const fixed_box &obj)
{
  char *str = isl_fixed_box_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::id id::alloc(isl::ctx ctx, const std::string &name, void * user)
{
  auto res = isl_id_alloc(ctx.release(), name.c_str(), user);
  return manage(res);
}

std::string id::name() const
{
  auto res = isl_id_get_name(get());
  std::string tmp(res);
  return tmp;
}

std::string id::get_name() const
{
  return name();
}

isl::id_list id::to_list() const
{
  auto res = isl_id_to_list(copy());
  return manage(res);
}

void * id::user() const
{
  auto res = isl_id_get_user(get());
  return res;
}

void * id::get_user() const
{
  return user();
}

inline std::ostream &operator<<(std::ostream &os, const id &obj)
{
  char *str = isl_id_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

id_list::id_list(isl::ctx ctx, int n)
{
  auto res = isl_id_list_alloc(ctx.release(), n);
  ptr = res;
}

id_list::id_list(isl::id el)
{
  auto res = isl_id_list_from_id(el.release());
  ptr = res;
}

id_list::id_list(isl::ctx ctx, const std::string &str)
{
  auto res = isl_id_list_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

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

isl::id_list id_list::add(isl::id el) const
{
  auto res = isl_id_list_add(copy(), el.release());
  return manage(res);
}

isl::id_list id_list::add(const std::string &el) const
{
  return this->add(isl::id(ctx(), el));
}

isl::id id_list::at(int index) const
{
  auto res = isl_id_list_get_at(get(), index);
  return manage(res);
}

isl::id id_list::get_at(int index) const
{
  return at(index);
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

stat id_list::foreach(const std::function<stat(isl::id)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::id)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_id *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_id_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

isl::id_list id_list::insert(unsigned int pos, isl::id el) const
{
  auto res = isl_id_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl::id_list id_list::insert(unsigned int pos, const std::string &el) const
{
  return this->insert(pos, isl::id(ctx(), el));
}

class size id_list::size() const
{
  auto res = isl_id_list_size(get());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const id_list &obj)
{
  char *str = isl_id_list_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::id_to_ast_expr id_to_ast_expr::alloc(isl::ctx ctx, int min_size)
{
  auto res = isl_id_to_ast_expr_alloc(ctx.release(), min_size);
  return manage(res);
}

isl::id_to_ast_expr id_to_ast_expr::set(isl::id key, isl::ast_expr val) const
{
  auto res = isl_id_to_ast_expr_set(copy(), key.release(), val.release());
  return manage(res);
}

isl::id_to_ast_expr id_to_ast_expr::set(const std::string &key, const isl::ast_expr &val) const
{
  return this->set(isl::id(ctx(), key), val);
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

isl::union_map map::apply_domain(const isl::union_map &umap2) const
{
  return isl::union_map(*this).apply_domain(umap2);
}

isl::map map::apply_domain(const isl::basic_map &map2) const
{
  return this->apply_domain(isl::map(map2));
}

isl::map map::apply_range(isl::map map2) const
{
  auto res = isl_map_apply_range(copy(), map2.release());
  return manage(res);
}

isl::union_map map::apply_range(const isl::union_map &umap2) const
{
  return isl::union_map(*this).apply_range(umap2);
}

isl::map map::apply_range(const isl::basic_map &map2) const
{
  return this->apply_range(isl::map(map2));
}

isl::map map::as_map() const
{
  return isl::union_map(*this).as_map();
}

isl::multi_union_pw_aff map::as_multi_union_pw_aff() const
{
  return isl::union_map(*this).as_multi_union_pw_aff();
}

isl::pw_multi_aff map::as_pw_multi_aff() const
{
  auto res = isl_map_as_pw_multi_aff(copy());
  return manage(res);
}

isl::union_pw_multi_aff map::as_union_pw_multi_aff() const
{
  return isl::union_map(*this).as_union_pw_multi_aff();
}

isl::basic_map_list map::basic_map_list() const
{
  auto res = isl_map_get_basic_map_list(get());
  return manage(res);
}

isl::basic_map_list map::get_basic_map_list() const
{
  return basic_map_list();
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

isl::union_map map::compute_divs() const
{
  return isl::union_map(*this).compute_divs();
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

isl::map map::detect_equalities() const
{
  auto res = isl_map_detect_equalities(copy());
  return manage(res);
}

class size map::dim(isl::dim type) const
{
  auto res = isl_map_dim(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
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

isl::map map::domain_map() const
{
  auto res = isl_map_domain_map(copy());
  return manage(res);
}

isl::union_pw_multi_aff map::domain_map_union_pw_multi_aff() const
{
  return isl::union_map(*this).domain_map_union_pw_multi_aff();
}

isl::map map::domain_product(isl::map map2) const
{
  auto res = isl_map_domain_product(copy(), map2.release());
  return manage(res);
}

isl::union_map map::domain_product(const isl::union_map &umap2) const
{
  return isl::union_map(*this).domain_product(umap2);
}

isl::map map::domain_product(const isl::basic_map &map2) const
{
  return this->domain_product(isl::map(map2));
}

class size map::domain_tuple_dim() const
{
  auto res = isl_map_domain_tuple_dim(get());
  return manage(res);
}

isl::id map::domain_tuple_id() const
{
  auto res = isl_map_get_domain_tuple_id(get());
  return manage(res);
}

isl::id map::get_domain_tuple_id() const
{
  return domain_tuple_id();
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

isl::union_map map::eq_at(const isl::multi_union_pw_aff &mupa) const
{
  return isl::union_map(*this).eq_at(mupa);
}

isl::map map::eq_at(const isl::aff &mpa) const
{
  return this->eq_at(isl::multi_pw_aff(mpa));
}

isl::map map::eq_at(const isl::multi_aff &mpa) const
{
  return this->eq_at(isl::multi_pw_aff(mpa));
}

isl::map map::eq_at(const isl::pw_aff &mpa) const
{
  return this->eq_at(isl::multi_pw_aff(mpa));
}

isl::map map::eq_at(const isl::pw_multi_aff &mpa) const
{
  return this->eq_at(isl::multi_pw_aff(mpa));
}

isl::map map::equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_map_equate(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

boolean map::every_map(const std::function<boolean(isl::map)> &test) const
{
  return isl::union_map(*this).every_map(test);
}

isl::map map::extract_map(const isl::space &space) const
{
  return isl::union_map(*this).extract_map(space);
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

isl::map map::fix_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_map_fix_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::union_map map::fixed_power(const isl::val &exp) const
{
  return isl::union_map(*this).fixed_power(exp);
}

isl::union_map map::fixed_power(long exp) const
{
  return this->fixed_power(isl::val(ctx(), exp));
}

isl::map map::flat_range_product(isl::map map2) const
{
  auto res = isl_map_flat_range_product(copy(), map2.release());
  return manage(res);
}

isl::union_map map::flat_range_product(const isl::union_map &umap2) const
{
  return isl::union_map(*this).flat_range_product(umap2);
}

isl::map map::flat_range_product(const isl::basic_map &map2) const
{
  return this->flat_range_product(isl::map(map2));
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

isl::map map::floordiv_val(long d) const
{
  return this->floordiv_val(isl::val(ctx(), d));
}

stat map::foreach_basic_map(const std::function<stat(isl::basic_map)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::basic_map)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_basic_map *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_map_foreach_basic_map(get(), fn_lambda, &fn_data);
  return manage(res);
}

stat map::foreach_map(const std::function<stat(isl::map)> &fn) const
{
  return isl::union_map(*this).foreach_map(fn);
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

isl::map map::from_pw_aff(isl::pw_aff pwaff)
{
  auto res = isl_map_from_pw_aff(pwaff.release());
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

isl::map map::gist(isl::map context) const
{
  auto res = isl_map_gist(copy(), context.release());
  return manage(res);
}

isl::union_map map::gist(const isl::union_map &context) const
{
  return isl::union_map(*this).gist(context);
}

isl::map map::gist(const isl::basic_map &context) const
{
  return this->gist(isl::map(context));
}

isl::map map::gist_domain(isl::set context) const
{
  auto res = isl_map_gist_domain(copy(), context.release());
  return manage(res);
}

isl::union_map map::gist_domain(const isl::union_set &uset) const
{
  return isl::union_map(*this).gist_domain(uset);
}

isl::map map::gist_domain(const isl::basic_set &context) const
{
  return this->gist_domain(isl::set(context));
}

isl::map map::gist_domain(const isl::point &context) const
{
  return this->gist_domain(isl::set(context));
}

isl::map map::gist_params(isl::set context) const
{
  auto res = isl_map_gist_params(copy(), context.release());
  return manage(res);
}

isl::union_map map::gist_range(const isl::union_set &uset) const
{
  return isl::union_map(*this).gist_range(uset);
}

boolean map::has_domain_tuple_id() const
{
  auto res = isl_map_has_domain_tuple_id(get());
  return manage(res);
}

boolean map::has_equal_space(const isl::map &map2) const
{
  auto res = isl_map_has_equal_space(get(), map2.get());
  return manage(res);
}

boolean map::has_range_tuple_id() const
{
  auto res = isl_map_has_range_tuple_id(get());
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

isl::map map::intersect(isl::map map2) const
{
  auto res = isl_map_intersect(copy(), map2.release());
  return manage(res);
}

isl::union_map map::intersect(const isl::union_map &umap2) const
{
  return isl::union_map(*this).intersect(umap2);
}

isl::map map::intersect(const isl::basic_map &map2) const
{
  return this->intersect(isl::map(map2));
}

isl::map map::intersect_domain(isl::set set) const
{
  auto res = isl_map_intersect_domain(copy(), set.release());
  return manage(res);
}

isl::union_map map::intersect_domain(const isl::space &space) const
{
  return isl::union_map(*this).intersect_domain(space);
}

isl::union_map map::intersect_domain(const isl::union_set &uset) const
{
  return isl::union_map(*this).intersect_domain(uset);
}

isl::map map::intersect_domain(const isl::basic_set &set) const
{
  return this->intersect_domain(isl::set(set));
}

isl::map map::intersect_domain(const isl::point &set) const
{
  return this->intersect_domain(isl::set(set));
}

isl::map map::intersect_domain_factor_domain(isl::map factor) const
{
  auto res = isl_map_intersect_domain_factor_domain(copy(), factor.release());
  return manage(res);
}

isl::union_map map::intersect_domain_factor_domain(const isl::union_map &factor) const
{
  return isl::union_map(*this).intersect_domain_factor_domain(factor);
}

isl::map map::intersect_domain_factor_domain(const isl::basic_map &factor) const
{
  return this->intersect_domain_factor_domain(isl::map(factor));
}

isl::map map::intersect_domain_factor_range(isl::map factor) const
{
  auto res = isl_map_intersect_domain_factor_range(copy(), factor.release());
  return manage(res);
}

isl::union_map map::intersect_domain_factor_range(const isl::union_map &factor) const
{
  return isl::union_map(*this).intersect_domain_factor_range(factor);
}

isl::map map::intersect_domain_factor_range(const isl::basic_map &factor) const
{
  return this->intersect_domain_factor_range(isl::map(factor));
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

isl::union_map map::intersect_range(const isl::space &space) const
{
  return isl::union_map(*this).intersect_range(space);
}

isl::union_map map::intersect_range(const isl::union_set &uset) const
{
  return isl::union_map(*this).intersect_range(uset);
}

isl::map map::intersect_range(const isl::basic_set &set) const
{
  return this->intersect_range(isl::set(set));
}

isl::map map::intersect_range(const isl::point &set) const
{
  return this->intersect_range(isl::set(set));
}

isl::map map::intersect_range_factor_domain(isl::map factor) const
{
  auto res = isl_map_intersect_range_factor_domain(copy(), factor.release());
  return manage(res);
}

isl::union_map map::intersect_range_factor_domain(const isl::union_map &factor) const
{
  return isl::union_map(*this).intersect_range_factor_domain(factor);
}

isl::map map::intersect_range_factor_domain(const isl::basic_map &factor) const
{
  return this->intersect_range_factor_domain(isl::map(factor));
}

isl::map map::intersect_range_factor_range(isl::map factor) const
{
  auto res = isl_map_intersect_range_factor_range(copy(), factor.release());
  return manage(res);
}

isl::union_map map::intersect_range_factor_range(const isl::union_map &factor) const
{
  return isl::union_map(*this).intersect_range_factor_range(factor);
}

isl::map map::intersect_range_factor_range(const isl::basic_map &factor) const
{
  return this->intersect_range_factor_range(isl::map(factor));
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

boolean map::is_disjoint(const isl::union_map &umap2) const
{
  return isl::union_map(*this).is_disjoint(umap2);
}

boolean map::is_disjoint(const isl::basic_map &map2) const
{
  return this->is_disjoint(isl::map(map2));
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

boolean map::is_equal(const isl::union_map &umap2) const
{
  return isl::union_map(*this).is_equal(umap2);
}

boolean map::is_equal(const isl::basic_map &map2) const
{
  return this->is_equal(isl::map(map2));
}

boolean map::is_injective() const
{
  auto res = isl_map_is_injective(get());
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

boolean map::is_strict_subset(const isl::union_map &umap2) const
{
  return isl::union_map(*this).is_strict_subset(umap2);
}

boolean map::is_strict_subset(const isl::basic_map &map2) const
{
  return this->is_strict_subset(isl::map(map2));
}

boolean map::is_subset(const isl::map &map2) const
{
  auto res = isl_map_is_subset(get(), map2.get());
  return manage(res);
}

boolean map::is_subset(const isl::union_map &umap2) const
{
  return isl::union_map(*this).is_subset(umap2);
}

boolean map::is_subset(const isl::basic_map &map2) const
{
  return this->is_subset(isl::map(map2));
}

boolean map::isa_map() const
{
  return isl::union_map(*this).isa_map();
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

isl::map_list map::map_list() const
{
  return isl::union_map(*this).map_list();
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

class size map::n_basic_map() const
{
  auto res = isl_map_n_basic_map(get());
  return manage(res);
}

isl::map map::order_lt(isl::dim type1, int pos1, isl::dim type2, int pos2) const
{
  auto res = isl_map_order_lt(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

isl::set map::params() const
{
  return isl::union_map(*this).params();
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

isl::union_map map::preimage_domain(const isl::union_pw_multi_aff &upma) const
{
  return isl::union_map(*this).preimage_domain(upma);
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

isl::union_map map::preimage_range(const isl::union_pw_multi_aff &upma) const
{
  return isl::union_map(*this).preimage_range(upma);
}

isl::map map::product(isl::map map2) const
{
  auto res = isl_map_product(copy(), map2.release());
  return manage(res);
}

isl::union_map map::product(const isl::union_map &umap2) const
{
  return isl::union_map(*this).product(umap2);
}

isl::map map::product(const isl::basic_map &map2) const
{
  return this->product(isl::map(map2));
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

isl::fixed_box map::range_lattice_tile() const
{
  auto res = isl_map_get_range_lattice_tile(get());
  return manage(res);
}

isl::fixed_box map::get_range_lattice_tile() const
{
  return range_lattice_tile();
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

isl::union_map map::range_product(const isl::union_map &umap2) const
{
  return isl::union_map(*this).range_product(umap2);
}

isl::map map::range_product(const isl::basic_map &map2) const
{
  return this->range_product(isl::map(map2));
}

isl::map map::range_reverse() const
{
  auto res = isl_map_range_reverse(copy());
  return manage(res);
}

isl::fixed_box map::range_simple_fixed_box_hull() const
{
  auto res = isl_map_get_range_simple_fixed_box_hull(get());
  return manage(res);
}

isl::fixed_box map::get_range_simple_fixed_box_hull() const
{
  return range_simple_fixed_box_hull();
}

class size map::range_tuple_dim() const
{
  auto res = isl_map_range_tuple_dim(get());
  return manage(res);
}

isl::id map::range_tuple_id() const
{
  auto res = isl_map_get_range_tuple_id(get());
  return manage(res);
}

isl::id map::get_range_tuple_id() const
{
  return range_tuple_id();
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

isl::map map::set_domain_tuple(isl::id id) const
{
  auto res = isl_map_set_domain_tuple_id(copy(), id.release());
  return manage(res);
}

isl::map map::set_domain_tuple(const std::string &id) const
{
  return this->set_domain_tuple(isl::id(ctx(), id));
}

isl::map map::set_range_tuple(isl::id id) const
{
  auto res = isl_map_set_range_tuple_id(copy(), id.release());
  return manage(res);
}

isl::map map::set_range_tuple(const std::string &id) const
{
  return this->set_range_tuple(isl::id(ctx(), id));
}

isl::map map::set_tuple_id(isl::dim type, isl::id id) const
{
  auto res = isl_map_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::map map::set_tuple_id(isl::dim type, const std::string &id) const
{
  return this->set_tuple_id(type, isl::id(ctx(), id));
}

isl::space map::space() const
{
  auto res = isl_map_get_space(get());
  return manage(res);
}

isl::space map::get_space() const
{
  return space();
}

isl::map map::subtract(isl::map map2) const
{
  auto res = isl_map_subtract(copy(), map2.release());
  return manage(res);
}

isl::union_map map::subtract(const isl::union_map &umap2) const
{
  return isl::union_map(*this).subtract(umap2);
}

isl::map map::subtract(const isl::basic_map &map2) const
{
  return this->subtract(isl::map(map2));
}

isl::union_map map::subtract_domain(const isl::union_set &dom) const
{
  return isl::union_map(*this).subtract_domain(dom);
}

isl::union_map map::subtract_range(const isl::union_set &dom) const
{
  return isl::union_map(*this).subtract_range(dom);
}

isl::map map::sum(isl::map map2) const
{
  auto res = isl_map_sum(copy(), map2.release());
  return manage(res);
}

isl::map_list map::to_list() const
{
  auto res = isl_map_to_list(copy());
  return manage(res);
}

isl::union_map map::to_union_map() const
{
  auto res = isl_map_to_union_map(copy());
  return manage(res);
}

isl::id map::tuple_id(isl::dim type) const
{
  auto res = isl_map_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::id map::get_tuple_id(isl::dim type) const
{
  return tuple_id(type);
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

isl::union_map map::unite(const isl::union_map &umap2) const
{
  return isl::union_map(*this).unite(umap2);
}

isl::map map::unite(const isl::basic_map &map2) const
{
  return this->unite(isl::map(map2));
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

inline std::ostream &operator<<(std::ostream &os, const map &obj)
{
  char *str = isl_map_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

map_list::map_list(isl::ctx ctx, int n)
{
  auto res = isl_map_list_alloc(ctx.release(), n);
  ptr = res;
}

map_list::map_list(isl::map el)
{
  auto res = isl_map_list_from_map(el.release());
  ptr = res;
}

map_list::map_list(isl::ctx ctx, const std::string &str)
{
  auto res = isl_map_list_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

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

isl::map_list map_list::add(isl::map el) const
{
  auto res = isl_map_list_add(copy(), el.release());
  return manage(res);
}

isl::map map_list::at(int index) const
{
  auto res = isl_map_list_get_at(get(), index);
  return manage(res);
}

isl::map map_list::get_at(int index) const
{
  return at(index);
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

stat map_list::foreach(const std::function<stat(isl::map)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::map)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_map *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_map_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

isl::map_list map_list::insert(unsigned int pos, isl::map el) const
{
  auto res = isl_map_list_insert(copy(), pos, el.release());
  return manage(res);
}

class size map_list::size() const
{
  auto res = isl_map_list_size(get());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const map_list &obj)
{
  char *str = isl_map_list_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::multi_aff multi_aff::add(isl::multi_aff multi2) const
{
  auto res = isl_multi_aff_add(copy(), multi2.release());
  return manage(res);
}

isl::multi_pw_aff multi_aff::add(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).add(multi2);
}

isl::multi_union_pw_aff multi_aff::add(const isl::multi_union_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).add(multi2);
}

isl::pw_multi_aff multi_aff::add(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_multi_aff(*this).add(pma2);
}

isl::union_pw_multi_aff multi_aff::add(const isl::union_pw_multi_aff &upma2) const
{
  return isl::pw_multi_aff(*this).add(upma2);
}

isl::multi_aff multi_aff::add(const isl::aff &multi2) const
{
  return this->add(isl::multi_aff(multi2));
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

isl::multi_aff multi_aff::add_constant(long v) const
{
  return this->add_constant(isl::val(ctx(), v));
}

isl::union_pw_multi_aff multi_aff::add_pw_multi_aff(const isl::pw_multi_aff &pma) const
{
  return isl::pw_multi_aff(*this).add_pw_multi_aff(pma);
}

isl::union_pw_multi_aff multi_aff::apply(const isl::union_pw_multi_aff &upma2) const
{
  return isl::pw_multi_aff(*this).apply(upma2);
}

isl::map multi_aff::as_map() const
{
  auto res = isl_multi_aff_as_map(copy());
  return manage(res);
}

isl::multi_aff multi_aff::as_multi_aff() const
{
  return isl::pw_multi_aff(*this).as_multi_aff();
}

isl::multi_union_pw_aff multi_aff::as_multi_union_pw_aff() const
{
  return isl::pw_multi_aff(*this).as_multi_union_pw_aff();
}

isl::pw_multi_aff multi_aff::as_pw_multi_aff() const
{
  return isl::pw_multi_aff(*this).as_pw_multi_aff();
}

isl::set multi_aff::as_set() const
{
  auto res = isl_multi_aff_as_set(copy());
  return manage(res);
}

isl::union_map multi_aff::as_union_map() const
{
  return isl::pw_multi_aff(*this).as_union_map();
}

isl::aff multi_aff::at(int pos) const
{
  auto res = isl_multi_aff_get_at(get(), pos);
  return manage(res);
}

isl::aff multi_aff::get_at(int pos) const
{
  return at(pos);
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

isl::pw_multi_aff multi_aff::coalesce() const
{
  return isl::pw_multi_aff(*this).coalesce();
}

isl::multi_val multi_aff::constant_multi_val() const
{
  auto res = isl_multi_aff_get_constant_multi_val(get());
  return manage(res);
}

isl::multi_val multi_aff::get_constant_multi_val() const
{
  return constant_multi_val();
}

class size multi_aff::dim(isl::dim type) const
{
  auto res = isl_multi_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::set multi_aff::domain() const
{
  return isl::pw_multi_aff(*this).domain();
}

isl::multi_aff multi_aff::domain_map(isl::space space)
{
  auto res = isl_multi_aff_domain_map(space.release());
  return manage(res);
}

isl::pw_multi_aff multi_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  return isl::pw_multi_aff(*this).drop_dims(type, first, n);
}

isl::pw_multi_aff multi_aff::extract_pw_multi_aff(const isl::space &space) const
{
  return isl::pw_multi_aff(*this).extract_pw_multi_aff(space);
}

isl::multi_aff multi_aff::flat_range_product(isl::multi_aff multi2) const
{
  auto res = isl_multi_aff_flat_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_pw_aff multi_aff::flat_range_product(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).flat_range_product(multi2);
}

isl::multi_union_pw_aff multi_aff::flat_range_product(const isl::multi_union_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).flat_range_product(multi2);
}

isl::pw_multi_aff multi_aff::flat_range_product(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_multi_aff(*this).flat_range_product(pma2);
}

isl::union_pw_multi_aff multi_aff::flat_range_product(const isl::union_pw_multi_aff &upma2) const
{
  return isl::pw_multi_aff(*this).flat_range_product(upma2);
}

isl::multi_aff multi_aff::flat_range_product(const isl::aff &multi2) const
{
  return this->flat_range_product(isl::multi_aff(multi2));
}

isl::multi_aff multi_aff::floor() const
{
  auto res = isl_multi_aff_floor(copy());
  return manage(res);
}

stat multi_aff::foreach_piece(const std::function<stat(isl::set, isl::multi_aff)> &fn) const
{
  return isl::pw_multi_aff(*this).foreach_piece(fn);
}

isl::multi_aff multi_aff::gist(isl::set context) const
{
  auto res = isl_multi_aff_gist(copy(), context.release());
  return manage(res);
}

isl::union_pw_multi_aff multi_aff::gist(const isl::union_set &context) const
{
  return isl::pw_multi_aff(*this).gist(context);
}

isl::multi_aff multi_aff::gist(const isl::basic_set &context) const
{
  return this->gist(isl::set(context));
}

isl::multi_aff multi_aff::gist(const isl::point &context) const
{
  return this->gist(isl::set(context));
}

boolean multi_aff::has_range_tuple_id() const
{
  auto res = isl_multi_aff_has_range_tuple_id(get());
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

isl::multi_aff multi_aff::insert_domain(isl::space domain) const
{
  auto res = isl_multi_aff_insert_domain(copy(), domain.release());
  return manage(res);
}

isl::pw_multi_aff multi_aff::intersect_domain(const isl::set &set) const
{
  return isl::pw_multi_aff(*this).intersect_domain(set);
}

isl::union_pw_multi_aff multi_aff::intersect_domain(const isl::space &space) const
{
  return isl::pw_multi_aff(*this).intersect_domain(space);
}

isl::union_pw_multi_aff multi_aff::intersect_domain(const isl::union_set &uset) const
{
  return isl::pw_multi_aff(*this).intersect_domain(uset);
}

isl::union_pw_multi_aff multi_aff::intersect_domain_wrapped_domain(const isl::union_set &uset) const
{
  return isl::pw_multi_aff(*this).intersect_domain_wrapped_domain(uset);
}

isl::union_pw_multi_aff multi_aff::intersect_domain_wrapped_range(const isl::union_set &uset) const
{
  return isl::pw_multi_aff(*this).intersect_domain_wrapped_range(uset);
}

isl::pw_multi_aff multi_aff::intersect_params(const isl::set &set) const
{
  return isl::pw_multi_aff(*this).intersect_params(set);
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

boolean multi_aff::involves_param(const isl::id &id) const
{
  return isl::pw_multi_aff(*this).involves_param(id);
}

boolean multi_aff::involves_param(const std::string &id) const
{
  return this->involves_param(isl::id(ctx(), id));
}

boolean multi_aff::involves_param(const isl::id_list &list) const
{
  return isl::pw_multi_aff(*this).involves_param(list);
}

boolean multi_aff::isa_multi_aff() const
{
  return isl::pw_multi_aff(*this).isa_multi_aff();
}

boolean multi_aff::isa_pw_multi_aff() const
{
  return isl::pw_multi_aff(*this).isa_pw_multi_aff();
}

isl::aff_list multi_aff::list() const
{
  auto res = isl_multi_aff_get_list(get());
  return manage(res);
}

isl::aff_list multi_aff::get_list() const
{
  return list();
}

isl::multi_pw_aff multi_aff::max(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).max(multi2);
}

isl::multi_val multi_aff::max_multi_val() const
{
  return isl::pw_multi_aff(*this).max_multi_val();
}

isl::multi_pw_aff multi_aff::min(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).min(multi2);
}

isl::multi_val multi_aff::min_multi_val() const
{
  return isl::pw_multi_aff(*this).min_multi_val();
}

isl::multi_aff multi_aff::multi_val_on_domain(isl::space space, isl::multi_val mv)
{
  auto res = isl_multi_aff_multi_val_on_domain_space(space.release(), mv.release());
  return manage(res);
}

class size multi_aff::n_piece() const
{
  return isl::pw_multi_aff(*this).n_piece();
}

isl::multi_aff multi_aff::neg() const
{
  auto res = isl_multi_aff_neg(copy());
  return manage(res);
}

boolean multi_aff::plain_is_empty() const
{
  return isl::pw_multi_aff(*this).plain_is_empty();
}

boolean multi_aff::plain_is_equal(const isl::multi_aff &multi2) const
{
  auto res = isl_multi_aff_plain_is_equal(get(), multi2.get());
  return manage(res);
}

boolean multi_aff::plain_is_equal(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).plain_is_equal(multi2);
}

boolean multi_aff::plain_is_equal(const isl::multi_union_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).plain_is_equal(multi2);
}

boolean multi_aff::plain_is_equal(const isl::aff &multi2) const
{
  return this->plain_is_equal(isl::multi_aff(multi2));
}

isl::pw_multi_aff multi_aff::preimage_domain_wrapped_domain(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_multi_aff(*this).preimage_domain_wrapped_domain(pma2);
}

isl::union_pw_multi_aff multi_aff::preimage_domain_wrapped_domain(const isl::union_pw_multi_aff &upma2) const
{
  return isl::pw_multi_aff(*this).preimage_domain_wrapped_domain(upma2);
}

isl::multi_aff multi_aff::product(isl::multi_aff multi2) const
{
  auto res = isl_multi_aff_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_pw_aff multi_aff::product(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).product(multi2);
}

isl::pw_multi_aff multi_aff::product(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_multi_aff(*this).product(pma2);
}

isl::multi_aff multi_aff::product(const isl::aff &multi2) const
{
  return this->product(isl::multi_aff(multi2));
}

isl::multi_aff multi_aff::pullback(isl::multi_aff ma2) const
{
  auto res = isl_multi_aff_pullback_multi_aff(copy(), ma2.release());
  return manage(res);
}

isl::multi_pw_aff multi_aff::pullback(const isl::multi_pw_aff &mpa2) const
{
  return isl::pw_multi_aff(*this).pullback(mpa2);
}

isl::pw_multi_aff multi_aff::pullback(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_multi_aff(*this).pullback(pma2);
}

isl::union_pw_multi_aff multi_aff::pullback(const isl::union_pw_multi_aff &upma2) const
{
  return isl::pw_multi_aff(*this).pullback(upma2);
}

isl::multi_aff multi_aff::pullback(const isl::aff &ma2) const
{
  return this->pullback(isl::multi_aff(ma2));
}

isl::pw_multi_aff_list multi_aff::pw_multi_aff_list() const
{
  return isl::pw_multi_aff(*this).pw_multi_aff_list();
}

isl::pw_multi_aff multi_aff::range_factor_domain() const
{
  return isl::pw_multi_aff(*this).range_factor_domain();
}

isl::pw_multi_aff multi_aff::range_factor_range() const
{
  return isl::pw_multi_aff(*this).range_factor_range();
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

isl::multi_pw_aff multi_aff::range_product(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).range_product(multi2);
}

isl::multi_union_pw_aff multi_aff::range_product(const isl::multi_union_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).range_product(multi2);
}

isl::pw_multi_aff multi_aff::range_product(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_multi_aff(*this).range_product(pma2);
}

isl::union_pw_multi_aff multi_aff::range_product(const isl::union_pw_multi_aff &upma2) const
{
  return isl::pw_multi_aff(*this).range_product(upma2);
}

isl::multi_aff multi_aff::range_product(const isl::aff &multi2) const
{
  return this->range_product(isl::multi_aff(multi2));
}

isl::id multi_aff::range_tuple_id() const
{
  auto res = isl_multi_aff_get_range_tuple_id(get());
  return manage(res);
}

isl::id multi_aff::get_range_tuple_id() const
{
  return range_tuple_id();
}

isl::multi_aff multi_aff::reset_range_tuple_id() const
{
  auto res = isl_multi_aff_reset_range_tuple_id(copy());
  return manage(res);
}

isl::multi_aff multi_aff::reset_tuple_id(isl::dim type) const
{
  auto res = isl_multi_aff_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
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

isl::multi_aff multi_aff::scale(long v) const
{
  return this->scale(isl::val(ctx(), v));
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

isl::multi_aff multi_aff::scale_down(long v) const
{
  return this->scale_down(isl::val(ctx(), v));
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

isl::multi_pw_aff multi_aff::set_at(int pos, const isl::pw_aff &el) const
{
  return isl::pw_multi_aff(*this).set_at(pos, el);
}

isl::multi_union_pw_aff multi_aff::set_at(int pos, const isl::union_pw_aff &el) const
{
  return isl::pw_multi_aff(*this).set_at(pos, el);
}

isl::multi_pw_aff multi_aff::set_pw_aff(int pos, const isl::pw_aff &el) const
{
  return isl::pw_multi_aff(*this).set_pw_aff(pos, el);
}

isl::pw_multi_aff multi_aff::set_pw_aff(unsigned int pos, const isl::pw_aff &pa) const
{
  return isl::pw_multi_aff(*this).set_pw_aff(pos, pa);
}

isl::multi_aff multi_aff::set_range_tuple(isl::id id) const
{
  auto res = isl_multi_aff_set_range_tuple_id(copy(), id.release());
  return manage(res);
}

isl::multi_aff multi_aff::set_range_tuple(const std::string &id) const
{
  return this->set_range_tuple(isl::id(ctx(), id));
}

isl::multi_union_pw_aff multi_aff::set_union_pw_aff(int pos, const isl::union_pw_aff &el) const
{
  return isl::pw_multi_aff(*this).set_union_pw_aff(pos, el);
}

class size multi_aff::size() const
{
  auto res = isl_multi_aff_size(get());
  return manage(res);
}

isl::space multi_aff::space() const
{
  auto res = isl_multi_aff_get_space(get());
  return manage(res);
}

isl::space multi_aff::get_space() const
{
  return space();
}

isl::multi_aff multi_aff::sub(isl::multi_aff multi2) const
{
  auto res = isl_multi_aff_sub(copy(), multi2.release());
  return manage(res);
}

isl::multi_pw_aff multi_aff::sub(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).sub(multi2);
}

isl::multi_union_pw_aff multi_aff::sub(const isl::multi_union_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).sub(multi2);
}

isl::pw_multi_aff multi_aff::sub(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_multi_aff(*this).sub(pma2);
}

isl::union_pw_multi_aff multi_aff::sub(const isl::union_pw_multi_aff &upma2) const
{
  return isl::pw_multi_aff(*this).sub(upma2);
}

isl::multi_aff multi_aff::sub(const isl::aff &multi2) const
{
  return this->sub(isl::multi_aff(multi2));
}

isl::pw_multi_aff multi_aff::subtract_domain(const isl::set &set) const
{
  return isl::pw_multi_aff(*this).subtract_domain(set);
}

isl::union_pw_multi_aff multi_aff::subtract_domain(const isl::space &space) const
{
  return isl::pw_multi_aff(*this).subtract_domain(space);
}

isl::union_pw_multi_aff multi_aff::subtract_domain(const isl::union_set &uset) const
{
  return isl::pw_multi_aff(*this).subtract_domain(uset);
}

isl::pw_multi_aff_list multi_aff::to_list() const
{
  return isl::pw_multi_aff(*this).to_list();
}

isl::multi_pw_aff multi_aff::to_multi_pw_aff() const
{
  auto res = isl_multi_aff_to_multi_pw_aff(copy());
  return manage(res);
}

isl::multi_union_pw_aff multi_aff::to_multi_union_pw_aff() const
{
  auto res = isl_multi_aff_to_multi_union_pw_aff(copy());
  return manage(res);
}

isl::pw_multi_aff multi_aff::to_pw_multi_aff() const
{
  auto res = isl_multi_aff_to_pw_multi_aff(copy());
  return manage(res);
}

isl::union_pw_multi_aff multi_aff::to_union_pw_multi_aff() const
{
  return isl::pw_multi_aff(*this).to_union_pw_multi_aff();
}

isl::id multi_aff::tuple_id(isl::dim type) const
{
  return isl::pw_multi_aff(*this).tuple_id(type);
}

isl::multi_aff multi_aff::unbind_params_insert_domain(isl::multi_id domain) const
{
  auto res = isl_multi_aff_unbind_params_insert_domain(copy(), domain.release());
  return manage(res);
}

isl::multi_pw_aff multi_aff::union_add(const isl::multi_pw_aff &mpa2) const
{
  return isl::pw_multi_aff(*this).union_add(mpa2);
}

isl::multi_union_pw_aff multi_aff::union_add(const isl::multi_union_pw_aff &mupa2) const
{
  return isl::pw_multi_aff(*this).union_add(mupa2);
}

isl::pw_multi_aff multi_aff::union_add(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_multi_aff(*this).union_add(pma2);
}

isl::union_pw_multi_aff multi_aff::union_add(const isl::union_pw_multi_aff &upma2) const
{
  return isl::pw_multi_aff(*this).union_add(upma2);
}

isl::multi_aff multi_aff::zero(isl::space space)
{
  auto res = isl_multi_aff_zero(space.release());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const multi_aff &obj)
{
  char *str = isl_multi_aff_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::id multi_id::at(int pos) const
{
  auto res = isl_multi_id_get_at(get(), pos);
  return manage(res);
}

isl::id multi_id::get_at(int pos) const
{
  return at(pos);
}

isl::multi_id multi_id::flat_range_product(isl::multi_id multi2) const
{
  auto res = isl_multi_id_flat_range_product(copy(), multi2.release());
  return manage(res);
}

isl::id_list multi_id::list() const
{
  auto res = isl_multi_id_get_list(get());
  return manage(res);
}

isl::id_list multi_id::get_list() const
{
  return list();
}

boolean multi_id::plain_is_equal(const isl::multi_id &multi2) const
{
  auto res = isl_multi_id_plain_is_equal(get(), multi2.get());
  return manage(res);
}

isl::multi_id multi_id::range_product(isl::multi_id multi2) const
{
  auto res = isl_multi_id_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_id multi_id::set_at(int pos, isl::id el) const
{
  auto res = isl_multi_id_set_at(copy(), pos, el.release());
  return manage(res);
}

isl::multi_id multi_id::set_at(int pos, const std::string &el) const
{
  return this->set_at(pos, isl::id(ctx(), el));
}

isl::multi_id multi_id::set_id(int pos, isl::id el) const
{
  auto res = isl_multi_id_set_id(copy(), pos, el.release());
  return manage(res);
}

isl::multi_id multi_id::set_id(int pos, const std::string &el) const
{
  return this->set_id(pos, isl::id(ctx(), el));
}

class size multi_id::size() const
{
  auto res = isl_multi_id_size(get());
  return manage(res);
}

isl::space multi_id::space() const
{
  auto res = isl_multi_id_get_space(get());
  return manage(res);
}

isl::space multi_id::get_space() const
{
  return space();
}

inline std::ostream &operator<<(std::ostream &os, const multi_id &obj)
{
  char *str = isl_multi_id_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::multi_pw_aff multi_pw_aff::add(isl::multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_add(copy(), multi2.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_pw_aff::add(const isl::multi_union_pw_aff &multi2) const
{
  return isl::multi_union_pw_aff(*this).add(multi2);
}

isl::multi_pw_aff multi_pw_aff::add(const isl::aff &multi2) const
{
  return this->add(isl::multi_pw_aff(multi2));
}

isl::multi_pw_aff multi_pw_aff::add(const isl::multi_aff &multi2) const
{
  return this->add(isl::multi_pw_aff(multi2));
}

isl::multi_pw_aff multi_pw_aff::add(const isl::pw_aff &multi2) const
{
  return this->add(isl::multi_pw_aff(multi2));
}

isl::multi_pw_aff multi_pw_aff::add(const isl::pw_multi_aff &multi2) const
{
  return this->add(isl::multi_pw_aff(multi2));
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

isl::multi_pw_aff multi_pw_aff::add_constant(long v) const
{
  return this->add_constant(isl::val(ctx(), v));
}

isl::map multi_pw_aff::as_map() const
{
  auto res = isl_multi_pw_aff_as_map(copy());
  return manage(res);
}

isl::multi_aff multi_pw_aff::as_multi_aff() const
{
  auto res = isl_multi_pw_aff_as_multi_aff(copy());
  return manage(res);
}

isl::set multi_pw_aff::as_set() const
{
  auto res = isl_multi_pw_aff_as_set(copy());
  return manage(res);
}

isl::pw_aff multi_pw_aff::at(int pos) const
{
  auto res = isl_multi_pw_aff_get_at(get(), pos);
  return manage(res);
}

isl::pw_aff multi_pw_aff::get_at(int pos) const
{
  return at(pos);
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

class size multi_pw_aff::dim(isl::dim type) const
{
  auto res = isl_multi_pw_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::set multi_pw_aff::domain() const
{
  auto res = isl_multi_pw_aff_domain(copy());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::flat_range_product(isl::multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_flat_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_pw_aff::flat_range_product(const isl::multi_union_pw_aff &multi2) const
{
  return isl::multi_union_pw_aff(*this).flat_range_product(multi2);
}

isl::multi_pw_aff multi_pw_aff::flat_range_product(const isl::aff &multi2) const
{
  return this->flat_range_product(isl::multi_pw_aff(multi2));
}

isl::multi_pw_aff multi_pw_aff::flat_range_product(const isl::multi_aff &multi2) const
{
  return this->flat_range_product(isl::multi_pw_aff(multi2));
}

isl::multi_pw_aff multi_pw_aff::flat_range_product(const isl::pw_aff &multi2) const
{
  return this->flat_range_product(isl::multi_pw_aff(multi2));
}

isl::multi_pw_aff multi_pw_aff::flat_range_product(const isl::pw_multi_aff &multi2) const
{
  return this->flat_range_product(isl::multi_pw_aff(multi2));
}

isl::multi_pw_aff multi_pw_aff::gist(isl::set set) const
{
  auto res = isl_multi_pw_aff_gist(copy(), set.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_pw_aff::gist(const isl::union_set &context) const
{
  return isl::multi_union_pw_aff(*this).gist(context);
}

isl::multi_pw_aff multi_pw_aff::gist(const isl::basic_set &set) const
{
  return this->gist(isl::set(set));
}

isl::multi_pw_aff multi_pw_aff::gist(const isl::point &set) const
{
  return this->gist(isl::set(set));
}

boolean multi_pw_aff::has_range_tuple_id() const
{
  auto res = isl_multi_pw_aff_has_range_tuple_id(get());
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

isl::multi_union_pw_aff multi_pw_aff::intersect_domain(const isl::union_set &uset) const
{
  return isl::multi_union_pw_aff(*this).intersect_domain(uset);
}

isl::multi_pw_aff multi_pw_aff::intersect_domain(const isl::basic_set &domain) const
{
  return this->intersect_domain(isl::set(domain));
}

isl::multi_pw_aff multi_pw_aff::intersect_domain(const isl::point &domain) const
{
  return this->intersect_domain(isl::set(domain));
}

isl::multi_pw_aff multi_pw_aff::intersect_params(isl::set set) const
{
  auto res = isl_multi_pw_aff_intersect_params(copy(), set.release());
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

boolean multi_pw_aff::involves_param(const std::string &id) const
{
  return this->involves_param(isl::id(ctx(), id));
}

boolean multi_pw_aff::involves_param(const isl::id_list &list) const
{
  auto res = isl_multi_pw_aff_involves_param_id_list(get(), list.get());
  return manage(res);
}

boolean multi_pw_aff::isa_multi_aff() const
{
  auto res = isl_multi_pw_aff_isa_multi_aff(get());
  return manage(res);
}

isl::pw_aff_list multi_pw_aff::list() const
{
  auto res = isl_multi_pw_aff_get_list(get());
  return manage(res);
}

isl::pw_aff_list multi_pw_aff::get_list() const
{
  return list();
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

boolean multi_pw_aff::plain_is_equal(const isl::multi_union_pw_aff &multi2) const
{
  return isl::multi_union_pw_aff(*this).plain_is_equal(multi2);
}

boolean multi_pw_aff::plain_is_equal(const isl::aff &multi2) const
{
  return this->plain_is_equal(isl::multi_pw_aff(multi2));
}

boolean multi_pw_aff::plain_is_equal(const isl::multi_aff &multi2) const
{
  return this->plain_is_equal(isl::multi_pw_aff(multi2));
}

boolean multi_pw_aff::plain_is_equal(const isl::pw_aff &multi2) const
{
  return this->plain_is_equal(isl::multi_pw_aff(multi2));
}

boolean multi_pw_aff::plain_is_equal(const isl::pw_multi_aff &multi2) const
{
  return this->plain_is_equal(isl::multi_pw_aff(multi2));
}

isl::multi_pw_aff multi_pw_aff::product(isl::multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_product(copy(), multi2.release());
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

isl::multi_union_pw_aff multi_pw_aff::pullback(const isl::union_pw_multi_aff &upma) const
{
  return isl::multi_union_pw_aff(*this).pullback(upma);
}

isl::multi_pw_aff multi_pw_aff::range_product(isl::multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_pw_aff::range_product(const isl::multi_union_pw_aff &multi2) const
{
  return isl::multi_union_pw_aff(*this).range_product(multi2);
}

isl::multi_pw_aff multi_pw_aff::range_product(const isl::aff &multi2) const
{
  return this->range_product(isl::multi_pw_aff(multi2));
}

isl::multi_pw_aff multi_pw_aff::range_product(const isl::multi_aff &multi2) const
{
  return this->range_product(isl::multi_pw_aff(multi2));
}

isl::multi_pw_aff multi_pw_aff::range_product(const isl::pw_aff &multi2) const
{
  return this->range_product(isl::multi_pw_aff(multi2));
}

isl::multi_pw_aff multi_pw_aff::range_product(const isl::pw_multi_aff &multi2) const
{
  return this->range_product(isl::multi_pw_aff(multi2));
}

isl::id multi_pw_aff::range_tuple_id() const
{
  auto res = isl_multi_pw_aff_get_range_tuple_id(get());
  return manage(res);
}

isl::id multi_pw_aff::get_range_tuple_id() const
{
  return range_tuple_id();
}

isl::multi_pw_aff multi_pw_aff::reset_range_tuple_id() const
{
  auto res = isl_multi_pw_aff_reset_range_tuple_id(copy());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::reset_tuple_id(isl::dim type) const
{
  auto res = isl_multi_pw_aff_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
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

isl::multi_pw_aff multi_pw_aff::scale(long v) const
{
  return this->scale(isl::val(ctx(), v));
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

isl::multi_pw_aff multi_pw_aff::scale_down(long v) const
{
  return this->scale_down(isl::val(ctx(), v));
}

isl::multi_pw_aff multi_pw_aff::set_at(int pos, isl::pw_aff el) const
{
  auto res = isl_multi_pw_aff_set_at(copy(), pos, el.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_pw_aff::set_at(int pos, const isl::union_pw_aff &el) const
{
  return isl::multi_union_pw_aff(*this).set_at(pos, el);
}

isl::multi_pw_aff multi_pw_aff::set_pw_aff(int pos, isl::pw_aff el) const
{
  auto res = isl_multi_pw_aff_set_pw_aff(copy(), pos, el.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::set_range_tuple(isl::id id) const
{
  auto res = isl_multi_pw_aff_set_range_tuple_id(copy(), id.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::set_range_tuple(const std::string &id) const
{
  return this->set_range_tuple(isl::id(ctx(), id));
}

isl::multi_union_pw_aff multi_pw_aff::set_union_pw_aff(int pos, const isl::union_pw_aff &el) const
{
  return isl::multi_union_pw_aff(*this).set_union_pw_aff(pos, el);
}

class size multi_pw_aff::size() const
{
  auto res = isl_multi_pw_aff_size(get());
  return manage(res);
}

isl::space multi_pw_aff::space() const
{
  auto res = isl_multi_pw_aff_get_space(get());
  return manage(res);
}

isl::space multi_pw_aff::get_space() const
{
  return space();
}

isl::multi_pw_aff multi_pw_aff::sub(isl::multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_sub(copy(), multi2.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_pw_aff::sub(const isl::multi_union_pw_aff &multi2) const
{
  return isl::multi_union_pw_aff(*this).sub(multi2);
}

isl::multi_pw_aff multi_pw_aff::sub(const isl::aff &multi2) const
{
  return this->sub(isl::multi_pw_aff(multi2));
}

isl::multi_pw_aff multi_pw_aff::sub(const isl::multi_aff &multi2) const
{
  return this->sub(isl::multi_pw_aff(multi2));
}

isl::multi_pw_aff multi_pw_aff::sub(const isl::pw_aff &multi2) const
{
  return this->sub(isl::multi_pw_aff(multi2));
}

isl::multi_pw_aff multi_pw_aff::sub(const isl::pw_multi_aff &multi2) const
{
  return this->sub(isl::multi_pw_aff(multi2));
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

isl::multi_union_pw_aff multi_pw_aff::union_add(const isl::multi_union_pw_aff &mupa2) const
{
  return isl::multi_union_pw_aff(*this).union_add(mupa2);
}

isl::multi_pw_aff multi_pw_aff::union_add(const isl::aff &mpa2) const
{
  return this->union_add(isl::multi_pw_aff(mpa2));
}

isl::multi_pw_aff multi_pw_aff::union_add(const isl::multi_aff &mpa2) const
{
  return this->union_add(isl::multi_pw_aff(mpa2));
}

isl::multi_pw_aff multi_pw_aff::union_add(const isl::pw_aff &mpa2) const
{
  return this->union_add(isl::multi_pw_aff(mpa2));
}

isl::multi_pw_aff multi_pw_aff::union_add(const isl::pw_multi_aff &mpa2) const
{
  return this->union_add(isl::multi_pw_aff(mpa2));
}

isl::multi_pw_aff multi_pw_aff::zero(isl::space space)
{
  auto res = isl_multi_pw_aff_zero(space.release());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const multi_pw_aff &obj)
{
  char *str = isl_multi_pw_aff_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::multi_union_pw_aff multi_union_pw_aff::add(isl::multi_union_pw_aff multi2) const
{
  auto res = isl_multi_union_pw_aff_add(copy(), multi2.release());
  return manage(res);
}

isl::union_pw_aff multi_union_pw_aff::at(int pos) const
{
  auto res = isl_multi_union_pw_aff_get_at(get(), pos);
  return manage(res);
}

isl::union_pw_aff multi_union_pw_aff::get_at(int pos) const
{
  return at(pos);
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

class size multi_union_pw_aff::dim(isl::dim type) const
{
  auto res = isl_multi_union_pw_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::union_set multi_union_pw_aff::domain() const
{
  auto res = isl_multi_union_pw_aff_domain(copy());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::flat_range_product(isl::multi_union_pw_aff multi2) const
{
  auto res = isl_multi_union_pw_aff_flat_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::from_union_map(isl::union_map umap)
{
  auto res = isl_multi_union_pw_aff_from_union_map(umap.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::gist(isl::union_set context) const
{
  auto res = isl_multi_union_pw_aff_gist(copy(), context.release());
  return manage(res);
}

boolean multi_union_pw_aff::has_range_tuple_id() const
{
  auto res = isl_multi_union_pw_aff_has_range_tuple_id(get());
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

boolean multi_union_pw_aff::involves_nan() const
{
  auto res = isl_multi_union_pw_aff_involves_nan(get());
  return manage(res);
}

isl::union_pw_aff_list multi_union_pw_aff::list() const
{
  auto res = isl_multi_union_pw_aff_get_list(get());
  return manage(res);
}

isl::union_pw_aff_list multi_union_pw_aff::get_list() const
{
  return list();
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

isl::multi_union_pw_aff multi_union_pw_aff::range_product(isl::multi_union_pw_aff multi2) const
{
  auto res = isl_multi_union_pw_aff_range_product(copy(), multi2.release());
  return manage(res);
}

isl::id multi_union_pw_aff::range_tuple_id() const
{
  auto res = isl_multi_union_pw_aff_get_range_tuple_id(get());
  return manage(res);
}

isl::id multi_union_pw_aff::get_range_tuple_id() const
{
  return range_tuple_id();
}

isl::multi_union_pw_aff multi_union_pw_aff::reset_range_tuple_id() const
{
  auto res = isl_multi_union_pw_aff_reset_range_tuple_id(copy());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::reset_tuple_id(isl::dim type) const
{
  auto res = isl_multi_union_pw_aff_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
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

isl::multi_union_pw_aff multi_union_pw_aff::scale(long v) const
{
  return this->scale(isl::val(ctx(), v));
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

isl::multi_union_pw_aff multi_union_pw_aff::scale_down(long v) const
{
  return this->scale_down(isl::val(ctx(), v));
}

isl::multi_union_pw_aff multi_union_pw_aff::set_at(int pos, isl::union_pw_aff el) const
{
  auto res = isl_multi_union_pw_aff_set_at(copy(), pos, el.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::set_range_tuple(isl::id id) const
{
  auto res = isl_multi_union_pw_aff_set_range_tuple_id(copy(), id.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::set_range_tuple(const std::string &id) const
{
  return this->set_range_tuple(isl::id(ctx(), id));
}

isl::multi_union_pw_aff multi_union_pw_aff::set_union_pw_aff(int pos, isl::union_pw_aff el) const
{
  auto res = isl_multi_union_pw_aff_set_union_pw_aff(copy(), pos, el.release());
  return manage(res);
}

class size multi_union_pw_aff::size() const
{
  auto res = isl_multi_union_pw_aff_size(get());
  return manage(res);
}

isl::space multi_union_pw_aff::space() const
{
  auto res = isl_multi_union_pw_aff_get_space(get());
  return manage(res);
}

isl::space multi_union_pw_aff::get_space() const
{
  return space();
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

inline std::ostream &operator<<(std::ostream &os, const multi_union_pw_aff &obj)
{
  char *str = isl_multi_union_pw_aff_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::multi_val multi_val::add(long v) const
{
  return this->add(isl::val(ctx(), v));
}

isl::val multi_val::at(int pos) const
{
  auto res = isl_multi_val_get_at(get(), pos);
  return manage(res);
}

isl::val multi_val::get_at(int pos) const
{
  return at(pos);
}

class size multi_val::dim(isl::dim type) const
{
  auto res = isl_multi_val_dim(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::multi_val multi_val::flat_range_product(isl::multi_val multi2) const
{
  auto res = isl_multi_val_flat_range_product(copy(), multi2.release());
  return manage(res);
}

boolean multi_val::has_range_tuple_id() const
{
  auto res = isl_multi_val_has_range_tuple_id(get());
  return manage(res);
}

boolean multi_val::involves_nan() const
{
  auto res = isl_multi_val_involves_nan(get());
  return manage(res);
}

isl::val_list multi_val::list() const
{
  auto res = isl_multi_val_get_list(get());
  return manage(res);
}

isl::val_list multi_val::get_list() const
{
  return list();
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

isl::multi_val multi_val::range_product(isl::multi_val multi2) const
{
  auto res = isl_multi_val_range_product(copy(), multi2.release());
  return manage(res);
}

isl::id multi_val::range_tuple_id() const
{
  auto res = isl_multi_val_get_range_tuple_id(get());
  return manage(res);
}

isl::id multi_val::get_range_tuple_id() const
{
  return range_tuple_id();
}

isl::multi_val multi_val::reset_range_tuple_id() const
{
  auto res = isl_multi_val_reset_range_tuple_id(copy());
  return manage(res);
}

isl::multi_val multi_val::reset_tuple_id(isl::dim type) const
{
  auto res = isl_multi_val_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
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

isl::multi_val multi_val::scale(long v) const
{
  return this->scale(isl::val(ctx(), v));
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

isl::multi_val multi_val::scale_down(long v) const
{
  return this->scale_down(isl::val(ctx(), v));
}

isl::multi_val multi_val::set_at(int pos, isl::val el) const
{
  auto res = isl_multi_val_set_at(copy(), pos, el.release());
  return manage(res);
}

isl::multi_val multi_val::set_at(int pos, long el) const
{
  return this->set_at(pos, isl::val(ctx(), el));
}

isl::multi_val multi_val::set_range_tuple(isl::id id) const
{
  auto res = isl_multi_val_set_range_tuple_id(copy(), id.release());
  return manage(res);
}

isl::multi_val multi_val::set_range_tuple(const std::string &id) const
{
  return this->set_range_tuple(isl::id(ctx(), id));
}

isl::multi_val multi_val::set_val(int pos, isl::val el) const
{
  auto res = isl_multi_val_set_val(copy(), pos, el.release());
  return manage(res);
}

isl::multi_val multi_val::set_val(int pos, long el) const
{
  return this->set_val(pos, isl::val(ctx(), el));
}

class size multi_val::size() const
{
  auto res = isl_multi_val_size(get());
  return manage(res);
}

isl::space multi_val::space() const
{
  auto res = isl_multi_val_get_space(get());
  return manage(res);
}

isl::space multi_val::get_space() const
{
  return space();
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

inline std::ostream &operator<<(std::ostream &os, const multi_val &obj)
{
  char *str = isl_multi_val_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

point::point(isl::space space)
{
  auto res = isl_point_zero(space.release());
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

isl::set point::add_constraint(const isl::constraint &constraint) const
{
  return isl::basic_set(*this).add_constraint(constraint);
}

isl::set point::add_dims(isl::dim type, unsigned int n) const
{
  return isl::basic_set(*this).add_dims(type, n);
}

isl::basic_set point::affine_hull() const
{
  return isl::basic_set(*this).affine_hull();
}

isl::set point::align_params(const isl::space &model) const
{
  return isl::basic_set(*this).align_params(model);
}

isl::basic_set point::apply(const isl::basic_map &bmap) const
{
  return isl::basic_set(*this).apply(bmap);
}

isl::set point::apply(const isl::map &map) const
{
  return isl::basic_set(*this).apply(map);
}

isl::union_set point::apply(const isl::union_map &umap) const
{
  return isl::basic_set(*this).apply(umap);
}

isl::pw_multi_aff point::as_pw_multi_aff() const
{
  return isl::basic_set(*this).as_pw_multi_aff();
}

isl::set point::as_set() const
{
  return isl::basic_set(*this).as_set();
}

isl::basic_set_list point::basic_set_list() const
{
  return isl::basic_set(*this).basic_set_list();
}

isl::set point::bind(const isl::multi_id &tuple) const
{
  return isl::basic_set(*this).bind(tuple);
}

isl::set point::coalesce() const
{
  return isl::basic_set(*this).coalesce();
}

isl::set point::complement() const
{
  return isl::basic_set(*this).complement();
}

isl::union_set point::compute_divs() const
{
  return isl::basic_set(*this).compute_divs();
}

boolean point::contains(const isl::space &space) const
{
  return isl::basic_set(*this).contains(space);
}

isl::basic_set point::convex_hull() const
{
  return isl::basic_set(*this).convex_hull();
}

isl::val point::coordinate_val(isl::dim type, int pos) const
{
  auto res = isl_point_get_coordinate_val(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::val point::get_coordinate_val(isl::dim type, int pos) const
{
  return coordinate_val(type, pos);
}

isl::basic_set point::detect_equalities() const
{
  return isl::basic_set(*this).detect_equalities();
}

class size point::dim(isl::dim type) const
{
  return isl::basic_set(*this).dim(type);
}

boolean point::dim_has_any_lower_bound(isl::dim type, unsigned int pos) const
{
  return isl::basic_set(*this).dim_has_any_lower_bound(type, pos);
}

isl::id point::dim_id(isl::dim type, unsigned int pos) const
{
  return isl::basic_set(*this).dim_id(type, pos);
}

isl::pw_aff point::dim_max(int pos) const
{
  return isl::basic_set(*this).dim_max(pos);
}

isl::val point::dim_max_val(int pos) const
{
  return isl::basic_set(*this).dim_max_val(pos);
}

isl::pw_aff point::dim_min(int pos) const
{
  return isl::basic_set(*this).dim_min(pos);
}

isl::val point::dim_min_val(int pos) const
{
  return isl::basic_set(*this).dim_min_val(pos);
}

std::string point::dim_name(isl::dim type, unsigned int pos) const
{
  return isl::basic_set(*this).dim_name(type, pos);
}

isl::aff point::div(int pos) const
{
  return isl::basic_set(*this).div(pos);
}

isl::set point::drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  return isl::basic_set(*this).drop_constraints_involving_dims(type, first, n);
}

isl::set point::eliminate(isl::dim type, unsigned int first, unsigned int n) const
{
  return isl::basic_set(*this).eliminate(type, first, n);
}

boolean point::every_set(const std::function<boolean(isl::set)> &test) const
{
  return isl::basic_set(*this).every_set(test);
}

isl::set point::extract_set(const isl::space &space) const
{
  return isl::basic_set(*this).extract_set(space);
}

int point::find_dim_by_id(isl::dim type, const isl::id &id) const
{
  return isl::basic_set(*this).find_dim_by_id(type, id);
}

int point::find_dim_by_id(isl::dim type, const std::string &id) const
{
  return this->find_dim_by_id(type, isl::id(ctx(), id));
}

isl::basic_set point::fix_si(isl::dim type, unsigned int pos, int value) const
{
  return isl::basic_set(*this).fix_si(type, pos, value);
}

isl::basic_set point::fix_val(isl::dim type, unsigned int pos, const isl::val &v) const
{
  return isl::basic_set(*this).fix_val(type, pos, v);
}

isl::basic_set point::fix_val(isl::dim type, unsigned int pos, long v) const
{
  return this->fix_val(type, pos, isl::val(ctx(), v));
}

isl::basic_set point::flatten() const
{
  return isl::basic_set(*this).flatten();
}

stat point::foreach_basic_set(const std::function<stat(isl::basic_set)> &fn) const
{
  return isl::basic_set(*this).foreach_basic_set(fn);
}

stat point::foreach_point(const std::function<stat(isl::point)> &fn) const
{
  return isl::basic_set(*this).foreach_point(fn);
}

stat point::foreach_set(const std::function<stat(isl::set)> &fn) const
{
  return isl::basic_set(*this).foreach_set(fn);
}

isl::basic_set point::gist(const isl::basic_set &context) const
{
  return isl::basic_set(*this).gist(context);
}

isl::set point::gist(const isl::set &context) const
{
  return isl::basic_set(*this).gist(context);
}

isl::union_set point::gist(const isl::union_set &context) const
{
  return isl::basic_set(*this).gist(context);
}

isl::set point::gist_params(const isl::set &context) const
{
  return isl::basic_set(*this).gist_params(context);
}

boolean point::has_equal_space(const isl::set &set2) const
{
  return isl::basic_set(*this).has_equal_space(set2);
}

isl::map point::identity() const
{
  return isl::basic_set(*this).identity();
}

isl::union_pw_multi_aff point::identity_union_pw_multi_aff() const
{
  return isl::basic_set(*this).identity_union_pw_multi_aff();
}

isl::pw_aff point::indicator_function() const
{
  return isl::basic_set(*this).indicator_function();
}

isl::set point::insert_dims(isl::dim type, unsigned int pos, unsigned int n) const
{
  return isl::basic_set(*this).insert_dims(type, pos, n);
}

isl::map point::insert_domain(const isl::space &domain) const
{
  return isl::basic_set(*this).insert_domain(domain);
}

isl::basic_set point::intersect(const isl::basic_set &bset2) const
{
  return isl::basic_set(*this).intersect(bset2);
}

isl::set point::intersect(const isl::set &set2) const
{
  return isl::basic_set(*this).intersect(set2);
}

isl::union_set point::intersect(const isl::union_set &uset2) const
{
  return isl::basic_set(*this).intersect(uset2);
}

isl::basic_set point::intersect_params(const isl::basic_set &bset2) const
{
  return isl::basic_set(*this).intersect_params(bset2);
}

isl::set point::intersect_params(const isl::set &params) const
{
  return isl::basic_set(*this).intersect_params(params);
}

boolean point::involves_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  return isl::basic_set(*this).involves_dims(type, first, n);
}

boolean point::involves_locals() const
{
  return isl::basic_set(*this).involves_locals();
}

boolean point::is_bounded() const
{
  return isl::basic_set(*this).is_bounded();
}

boolean point::is_disjoint(const isl::set &set2) const
{
  return isl::basic_set(*this).is_disjoint(set2);
}

boolean point::is_disjoint(const isl::union_set &uset2) const
{
  return isl::basic_set(*this).is_disjoint(uset2);
}

boolean point::is_empty() const
{
  return isl::basic_set(*this).is_empty();
}

boolean point::is_equal(const isl::basic_set &bset2) const
{
  return isl::basic_set(*this).is_equal(bset2);
}

boolean point::is_equal(const isl::set &set2) const
{
  return isl::basic_set(*this).is_equal(set2);
}

boolean point::is_equal(const isl::union_set &uset2) const
{
  return isl::basic_set(*this).is_equal(uset2);
}

boolean point::is_params() const
{
  return isl::basic_set(*this).is_params();
}

boolean point::is_singleton() const
{
  return isl::basic_set(*this).is_singleton();
}

boolean point::is_strict_subset(const isl::set &set2) const
{
  return isl::basic_set(*this).is_strict_subset(set2);
}

boolean point::is_strict_subset(const isl::union_set &uset2) const
{
  return isl::basic_set(*this).is_strict_subset(uset2);
}

boolean point::is_subset(const isl::basic_set &bset2) const
{
  return isl::basic_set(*this).is_subset(bset2);
}

boolean point::is_subset(const isl::set &set2) const
{
  return isl::basic_set(*this).is_subset(set2);
}

boolean point::is_subset(const isl::union_set &uset2) const
{
  return isl::basic_set(*this).is_subset(uset2);
}

boolean point::is_wrapping() const
{
  return isl::basic_set(*this).is_wrapping();
}

boolean point::isa_set() const
{
  return isl::basic_set(*this).isa_set();
}

isl::set point::lexmax() const
{
  return isl::basic_set(*this).lexmax();
}

isl::pw_multi_aff point::lexmax_pw_multi_aff() const
{
  return isl::basic_set(*this).lexmax_pw_multi_aff();
}

isl::set point::lexmin() const
{
  return isl::basic_set(*this).lexmin();
}

isl::pw_multi_aff point::lexmin_pw_multi_aff() const
{
  return isl::basic_set(*this).lexmin_pw_multi_aff();
}

isl::set point::lower_bound(const isl::multi_pw_aff &lower) const
{
  return isl::basic_set(*this).lower_bound(lower);
}

isl::set point::lower_bound(const isl::multi_val &lower) const
{
  return isl::basic_set(*this).lower_bound(lower);
}

isl::set point::lower_bound_si(isl::dim type, unsigned int pos, int value) const
{
  return isl::basic_set(*this).lower_bound_si(type, pos, value);
}

isl::set point::lower_bound_val(isl::dim type, unsigned int pos, const isl::val &value) const
{
  return isl::basic_set(*this).lower_bound_val(type, pos, value);
}

isl::set point::lower_bound_val(isl::dim type, unsigned int pos, long value) const
{
  return this->lower_bound_val(type, pos, isl::val(ctx(), value));
}

isl::multi_pw_aff point::max_multi_pw_aff() const
{
  return isl::basic_set(*this).max_multi_pw_aff();
}

isl::val point::max_val(const isl::aff &obj) const
{
  return isl::basic_set(*this).max_val(obj);
}

isl::multi_pw_aff point::min_multi_pw_aff() const
{
  return isl::basic_set(*this).min_multi_pw_aff();
}

isl::val point::min_val(const isl::aff &obj) const
{
  return isl::basic_set(*this).min_val(obj);
}

isl::multi_val point::multi_val() const
{
  auto res = isl_point_get_multi_val(get());
  return manage(res);
}

isl::multi_val point::get_multi_val() const
{
  return multi_val();
}

class size point::n_basic_set() const
{
  return isl::basic_set(*this).n_basic_set();
}

isl::basic_set point::params() const
{
  return isl::basic_set(*this).params();
}

isl::val point::plain_get_val_if_fixed(isl::dim type, unsigned int pos) const
{
  return isl::basic_set(*this).plain_get_val_if_fixed(type, pos);
}

isl::multi_val point::plain_multi_val_if_fixed() const
{
  return isl::basic_set(*this).plain_multi_val_if_fixed();
}

isl::basic_set point::polyhedral_hull() const
{
  return isl::basic_set(*this).polyhedral_hull();
}

isl::set point::preimage(const isl::multi_aff &ma) const
{
  return isl::basic_set(*this).preimage(ma);
}

isl::set point::preimage(const isl::multi_pw_aff &mpa) const
{
  return isl::basic_set(*this).preimage(mpa);
}

isl::set point::preimage(const isl::pw_multi_aff &pma) const
{
  return isl::basic_set(*this).preimage(pma);
}

isl::union_set point::preimage(const isl::union_pw_multi_aff &upma) const
{
  return isl::basic_set(*this).preimage(upma);
}

isl::set point::product(const isl::set &set2) const
{
  return isl::basic_set(*this).product(set2);
}

isl::basic_set point::project_out(isl::dim type, unsigned int first, unsigned int n) const
{
  return isl::basic_set(*this).project_out(type, first, n);
}

isl::set point::project_out_all_params() const
{
  return isl::basic_set(*this).project_out_all_params();
}

isl::set point::project_out_param(const isl::id &id) const
{
  return isl::basic_set(*this).project_out_param(id);
}

isl::set point::project_out_param(const std::string &id) const
{
  return this->project_out_param(isl::id(ctx(), id));
}

isl::set point::project_out_param(const isl::id_list &list) const
{
  return isl::basic_set(*this).project_out_param(list);
}

isl::pw_multi_aff point::pw_multi_aff_on_domain(const isl::multi_val &mv) const
{
  return isl::basic_set(*this).pw_multi_aff_on_domain(mv);
}

isl::set point::remove_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  return isl::basic_set(*this).remove_dims(type, first, n);
}

isl::set point::remove_divs() const
{
  return isl::basic_set(*this).remove_divs();
}

isl::set point::remove_redundancies() const
{
  return isl::basic_set(*this).remove_redundancies();
}

isl::set point::reset_tuple_id() const
{
  return isl::basic_set(*this).reset_tuple_id();
}

isl::basic_set point::sample() const
{
  return isl::basic_set(*this).sample();
}

isl::point point::sample_point() const
{
  return isl::basic_set(*this).sample_point();
}

isl::set point::set_dim_id(isl::dim type, unsigned int pos, const isl::id &id) const
{
  return isl::basic_set(*this).set_dim_id(type, pos, id);
}

isl::set point::set_dim_id(isl::dim type, unsigned int pos, const std::string &id) const
{
  return this->set_dim_id(type, pos, isl::id(ctx(), id));
}

isl::set_list point::set_list() const
{
  return isl::basic_set(*this).set_list();
}

isl::set point::set_tuple_id(const isl::id &id) const
{
  return isl::basic_set(*this).set_tuple_id(id);
}

isl::set point::set_tuple_id(const std::string &id) const
{
  return this->set_tuple_id(isl::id(ctx(), id));
}

isl::fixed_box point::simple_fixed_box_hull() const
{
  return isl::basic_set(*this).simple_fixed_box_hull();
}

isl::basic_set point::simple_hull() const
{
  return isl::basic_set(*this).simple_hull();
}

isl::space point::space() const
{
  return isl::basic_set(*this).space();
}

isl::val point::stride(int pos) const
{
  return isl::basic_set(*this).stride(pos);
}

isl::set point::subtract(const isl::set &set2) const
{
  return isl::basic_set(*this).subtract(set2);
}

isl::union_set point::subtract(const isl::union_set &uset2) const
{
  return isl::basic_set(*this).subtract(uset2);
}

isl::basic_set_list point::to_list() const
{
  return isl::basic_set(*this).to_list();
}

isl::set point::to_set() const
{
  auto res = isl_point_to_set(copy());
  return manage(res);
}

isl::union_set point::to_union_set() const
{
  return isl::basic_set(*this).to_union_set();
}

isl::map point::translation() const
{
  return isl::basic_set(*this).translation();
}

class size point::tuple_dim() const
{
  return isl::basic_set(*this).tuple_dim();
}

isl::id point::tuple_id() const
{
  return isl::basic_set(*this).tuple_id();
}

std::string point::tuple_name() const
{
  return isl::basic_set(*this).tuple_name();
}

isl::set point::unbind_params(const isl::multi_id &tuple) const
{
  return isl::basic_set(*this).unbind_params(tuple);
}

isl::map point::unbind_params_insert_domain(const isl::multi_id &domain) const
{
  return isl::basic_set(*this).unbind_params_insert_domain(domain);
}

isl::set point::unite(const isl::basic_set &bset2) const
{
  return isl::basic_set(*this).unite(bset2);
}

isl::set point::unite(const isl::set &set2) const
{
  return isl::basic_set(*this).unite(set2);
}

isl::union_set point::unite(const isl::union_set &uset2) const
{
  return isl::basic_set(*this).unite(uset2);
}

isl::basic_set point::unshifted_simple_hull() const
{
  return isl::basic_set(*this).unshifted_simple_hull();
}

isl::map point::unwrap() const
{
  return isl::basic_set(*this).unwrap();
}

isl::set point::upper_bound(const isl::multi_pw_aff &upper) const
{
  return isl::basic_set(*this).upper_bound(upper);
}

isl::set point::upper_bound(const isl::multi_val &upper) const
{
  return isl::basic_set(*this).upper_bound(upper);
}

isl::set point::upper_bound_val(isl::dim type, unsigned int pos, const isl::val &value) const
{
  return isl::basic_set(*this).upper_bound_val(type, pos, value);
}

isl::set point::upper_bound_val(isl::dim type, unsigned int pos, long value) const
{
  return this->upper_bound_val(type, pos, isl::val(ctx(), value));
}

inline std::ostream &operator<<(std::ostream &os, const point &obj)
{
  char *str = isl_point_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::multi_pw_aff pw_aff::add(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).add(multi2);
}

isl::multi_union_pw_aff pw_aff::add(const isl::multi_union_pw_aff &multi2) const
{
  return isl::union_pw_aff(*this).add(multi2);
}

isl::pw_aff pw_aff::add(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_add(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_multi_aff pw_aff::add(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_multi_aff(*this).add(pma2);
}

isl::union_pw_aff pw_aff::add(const isl::union_pw_aff &upa2) const
{
  return isl::union_pw_aff(*this).add(upa2);
}

isl::union_pw_multi_aff pw_aff::add(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_aff(*this).add(upma2);
}

isl::pw_aff pw_aff::add(const isl::aff &pwaff2) const
{
  return this->add(isl::pw_aff(pwaff2));
}

isl::pw_aff pw_aff::add_constant(isl::val v) const
{
  auto res = isl_pw_aff_add_constant_val(copy(), v.release());
  return manage(res);
}

isl::pw_aff pw_aff::add_constant(long v) const
{
  return this->add_constant(isl::val(ctx(), v));
}

isl::pw_multi_aff pw_aff::add_constant(const isl::multi_val &mv) const
{
  return isl::pw_multi_aff(*this).add_constant(mv);
}

isl::pw_aff pw_aff::add_dims(isl::dim type, unsigned int n) const
{
  auto res = isl_pw_aff_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::union_pw_multi_aff pw_aff::add_pw_multi_aff(const isl::pw_multi_aff &pma) const
{
  return isl::union_pw_aff(*this).add_pw_multi_aff(pma);
}

isl::union_pw_multi_aff pw_aff::apply(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_aff(*this).apply(upma2);
}

isl::aff pw_aff::as_aff() const
{
  auto res = isl_pw_aff_as_aff(copy());
  return manage(res);
}

isl::map pw_aff::as_map() const
{
  auto res = isl_pw_aff_as_map(copy());
  return manage(res);
}

isl::multi_aff pw_aff::as_multi_aff() const
{
  return isl::pw_multi_aff(*this).as_multi_aff();
}

isl::multi_union_pw_aff pw_aff::as_multi_union_pw_aff() const
{
  return isl::union_pw_aff(*this).as_multi_union_pw_aff();
}

isl::pw_multi_aff pw_aff::as_pw_multi_aff() const
{
  return isl::union_pw_aff(*this).as_pw_multi_aff();
}

isl::set pw_aff::as_set() const
{
  return isl::pw_multi_aff(*this).as_set();
}

isl::union_map pw_aff::as_union_map() const
{
  return isl::union_pw_aff(*this).as_union_map();
}

isl::pw_aff pw_aff::at(int pos) const
{
  return isl::pw_multi_aff(*this).at(pos);
}

isl::set pw_aff::bind(const isl::multi_id &tuple) const
{
  return isl::multi_pw_aff(*this).bind(tuple);
}

isl::set pw_aff::bind(isl::id id) const
{
  auto res = isl_pw_aff_bind_id(copy(), id.release());
  return manage(res);
}

isl::set pw_aff::bind(const std::string &id) const
{
  return this->bind(isl::id(ctx(), id));
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

class size pw_aff::dim(isl::dim type) const
{
  return isl::pw_multi_aff(*this).dim(type);
}

isl::id pw_aff::dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_pw_aff_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::id pw_aff::get_dim_id(isl::dim type, unsigned int pos) const
{
  return dim_id(type, pos);
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

isl::space pw_aff::domain_space() const
{
  auto res = isl_pw_aff_get_domain_space(get());
  return manage(res);
}

isl::space pw_aff::get_domain_space() const
{
  return domain_space();
}

isl::pw_multi_aff pw_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  return isl::pw_multi_aff(*this).drop_dims(type, first, n);
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

isl::pw_multi_aff pw_aff::extract_pw_multi_aff(const isl::space &space) const
{
  return isl::union_pw_aff(*this).extract_pw_multi_aff(space);
}

isl::multi_pw_aff pw_aff::flat_range_product(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).flat_range_product(multi2);
}

isl::multi_union_pw_aff pw_aff::flat_range_product(const isl::multi_union_pw_aff &multi2) const
{
  return isl::union_pw_aff(*this).flat_range_product(multi2);
}

isl::pw_multi_aff pw_aff::flat_range_product(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_multi_aff(*this).flat_range_product(pma2);
}

isl::union_pw_multi_aff pw_aff::flat_range_product(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_aff(*this).flat_range_product(upma2);
}

isl::pw_aff pw_aff::floor() const
{
  auto res = isl_pw_aff_floor(copy());
  return manage(res);
}

stat pw_aff::foreach_piece(const std::function<stat(isl::set, isl::aff)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::set, isl::aff)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_set *arg_0, isl_aff *arg_1, void *arg_2) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_2);
    auto ret = (data->func)(manage(arg_0), manage(arg_1));
    return ret.release();
  };
  auto res = isl_pw_aff_foreach_piece(get(), fn_lambda, &fn_data);
  return manage(res);
}

stat pw_aff::foreach_piece(const std::function<stat(isl::set, isl::multi_aff)> &fn) const
{
  return isl::pw_multi_aff(*this).foreach_piece(fn);
}

stat pw_aff::foreach_pw_aff(const std::function<stat(isl::pw_aff)> &fn) const
{
  return isl::union_pw_aff(*this).foreach_pw_aff(fn);
}

isl::set pw_aff::ge_set(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_ge_set(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::gist(isl::set context) const
{
  auto res = isl_pw_aff_gist(copy(), context.release());
  return manage(res);
}

isl::union_pw_aff pw_aff::gist(const isl::union_set &context) const
{
  return isl::union_pw_aff(*this).gist(context);
}

isl::pw_aff pw_aff::gist(const isl::basic_set &context) const
{
  return this->gist(isl::set(context));
}

isl::pw_aff pw_aff::gist(const isl::point &context) const
{
  return this->gist(isl::set(context));
}

isl::set pw_aff::gt_set(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_gt_set(copy(), pwaff2.release());
  return manage(res);
}

boolean pw_aff::has_range_tuple_id() const
{
  return isl::pw_multi_aff(*this).has_range_tuple_id();
}

isl::multi_pw_aff pw_aff::identity() const
{
  return isl::pw_multi_aff(*this).identity();
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

isl::union_pw_aff pw_aff::intersect_domain(const isl::space &space) const
{
  return isl::union_pw_aff(*this).intersect_domain(space);
}

isl::union_pw_aff pw_aff::intersect_domain(const isl::union_set &uset) const
{
  return isl::union_pw_aff(*this).intersect_domain(uset);
}

isl::pw_aff pw_aff::intersect_domain(const isl::basic_set &set) const
{
  return this->intersect_domain(isl::set(set));
}

isl::pw_aff pw_aff::intersect_domain(const isl::point &set) const
{
  return this->intersect_domain(isl::set(set));
}

isl::union_pw_aff pw_aff::intersect_domain_wrapped_domain(const isl::union_set &uset) const
{
  return isl::union_pw_aff(*this).intersect_domain_wrapped_domain(uset);
}

isl::union_pw_aff pw_aff::intersect_domain_wrapped_range(const isl::union_set &uset) const
{
  return isl::union_pw_aff(*this).intersect_domain_wrapped_range(uset);
}

isl::pw_aff pw_aff::intersect_params(isl::set set) const
{
  auto res = isl_pw_aff_intersect_params(copy(), set.release());
  return manage(res);
}

boolean pw_aff::involves_locals() const
{
  return isl::pw_multi_aff(*this).involves_locals();
}

boolean pw_aff::involves_nan() const
{
  return isl::multi_pw_aff(*this).involves_nan();
}

boolean pw_aff::involves_param(const isl::id &id) const
{
  return isl::pw_multi_aff(*this).involves_param(id);
}

boolean pw_aff::involves_param(const std::string &id) const
{
  return this->involves_param(isl::id(ctx(), id));
}

boolean pw_aff::involves_param(const isl::id_list &list) const
{
  return isl::pw_multi_aff(*this).involves_param(list);
}

boolean pw_aff::is_cst() const
{
  auto res = isl_pw_aff_is_cst(get());
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

boolean pw_aff::isa_multi_aff() const
{
  return isl::pw_multi_aff(*this).isa_multi_aff();
}

boolean pw_aff::isa_pw_multi_aff() const
{
  return isl::union_pw_aff(*this).isa_pw_multi_aff();
}

isl::set pw_aff::le_set(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_le_set(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff_list pw_aff::list() const
{
  return isl::multi_pw_aff(*this).list();
}

isl::set pw_aff::lt_set(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_lt_set(copy(), pwaff2.release());
  return manage(res);
}

isl::multi_pw_aff pw_aff::max(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).max(multi2);
}

isl::pw_aff pw_aff::max(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_max(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::max(const isl::aff &pwaff2) const
{
  return this->max(isl::pw_aff(pwaff2));
}

isl::multi_val pw_aff::max_multi_val() const
{
  return isl::pw_multi_aff(*this).max_multi_val();
}

isl::multi_pw_aff pw_aff::min(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).min(multi2);
}

isl::pw_aff pw_aff::min(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_min(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::min(const isl::aff &pwaff2) const
{
  return this->min(isl::pw_aff(pwaff2));
}

isl::multi_val pw_aff::min_multi_val() const
{
  return isl::pw_multi_aff(*this).min_multi_val();
}

isl::pw_aff pw_aff::mod(isl::val mod) const
{
  auto res = isl_pw_aff_mod_val(copy(), mod.release());
  return manage(res);
}

isl::pw_aff pw_aff::mod(long mod) const
{
  return this->mod(isl::val(ctx(), mod));
}

isl::pw_aff pw_aff::mul(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_mul(copy(), pwaff2.release());
  return manage(res);
}

class size pw_aff::n_piece() const
{
  return isl::pw_multi_aff(*this).n_piece();
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

isl::pw_aff pw_aff::param_on_domain(isl::set domain, isl::id id)
{
  auto res = isl_pw_aff_param_on_domain_id(domain.release(), id.release());
  return manage(res);
}

boolean pw_aff::plain_is_empty() const
{
  return isl::union_pw_aff(*this).plain_is_empty();
}

boolean pw_aff::plain_is_equal(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).plain_is_equal(multi2);
}

boolean pw_aff::plain_is_equal(const isl::multi_union_pw_aff &multi2) const
{
  return isl::union_pw_aff(*this).plain_is_equal(multi2);
}

isl::pw_multi_aff pw_aff::preimage_domain_wrapped_domain(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_multi_aff(*this).preimage_domain_wrapped_domain(pma2);
}

isl::union_pw_multi_aff pw_aff::preimage_domain_wrapped_domain(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_aff(*this).preimage_domain_wrapped_domain(upma2);
}

isl::multi_pw_aff pw_aff::product(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).product(multi2);
}

isl::pw_multi_aff pw_aff::product(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_multi_aff(*this).product(pma2);
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

isl::union_pw_aff pw_aff::pullback(const isl::union_pw_multi_aff &upma) const
{
  return isl::union_pw_aff(*this).pullback(upma);
}

isl::pw_multi_aff_list pw_aff::pw_multi_aff_list() const
{
  return isl::union_pw_aff(*this).pw_multi_aff_list();
}

isl::pw_multi_aff pw_aff::range_factor_domain() const
{
  return isl::pw_multi_aff(*this).range_factor_domain();
}

isl::pw_multi_aff pw_aff::range_factor_range() const
{
  return isl::pw_multi_aff(*this).range_factor_range();
}

isl::multi_pw_aff pw_aff::range_product(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).range_product(multi2);
}

isl::multi_union_pw_aff pw_aff::range_product(const isl::multi_union_pw_aff &multi2) const
{
  return isl::union_pw_aff(*this).range_product(multi2);
}

isl::pw_multi_aff pw_aff::range_product(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_multi_aff(*this).range_product(pma2);
}

isl::union_pw_multi_aff pw_aff::range_product(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_aff(*this).range_product(upma2);
}

isl::id pw_aff::range_tuple_id() const
{
  return isl::pw_multi_aff(*this).range_tuple_id();
}

isl::multi_pw_aff pw_aff::reset_range_tuple_id() const
{
  return isl::multi_pw_aff(*this).reset_range_tuple_id();
}

isl::multi_pw_aff pw_aff::reset_tuple_id(isl::dim type) const
{
  return isl::multi_pw_aff(*this).reset_tuple_id(type);
}

isl::multi_pw_aff pw_aff::scale(const isl::multi_val &mv) const
{
  return isl::multi_pw_aff(*this).scale(mv);
}

isl::pw_aff pw_aff::scale(isl::val v) const
{
  auto res = isl_pw_aff_scale_val(copy(), v.release());
  return manage(res);
}

isl::pw_aff pw_aff::scale(long v) const
{
  return this->scale(isl::val(ctx(), v));
}

isl::multi_pw_aff pw_aff::scale_down(const isl::multi_val &mv) const
{
  return isl::multi_pw_aff(*this).scale_down(mv);
}

isl::pw_aff pw_aff::scale_down(isl::val f) const
{
  auto res = isl_pw_aff_scale_down_val(copy(), f.release());
  return manage(res);
}

isl::pw_aff pw_aff::scale_down(long f) const
{
  return this->scale_down(isl::val(ctx(), f));
}

isl::multi_pw_aff pw_aff::set_at(int pos, const isl::pw_aff &el) const
{
  return isl::pw_multi_aff(*this).set_at(pos, el);
}

isl::multi_union_pw_aff pw_aff::set_at(int pos, const isl::union_pw_aff &el) const
{
  return isl::union_pw_aff(*this).set_at(pos, el);
}

isl::multi_pw_aff pw_aff::set_pw_aff(int pos, const isl::pw_aff &el) const
{
  return isl::pw_multi_aff(*this).set_pw_aff(pos, el);
}

isl::pw_multi_aff pw_aff::set_pw_aff(unsigned int pos, const isl::pw_aff &pa) const
{
  return isl::pw_multi_aff(*this).set_pw_aff(pos, pa);
}

isl::pw_multi_aff pw_aff::set_range_tuple(const isl::id &id) const
{
  return isl::pw_multi_aff(*this).set_range_tuple(id);
}

isl::pw_multi_aff pw_aff::set_range_tuple(const std::string &id) const
{
  return this->set_range_tuple(isl::id(ctx(), id));
}

isl::pw_aff pw_aff::set_tuple_id(isl::dim type, isl::id id) const
{
  auto res = isl_pw_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::pw_aff pw_aff::set_tuple_id(isl::dim type, const std::string &id) const
{
  return this->set_tuple_id(type, isl::id(ctx(), id));
}

isl::multi_union_pw_aff pw_aff::set_union_pw_aff(int pos, const isl::union_pw_aff &el) const
{
  return isl::union_pw_aff(*this).set_union_pw_aff(pos, el);
}

class size pw_aff::size() const
{
  return isl::multi_pw_aff(*this).size();
}

isl::space pw_aff::space() const
{
  auto res = isl_pw_aff_get_space(get());
  return manage(res);
}

isl::space pw_aff::get_space() const
{
  return space();
}

isl::multi_pw_aff pw_aff::sub(const isl::multi_pw_aff &multi2) const
{
  return isl::pw_multi_aff(*this).sub(multi2);
}

isl::multi_union_pw_aff pw_aff::sub(const isl::multi_union_pw_aff &multi2) const
{
  return isl::union_pw_aff(*this).sub(multi2);
}

isl::pw_aff pw_aff::sub(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_sub(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_multi_aff pw_aff::sub(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_multi_aff(*this).sub(pma2);
}

isl::union_pw_aff pw_aff::sub(const isl::union_pw_aff &upa2) const
{
  return isl::union_pw_aff(*this).sub(upa2);
}

isl::union_pw_multi_aff pw_aff::sub(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_aff(*this).sub(upma2);
}

isl::pw_aff pw_aff::sub(const isl::aff &pwaff2) const
{
  return this->sub(isl::pw_aff(pwaff2));
}

isl::pw_aff pw_aff::subtract_domain(isl::set set) const
{
  auto res = isl_pw_aff_subtract_domain(copy(), set.release());
  return manage(res);
}

isl::union_pw_aff pw_aff::subtract_domain(const isl::space &space) const
{
  return isl::union_pw_aff(*this).subtract_domain(space);
}

isl::union_pw_aff pw_aff::subtract_domain(const isl::union_set &uset) const
{
  return isl::union_pw_aff(*this).subtract_domain(uset);
}

isl::pw_aff pw_aff::subtract_domain(const isl::basic_set &set) const
{
  return this->subtract_domain(isl::set(set));
}

isl::pw_aff pw_aff::subtract_domain(const isl::point &set) const
{
  return this->subtract_domain(isl::set(set));
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

isl::pw_aff_list pw_aff::to_list() const
{
  auto res = isl_pw_aff_to_list(copy());
  return manage(res);
}

isl::multi_pw_aff pw_aff::to_multi_pw_aff() const
{
  return isl::pw_multi_aff(*this).to_multi_pw_aff();
}

isl::union_pw_aff pw_aff::to_union_pw_aff() const
{
  auto res = isl_pw_aff_to_union_pw_aff(copy());
  return manage(res);
}

isl::union_pw_multi_aff pw_aff::to_union_pw_multi_aff() const
{
  return isl::pw_multi_aff(*this).to_union_pw_multi_aff();
}

isl::id pw_aff::tuple_id(isl::dim type) const
{
  auto res = isl_pw_aff_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::id pw_aff::get_tuple_id(isl::dim type) const
{
  return tuple_id(type);
}

isl::multi_pw_aff pw_aff::unbind_params_insert_domain(const isl::multi_id &domain) const
{
  return isl::pw_multi_aff(*this).unbind_params_insert_domain(domain);
}

isl::multi_pw_aff pw_aff::union_add(const isl::multi_pw_aff &mpa2) const
{
  return isl::pw_multi_aff(*this).union_add(mpa2);
}

isl::multi_union_pw_aff pw_aff::union_add(const isl::multi_union_pw_aff &mupa2) const
{
  return isl::union_pw_aff(*this).union_add(mupa2);
}

isl::pw_aff pw_aff::union_add(isl::pw_aff pwaff2) const
{
  auto res = isl_pw_aff_union_add(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_multi_aff pw_aff::union_add(const isl::pw_multi_aff &pma2) const
{
  return isl::pw_multi_aff(*this).union_add(pma2);
}

isl::union_pw_aff pw_aff::union_add(const isl::union_pw_aff &upa2) const
{
  return isl::union_pw_aff(*this).union_add(upa2);
}

isl::union_pw_multi_aff pw_aff::union_add(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_aff(*this).union_add(upma2);
}

isl::pw_aff pw_aff::union_add(const isl::aff &pwaff2) const
{
  return this->union_add(isl::pw_aff(pwaff2));
}

isl::pw_aff pw_aff::var_on_domain(isl::local_space ls, isl::dim type, unsigned int pos)
{
  auto res = isl_pw_aff_var_on_domain(ls.release(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const pw_aff &obj)
{
  char *str = isl_pw_aff_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

pw_aff_list::pw_aff_list(isl::ctx ctx, int n)
{
  auto res = isl_pw_aff_list_alloc(ctx.release(), n);
  ptr = res;
}

pw_aff_list::pw_aff_list(isl::pw_aff el)
{
  auto res = isl_pw_aff_list_from_pw_aff(el.release());
  ptr = res;
}

pw_aff_list::pw_aff_list(isl::ctx ctx, const std::string &str)
{
  auto res = isl_pw_aff_list_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

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

isl::pw_aff_list pw_aff_list::add(isl::pw_aff el) const
{
  auto res = isl_pw_aff_list_add(copy(), el.release());
  return manage(res);
}

isl::pw_aff pw_aff_list::at(int index) const
{
  auto res = isl_pw_aff_list_get_at(get(), index);
  return manage(res);
}

isl::pw_aff pw_aff_list::get_at(int index) const
{
  return at(index);
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

stat pw_aff_list::foreach(const std::function<stat(isl::pw_aff)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::pw_aff)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_pw_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_pw_aff_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

isl::pw_aff_list pw_aff_list::insert(unsigned int pos, isl::pw_aff el) const
{
  auto res = isl_pw_aff_list_insert(copy(), pos, el.release());
  return manage(res);
}

class size pw_aff_list::size() const
{
  auto res = isl_pw_aff_list_size(get());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const pw_aff_list &obj)
{
  char *str = isl_pw_aff_list_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::multi_pw_aff pw_multi_aff::add(const isl::multi_pw_aff &multi2) const
{
  return isl::multi_pw_aff(*this).add(multi2);
}

isl::multi_union_pw_aff pw_multi_aff::add(const isl::multi_union_pw_aff &multi2) const
{
  return isl::multi_pw_aff(*this).add(multi2);
}

isl::pw_multi_aff pw_multi_aff::add(isl::pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_add(copy(), pma2.release());
  return manage(res);
}

isl::union_pw_multi_aff pw_multi_aff::add(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_multi_aff(*this).add(upma2);
}

isl::pw_multi_aff pw_multi_aff::add(const isl::multi_aff &pma2) const
{
  return this->add(isl::pw_multi_aff(pma2));
}

isl::pw_multi_aff pw_multi_aff::add(const isl::pw_aff &pma2) const
{
  return this->add(isl::pw_multi_aff(pma2));
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

isl::pw_multi_aff pw_multi_aff::add_constant(long v) const
{
  return this->add_constant(isl::val(ctx(), v));
}

isl::union_pw_multi_aff pw_multi_aff::add_pw_multi_aff(const isl::pw_multi_aff &pma) const
{
  return isl::union_pw_multi_aff(*this).add_pw_multi_aff(pma);
}

isl::union_pw_multi_aff pw_multi_aff::apply(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_multi_aff(*this).apply(upma2);
}

isl::map pw_multi_aff::as_map() const
{
  auto res = isl_pw_multi_aff_as_map(copy());
  return manage(res);
}

isl::multi_aff pw_multi_aff::as_multi_aff() const
{
  auto res = isl_pw_multi_aff_as_multi_aff(copy());
  return manage(res);
}

isl::multi_union_pw_aff pw_multi_aff::as_multi_union_pw_aff() const
{
  return isl::union_pw_multi_aff(*this).as_multi_union_pw_aff();
}

isl::pw_multi_aff pw_multi_aff::as_pw_multi_aff() const
{
  return isl::union_pw_multi_aff(*this).as_pw_multi_aff();
}

isl::set pw_multi_aff::as_set() const
{
  auto res = isl_pw_multi_aff_as_set(copy());
  return manage(res);
}

isl::union_map pw_multi_aff::as_union_map() const
{
  return isl::union_pw_multi_aff(*this).as_union_map();
}

isl::pw_aff pw_multi_aff::at(int pos) const
{
  auto res = isl_pw_multi_aff_get_at(get(), pos);
  return manage(res);
}

isl::pw_aff pw_multi_aff::get_at(int pos) const
{
  return at(pos);
}

isl::set pw_multi_aff::bind(const isl::multi_id &tuple) const
{
  return isl::multi_pw_aff(*this).bind(tuple);
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

class size pw_multi_aff::dim(isl::dim type) const
{
  auto res = isl_pw_multi_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
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

isl::pw_multi_aff pw_multi_aff::extract_pw_multi_aff(const isl::space &space) const
{
  return isl::union_pw_multi_aff(*this).extract_pw_multi_aff(space);
}

isl::multi_pw_aff pw_multi_aff::flat_range_product(const isl::multi_pw_aff &multi2) const
{
  return isl::multi_pw_aff(*this).flat_range_product(multi2);
}

isl::multi_union_pw_aff pw_multi_aff::flat_range_product(const isl::multi_union_pw_aff &multi2) const
{
  return isl::multi_pw_aff(*this).flat_range_product(multi2);
}

isl::pw_multi_aff pw_multi_aff::flat_range_product(isl::pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_flat_range_product(copy(), pma2.release());
  return manage(res);
}

isl::union_pw_multi_aff pw_multi_aff::flat_range_product(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_multi_aff(*this).flat_range_product(upma2);
}

isl::pw_multi_aff pw_multi_aff::flat_range_product(const isl::multi_aff &pma2) const
{
  return this->flat_range_product(isl::pw_multi_aff(pma2));
}

isl::pw_multi_aff pw_multi_aff::flat_range_product(const isl::pw_aff &pma2) const
{
  return this->flat_range_product(isl::pw_multi_aff(pma2));
}

stat pw_multi_aff::foreach_piece(const std::function<stat(isl::set, isl::multi_aff)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::set, isl::multi_aff)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_set *arg_0, isl_multi_aff *arg_1, void *arg_2) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_2);
    auto ret = (data->func)(manage(arg_0), manage(arg_1));
    return ret.release();
  };
  auto res = isl_pw_multi_aff_foreach_piece(get(), fn_lambda, &fn_data);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::from_map(isl::map map)
{
  auto res = isl_pw_multi_aff_from_map(map.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::gist(isl::set set) const
{
  auto res = isl_pw_multi_aff_gist(copy(), set.release());
  return manage(res);
}

isl::union_pw_multi_aff pw_multi_aff::gist(const isl::union_set &context) const
{
  return isl::union_pw_multi_aff(*this).gist(context);
}

isl::pw_multi_aff pw_multi_aff::gist(const isl::basic_set &set) const
{
  return this->gist(isl::set(set));
}

isl::pw_multi_aff pw_multi_aff::gist(const isl::point &set) const
{
  return this->gist(isl::set(set));
}

boolean pw_multi_aff::has_range_tuple_id() const
{
  auto res = isl_pw_multi_aff_has_range_tuple_id(get());
  return manage(res);
}

isl::multi_pw_aff pw_multi_aff::identity() const
{
  return isl::multi_pw_aff(*this).identity();
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

isl::union_pw_multi_aff pw_multi_aff::intersect_domain(const isl::space &space) const
{
  return isl::union_pw_multi_aff(*this).intersect_domain(space);
}

isl::union_pw_multi_aff pw_multi_aff::intersect_domain(const isl::union_set &uset) const
{
  return isl::union_pw_multi_aff(*this).intersect_domain(uset);
}

isl::pw_multi_aff pw_multi_aff::intersect_domain(const isl::basic_set &set) const
{
  return this->intersect_domain(isl::set(set));
}

isl::pw_multi_aff pw_multi_aff::intersect_domain(const isl::point &set) const
{
  return this->intersect_domain(isl::set(set));
}

isl::union_pw_multi_aff pw_multi_aff::intersect_domain_wrapped_domain(const isl::union_set &uset) const
{
  return isl::union_pw_multi_aff(*this).intersect_domain_wrapped_domain(uset);
}

isl::union_pw_multi_aff pw_multi_aff::intersect_domain_wrapped_range(const isl::union_set &uset) const
{
  return isl::union_pw_multi_aff(*this).intersect_domain_wrapped_range(uset);
}

isl::pw_multi_aff pw_multi_aff::intersect_params(isl::set set) const
{
  auto res = isl_pw_multi_aff_intersect_params(copy(), set.release());
  return manage(res);
}

boolean pw_multi_aff::involves_locals() const
{
  auto res = isl_pw_multi_aff_involves_locals(get());
  return manage(res);
}

boolean pw_multi_aff::involves_nan() const
{
  return isl::multi_pw_aff(*this).involves_nan();
}

boolean pw_multi_aff::involves_param(const isl::id &id) const
{
  return isl::multi_pw_aff(*this).involves_param(id);
}

boolean pw_multi_aff::involves_param(const std::string &id) const
{
  return this->involves_param(isl::id(ctx(), id));
}

boolean pw_multi_aff::involves_param(const isl::id_list &list) const
{
  return isl::multi_pw_aff(*this).involves_param(list);
}

boolean pw_multi_aff::isa_multi_aff() const
{
  auto res = isl_pw_multi_aff_isa_multi_aff(get());
  return manage(res);
}

boolean pw_multi_aff::isa_pw_multi_aff() const
{
  return isl::union_pw_multi_aff(*this).isa_pw_multi_aff();
}

isl::pw_aff_list pw_multi_aff::list() const
{
  return isl::multi_pw_aff(*this).list();
}

isl::multi_pw_aff pw_multi_aff::max(const isl::multi_pw_aff &multi2) const
{
  return isl::multi_pw_aff(*this).max(multi2);
}

isl::multi_val pw_multi_aff::max_multi_val() const
{
  auto res = isl_pw_multi_aff_max_multi_val(copy());
  return manage(res);
}

isl::multi_pw_aff pw_multi_aff::min(const isl::multi_pw_aff &multi2) const
{
  return isl::multi_pw_aff(*this).min(multi2);
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

class size pw_multi_aff::n_piece() const
{
  auto res = isl_pw_multi_aff_n_piece(get());
  return manage(res);
}

isl::multi_pw_aff pw_multi_aff::neg() const
{
  return isl::multi_pw_aff(*this).neg();
}

boolean pw_multi_aff::plain_is_empty() const
{
  return isl::union_pw_multi_aff(*this).plain_is_empty();
}

boolean pw_multi_aff::plain_is_equal(const isl::multi_pw_aff &multi2) const
{
  return isl::multi_pw_aff(*this).plain_is_equal(multi2);
}

boolean pw_multi_aff::plain_is_equal(const isl::multi_union_pw_aff &multi2) const
{
  return isl::multi_pw_aff(*this).plain_is_equal(multi2);
}

isl::pw_multi_aff pw_multi_aff::preimage_domain_wrapped_domain(isl::pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_preimage_domain_wrapped_domain_pw_multi_aff(copy(), pma2.release());
  return manage(res);
}

isl::union_pw_multi_aff pw_multi_aff::preimage_domain_wrapped_domain(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_multi_aff(*this).preimage_domain_wrapped_domain(upma2);
}

isl::pw_multi_aff pw_multi_aff::preimage_domain_wrapped_domain(const isl::multi_aff &pma2) const
{
  return this->preimage_domain_wrapped_domain(isl::pw_multi_aff(pma2));
}

isl::pw_multi_aff pw_multi_aff::preimage_domain_wrapped_domain(const isl::pw_aff &pma2) const
{
  return this->preimage_domain_wrapped_domain(isl::pw_multi_aff(pma2));
}

isl::multi_pw_aff pw_multi_aff::product(const isl::multi_pw_aff &multi2) const
{
  return isl::multi_pw_aff(*this).product(multi2);
}

isl::pw_multi_aff pw_multi_aff::product(isl::pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_product(copy(), pma2.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::product(const isl::multi_aff &pma2) const
{
  return this->product(isl::pw_multi_aff(pma2));
}

isl::pw_multi_aff pw_multi_aff::product(const isl::pw_aff &pma2) const
{
  return this->product(isl::pw_multi_aff(pma2));
}

isl::pw_multi_aff pw_multi_aff::project_out_map(isl::space space, isl::dim type, unsigned int first, unsigned int n)
{
  auto res = isl_pw_multi_aff_project_out_map(space.release(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::multi_pw_aff pw_multi_aff::pullback(const isl::multi_pw_aff &mpa2) const
{
  return isl::multi_pw_aff(*this).pullback(mpa2);
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

isl::union_pw_multi_aff pw_multi_aff::pullback(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_multi_aff(*this).pullback(upma2);
}

isl::pw_multi_aff_list pw_multi_aff::pw_multi_aff_list() const
{
  return isl::union_pw_multi_aff(*this).pw_multi_aff_list();
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

isl::multi_pw_aff pw_multi_aff::range_product(const isl::multi_pw_aff &multi2) const
{
  return isl::multi_pw_aff(*this).range_product(multi2);
}

isl::multi_union_pw_aff pw_multi_aff::range_product(const isl::multi_union_pw_aff &multi2) const
{
  return isl::multi_pw_aff(*this).range_product(multi2);
}

isl::pw_multi_aff pw_multi_aff::range_product(isl::pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_range_product(copy(), pma2.release());
  return manage(res);
}

isl::union_pw_multi_aff pw_multi_aff::range_product(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_multi_aff(*this).range_product(upma2);
}

isl::pw_multi_aff pw_multi_aff::range_product(const isl::multi_aff &pma2) const
{
  return this->range_product(isl::pw_multi_aff(pma2));
}

isl::pw_multi_aff pw_multi_aff::range_product(const isl::pw_aff &pma2) const
{
  return this->range_product(isl::pw_multi_aff(pma2));
}

isl::id pw_multi_aff::range_tuple_id() const
{
  auto res = isl_pw_multi_aff_get_range_tuple_id(get());
  return manage(res);
}

isl::id pw_multi_aff::get_range_tuple_id() const
{
  return range_tuple_id();
}

isl::multi_pw_aff pw_multi_aff::reset_range_tuple_id() const
{
  return isl::multi_pw_aff(*this).reset_range_tuple_id();
}

isl::multi_pw_aff pw_multi_aff::reset_tuple_id(isl::dim type) const
{
  return isl::multi_pw_aff(*this).reset_tuple_id(type);
}

isl::multi_pw_aff pw_multi_aff::scale(const isl::multi_val &mv) const
{
  return isl::multi_pw_aff(*this).scale(mv);
}

isl::pw_multi_aff pw_multi_aff::scale(isl::val v) const
{
  auto res = isl_pw_multi_aff_scale_val(copy(), v.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::scale(long v) const
{
  return this->scale(isl::val(ctx(), v));
}

isl::multi_pw_aff pw_multi_aff::scale_down(const isl::multi_val &mv) const
{
  return isl::multi_pw_aff(*this).scale_down(mv);
}

isl::pw_multi_aff pw_multi_aff::scale_down(isl::val v) const
{
  auto res = isl_pw_multi_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::scale_down(long v) const
{
  return this->scale_down(isl::val(ctx(), v));
}

isl::multi_pw_aff pw_multi_aff::set_at(int pos, const isl::pw_aff &el) const
{
  return isl::multi_pw_aff(*this).set_at(pos, el);
}

isl::multi_union_pw_aff pw_multi_aff::set_at(int pos, const isl::union_pw_aff &el) const
{
  return isl::multi_pw_aff(*this).set_at(pos, el);
}

isl::multi_pw_aff pw_multi_aff::set_pw_aff(int pos, const isl::pw_aff &el) const
{
  return isl::multi_pw_aff(*this).set_pw_aff(pos, el);
}

isl::pw_multi_aff pw_multi_aff::set_pw_aff(unsigned int pos, isl::pw_aff pa) const
{
  auto res = isl_pw_multi_aff_set_pw_aff(copy(), pos, pa.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::set_range_tuple(isl::id id) const
{
  auto res = isl_pw_multi_aff_set_range_tuple_id(copy(), id.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::set_range_tuple(const std::string &id) const
{
  return this->set_range_tuple(isl::id(ctx(), id));
}

isl::multi_union_pw_aff pw_multi_aff::set_union_pw_aff(int pos, const isl::union_pw_aff &el) const
{
  return isl::multi_pw_aff(*this).set_union_pw_aff(pos, el);
}

class size pw_multi_aff::size() const
{
  return isl::multi_pw_aff(*this).size();
}

isl::space pw_multi_aff::space() const
{
  auto res = isl_pw_multi_aff_get_space(get());
  return manage(res);
}

isl::space pw_multi_aff::get_space() const
{
  return space();
}

isl::multi_pw_aff pw_multi_aff::sub(const isl::multi_pw_aff &multi2) const
{
  return isl::multi_pw_aff(*this).sub(multi2);
}

isl::multi_union_pw_aff pw_multi_aff::sub(const isl::multi_union_pw_aff &multi2) const
{
  return isl::multi_pw_aff(*this).sub(multi2);
}

isl::pw_multi_aff pw_multi_aff::sub(isl::pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_sub(copy(), pma2.release());
  return manage(res);
}

isl::union_pw_multi_aff pw_multi_aff::sub(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_multi_aff(*this).sub(upma2);
}

isl::pw_multi_aff pw_multi_aff::sub(const isl::multi_aff &pma2) const
{
  return this->sub(isl::pw_multi_aff(pma2));
}

isl::pw_multi_aff pw_multi_aff::sub(const isl::pw_aff &pma2) const
{
  return this->sub(isl::pw_multi_aff(pma2));
}

isl::pw_multi_aff pw_multi_aff::subtract_domain(isl::set set) const
{
  auto res = isl_pw_multi_aff_subtract_domain(copy(), set.release());
  return manage(res);
}

isl::union_pw_multi_aff pw_multi_aff::subtract_domain(const isl::space &space) const
{
  return isl::union_pw_multi_aff(*this).subtract_domain(space);
}

isl::union_pw_multi_aff pw_multi_aff::subtract_domain(const isl::union_set &uset) const
{
  return isl::union_pw_multi_aff(*this).subtract_domain(uset);
}

isl::pw_multi_aff pw_multi_aff::subtract_domain(const isl::basic_set &set) const
{
  return this->subtract_domain(isl::set(set));
}

isl::pw_multi_aff pw_multi_aff::subtract_domain(const isl::point &set) const
{
  return this->subtract_domain(isl::set(set));
}

isl::pw_multi_aff_list pw_multi_aff::to_list() const
{
  auto res = isl_pw_multi_aff_to_list(copy());
  return manage(res);
}

isl::multi_pw_aff pw_multi_aff::to_multi_pw_aff() const
{
  auto res = isl_pw_multi_aff_to_multi_pw_aff(copy());
  return manage(res);
}

isl::union_pw_multi_aff pw_multi_aff::to_union_pw_multi_aff() const
{
  auto res = isl_pw_multi_aff_to_union_pw_multi_aff(copy());
  return manage(res);
}

isl::id pw_multi_aff::tuple_id(isl::dim type) const
{
  auto res = isl_pw_multi_aff_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::id pw_multi_aff::get_tuple_id(isl::dim type) const
{
  return tuple_id(type);
}

isl::multi_pw_aff pw_multi_aff::unbind_params_insert_domain(const isl::multi_id &domain) const
{
  return isl::multi_pw_aff(*this).unbind_params_insert_domain(domain);
}

isl::multi_pw_aff pw_multi_aff::union_add(const isl::multi_pw_aff &mpa2) const
{
  return isl::multi_pw_aff(*this).union_add(mpa2);
}

isl::multi_union_pw_aff pw_multi_aff::union_add(const isl::multi_union_pw_aff &mupa2) const
{
  return isl::multi_pw_aff(*this).union_add(mupa2);
}

isl::pw_multi_aff pw_multi_aff::union_add(isl::pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_union_add(copy(), pma2.release());
  return manage(res);
}

isl::union_pw_multi_aff pw_multi_aff::union_add(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_multi_aff(*this).union_add(upma2);
}

isl::pw_multi_aff pw_multi_aff::union_add(const isl::multi_aff &pma2) const
{
  return this->union_add(isl::pw_multi_aff(pma2));
}

isl::pw_multi_aff pw_multi_aff::union_add(const isl::pw_aff &pma2) const
{
  return this->union_add(isl::pw_multi_aff(pma2));
}

isl::pw_multi_aff pw_multi_aff::zero(isl::space space)
{
  auto res = isl_pw_multi_aff_zero(space.release());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const pw_multi_aff &obj)
{
  char *str = isl_pw_multi_aff_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

pw_multi_aff_list::pw_multi_aff_list(isl::ctx ctx, int n)
{
  auto res = isl_pw_multi_aff_list_alloc(ctx.release(), n);
  ptr = res;
}

pw_multi_aff_list::pw_multi_aff_list(isl::pw_multi_aff el)
{
  auto res = isl_pw_multi_aff_list_from_pw_multi_aff(el.release());
  ptr = res;
}

pw_multi_aff_list::pw_multi_aff_list(isl::ctx ctx, const std::string &str)
{
  auto res = isl_pw_multi_aff_list_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

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

isl::pw_multi_aff_list pw_multi_aff_list::add(isl::pw_multi_aff el) const
{
  auto res = isl_pw_multi_aff_list_add(copy(), el.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff_list::at(int index) const
{
  auto res = isl_pw_multi_aff_list_get_at(get(), index);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff_list::get_at(int index) const
{
  return at(index);
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

stat pw_multi_aff_list::foreach(const std::function<stat(isl::pw_multi_aff)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::pw_multi_aff)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_pw_multi_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_pw_multi_aff_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

isl::pw_multi_aff_list pw_multi_aff_list::insert(unsigned int pos, isl::pw_multi_aff el) const
{
  auto res = isl_pw_multi_aff_list_insert(copy(), pos, el.release());
  return manage(res);
}

class size pw_multi_aff_list::size() const
{
  auto res = isl_pw_multi_aff_list_size(get());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const pw_multi_aff_list &obj)
{
  char *str = isl_pw_multi_aff_list_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::schedule schedule::align_params(isl::space space) const
{
  auto res = isl_schedule_align_params(copy(), space.release());
  return manage(res);
}

isl::union_set schedule::domain() const
{
  auto res = isl_schedule_get_domain(get());
  return manage(res);
}

isl::union_set schedule::get_domain() const
{
  return domain();
}

isl::schedule schedule::from_domain(isl::union_set domain)
{
  auto res = isl_schedule_from_domain(domain.release());
  return manage(res);
}

isl::schedule schedule::gist_domain_params(isl::set context) const
{
  auto res = isl_schedule_gist_domain_params(copy(), context.release());
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

isl::union_map schedule::map() const
{
  auto res = isl_schedule_get_map(get());
  return manage(res);
}

isl::union_map schedule::get_map() const
{
  return map();
}

isl::schedule schedule::pullback(isl::union_pw_multi_aff upma) const
{
  auto res = isl_schedule_pullback_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

isl::schedule_node schedule::root() const
{
  auto res = isl_schedule_get_root(get());
  return manage(res);
}

isl::schedule_node schedule::get_root() const
{
  return root();
}

isl::schedule schedule::sequence(isl::schedule schedule2) const
{
  auto res = isl_schedule_sequence(copy(), schedule2.release());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const schedule &obj)
{
  char *str = isl_schedule_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::union_map schedule_constraints::coincidence() const
{
  auto res = isl_schedule_constraints_get_coincidence(get());
  return manage(res);
}

isl::union_map schedule_constraints::get_coincidence() const
{
  return coincidence();
}

isl::schedule schedule_constraints::compute_schedule() const
{
  auto res = isl_schedule_constraints_compute_schedule(copy());
  return manage(res);
}

isl::union_map schedule_constraints::conditional_validity() const
{
  auto res = isl_schedule_constraints_get_conditional_validity(get());
  return manage(res);
}

isl::union_map schedule_constraints::get_conditional_validity() const
{
  return conditional_validity();
}

isl::union_map schedule_constraints::conditional_validity_condition() const
{
  auto res = isl_schedule_constraints_get_conditional_validity_condition(get());
  return manage(res);
}

isl::union_map schedule_constraints::get_conditional_validity_condition() const
{
  return conditional_validity_condition();
}

isl::set schedule_constraints::context() const
{
  auto res = isl_schedule_constraints_get_context(get());
  return manage(res);
}

isl::set schedule_constraints::get_context() const
{
  return context();
}

isl::union_set schedule_constraints::domain() const
{
  auto res = isl_schedule_constraints_get_domain(get());
  return manage(res);
}

isl::union_set schedule_constraints::get_domain() const
{
  return domain();
}

isl::schedule_constraints schedule_constraints::on_domain(isl::union_set domain)
{
  auto res = isl_schedule_constraints_on_domain(domain.release());
  return manage(res);
}

isl::union_map schedule_constraints::proximity() const
{
  auto res = isl_schedule_constraints_get_proximity(get());
  return manage(res);
}

isl::union_map schedule_constraints::get_proximity() const
{
  return proximity();
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

isl::union_map schedule_constraints::validity() const
{
  auto res = isl_schedule_constraints_get_validity(get());
  return manage(res);
}

isl::union_map schedule_constraints::get_validity() const
{
  return validity();
}

inline std::ostream &operator<<(std::ostream &os, const schedule_constraints &obj)
{
  char *str = isl_schedule_constraints_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

template <typename T, typename>
boolean schedule_node::isa_type(T subtype) const
{
  if (is_null())
    return boolean();
  return isl_schedule_node_get_type(get()) == subtype;
}
template <class T>
boolean schedule_node::isa() const
{
  return isa_type<decltype(T::type)>(T::type);
}
template <class T>
T schedule_node::as() const
{
 if (isa<T>().is_false())
    isl_die(ctx().get(), isl_error_invalid, "not an object of the requested subtype", return T());
  return T(copy());
}

isl::ctx schedule_node::ctx() const {
  return isl::ctx(isl_schedule_node_get_ctx(ptr));
}

isl::schedule_node schedule_node::ancestor(int generation) const
{
  auto res = isl_schedule_node_ancestor(copy(), generation);
  return manage(res);
}

class size schedule_node::ancestor_child_position(const isl::schedule_node &ancestor) const
{
  auto res = isl_schedule_node_get_ancestor_child_position(get(), ancestor.get());
  return manage(res);
}

class size schedule_node::get_ancestor_child_position(const isl::schedule_node &ancestor) const
{
  return ancestor_child_position(ancestor);
}

isl::schedule_node schedule_node::child(int pos) const
{
  auto res = isl_schedule_node_child(copy(), pos);
  return manage(res);
}

class size schedule_node::child_position() const
{
  auto res = isl_schedule_node_get_child_position(get());
  return manage(res);
}

class size schedule_node::get_child_position() const
{
  return child_position();
}

isl::union_set schedule_node::domain() const
{
  auto res = isl_schedule_node_get_domain(get());
  return manage(res);
}

isl::union_set schedule_node::get_domain() const
{
  return domain();
}

boolean schedule_node::every_descendant(const std::function<boolean(isl::schedule_node)> &test) const
{
  struct test_data {
    std::function<boolean(isl::schedule_node)> func;
  } test_data = { test };
  auto test_lambda = [](isl_schedule_node *arg_0, void *arg_1) -> isl_bool {
    auto *data = static_cast<struct test_data *>(arg_1);
    auto ret = (data->func)(manage_copy(arg_0));
    return ret.release();
  };
  auto res = isl_schedule_node_every_descendant(get(), test_lambda, &test_data);
  return manage(res);
}

isl::schedule_node schedule_node::first_child() const
{
  auto res = isl_schedule_node_first_child(copy());
  return manage(res);
}

stat schedule_node::foreach_ancestor_top_down(const std::function<stat(isl::schedule_node)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::schedule_node)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_schedule_node *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage_copy(arg_0));
    return ret.release();
  };
  auto res = isl_schedule_node_foreach_ancestor_top_down(get(), fn_lambda, &fn_data);
  return manage(res);
}

stat schedule_node::foreach_descendant_top_down(const std::function<boolean(isl::schedule_node)> &fn) const
{
  struct fn_data {
    std::function<boolean(isl::schedule_node)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_schedule_node *arg_0, void *arg_1) -> isl_bool {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage_copy(arg_0));
    return ret.release();
  };
  auto res = isl_schedule_node_foreach_descendant_top_down(get(), fn_lambda, &fn_data);
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

isl::schedule_node schedule_node::insert_mark(const std::string &mark) const
{
  return this->insert_mark(isl::id(ctx(), mark));
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

isl::schedule_node schedule_node::map_descendant_bottom_up(const std::function<isl::schedule_node(isl::schedule_node)> &fn) const
{
  struct fn_data {
    std::function<isl::schedule_node(isl::schedule_node)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_schedule_node *arg_0, void *arg_1) -> isl_schedule_node * {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_schedule_node_map_descendant_bottom_up(copy(), fn_lambda, &fn_data);
  return manage(res);
}

class size schedule_node::n_children() const
{
  auto res = isl_schedule_node_n_children(get());
  return manage(res);
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

isl::multi_union_pw_aff schedule_node::prefix_schedule_multi_union_pw_aff() const
{
  auto res = isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(get());
  return manage(res);
}

isl::multi_union_pw_aff schedule_node::get_prefix_schedule_multi_union_pw_aff() const
{
  return prefix_schedule_multi_union_pw_aff();
}

isl::union_map schedule_node::prefix_schedule_relation() const
{
  auto res = isl_schedule_node_get_prefix_schedule_relation(get());
  return manage(res);
}

isl::union_map schedule_node::get_prefix_schedule_relation() const
{
  return prefix_schedule_relation();
}

isl::union_map schedule_node::prefix_schedule_union_map() const
{
  auto res = isl_schedule_node_get_prefix_schedule_union_map(get());
  return manage(res);
}

isl::union_map schedule_node::get_prefix_schedule_union_map() const
{
  return prefix_schedule_union_map();
}

isl::union_pw_multi_aff schedule_node::prefix_schedule_union_pw_multi_aff() const
{
  auto res = isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(get());
  return manage(res);
}

isl::union_pw_multi_aff schedule_node::get_prefix_schedule_union_pw_multi_aff() const
{
  return prefix_schedule_union_pw_multi_aff();
}

isl::schedule_node schedule_node::previous_sibling() const
{
  auto res = isl_schedule_node_previous_sibling(copy());
  return manage(res);
}

isl::schedule_node schedule_node::root() const
{
  auto res = isl_schedule_node_root(copy());
  return manage(res);
}

isl::schedule schedule_node::schedule() const
{
  auto res = isl_schedule_node_get_schedule(get());
  return manage(res);
}

isl::schedule schedule_node::get_schedule() const
{
  return schedule();
}

class size schedule_node::schedule_depth() const
{
  auto res = isl_schedule_node_get_schedule_depth(get());
  return manage(res);
}

class size schedule_node::get_schedule_depth() const
{
  return schedule_depth();
}

isl::schedule_node schedule_node::shared_ancestor(const isl::schedule_node &node2) const
{
  auto res = isl_schedule_node_get_shared_ancestor(get(), node2.get());
  return manage(res);
}

isl::schedule_node schedule_node::get_shared_ancestor(const isl::schedule_node &node2) const
{
  return shared_ancestor(node2);
}

class size schedule_node::tree_depth() const
{
  auto res = isl_schedule_node_get_tree_depth(get());
  return manage(res);
}

class size schedule_node::get_tree_depth() const
{
  return tree_depth();
}

isl::union_set schedule_node::universe_domain() const
{
  auto res = isl_schedule_node_get_universe_domain(get());
  return manage(res);
}

isl::union_set schedule_node::get_universe_domain() const
{
  return universe_domain();
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node &obj)
{
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::schedule_node_band
schedule_node_band::schedule_node_band()
    : schedule_node() {}

schedule_node_band::schedule_node_band(const schedule_node_band &obj)
    : schedule_node(obj)
{
}

schedule_node_band::schedule_node_band(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}

schedule_node_band &schedule_node_band::operator=(schedule_node_band obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx schedule_node_band::ctx() const {
  return isl::ctx(isl_schedule_node_get_ctx(ptr));
}

isl::union_set schedule_node_band::ast_build_options() const
{
  auto res = isl_schedule_node_band_get_ast_build_options(get());
  return manage(res);
}

isl::union_set schedule_node_band::get_ast_build_options() const
{
  return ast_build_options();
}

isl::set schedule_node_band::ast_isolate_option() const
{
  auto res = isl_schedule_node_band_get_ast_isolate_option(get());
  return manage(res);
}

isl::set schedule_node_band::get_ast_isolate_option() const
{
  return ast_isolate_option();
}

boolean schedule_node_band::member_get_coincident(int pos) const
{
  auto res = isl_schedule_node_band_member_get_coincident(get(), pos);
  return manage(res);
}

schedule_node_band schedule_node_band::member_set_coincident(int pos, int coincident) const
{
  auto res = isl_schedule_node_band_member_set_coincident(copy(), pos, coincident);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::mod(isl::multi_val mv) const
{
  auto res = isl_schedule_node_band_mod(copy(), mv.release());
  return manage(res).as<schedule_node_band>();
}

class size schedule_node_band::n_member() const
{
  auto res = isl_schedule_node_band_n_member(get());
  return manage(res);
}

isl::multi_union_pw_aff schedule_node_band::partial_schedule() const
{
  auto res = isl_schedule_node_band_get_partial_schedule(get());
  return manage(res);
}

isl::multi_union_pw_aff schedule_node_band::get_partial_schedule() const
{
  return partial_schedule();
}

boolean schedule_node_band::permutable() const
{
  auto res = isl_schedule_node_band_get_permutable(get());
  return manage(res);
}

boolean schedule_node_band::get_permutable() const
{
  return permutable();
}

schedule_node_band schedule_node_band::scale(isl::multi_val mv) const
{
  auto res = isl_schedule_node_band_scale(copy(), mv.release());
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::scale_down(isl::multi_val mv) const
{
  auto res = isl_schedule_node_band_scale_down(copy(), mv.release());
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::set_ast_build_options(isl::union_set options) const
{
  auto res = isl_schedule_node_band_set_ast_build_options(copy(), options.release());
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::set_permutable(int permutable) const
{
  auto res = isl_schedule_node_band_set_permutable(copy(), permutable);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::shift(isl::multi_union_pw_aff shift) const
{
  auto res = isl_schedule_node_band_shift(copy(), shift.release());
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::split(int pos) const
{
  auto res = isl_schedule_node_band_split(copy(), pos);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::tile(isl::multi_val sizes) const
{
  auto res = isl_schedule_node_band_tile(copy(), sizes.release());
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::member_set_ast_loop_default(int pos) const
{
  auto res = isl_schedule_node_band_member_set_ast_loop_type(copy(), pos, isl_ast_loop_default);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::member_set_ast_loop_atomic(int pos) const
{
  auto res = isl_schedule_node_band_member_set_ast_loop_type(copy(), pos, isl_ast_loop_atomic);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::member_set_ast_loop_unroll(int pos) const
{
  auto res = isl_schedule_node_band_member_set_ast_loop_type(copy(), pos, isl_ast_loop_unroll);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::member_set_ast_loop_separate(int pos) const
{
  auto res = isl_schedule_node_band_member_set_ast_loop_type(copy(), pos, isl_ast_loop_separate);
  return manage(res).as<schedule_node_band>();
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node_band &obj)
{
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::schedule_node_context
schedule_node_context::schedule_node_context()
    : schedule_node() {}

schedule_node_context::schedule_node_context(const schedule_node_context &obj)
    : schedule_node(obj)
{
}

schedule_node_context::schedule_node_context(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}

schedule_node_context &schedule_node_context::operator=(schedule_node_context obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx schedule_node_context::ctx() const {
  return isl::ctx(isl_schedule_node_get_ctx(ptr));
}

isl::set schedule_node_context::context() const
{
  auto res = isl_schedule_node_context_get_context(get());
  return manage(res);
}

isl::set schedule_node_context::get_context() const
{
  return context();
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node_context &obj)
{
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::schedule_node_domain
schedule_node_domain::schedule_node_domain()
    : schedule_node() {}

schedule_node_domain::schedule_node_domain(const schedule_node_domain &obj)
    : schedule_node(obj)
{
}

schedule_node_domain::schedule_node_domain(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}

schedule_node_domain &schedule_node_domain::operator=(schedule_node_domain obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx schedule_node_domain::ctx() const {
  return isl::ctx(isl_schedule_node_get_ctx(ptr));
}

isl::union_set schedule_node_domain::domain() const
{
  auto res = isl_schedule_node_domain_get_domain(get());
  return manage(res);
}

isl::union_set schedule_node_domain::get_domain() const
{
  return domain();
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node_domain &obj)
{
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::schedule_node_expansion
schedule_node_expansion::schedule_node_expansion()
    : schedule_node() {}

schedule_node_expansion::schedule_node_expansion(const schedule_node_expansion &obj)
    : schedule_node(obj)
{
}

schedule_node_expansion::schedule_node_expansion(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}

schedule_node_expansion &schedule_node_expansion::operator=(schedule_node_expansion obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx schedule_node_expansion::ctx() const {
  return isl::ctx(isl_schedule_node_get_ctx(ptr));
}

isl::union_pw_multi_aff schedule_node_expansion::contraction() const
{
  auto res = isl_schedule_node_expansion_get_contraction(get());
  return manage(res);
}

isl::union_pw_multi_aff schedule_node_expansion::get_contraction() const
{
  return contraction();
}

isl::union_map schedule_node_expansion::expansion() const
{
  auto res = isl_schedule_node_expansion_get_expansion(get());
  return manage(res);
}

isl::union_map schedule_node_expansion::get_expansion() const
{
  return expansion();
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node_expansion &obj)
{
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::schedule_node_extension
schedule_node_extension::schedule_node_extension()
    : schedule_node() {}

schedule_node_extension::schedule_node_extension(const schedule_node_extension &obj)
    : schedule_node(obj)
{
}

schedule_node_extension::schedule_node_extension(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}

schedule_node_extension &schedule_node_extension::operator=(schedule_node_extension obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx schedule_node_extension::ctx() const {
  return isl::ctx(isl_schedule_node_get_ctx(ptr));
}

isl::union_map schedule_node_extension::extension() const
{
  auto res = isl_schedule_node_extension_get_extension(get());
  return manage(res);
}

isl::union_map schedule_node_extension::get_extension() const
{
  return extension();
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node_extension &obj)
{
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::schedule_node_filter
schedule_node_filter::schedule_node_filter()
    : schedule_node() {}

schedule_node_filter::schedule_node_filter(const schedule_node_filter &obj)
    : schedule_node(obj)
{
}

schedule_node_filter::schedule_node_filter(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}

schedule_node_filter &schedule_node_filter::operator=(schedule_node_filter obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx schedule_node_filter::ctx() const {
  return isl::ctx(isl_schedule_node_get_ctx(ptr));
}

isl::union_set schedule_node_filter::filter() const
{
  auto res = isl_schedule_node_filter_get_filter(get());
  return manage(res);
}

isl::union_set schedule_node_filter::get_filter() const
{
  return filter();
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node_filter &obj)
{
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::schedule_node_guard
schedule_node_guard::schedule_node_guard()
    : schedule_node() {}

schedule_node_guard::schedule_node_guard(const schedule_node_guard &obj)
    : schedule_node(obj)
{
}

schedule_node_guard::schedule_node_guard(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}

schedule_node_guard &schedule_node_guard::operator=(schedule_node_guard obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx schedule_node_guard::ctx() const {
  return isl::ctx(isl_schedule_node_get_ctx(ptr));
}

isl::set schedule_node_guard::guard() const
{
  auto res = isl_schedule_node_guard_get_guard(get());
  return manage(res);
}

isl::set schedule_node_guard::get_guard() const
{
  return guard();
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node_guard &obj)
{
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::schedule_node_leaf
schedule_node_leaf::schedule_node_leaf()
    : schedule_node() {}

schedule_node_leaf::schedule_node_leaf(const schedule_node_leaf &obj)
    : schedule_node(obj)
{
}

schedule_node_leaf::schedule_node_leaf(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}

schedule_node_leaf &schedule_node_leaf::operator=(schedule_node_leaf obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx schedule_node_leaf::ctx() const {
  return isl::ctx(isl_schedule_node_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node_leaf &obj)
{
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::schedule_node_mark
schedule_node_mark::schedule_node_mark()
    : schedule_node() {}

schedule_node_mark::schedule_node_mark(const schedule_node_mark &obj)
    : schedule_node(obj)
{
}

schedule_node_mark::schedule_node_mark(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}

schedule_node_mark &schedule_node_mark::operator=(schedule_node_mark obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx schedule_node_mark::ctx() const {
  return isl::ctx(isl_schedule_node_get_ctx(ptr));
}

isl::id schedule_node_mark::id() const
{
  auto res = isl_schedule_node_mark_get_id(get());
  return manage(res);
}

isl::id schedule_node_mark::get_id() const
{
  return id();
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node_mark &obj)
{
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::schedule_node_sequence
schedule_node_sequence::schedule_node_sequence()
    : schedule_node() {}

schedule_node_sequence::schedule_node_sequence(const schedule_node_sequence &obj)
    : schedule_node(obj)
{
}

schedule_node_sequence::schedule_node_sequence(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}

schedule_node_sequence &schedule_node_sequence::operator=(schedule_node_sequence obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx schedule_node_sequence::ctx() const {
  return isl::ctx(isl_schedule_node_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node_sequence &obj)
{
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}

// implementations for isl::schedule_node_set
schedule_node_set::schedule_node_set()
    : schedule_node() {}

schedule_node_set::schedule_node_set(const schedule_node_set &obj)
    : schedule_node(obj)
{
}

schedule_node_set::schedule_node_set(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}

schedule_node_set &schedule_node_set::operator=(schedule_node_set obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

isl::ctx schedule_node_set::ctx() const {
  return isl::ctx(isl_schedule_node_get_ctx(ptr));
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node_set &obj)
{
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::union_set set::apply(const isl::union_map &umap) const
{
  return isl::union_set(*this).apply(umap);
}

isl::set set::apply(const isl::basic_map &map) const
{
  return this->apply(isl::map(map));
}

isl::pw_multi_aff set::as_pw_multi_aff() const
{
  auto res = isl_set_as_pw_multi_aff(copy());
  return manage(res);
}

isl::set set::as_set() const
{
  return isl::union_set(*this).as_set();
}

isl::basic_set_list set::basic_set_list() const
{
  auto res = isl_set_get_basic_set_list(get());
  return manage(res);
}

isl::basic_set_list set::get_basic_set_list() const
{
  return basic_set_list();
}

isl::set set::bind(isl::multi_id tuple) const
{
  auto res = isl_set_bind(copy(), tuple.release());
  return manage(res);
}

isl::set set::coalesce() const
{
  auto res = isl_set_coalesce(copy());
  return manage(res);
}

isl::set set::complement() const
{
  auto res = isl_set_complement(copy());
  return manage(res);
}

isl::union_set set::compute_divs() const
{
  return isl::union_set(*this).compute_divs();
}

boolean set::contains(const isl::space &space) const
{
  return isl::union_set(*this).contains(space);
}

isl::basic_set set::convex_hull() const
{
  auto res = isl_set_convex_hull(copy());
  return manage(res);
}

isl::set set::detect_equalities() const
{
  auto res = isl_set_detect_equalities(copy());
  return manage(res);
}

class size set::dim(isl::dim type) const
{
  auto res = isl_set_dim(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

boolean set::dim_has_any_lower_bound(isl::dim type, unsigned int pos) const
{
  auto res = isl_set_dim_has_any_lower_bound(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::id set::dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_set_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::id set::get_dim_id(isl::dim type, unsigned int pos) const
{
  return dim_id(type, pos);
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

std::string set::dim_name(isl::dim type, unsigned int pos) const
{
  auto res = isl_set_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

std::string set::get_dim_name(isl::dim type, unsigned int pos) const
{
  return dim_name(type, pos);
}

isl::set set::drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const
{
  auto res = isl_set_drop_constraints_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
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

boolean set::every_set(const std::function<boolean(isl::set)> &test) const
{
  return isl::union_set(*this).every_set(test);
}

isl::set set::extract_set(const isl::space &space) const
{
  return isl::union_set(*this).extract_set(space);
}

int set::find_dim_by_id(isl::dim type, const isl::id &id) const
{
  auto res = isl_set_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int set::find_dim_by_id(isl::dim type, const std::string &id) const
{
  return this->find_dim_by_id(type, isl::id(ctx(), id));
}

isl::set set::fix_si(isl::dim type, unsigned int pos, int value) const
{
  auto res = isl_set_fix_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::set set::flatten() const
{
  auto res = isl_set_flatten(copy());
  return manage(res);
}

stat set::foreach_basic_set(const std::function<stat(isl::basic_set)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::basic_set)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_basic_set *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_set_foreach_basic_set(get(), fn_lambda, &fn_data);
  return manage(res);
}

stat set::foreach_point(const std::function<stat(isl::point)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::point)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_point *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_set_foreach_point(get(), fn_lambda, &fn_data);
  return manage(res);
}

stat set::foreach_set(const std::function<stat(isl::set)> &fn) const
{
  return isl::union_set(*this).foreach_set(fn);
}

isl::set set::gist(isl::set context) const
{
  auto res = isl_set_gist(copy(), context.release());
  return manage(res);
}

isl::union_set set::gist(const isl::union_set &context) const
{
  return isl::union_set(*this).gist(context);
}

isl::set set::gist(const isl::basic_set &context) const
{
  return this->gist(isl::set(context));
}

isl::set set::gist(const isl::point &context) const
{
  return this->gist(isl::set(context));
}

isl::set set::gist_params(isl::set context) const
{
  auto res = isl_set_gist_params(copy(), context.release());
  return manage(res);
}

boolean set::has_equal_space(const isl::set &set2) const
{
  auto res = isl_set_has_equal_space(get(), set2.get());
  return manage(res);
}

isl::map set::identity() const
{
  auto res = isl_set_identity(copy());
  return manage(res);
}

isl::union_pw_multi_aff set::identity_union_pw_multi_aff() const
{
  return isl::union_set(*this).identity_union_pw_multi_aff();
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

isl::union_set set::intersect(const isl::union_set &uset2) const
{
  return isl::union_set(*this).intersect(uset2);
}

isl::set set::intersect(const isl::basic_set &set2) const
{
  return this->intersect(isl::set(set2));
}

isl::set set::intersect(const isl::point &set2) const
{
  return this->intersect(isl::set(set2));
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

boolean set::is_disjoint(const isl::set &set2) const
{
  auto res = isl_set_is_disjoint(get(), set2.get());
  return manage(res);
}

boolean set::is_disjoint(const isl::union_set &uset2) const
{
  return isl::union_set(*this).is_disjoint(uset2);
}

boolean set::is_disjoint(const isl::basic_set &set2) const
{
  return this->is_disjoint(isl::set(set2));
}

boolean set::is_disjoint(const isl::point &set2) const
{
  return this->is_disjoint(isl::set(set2));
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

boolean set::is_equal(const isl::union_set &uset2) const
{
  return isl::union_set(*this).is_equal(uset2);
}

boolean set::is_equal(const isl::basic_set &set2) const
{
  return this->is_equal(isl::set(set2));
}

boolean set::is_equal(const isl::point &set2) const
{
  return this->is_equal(isl::set(set2));
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

boolean set::is_strict_subset(const isl::union_set &uset2) const
{
  return isl::union_set(*this).is_strict_subset(uset2);
}

boolean set::is_strict_subset(const isl::basic_set &set2) const
{
  return this->is_strict_subset(isl::set(set2));
}

boolean set::is_strict_subset(const isl::point &set2) const
{
  return this->is_strict_subset(isl::set(set2));
}

boolean set::is_subset(const isl::set &set2) const
{
  auto res = isl_set_is_subset(get(), set2.get());
  return manage(res);
}

boolean set::is_subset(const isl::union_set &uset2) const
{
  return isl::union_set(*this).is_subset(uset2);
}

boolean set::is_subset(const isl::basic_set &set2) const
{
  return this->is_subset(isl::set(set2));
}

boolean set::is_subset(const isl::point &set2) const
{
  return this->is_subset(isl::set(set2));
}

boolean set::is_wrapping() const
{
  auto res = isl_set_is_wrapping(get());
  return manage(res);
}

boolean set::isa_set() const
{
  return isl::union_set(*this).isa_set();
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

isl::set set::lower_bound_val(isl::dim type, unsigned int pos, long value) const
{
  return this->lower_bound_val(type, pos, isl::val(ctx(), value));
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

class size set::n_basic_set() const
{
  auto res = isl_set_n_basic_set(get());
  return manage(res);
}

isl::set set::params() const
{
  auto res = isl_set_params(copy());
  return manage(res);
}

isl::val set::plain_get_val_if_fixed(isl::dim type, unsigned int pos) const
{
  auto res = isl_set_plain_get_val_if_fixed(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::multi_val set::plain_multi_val_if_fixed() const
{
  auto res = isl_set_get_plain_multi_val_if_fixed(get());
  return manage(res);
}

isl::multi_val set::get_plain_multi_val_if_fixed() const
{
  return plain_multi_val_if_fixed();
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

isl::union_set set::preimage(const isl::union_pw_multi_aff &upma) const
{
  return isl::union_set(*this).preimage(upma);
}

isl::set set::product(isl::set set2) const
{
  auto res = isl_set_product(copy(), set2.release());
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

isl::set set::project_out_param(const std::string &id) const
{
  return this->project_out_param(isl::id(ctx(), id));
}

isl::set set::project_out_param(isl::id_list list) const
{
  auto res = isl_set_project_out_param_id_list(copy(), list.release());
  return manage(res);
}

isl::pw_multi_aff set::pw_multi_aff_on_domain(isl::multi_val mv) const
{
  auto res = isl_set_pw_multi_aff_on_domain_multi_val(copy(), mv.release());
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

isl::set set::remove_redundancies() const
{
  auto res = isl_set_remove_redundancies(copy());
  return manage(res);
}

isl::set set::reset_tuple_id() const
{
  auto res = isl_set_reset_tuple_id(copy());
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

isl::set set::set_dim_id(isl::dim type, unsigned int pos, const std::string &id) const
{
  return this->set_dim_id(type, pos, isl::id(ctx(), id));
}

isl::set_list set::set_list() const
{
  return isl::union_set(*this).set_list();
}

isl::set set::set_tuple_id(isl::id id) const
{
  auto res = isl_set_set_tuple_id(copy(), id.release());
  return manage(res);
}

isl::set set::set_tuple_id(const std::string &id) const
{
  return this->set_tuple_id(isl::id(ctx(), id));
}

isl::fixed_box set::simple_fixed_box_hull() const
{
  auto res = isl_set_get_simple_fixed_box_hull(get());
  return manage(res);
}

isl::fixed_box set::get_simple_fixed_box_hull() const
{
  return simple_fixed_box_hull();
}

isl::basic_set set::simple_hull() const
{
  auto res = isl_set_simple_hull(copy());
  return manage(res);
}

isl::space set::space() const
{
  auto res = isl_set_get_space(get());
  return manage(res);
}

isl::space set::get_space() const
{
  return space();
}

isl::val set::stride(int pos) const
{
  auto res = isl_set_get_stride(get(), pos);
  return manage(res);
}

isl::val set::get_stride(int pos) const
{
  return stride(pos);
}

isl::set set::subtract(isl::set set2) const
{
  auto res = isl_set_subtract(copy(), set2.release());
  return manage(res);
}

isl::union_set set::subtract(const isl::union_set &uset2) const
{
  return isl::union_set(*this).subtract(uset2);
}

isl::set set::subtract(const isl::basic_set &set2) const
{
  return this->subtract(isl::set(set2));
}

isl::set set::subtract(const isl::point &set2) const
{
  return this->subtract(isl::set(set2));
}

isl::set_list set::to_list() const
{
  auto res = isl_set_to_list(copy());
  return manage(res);
}

isl::union_set set::to_union_set() const
{
  auto res = isl_set_to_union_set(copy());
  return manage(res);
}

isl::map set::translation() const
{
  auto res = isl_set_translation(copy());
  return manage(res);
}

class size set::tuple_dim() const
{
  auto res = isl_set_tuple_dim(get());
  return manage(res);
}

isl::id set::tuple_id() const
{
  auto res = isl_set_get_tuple_id(get());
  return manage(res);
}

isl::id set::get_tuple_id() const
{
  return tuple_id();
}

std::string set::tuple_name() const
{
  auto res = isl_set_get_tuple_name(get());
  std::string tmp(res);
  return tmp;
}

std::string set::get_tuple_name() const
{
  return tuple_name();
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

isl::union_set set::unite(const isl::union_set &uset2) const
{
  return isl::union_set(*this).unite(uset2);
}

isl::set set::unite(const isl::basic_set &set2) const
{
  return this->unite(isl::set(set2));
}

isl::set set::unite(const isl::point &set2) const
{
  return this->unite(isl::set(set2));
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

isl::set set::upper_bound_val(isl::dim type, unsigned int pos, isl::val value) const
{
  auto res = isl_set_upper_bound_val(copy(), static_cast<enum isl_dim_type>(type), pos, value.release());
  return manage(res);
}

isl::set set::upper_bound_val(isl::dim type, unsigned int pos, long value) const
{
  return this->upper_bound_val(type, pos, isl::val(ctx(), value));
}

inline std::ostream &operator<<(std::ostream &os, const set &obj)
{
  char *str = isl_set_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

set_list::set_list(isl::ctx ctx, int n)
{
  auto res = isl_set_list_alloc(ctx.release(), n);
  ptr = res;
}

set_list::set_list(isl::set el)
{
  auto res = isl_set_list_from_set(el.release());
  ptr = res;
}

set_list::set_list(isl::ctx ctx, const std::string &str)
{
  auto res = isl_set_list_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

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

isl::set_list set_list::add(isl::set el) const
{
  auto res = isl_set_list_add(copy(), el.release());
  return manage(res);
}

isl::set set_list::at(int index) const
{
  auto res = isl_set_list_get_at(get(), index);
  return manage(res);
}

isl::set set_list::get_at(int index) const
{
  return at(index);
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

stat set_list::foreach(const std::function<stat(isl::set)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::set)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_set *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_set_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

isl::set_list set_list::insert(unsigned int pos, isl::set el) const
{
  auto res = isl_set_list_insert(copy(), pos, el.release());
  return manage(res);
}

class size set_list::size() const
{
  auto res = isl_set_list_size(get());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const set_list &obj)
{
  char *str = isl_set_list_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::space space::add_named_tuple(const std::string &tuple_id, unsigned int dim) const
{
  return this->add_named_tuple(isl::id(ctx(), tuple_id), dim);
}

isl::space space::add_param(isl::id id) const
{
  auto res = isl_space_add_param_id(copy(), id.release());
  return manage(res);
}

isl::space space::add_param(const std::string &id) const
{
  return this->add_param(isl::id(ctx(), id));
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

isl::space space::curry() const
{
  auto res = isl_space_curry(copy());
  return manage(res);
}

class size space::dim(isl::dim type) const
{
  auto res = isl_space_dim(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::id space::dim_id(isl::dim type, unsigned int pos) const
{
  auto res = isl_space_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::id space::get_dim_id(isl::dim type, unsigned int pos) const
{
  return dim_id(type, pos);
}

isl::space space::domain() const
{
  auto res = isl_space_domain(copy());
  return manage(res);
}

isl::multi_aff space::domain_map_multi_aff() const
{
  auto res = isl_space_domain_map_multi_aff(copy());
  return manage(res);
}

isl::pw_multi_aff space::domain_map_pw_multi_aff() const
{
  auto res = isl_space_domain_map_pw_multi_aff(copy());
  return manage(res);
}

isl::id space::domain_tuple_id() const
{
  auto res = isl_space_get_domain_tuple_id(get());
  return manage(res);
}

isl::id space::get_domain_tuple_id() const
{
  return domain_tuple_id();
}

isl::space space::drop_dims(isl::dim type, unsigned int first, unsigned int num) const
{
  auto res = isl_space_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, num);
  return manage(res);
}

int space::find_dim_by_id(isl::dim type, const isl::id &id) const
{
  auto res = isl_space_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int space::find_dim_by_id(isl::dim type, const std::string &id) const
{
  return this->find_dim_by_id(type, isl::id(ctx(), id));
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

boolean space::has_domain_tuple_id() const
{
  auto res = isl_space_has_domain_tuple_id(get());
  return manage(res);
}

boolean space::has_equal_tuples(const isl::space &space2) const
{
  auto res = isl_space_has_equal_tuples(get(), space2.get());
  return manage(res);
}

boolean space::has_range_tuple_id() const
{
  auto res = isl_space_has_range_tuple_id(get());
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

isl::multi_aff space::identity_multi_aff_on_domain() const
{
  auto res = isl_space_identity_multi_aff_on_domain(copy());
  return manage(res);
}

isl::multi_pw_aff space::identity_multi_pw_aff_on_domain() const
{
  auto res = isl_space_identity_multi_pw_aff_on_domain(copy());
  return manage(res);
}

isl::pw_multi_aff space::identity_pw_multi_aff_on_domain() const
{
  auto res = isl_space_identity_pw_multi_aff_on_domain(copy());
  return manage(res);
}

boolean space::is_equal(const isl::space &space2) const
{
  auto res = isl_space_is_equal(get(), space2.get());
  return manage(res);
}

boolean space::is_params() const
{
  auto res = isl_space_is_params(get());
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

isl::multi_aff space::multi_aff(isl::aff_list list) const
{
  auto res = isl_space_multi_aff(copy(), list.release());
  return manage(res);
}

isl::multi_aff space::multi_aff_on_domain(isl::multi_val mv) const
{
  auto res = isl_space_multi_aff_on_domain_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_id space::multi_id(isl::id_list list) const
{
  auto res = isl_space_multi_id(copy(), list.release());
  return manage(res);
}

isl::multi_pw_aff space::multi_pw_aff(isl::pw_aff_list list) const
{
  auto res = isl_space_multi_pw_aff(copy(), list.release());
  return manage(res);
}

isl::multi_union_pw_aff space::multi_union_pw_aff(isl::union_pw_aff_list list) const
{
  auto res = isl_space_multi_union_pw_aff(copy(), list.release());
  return manage(res);
}

isl::multi_val space::multi_val(isl::val_list list) const
{
  auto res = isl_space_multi_val(copy(), list.release());
  return manage(res);
}

isl::aff space::param_aff_on_domain(isl::id id) const
{
  auto res = isl_space_param_aff_on_domain_id(copy(), id.release());
  return manage(res);
}

isl::aff space::param_aff_on_domain(const std::string &id) const
{
  return this->param_aff_on_domain(isl::id(ctx(), id));
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

isl::multi_aff space::range_map_multi_aff() const
{
  auto res = isl_space_range_map_multi_aff(copy());
  return manage(res);
}

isl::pw_multi_aff space::range_map_pw_multi_aff() const
{
  auto res = isl_space_range_map_pw_multi_aff(copy());
  return manage(res);
}

isl::space space::range_reverse() const
{
  auto res = isl_space_range_reverse(copy());
  return manage(res);
}

isl::id space::range_tuple_id() const
{
  auto res = isl_space_get_range_tuple_id(get());
  return manage(res);
}

isl::id space::get_range_tuple_id() const
{
  return range_tuple_id();
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

isl::space space::set_dim_id(isl::dim type, unsigned int pos, const std::string &id) const
{
  return this->set_dim_id(type, pos, isl::id(ctx(), id));
}

isl::space space::set_domain_tuple(isl::id id) const
{
  auto res = isl_space_set_domain_tuple_id(copy(), id.release());
  return manage(res);
}

isl::space space::set_domain_tuple(const std::string &id) const
{
  return this->set_domain_tuple(isl::id(ctx(), id));
}

isl::space space::set_from_params() const
{
  auto res = isl_space_set_from_params(copy());
  return manage(res);
}

isl::space space::set_range_tuple(isl::id id) const
{
  auto res = isl_space_set_range_tuple_id(copy(), id.release());
  return manage(res);
}

isl::space space::set_range_tuple(const std::string &id) const
{
  return this->set_range_tuple(isl::id(ctx(), id));
}

isl::space space::set_tuple_id(isl::dim type, isl::id id) const
{
  auto res = isl_space_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::space space::set_tuple_id(isl::dim type, const std::string &id) const
{
  return this->set_tuple_id(type, isl::id(ctx(), id));
}

isl::id space::tuple_id(isl::dim type) const
{
  auto res = isl_space_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::id space::get_tuple_id(isl::dim type) const
{
  return tuple_id(type);
}

std::string space::tuple_name(isl::dim type) const
{
  auto res = isl_space_get_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  std::string tmp(res);
  return tmp;
}

std::string space::get_tuple_name(isl::dim type) const
{
  return tuple_name(type);
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

isl::map space::universe_map() const
{
  auto res = isl_space_universe_map(copy());
  return manage(res);
}

isl::set space::universe_set() const
{
  auto res = isl_space_universe_set(copy());
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

isl::aff space::zero_aff_on_domain() const
{
  auto res = isl_space_zero_aff_on_domain(copy());
  return manage(res);
}

isl::multi_aff space::zero_multi_aff() const
{
  auto res = isl_space_zero_multi_aff(copy());
  return manage(res);
}

isl::multi_pw_aff space::zero_multi_pw_aff() const
{
  auto res = isl_space_zero_multi_pw_aff(copy());
  return manage(res);
}

isl::multi_union_pw_aff space::zero_multi_union_pw_aff() const
{
  auto res = isl_space_zero_multi_union_pw_aff(copy());
  return manage(res);
}

isl::multi_val space::zero_multi_val() const
{
  auto res = isl_space_zero_multi_val(copy());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const space &obj)
{
  char *str = isl_space_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

inline std::ostream &operator<<(std::ostream &os, const union_access_info &obj)
{
  char *str = isl_union_access_info_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::union_map union_flow::full_may_dependence() const
{
  auto res = isl_union_flow_get_full_may_dependence(get());
  return manage(res);
}

isl::union_map union_flow::get_full_may_dependence() const
{
  return full_may_dependence();
}

isl::union_map union_flow::full_must_dependence() const
{
  auto res = isl_union_flow_get_full_must_dependence(get());
  return manage(res);
}

isl::union_map union_flow::get_full_must_dependence() const
{
  return full_must_dependence();
}

isl::union_map union_flow::may_dependence() const
{
  auto res = isl_union_flow_get_may_dependence(get());
  return manage(res);
}

isl::union_map union_flow::get_may_dependence() const
{
  return may_dependence();
}

isl::union_map union_flow::may_no_source() const
{
  auto res = isl_union_flow_get_may_no_source(get());
  return manage(res);
}

isl::union_map union_flow::get_may_no_source() const
{
  return may_no_source();
}

isl::union_map union_flow::must_dependence() const
{
  auto res = isl_union_flow_get_must_dependence(get());
  return manage(res);
}

isl::union_map union_flow::get_must_dependence() const
{
  return must_dependence();
}

isl::union_map union_flow::must_no_source() const
{
  auto res = isl_union_flow_get_must_no_source(get());
  return manage(res);
}

isl::union_map union_flow::get_must_no_source() const
{
  return must_no_source();
}

inline std::ostream &operator<<(std::ostream &os, const union_flow &obj)
{
  char *str = isl_union_flow_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::union_map union_map::affine_hull() const
{
  auto res = isl_union_map_affine_hull(copy());
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

isl::map union_map::as_map() const
{
  auto res = isl_union_map_as_map(copy());
  return manage(res);
}

isl::multi_union_pw_aff union_map::as_multi_union_pw_aff() const
{
  auto res = isl_union_map_as_multi_union_pw_aff(copy());
  return manage(res);
}

isl::union_pw_multi_aff union_map::as_union_pw_multi_aff() const
{
  auto res = isl_union_map_as_union_pw_multi_aff(copy());
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

isl::union_map union_map::compute_divs() const
{
  auto res = isl_union_map_compute_divs(copy());
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

isl::union_map union_map::detect_equalities() const
{
  auto res = isl_union_map_detect_equalities(copy());
  return manage(res);
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

boolean union_map::every_map(const std::function<boolean(isl::map)> &test) const
{
  struct test_data {
    std::function<boolean(isl::map)> func;
  } test_data = { test };
  auto test_lambda = [](isl_map *arg_0, void *arg_1) -> isl_bool {
    auto *data = static_cast<struct test_data *>(arg_1);
    auto ret = (data->func)(manage_copy(arg_0));
    return ret.release();
  };
  auto res = isl_union_map_every_map(get(), test_lambda, &test_data);
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

isl::union_map union_map::fixed_power(isl::val exp) const
{
  auto res = isl_union_map_fixed_power_val(copy(), exp.release());
  return manage(res);
}

isl::union_map union_map::fixed_power(long exp) const
{
  return this->fixed_power(isl::val(ctx(), exp));
}

isl::union_map union_map::flat_range_product(isl::union_map umap2) const
{
  auto res = isl_union_map_flat_range_product(copy(), umap2.release());
  return manage(res);
}

stat union_map::foreach_map(const std::function<stat(isl::map)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::map)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_map *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
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

isl::union_map union_map::from(isl::union_pw_multi_aff upma)
{
  auto res = isl_union_map_from_union_pw_multi_aff(upma.release());
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

isl::map_list union_map::map_list() const
{
  auto res = isl_union_map_get_map_list(get());
  return manage(res);
}

isl::map_list union_map::get_map_list() const
{
  return map_list();
}

isl::set union_map::params() const
{
  auto res = isl_union_map_params(copy());
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

isl::union_map union_map::reverse() const
{
  auto res = isl_union_map_reverse(copy());
  return manage(res);
}

isl::space union_map::space() const
{
  auto res = isl_union_map_get_space(get());
  return manage(res);
}

isl::space union_map::get_space() const
{
  return space();
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

inline std::ostream &operator<<(std::ostream &os, const union_map &obj)
{
  char *str = isl_union_map_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::multi_union_pw_aff union_pw_aff::add(const isl::multi_union_pw_aff &multi2) const
{
  return isl::multi_union_pw_aff(*this).add(multi2);
}

isl::union_pw_aff union_pw_aff::add(isl::union_pw_aff upa2) const
{
  auto res = isl_union_pw_aff_add(copy(), upa2.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_aff::add(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_multi_aff(*this).add(upma2);
}

isl::union_pw_aff union_pw_aff::add(const isl::aff &upa2) const
{
  return this->add(isl::union_pw_aff(upa2));
}

isl::union_pw_aff union_pw_aff::add(const isl::pw_aff &upa2) const
{
  return this->add(isl::union_pw_aff(upa2));
}

isl::union_pw_multi_aff union_pw_aff::add_pw_multi_aff(const isl::pw_multi_aff &pma) const
{
  return isl::union_pw_multi_aff(*this).add_pw_multi_aff(pma);
}

isl::union_pw_multi_aff union_pw_aff::apply(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_multi_aff(*this).apply(upma2);
}

isl::multi_union_pw_aff union_pw_aff::as_multi_union_pw_aff() const
{
  return isl::union_pw_multi_aff(*this).as_multi_union_pw_aff();
}

isl::pw_multi_aff union_pw_aff::as_pw_multi_aff() const
{
  return isl::union_pw_multi_aff(*this).as_pw_multi_aff();
}

isl::union_map union_pw_aff::as_union_map() const
{
  return isl::union_pw_multi_aff(*this).as_union_map();
}

isl::union_pw_aff union_pw_aff::at(int pos) const
{
  return isl::multi_union_pw_aff(*this).at(pos);
}

isl::union_set union_pw_aff::bind(const isl::multi_id &tuple) const
{
  return isl::multi_union_pw_aff(*this).bind(tuple);
}

isl::union_set union_pw_aff::bind(isl::id id) const
{
  auto res = isl_union_pw_aff_bind_id(copy(), id.release());
  return manage(res);
}

isl::union_set union_pw_aff::bind(const std::string &id) const
{
  return this->bind(isl::id(ctx(), id));
}

isl::union_pw_aff union_pw_aff::coalesce() const
{
  auto res = isl_union_pw_aff_coalesce(copy());
  return manage(res);
}

class size union_pw_aff::dim(isl::dim type) const
{
  return isl::multi_union_pw_aff(*this).dim(type);
}

isl::union_set union_pw_aff::domain() const
{
  auto res = isl_union_pw_aff_domain(copy());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::empty(isl::space space)
{
  auto res = isl_union_pw_aff_empty(space.release());
  return manage(res);
}

isl::pw_multi_aff union_pw_aff::extract_pw_multi_aff(const isl::space &space) const
{
  return isl::union_pw_multi_aff(*this).extract_pw_multi_aff(space);
}

isl::multi_union_pw_aff union_pw_aff::flat_range_product(const isl::multi_union_pw_aff &multi2) const
{
  return isl::multi_union_pw_aff(*this).flat_range_product(multi2);
}

isl::union_pw_multi_aff union_pw_aff::flat_range_product(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_multi_aff(*this).flat_range_product(upma2);
}

stat union_pw_aff::foreach_pw_aff(const std::function<stat(isl::pw_aff)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::pw_aff)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_pw_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_union_pw_aff_foreach_pw_aff(get(), fn_lambda, &fn_data);
  return manage(res);
}

isl::union_pw_aff union_pw_aff::gist(isl::union_set context) const
{
  auto res = isl_union_pw_aff_gist(copy(), context.release());
  return manage(res);
}

boolean union_pw_aff::has_range_tuple_id() const
{
  return isl::multi_union_pw_aff(*this).has_range_tuple_id();
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

boolean union_pw_aff::involves_locals() const
{
  return isl::union_pw_multi_aff(*this).involves_locals();
}

boolean union_pw_aff::involves_nan() const
{
  return isl::multi_union_pw_aff(*this).involves_nan();
}

boolean union_pw_aff::isa_pw_multi_aff() const
{
  return isl::union_pw_multi_aff(*this).isa_pw_multi_aff();
}

isl::union_pw_aff_list union_pw_aff::list() const
{
  return isl::multi_union_pw_aff(*this).list();
}

isl::multi_union_pw_aff union_pw_aff::neg() const
{
  return isl::multi_union_pw_aff(*this).neg();
}

boolean union_pw_aff::plain_is_empty() const
{
  return isl::union_pw_multi_aff(*this).plain_is_empty();
}

boolean union_pw_aff::plain_is_equal(const isl::multi_union_pw_aff &multi2) const
{
  return isl::multi_union_pw_aff(*this).plain_is_equal(multi2);
}

isl::union_pw_multi_aff union_pw_aff::preimage_domain_wrapped_domain(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_multi_aff(*this).preimage_domain_wrapped_domain(upma2);
}

isl::union_pw_aff union_pw_aff::pullback(isl::union_pw_multi_aff upma) const
{
  auto res = isl_union_pw_aff_pullback_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

isl::pw_multi_aff_list union_pw_aff::pw_multi_aff_list() const
{
  return isl::union_pw_multi_aff(*this).pw_multi_aff_list();
}

isl::union_pw_multi_aff union_pw_aff::range_factor_domain() const
{
  return isl::union_pw_multi_aff(*this).range_factor_domain();
}

isl::union_pw_multi_aff union_pw_aff::range_factor_range() const
{
  return isl::union_pw_multi_aff(*this).range_factor_range();
}

isl::multi_union_pw_aff union_pw_aff::range_product(const isl::multi_union_pw_aff &multi2) const
{
  return isl::multi_union_pw_aff(*this).range_product(multi2);
}

isl::union_pw_multi_aff union_pw_aff::range_product(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_multi_aff(*this).range_product(upma2);
}

isl::id union_pw_aff::range_tuple_id() const
{
  return isl::multi_union_pw_aff(*this).range_tuple_id();
}

isl::multi_union_pw_aff union_pw_aff::reset_range_tuple_id() const
{
  return isl::multi_union_pw_aff(*this).reset_range_tuple_id();
}

isl::multi_union_pw_aff union_pw_aff::reset_tuple_id(isl::dim type) const
{
  return isl::multi_union_pw_aff(*this).reset_tuple_id(type);
}

isl::multi_union_pw_aff union_pw_aff::scale(const isl::multi_val &mv) const
{
  return isl::multi_union_pw_aff(*this).scale(mv);
}

isl::multi_union_pw_aff union_pw_aff::scale(const isl::val &v) const
{
  return isl::multi_union_pw_aff(*this).scale(v);
}

isl::multi_union_pw_aff union_pw_aff::scale(long v) const
{
  return this->scale(isl::val(ctx(), v));
}

isl::multi_union_pw_aff union_pw_aff::scale_down(const isl::multi_val &mv) const
{
  return isl::multi_union_pw_aff(*this).scale_down(mv);
}

isl::multi_union_pw_aff union_pw_aff::scale_down(const isl::val &v) const
{
  return isl::multi_union_pw_aff(*this).scale_down(v);
}

isl::multi_union_pw_aff union_pw_aff::scale_down(long v) const
{
  return this->scale_down(isl::val(ctx(), v));
}

isl::multi_union_pw_aff union_pw_aff::set_at(int pos, const isl::union_pw_aff &el) const
{
  return isl::multi_union_pw_aff(*this).set_at(pos, el);
}

isl::multi_union_pw_aff union_pw_aff::set_range_tuple(const isl::id &id) const
{
  return isl::multi_union_pw_aff(*this).set_range_tuple(id);
}

isl::multi_union_pw_aff union_pw_aff::set_range_tuple(const std::string &id) const
{
  return this->set_range_tuple(isl::id(ctx(), id));
}

isl::multi_union_pw_aff union_pw_aff::set_union_pw_aff(int pos, const isl::union_pw_aff &el) const
{
  return isl::multi_union_pw_aff(*this).set_union_pw_aff(pos, el);
}

class size union_pw_aff::size() const
{
  return isl::multi_union_pw_aff(*this).size();
}

isl::space union_pw_aff::space() const
{
  auto res = isl_union_pw_aff_get_space(get());
  return manage(res);
}

isl::space union_pw_aff::get_space() const
{
  return space();
}

isl::multi_union_pw_aff union_pw_aff::sub(const isl::multi_union_pw_aff &multi2) const
{
  return isl::multi_union_pw_aff(*this).sub(multi2);
}

isl::union_pw_aff union_pw_aff::sub(isl::union_pw_aff upa2) const
{
  auto res = isl_union_pw_aff_sub(copy(), upa2.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_aff::sub(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_multi_aff(*this).sub(upma2);
}

isl::union_pw_aff union_pw_aff::sub(const isl::aff &upa2) const
{
  return this->sub(isl::union_pw_aff(upa2));
}

isl::union_pw_aff union_pw_aff::sub(const isl::pw_aff &upa2) const
{
  return this->sub(isl::union_pw_aff(upa2));
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

isl::union_pw_aff_list union_pw_aff::to_list() const
{
  auto res = isl_union_pw_aff_to_list(copy());
  return manage(res);
}

isl::multi_union_pw_aff union_pw_aff::union_add(const isl::multi_union_pw_aff &mupa2) const
{
  return isl::multi_union_pw_aff(*this).union_add(mupa2);
}

isl::union_pw_aff union_pw_aff::union_add(isl::union_pw_aff upa2) const
{
  auto res = isl_union_pw_aff_union_add(copy(), upa2.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_aff::union_add(const isl::union_pw_multi_aff &upma2) const
{
  return isl::union_pw_multi_aff(*this).union_add(upma2);
}

isl::union_pw_aff union_pw_aff::union_add(const isl::aff &upa2) const
{
  return this->union_add(isl::union_pw_aff(upa2));
}

isl::union_pw_aff union_pw_aff::union_add(const isl::pw_aff &upa2) const
{
  return this->union_add(isl::union_pw_aff(upa2));
}

inline std::ostream &operator<<(std::ostream &os, const union_pw_aff &obj)
{
  char *str = isl_union_pw_aff_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

union_pw_aff_list::union_pw_aff_list(isl::ctx ctx, int n)
{
  auto res = isl_union_pw_aff_list_alloc(ctx.release(), n);
  ptr = res;
}

union_pw_aff_list::union_pw_aff_list(isl::union_pw_aff el)
{
  auto res = isl_union_pw_aff_list_from_union_pw_aff(el.release());
  ptr = res;
}

union_pw_aff_list::union_pw_aff_list(isl::ctx ctx, const std::string &str)
{
  auto res = isl_union_pw_aff_list_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

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

isl::union_pw_aff_list union_pw_aff_list::add(isl::union_pw_aff el) const
{
  auto res = isl_union_pw_aff_list_add(copy(), el.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff_list::at(int index) const
{
  auto res = isl_union_pw_aff_list_get_at(get(), index);
  return manage(res);
}

isl::union_pw_aff union_pw_aff_list::get_at(int index) const
{
  return at(index);
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

stat union_pw_aff_list::foreach(const std::function<stat(isl::union_pw_aff)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::union_pw_aff)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_union_pw_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_union_pw_aff_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

isl::union_pw_aff_list union_pw_aff_list::insert(unsigned int pos, isl::union_pw_aff el) const
{
  auto res = isl_union_pw_aff_list_insert(copy(), pos, el.release());
  return manage(res);
}

class size union_pw_aff_list::size() const
{
  auto res = isl_union_pw_aff_list_size(get());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const union_pw_aff_list &obj)
{
  char *str = isl_union_pw_aff_list_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::union_pw_multi_aff union_pw_multi_aff::apply(isl::union_pw_multi_aff upma2) const
{
  auto res = isl_union_pw_multi_aff_apply_union_pw_multi_aff(copy(), upma2.release());
  return manage(res);
}

isl::multi_union_pw_aff union_pw_multi_aff::as_multi_union_pw_aff() const
{
  auto res = isl_union_pw_multi_aff_as_multi_union_pw_aff(copy());
  return manage(res);
}

isl::pw_multi_aff union_pw_multi_aff::as_pw_multi_aff() const
{
  auto res = isl_union_pw_multi_aff_as_pw_multi_aff(copy());
  return manage(res);
}

isl::union_map union_pw_multi_aff::as_union_map() const
{
  auto res = isl_union_pw_multi_aff_as_union_map(copy());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::coalesce() const
{
  auto res = isl_union_pw_multi_aff_coalesce(copy());
  return manage(res);
}

isl::union_set union_pw_multi_aff::domain() const
{
  auto res = isl_union_pw_multi_aff_domain(copy());
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

isl::pw_multi_aff union_pw_multi_aff::extract_pw_multi_aff(isl::space space) const
{
  auto res = isl_union_pw_multi_aff_extract_pw_multi_aff(get(), space.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::flat_range_product(isl::union_pw_multi_aff upma2) const
{
  auto res = isl_union_pw_multi_aff_flat_range_product(copy(), upma2.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::gist(isl::union_set context) const
{
  auto res = isl_union_pw_multi_aff_gist(copy(), context.release());
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

boolean union_pw_multi_aff::isa_pw_multi_aff() const
{
  auto res = isl_union_pw_multi_aff_isa_pw_multi_aff(get());
  return manage(res);
}

boolean union_pw_multi_aff::plain_is_empty() const
{
  auto res = isl_union_pw_multi_aff_plain_is_empty(get());
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

isl::pw_multi_aff_list union_pw_multi_aff::pw_multi_aff_list() const
{
  auto res = isl_union_pw_multi_aff_get_pw_multi_aff_list(get());
  return manage(res);
}

isl::pw_multi_aff_list union_pw_multi_aff::get_pw_multi_aff_list() const
{
  return pw_multi_aff_list();
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

isl::space union_pw_multi_aff::space() const
{
  auto res = isl_union_pw_multi_aff_get_space(get());
  return manage(res);
}

isl::space union_pw_multi_aff::get_space() const
{
  return space();
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

inline std::ostream &operator<<(std::ostream &os, const union_pw_multi_aff &obj)
{
  char *str = isl_union_pw_multi_aff_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

isl::union_set union_set::affine_hull() const
{
  auto res = isl_union_set_affine_hull(copy());
  return manage(res);
}

isl::union_set union_set::apply(isl::union_map umap) const
{
  auto res = isl_union_set_apply(copy(), umap.release());
  return manage(res);
}

isl::set union_set::as_set() const
{
  auto res = isl_union_set_as_set(copy());
  return manage(res);
}

isl::union_set union_set::coalesce() const
{
  auto res = isl_union_set_coalesce(copy());
  return manage(res);
}

isl::union_set union_set::compute_divs() const
{
  auto res = isl_union_set_compute_divs(copy());
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

isl::union_set union_set::empty(isl::ctx ctx)
{
  auto res = isl_union_set_empty_ctx(ctx.release());
  return manage(res);
}

boolean union_set::every_set(const std::function<boolean(isl::set)> &test) const
{
  struct test_data {
    std::function<boolean(isl::set)> func;
  } test_data = { test };
  auto test_lambda = [](isl_set *arg_0, void *arg_1) -> isl_bool {
    auto *data = static_cast<struct test_data *>(arg_1);
    auto ret = (data->func)(manage_copy(arg_0));
    return ret.release();
  };
  auto res = isl_union_set_every_set(get(), test_lambda, &test_data);
  return manage(res);
}

isl::set union_set::extract_set(isl::space space) const
{
  auto res = isl_union_set_extract_set(get(), space.release());
  return manage(res);
}

stat union_set::foreach_point(const std::function<stat(isl::point)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::point)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_point *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_union_set_foreach_point(get(), fn_lambda, &fn_data);
  return manage(res);
}

stat union_set::foreach_set(const std::function<stat(isl::set)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::set)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_set *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_union_set_foreach_set(get(), fn_lambda, &fn_data);
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

isl::point union_set::sample_point() const
{
  auto res = isl_union_set_sample_point(copy());
  return manage(res);
}

isl::set_list union_set::set_list() const
{
  auto res = isl_union_set_get_set_list(get());
  return manage(res);
}

isl::set_list union_set::get_set_list() const
{
  return set_list();
}

isl::space union_set::space() const
{
  auto res = isl_union_set_get_space(get());
  return manage(res);
}

isl::space union_set::get_space() const
{
  return space();
}

isl::union_set union_set::subtract(isl::union_set uset2) const
{
  auto res = isl_union_set_subtract(copy(), uset2.release());
  return manage(res);
}

isl::union_set_list union_set::to_list() const
{
  auto res = isl_union_set_to_list(copy());
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

inline std::ostream &operator<<(std::ostream &os, const union_set &obj)
{
  char *str = isl_union_set_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

union_set_list::union_set_list(isl::ctx ctx, int n)
{
  auto res = isl_union_set_list_alloc(ctx.release(), n);
  ptr = res;
}

union_set_list::union_set_list(isl::union_set el)
{
  auto res = isl_union_set_list_from_union_set(el.release());
  ptr = res;
}

union_set_list::union_set_list(isl::ctx ctx, const std::string &str)
{
  auto res = isl_union_set_list_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

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

isl::union_set_list union_set_list::add(isl::union_set el) const
{
  auto res = isl_union_set_list_add(copy(), el.release());
  return manage(res);
}

isl::union_set union_set_list::at(int index) const
{
  auto res = isl_union_set_list_get_at(get(), index);
  return manage(res);
}

isl::union_set union_set_list::get_at(int index) const
{
  return at(index);
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

stat union_set_list::foreach(const std::function<stat(isl::union_set)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::union_set)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_union_set *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_union_set_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

isl::union_set_list union_set_list::insert(unsigned int pos, isl::union_set el) const
{
  auto res = isl_union_set_list_insert(copy(), pos, el.release());
  return manage(res);
}

class size union_set_list::size() const
{
  auto res = isl_union_set_list_size(get());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const union_set_list &obj)
{
  char *str = isl_union_set_list_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

boolean val::abs_eq(long v2) const
{
  return this->abs_eq(isl::val(ctx(), v2));
}

isl::val val::add(isl::val v2) const
{
  auto res = isl_val_add(copy(), v2.release());
  return manage(res);
}

isl::val val::add(long v2) const
{
  return this->add(isl::val(ctx(), v2));
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

long val::den_si() const
{
  auto res = isl_val_get_den_si(get());
  return res;
}

long val::get_den_si() const
{
  return den_si();
}

isl::val val::div(isl::val v2) const
{
  auto res = isl_val_div(copy(), v2.release());
  return manage(res);
}

isl::val val::div(long v2) const
{
  return this->div(isl::val(ctx(), v2));
}

boolean val::eq(const isl::val &v2) const
{
  auto res = isl_val_eq(get(), v2.get());
  return manage(res);
}

boolean val::eq(long v2) const
{
  return this->eq(isl::val(ctx(), v2));
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

isl::val val::gcd(long v2) const
{
  return this->gcd(isl::val(ctx(), v2));
}

boolean val::ge(const isl::val &v2) const
{
  auto res = isl_val_ge(get(), v2.get());
  return manage(res);
}

boolean val::ge(long v2) const
{
  return this->ge(isl::val(ctx(), v2));
}

boolean val::gt(const isl::val &v2) const
{
  auto res = isl_val_gt(get(), v2.get());
  return manage(res);
}

boolean val::gt(long v2) const
{
  return this->gt(isl::val(ctx(), v2));
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

boolean val::is_divisible_by(long v2) const
{
  return this->is_divisible_by(isl::val(ctx(), v2));
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

boolean val::le(long v2) const
{
  return this->le(isl::val(ctx(), v2));
}

boolean val::lt(const isl::val &v2) const
{
  auto res = isl_val_lt(get(), v2.get());
  return manage(res);
}

boolean val::lt(long v2) const
{
  return this->lt(isl::val(ctx(), v2));
}

isl::val val::max(isl::val v2) const
{
  auto res = isl_val_max(copy(), v2.release());
  return manage(res);
}

isl::val val::max(long v2) const
{
  return this->max(isl::val(ctx(), v2));
}

isl::val val::min(isl::val v2) const
{
  auto res = isl_val_min(copy(), v2.release());
  return manage(res);
}

isl::val val::min(long v2) const
{
  return this->min(isl::val(ctx(), v2));
}

isl::val val::mod(isl::val v2) const
{
  auto res = isl_val_mod(copy(), v2.release());
  return manage(res);
}

isl::val val::mod(long v2) const
{
  return this->mod(isl::val(ctx(), v2));
}

isl::val val::mul(isl::val v2) const
{
  auto res = isl_val_mul(copy(), v2.release());
  return manage(res);
}

isl::val val::mul(long v2) const
{
  return this->mul(isl::val(ctx(), v2));
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

boolean val::ne(long v2) const
{
  return this->ne(isl::val(ctx(), v2));
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

long val::num_si() const
{
  auto res = isl_val_get_num_si(get());
  return res;
}

long val::get_num_si() const
{
  return num_si();
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

isl::val val::sub(long v2) const
{
  return this->sub(isl::val(ctx(), v2));
}

isl::val_list val::to_list() const
{
  auto res = isl_val_to_list(copy());
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

inline std::ostream &operator<<(std::ostream &os, const val &obj)
{
  char *str = isl_val_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
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

val_list::val_list(isl::ctx ctx, int n)
{
  auto res = isl_val_list_alloc(ctx.release(), n);
  ptr = res;
}

val_list::val_list(isl::val el)
{
  auto res = isl_val_list_from_val(el.release());
  ptr = res;
}

val_list::val_list(isl::ctx ctx, const std::string &str)
{
  auto res = isl_val_list_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

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

isl::val_list val_list::add(isl::val el) const
{
  auto res = isl_val_list_add(copy(), el.release());
  return manage(res);
}

isl::val_list val_list::add(long el) const
{
  return this->add(isl::val(ctx(), el));
}

isl::val val_list::at(int index) const
{
  auto res = isl_val_list_get_at(get(), index);
  return manage(res);
}

isl::val val_list::get_at(int index) const
{
  return at(index);
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

stat val_list::foreach(const std::function<stat(isl::val)> &fn) const
{
  struct fn_data {
    std::function<stat(isl::val)> func;
  } fn_data = { fn };
  auto fn_lambda = [](isl_val *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    auto ret = (data->func)(manage(arg_0));
    return ret.release();
  };
  auto res = isl_val_list_foreach(get(), fn_lambda, &fn_data);
  return manage(res);
}

isl::val_list val_list::insert(unsigned int pos, isl::val el) const
{
  auto res = isl_val_list_insert(copy(), pos, el.release());
  return manage(res);
}

isl::val_list val_list::insert(unsigned int pos, long el) const
{
  return this->insert(pos, isl::val(ctx(), el));
}

class size val_list::size() const
{
  auto res = isl_val_list_size(get());
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const val_list &obj)
{
  char *str = isl_val_list_to_str(obj.get());
  if (!str) {
    os.setstate(std::ios_base::badbit);
    return os;
  }
  os << str;
  free(str);
  return os;
}
} // namespace isl

#endif /* ISL_CPP_CHECKED */
