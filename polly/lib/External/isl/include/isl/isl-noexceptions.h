/// These are automatically generated C++ bindings for isl.
///
/// isl is a library for computing with integer sets and maps described by
/// Presburger formulas. On top of this, isl provides various tools for
/// polyhedral compilation, ranging from dependence analysis over scheduling
/// to AST generation.

#ifndef ISL_CPP_NOEXCEPTIONS
#define ISL_CPP_NOEXCEPTIONS

#include <isl/aff.h>
#include <isl/ast_build.h>
#include <isl/constraint.h>
#include <isl/flow.h>
#include <isl/id.h>
#include <isl/ilp.h>
#include <isl/map.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/val.h>
#include <isl/polynomial.h>

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
  } while (0)

class boolean {
private:
  isl_bool val;

  friend isl::boolean manage(isl_bool val);
  boolean(isl_bool val): val(val) {}
public:
  boolean()
      : val(isl_bool_error) {}

  /* implicit */ boolean(bool val)
      : val(val ? isl_bool_true : isl_bool_false) {}

  bool is_error() const { return val == isl_bool_error; }
  bool is_false() const { return val == isl_bool_false; }
  bool is_true() const { return val == isl_bool_true; }

  operator bool() const {
    ISLPP_ASSERT(!is_error(), "IMPLEMENTATION ERROR: Unhandled error state");
    return is_true();
  }

  boolean operator!() const {
    if (is_error())
      return *this;
    return !is_true();
  }
};

inline isl::boolean manage(isl_bool val) {
  return isl::boolean(val);
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

enum class stat {
  ok = isl_stat_ok,
  error = isl_stat_error
};

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
class band_list;
class basic_map;
class basic_map_list;
class basic_set;
class basic_set_list;
class constraint;
class constraint_list;
class id;
class id_list;
class id_to_ast_expr;
class local_space;
class map;
class map_list;
class multi_aff;
class multi_pw_aff;
class multi_union_pw_aff;
class multi_val;
class point;
class pw_aff;
class pw_aff_list;
class pw_multi_aff;
class pw_qpolynomial;
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

// declarations for isl::aff
inline isl::aff manage(__isl_take isl_aff *ptr);
inline isl::aff give(__isl_take isl_aff *ptr);


class aff {
  friend inline isl::aff manage(__isl_take isl_aff *ptr);

  isl_aff *ptr = nullptr;

  inline explicit aff(__isl_take isl_aff *ptr);

public:
  inline /* implicit */ aff();
  inline /* implicit */ aff(const isl::aff &obj);
  inline /* implicit */ aff(std::nullptr_t);
  inline explicit aff(isl::local_space ls);
  inline explicit aff(isl::local_space ls, isl::val val);
  inline explicit aff(isl::ctx ctx, const std::string &str);
  inline isl::aff &operator=(isl::aff obj);
  inline ~aff();
  inline __isl_give isl_aff *copy() const &;
  inline __isl_give isl_aff *copy() && = delete;
  inline __isl_keep isl_aff *get() const;
  inline __isl_give isl_aff *release();
  inline bool is_null() const;
  inline __isl_keep isl_aff *keep() const;
  inline __isl_give isl_aff *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::aff add(isl::aff aff2) const;
  inline isl::aff add_coefficient_si(isl::dim type, int pos, int v) const;
  inline isl::aff add_coefficient_val(isl::dim type, int pos, isl::val v) const;
  inline isl::aff add_constant_num_si(int v) const;
  inline isl::aff add_constant_si(int v) const;
  inline isl::aff add_constant_val(isl::val v) const;
  inline isl::aff add_dims(isl::dim type, unsigned int n) const;
  inline isl::aff align_params(isl::space model) const;
  inline isl::aff ceil() const;
  inline int coefficient_sgn(isl::dim type, int pos) const;
  inline int dim(isl::dim type) const;
  inline isl::aff div(isl::aff aff2) const;
  inline isl::aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::basic_set eq_basic_set(isl::aff aff2) const;
  inline isl::set eq_set(isl::aff aff2) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::aff floor() const;
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
  inline isl::boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::boolean is_cst() const;
  inline isl::boolean is_nan() const;
  inline isl::basic_set le_basic_set(isl::aff aff2) const;
  inline isl::set le_set(isl::aff aff2) const;
  inline isl::basic_set lt_basic_set(isl::aff aff2) const;
  inline isl::set lt_set(isl::aff aff2) const;
  inline isl::aff mod(isl::val mod) const;
  inline isl::aff move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline isl::aff mul(isl::aff aff2) const;
  static inline isl::aff nan_on_domain(isl::local_space ls);
  inline isl::set ne_set(isl::aff aff2) const;
  inline isl::aff neg() const;
  inline isl::basic_set neg_basic_set() const;
  inline isl::boolean plain_is_equal(const isl::aff &aff2) const;
  inline isl::boolean plain_is_zero() const;
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
  static inline isl::aff var_on_domain(isl::local_space ls, isl::dim type, unsigned int pos);
  inline isl::basic_set zero_basic_set() const;
};

// declarations for isl::aff_list
inline isl::aff_list manage(__isl_take isl_aff_list *ptr);
inline isl::aff_list give(__isl_take isl_aff_list *ptr);


class aff_list {
  friend inline isl::aff_list manage(__isl_take isl_aff_list *ptr);

  isl_aff_list *ptr = nullptr;

  inline explicit aff_list(__isl_take isl_aff_list *ptr);

public:
  inline /* implicit */ aff_list();
  inline /* implicit */ aff_list(const isl::aff_list &obj);
  inline /* implicit */ aff_list(std::nullptr_t);
  inline isl::aff_list &operator=(isl::aff_list obj);
  inline ~aff_list();
  inline __isl_give isl_aff_list *copy() const &;
  inline __isl_give isl_aff_list *copy() && = delete;
  inline __isl_keep isl_aff_list *get() const;
  inline __isl_give isl_aff_list *release();
  inline bool is_null() const;
  inline __isl_keep isl_aff_list *keep() const;
  inline __isl_give isl_aff_list *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

};

// declarations for isl::ast_build
inline isl::ast_build manage(__isl_take isl_ast_build *ptr);
inline isl::ast_build give(__isl_take isl_ast_build *ptr);


class ast_build {
  friend inline isl::ast_build manage(__isl_take isl_ast_build *ptr);

  isl_ast_build *ptr = nullptr;

  inline explicit ast_build(__isl_take isl_ast_build *ptr);

public:
  inline /* implicit */ ast_build();
  inline /* implicit */ ast_build(const isl::ast_build &obj);
  inline /* implicit */ ast_build(std::nullptr_t);
  inline explicit ast_build(isl::ctx ctx);
  inline isl::ast_build &operator=(isl::ast_build obj);
  inline ~ast_build();
  inline __isl_give isl_ast_build *copy() const &;
  inline __isl_give isl_ast_build *copy() && = delete;
  inline __isl_keep isl_ast_build *get() const;
  inline __isl_give isl_ast_build *release();
  inline bool is_null() const;
  inline __isl_keep isl_ast_build *keep() const;
  inline __isl_give isl_ast_build *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;

  inline isl::ast_expr access_from(isl::pw_multi_aff pma) const;
  inline isl::ast_expr access_from(isl::multi_pw_aff mpa) const;
  inline isl::ast_node ast_from_schedule(isl::union_map schedule) const;
  inline isl::ast_expr call_from(isl::pw_multi_aff pma) const;
  inline isl::ast_expr call_from(isl::multi_pw_aff mpa) const;
  inline isl::ast_expr expr_from(isl::set set) const;
  inline isl::ast_expr expr_from(isl::pw_aff pa) const;
  static inline isl::ast_build from_context(isl::set set);
  inline isl::union_map get_schedule() const;
  inline isl::space get_schedule_space() const;
  inline isl::ast_node node_from_schedule(isl::schedule schedule) const;
  inline isl::ast_node node_from_schedule_map(isl::union_map schedule) const;
  inline isl::ast_build restrict(isl::set set) const;
};

// declarations for isl::ast_expr
inline isl::ast_expr manage(__isl_take isl_ast_expr *ptr);
inline isl::ast_expr give(__isl_take isl_ast_expr *ptr);


class ast_expr {
  friend inline isl::ast_expr manage(__isl_take isl_ast_expr *ptr);

  isl_ast_expr *ptr = nullptr;

  inline explicit ast_expr(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr();
  inline /* implicit */ ast_expr(const isl::ast_expr &obj);
  inline /* implicit */ ast_expr(std::nullptr_t);
  inline isl::ast_expr &operator=(isl::ast_expr obj);
  inline ~ast_expr();
  inline __isl_give isl_ast_expr *copy() const &;
  inline __isl_give isl_ast_expr *copy() && = delete;
  inline __isl_keep isl_ast_expr *get() const;
  inline __isl_give isl_ast_expr *release();
  inline bool is_null() const;
  inline __isl_keep isl_ast_expr *keep() const;
  inline __isl_give isl_ast_expr *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
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
  inline isl::val get_val() const;
  inline isl::ast_expr gt(isl::ast_expr expr2) const;
  inline isl::boolean is_equal(const isl::ast_expr &expr2) const;
  inline isl::ast_expr le(isl::ast_expr expr2) const;
  inline isl::ast_expr lt(isl::ast_expr expr2) const;
  inline isl::ast_expr mul(isl::ast_expr expr2) const;
  inline isl::ast_expr neg() const;
  inline isl::ast_expr pdiv_q(isl::ast_expr expr2) const;
  inline isl::ast_expr pdiv_r(isl::ast_expr expr2) const;
  inline isl::ast_expr set_op_arg(int pos, isl::ast_expr arg) const;
  inline isl::ast_expr sub(isl::ast_expr expr2) const;
  inline isl::ast_expr substitute_ids(isl::id_to_ast_expr id2expr) const;
  inline std::string to_C_str() const;
};

// declarations for isl::ast_expr_list
inline isl::ast_expr_list manage(__isl_take isl_ast_expr_list *ptr);
inline isl::ast_expr_list give(__isl_take isl_ast_expr_list *ptr);


class ast_expr_list {
  friend inline isl::ast_expr_list manage(__isl_take isl_ast_expr_list *ptr);

  isl_ast_expr_list *ptr = nullptr;

  inline explicit ast_expr_list(__isl_take isl_ast_expr_list *ptr);

public:
  inline /* implicit */ ast_expr_list();
  inline /* implicit */ ast_expr_list(const isl::ast_expr_list &obj);
  inline /* implicit */ ast_expr_list(std::nullptr_t);
  inline isl::ast_expr_list &operator=(isl::ast_expr_list obj);
  inline ~ast_expr_list();
  inline __isl_give isl_ast_expr_list *copy() const &;
  inline __isl_give isl_ast_expr_list *copy() && = delete;
  inline __isl_keep isl_ast_expr_list *get() const;
  inline __isl_give isl_ast_expr_list *release();
  inline bool is_null() const;
  inline __isl_keep isl_ast_expr_list *keep() const;
  inline __isl_give isl_ast_expr_list *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

};

// declarations for isl::ast_node
inline isl::ast_node manage(__isl_take isl_ast_node *ptr);
inline isl::ast_node give(__isl_take isl_ast_node *ptr);


class ast_node {
  friend inline isl::ast_node manage(__isl_take isl_ast_node *ptr);

  isl_ast_node *ptr = nullptr;

  inline explicit ast_node(__isl_take isl_ast_node *ptr);

public:
  inline /* implicit */ ast_node();
  inline /* implicit */ ast_node(const isl::ast_node &obj);
  inline /* implicit */ ast_node(std::nullptr_t);
  inline isl::ast_node &operator=(isl::ast_node obj);
  inline ~ast_node();
  inline __isl_give isl_ast_node *copy() const &;
  inline __isl_give isl_ast_node *copy() && = delete;
  inline __isl_keep isl_ast_node *get() const;
  inline __isl_give isl_ast_node *release();
  inline bool is_null() const;
  inline __isl_keep isl_ast_node *keep() const;
  inline __isl_give isl_ast_node *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  static inline isl::ast_node alloc_user(isl::ast_expr expr);
  inline isl::ast_node_list block_get_children() const;
  inline isl::ast_node for_get_body() const;
  inline isl::ast_expr for_get_cond() const;
  inline isl::ast_expr for_get_inc() const;
  inline isl::ast_expr for_get_init() const;
  inline isl::ast_expr for_get_iterator() const;
  inline isl::boolean for_is_degenerate() const;
  inline isl::id get_annotation() const;
  inline isl::ast_expr if_get_cond() const;
  inline isl::ast_node if_get_else() const;
  inline isl::ast_node if_get_then() const;
  inline isl::boolean if_has_else() const;
  inline isl::id mark_get_id() const;
  inline isl::ast_node mark_get_node() const;
  inline isl::ast_node set_annotation(isl::id annotation) const;
  inline std::string to_C_str() const;
  inline isl::ast_expr user_get_expr() const;
};

// declarations for isl::ast_node_list
inline isl::ast_node_list manage(__isl_take isl_ast_node_list *ptr);
inline isl::ast_node_list give(__isl_take isl_ast_node_list *ptr);


class ast_node_list {
  friend inline isl::ast_node_list manage(__isl_take isl_ast_node_list *ptr);

  isl_ast_node_list *ptr = nullptr;

  inline explicit ast_node_list(__isl_take isl_ast_node_list *ptr);

public:
  inline /* implicit */ ast_node_list();
  inline /* implicit */ ast_node_list(const isl::ast_node_list &obj);
  inline /* implicit */ ast_node_list(std::nullptr_t);
  inline isl::ast_node_list &operator=(isl::ast_node_list obj);
  inline ~ast_node_list();
  inline __isl_give isl_ast_node_list *copy() const &;
  inline __isl_give isl_ast_node_list *copy() && = delete;
  inline __isl_keep isl_ast_node_list *get() const;
  inline __isl_give isl_ast_node_list *release();
  inline bool is_null() const;
  inline __isl_keep isl_ast_node_list *keep() const;
  inline __isl_give isl_ast_node_list *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

};

// declarations for isl::band_list
inline isl::band_list manage(__isl_take isl_band_list *ptr);
inline isl::band_list give(__isl_take isl_band_list *ptr);


class band_list {
  friend inline isl::band_list manage(__isl_take isl_band_list *ptr);

  isl_band_list *ptr = nullptr;

  inline explicit band_list(__isl_take isl_band_list *ptr);

public:
  inline /* implicit */ band_list();
  inline /* implicit */ band_list(const isl::band_list &obj);
  inline /* implicit */ band_list(std::nullptr_t);
  inline isl::band_list &operator=(isl::band_list obj);
  inline ~band_list();
  inline __isl_give isl_band_list *copy() const &;
  inline __isl_give isl_band_list *copy() && = delete;
  inline __isl_keep isl_band_list *get() const;
  inline __isl_give isl_band_list *release();
  inline bool is_null() const;
  inline __isl_keep isl_band_list *keep() const;
  inline __isl_give isl_band_list *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

};

// declarations for isl::basic_map
inline isl::basic_map manage(__isl_take isl_basic_map *ptr);
inline isl::basic_map give(__isl_take isl_basic_map *ptr);


class basic_map {
  friend inline isl::basic_map manage(__isl_take isl_basic_map *ptr);

  isl_basic_map *ptr = nullptr;

  inline explicit basic_map(__isl_take isl_basic_map *ptr);

public:
  inline /* implicit */ basic_map();
  inline /* implicit */ basic_map(const isl::basic_map &obj);
  inline /* implicit */ basic_map(std::nullptr_t);
  inline explicit basic_map(isl::ctx ctx, const std::string &str);
  inline isl::basic_map &operator=(isl::basic_map obj);
  inline ~basic_map();
  inline __isl_give isl_basic_map *copy() const &;
  inline __isl_give isl_basic_map *copy() && = delete;
  inline __isl_keep isl_basic_map *get() const;
  inline __isl_give isl_basic_map *release();
  inline bool is_null() const;
  inline __isl_keep isl_basic_map *keep() const;
  inline __isl_give isl_basic_map *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::basic_map add_constraint(isl::constraint constraint) const;
  inline isl::basic_map add_dims(isl::dim type, unsigned int n) const;
  inline isl::basic_map affine_hull() const;
  inline isl::basic_map align_params(isl::space model) const;
  inline isl::basic_map apply_domain(isl::basic_map bmap2) const;
  inline isl::basic_map apply_range(isl::basic_map bmap2) const;
  inline isl::boolean can_curry() const;
  inline isl::boolean can_uncurry() const;
  inline isl::boolean can_zip() const;
  inline isl::basic_map curry() const;
  inline isl::basic_set deltas() const;
  inline isl::basic_map deltas_map() const;
  inline isl::basic_map detect_equalities() const;
  inline unsigned int dim(isl::dim type) const;
  inline isl::basic_set domain() const;
  inline isl::basic_map domain_map() const;
  inline isl::basic_map domain_product(isl::basic_map bmap2) const;
  inline isl::basic_map drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::basic_map drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::basic_map eliminate(isl::dim type, unsigned int first, unsigned int n) const;
  static inline isl::basic_map empty(isl::space dim);
  static inline isl::basic_map equal(isl::space dim, unsigned int n_equal);
  inline isl::basic_map equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::basic_map fix_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::basic_map fix_val(isl::dim type, unsigned int pos, isl::val v) const;
  inline isl::basic_map flat_product(isl::basic_map bmap2) const;
  inline isl::basic_map flat_range_product(isl::basic_map bmap2) const;
  inline isl::basic_map flatten() const;
  inline isl::basic_map flatten_domain() const;
  inline isl::basic_map flatten_range() const;
  inline isl::stat foreach_constraint(const std::function<isl::stat(isl::constraint)> &fn) const;
  static inline isl::basic_map from_aff(isl::aff aff);
  static inline isl::basic_map from_aff_list(isl::space domain_dim, isl::aff_list list);
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
  inline isl::boolean has_dim_id(isl::dim type, unsigned int pos) const;
  static inline isl::basic_map identity(isl::space dim);
  inline isl::boolean image_is_bounded() const;
  inline isl::basic_map insert_dims(isl::dim type, unsigned int pos, unsigned int n) const;
  inline isl::basic_map intersect(isl::basic_map bmap2) const;
  inline isl::basic_map intersect_domain(isl::basic_set bset) const;
  inline isl::basic_map intersect_range(isl::basic_set bset) const;
  inline isl::boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::boolean is_disjoint(const isl::basic_map &bmap2) const;
  inline isl::boolean is_empty() const;
  inline isl::boolean is_equal(const isl::basic_map &bmap2) const;
  inline isl::boolean is_rational() const;
  inline isl::boolean is_single_valued() const;
  inline isl::boolean is_strict_subset(const isl::basic_map &bmap2) const;
  inline isl::boolean is_subset(const isl::basic_map &bmap2) const;
  inline isl::boolean is_universe() const;
  static inline isl::basic_map less_at(isl::space dim, unsigned int pos);
  inline isl::map lexmax() const;
  inline isl::map lexmin() const;
  inline isl::pw_multi_aff lexmin_pw_multi_aff() const;
  inline isl::basic_map lower_bound_si(isl::dim type, unsigned int pos, int value) const;
  static inline isl::basic_map more_at(isl::space dim, unsigned int pos);
  inline isl::basic_map move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  static inline isl::basic_map nat_universe(isl::space dim);
  inline isl::basic_map neg() const;
  inline isl::basic_map order_ge(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline isl::basic_map order_gt(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline isl::val plain_get_val_if_fixed(isl::dim type, unsigned int pos) const;
  inline isl::boolean plain_is_empty() const;
  inline isl::boolean plain_is_universe() const;
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
  static inline isl::basic_map universe(isl::space dim);
  inline isl::basic_map upper_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::basic_set wrap() const;
  inline isl::basic_map zip() const;
};

// declarations for isl::basic_map_list
inline isl::basic_map_list manage(__isl_take isl_basic_map_list *ptr);
inline isl::basic_map_list give(__isl_take isl_basic_map_list *ptr);


class basic_map_list {
  friend inline isl::basic_map_list manage(__isl_take isl_basic_map_list *ptr);

  isl_basic_map_list *ptr = nullptr;

  inline explicit basic_map_list(__isl_take isl_basic_map_list *ptr);

public:
  inline /* implicit */ basic_map_list();
  inline /* implicit */ basic_map_list(const isl::basic_map_list &obj);
  inline /* implicit */ basic_map_list(std::nullptr_t);
  inline isl::basic_map_list &operator=(isl::basic_map_list obj);
  inline ~basic_map_list();
  inline __isl_give isl_basic_map_list *copy() const &;
  inline __isl_give isl_basic_map_list *copy() && = delete;
  inline __isl_keep isl_basic_map_list *get() const;
  inline __isl_give isl_basic_map_list *release();
  inline bool is_null() const;
  inline __isl_keep isl_basic_map_list *keep() const;
  inline __isl_give isl_basic_map_list *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

};

// declarations for isl::basic_set
inline isl::basic_set manage(__isl_take isl_basic_set *ptr);
inline isl::basic_set give(__isl_take isl_basic_set *ptr);


class basic_set {
  friend inline isl::basic_set manage(__isl_take isl_basic_set *ptr);

  isl_basic_set *ptr = nullptr;

  inline explicit basic_set(__isl_take isl_basic_set *ptr);

public:
  inline /* implicit */ basic_set();
  inline /* implicit */ basic_set(const isl::basic_set &obj);
  inline /* implicit */ basic_set(std::nullptr_t);
  inline explicit basic_set(isl::ctx ctx, const std::string &str);
  inline /* implicit */ basic_set(isl::point pnt);
  inline isl::basic_set &operator=(isl::basic_set obj);
  inline ~basic_set();
  inline __isl_give isl_basic_set *copy() const &;
  inline __isl_give isl_basic_set *copy() && = delete;
  inline __isl_keep isl_basic_set *get() const;
  inline __isl_give isl_basic_set *release();
  inline bool is_null() const;
  inline __isl_keep isl_basic_set *keep() const;
  inline __isl_give isl_basic_set *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::basic_set affine_hull() const;
  inline isl::basic_set align_params(isl::space model) const;
  inline isl::basic_set apply(isl::basic_map bmap) const;
  static inline isl::basic_set box_from_points(isl::point pnt1, isl::point pnt2);
  inline isl::basic_set coefficients() const;
  inline isl::basic_set detect_equalities() const;
  inline unsigned int dim(isl::dim type) const;
  inline isl::basic_set drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::basic_set drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::basic_set eliminate(isl::dim type, unsigned int first, unsigned int n) const;
  static inline isl::basic_set empty(isl::space dim);
  inline isl::basic_set fix_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::basic_set fix_val(isl::dim type, unsigned int pos, isl::val v) const;
  inline isl::basic_set flat_product(isl::basic_set bset2) const;
  inline isl::basic_set flatten() const;
  inline isl::stat foreach_bound_pair(isl::dim type, unsigned int pos, const std::function<isl::stat(isl::constraint, isl::constraint, isl::basic_set)> &fn) const;
  inline isl::stat foreach_constraint(const std::function<isl::stat(isl::constraint)> &fn) const;
  static inline isl::basic_set from_constraint(isl::constraint constraint);
  inline isl::basic_set from_params() const;
  inline isl::constraint_list get_constraint_list() const;
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline isl::aff get_div(int pos) const;
  inline isl::local_space get_local_space() const;
  inline isl::space get_space() const;
  inline std::string get_tuple_name() const;
  inline isl::basic_set gist(isl::basic_set context) const;
  inline isl::basic_set insert_dims(isl::dim type, unsigned int pos, unsigned int n) const;
  inline isl::basic_set intersect(isl::basic_set bset2) const;
  inline isl::basic_set intersect_params(isl::basic_set bset2) const;
  inline isl::boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::boolean is_bounded() const;
  inline isl::boolean is_disjoint(const isl::basic_set &bset2) const;
  inline isl::boolean is_empty() const;
  inline isl::boolean is_equal(const isl::basic_set &bset2) const;
  inline int is_rational() const;
  inline isl::boolean is_subset(const isl::basic_set &bset2) const;
  inline isl::boolean is_universe() const;
  inline isl::boolean is_wrapping() const;
  inline isl::set lexmax() const;
  inline isl::set lexmin() const;
  inline isl::basic_set lower_bound_val(isl::dim type, unsigned int pos, isl::val value) const;
  inline isl::val max_val(const isl::aff &obj) const;
  inline isl::basic_set move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  static inline isl::basic_set nat_universe(isl::space dim);
  inline isl::basic_set neg() const;
  inline isl::basic_set params() const;
  inline isl::boolean plain_is_empty() const;
  inline isl::boolean plain_is_equal(const isl::basic_set &bset2) const;
  inline isl::boolean plain_is_universe() const;
  static inline isl::basic_set positive_orthant(isl::space space);
  inline isl::basic_set preimage_multi_aff(isl::multi_aff ma) const;
  inline isl::basic_set project_out(isl::dim type, unsigned int first, unsigned int n) const;
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
  static inline isl::basic_set universe(isl::space dim);
  inline isl::basic_map unwrap() const;
  inline isl::basic_set upper_bound_val(isl::dim type, unsigned int pos, isl::val value) const;
};

// declarations for isl::basic_set_list
inline isl::basic_set_list manage(__isl_take isl_basic_set_list *ptr);
inline isl::basic_set_list give(__isl_take isl_basic_set_list *ptr);


class basic_set_list {
  friend inline isl::basic_set_list manage(__isl_take isl_basic_set_list *ptr);

  isl_basic_set_list *ptr = nullptr;

  inline explicit basic_set_list(__isl_take isl_basic_set_list *ptr);

public:
  inline /* implicit */ basic_set_list();
  inline /* implicit */ basic_set_list(const isl::basic_set_list &obj);
  inline /* implicit */ basic_set_list(std::nullptr_t);
  inline isl::basic_set_list &operator=(isl::basic_set_list obj);
  inline ~basic_set_list();
  inline __isl_give isl_basic_set_list *copy() const &;
  inline __isl_give isl_basic_set_list *copy() && = delete;
  inline __isl_keep isl_basic_set_list *get() const;
  inline __isl_give isl_basic_set_list *release();
  inline bool is_null() const;
  inline __isl_keep isl_basic_set_list *keep() const;
  inline __isl_give isl_basic_set_list *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

};

// declarations for isl::constraint
inline isl::constraint manage(__isl_take isl_constraint *ptr);
inline isl::constraint give(__isl_take isl_constraint *ptr);


class constraint {
  friend inline isl::constraint manage(__isl_take isl_constraint *ptr);

  isl_constraint *ptr = nullptr;

  inline explicit constraint(__isl_take isl_constraint *ptr);

public:
  inline /* implicit */ constraint();
  inline /* implicit */ constraint(const isl::constraint &obj);
  inline /* implicit */ constraint(std::nullptr_t);
  inline isl::constraint &operator=(isl::constraint obj);
  inline ~constraint();
  inline __isl_give isl_constraint *copy() const &;
  inline __isl_give isl_constraint *copy() && = delete;
  inline __isl_keep isl_constraint *get() const;
  inline __isl_give isl_constraint *release();
  inline bool is_null() const;
  inline __isl_keep isl_constraint *keep() const;
  inline __isl_give isl_constraint *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
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
  inline isl::boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline int is_div_constraint() const;
  inline isl::boolean is_lower_bound(isl::dim type, unsigned int pos) const;
  inline isl::boolean is_upper_bound(isl::dim type, unsigned int pos) const;
  inline int plain_cmp(const isl::constraint &c2) const;
  inline isl::constraint set_coefficient_si(isl::dim type, int pos, int v) const;
  inline isl::constraint set_coefficient_val(isl::dim type, int pos, isl::val v) const;
  inline isl::constraint set_constant_si(int v) const;
  inline isl::constraint set_constant_val(isl::val v) const;
};

// declarations for isl::constraint_list
inline isl::constraint_list manage(__isl_take isl_constraint_list *ptr);
inline isl::constraint_list give(__isl_take isl_constraint_list *ptr);


class constraint_list {
  friend inline isl::constraint_list manage(__isl_take isl_constraint_list *ptr);

  isl_constraint_list *ptr = nullptr;

  inline explicit constraint_list(__isl_take isl_constraint_list *ptr);

public:
  inline /* implicit */ constraint_list();
  inline /* implicit */ constraint_list(const isl::constraint_list &obj);
  inline /* implicit */ constraint_list(std::nullptr_t);
  inline isl::constraint_list &operator=(isl::constraint_list obj);
  inline ~constraint_list();
  inline __isl_give isl_constraint_list *copy() const &;
  inline __isl_give isl_constraint_list *copy() && = delete;
  inline __isl_keep isl_constraint_list *get() const;
  inline __isl_give isl_constraint_list *release();
  inline bool is_null() const;
  inline __isl_keep isl_constraint_list *keep() const;
  inline __isl_give isl_constraint_list *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

};

// declarations for isl::id
inline isl::id manage(__isl_take isl_id *ptr);
inline isl::id give(__isl_take isl_id *ptr);


class id {
  friend inline isl::id manage(__isl_take isl_id *ptr);

  isl_id *ptr = nullptr;

  inline explicit id(__isl_take isl_id *ptr);

public:
  inline /* implicit */ id();
  inline /* implicit */ id(const isl::id &obj);
  inline /* implicit */ id(std::nullptr_t);
  inline isl::id &operator=(isl::id obj);
  inline ~id();
  inline __isl_give isl_id *copy() const &;
  inline __isl_give isl_id *copy() && = delete;
  inline __isl_keep isl_id *get() const;
  inline __isl_give isl_id *release();
  inline bool is_null() const;
  inline __isl_keep isl_id *keep() const;
  inline __isl_give isl_id *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  static inline isl::id alloc(isl::ctx ctx, const std::string &name, void * user);
  inline uint32_t get_hash() const;
  inline std::string get_name() const;
  inline void * get_user() const;
};

// declarations for isl::id_list
inline isl::id_list manage(__isl_take isl_id_list *ptr);
inline isl::id_list give(__isl_take isl_id_list *ptr);


class id_list {
  friend inline isl::id_list manage(__isl_take isl_id_list *ptr);

  isl_id_list *ptr = nullptr;

  inline explicit id_list(__isl_take isl_id_list *ptr);

public:
  inline /* implicit */ id_list();
  inline /* implicit */ id_list(const isl::id_list &obj);
  inline /* implicit */ id_list(std::nullptr_t);
  inline isl::id_list &operator=(isl::id_list obj);
  inline ~id_list();
  inline __isl_give isl_id_list *copy() const &;
  inline __isl_give isl_id_list *copy() && = delete;
  inline __isl_keep isl_id_list *get() const;
  inline __isl_give isl_id_list *release();
  inline bool is_null() const;
  inline __isl_keep isl_id_list *keep() const;
  inline __isl_give isl_id_list *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

};

// declarations for isl::id_to_ast_expr
inline isl::id_to_ast_expr manage(__isl_take isl_id_to_ast_expr *ptr);
inline isl::id_to_ast_expr give(__isl_take isl_id_to_ast_expr *ptr);


class id_to_ast_expr {
  friend inline isl::id_to_ast_expr manage(__isl_take isl_id_to_ast_expr *ptr);

  isl_id_to_ast_expr *ptr = nullptr;

  inline explicit id_to_ast_expr(__isl_take isl_id_to_ast_expr *ptr);

public:
  inline /* implicit */ id_to_ast_expr();
  inline /* implicit */ id_to_ast_expr(const isl::id_to_ast_expr &obj);
  inline /* implicit */ id_to_ast_expr(std::nullptr_t);
  inline isl::id_to_ast_expr &operator=(isl::id_to_ast_expr obj);
  inline ~id_to_ast_expr();
  inline __isl_give isl_id_to_ast_expr *copy() const &;
  inline __isl_give isl_id_to_ast_expr *copy() && = delete;
  inline __isl_keep isl_id_to_ast_expr *get() const;
  inline __isl_give isl_id_to_ast_expr *release();
  inline bool is_null() const;
  inline __isl_keep isl_id_to_ast_expr *keep() const;
  inline __isl_give isl_id_to_ast_expr *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

  static inline isl::id_to_ast_expr alloc(isl::ctx ctx, int min_size);
  inline isl::id_to_ast_expr drop(isl::id key) const;
  inline isl::stat foreach(const std::function<isl::stat(isl::id, isl::ast_expr)> &fn) const;
  inline isl::ast_expr get(isl::id key) const;
  inline isl::boolean has(const isl::id &key) const;
  inline isl::id_to_ast_expr set(isl::id key, isl::ast_expr val) const;
};

// declarations for isl::local_space
inline isl::local_space manage(__isl_take isl_local_space *ptr);
inline isl::local_space give(__isl_take isl_local_space *ptr);


class local_space {
  friend inline isl::local_space manage(__isl_take isl_local_space *ptr);

  isl_local_space *ptr = nullptr;

  inline explicit local_space(__isl_take isl_local_space *ptr);

public:
  inline /* implicit */ local_space();
  inline /* implicit */ local_space(const isl::local_space &obj);
  inline /* implicit */ local_space(std::nullptr_t);
  inline explicit local_space(isl::space dim);
  inline isl::local_space &operator=(isl::local_space obj);
  inline ~local_space();
  inline __isl_give isl_local_space *copy() const &;
  inline __isl_give isl_local_space *copy() && = delete;
  inline __isl_keep isl_local_space *get() const;
  inline __isl_give isl_local_space *release();
  inline bool is_null() const;
  inline __isl_keep isl_local_space *keep() const;
  inline __isl_give isl_local_space *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

  inline isl::local_space add_dims(isl::dim type, unsigned int n) const;
  inline int dim(isl::dim type) const;
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
  inline isl::boolean has_dim_id(isl::dim type, unsigned int pos) const;
  inline isl::boolean has_dim_name(isl::dim type, unsigned int pos) const;
  inline isl::local_space insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::local_space intersect(isl::local_space ls2) const;
  inline isl::boolean is_equal(const isl::local_space &ls2) const;
  inline isl::boolean is_params() const;
  inline isl::boolean is_set() const;
  inline isl::local_space range() const;
  inline isl::local_space set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::local_space set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::local_space wrap() const;
};

// declarations for isl::map
inline isl::map manage(__isl_take isl_map *ptr);
inline isl::map give(__isl_take isl_map *ptr);


class map {
  friend inline isl::map manage(__isl_take isl_map *ptr);

  isl_map *ptr = nullptr;

  inline explicit map(__isl_take isl_map *ptr);

public:
  inline /* implicit */ map();
  inline /* implicit */ map(const isl::map &obj);
  inline /* implicit */ map(std::nullptr_t);
  inline explicit map(isl::ctx ctx, const std::string &str);
  inline /* implicit */ map(isl::basic_map bmap);
  inline isl::map &operator=(isl::map obj);
  inline ~map();
  inline __isl_give isl_map *copy() const &;
  inline __isl_give isl_map *copy() && = delete;
  inline __isl_keep isl_map *get() const;
  inline __isl_give isl_map *release();
  inline bool is_null() const;
  inline __isl_keep isl_map *keep() const;
  inline __isl_give isl_map *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::map add_constraint(isl::constraint constraint) const;
  inline isl::map add_dims(isl::dim type, unsigned int n) const;
  inline isl::basic_map affine_hull() const;
  inline isl::map align_params(isl::space model) const;
  inline isl::map apply_domain(isl::map map2) const;
  inline isl::map apply_range(isl::map map2) const;
  inline isl::boolean can_curry() const;
  inline isl::boolean can_range_curry() const;
  inline isl::boolean can_uncurry() const;
  inline isl::boolean can_zip() const;
  inline isl::map coalesce() const;
  inline isl::map complement() const;
  inline isl::basic_map convex_hull() const;
  inline isl::map curry() const;
  inline isl::set deltas() const;
  inline isl::map deltas_map() const;
  inline isl::map detect_equalities() const;
  inline unsigned int dim(isl::dim type) const;
  inline isl::pw_aff dim_max(int pos) const;
  inline isl::pw_aff dim_min(int pos) const;
  inline isl::set domain() const;
  inline isl::map domain_factor_domain() const;
  inline isl::map domain_factor_range() const;
  inline isl::boolean domain_is_wrapping() const;
  inline isl::map domain_map() const;
  inline isl::map domain_product(isl::map map2) const;
  inline isl::map drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::map drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::map eliminate(isl::dim type, unsigned int first, unsigned int n) const;
  static inline isl::map empty(isl::space dim);
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
  inline isl::stat foreach_basic_map(const std::function<isl::stat(isl::basic_map)> &fn) const;
  static inline isl::map from_aff(isl::aff aff);
  static inline isl::map from_domain(isl::set set);
  static inline isl::map from_domain_and_range(isl::set domain, isl::set range);
  static inline isl::map from_multi_aff(isl::multi_aff maff);
  static inline isl::map from_multi_pw_aff(isl::multi_pw_aff mpa);
  static inline isl::map from_pw_aff(isl::pw_aff pwaff);
  static inline isl::map from_pw_multi_aff(isl::pw_multi_aff pma);
  static inline isl::map from_range(isl::set set);
  static inline isl::map from_union_map(isl::union_map umap);
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline uint32_t get_hash() const;
  inline isl::space get_space() const;
  inline isl::id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline isl::map gist(isl::map context) const;
  inline isl::map gist_basic_map(isl::basic_map context) const;
  inline isl::map gist_domain(isl::set context) const;
  inline isl::map gist_params(isl::set context) const;
  inline isl::map gist_range(isl::set context) const;
  inline isl::boolean has_dim_id(isl::dim type, unsigned int pos) const;
  inline isl::boolean has_dim_name(isl::dim type, unsigned int pos) const;
  inline isl::boolean has_equal_space(const isl::map &map2) const;
  inline isl::boolean has_tuple_id(isl::dim type) const;
  inline isl::boolean has_tuple_name(isl::dim type) const;
  static inline isl::map identity(isl::space dim);
  inline isl::map insert_dims(isl::dim type, unsigned int pos, unsigned int n) const;
  inline isl::map intersect(isl::map map2) const;
  inline isl::map intersect_domain(isl::set set) const;
  inline isl::map intersect_domain_factor_range(isl::map factor) const;
  inline isl::map intersect_params(isl::set params) const;
  inline isl::map intersect_range(isl::set set) const;
  inline isl::map intersect_range_factor_range(isl::map factor) const;
  inline isl::boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::boolean is_bijective() const;
  inline isl::boolean is_disjoint(const isl::map &map2) const;
  inline isl::boolean is_empty() const;
  inline isl::boolean is_equal(const isl::map &map2) const;
  inline isl::boolean is_identity() const;
  inline isl::boolean is_injective() const;
  inline isl::boolean is_product() const;
  inline isl::boolean is_single_valued() const;
  inline isl::boolean is_strict_subset(const isl::map &map2) const;
  inline isl::boolean is_subset(const isl::map &map2) const;
  inline int is_translation() const;
  static inline isl::map lex_ge(isl::space set_dim);
  static inline isl::map lex_ge_first(isl::space dim, unsigned int n);
  inline isl::map lex_ge_map(isl::map map2) const;
  static inline isl::map lex_gt(isl::space set_dim);
  static inline isl::map lex_gt_first(isl::space dim, unsigned int n);
  inline isl::map lex_gt_map(isl::map map2) const;
  static inline isl::map lex_le(isl::space set_dim);
  static inline isl::map lex_le_first(isl::space dim, unsigned int n);
  inline isl::map lex_le_map(isl::map map2) const;
  static inline isl::map lex_lt(isl::space set_dim);
  static inline isl::map lex_lt_first(isl::space dim, unsigned int n);
  inline isl::map lex_lt_map(isl::map map2) const;
  inline isl::map lexmax() const;
  inline isl::pw_multi_aff lexmax_pw_multi_aff() const;
  inline isl::map lexmin() const;
  inline isl::pw_multi_aff lexmin_pw_multi_aff() const;
  inline isl::map lower_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::map move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  static inline isl::map nat_universe(isl::space dim);
  inline isl::map neg() const;
  inline isl::map oppose(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline isl::map order_ge(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline isl::map order_gt(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline isl::map order_le(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline isl::map order_lt(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline isl::set params() const;
  inline isl::val plain_get_val_if_fixed(isl::dim type, unsigned int pos) const;
  inline isl::boolean plain_is_empty() const;
  inline isl::boolean plain_is_equal(const isl::map &map2) const;
  inline isl::boolean plain_is_injective() const;
  inline isl::boolean plain_is_single_valued() const;
  inline isl::boolean plain_is_universe() const;
  inline isl::basic_map plain_unshifted_simple_hull() const;
  inline isl::basic_map polyhedral_hull() const;
  inline isl::map preimage_domain_multi_aff(isl::multi_aff ma) const;
  inline isl::map preimage_domain_multi_pw_aff(isl::multi_pw_aff mpa) const;
  inline isl::map preimage_domain_pw_multi_aff(isl::pw_multi_aff pma) const;
  inline isl::map preimage_range_multi_aff(isl::multi_aff ma) const;
  inline isl::map preimage_range_pw_multi_aff(isl::pw_multi_aff pma) const;
  inline isl::map product(isl::map map2) const;
  inline isl::map project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set range() const;
  inline isl::map range_curry() const;
  inline isl::map range_factor_domain() const;
  inline isl::map range_factor_range() const;
  inline isl::boolean range_is_wrapping() const;
  inline isl::map range_map() const;
  inline isl::map range_product(isl::map map2) const;
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
  static inline isl::map universe(isl::space dim);
  inline isl::basic_map unshifted_simple_hull() const;
  inline isl::basic_map unshifted_simple_hull_from_map_list(isl::map_list list) const;
  inline isl::map upper_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::set wrap() const;
  inline isl::map zip() const;
};

// declarations for isl::map_list
inline isl::map_list manage(__isl_take isl_map_list *ptr);
inline isl::map_list give(__isl_take isl_map_list *ptr);


class map_list {
  friend inline isl::map_list manage(__isl_take isl_map_list *ptr);

  isl_map_list *ptr = nullptr;

  inline explicit map_list(__isl_take isl_map_list *ptr);

public:
  inline /* implicit */ map_list();
  inline /* implicit */ map_list(const isl::map_list &obj);
  inline /* implicit */ map_list(std::nullptr_t);
  inline isl::map_list &operator=(isl::map_list obj);
  inline ~map_list();
  inline __isl_give isl_map_list *copy() const &;
  inline __isl_give isl_map_list *copy() && = delete;
  inline __isl_keep isl_map_list *get() const;
  inline __isl_give isl_map_list *release();
  inline bool is_null() const;
  inline __isl_keep isl_map_list *keep() const;
  inline __isl_give isl_map_list *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

};

// declarations for isl::multi_aff
inline isl::multi_aff manage(__isl_take isl_multi_aff *ptr);
inline isl::multi_aff give(__isl_take isl_multi_aff *ptr);


class multi_aff {
  friend inline isl::multi_aff manage(__isl_take isl_multi_aff *ptr);

  isl_multi_aff *ptr = nullptr;

  inline explicit multi_aff(__isl_take isl_multi_aff *ptr);

public:
  inline /* implicit */ multi_aff();
  inline /* implicit */ multi_aff(const isl::multi_aff &obj);
  inline /* implicit */ multi_aff(std::nullptr_t);
  inline explicit multi_aff(isl::ctx ctx, const std::string &str);
  inline /* implicit */ multi_aff(isl::aff aff);
  inline isl::multi_aff &operator=(isl::multi_aff obj);
  inline ~multi_aff();
  inline __isl_give isl_multi_aff *copy() const &;
  inline __isl_give isl_multi_aff *copy() && = delete;
  inline __isl_keep isl_multi_aff *get() const;
  inline __isl_give isl_multi_aff *release();
  inline bool is_null() const;
  inline __isl_keep isl_multi_aff *keep() const;
  inline __isl_give isl_multi_aff *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::multi_aff add(isl::multi_aff multi2) const;
  inline isl::multi_aff add_dims(isl::dim type, unsigned int n) const;
  inline isl::multi_aff align_params(isl::space model) const;
  inline unsigned int dim(isl::dim type) const;
  static inline isl::multi_aff domain_map(isl::space space);
  inline isl::multi_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::multi_aff factor_range() const;
  inline int find_dim_by_id(isl::dim type, const isl::id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::multi_aff flat_range_product(isl::multi_aff multi2) const;
  inline isl::multi_aff flatten_domain() const;
  inline isl::multi_aff flatten_range() const;
  inline isl::multi_aff floor() const;
  static inline isl::multi_aff from_aff_list(isl::space space, isl::aff_list list);
  inline isl::multi_aff from_range() const;
  inline isl::aff get_aff(int pos) const;
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline isl::space get_domain_space() const;
  inline isl::space get_space() const;
  inline isl::id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline isl::multi_aff gist(isl::set context) const;
  inline isl::multi_aff gist_params(isl::set context) const;
  inline isl::boolean has_tuple_id(isl::dim type) const;
  static inline isl::multi_aff identity(isl::space space);
  inline isl::multi_aff insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::boolean involves_nan() const;
  inline isl::set lex_ge_set(isl::multi_aff ma2) const;
  inline isl::set lex_gt_set(isl::multi_aff ma2) const;
  inline isl::set lex_le_set(isl::multi_aff ma2) const;
  inline isl::set lex_lt_set(isl::multi_aff ma2) const;
  inline isl::multi_aff mod_multi_val(isl::multi_val mv) const;
  inline isl::multi_aff move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  static inline isl::multi_aff multi_val_on_space(isl::space space, isl::multi_val mv);
  inline isl::multi_aff neg() const;
  inline int plain_cmp(const isl::multi_aff &multi2) const;
  inline isl::boolean plain_is_equal(const isl::multi_aff &multi2) const;
  inline isl::multi_aff product(isl::multi_aff multi2) const;
  static inline isl::multi_aff project_out_map(isl::space space, isl::dim type, unsigned int first, unsigned int n);
  inline isl::multi_aff pullback(isl::multi_aff ma2) const;
  inline isl::multi_aff range_factor_domain() const;
  inline isl::multi_aff range_factor_range() const;
  inline isl::boolean range_is_wrapping() const;
  static inline isl::multi_aff range_map(isl::space space);
  inline isl::multi_aff range_product(isl::multi_aff multi2) const;
  inline isl::multi_aff range_splice(unsigned int pos, isl::multi_aff multi2) const;
  inline isl::multi_aff reset_tuple_id(isl::dim type) const;
  inline isl::multi_aff reset_user() const;
  inline isl::multi_aff scale_down_multi_val(isl::multi_val mv) const;
  inline isl::multi_aff scale_down_val(isl::val v) const;
  inline isl::multi_aff scale_multi_val(isl::multi_val mv) const;
  inline isl::multi_aff scale_val(isl::val v) const;
  inline isl::multi_aff set_aff(int pos, isl::aff el) const;
  inline isl::multi_aff set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::multi_aff set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::multi_aff set_tuple_name(isl::dim type, const std::string &s) const;
  inline isl::multi_aff splice(unsigned int in_pos, unsigned int out_pos, isl::multi_aff multi2) const;
  inline isl::multi_aff sub(isl::multi_aff multi2) const;
  static inline isl::multi_aff zero(isl::space space);
};

// declarations for isl::multi_pw_aff
inline isl::multi_pw_aff manage(__isl_take isl_multi_pw_aff *ptr);
inline isl::multi_pw_aff give(__isl_take isl_multi_pw_aff *ptr);


class multi_pw_aff {
  friend inline isl::multi_pw_aff manage(__isl_take isl_multi_pw_aff *ptr);

  isl_multi_pw_aff *ptr = nullptr;

  inline explicit multi_pw_aff(__isl_take isl_multi_pw_aff *ptr);

public:
  inline /* implicit */ multi_pw_aff();
  inline /* implicit */ multi_pw_aff(const isl::multi_pw_aff &obj);
  inline /* implicit */ multi_pw_aff(std::nullptr_t);
  inline /* implicit */ multi_pw_aff(isl::multi_aff ma);
  inline /* implicit */ multi_pw_aff(isl::pw_aff pa);
  inline /* implicit */ multi_pw_aff(isl::pw_multi_aff pma);
  inline explicit multi_pw_aff(isl::ctx ctx, const std::string &str);
  inline isl::multi_pw_aff &operator=(isl::multi_pw_aff obj);
  inline ~multi_pw_aff();
  inline __isl_give isl_multi_pw_aff *copy() const &;
  inline __isl_give isl_multi_pw_aff *copy() && = delete;
  inline __isl_keep isl_multi_pw_aff *get() const;
  inline __isl_give isl_multi_pw_aff *release();
  inline bool is_null() const;
  inline __isl_keep isl_multi_pw_aff *keep() const;
  inline __isl_give isl_multi_pw_aff *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::multi_pw_aff add(isl::multi_pw_aff multi2) const;
  inline isl::multi_pw_aff add_dims(isl::dim type, unsigned int n) const;
  inline isl::multi_pw_aff align_params(isl::space model) const;
  inline isl::multi_pw_aff coalesce() const;
  inline unsigned int dim(isl::dim type) const;
  inline isl::set domain() const;
  inline isl::multi_pw_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::map eq_map(isl::multi_pw_aff mpa2) const;
  inline isl::multi_pw_aff factor_range() const;
  inline int find_dim_by_id(isl::dim type, const isl::id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::multi_pw_aff flat_range_product(isl::multi_pw_aff multi2) const;
  inline isl::multi_pw_aff flatten_range() const;
  static inline isl::multi_pw_aff from_pw_aff_list(isl::space space, isl::pw_aff_list list);
  inline isl::multi_pw_aff from_range() const;
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline isl::space get_domain_space() const;
  inline uint32_t get_hash() const;
  inline isl::pw_aff get_pw_aff(int pos) const;
  inline isl::space get_space() const;
  inline isl::id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline isl::multi_pw_aff gist(isl::set set) const;
  inline isl::multi_pw_aff gist_params(isl::set set) const;
  inline isl::boolean has_tuple_id(isl::dim type) const;
  static inline isl::multi_pw_aff identity(isl::space space);
  inline isl::multi_pw_aff insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::multi_pw_aff intersect_domain(isl::set domain) const;
  inline isl::multi_pw_aff intersect_params(isl::set set) const;
  inline isl::boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::boolean involves_nan() const;
  inline isl::boolean is_cst() const;
  inline isl::boolean is_equal(const isl::multi_pw_aff &mpa2) const;
  inline isl::map lex_gt_map(isl::multi_pw_aff mpa2) const;
  inline isl::map lex_lt_map(isl::multi_pw_aff mpa2) const;
  inline isl::multi_pw_aff mod_multi_val(isl::multi_val mv) const;
  inline isl::multi_pw_aff move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline isl::multi_pw_aff neg() const;
  inline isl::boolean plain_is_equal(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_pw_aff product(isl::multi_pw_aff multi2) const;
  inline isl::multi_pw_aff pullback(isl::multi_aff ma) const;
  inline isl::multi_pw_aff pullback(isl::pw_multi_aff pma) const;
  inline isl::multi_pw_aff pullback(isl::multi_pw_aff mpa2) const;
  inline isl::multi_pw_aff range_factor_domain() const;
  inline isl::multi_pw_aff range_factor_range() const;
  inline isl::boolean range_is_wrapping() const;
  inline isl::multi_pw_aff range_product(isl::multi_pw_aff multi2) const;
  inline isl::multi_pw_aff range_splice(unsigned int pos, isl::multi_pw_aff multi2) const;
  inline isl::multi_pw_aff reset_tuple_id(isl::dim type) const;
  inline isl::multi_pw_aff reset_user() const;
  inline isl::multi_pw_aff scale_down_multi_val(isl::multi_val mv) const;
  inline isl::multi_pw_aff scale_down_val(isl::val v) const;
  inline isl::multi_pw_aff scale_multi_val(isl::multi_val mv) const;
  inline isl::multi_pw_aff scale_val(isl::val v) const;
  inline isl::multi_pw_aff set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::multi_pw_aff set_pw_aff(int pos, isl::pw_aff el) const;
  inline isl::multi_pw_aff set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::multi_pw_aff set_tuple_name(isl::dim type, const std::string &s) const;
  inline isl::multi_pw_aff splice(unsigned int in_pos, unsigned int out_pos, isl::multi_pw_aff multi2) const;
  inline isl::multi_pw_aff sub(isl::multi_pw_aff multi2) const;
  static inline isl::multi_pw_aff zero(isl::space space);
};

// declarations for isl::multi_union_pw_aff
inline isl::multi_union_pw_aff manage(__isl_take isl_multi_union_pw_aff *ptr);
inline isl::multi_union_pw_aff give(__isl_take isl_multi_union_pw_aff *ptr);


class multi_union_pw_aff {
  friend inline isl::multi_union_pw_aff manage(__isl_take isl_multi_union_pw_aff *ptr);

  isl_multi_union_pw_aff *ptr = nullptr;

  inline explicit multi_union_pw_aff(__isl_take isl_multi_union_pw_aff *ptr);

public:
  inline /* implicit */ multi_union_pw_aff();
  inline /* implicit */ multi_union_pw_aff(const isl::multi_union_pw_aff &obj);
  inline /* implicit */ multi_union_pw_aff(std::nullptr_t);
  inline /* implicit */ multi_union_pw_aff(isl::union_pw_aff upa);
  inline /* implicit */ multi_union_pw_aff(isl::multi_pw_aff mpa);
  inline explicit multi_union_pw_aff(isl::union_pw_multi_aff upma);
  inline explicit multi_union_pw_aff(isl::ctx ctx, const std::string &str);
  inline isl::multi_union_pw_aff &operator=(isl::multi_union_pw_aff obj);
  inline ~multi_union_pw_aff();
  inline __isl_give isl_multi_union_pw_aff *copy() const &;
  inline __isl_give isl_multi_union_pw_aff *copy() && = delete;
  inline __isl_keep isl_multi_union_pw_aff *get() const;
  inline __isl_give isl_multi_union_pw_aff *release();
  inline bool is_null() const;
  inline __isl_keep isl_multi_union_pw_aff *keep() const;
  inline __isl_give isl_multi_union_pw_aff *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::multi_union_pw_aff add(isl::multi_union_pw_aff multi2) const;
  inline isl::multi_union_pw_aff align_params(isl::space model) const;
  inline isl::union_pw_aff apply_aff(isl::aff aff) const;
  inline isl::union_pw_aff apply_pw_aff(isl::pw_aff pa) const;
  inline isl::multi_union_pw_aff apply_pw_multi_aff(isl::pw_multi_aff pma) const;
  inline isl::multi_union_pw_aff coalesce() const;
  inline unsigned int dim(isl::dim type) const;
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
  static inline isl::multi_union_pw_aff from_union_pw_aff_list(isl::space space, isl::union_pw_aff_list list);
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline isl::space get_domain_space() const;
  inline isl::space get_space() const;
  inline isl::id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline isl::union_pw_aff get_union_pw_aff(int pos) const;
  inline isl::multi_union_pw_aff gist(isl::union_set context) const;
  inline isl::multi_union_pw_aff gist_params(isl::set context) const;
  inline isl::boolean has_tuple_id(isl::dim type) const;
  inline isl::multi_union_pw_aff intersect_domain(isl::union_set uset) const;
  inline isl::multi_union_pw_aff intersect_params(isl::set params) const;
  inline isl::multi_union_pw_aff intersect_range(isl::set set) const;
  inline isl::boolean involves_nan() const;
  inline isl::multi_union_pw_aff mod_multi_val(isl::multi_val mv) const;
  static inline isl::multi_union_pw_aff multi_aff_on_domain(isl::union_set domain, isl::multi_aff ma);
  static inline isl::multi_union_pw_aff multi_val_on_domain(isl::union_set domain, isl::multi_val mv);
  inline isl::multi_union_pw_aff neg() const;
  inline isl::boolean plain_is_equal(const isl::multi_union_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff pullback(isl::union_pw_multi_aff upma) const;
  inline isl::multi_union_pw_aff range_factor_domain() const;
  inline isl::multi_union_pw_aff range_factor_range() const;
  inline isl::boolean range_is_wrapping() const;
  inline isl::multi_union_pw_aff range_product(isl::multi_union_pw_aff multi2) const;
  inline isl::multi_union_pw_aff range_splice(unsigned int pos, isl::multi_union_pw_aff multi2) const;
  inline isl::multi_union_pw_aff reset_tuple_id(isl::dim type) const;
  inline isl::multi_union_pw_aff reset_user() const;
  inline isl::multi_union_pw_aff scale_down_multi_val(isl::multi_val mv) const;
  inline isl::multi_union_pw_aff scale_down_val(isl::val v) const;
  inline isl::multi_union_pw_aff scale_multi_val(isl::multi_val mv) const;
  inline isl::multi_union_pw_aff scale_val(isl::val v) const;
  inline isl::multi_union_pw_aff set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::multi_union_pw_aff set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::multi_union_pw_aff set_tuple_name(isl::dim type, const std::string &s) const;
  inline isl::multi_union_pw_aff set_union_pw_aff(int pos, isl::union_pw_aff el) const;
  inline isl::multi_union_pw_aff sub(isl::multi_union_pw_aff multi2) const;
  inline isl::multi_union_pw_aff union_add(isl::multi_union_pw_aff mupa2) const;
  static inline isl::multi_union_pw_aff zero(isl::space space);
  inline isl::union_set zero_union_set() const;
};

// declarations for isl::multi_val
inline isl::multi_val manage(__isl_take isl_multi_val *ptr);
inline isl::multi_val give(__isl_take isl_multi_val *ptr);


class multi_val {
  friend inline isl::multi_val manage(__isl_take isl_multi_val *ptr);

  isl_multi_val *ptr = nullptr;

  inline explicit multi_val(__isl_take isl_multi_val *ptr);

public:
  inline /* implicit */ multi_val();
  inline /* implicit */ multi_val(const isl::multi_val &obj);
  inline /* implicit */ multi_val(std::nullptr_t);
  inline explicit multi_val(isl::ctx ctx, const std::string &str);
  inline isl::multi_val &operator=(isl::multi_val obj);
  inline ~multi_val();
  inline __isl_give isl_multi_val *copy() const &;
  inline __isl_give isl_multi_val *copy() && = delete;
  inline __isl_keep isl_multi_val *get() const;
  inline __isl_give isl_multi_val *release();
  inline bool is_null() const;
  inline __isl_keep isl_multi_val *keep() const;
  inline __isl_give isl_multi_val *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::multi_val add(isl::multi_val multi2) const;
  inline isl::multi_val add_dims(isl::dim type, unsigned int n) const;
  inline isl::multi_val add_val(isl::val v) const;
  inline isl::multi_val align_params(isl::space model) const;
  inline unsigned int dim(isl::dim type) const;
  inline isl::multi_val drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::multi_val factor_range() const;
  inline int find_dim_by_id(isl::dim type, const isl::id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::multi_val flat_range_product(isl::multi_val multi2) const;
  inline isl::multi_val flatten_range() const;
  inline isl::multi_val from_range() const;
  static inline isl::multi_val from_val_list(isl::space space, isl::val_list list);
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline isl::space get_domain_space() const;
  inline isl::space get_space() const;
  inline isl::id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline isl::val get_val(int pos) const;
  inline isl::boolean has_tuple_id(isl::dim type) const;
  inline isl::multi_val insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::boolean involves_nan() const;
  inline isl::multi_val mod_multi_val(isl::multi_val mv) const;
  inline isl::multi_val mod_val(isl::val v) const;
  inline isl::multi_val neg() const;
  inline isl::boolean plain_is_equal(const isl::multi_val &multi2) const;
  inline isl::multi_val product(isl::multi_val multi2) const;
  inline isl::multi_val range_factor_domain() const;
  inline isl::multi_val range_factor_range() const;
  inline isl::boolean range_is_wrapping() const;
  inline isl::multi_val range_product(isl::multi_val multi2) const;
  inline isl::multi_val range_splice(unsigned int pos, isl::multi_val multi2) const;
  inline isl::multi_val reset_tuple_id(isl::dim type) const;
  inline isl::multi_val reset_user() const;
  inline isl::multi_val scale_down_multi_val(isl::multi_val mv) const;
  inline isl::multi_val scale_down_val(isl::val v) const;
  inline isl::multi_val scale_multi_val(isl::multi_val mv) const;
  inline isl::multi_val scale_val(isl::val v) const;
  inline isl::multi_val set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::multi_val set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::multi_val set_tuple_name(isl::dim type, const std::string &s) const;
  inline isl::multi_val set_val(int pos, isl::val el) const;
  inline isl::multi_val splice(unsigned int in_pos, unsigned int out_pos, isl::multi_val multi2) const;
  inline isl::multi_val sub(isl::multi_val multi2) const;
  static inline isl::multi_val zero(isl::space space);
};

// declarations for isl::point
inline isl::point manage(__isl_take isl_point *ptr);
inline isl::point give(__isl_take isl_point *ptr);


class point {
  friend inline isl::point manage(__isl_take isl_point *ptr);

  isl_point *ptr = nullptr;

  inline explicit point(__isl_take isl_point *ptr);

public:
  inline /* implicit */ point();
  inline /* implicit */ point(const isl::point &obj);
  inline /* implicit */ point(std::nullptr_t);
  inline explicit point(isl::space dim);
  inline isl::point &operator=(isl::point obj);
  inline ~point();
  inline __isl_give isl_point *copy() const &;
  inline __isl_give isl_point *copy() && = delete;
  inline __isl_keep isl_point *get() const;
  inline __isl_give isl_point *release();
  inline bool is_null() const;
  inline __isl_keep isl_point *keep() const;
  inline __isl_give isl_point *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::point add_ui(isl::dim type, int pos, unsigned int val) const;
  inline isl::val get_coordinate_val(isl::dim type, int pos) const;
  inline isl::space get_space() const;
  inline isl::point set_coordinate_val(isl::dim type, int pos, isl::val v) const;
  inline isl::point sub_ui(isl::dim type, int pos, unsigned int val) const;
};

// declarations for isl::pw_aff
inline isl::pw_aff manage(__isl_take isl_pw_aff *ptr);
inline isl::pw_aff give(__isl_take isl_pw_aff *ptr);


class pw_aff {
  friend inline isl::pw_aff manage(__isl_take isl_pw_aff *ptr);

  isl_pw_aff *ptr = nullptr;

  inline explicit pw_aff(__isl_take isl_pw_aff *ptr);

public:
  inline /* implicit */ pw_aff();
  inline /* implicit */ pw_aff(const isl::pw_aff &obj);
  inline /* implicit */ pw_aff(std::nullptr_t);
  inline /* implicit */ pw_aff(isl::aff aff);
  inline explicit pw_aff(isl::local_space ls);
  inline explicit pw_aff(isl::set domain, isl::val v);
  inline explicit pw_aff(isl::ctx ctx, const std::string &str);
  inline isl::pw_aff &operator=(isl::pw_aff obj);
  inline ~pw_aff();
  inline __isl_give isl_pw_aff *copy() const &;
  inline __isl_give isl_pw_aff *copy() && = delete;
  inline __isl_keep isl_pw_aff *get() const;
  inline __isl_give isl_pw_aff *release();
  inline bool is_null() const;
  inline __isl_keep isl_pw_aff *keep() const;
  inline __isl_give isl_pw_aff *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::pw_aff add(isl::pw_aff pwaff2) const;
  inline isl::pw_aff add_dims(isl::dim type, unsigned int n) const;
  inline isl::pw_aff align_params(isl::space model) const;
  static inline isl::pw_aff alloc(isl::set set, isl::aff aff);
  inline isl::pw_aff ceil() const;
  inline isl::pw_aff coalesce() const;
  inline isl::pw_aff cond(isl::pw_aff pwaff_true, isl::pw_aff pwaff_false) const;
  inline unsigned int dim(isl::dim type) const;
  inline isl::pw_aff div(isl::pw_aff pa2) const;
  inline isl::set domain() const;
  inline isl::pw_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  static inline isl::pw_aff empty(isl::space dim);
  inline isl::map eq_map(isl::pw_aff pa2) const;
  inline isl::set eq_set(isl::pw_aff pwaff2) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::pw_aff floor() const;
  inline isl::stat foreach_piece(const std::function<isl::stat(isl::set, isl::aff)> &fn) const;
  inline isl::pw_aff from_range() const;
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
  inline isl::boolean has_dim_id(isl::dim type, unsigned int pos) const;
  inline isl::boolean has_tuple_id(isl::dim type) const;
  inline isl::pw_aff insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::pw_aff intersect_domain(isl::set set) const;
  inline isl::pw_aff intersect_params(isl::set set) const;
  inline isl::boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::boolean involves_nan() const;
  inline isl::boolean is_cst() const;
  inline isl::boolean is_empty() const;
  inline isl::boolean is_equal(const isl::pw_aff &pa2) const;
  inline isl::set le_set(isl::pw_aff pwaff2) const;
  inline isl::map lt_map(isl::pw_aff pa2) const;
  inline isl::set lt_set(isl::pw_aff pwaff2) const;
  inline isl::pw_aff max(isl::pw_aff pwaff2) const;
  inline isl::pw_aff min(isl::pw_aff pwaff2) const;
  inline isl::pw_aff mod(isl::val mod) const;
  inline isl::pw_aff move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline isl::pw_aff mul(isl::pw_aff pwaff2) const;
  static inline isl::pw_aff nan_on_domain(isl::local_space ls);
  inline isl::set ne_set(isl::pw_aff pwaff2) const;
  inline isl::pw_aff neg() const;
  inline isl::set non_zero_set() const;
  inline isl::set nonneg_set() const;
  inline isl::set params() const;
  inline int plain_cmp(const isl::pw_aff &pa2) const;
  inline isl::boolean plain_is_equal(const isl::pw_aff &pwaff2) const;
  inline isl::set pos_set() const;
  inline isl::pw_aff project_domain_on_params() const;
  inline isl::pw_aff pullback(isl::multi_aff ma) const;
  inline isl::pw_aff pullback(isl::pw_multi_aff pma) const;
  inline isl::pw_aff pullback(isl::multi_pw_aff mpa) const;
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
inline isl::pw_aff_list manage(__isl_take isl_pw_aff_list *ptr);
inline isl::pw_aff_list give(__isl_take isl_pw_aff_list *ptr);


class pw_aff_list {
  friend inline isl::pw_aff_list manage(__isl_take isl_pw_aff_list *ptr);

  isl_pw_aff_list *ptr = nullptr;

  inline explicit pw_aff_list(__isl_take isl_pw_aff_list *ptr);

public:
  inline /* implicit */ pw_aff_list();
  inline /* implicit */ pw_aff_list(const isl::pw_aff_list &obj);
  inline /* implicit */ pw_aff_list(std::nullptr_t);
  inline isl::pw_aff_list &operator=(isl::pw_aff_list obj);
  inline ~pw_aff_list();
  inline __isl_give isl_pw_aff_list *copy() const &;
  inline __isl_give isl_pw_aff_list *copy() && = delete;
  inline __isl_keep isl_pw_aff_list *get() const;
  inline __isl_give isl_pw_aff_list *release();
  inline bool is_null() const;
  inline __isl_keep isl_pw_aff_list *keep() const;
  inline __isl_give isl_pw_aff_list *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

};

// declarations for isl::pw_multi_aff
inline isl::pw_multi_aff manage(__isl_take isl_pw_multi_aff *ptr);
inline isl::pw_multi_aff give(__isl_take isl_pw_multi_aff *ptr);


class pw_multi_aff {
  friend inline isl::pw_multi_aff manage(__isl_take isl_pw_multi_aff *ptr);

  isl_pw_multi_aff *ptr = nullptr;

  inline explicit pw_multi_aff(__isl_take isl_pw_multi_aff *ptr);

public:
  inline /* implicit */ pw_multi_aff();
  inline /* implicit */ pw_multi_aff(const isl::pw_multi_aff &obj);
  inline /* implicit */ pw_multi_aff(std::nullptr_t);
  inline /* implicit */ pw_multi_aff(isl::multi_aff ma);
  inline /* implicit */ pw_multi_aff(isl::pw_aff pa);
  inline explicit pw_multi_aff(isl::ctx ctx, const std::string &str);
  inline isl::pw_multi_aff &operator=(isl::pw_multi_aff obj);
  inline ~pw_multi_aff();
  inline __isl_give isl_pw_multi_aff *copy() const &;
  inline __isl_give isl_pw_multi_aff *copy() && = delete;
  inline __isl_keep isl_pw_multi_aff *get() const;
  inline __isl_give isl_pw_multi_aff *release();
  inline bool is_null() const;
  inline __isl_keep isl_pw_multi_aff *keep() const;
  inline __isl_give isl_pw_multi_aff *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::pw_multi_aff add(isl::pw_multi_aff pma2) const;
  inline isl::pw_multi_aff align_params(isl::space model) const;
  static inline isl::pw_multi_aff alloc(isl::set set, isl::multi_aff maff);
  inline isl::pw_multi_aff coalesce() const;
  inline unsigned int dim(isl::dim type) const;
  inline isl::set domain() const;
  inline isl::pw_multi_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  static inline isl::pw_multi_aff empty(isl::space space);
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::pw_multi_aff fix_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::pw_multi_aff flat_range_product(isl::pw_multi_aff pma2) const;
  inline isl::stat foreach_piece(const std::function<isl::stat(isl::set, isl::multi_aff)> &fn) const;
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
  inline isl::boolean has_tuple_id(isl::dim type) const;
  inline isl::boolean has_tuple_name(isl::dim type) const;
  static inline isl::pw_multi_aff identity(isl::space space);
  inline isl::pw_multi_aff intersect_domain(isl::set set) const;
  inline isl::pw_multi_aff intersect_params(isl::set set) const;
  inline isl::boolean involves_nan() const;
  inline isl::boolean is_equal(const isl::pw_multi_aff &pma2) const;
  static inline isl::pw_multi_aff multi_val_on_domain(isl::set domain, isl::multi_val mv);
  inline isl::pw_multi_aff neg() const;
  inline isl::boolean plain_is_equal(const isl::pw_multi_aff &pma2) const;
  inline isl::pw_multi_aff product(isl::pw_multi_aff pma2) const;
  inline isl::pw_multi_aff project_domain_on_params() const;
  static inline isl::pw_multi_aff project_out_map(isl::space space, isl::dim type, unsigned int first, unsigned int n);
  inline isl::pw_multi_aff pullback(isl::multi_aff ma) const;
  inline isl::pw_multi_aff pullback(isl::pw_multi_aff pma2) const;
  static inline isl::pw_multi_aff range_map(isl::space space);
  inline isl::pw_multi_aff range_product(isl::pw_multi_aff pma2) const;
  inline isl::pw_multi_aff reset_tuple_id(isl::dim type) const;
  inline isl::pw_multi_aff reset_user() const;
  inline isl::pw_multi_aff scale_down_val(isl::val v) const;
  inline isl::pw_multi_aff scale_multi_val(isl::multi_val mv) const;
  inline isl::pw_multi_aff scale_val(isl::val v) const;
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

// declarations for isl::pw_qpolynomial
inline isl::pw_qpolynomial manage(__isl_take isl_pw_qpolynomial *ptr);
inline isl::pw_qpolynomial give(__isl_take isl_pw_qpolynomial *ptr);


class pw_qpolynomial {
  friend inline isl::pw_qpolynomial manage(__isl_take isl_pw_qpolynomial *ptr);

  isl_pw_qpolynomial *ptr = nullptr;

  inline explicit pw_qpolynomial(__isl_take isl_pw_qpolynomial *ptr);

public:
  inline /* implicit */ pw_qpolynomial();
  inline /* implicit */ pw_qpolynomial(const isl::pw_qpolynomial &obj);
  inline /* implicit */ pw_qpolynomial(std::nullptr_t);
  inline explicit pw_qpolynomial(isl::ctx ctx, const std::string &str);
  inline isl::pw_qpolynomial &operator=(isl::pw_qpolynomial obj);
  inline ~pw_qpolynomial();
  inline __isl_give isl_pw_qpolynomial *copy() const &;
  inline __isl_give isl_pw_qpolynomial *copy() && = delete;
  inline __isl_keep isl_pw_qpolynomial *get() const;
  inline __isl_give isl_pw_qpolynomial *release();
  inline bool is_null() const;
  inline __isl_keep isl_pw_qpolynomial *keep() const;
  inline __isl_give isl_pw_qpolynomial *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::pw_qpolynomial add(isl::pw_qpolynomial pwqp2) const;
  inline isl::pw_qpolynomial add_dims(isl::dim type, unsigned int n) const;
  static inline isl::pw_qpolynomial alloc(isl::set set, isl::qpolynomial qp);
  inline isl::pw_qpolynomial coalesce() const;
  inline unsigned int dim(isl::dim type) const;
  inline isl::set domain() const;
  inline isl::pw_qpolynomial drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::val eval(isl::point pnt) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::pw_qpolynomial fix_val(isl::dim type, unsigned int n, isl::val v) const;
  inline isl::stat foreach_piece(const std::function<isl::stat(isl::set, isl::qpolynomial)> &fn) const;
  static inline isl::pw_qpolynomial from_pw_aff(isl::pw_aff pwaff);
  static inline isl::pw_qpolynomial from_qpolynomial(isl::qpolynomial qp);
  inline isl::pw_qpolynomial from_range() const;
  inline isl::space get_domain_space() const;
  inline isl::space get_space() const;
  inline isl::pw_qpolynomial gist(isl::set context) const;
  inline isl::pw_qpolynomial gist_params(isl::set context) const;
  inline isl::boolean has_equal_space(const isl::pw_qpolynomial &pwqp2) const;
  inline isl::pw_qpolynomial insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::pw_qpolynomial intersect_domain(isl::set set) const;
  inline isl::pw_qpolynomial intersect_params(isl::set set) const;
  inline isl::boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::boolean involves_nan() const;
  inline isl::boolean is_zero() const;
  inline isl::val max() const;
  inline isl::val min() const;
  inline isl::pw_qpolynomial move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline isl::pw_qpolynomial mul(isl::pw_qpolynomial pwqp2) const;
  inline isl::pw_qpolynomial neg() const;
  inline isl::boolean plain_is_equal(const isl::pw_qpolynomial &pwqp2) const;
  inline isl::pw_qpolynomial pow(unsigned int exponent) const;
  inline isl::pw_qpolynomial project_domain_on_params() const;
  inline isl::pw_qpolynomial reset_domain_space(isl::space dim) const;
  inline isl::pw_qpolynomial reset_user() const;
  inline isl::pw_qpolynomial scale_down_val(isl::val v) const;
  inline isl::pw_qpolynomial scale_val(isl::val v) const;
  inline isl::pw_qpolynomial split_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::pw_qpolynomial split_periods(int max_periods) const;
  inline isl::pw_qpolynomial sub(isl::pw_qpolynomial pwqp2) const;
  inline isl::pw_qpolynomial subtract_domain(isl::set set) const;
  inline isl::pw_qpolynomial to_polynomial(int sign) const;
  static inline isl::pw_qpolynomial zero(isl::space dim);
};

// declarations for isl::qpolynomial
inline isl::qpolynomial manage(__isl_take isl_qpolynomial *ptr);
inline isl::qpolynomial give(__isl_take isl_qpolynomial *ptr);


class qpolynomial {
  friend inline isl::qpolynomial manage(__isl_take isl_qpolynomial *ptr);

  isl_qpolynomial *ptr = nullptr;

  inline explicit qpolynomial(__isl_take isl_qpolynomial *ptr);

public:
  inline /* implicit */ qpolynomial();
  inline /* implicit */ qpolynomial(const isl::qpolynomial &obj);
  inline /* implicit */ qpolynomial(std::nullptr_t);
  inline isl::qpolynomial &operator=(isl::qpolynomial obj);
  inline ~qpolynomial();
  inline __isl_give isl_qpolynomial *copy() const &;
  inline __isl_give isl_qpolynomial *copy() && = delete;
  inline __isl_keep isl_qpolynomial *get() const;
  inline __isl_give isl_qpolynomial *release();
  inline bool is_null() const;
  inline __isl_keep isl_qpolynomial *keep() const;
  inline __isl_give isl_qpolynomial *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

  inline isl::qpolynomial add(isl::qpolynomial qp2) const;
  inline isl::qpolynomial add_dims(isl::dim type, unsigned int n) const;
  inline isl::qpolynomial align_params(isl::space model) const;
  inline isl::stat as_polynomial_on_domain(const isl::basic_set &bset, const std::function<isl::stat(isl::basic_set, isl::qpolynomial)> &fn) const;
  inline unsigned int dim(isl::dim type) const;
  inline isl::qpolynomial drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::val eval(isl::point pnt) const;
  inline isl::stat foreach_term(const std::function<isl::stat(isl::term)> &fn) const;
  static inline isl::qpolynomial from_aff(isl::aff aff);
  static inline isl::qpolynomial from_constraint(isl::constraint c, isl::dim type, unsigned int pos);
  static inline isl::qpolynomial from_term(isl::term term);
  inline isl::val get_constant_val() const;
  inline isl::space get_domain_space() const;
  inline isl::space get_space() const;
  inline isl::qpolynomial gist(isl::set context) const;
  inline isl::qpolynomial gist_params(isl::set context) const;
  inline isl::qpolynomial homogenize() const;
  static inline isl::qpolynomial infty_on_domain(isl::space dim);
  inline isl::qpolynomial insert_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::boolean is_infty() const;
  inline isl::boolean is_nan() const;
  inline isl::boolean is_neginfty() const;
  inline isl::boolean is_zero() const;
  inline isl::qpolynomial move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  inline isl::qpolynomial mul(isl::qpolynomial qp2) const;
  static inline isl::qpolynomial nan_on_domain(isl::space dim);
  inline isl::qpolynomial neg() const;
  static inline isl::qpolynomial neginfty_on_domain(isl::space dim);
  static inline isl::qpolynomial one_on_domain(isl::space dim);
  inline isl::boolean plain_is_equal(const isl::qpolynomial &qp2) const;
  inline isl::qpolynomial pow(unsigned int power) const;
  inline isl::qpolynomial project_domain_on_params() const;
  inline isl::qpolynomial scale_down_val(isl::val v) const;
  inline isl::qpolynomial scale_val(isl::val v) const;
  inline int sgn() const;
  inline isl::qpolynomial sub(isl::qpolynomial qp2) const;
  static inline isl::qpolynomial val_on_domain(isl::space space, isl::val val);
  static inline isl::qpolynomial var_on_domain(isl::space dim, isl::dim type, unsigned int pos);
  static inline isl::qpolynomial zero_on_domain(isl::space dim);
};

// declarations for isl::schedule
inline isl::schedule manage(__isl_take isl_schedule *ptr);
inline isl::schedule give(__isl_take isl_schedule *ptr);


class schedule {
  friend inline isl::schedule manage(__isl_take isl_schedule *ptr);

  isl_schedule *ptr = nullptr;

  inline explicit schedule(__isl_take isl_schedule *ptr);

public:
  inline /* implicit */ schedule();
  inline /* implicit */ schedule(const isl::schedule &obj);
  inline /* implicit */ schedule(std::nullptr_t);
  inline explicit schedule(isl::ctx ctx, const std::string &str);
  inline isl::schedule &operator=(isl::schedule obj);
  inline ~schedule();
  inline __isl_give isl_schedule *copy() const &;
  inline __isl_give isl_schedule *copy() && = delete;
  inline __isl_keep isl_schedule *get() const;
  inline __isl_give isl_schedule *release();
  inline bool is_null() const;
  inline __isl_keep isl_schedule *keep() const;
  inline __isl_give isl_schedule *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
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
  inline isl::boolean plain_is_equal(const isl::schedule &schedule2) const;
  inline isl::schedule pullback(isl::union_pw_multi_aff upma) const;
  inline isl::schedule reset_user() const;
  inline isl::schedule sequence(isl::schedule schedule2) const;
  inline isl::schedule set(isl::schedule schedule2) const;
};

// declarations for isl::schedule_constraints
inline isl::schedule_constraints manage(__isl_take isl_schedule_constraints *ptr);
inline isl::schedule_constraints give(__isl_take isl_schedule_constraints *ptr);


class schedule_constraints {
  friend inline isl::schedule_constraints manage(__isl_take isl_schedule_constraints *ptr);

  isl_schedule_constraints *ptr = nullptr;

  inline explicit schedule_constraints(__isl_take isl_schedule_constraints *ptr);

public:
  inline /* implicit */ schedule_constraints();
  inline /* implicit */ schedule_constraints(const isl::schedule_constraints &obj);
  inline /* implicit */ schedule_constraints(std::nullptr_t);
  inline explicit schedule_constraints(isl::ctx ctx, const std::string &str);
  inline isl::schedule_constraints &operator=(isl::schedule_constraints obj);
  inline ~schedule_constraints();
  inline __isl_give isl_schedule_constraints *copy() const &;
  inline __isl_give isl_schedule_constraints *copy() && = delete;
  inline __isl_keep isl_schedule_constraints *get() const;
  inline __isl_give isl_schedule_constraints *release();
  inline bool is_null() const;
  inline __isl_keep isl_schedule_constraints *keep() const;
  inline __isl_give isl_schedule_constraints *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
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
inline isl::schedule_node manage(__isl_take isl_schedule_node *ptr);
inline isl::schedule_node give(__isl_take isl_schedule_node *ptr);


class schedule_node {
  friend inline isl::schedule_node manage(__isl_take isl_schedule_node *ptr);

  isl_schedule_node *ptr = nullptr;

  inline explicit schedule_node(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node();
  inline /* implicit */ schedule_node(const isl::schedule_node &obj);
  inline /* implicit */ schedule_node(std::nullptr_t);
  inline isl::schedule_node &operator=(isl::schedule_node obj);
  inline ~schedule_node();
  inline __isl_give isl_schedule_node *copy() const &;
  inline __isl_give isl_schedule_node *copy() && = delete;
  inline __isl_keep isl_schedule_node *get() const;
  inline __isl_give isl_schedule_node *release();
  inline bool is_null() const;
  inline __isl_keep isl_schedule_node *keep() const;
  inline __isl_give isl_schedule_node *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::schedule_node align_params(isl::space space) const;
  inline isl::schedule_node ancestor(int generation) const;
  inline isl::boolean band_member_get_coincident(int pos) const;
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
  inline isl::stat foreach_ancestor_top_down(const std::function<isl::stat(isl::schedule_node)> &fn) const;
  static inline isl::schedule_node from_domain(isl::union_set domain);
  static inline isl::schedule_node from_extension(isl::union_map extension);
  inline int get_ancestor_child_position(const isl::schedule_node &ancestor) const;
  inline isl::schedule_node get_child(int pos) const;
  inline int get_child_position() const;
  inline isl::union_set get_domain() const;
  inline isl::multi_union_pw_aff get_prefix_schedule_multi_union_pw_aff() const;
  inline isl::union_map get_prefix_schedule_relation() const;
  inline isl::union_map get_prefix_schedule_union_map() const;
  inline isl::union_pw_multi_aff get_prefix_schedule_union_pw_multi_aff() const;
  inline isl::schedule get_schedule() const;
  inline int get_schedule_depth() const;
  inline isl::schedule_node get_shared_ancestor(const isl::schedule_node &node2) const;
  inline isl::union_pw_multi_aff get_subtree_contraction() const;
  inline isl::union_map get_subtree_expansion() const;
  inline isl::union_map get_subtree_schedule_union_map() const;
  inline int get_tree_depth() const;
  inline isl::union_set get_universe_domain() const;
  inline isl::schedule_node graft_after(isl::schedule_node graft) const;
  inline isl::schedule_node graft_before(isl::schedule_node graft) const;
  inline isl::schedule_node group(isl::id group_id) const;
  inline isl::set guard_get_guard() const;
  inline isl::boolean has_children() const;
  inline isl::boolean has_next_sibling() const;
  inline isl::boolean has_parent() const;
  inline isl::boolean has_previous_sibling() const;
  inline isl::schedule_node insert_context(isl::set context) const;
  inline isl::schedule_node insert_filter(isl::union_set filter) const;
  inline isl::schedule_node insert_guard(isl::set context) const;
  inline isl::schedule_node insert_mark(isl::id mark) const;
  inline isl::schedule_node insert_partial_schedule(isl::multi_union_pw_aff schedule) const;
  inline isl::schedule_node insert_sequence(isl::union_set_list filters) const;
  inline isl::schedule_node insert_set(isl::union_set_list filters) const;
  inline isl::boolean is_equal(const isl::schedule_node &node2) const;
  inline isl::boolean is_subtree_anchored() const;
  inline isl::id mark_get_id() const;
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
inline isl::set manage(__isl_take isl_set *ptr);
inline isl::set give(__isl_take isl_set *ptr);


class set {
  friend inline isl::set manage(__isl_take isl_set *ptr);

  isl_set *ptr = nullptr;

  inline explicit set(__isl_take isl_set *ptr);

public:
  inline /* implicit */ set();
  inline /* implicit */ set(const isl::set &obj);
  inline /* implicit */ set(std::nullptr_t);
  inline explicit set(isl::union_set uset);
  inline explicit set(isl::ctx ctx, const std::string &str);
  inline /* implicit */ set(isl::basic_set bset);
  inline /* implicit */ set(isl::point pnt);
  inline isl::set &operator=(isl::set obj);
  inline ~set();
  inline __isl_give isl_set *copy() const &;
  inline __isl_give isl_set *copy() && = delete;
  inline __isl_keep isl_set *get() const;
  inline __isl_give isl_set *release();
  inline bool is_null() const;
  inline __isl_keep isl_set *keep() const;
  inline __isl_give isl_set *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::set add_constraint(isl::constraint constraint) const;
  inline isl::set add_dims(isl::dim type, unsigned int n) const;
  inline isl::basic_set affine_hull() const;
  inline isl::set align_params(isl::space model) const;
  inline isl::set apply(isl::map map) const;
  inline isl::basic_set bounded_simple_hull() const;
  static inline isl::set box_from_points(isl::point pnt1, isl::point pnt2);
  inline isl::set coalesce() const;
  inline isl::basic_set coefficients() const;
  inline isl::set complement() const;
  inline isl::basic_set convex_hull() const;
  inline isl::val count_val() const;
  inline isl::set detect_equalities() const;
  inline unsigned int dim(isl::dim type) const;
  inline isl::boolean dim_has_any_lower_bound(isl::dim type, unsigned int pos) const;
  inline isl::boolean dim_has_any_upper_bound(isl::dim type, unsigned int pos) const;
  inline isl::boolean dim_has_lower_bound(isl::dim type, unsigned int pos) const;
  inline isl::boolean dim_has_upper_bound(isl::dim type, unsigned int pos) const;
  inline isl::boolean dim_is_bounded(isl::dim type, unsigned int pos) const;
  inline isl::pw_aff dim_max(int pos) const;
  inline isl::pw_aff dim_min(int pos) const;
  inline isl::set drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set eliminate(isl::dim type, unsigned int first, unsigned int n) const;
  static inline isl::set empty(isl::space dim);
  inline isl::set equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const;
  inline int find_dim_by_id(isl::dim type, const isl::id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::set fix_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::set fix_val(isl::dim type, unsigned int pos, isl::val v) const;
  inline isl::set flat_product(isl::set set2) const;
  inline isl::set flatten() const;
  inline isl::map flatten_map() const;
  inline int follows_at(const isl::set &set2, int pos) const;
  inline isl::stat foreach_basic_set(const std::function<isl::stat(isl::basic_set)> &fn) const;
  inline isl::stat foreach_point(const std::function<isl::stat(isl::point)> &fn) const;
  static inline isl::set from_multi_pw_aff(isl::multi_pw_aff mpa);
  inline isl::set from_params() const;
  static inline isl::set from_pw_aff(isl::pw_aff pwaff);
  static inline isl::set from_pw_multi_aff(isl::pw_multi_aff pma);
  inline isl::basic_set_list get_basic_set_list() const;
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline isl::space get_space() const;
  inline isl::id get_tuple_id() const;
  inline std::string get_tuple_name() const;
  inline isl::set gist(isl::set context) const;
  inline isl::set gist_basic_set(isl::basic_set context) const;
  inline isl::set gist_params(isl::set context) const;
  inline isl::boolean has_dim_id(isl::dim type, unsigned int pos) const;
  inline isl::boolean has_dim_name(isl::dim type, unsigned int pos) const;
  inline isl::boolean has_equal_space(const isl::set &set2) const;
  inline isl::boolean has_tuple_id() const;
  inline isl::boolean has_tuple_name() const;
  inline isl::map identity() const;
  inline isl::pw_aff indicator_function() const;
  inline isl::set insert_dims(isl::dim type, unsigned int pos, unsigned int n) const;
  inline isl::set intersect(isl::set set2) const;
  inline isl::set intersect_params(isl::set params) const;
  inline isl::boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::boolean is_bounded() const;
  inline isl::boolean is_box() const;
  inline isl::boolean is_disjoint(const isl::set &set2) const;
  inline isl::boolean is_empty() const;
  inline isl::boolean is_equal(const isl::set &set2) const;
  inline isl::boolean is_params() const;
  inline isl::boolean is_singleton() const;
  inline isl::boolean is_strict_subset(const isl::set &set2) const;
  inline isl::boolean is_subset(const isl::set &set2) const;
  inline isl::boolean is_wrapping() const;
  inline isl::map lex_ge_set(isl::set set2) const;
  inline isl::map lex_gt_set(isl::set set2) const;
  inline isl::map lex_le_set(isl::set set2) const;
  inline isl::map lex_lt_set(isl::set set2) const;
  inline isl::set lexmax() const;
  inline isl::pw_multi_aff lexmax_pw_multi_aff() const;
  inline isl::set lexmin() const;
  inline isl::pw_multi_aff lexmin_pw_multi_aff() const;
  inline isl::set lower_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::set lower_bound_val(isl::dim type, unsigned int pos, isl::val value) const;
  inline isl::val max_val(const isl::aff &obj) const;
  inline isl::val min_val(const isl::aff &obj) const;
  inline isl::set move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const;
  static inline isl::set nat_universe(isl::space dim);
  inline isl::set neg() const;
  inline isl::set params() const;
  inline int plain_cmp(const isl::set &set2) const;
  inline isl::val plain_get_val_if_fixed(isl::dim type, unsigned int pos) const;
  inline isl::boolean plain_is_disjoint(const isl::set &set2) const;
  inline isl::boolean plain_is_empty() const;
  inline isl::boolean plain_is_equal(const isl::set &set2) const;
  inline isl::boolean plain_is_universe() const;
  inline isl::basic_set plain_unshifted_simple_hull() const;
  inline isl::basic_set polyhedral_hull() const;
  inline isl::set preimage_multi_aff(isl::multi_aff ma) const;
  inline isl::set preimage_multi_pw_aff(isl::multi_pw_aff mpa) const;
  inline isl::set preimage_pw_multi_aff(isl::pw_multi_aff pma) const;
  inline isl::set product(isl::set set2) const;
  inline isl::map project_onto_map(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set remove_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set remove_divs() const;
  inline isl::set remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::set remove_redundancies() const;
  inline isl::set remove_unknown_divs() const;
  inline isl::set reset_space(isl::space dim) const;
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
  inline isl::set unite(isl::set set2) const;
  static inline isl::set universe(isl::space dim);
  inline isl::basic_set unshifted_simple_hull() const;
  inline isl::basic_set unshifted_simple_hull_from_set_list(isl::set_list list) const;
  inline isl::map unwrap() const;
  inline isl::set upper_bound_si(isl::dim type, unsigned int pos, int value) const;
  inline isl::set upper_bound_val(isl::dim type, unsigned int pos, isl::val value) const;
  inline isl::map wrapped_domain_map() const;
};

// declarations for isl::set_list
inline isl::set_list manage(__isl_take isl_set_list *ptr);
inline isl::set_list give(__isl_take isl_set_list *ptr);


class set_list {
  friend inline isl::set_list manage(__isl_take isl_set_list *ptr);

  isl_set_list *ptr = nullptr;

  inline explicit set_list(__isl_take isl_set_list *ptr);

public:
  inline /* implicit */ set_list();
  inline /* implicit */ set_list(const isl::set_list &obj);
  inline /* implicit */ set_list(std::nullptr_t);
  inline isl::set_list &operator=(isl::set_list obj);
  inline ~set_list();
  inline __isl_give isl_set_list *copy() const &;
  inline __isl_give isl_set_list *copy() && = delete;
  inline __isl_keep isl_set_list *get() const;
  inline __isl_give isl_set_list *release();
  inline bool is_null() const;
  inline __isl_keep isl_set_list *keep() const;
  inline __isl_give isl_set_list *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

};

// declarations for isl::space
inline isl::space manage(__isl_take isl_space *ptr);
inline isl::space give(__isl_take isl_space *ptr);


class space {
  friend inline isl::space manage(__isl_take isl_space *ptr);

  isl_space *ptr = nullptr;

  inline explicit space(__isl_take isl_space *ptr);

public:
  inline /* implicit */ space();
  inline /* implicit */ space(const isl::space &obj);
  inline /* implicit */ space(std::nullptr_t);
  inline explicit space(isl::ctx ctx, unsigned int nparam, unsigned int n_in, unsigned int n_out);
  inline explicit space(isl::ctx ctx, unsigned int nparam, unsigned int dim);
  inline isl::space &operator=(isl::space obj);
  inline ~space();
  inline __isl_give isl_space *copy() const &;
  inline __isl_give isl_space *copy() && = delete;
  inline __isl_keep isl_space *get() const;
  inline __isl_give isl_space *release();
  inline bool is_null() const;
  inline __isl_keep isl_space *keep() const;
  inline __isl_give isl_space *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::space add_dims(isl::dim type, unsigned int n) const;
  inline isl::space align_params(isl::space dim2) const;
  inline isl::boolean can_curry() const;
  inline isl::boolean can_range_curry() const;
  inline isl::boolean can_uncurry() const;
  inline isl::boolean can_zip() const;
  inline isl::space curry() const;
  inline unsigned int dim(isl::dim type) const;
  inline isl::space domain() const;
  inline isl::space domain_factor_domain() const;
  inline isl::space domain_factor_range() const;
  inline isl::boolean domain_is_wrapping() const;
  inline isl::space domain_map() const;
  inline isl::space domain_product(isl::space right) const;
  inline isl::space drop_dims(isl::dim type, unsigned int first, unsigned int num) const;
  inline isl::space factor_domain() const;
  inline isl::space factor_range() const;
  inline int find_dim_by_id(isl::dim type, const isl::id &id) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::space from_domain() const;
  inline isl::space from_range() const;
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline std::string get_dim_name(isl::dim type, unsigned int pos) const;
  inline isl::id get_tuple_id(isl::dim type) const;
  inline std::string get_tuple_name(isl::dim type) const;
  inline isl::boolean has_dim_id(isl::dim type, unsigned int pos) const;
  inline isl::boolean has_dim_name(isl::dim type, unsigned int pos) const;
  inline isl::boolean has_equal_params(const isl::space &space2) const;
  inline isl::boolean has_equal_tuples(const isl::space &space2) const;
  inline isl::boolean has_tuple_id(isl::dim type) const;
  inline isl::boolean has_tuple_name(isl::dim type) const;
  inline isl::space insert_dims(isl::dim type, unsigned int pos, unsigned int n) const;
  inline isl::boolean is_domain(const isl::space &space2) const;
  inline isl::boolean is_equal(const isl::space &space2) const;
  inline isl::boolean is_map() const;
  inline isl::boolean is_params() const;
  inline isl::boolean is_product() const;
  inline isl::boolean is_range(const isl::space &space2) const;
  inline isl::boolean is_set() const;
  inline isl::boolean is_wrapping() const;
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
  inline isl::boolean range_is_wrapping() const;
  inline isl::space range_map() const;
  inline isl::space range_product(isl::space right) const;
  inline isl::space reset_tuple_id(isl::dim type) const;
  inline isl::space reset_user() const;
  inline isl::space reverse() const;
  inline isl::space set_dim_id(isl::dim type, unsigned int pos, isl::id id) const;
  inline isl::space set_from_params() const;
  inline isl::space set_tuple_id(isl::dim type, isl::id id) const;
  inline isl::space set_tuple_name(isl::dim type, const std::string &s) const;
  inline isl::boolean tuple_is_equal(isl::dim type1, const isl::space &space2, isl::dim type2) const;
  inline isl::space uncurry() const;
  inline isl::space unwrap() const;
  inline isl::space wrap() const;
  inline isl::space zip() const;
};

// declarations for isl::term
inline isl::term manage(__isl_take isl_term *ptr);
inline isl::term give(__isl_take isl_term *ptr);


class term {
  friend inline isl::term manage(__isl_take isl_term *ptr);

  isl_term *ptr = nullptr;

  inline explicit term(__isl_take isl_term *ptr);

public:
  inline /* implicit */ term();
  inline /* implicit */ term(const isl::term &obj);
  inline /* implicit */ term(std::nullptr_t);
  inline isl::term &operator=(isl::term obj);
  inline ~term();
  inline __isl_give isl_term *copy() const &;
  inline __isl_give isl_term *copy() && = delete;
  inline __isl_keep isl_term *get() const;
  inline __isl_give isl_term *release();
  inline bool is_null() const;
  inline __isl_keep isl_term *keep() const;
  inline __isl_give isl_term *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;

  inline unsigned int dim(isl::dim type) const;
  inline isl::val get_coefficient_val() const;
  inline isl::aff get_div(unsigned int pos) const;
  inline int get_exp(isl::dim type, unsigned int pos) const;
};

// declarations for isl::union_access_info
inline isl::union_access_info manage(__isl_take isl_union_access_info *ptr);
inline isl::union_access_info give(__isl_take isl_union_access_info *ptr);


class union_access_info {
  friend inline isl::union_access_info manage(__isl_take isl_union_access_info *ptr);

  isl_union_access_info *ptr = nullptr;

  inline explicit union_access_info(__isl_take isl_union_access_info *ptr);

public:
  inline /* implicit */ union_access_info();
  inline /* implicit */ union_access_info(const isl::union_access_info &obj);
  inline /* implicit */ union_access_info(std::nullptr_t);
  inline explicit union_access_info(isl::union_map sink);
  inline isl::union_access_info &operator=(isl::union_access_info obj);
  inline ~union_access_info();
  inline __isl_give isl_union_access_info *copy() const &;
  inline __isl_give isl_union_access_info *copy() && = delete;
  inline __isl_keep isl_union_access_info *get() const;
  inline __isl_give isl_union_access_info *release();
  inline bool is_null() const;
  inline __isl_keep isl_union_access_info *keep() const;
  inline __isl_give isl_union_access_info *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;

  inline isl::union_flow compute_flow() const;
  inline isl::union_access_info set_kill(isl::union_map kill) const;
  inline isl::union_access_info set_may_source(isl::union_map may_source) const;
  inline isl::union_access_info set_must_source(isl::union_map must_source) const;
  inline isl::union_access_info set_schedule(isl::schedule schedule) const;
  inline isl::union_access_info set_schedule_map(isl::union_map schedule_map) const;
};

// declarations for isl::union_flow
inline isl::union_flow manage(__isl_take isl_union_flow *ptr);
inline isl::union_flow give(__isl_take isl_union_flow *ptr);


class union_flow {
  friend inline isl::union_flow manage(__isl_take isl_union_flow *ptr);

  isl_union_flow *ptr = nullptr;

  inline explicit union_flow(__isl_take isl_union_flow *ptr);

public:
  inline /* implicit */ union_flow();
  inline /* implicit */ union_flow(const isl::union_flow &obj);
  inline /* implicit */ union_flow(std::nullptr_t);
  inline isl::union_flow &operator=(isl::union_flow obj);
  inline ~union_flow();
  inline __isl_give isl_union_flow *copy() const &;
  inline __isl_give isl_union_flow *copy() && = delete;
  inline __isl_keep isl_union_flow *get() const;
  inline __isl_give isl_union_flow *release();
  inline bool is_null() const;
  inline __isl_keep isl_union_flow *keep() const;
  inline __isl_give isl_union_flow *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;

  inline isl::union_map get_full_may_dependence() const;
  inline isl::union_map get_full_must_dependence() const;
  inline isl::union_map get_may_dependence() const;
  inline isl::union_map get_may_no_source() const;
  inline isl::union_map get_must_dependence() const;
  inline isl::union_map get_must_no_source() const;
};

// declarations for isl::union_map
inline isl::union_map manage(__isl_take isl_union_map *ptr);
inline isl::union_map give(__isl_take isl_union_map *ptr);


class union_map {
  friend inline isl::union_map manage(__isl_take isl_union_map *ptr);

  isl_union_map *ptr = nullptr;

  inline explicit union_map(__isl_take isl_union_map *ptr);

public:
  inline /* implicit */ union_map();
  inline /* implicit */ union_map(const isl::union_map &obj);
  inline /* implicit */ union_map(std::nullptr_t);
  inline explicit union_map(isl::union_pw_aff upa);
  inline /* implicit */ union_map(isl::basic_map bmap);
  inline /* implicit */ union_map(isl::map map);
  inline explicit union_map(isl::ctx ctx, const std::string &str);
  inline isl::union_map &operator=(isl::union_map obj);
  inline ~union_map();
  inline __isl_give isl_union_map *copy() const &;
  inline __isl_give isl_union_map *copy() && = delete;
  inline __isl_keep isl_union_map *get() const;
  inline __isl_give isl_union_map *release();
  inline bool is_null() const;
  inline __isl_keep isl_union_map *keep() const;
  inline __isl_give isl_union_map *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::union_map add_map(isl::map map) const;
  inline isl::union_map affine_hull() const;
  inline isl::union_map align_params(isl::space model) const;
  inline isl::union_map apply_domain(isl::union_map umap2) const;
  inline isl::union_map apply_range(isl::union_map umap2) const;
  inline isl::union_map coalesce() const;
  inline isl::boolean contains(const isl::space &space) const;
  inline isl::union_map curry() const;
  inline isl::union_set deltas() const;
  inline isl::union_map deltas_map() const;
  inline isl::union_map detect_equalities() const;
  inline unsigned int dim(isl::dim type) const;
  inline isl::union_set domain() const;
  inline isl::union_map domain_factor_domain() const;
  inline isl::union_map domain_factor_range() const;
  inline isl::union_map domain_map() const;
  inline isl::union_pw_multi_aff domain_map_union_pw_multi_aff() const;
  inline isl::union_map domain_product(isl::union_map umap2) const;
  static inline isl::union_map empty(isl::space dim);
  inline isl::union_map eq_at_multi_union_pw_aff(isl::multi_union_pw_aff mupa) const;
  inline isl::map extract_map(isl::space dim) const;
  inline isl::union_map factor_domain() const;
  inline isl::union_map factor_range() const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::union_map fixed_power(isl::val exp) const;
  inline isl::union_map flat_domain_product(isl::union_map umap2) const;
  inline isl::union_map flat_range_product(isl::union_map umap2) const;
  inline isl::stat foreach_map(const std::function<isl::stat(isl::map)> &fn) const;
  static inline isl::union_map from(isl::union_pw_multi_aff upma);
  static inline isl::union_map from(isl::multi_union_pw_aff mupa);
  static inline isl::union_map from_domain(isl::union_set uset);
  static inline isl::union_map from_domain_and_range(isl::union_set domain, isl::union_set range);
  static inline isl::union_map from_range(isl::union_set uset);
  inline isl::id get_dim_id(isl::dim type, unsigned int pos) const;
  inline uint32_t get_hash() const;
  inline isl::space get_space() const;
  inline isl::union_map gist(isl::union_map context) const;
  inline isl::union_map gist_domain(isl::union_set uset) const;
  inline isl::union_map gist_params(isl::set set) const;
  inline isl::union_map gist_range(isl::union_set uset) const;
  inline isl::union_map intersect(isl::union_map umap2) const;
  inline isl::union_map intersect_domain(isl::union_set uset) const;
  inline isl::union_map intersect_params(isl::set set) const;
  inline isl::union_map intersect_range(isl::union_set uset) const;
  inline isl::union_map intersect_range_factor_range(isl::union_map factor) const;
  inline isl::boolean involves_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::boolean is_bijective() const;
  inline isl::boolean is_disjoint(const isl::union_map &umap2) const;
  inline isl::boolean is_empty() const;
  inline isl::boolean is_equal(const isl::union_map &umap2) const;
  inline isl::boolean is_identity() const;
  inline isl::boolean is_injective() const;
  inline isl::boolean is_single_valued() const;
  inline isl::boolean is_strict_subset(const isl::union_map &umap2) const;
  inline isl::boolean is_subset(const isl::union_map &umap2) const;
  inline isl::union_map lex_ge_union_map(isl::union_map umap2) const;
  inline isl::union_map lex_gt_at_multi_union_pw_aff(isl::multi_union_pw_aff mupa) const;
  inline isl::union_map lex_gt_union_map(isl::union_map umap2) const;
  inline isl::union_map lex_le_union_map(isl::union_map umap2) const;
  inline isl::union_map lex_lt_at_multi_union_pw_aff(isl::multi_union_pw_aff mupa) const;
  inline isl::union_map lex_lt_union_map(isl::union_map umap2) const;
  inline isl::union_map lexmax() const;
  inline isl::union_map lexmin() const;
  inline isl::set params() const;
  inline isl::boolean plain_is_injective() const;
  inline isl::union_map polyhedral_hull() const;
  inline isl::union_map preimage_domain_multi_aff(isl::multi_aff ma) const;
  inline isl::union_map preimage_domain_multi_pw_aff(isl::multi_pw_aff mpa) const;
  inline isl::union_map preimage_domain_pw_multi_aff(isl::pw_multi_aff pma) const;
  inline isl::union_map preimage_domain_union_pw_multi_aff(isl::union_pw_multi_aff upma) const;
  inline isl::union_map preimage_range_multi_aff(isl::multi_aff ma) const;
  inline isl::union_map preimage_range_pw_multi_aff(isl::pw_multi_aff pma) const;
  inline isl::union_map preimage_range_union_pw_multi_aff(isl::union_pw_multi_aff upma) const;
  inline isl::union_map product(isl::union_map umap2) const;
  inline isl::union_map project_out(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::union_set range() const;
  inline isl::union_map range_curry() const;
  inline isl::union_map range_factor_domain() const;
  inline isl::union_map range_factor_range() const;
  inline isl::union_map range_map() const;
  inline isl::union_map range_product(isl::union_map umap2) const;
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
inline isl::union_map_list manage(__isl_take isl_union_map_list *ptr);
inline isl::union_map_list give(__isl_take isl_union_map_list *ptr);


class union_map_list {
  friend inline isl::union_map_list manage(__isl_take isl_union_map_list *ptr);

  isl_union_map_list *ptr = nullptr;

  inline explicit union_map_list(__isl_take isl_union_map_list *ptr);

public:
  inline /* implicit */ union_map_list();
  inline /* implicit */ union_map_list(const isl::union_map_list &obj);
  inline /* implicit */ union_map_list(std::nullptr_t);
  inline isl::union_map_list &operator=(isl::union_map_list obj);
  inline ~union_map_list();
  inline __isl_give isl_union_map_list *copy() const &;
  inline __isl_give isl_union_map_list *copy() && = delete;
  inline __isl_keep isl_union_map_list *get() const;
  inline __isl_give isl_union_map_list *release();
  inline bool is_null() const;
  inline __isl_keep isl_union_map_list *keep() const;
  inline __isl_give isl_union_map_list *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

};

// declarations for isl::union_pw_aff
inline isl::union_pw_aff manage(__isl_take isl_union_pw_aff *ptr);
inline isl::union_pw_aff give(__isl_take isl_union_pw_aff *ptr);


class union_pw_aff {
  friend inline isl::union_pw_aff manage(__isl_take isl_union_pw_aff *ptr);

  isl_union_pw_aff *ptr = nullptr;

  inline explicit union_pw_aff(__isl_take isl_union_pw_aff *ptr);

public:
  inline /* implicit */ union_pw_aff();
  inline /* implicit */ union_pw_aff(const isl::union_pw_aff &obj);
  inline /* implicit */ union_pw_aff(std::nullptr_t);
  inline /* implicit */ union_pw_aff(isl::pw_aff pa);
  inline explicit union_pw_aff(isl::union_set domain, isl::val v);
  inline explicit union_pw_aff(isl::ctx ctx, const std::string &str);
  inline isl::union_pw_aff &operator=(isl::union_pw_aff obj);
  inline ~union_pw_aff();
  inline __isl_give isl_union_pw_aff *copy() const &;
  inline __isl_give isl_union_pw_aff *copy() && = delete;
  inline __isl_keep isl_union_pw_aff *get() const;
  inline __isl_give isl_union_pw_aff *release();
  inline bool is_null() const;
  inline __isl_keep isl_union_pw_aff *keep() const;
  inline __isl_give isl_union_pw_aff *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::union_pw_aff add(isl::union_pw_aff upa2) const;
  inline isl::union_pw_aff add_pw_aff(isl::pw_aff pa) const;
  static inline isl::union_pw_aff aff_on_domain(isl::union_set domain, isl::aff aff);
  inline isl::union_pw_aff align_params(isl::space model) const;
  inline isl::union_pw_aff coalesce() const;
  inline unsigned int dim(isl::dim type) const;
  inline isl::union_set domain() const;
  inline isl::union_pw_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  static inline isl::union_pw_aff empty(isl::space space);
  inline isl::pw_aff extract_pw_aff(isl::space space) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::union_pw_aff floor() const;
  inline isl::stat foreach_pw_aff(const std::function<isl::stat(isl::pw_aff)> &fn) const;
  inline isl::space get_space() const;
  inline isl::union_pw_aff gist(isl::union_set context) const;
  inline isl::union_pw_aff gist_params(isl::set context) const;
  inline isl::union_pw_aff intersect_domain(isl::union_set uset) const;
  inline isl::union_pw_aff intersect_params(isl::set set) const;
  inline isl::boolean involves_nan() const;
  inline isl::union_pw_aff mod_val(isl::val f) const;
  inline isl::union_pw_aff neg() const;
  inline isl::boolean plain_is_equal(const isl::union_pw_aff &upa2) const;
  inline isl::union_pw_aff pullback(isl::union_pw_multi_aff upma) const;
  inline isl::union_pw_aff reset_user() const;
  inline isl::union_pw_aff scale_down_val(isl::val v) const;
  inline isl::union_pw_aff scale_val(isl::val v) const;
  inline isl::union_pw_aff sub(isl::union_pw_aff upa2) const;
  inline isl::union_pw_aff subtract_domain(isl::union_set uset) const;
  inline isl::union_pw_aff union_add(isl::union_pw_aff upa2) const;
  inline isl::union_set zero_union_set() const;
};

// declarations for isl::union_pw_aff_list
inline isl::union_pw_aff_list manage(__isl_take isl_union_pw_aff_list *ptr);
inline isl::union_pw_aff_list give(__isl_take isl_union_pw_aff_list *ptr);


class union_pw_aff_list {
  friend inline isl::union_pw_aff_list manage(__isl_take isl_union_pw_aff_list *ptr);

  isl_union_pw_aff_list *ptr = nullptr;

  inline explicit union_pw_aff_list(__isl_take isl_union_pw_aff_list *ptr);

public:
  inline /* implicit */ union_pw_aff_list();
  inline /* implicit */ union_pw_aff_list(const isl::union_pw_aff_list &obj);
  inline /* implicit */ union_pw_aff_list(std::nullptr_t);
  inline isl::union_pw_aff_list &operator=(isl::union_pw_aff_list obj);
  inline ~union_pw_aff_list();
  inline __isl_give isl_union_pw_aff_list *copy() const &;
  inline __isl_give isl_union_pw_aff_list *copy() && = delete;
  inline __isl_keep isl_union_pw_aff_list *get() const;
  inline __isl_give isl_union_pw_aff_list *release();
  inline bool is_null() const;
  inline __isl_keep isl_union_pw_aff_list *keep() const;
  inline __isl_give isl_union_pw_aff_list *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

};

// declarations for isl::union_pw_multi_aff
inline isl::union_pw_multi_aff manage(__isl_take isl_union_pw_multi_aff *ptr);
inline isl::union_pw_multi_aff give(__isl_take isl_union_pw_multi_aff *ptr);


class union_pw_multi_aff {
  friend inline isl::union_pw_multi_aff manage(__isl_take isl_union_pw_multi_aff *ptr);

  isl_union_pw_multi_aff *ptr = nullptr;

  inline explicit union_pw_multi_aff(__isl_take isl_union_pw_multi_aff *ptr);

public:
  inline /* implicit */ union_pw_multi_aff();
  inline /* implicit */ union_pw_multi_aff(const isl::union_pw_multi_aff &obj);
  inline /* implicit */ union_pw_multi_aff(std::nullptr_t);
  inline /* implicit */ union_pw_multi_aff(isl::pw_multi_aff pma);
  inline explicit union_pw_multi_aff(isl::union_set uset);
  inline explicit union_pw_multi_aff(isl::union_map umap);
  inline explicit union_pw_multi_aff(isl::ctx ctx, const std::string &str);
  inline /* implicit */ union_pw_multi_aff(isl::union_pw_aff upa);
  inline isl::union_pw_multi_aff &operator=(isl::union_pw_multi_aff obj);
  inline ~union_pw_multi_aff();
  inline __isl_give isl_union_pw_multi_aff *copy() const &;
  inline __isl_give isl_union_pw_multi_aff *copy() && = delete;
  inline __isl_keep isl_union_pw_multi_aff *get() const;
  inline __isl_give isl_union_pw_multi_aff *release();
  inline bool is_null() const;
  inline __isl_keep isl_union_pw_multi_aff *keep() const;
  inline __isl_give isl_union_pw_multi_aff *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::union_pw_multi_aff add(isl::union_pw_multi_aff upma2) const;
  inline isl::union_pw_multi_aff add_pw_multi_aff(isl::pw_multi_aff pma) const;
  inline isl::union_pw_multi_aff align_params(isl::space model) const;
  inline isl::union_pw_multi_aff coalesce() const;
  inline unsigned int dim(isl::dim type) const;
  inline isl::union_set domain() const;
  inline isl::union_pw_multi_aff drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  static inline isl::union_pw_multi_aff empty(isl::space space);
  inline isl::pw_multi_aff extract_pw_multi_aff(isl::space space) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::union_pw_multi_aff flat_range_product(isl::union_pw_multi_aff upma2) const;
  inline isl::stat foreach_pw_multi_aff(const std::function<isl::stat(isl::pw_multi_aff)> &fn) const;
  static inline isl::union_pw_multi_aff from_aff(isl::aff aff);
  static inline isl::union_pw_multi_aff from_multi_union_pw_aff(isl::multi_union_pw_aff mupa);
  static inline isl::union_pw_multi_aff from_union_set(isl::union_set uset);
  inline isl::space get_space() const;
  inline isl::union_pw_aff get_union_pw_aff(int pos) const;
  inline isl::union_pw_multi_aff gist(isl::union_set context) const;
  inline isl::union_pw_multi_aff gist_params(isl::set context) const;
  inline isl::union_pw_multi_aff intersect_domain(isl::union_set uset) const;
  inline isl::union_pw_multi_aff intersect_params(isl::set set) const;
  inline isl::boolean involves_nan() const;
  static inline isl::union_pw_multi_aff multi_val_on_domain(isl::union_set domain, isl::multi_val mv);
  inline isl::union_pw_multi_aff neg() const;
  inline isl::boolean plain_is_equal(const isl::union_pw_multi_aff &upma2) const;
  inline isl::union_pw_multi_aff pullback(isl::union_pw_multi_aff upma2) const;
  inline isl::union_pw_multi_aff reset_user() const;
  inline isl::union_pw_multi_aff scale_down_val(isl::val val) const;
  inline isl::union_pw_multi_aff scale_multi_val(isl::multi_val mv) const;
  inline isl::union_pw_multi_aff scale_val(isl::val val) const;
  inline isl::union_pw_multi_aff sub(isl::union_pw_multi_aff upma2) const;
  inline isl::union_pw_multi_aff subtract_domain(isl::union_set uset) const;
  inline isl::union_pw_multi_aff union_add(isl::union_pw_multi_aff upma2) const;
};

// declarations for isl::union_pw_multi_aff_list
inline isl::union_pw_multi_aff_list manage(__isl_take isl_union_pw_multi_aff_list *ptr);
inline isl::union_pw_multi_aff_list give(__isl_take isl_union_pw_multi_aff_list *ptr);


class union_pw_multi_aff_list {
  friend inline isl::union_pw_multi_aff_list manage(__isl_take isl_union_pw_multi_aff_list *ptr);

  isl_union_pw_multi_aff_list *ptr = nullptr;

  inline explicit union_pw_multi_aff_list(__isl_take isl_union_pw_multi_aff_list *ptr);

public:
  inline /* implicit */ union_pw_multi_aff_list();
  inline /* implicit */ union_pw_multi_aff_list(const isl::union_pw_multi_aff_list &obj);
  inline /* implicit */ union_pw_multi_aff_list(std::nullptr_t);
  inline isl::union_pw_multi_aff_list &operator=(isl::union_pw_multi_aff_list obj);
  inline ~union_pw_multi_aff_list();
  inline __isl_give isl_union_pw_multi_aff_list *copy() const &;
  inline __isl_give isl_union_pw_multi_aff_list *copy() && = delete;
  inline __isl_keep isl_union_pw_multi_aff_list *get() const;
  inline __isl_give isl_union_pw_multi_aff_list *release();
  inline bool is_null() const;
  inline __isl_keep isl_union_pw_multi_aff_list *keep() const;
  inline __isl_give isl_union_pw_multi_aff_list *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

};

// declarations for isl::union_pw_qpolynomial
inline isl::union_pw_qpolynomial manage(__isl_take isl_union_pw_qpolynomial *ptr);
inline isl::union_pw_qpolynomial give(__isl_take isl_union_pw_qpolynomial *ptr);


class union_pw_qpolynomial {
  friend inline isl::union_pw_qpolynomial manage(__isl_take isl_union_pw_qpolynomial *ptr);

  isl_union_pw_qpolynomial *ptr = nullptr;

  inline explicit union_pw_qpolynomial(__isl_take isl_union_pw_qpolynomial *ptr);

public:
  inline /* implicit */ union_pw_qpolynomial();
  inline /* implicit */ union_pw_qpolynomial(const isl::union_pw_qpolynomial &obj);
  inline /* implicit */ union_pw_qpolynomial(std::nullptr_t);
  inline explicit union_pw_qpolynomial(isl::ctx ctx, const std::string &str);
  inline isl::union_pw_qpolynomial &operator=(isl::union_pw_qpolynomial obj);
  inline ~union_pw_qpolynomial();
  inline __isl_give isl_union_pw_qpolynomial *copy() const &;
  inline __isl_give isl_union_pw_qpolynomial *copy() && = delete;
  inline __isl_keep isl_union_pw_qpolynomial *get() const;
  inline __isl_give isl_union_pw_qpolynomial *release();
  inline bool is_null() const;
  inline __isl_keep isl_union_pw_qpolynomial *keep() const;
  inline __isl_give isl_union_pw_qpolynomial *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;

  inline isl::union_pw_qpolynomial add(isl::union_pw_qpolynomial upwqp2) const;
  inline isl::union_pw_qpolynomial add_pw_qpolynomial(isl::pw_qpolynomial pwqp) const;
  inline isl::union_pw_qpolynomial align_params(isl::space model) const;
  inline isl::union_pw_qpolynomial coalesce() const;
  inline unsigned int dim(isl::dim type) const;
  inline isl::union_set domain() const;
  inline isl::union_pw_qpolynomial drop_dims(isl::dim type, unsigned int first, unsigned int n) const;
  inline isl::val eval(isl::point pnt) const;
  inline isl::pw_qpolynomial extract_pw_qpolynomial(isl::space dim) const;
  inline int find_dim_by_name(isl::dim type, const std::string &name) const;
  inline isl::stat foreach_pw_qpolynomial(const std::function<isl::stat(isl::pw_qpolynomial)> &fn) const;
  static inline isl::union_pw_qpolynomial from_pw_qpolynomial(isl::pw_qpolynomial pwqp);
  inline isl::space get_space() const;
  inline isl::union_pw_qpolynomial gist(isl::union_set context) const;
  inline isl::union_pw_qpolynomial gist_params(isl::set context) const;
  inline isl::union_pw_qpolynomial intersect_domain(isl::union_set uset) const;
  inline isl::union_pw_qpolynomial intersect_params(isl::set set) const;
  inline isl::boolean involves_nan() const;
  inline isl::union_pw_qpolynomial mul(isl::union_pw_qpolynomial upwqp2) const;
  inline isl::union_pw_qpolynomial neg() const;
  inline isl::boolean plain_is_equal(const isl::union_pw_qpolynomial &upwqp2) const;
  inline isl::union_pw_qpolynomial reset_user() const;
  inline isl::union_pw_qpolynomial scale_down_val(isl::val v) const;
  inline isl::union_pw_qpolynomial scale_val(isl::val v) const;
  inline isl::union_pw_qpolynomial sub(isl::union_pw_qpolynomial upwqp2) const;
  inline isl::union_pw_qpolynomial subtract_domain(isl::union_set uset) const;
  inline isl::union_pw_qpolynomial to_polynomial(int sign) const;
  static inline isl::union_pw_qpolynomial zero(isl::space dim);
};

// declarations for isl::union_set
inline isl::union_set manage(__isl_take isl_union_set *ptr);
inline isl::union_set give(__isl_take isl_union_set *ptr);


class union_set {
  friend inline isl::union_set manage(__isl_take isl_union_set *ptr);

  isl_union_set *ptr = nullptr;

  inline explicit union_set(__isl_take isl_union_set *ptr);

public:
  inline /* implicit */ union_set();
  inline /* implicit */ union_set(const isl::union_set &obj);
  inline /* implicit */ union_set(std::nullptr_t);
  inline /* implicit */ union_set(isl::point pnt);
  inline explicit union_set(isl::ctx ctx, const std::string &str);
  inline /* implicit */ union_set(isl::basic_set bset);
  inline /* implicit */ union_set(isl::set set);
  inline isl::union_set &operator=(isl::union_set obj);
  inline ~union_set();
  inline __isl_give isl_union_set *copy() const &;
  inline __isl_give isl_union_set *copy() && = delete;
  inline __isl_keep isl_union_set *get() const;
  inline __isl_give isl_union_set *release();
  inline bool is_null() const;
  inline __isl_keep isl_union_set *keep() const;
  inline __isl_give isl_union_set *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::union_set add_set(isl::set set) const;
  inline isl::union_set affine_hull() const;
  inline isl::union_set align_params(isl::space model) const;
  inline isl::union_set apply(isl::union_map umap) const;
  inline isl::union_set coalesce() const;
  inline isl::union_set coefficients() const;
  inline isl::schedule compute_schedule(isl::union_map validity, isl::union_map proximity) const;
  inline isl::boolean contains(const isl::space &space) const;
  inline isl::union_set detect_equalities() const;
  inline unsigned int dim(isl::dim type) const;
  static inline isl::union_set empty(isl::space dim);
  inline isl::set extract_set(isl::space dim) const;
  inline isl::stat foreach_point(const std::function<isl::stat(isl::point)> &fn) const;
  inline isl::stat foreach_set(const std::function<isl::stat(isl::set)> &fn) const;
  inline isl::basic_set_list get_basic_set_list() const;
  inline uint32_t get_hash() const;
  inline isl::space get_space() const;
  inline isl::union_set gist(isl::union_set context) const;
  inline isl::union_set gist_params(isl::set set) const;
  inline isl::union_map identity() const;
  inline isl::union_pw_multi_aff identity_union_pw_multi_aff() const;
  inline isl::union_set intersect(isl::union_set uset2) const;
  inline isl::union_set intersect_params(isl::set set) const;
  inline isl::boolean is_disjoint(const isl::union_set &uset2) const;
  inline isl::boolean is_empty() const;
  inline isl::boolean is_equal(const isl::union_set &uset2) const;
  inline isl::boolean is_params() const;
  inline isl::boolean is_strict_subset(const isl::union_set &uset2) const;
  inline isl::boolean is_subset(const isl::union_set &uset2) const;
  inline isl::union_map lex_ge_union_set(isl::union_set uset2) const;
  inline isl::union_map lex_gt_union_set(isl::union_set uset2) const;
  inline isl::union_map lex_le_union_set(isl::union_set uset2) const;
  inline isl::union_map lex_lt_union_set(isl::union_set uset2) const;
  inline isl::union_set lexmax() const;
  inline isl::union_set lexmin() const;
  inline isl::multi_val min_multi_union_pw_aff(const isl::multi_union_pw_aff &obj) const;
  inline isl::set params() const;
  inline isl::union_set polyhedral_hull() const;
  inline isl::union_set preimage_multi_aff(isl::multi_aff ma) const;
  inline isl::union_set preimage_pw_multi_aff(isl::pw_multi_aff pma) const;
  inline isl::union_set preimage_union_pw_multi_aff(isl::union_pw_multi_aff upma) const;
  inline isl::union_set product(isl::union_set uset2) const;
  inline isl::union_set project_out(isl::dim type, unsigned int first, unsigned int n) const;
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
inline isl::union_set_list manage(__isl_take isl_union_set_list *ptr);
inline isl::union_set_list give(__isl_take isl_union_set_list *ptr);


class union_set_list {
  friend inline isl::union_set_list manage(__isl_take isl_union_set_list *ptr);

  isl_union_set_list *ptr = nullptr;

  inline explicit union_set_list(__isl_take isl_union_set_list *ptr);

public:
  inline /* implicit */ union_set_list();
  inline /* implicit */ union_set_list(const isl::union_set_list &obj);
  inline /* implicit */ union_set_list(std::nullptr_t);
  inline isl::union_set_list &operator=(isl::union_set_list obj);
  inline ~union_set_list();
  inline __isl_give isl_union_set_list *copy() const &;
  inline __isl_give isl_union_set_list *copy() && = delete;
  inline __isl_keep isl_union_set_list *get() const;
  inline __isl_give isl_union_set_list *release();
  inline bool is_null() const;
  inline __isl_keep isl_union_set_list *keep() const;
  inline __isl_give isl_union_set_list *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

};

// declarations for isl::val
inline isl::val manage(__isl_take isl_val *ptr);
inline isl::val give(__isl_take isl_val *ptr);


class val {
  friend inline isl::val manage(__isl_take isl_val *ptr);

  isl_val *ptr = nullptr;

  inline explicit val(__isl_take isl_val *ptr);

public:
  inline /* implicit */ val();
  inline /* implicit */ val(const isl::val &obj);
  inline /* implicit */ val(std::nullptr_t);
  inline explicit val(isl::ctx ctx, long i);
  inline explicit val(isl::ctx ctx, const std::string &str);
  inline isl::val &operator=(isl::val obj);
  inline ~val();
  inline __isl_give isl_val *copy() const &;
  inline __isl_give isl_val *copy() && = delete;
  inline __isl_keep isl_val *get() const;
  inline __isl_give isl_val *release();
  inline bool is_null() const;
  inline __isl_keep isl_val *keep() const;
  inline __isl_give isl_val *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline std::string to_str() const;
  inline void dump() const;

  inline isl::val two_exp() const;
  inline isl::val abs() const;
  inline isl::boolean abs_eq(const isl::val &v2) const;
  inline isl::val add(isl::val v2) const;
  inline isl::val add_ui(unsigned long v2) const;
  inline isl::val ceil() const;
  inline int cmp_si(long i) const;
  inline isl::val div(isl::val v2) const;
  inline isl::val div_ui(unsigned long v2) const;
  inline isl::boolean eq(const isl::val &v2) const;
  inline isl::val floor() const;
  inline isl::val gcd(isl::val v2) const;
  inline isl::boolean ge(const isl::val &v2) const;
  inline uint32_t get_hash() const;
  inline long get_num_si() const;
  inline isl::boolean gt(const isl::val &v2) const;
  static inline isl::val infty(isl::ctx ctx);
  static inline isl::val int_from_ui(isl::ctx ctx, unsigned long u);
  inline isl::val inv() const;
  inline isl::boolean is_divisible_by(const isl::val &v2) const;
  inline isl::boolean is_infty() const;
  inline isl::boolean is_int() const;
  inline isl::boolean is_nan() const;
  inline isl::boolean is_neg() const;
  inline isl::boolean is_neginfty() const;
  inline isl::boolean is_negone() const;
  inline isl::boolean is_nonneg() const;
  inline isl::boolean is_nonpos() const;
  inline isl::boolean is_one() const;
  inline isl::boolean is_pos() const;
  inline isl::boolean is_rat() const;
  inline isl::boolean is_zero() const;
  inline isl::boolean le(const isl::val &v2) const;
  inline isl::boolean lt(const isl::val &v2) const;
  inline isl::val max(isl::val v2) const;
  inline isl::val min(isl::val v2) const;
  inline isl::val mod(isl::val v2) const;
  inline isl::val mul(isl::val v2) const;
  inline isl::val mul_ui(unsigned long v2) const;
  static inline isl::val nan(isl::ctx ctx);
  inline isl::boolean ne(const isl::val &v2) const;
  inline isl::val neg() const;
  static inline isl::val neginfty(isl::ctx ctx);
  static inline isl::val negone(isl::ctx ctx);
  static inline isl::val one(isl::ctx ctx);
  inline isl::val set_si(long i) const;
  inline int sgn() const;
  inline isl::val sub(isl::val v2) const;
  inline isl::val sub_ui(unsigned long v2) const;
  inline isl::val trunc() const;
  static inline isl::val zero(isl::ctx ctx);
};

// declarations for isl::val_list
inline isl::val_list manage(__isl_take isl_val_list *ptr);
inline isl::val_list give(__isl_take isl_val_list *ptr);


class val_list {
  friend inline isl::val_list manage(__isl_take isl_val_list *ptr);

  isl_val_list *ptr = nullptr;

  inline explicit val_list(__isl_take isl_val_list *ptr);

public:
  inline /* implicit */ val_list();
  inline /* implicit */ val_list(const isl::val_list &obj);
  inline /* implicit */ val_list(std::nullptr_t);
  inline isl::val_list &operator=(isl::val_list obj);
  inline ~val_list();
  inline __isl_give isl_val_list *copy() const &;
  inline __isl_give isl_val_list *copy() && = delete;
  inline __isl_keep isl_val_list *get() const;
  inline __isl_give isl_val_list *release();
  inline bool is_null() const;
  inline __isl_keep isl_val_list *keep() const;
  inline __isl_give isl_val_list *take();
  inline explicit operator bool() const;
  inline isl::ctx get_ctx() const;
  inline void dump() const;

};

// implementations for isl::aff
isl::aff manage(__isl_take isl_aff *ptr) {
  return aff(ptr);
}
isl::aff give(__isl_take isl_aff *ptr) {
  return manage(ptr);
}


aff::aff()
    : ptr(nullptr) {}

aff::aff(const isl::aff &obj)
    : ptr(obj.copy()) {}
aff::aff(std::nullptr_t)
    : ptr(nullptr) {}


aff::aff(__isl_take isl_aff *ptr)
    : ptr(ptr) {}

aff::aff(isl::local_space ls) {
  auto res = isl_aff_zero_on_domain(ls.release());
  ptr = res;
}
aff::aff(isl::local_space ls, isl::val val) {
  auto res = isl_aff_val_on_domain(ls.release(), val.release());
  ptr = res;
}
aff::aff(isl::ctx ctx, const std::string &str) {
  auto res = isl_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

aff &aff::operator=(isl::aff obj) {
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
__isl_keep isl_aff *aff::keep() const {
  return get();
}

__isl_give isl_aff *aff::take() {
  return release();
}

aff::operator bool() const {
  return !is_null();
}

isl::ctx aff::get_ctx() const {
  return isl::ctx(isl_aff_get_ctx(ptr));
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


isl::aff aff::add(isl::aff aff2) const {
  auto res = isl_aff_add(copy(), aff2.release());
  return manage(res);
}

isl::aff aff::add_coefficient_si(isl::dim type, int pos, int v) const {
  auto res = isl_aff_add_coefficient_si(copy(), static_cast<enum isl_dim_type>(type), pos, v);
  return manage(res);
}

isl::aff aff::add_coefficient_val(isl::dim type, int pos, isl::val v) const {
  auto res = isl_aff_add_coefficient_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

isl::aff aff::add_constant_num_si(int v) const {
  auto res = isl_aff_add_constant_num_si(copy(), v);
  return manage(res);
}

isl::aff aff::add_constant_si(int v) const {
  auto res = isl_aff_add_constant_si(copy(), v);
  return manage(res);
}

isl::aff aff::add_constant_val(isl::val v) const {
  auto res = isl_aff_add_constant_val(copy(), v.release());
  return manage(res);
}

isl::aff aff::add_dims(isl::dim type, unsigned int n) const {
  auto res = isl_aff_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::aff aff::align_params(isl::space model) const {
  auto res = isl_aff_align_params(copy(), model.release());
  return manage(res);
}

isl::aff aff::ceil() const {
  auto res = isl_aff_ceil(copy());
  return manage(res);
}

int aff::coefficient_sgn(isl::dim type, int pos) const {
  auto res = isl_aff_coefficient_sgn(get(), static_cast<enum isl_dim_type>(type), pos);
  return res;
}

int aff::dim(isl::dim type) const {
  auto res = isl_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::aff aff::div(isl::aff aff2) const {
  auto res = isl_aff_div(copy(), aff2.release());
  return manage(res);
}

isl::aff aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_set aff::eq_basic_set(isl::aff aff2) const {
  auto res = isl_aff_eq_basic_set(copy(), aff2.release());
  return manage(res);
}

isl::set aff::eq_set(isl::aff aff2) const {
  auto res = isl_aff_eq_set(copy(), aff2.release());
  return manage(res);
}

int aff::find_dim_by_name(isl::dim type, const std::string &name) const {
  auto res = isl_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::aff aff::floor() const {
  auto res = isl_aff_floor(copy());
  return manage(res);
}

isl::basic_set aff::ge_basic_set(isl::aff aff2) const {
  auto res = isl_aff_ge_basic_set(copy(), aff2.release());
  return manage(res);
}

isl::set aff::ge_set(isl::aff aff2) const {
  auto res = isl_aff_ge_set(copy(), aff2.release());
  return manage(res);
}

isl::val aff::get_coefficient_val(isl::dim type, int pos) const {
  auto res = isl_aff_get_coefficient_val(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::val aff::get_constant_val() const {
  auto res = isl_aff_get_constant_val(get());
  return manage(res);
}

isl::val aff::get_denominator_val() const {
  auto res = isl_aff_get_denominator_val(get());
  return manage(res);
}

std::string aff::get_dim_name(isl::dim type, unsigned int pos) const {
  auto res = isl_aff_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

isl::aff aff::get_div(int pos) const {
  auto res = isl_aff_get_div(get(), pos);
  return manage(res);
}

isl::local_space aff::get_domain_local_space() const {
  auto res = isl_aff_get_domain_local_space(get());
  return manage(res);
}

isl::space aff::get_domain_space() const {
  auto res = isl_aff_get_domain_space(get());
  return manage(res);
}

uint32_t aff::get_hash() const {
  auto res = isl_aff_get_hash(get());
  return res;
}

isl::local_space aff::get_local_space() const {
  auto res = isl_aff_get_local_space(get());
  return manage(res);
}

isl::space aff::get_space() const {
  auto res = isl_aff_get_space(get());
  return manage(res);
}

isl::aff aff::gist(isl::set context) const {
  auto res = isl_aff_gist(copy(), context.release());
  return manage(res);
}

isl::aff aff::gist_params(isl::set context) const {
  auto res = isl_aff_gist_params(copy(), context.release());
  return manage(res);
}

isl::basic_set aff::gt_basic_set(isl::aff aff2) const {
  auto res = isl_aff_gt_basic_set(copy(), aff2.release());
  return manage(res);
}

isl::set aff::gt_set(isl::aff aff2) const {
  auto res = isl_aff_gt_set(copy(), aff2.release());
  return manage(res);
}

isl::aff aff::insert_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_aff_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::boolean aff::involves_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_aff_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::boolean aff::is_cst() const {
  auto res = isl_aff_is_cst(get());
  return manage(res);
}

isl::boolean aff::is_nan() const {
  auto res = isl_aff_is_nan(get());
  return manage(res);
}

isl::basic_set aff::le_basic_set(isl::aff aff2) const {
  auto res = isl_aff_le_basic_set(copy(), aff2.release());
  return manage(res);
}

isl::set aff::le_set(isl::aff aff2) const {
  auto res = isl_aff_le_set(copy(), aff2.release());
  return manage(res);
}

isl::basic_set aff::lt_basic_set(isl::aff aff2) const {
  auto res = isl_aff_lt_basic_set(copy(), aff2.release());
  return manage(res);
}

isl::set aff::lt_set(isl::aff aff2) const {
  auto res = isl_aff_lt_set(copy(), aff2.release());
  return manage(res);
}

isl::aff aff::mod(isl::val mod) const {
  auto res = isl_aff_mod_val(copy(), mod.release());
  return manage(res);
}

isl::aff aff::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const {
  auto res = isl_aff_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl::aff aff::mul(isl::aff aff2) const {
  auto res = isl_aff_mul(copy(), aff2.release());
  return manage(res);
}

isl::aff aff::nan_on_domain(isl::local_space ls) {
  auto res = isl_aff_nan_on_domain(ls.release());
  return manage(res);
}

isl::set aff::ne_set(isl::aff aff2) const {
  auto res = isl_aff_ne_set(copy(), aff2.release());
  return manage(res);
}

isl::aff aff::neg() const {
  auto res = isl_aff_neg(copy());
  return manage(res);
}

isl::basic_set aff::neg_basic_set() const {
  auto res = isl_aff_neg_basic_set(copy());
  return manage(res);
}

isl::boolean aff::plain_is_equal(const isl::aff &aff2) const {
  auto res = isl_aff_plain_is_equal(get(), aff2.get());
  return manage(res);
}

isl::boolean aff::plain_is_zero() const {
  auto res = isl_aff_plain_is_zero(get());
  return manage(res);
}

isl::aff aff::project_domain_on_params() const {
  auto res = isl_aff_project_domain_on_params(copy());
  return manage(res);
}

isl::aff aff::pullback(isl::multi_aff ma) const {
  auto res = isl_aff_pullback_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::aff aff::pullback_aff(isl::aff aff2) const {
  auto res = isl_aff_pullback_aff(copy(), aff2.release());
  return manage(res);
}

isl::aff aff::scale(isl::val v) const {
  auto res = isl_aff_scale_val(copy(), v.release());
  return manage(res);
}

isl::aff aff::scale_down(isl::val v) const {
  auto res = isl_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::aff aff::scale_down_ui(unsigned int f) const {
  auto res = isl_aff_scale_down_ui(copy(), f);
  return manage(res);
}

isl::aff aff::set_coefficient_si(isl::dim type, int pos, int v) const {
  auto res = isl_aff_set_coefficient_si(copy(), static_cast<enum isl_dim_type>(type), pos, v);
  return manage(res);
}

isl::aff aff::set_coefficient_val(isl::dim type, int pos, isl::val v) const {
  auto res = isl_aff_set_coefficient_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

isl::aff aff::set_constant_si(int v) const {
  auto res = isl_aff_set_constant_si(copy(), v);
  return manage(res);
}

isl::aff aff::set_constant_val(isl::val v) const {
  auto res = isl_aff_set_constant_val(copy(), v.release());
  return manage(res);
}

isl::aff aff::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const {
  auto res = isl_aff_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::aff aff::set_tuple_id(isl::dim type, isl::id id) const {
  auto res = isl_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::aff aff::sub(isl::aff aff2) const {
  auto res = isl_aff_sub(copy(), aff2.release());
  return manage(res);
}

isl::aff aff::var_on_domain(isl::local_space ls, isl::dim type, unsigned int pos) {
  auto res = isl_aff_var_on_domain(ls.release(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::basic_set aff::zero_basic_set() const {
  auto res = isl_aff_zero_basic_set(copy());
  return manage(res);
}

// implementations for isl::aff_list
isl::aff_list manage(__isl_take isl_aff_list *ptr) {
  return aff_list(ptr);
}
isl::aff_list give(__isl_take isl_aff_list *ptr) {
  return manage(ptr);
}


aff_list::aff_list()
    : ptr(nullptr) {}

aff_list::aff_list(const isl::aff_list &obj)
    : ptr(obj.copy()) {}
aff_list::aff_list(std::nullptr_t)
    : ptr(nullptr) {}


aff_list::aff_list(__isl_take isl_aff_list *ptr)
    : ptr(ptr) {}


aff_list &aff_list::operator=(isl::aff_list obj) {
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
__isl_keep isl_aff_list *aff_list::keep() const {
  return get();
}

__isl_give isl_aff_list *aff_list::take() {
  return release();
}

aff_list::operator bool() const {
  return !is_null();
}

isl::ctx aff_list::get_ctx() const {
  return isl::ctx(isl_aff_list_get_ctx(ptr));
}



void aff_list::dump() const {
  isl_aff_list_dump(get());
}



// implementations for isl::ast_build
isl::ast_build manage(__isl_take isl_ast_build *ptr) {
  return ast_build(ptr);
}
isl::ast_build give(__isl_take isl_ast_build *ptr) {
  return manage(ptr);
}


ast_build::ast_build()
    : ptr(nullptr) {}

ast_build::ast_build(const isl::ast_build &obj)
    : ptr(obj.copy()) {}
ast_build::ast_build(std::nullptr_t)
    : ptr(nullptr) {}


ast_build::ast_build(__isl_take isl_ast_build *ptr)
    : ptr(ptr) {}

ast_build::ast_build(isl::ctx ctx) {
  auto res = isl_ast_build_alloc(ctx.release());
  ptr = res;
}

ast_build &ast_build::operator=(isl::ast_build obj) {
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
__isl_keep isl_ast_build *ast_build::keep() const {
  return get();
}

__isl_give isl_ast_build *ast_build::take() {
  return release();
}

ast_build::operator bool() const {
  return !is_null();
}

isl::ctx ast_build::get_ctx() const {
  return isl::ctx(isl_ast_build_get_ctx(ptr));
}




isl::ast_expr ast_build::access_from(isl::pw_multi_aff pma) const {
  auto res = isl_ast_build_access_from_pw_multi_aff(get(), pma.release());
  return manage(res);
}

isl::ast_expr ast_build::access_from(isl::multi_pw_aff mpa) const {
  auto res = isl_ast_build_access_from_multi_pw_aff(get(), mpa.release());
  return manage(res);
}

isl::ast_node ast_build::ast_from_schedule(isl::union_map schedule) const {
  auto res = isl_ast_build_ast_from_schedule(get(), schedule.release());
  return manage(res);
}

isl::ast_expr ast_build::call_from(isl::pw_multi_aff pma) const {
  auto res = isl_ast_build_call_from_pw_multi_aff(get(), pma.release());
  return manage(res);
}

isl::ast_expr ast_build::call_from(isl::multi_pw_aff mpa) const {
  auto res = isl_ast_build_call_from_multi_pw_aff(get(), mpa.release());
  return manage(res);
}

isl::ast_expr ast_build::expr_from(isl::set set) const {
  auto res = isl_ast_build_expr_from_set(get(), set.release());
  return manage(res);
}

isl::ast_expr ast_build::expr_from(isl::pw_aff pa) const {
  auto res = isl_ast_build_expr_from_pw_aff(get(), pa.release());
  return manage(res);
}

isl::ast_build ast_build::from_context(isl::set set) {
  auto res = isl_ast_build_from_context(set.release());
  return manage(res);
}

isl::union_map ast_build::get_schedule() const {
  auto res = isl_ast_build_get_schedule(get());
  return manage(res);
}

isl::space ast_build::get_schedule_space() const {
  auto res = isl_ast_build_get_schedule_space(get());
  return manage(res);
}

isl::ast_node ast_build::node_from_schedule(isl::schedule schedule) const {
  auto res = isl_ast_build_node_from_schedule(get(), schedule.release());
  return manage(res);
}

isl::ast_node ast_build::node_from_schedule_map(isl::union_map schedule) const {
  auto res = isl_ast_build_node_from_schedule_map(get(), schedule.release());
  return manage(res);
}

isl::ast_build ast_build::restrict(isl::set set) const {
  auto res = isl_ast_build_restrict(copy(), set.release());
  return manage(res);
}

// implementations for isl::ast_expr
isl::ast_expr manage(__isl_take isl_ast_expr *ptr) {
  return ast_expr(ptr);
}
isl::ast_expr give(__isl_take isl_ast_expr *ptr) {
  return manage(ptr);
}


ast_expr::ast_expr()
    : ptr(nullptr) {}

ast_expr::ast_expr(const isl::ast_expr &obj)
    : ptr(obj.copy()) {}
ast_expr::ast_expr(std::nullptr_t)
    : ptr(nullptr) {}


ast_expr::ast_expr(__isl_take isl_ast_expr *ptr)
    : ptr(ptr) {}


ast_expr &ast_expr::operator=(isl::ast_expr obj) {
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
__isl_keep isl_ast_expr *ast_expr::keep() const {
  return get();
}

__isl_give isl_ast_expr *ast_expr::take() {
  return release();
}

ast_expr::operator bool() const {
  return !is_null();
}

isl::ctx ast_expr::get_ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
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


isl::ast_expr ast_expr::access(isl::ast_expr_list indices) const {
  auto res = isl_ast_expr_access(copy(), indices.release());
  return manage(res);
}

isl::ast_expr ast_expr::add(isl::ast_expr expr2) const {
  auto res = isl_ast_expr_add(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::address_of() const {
  auto res = isl_ast_expr_address_of(copy());
  return manage(res);
}

isl::ast_expr ast_expr::call(isl::ast_expr_list arguments) const {
  auto res = isl_ast_expr_call(copy(), arguments.release());
  return manage(res);
}

isl::ast_expr ast_expr::div(isl::ast_expr expr2) const {
  auto res = isl_ast_expr_div(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::eq(isl::ast_expr expr2) const {
  auto res = isl_ast_expr_eq(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::from_id(isl::id id) {
  auto res = isl_ast_expr_from_id(id.release());
  return manage(res);
}

isl::ast_expr ast_expr::from_val(isl::val v) {
  auto res = isl_ast_expr_from_val(v.release());
  return manage(res);
}

isl::ast_expr ast_expr::ge(isl::ast_expr expr2) const {
  auto res = isl_ast_expr_ge(copy(), expr2.release());
  return manage(res);
}

isl::id ast_expr::get_id() const {
  auto res = isl_ast_expr_get_id(get());
  return manage(res);
}

isl::ast_expr ast_expr::get_op_arg(int pos) const {
  auto res = isl_ast_expr_get_op_arg(get(), pos);
  return manage(res);
}

isl::val ast_expr::get_val() const {
  auto res = isl_ast_expr_get_val(get());
  return manage(res);
}

isl::ast_expr ast_expr::gt(isl::ast_expr expr2) const {
  auto res = isl_ast_expr_gt(copy(), expr2.release());
  return manage(res);
}

isl::boolean ast_expr::is_equal(const isl::ast_expr &expr2) const {
  auto res = isl_ast_expr_is_equal(get(), expr2.get());
  return manage(res);
}

isl::ast_expr ast_expr::le(isl::ast_expr expr2) const {
  auto res = isl_ast_expr_le(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::lt(isl::ast_expr expr2) const {
  auto res = isl_ast_expr_lt(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::mul(isl::ast_expr expr2) const {
  auto res = isl_ast_expr_mul(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::neg() const {
  auto res = isl_ast_expr_neg(copy());
  return manage(res);
}

isl::ast_expr ast_expr::pdiv_q(isl::ast_expr expr2) const {
  auto res = isl_ast_expr_pdiv_q(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::pdiv_r(isl::ast_expr expr2) const {
  auto res = isl_ast_expr_pdiv_r(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::set_op_arg(int pos, isl::ast_expr arg) const {
  auto res = isl_ast_expr_set_op_arg(copy(), pos, arg.release());
  return manage(res);
}

isl::ast_expr ast_expr::sub(isl::ast_expr expr2) const {
  auto res = isl_ast_expr_sub(copy(), expr2.release());
  return manage(res);
}

isl::ast_expr ast_expr::substitute_ids(isl::id_to_ast_expr id2expr) const {
  auto res = isl_ast_expr_substitute_ids(copy(), id2expr.release());
  return manage(res);
}

std::string ast_expr::to_C_str() const {
  auto res = isl_ast_expr_to_C_str(get());
  std::string tmp(res);
  free(res);
  return tmp;
}

// implementations for isl::ast_expr_list
isl::ast_expr_list manage(__isl_take isl_ast_expr_list *ptr) {
  return ast_expr_list(ptr);
}
isl::ast_expr_list give(__isl_take isl_ast_expr_list *ptr) {
  return manage(ptr);
}


ast_expr_list::ast_expr_list()
    : ptr(nullptr) {}

ast_expr_list::ast_expr_list(const isl::ast_expr_list &obj)
    : ptr(obj.copy()) {}
ast_expr_list::ast_expr_list(std::nullptr_t)
    : ptr(nullptr) {}


ast_expr_list::ast_expr_list(__isl_take isl_ast_expr_list *ptr)
    : ptr(ptr) {}


ast_expr_list &ast_expr_list::operator=(isl::ast_expr_list obj) {
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
__isl_keep isl_ast_expr_list *ast_expr_list::keep() const {
  return get();
}

__isl_give isl_ast_expr_list *ast_expr_list::take() {
  return release();
}

ast_expr_list::operator bool() const {
  return !is_null();
}

isl::ctx ast_expr_list::get_ctx() const {
  return isl::ctx(isl_ast_expr_list_get_ctx(ptr));
}



void ast_expr_list::dump() const {
  isl_ast_expr_list_dump(get());
}



// implementations for isl::ast_node
isl::ast_node manage(__isl_take isl_ast_node *ptr) {
  return ast_node(ptr);
}
isl::ast_node give(__isl_take isl_ast_node *ptr) {
  return manage(ptr);
}


ast_node::ast_node()
    : ptr(nullptr) {}

ast_node::ast_node(const isl::ast_node &obj)
    : ptr(obj.copy()) {}
ast_node::ast_node(std::nullptr_t)
    : ptr(nullptr) {}


ast_node::ast_node(__isl_take isl_ast_node *ptr)
    : ptr(ptr) {}


ast_node &ast_node::operator=(isl::ast_node obj) {
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
__isl_keep isl_ast_node *ast_node::keep() const {
  return get();
}

__isl_give isl_ast_node *ast_node::take() {
  return release();
}

ast_node::operator bool() const {
  return !is_null();
}

isl::ctx ast_node::get_ctx() const {
  return isl::ctx(isl_ast_node_get_ctx(ptr));
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


isl::ast_node ast_node::alloc_user(isl::ast_expr expr) {
  auto res = isl_ast_node_alloc_user(expr.release());
  return manage(res);
}

isl::ast_node_list ast_node::block_get_children() const {
  auto res = isl_ast_node_block_get_children(get());
  return manage(res);
}

isl::ast_node ast_node::for_get_body() const {
  auto res = isl_ast_node_for_get_body(get());
  return manage(res);
}

isl::ast_expr ast_node::for_get_cond() const {
  auto res = isl_ast_node_for_get_cond(get());
  return manage(res);
}

isl::ast_expr ast_node::for_get_inc() const {
  auto res = isl_ast_node_for_get_inc(get());
  return manage(res);
}

isl::ast_expr ast_node::for_get_init() const {
  auto res = isl_ast_node_for_get_init(get());
  return manage(res);
}

isl::ast_expr ast_node::for_get_iterator() const {
  auto res = isl_ast_node_for_get_iterator(get());
  return manage(res);
}

isl::boolean ast_node::for_is_degenerate() const {
  auto res = isl_ast_node_for_is_degenerate(get());
  return manage(res);
}

isl::id ast_node::get_annotation() const {
  auto res = isl_ast_node_get_annotation(get());
  return manage(res);
}

isl::ast_expr ast_node::if_get_cond() const {
  auto res = isl_ast_node_if_get_cond(get());
  return manage(res);
}

isl::ast_node ast_node::if_get_else() const {
  auto res = isl_ast_node_if_get_else(get());
  return manage(res);
}

isl::ast_node ast_node::if_get_then() const {
  auto res = isl_ast_node_if_get_then(get());
  return manage(res);
}

isl::boolean ast_node::if_has_else() const {
  auto res = isl_ast_node_if_has_else(get());
  return manage(res);
}

isl::id ast_node::mark_get_id() const {
  auto res = isl_ast_node_mark_get_id(get());
  return manage(res);
}

isl::ast_node ast_node::mark_get_node() const {
  auto res = isl_ast_node_mark_get_node(get());
  return manage(res);
}

isl::ast_node ast_node::set_annotation(isl::id annotation) const {
  auto res = isl_ast_node_set_annotation(copy(), annotation.release());
  return manage(res);
}

std::string ast_node::to_C_str() const {
  auto res = isl_ast_node_to_C_str(get());
  std::string tmp(res);
  free(res);
  return tmp;
}

isl::ast_expr ast_node::user_get_expr() const {
  auto res = isl_ast_node_user_get_expr(get());
  return manage(res);
}

// implementations for isl::ast_node_list
isl::ast_node_list manage(__isl_take isl_ast_node_list *ptr) {
  return ast_node_list(ptr);
}
isl::ast_node_list give(__isl_take isl_ast_node_list *ptr) {
  return manage(ptr);
}


ast_node_list::ast_node_list()
    : ptr(nullptr) {}

ast_node_list::ast_node_list(const isl::ast_node_list &obj)
    : ptr(obj.copy()) {}
ast_node_list::ast_node_list(std::nullptr_t)
    : ptr(nullptr) {}


ast_node_list::ast_node_list(__isl_take isl_ast_node_list *ptr)
    : ptr(ptr) {}


ast_node_list &ast_node_list::operator=(isl::ast_node_list obj) {
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
__isl_keep isl_ast_node_list *ast_node_list::keep() const {
  return get();
}

__isl_give isl_ast_node_list *ast_node_list::take() {
  return release();
}

ast_node_list::operator bool() const {
  return !is_null();
}

isl::ctx ast_node_list::get_ctx() const {
  return isl::ctx(isl_ast_node_list_get_ctx(ptr));
}



void ast_node_list::dump() const {
  isl_ast_node_list_dump(get());
}



// implementations for isl::band_list
isl::band_list manage(__isl_take isl_band_list *ptr) {
  return band_list(ptr);
}
isl::band_list give(__isl_take isl_band_list *ptr) {
  return manage(ptr);
}


band_list::band_list()
    : ptr(nullptr) {}

band_list::band_list(const isl::band_list &obj)
    : ptr(obj.copy()) {}
band_list::band_list(std::nullptr_t)
    : ptr(nullptr) {}


band_list::band_list(__isl_take isl_band_list *ptr)
    : ptr(ptr) {}


band_list &band_list::operator=(isl::band_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

band_list::~band_list() {
  if (ptr)
    isl_band_list_free(ptr);
}

__isl_give isl_band_list *band_list::copy() const & {
  return isl_band_list_copy(ptr);
}

__isl_keep isl_band_list *band_list::get() const {
  return ptr;
}

__isl_give isl_band_list *band_list::release() {
  isl_band_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool band_list::is_null() const {
  return ptr == nullptr;
}
__isl_keep isl_band_list *band_list::keep() const {
  return get();
}

__isl_give isl_band_list *band_list::take() {
  return release();
}

band_list::operator bool() const {
  return !is_null();
}

isl::ctx band_list::get_ctx() const {
  return isl::ctx(isl_band_list_get_ctx(ptr));
}



void band_list::dump() const {
  isl_band_list_dump(get());
}



// implementations for isl::basic_map
isl::basic_map manage(__isl_take isl_basic_map *ptr) {
  return basic_map(ptr);
}
isl::basic_map give(__isl_take isl_basic_map *ptr) {
  return manage(ptr);
}


basic_map::basic_map()
    : ptr(nullptr) {}

basic_map::basic_map(const isl::basic_map &obj)
    : ptr(obj.copy()) {}
basic_map::basic_map(std::nullptr_t)
    : ptr(nullptr) {}


basic_map::basic_map(__isl_take isl_basic_map *ptr)
    : ptr(ptr) {}

basic_map::basic_map(isl::ctx ctx, const std::string &str) {
  auto res = isl_basic_map_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

basic_map &basic_map::operator=(isl::basic_map obj) {
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
__isl_keep isl_basic_map *basic_map::keep() const {
  return get();
}

__isl_give isl_basic_map *basic_map::take() {
  return release();
}

basic_map::operator bool() const {
  return !is_null();
}

isl::ctx basic_map::get_ctx() const {
  return isl::ctx(isl_basic_map_get_ctx(ptr));
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


isl::basic_map basic_map::add_constraint(isl::constraint constraint) const {
  auto res = isl_basic_map_add_constraint(copy(), constraint.release());
  return manage(res);
}

isl::basic_map basic_map::add_dims(isl::dim type, unsigned int n) const {
  auto res = isl_basic_map_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::basic_map basic_map::affine_hull() const {
  auto res = isl_basic_map_affine_hull(copy());
  return manage(res);
}

isl::basic_map basic_map::align_params(isl::space model) const {
  auto res = isl_basic_map_align_params(copy(), model.release());
  return manage(res);
}

isl::basic_map basic_map::apply_domain(isl::basic_map bmap2) const {
  auto res = isl_basic_map_apply_domain(copy(), bmap2.release());
  return manage(res);
}

isl::basic_map basic_map::apply_range(isl::basic_map bmap2) const {
  auto res = isl_basic_map_apply_range(copy(), bmap2.release());
  return manage(res);
}

isl::boolean basic_map::can_curry() const {
  auto res = isl_basic_map_can_curry(get());
  return manage(res);
}

isl::boolean basic_map::can_uncurry() const {
  auto res = isl_basic_map_can_uncurry(get());
  return manage(res);
}

isl::boolean basic_map::can_zip() const {
  auto res = isl_basic_map_can_zip(get());
  return manage(res);
}

isl::basic_map basic_map::curry() const {
  auto res = isl_basic_map_curry(copy());
  return manage(res);
}

isl::basic_set basic_map::deltas() const {
  auto res = isl_basic_map_deltas(copy());
  return manage(res);
}

isl::basic_map basic_map::deltas_map() const {
  auto res = isl_basic_map_deltas_map(copy());
  return manage(res);
}

isl::basic_map basic_map::detect_equalities() const {
  auto res = isl_basic_map_detect_equalities(copy());
  return manage(res);
}

unsigned int basic_map::dim(isl::dim type) const {
  auto res = isl_basic_map_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::basic_set basic_map::domain() const {
  auto res = isl_basic_map_domain(copy());
  return manage(res);
}

isl::basic_map basic_map::domain_map() const {
  auto res = isl_basic_map_domain_map(copy());
  return manage(res);
}

isl::basic_map basic_map::domain_product(isl::basic_map bmap2) const {
  auto res = isl_basic_map_domain_product(copy(), bmap2.release());
  return manage(res);
}

isl::basic_map basic_map::drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_basic_map_drop_constraints_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_map basic_map::drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_basic_map_drop_constraints_not_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_map basic_map::eliminate(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_basic_map_eliminate(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_map basic_map::empty(isl::space dim) {
  auto res = isl_basic_map_empty(dim.release());
  return manage(res);
}

isl::basic_map basic_map::equal(isl::space dim, unsigned int n_equal) {
  auto res = isl_basic_map_equal(dim.release(), n_equal);
  return manage(res);
}

isl::basic_map basic_map::equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const {
  auto res = isl_basic_map_equate(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

int basic_map::find_dim_by_name(isl::dim type, const std::string &name) const {
  auto res = isl_basic_map_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::basic_map basic_map::fix_si(isl::dim type, unsigned int pos, int value) const {
  auto res = isl_basic_map_fix_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::basic_map basic_map::fix_val(isl::dim type, unsigned int pos, isl::val v) const {
  auto res = isl_basic_map_fix_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

isl::basic_map basic_map::flat_product(isl::basic_map bmap2) const {
  auto res = isl_basic_map_flat_product(copy(), bmap2.release());
  return manage(res);
}

isl::basic_map basic_map::flat_range_product(isl::basic_map bmap2) const {
  auto res = isl_basic_map_flat_range_product(copy(), bmap2.release());
  return manage(res);
}

isl::basic_map basic_map::flatten() const {
  auto res = isl_basic_map_flatten(copy());
  return manage(res);
}

isl::basic_map basic_map::flatten_domain() const {
  auto res = isl_basic_map_flatten_domain(copy());
  return manage(res);
}

isl::basic_map basic_map::flatten_range() const {
  auto res = isl_basic_map_flatten_range(copy());
  return manage(res);
}

isl::stat basic_map::foreach_constraint(const std::function<isl::stat(isl::constraint)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_constraint *arg_0, void *arg_1) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::constraint)> **>(arg_1);
    stat ret = (*func)(isl::manage(arg_0));
    return isl_stat(ret);
  };
  auto res = isl_basic_map_foreach_constraint(get(), fn_lambda, &fn_p);
  return isl::stat(res);
}

isl::basic_map basic_map::from_aff(isl::aff aff) {
  auto res = isl_basic_map_from_aff(aff.release());
  return manage(res);
}

isl::basic_map basic_map::from_aff_list(isl::space domain_dim, isl::aff_list list) {
  auto res = isl_basic_map_from_aff_list(domain_dim.release(), list.release());
  return manage(res);
}

isl::basic_map basic_map::from_constraint(isl::constraint constraint) {
  auto res = isl_basic_map_from_constraint(constraint.release());
  return manage(res);
}

isl::basic_map basic_map::from_domain(isl::basic_set bset) {
  auto res = isl_basic_map_from_domain(bset.release());
  return manage(res);
}

isl::basic_map basic_map::from_domain_and_range(isl::basic_set domain, isl::basic_set range) {
  auto res = isl_basic_map_from_domain_and_range(domain.release(), range.release());
  return manage(res);
}

isl::basic_map basic_map::from_multi_aff(isl::multi_aff maff) {
  auto res = isl_basic_map_from_multi_aff(maff.release());
  return manage(res);
}

isl::basic_map basic_map::from_qpolynomial(isl::qpolynomial qp) {
  auto res = isl_basic_map_from_qpolynomial(qp.release());
  return manage(res);
}

isl::basic_map basic_map::from_range(isl::basic_set bset) {
  auto res = isl_basic_map_from_range(bset.release());
  return manage(res);
}

isl::constraint_list basic_map::get_constraint_list() const {
  auto res = isl_basic_map_get_constraint_list(get());
  return manage(res);
}

std::string basic_map::get_dim_name(isl::dim type, unsigned int pos) const {
  auto res = isl_basic_map_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

isl::aff basic_map::get_div(int pos) const {
  auto res = isl_basic_map_get_div(get(), pos);
  return manage(res);
}

isl::local_space basic_map::get_local_space() const {
  auto res = isl_basic_map_get_local_space(get());
  return manage(res);
}

isl::space basic_map::get_space() const {
  auto res = isl_basic_map_get_space(get());
  return manage(res);
}

std::string basic_map::get_tuple_name(isl::dim type) const {
  auto res = isl_basic_map_get_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  std::string tmp(res);
  return tmp;
}

isl::basic_map basic_map::gist(isl::basic_map context) const {
  auto res = isl_basic_map_gist(copy(), context.release());
  return manage(res);
}

isl::basic_map basic_map::gist_domain(isl::basic_set context) const {
  auto res = isl_basic_map_gist_domain(copy(), context.release());
  return manage(res);
}

isl::boolean basic_map::has_dim_id(isl::dim type, unsigned int pos) const {
  auto res = isl_basic_map_has_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::basic_map basic_map::identity(isl::space dim) {
  auto res = isl_basic_map_identity(dim.release());
  return manage(res);
}

isl::boolean basic_map::image_is_bounded() const {
  auto res = isl_basic_map_image_is_bounded(get());
  return manage(res);
}

isl::basic_map basic_map::insert_dims(isl::dim type, unsigned int pos, unsigned int n) const {
  auto res = isl_basic_map_insert_dims(copy(), static_cast<enum isl_dim_type>(type), pos, n);
  return manage(res);
}

isl::basic_map basic_map::intersect(isl::basic_map bmap2) const {
  auto res = isl_basic_map_intersect(copy(), bmap2.release());
  return manage(res);
}

isl::basic_map basic_map::intersect_domain(isl::basic_set bset) const {
  auto res = isl_basic_map_intersect_domain(copy(), bset.release());
  return manage(res);
}

isl::basic_map basic_map::intersect_range(isl::basic_set bset) const {
  auto res = isl_basic_map_intersect_range(copy(), bset.release());
  return manage(res);
}

isl::boolean basic_map::involves_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_basic_map_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::boolean basic_map::is_disjoint(const isl::basic_map &bmap2) const {
  auto res = isl_basic_map_is_disjoint(get(), bmap2.get());
  return manage(res);
}

isl::boolean basic_map::is_empty() const {
  auto res = isl_basic_map_is_empty(get());
  return manage(res);
}

isl::boolean basic_map::is_equal(const isl::basic_map &bmap2) const {
  auto res = isl_basic_map_is_equal(get(), bmap2.get());
  return manage(res);
}

isl::boolean basic_map::is_rational() const {
  auto res = isl_basic_map_is_rational(get());
  return manage(res);
}

isl::boolean basic_map::is_single_valued() const {
  auto res = isl_basic_map_is_single_valued(get());
  return manage(res);
}

isl::boolean basic_map::is_strict_subset(const isl::basic_map &bmap2) const {
  auto res = isl_basic_map_is_strict_subset(get(), bmap2.get());
  return manage(res);
}

isl::boolean basic_map::is_subset(const isl::basic_map &bmap2) const {
  auto res = isl_basic_map_is_subset(get(), bmap2.get());
  return manage(res);
}

isl::boolean basic_map::is_universe() const {
  auto res = isl_basic_map_is_universe(get());
  return manage(res);
}

isl::basic_map basic_map::less_at(isl::space dim, unsigned int pos) {
  auto res = isl_basic_map_less_at(dim.release(), pos);
  return manage(res);
}

isl::map basic_map::lexmax() const {
  auto res = isl_basic_map_lexmax(copy());
  return manage(res);
}

isl::map basic_map::lexmin() const {
  auto res = isl_basic_map_lexmin(copy());
  return manage(res);
}

isl::pw_multi_aff basic_map::lexmin_pw_multi_aff() const {
  auto res = isl_basic_map_lexmin_pw_multi_aff(copy());
  return manage(res);
}

isl::basic_map basic_map::lower_bound_si(isl::dim type, unsigned int pos, int value) const {
  auto res = isl_basic_map_lower_bound_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::basic_map basic_map::more_at(isl::space dim, unsigned int pos) {
  auto res = isl_basic_map_more_at(dim.release(), pos);
  return manage(res);
}

isl::basic_map basic_map::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const {
  auto res = isl_basic_map_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl::basic_map basic_map::nat_universe(isl::space dim) {
  auto res = isl_basic_map_nat_universe(dim.release());
  return manage(res);
}

isl::basic_map basic_map::neg() const {
  auto res = isl_basic_map_neg(copy());
  return manage(res);
}

isl::basic_map basic_map::order_ge(isl::dim type1, int pos1, isl::dim type2, int pos2) const {
  auto res = isl_basic_map_order_ge(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

isl::basic_map basic_map::order_gt(isl::dim type1, int pos1, isl::dim type2, int pos2) const {
  auto res = isl_basic_map_order_gt(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

isl::val basic_map::plain_get_val_if_fixed(isl::dim type, unsigned int pos) const {
  auto res = isl_basic_map_plain_get_val_if_fixed(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::boolean basic_map::plain_is_empty() const {
  auto res = isl_basic_map_plain_is_empty(get());
  return manage(res);
}

isl::boolean basic_map::plain_is_universe() const {
  auto res = isl_basic_map_plain_is_universe(get());
  return manage(res);
}

isl::basic_map basic_map::preimage_domain_multi_aff(isl::multi_aff ma) const {
  auto res = isl_basic_map_preimage_domain_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::basic_map basic_map::preimage_range_multi_aff(isl::multi_aff ma) const {
  auto res = isl_basic_map_preimage_range_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::basic_map basic_map::product(isl::basic_map bmap2) const {
  auto res = isl_basic_map_product(copy(), bmap2.release());
  return manage(res);
}

isl::basic_map basic_map::project_out(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_basic_map_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_set basic_map::range() const {
  auto res = isl_basic_map_range(copy());
  return manage(res);
}

isl::basic_map basic_map::range_map() const {
  auto res = isl_basic_map_range_map(copy());
  return manage(res);
}

isl::basic_map basic_map::range_product(isl::basic_map bmap2) const {
  auto res = isl_basic_map_range_product(copy(), bmap2.release());
  return manage(res);
}

isl::basic_map basic_map::remove_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_basic_map_remove_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_map basic_map::remove_divs() const {
  auto res = isl_basic_map_remove_divs(copy());
  return manage(res);
}

isl::basic_map basic_map::remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_basic_map_remove_divs_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_map basic_map::remove_redundancies() const {
  auto res = isl_basic_map_remove_redundancies(copy());
  return manage(res);
}

isl::basic_map basic_map::reverse() const {
  auto res = isl_basic_map_reverse(copy());
  return manage(res);
}

isl::basic_map basic_map::sample() const {
  auto res = isl_basic_map_sample(copy());
  return manage(res);
}

isl::basic_map basic_map::set_tuple_id(isl::dim type, isl::id id) const {
  auto res = isl_basic_map_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::basic_map basic_map::set_tuple_name(isl::dim type, const std::string &s) const {
  auto res = isl_basic_map_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

isl::basic_map basic_map::sum(isl::basic_map bmap2) const {
  auto res = isl_basic_map_sum(copy(), bmap2.release());
  return manage(res);
}

isl::basic_map basic_map::uncurry() const {
  auto res = isl_basic_map_uncurry(copy());
  return manage(res);
}

isl::map basic_map::unite(isl::basic_map bmap2) const {
  auto res = isl_basic_map_union(copy(), bmap2.release());
  return manage(res);
}

isl::basic_map basic_map::universe(isl::space dim) {
  auto res = isl_basic_map_universe(dim.release());
  return manage(res);
}

isl::basic_map basic_map::upper_bound_si(isl::dim type, unsigned int pos, int value) const {
  auto res = isl_basic_map_upper_bound_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::basic_set basic_map::wrap() const {
  auto res = isl_basic_map_wrap(copy());
  return manage(res);
}

isl::basic_map basic_map::zip() const {
  auto res = isl_basic_map_zip(copy());
  return manage(res);
}

// implementations for isl::basic_map_list
isl::basic_map_list manage(__isl_take isl_basic_map_list *ptr) {
  return basic_map_list(ptr);
}
isl::basic_map_list give(__isl_take isl_basic_map_list *ptr) {
  return manage(ptr);
}


basic_map_list::basic_map_list()
    : ptr(nullptr) {}

basic_map_list::basic_map_list(const isl::basic_map_list &obj)
    : ptr(obj.copy()) {}
basic_map_list::basic_map_list(std::nullptr_t)
    : ptr(nullptr) {}


basic_map_list::basic_map_list(__isl_take isl_basic_map_list *ptr)
    : ptr(ptr) {}


basic_map_list &basic_map_list::operator=(isl::basic_map_list obj) {
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
__isl_keep isl_basic_map_list *basic_map_list::keep() const {
  return get();
}

__isl_give isl_basic_map_list *basic_map_list::take() {
  return release();
}

basic_map_list::operator bool() const {
  return !is_null();
}

isl::ctx basic_map_list::get_ctx() const {
  return isl::ctx(isl_basic_map_list_get_ctx(ptr));
}



void basic_map_list::dump() const {
  isl_basic_map_list_dump(get());
}



// implementations for isl::basic_set
isl::basic_set manage(__isl_take isl_basic_set *ptr) {
  return basic_set(ptr);
}
isl::basic_set give(__isl_take isl_basic_set *ptr) {
  return manage(ptr);
}


basic_set::basic_set()
    : ptr(nullptr) {}

basic_set::basic_set(const isl::basic_set &obj)
    : ptr(obj.copy()) {}
basic_set::basic_set(std::nullptr_t)
    : ptr(nullptr) {}


basic_set::basic_set(__isl_take isl_basic_set *ptr)
    : ptr(ptr) {}

basic_set::basic_set(isl::ctx ctx, const std::string &str) {
  auto res = isl_basic_set_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}
basic_set::basic_set(isl::point pnt) {
  auto res = isl_basic_set_from_point(pnt.release());
  ptr = res;
}

basic_set &basic_set::operator=(isl::basic_set obj) {
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
__isl_keep isl_basic_set *basic_set::keep() const {
  return get();
}

__isl_give isl_basic_set *basic_set::take() {
  return release();
}

basic_set::operator bool() const {
  return !is_null();
}

isl::ctx basic_set::get_ctx() const {
  return isl::ctx(isl_basic_set_get_ctx(ptr));
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


isl::basic_set basic_set::affine_hull() const {
  auto res = isl_basic_set_affine_hull(copy());
  return manage(res);
}

isl::basic_set basic_set::align_params(isl::space model) const {
  auto res = isl_basic_set_align_params(copy(), model.release());
  return manage(res);
}

isl::basic_set basic_set::apply(isl::basic_map bmap) const {
  auto res = isl_basic_set_apply(copy(), bmap.release());
  return manage(res);
}

isl::basic_set basic_set::box_from_points(isl::point pnt1, isl::point pnt2) {
  auto res = isl_basic_set_box_from_points(pnt1.release(), pnt2.release());
  return manage(res);
}

isl::basic_set basic_set::coefficients() const {
  auto res = isl_basic_set_coefficients(copy());
  return manage(res);
}

isl::basic_set basic_set::detect_equalities() const {
  auto res = isl_basic_set_detect_equalities(copy());
  return manage(res);
}

unsigned int basic_set::dim(isl::dim type) const {
  auto res = isl_basic_set_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::basic_set basic_set::drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_basic_set_drop_constraints_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_set basic_set::drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_basic_set_drop_constraints_not_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_set basic_set::eliminate(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_basic_set_eliminate(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_set basic_set::empty(isl::space dim) {
  auto res = isl_basic_set_empty(dim.release());
  return manage(res);
}

isl::basic_set basic_set::fix_si(isl::dim type, unsigned int pos, int value) const {
  auto res = isl_basic_set_fix_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::basic_set basic_set::fix_val(isl::dim type, unsigned int pos, isl::val v) const {
  auto res = isl_basic_set_fix_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

isl::basic_set basic_set::flat_product(isl::basic_set bset2) const {
  auto res = isl_basic_set_flat_product(copy(), bset2.release());
  return manage(res);
}

isl::basic_set basic_set::flatten() const {
  auto res = isl_basic_set_flatten(copy());
  return manage(res);
}

isl::stat basic_set::foreach_bound_pair(isl::dim type, unsigned int pos, const std::function<isl::stat(isl::constraint, isl::constraint, isl::basic_set)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_constraint *arg_0, isl_constraint *arg_1, isl_basic_set *arg_2, void *arg_3) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::constraint, isl::constraint, isl::basic_set)> **>(arg_3);
    stat ret = (*func)(isl::manage(arg_0), isl::manage(arg_1), isl::manage(arg_2));
    return isl_stat(ret);
  };
  auto res = isl_basic_set_foreach_bound_pair(get(), static_cast<enum isl_dim_type>(type), pos, fn_lambda, &fn_p);
  return isl::stat(res);
}

isl::stat basic_set::foreach_constraint(const std::function<isl::stat(isl::constraint)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_constraint *arg_0, void *arg_1) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::constraint)> **>(arg_1);
    stat ret = (*func)(isl::manage(arg_0));
    return isl_stat(ret);
  };
  auto res = isl_basic_set_foreach_constraint(get(), fn_lambda, &fn_p);
  return isl::stat(res);
}

isl::basic_set basic_set::from_constraint(isl::constraint constraint) {
  auto res = isl_basic_set_from_constraint(constraint.release());
  return manage(res);
}

isl::basic_set basic_set::from_params() const {
  auto res = isl_basic_set_from_params(copy());
  return manage(res);
}

isl::constraint_list basic_set::get_constraint_list() const {
  auto res = isl_basic_set_get_constraint_list(get());
  return manage(res);
}

isl::id basic_set::get_dim_id(isl::dim type, unsigned int pos) const {
  auto res = isl_basic_set_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

std::string basic_set::get_dim_name(isl::dim type, unsigned int pos) const {
  auto res = isl_basic_set_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

isl::aff basic_set::get_div(int pos) const {
  auto res = isl_basic_set_get_div(get(), pos);
  return manage(res);
}

isl::local_space basic_set::get_local_space() const {
  auto res = isl_basic_set_get_local_space(get());
  return manage(res);
}

isl::space basic_set::get_space() const {
  auto res = isl_basic_set_get_space(get());
  return manage(res);
}

std::string basic_set::get_tuple_name() const {
  auto res = isl_basic_set_get_tuple_name(get());
  std::string tmp(res);
  return tmp;
}

isl::basic_set basic_set::gist(isl::basic_set context) const {
  auto res = isl_basic_set_gist(copy(), context.release());
  return manage(res);
}

isl::basic_set basic_set::insert_dims(isl::dim type, unsigned int pos, unsigned int n) const {
  auto res = isl_basic_set_insert_dims(copy(), static_cast<enum isl_dim_type>(type), pos, n);
  return manage(res);
}

isl::basic_set basic_set::intersect(isl::basic_set bset2) const {
  auto res = isl_basic_set_intersect(copy(), bset2.release());
  return manage(res);
}

isl::basic_set basic_set::intersect_params(isl::basic_set bset2) const {
  auto res = isl_basic_set_intersect_params(copy(), bset2.release());
  return manage(res);
}

isl::boolean basic_set::involves_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_basic_set_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::boolean basic_set::is_bounded() const {
  auto res = isl_basic_set_is_bounded(get());
  return manage(res);
}

isl::boolean basic_set::is_disjoint(const isl::basic_set &bset2) const {
  auto res = isl_basic_set_is_disjoint(get(), bset2.get());
  return manage(res);
}

isl::boolean basic_set::is_empty() const {
  auto res = isl_basic_set_is_empty(get());
  return manage(res);
}

isl::boolean basic_set::is_equal(const isl::basic_set &bset2) const {
  auto res = isl_basic_set_is_equal(get(), bset2.get());
  return manage(res);
}

int basic_set::is_rational() const {
  auto res = isl_basic_set_is_rational(get());
  return res;
}

isl::boolean basic_set::is_subset(const isl::basic_set &bset2) const {
  auto res = isl_basic_set_is_subset(get(), bset2.get());
  return manage(res);
}

isl::boolean basic_set::is_universe() const {
  auto res = isl_basic_set_is_universe(get());
  return manage(res);
}

isl::boolean basic_set::is_wrapping() const {
  auto res = isl_basic_set_is_wrapping(get());
  return manage(res);
}

isl::set basic_set::lexmax() const {
  auto res = isl_basic_set_lexmax(copy());
  return manage(res);
}

isl::set basic_set::lexmin() const {
  auto res = isl_basic_set_lexmin(copy());
  return manage(res);
}

isl::basic_set basic_set::lower_bound_val(isl::dim type, unsigned int pos, isl::val value) const {
  auto res = isl_basic_set_lower_bound_val(copy(), static_cast<enum isl_dim_type>(type), pos, value.release());
  return manage(res);
}

isl::val basic_set::max_val(const isl::aff &obj) const {
  auto res = isl_basic_set_max_val(get(), obj.get());
  return manage(res);
}

isl::basic_set basic_set::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const {
  auto res = isl_basic_set_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl::basic_set basic_set::nat_universe(isl::space dim) {
  auto res = isl_basic_set_nat_universe(dim.release());
  return manage(res);
}

isl::basic_set basic_set::neg() const {
  auto res = isl_basic_set_neg(copy());
  return manage(res);
}

isl::basic_set basic_set::params() const {
  auto res = isl_basic_set_params(copy());
  return manage(res);
}

isl::boolean basic_set::plain_is_empty() const {
  auto res = isl_basic_set_plain_is_empty(get());
  return manage(res);
}

isl::boolean basic_set::plain_is_equal(const isl::basic_set &bset2) const {
  auto res = isl_basic_set_plain_is_equal(get(), bset2.get());
  return manage(res);
}

isl::boolean basic_set::plain_is_universe() const {
  auto res = isl_basic_set_plain_is_universe(get());
  return manage(res);
}

isl::basic_set basic_set::positive_orthant(isl::space space) {
  auto res = isl_basic_set_positive_orthant(space.release());
  return manage(res);
}

isl::basic_set basic_set::preimage_multi_aff(isl::multi_aff ma) const {
  auto res = isl_basic_set_preimage_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::basic_set basic_set::project_out(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_basic_set_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_set basic_set::remove_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_basic_set_remove_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_set basic_set::remove_divs() const {
  auto res = isl_basic_set_remove_divs(copy());
  return manage(res);
}

isl::basic_set basic_set::remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_basic_set_remove_divs_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::basic_set basic_set::remove_redundancies() const {
  auto res = isl_basic_set_remove_redundancies(copy());
  return manage(res);
}

isl::basic_set basic_set::remove_unknown_divs() const {
  auto res = isl_basic_set_remove_unknown_divs(copy());
  return manage(res);
}

isl::basic_set basic_set::sample() const {
  auto res = isl_basic_set_sample(copy());
  return manage(res);
}

isl::point basic_set::sample_point() const {
  auto res = isl_basic_set_sample_point(copy());
  return manage(res);
}

isl::basic_set basic_set::set_tuple_id(isl::id id) const {
  auto res = isl_basic_set_set_tuple_id(copy(), id.release());
  return manage(res);
}

isl::basic_set basic_set::set_tuple_name(const std::string &s) const {
  auto res = isl_basic_set_set_tuple_name(copy(), s.c_str());
  return manage(res);
}

isl::basic_set basic_set::solutions() const {
  auto res = isl_basic_set_solutions(copy());
  return manage(res);
}

isl::set basic_set::unite(isl::basic_set bset2) const {
  auto res = isl_basic_set_union(copy(), bset2.release());
  return manage(res);
}

isl::basic_set basic_set::universe(isl::space dim) {
  auto res = isl_basic_set_universe(dim.release());
  return manage(res);
}

isl::basic_map basic_set::unwrap() const {
  auto res = isl_basic_set_unwrap(copy());
  return manage(res);
}

isl::basic_set basic_set::upper_bound_val(isl::dim type, unsigned int pos, isl::val value) const {
  auto res = isl_basic_set_upper_bound_val(copy(), static_cast<enum isl_dim_type>(type), pos, value.release());
  return manage(res);
}

// implementations for isl::basic_set_list
isl::basic_set_list manage(__isl_take isl_basic_set_list *ptr) {
  return basic_set_list(ptr);
}
isl::basic_set_list give(__isl_take isl_basic_set_list *ptr) {
  return manage(ptr);
}


basic_set_list::basic_set_list()
    : ptr(nullptr) {}

basic_set_list::basic_set_list(const isl::basic_set_list &obj)
    : ptr(obj.copy()) {}
basic_set_list::basic_set_list(std::nullptr_t)
    : ptr(nullptr) {}


basic_set_list::basic_set_list(__isl_take isl_basic_set_list *ptr)
    : ptr(ptr) {}


basic_set_list &basic_set_list::operator=(isl::basic_set_list obj) {
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
__isl_keep isl_basic_set_list *basic_set_list::keep() const {
  return get();
}

__isl_give isl_basic_set_list *basic_set_list::take() {
  return release();
}

basic_set_list::operator bool() const {
  return !is_null();
}

isl::ctx basic_set_list::get_ctx() const {
  return isl::ctx(isl_basic_set_list_get_ctx(ptr));
}



void basic_set_list::dump() const {
  isl_basic_set_list_dump(get());
}



// implementations for isl::constraint
isl::constraint manage(__isl_take isl_constraint *ptr) {
  return constraint(ptr);
}
isl::constraint give(__isl_take isl_constraint *ptr) {
  return manage(ptr);
}


constraint::constraint()
    : ptr(nullptr) {}

constraint::constraint(const isl::constraint &obj)
    : ptr(obj.copy()) {}
constraint::constraint(std::nullptr_t)
    : ptr(nullptr) {}


constraint::constraint(__isl_take isl_constraint *ptr)
    : ptr(ptr) {}


constraint &constraint::operator=(isl::constraint obj) {
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
__isl_keep isl_constraint *constraint::keep() const {
  return get();
}

__isl_give isl_constraint *constraint::take() {
  return release();
}

constraint::operator bool() const {
  return !is_null();
}

isl::ctx constraint::get_ctx() const {
  return isl::ctx(isl_constraint_get_ctx(ptr));
}



void constraint::dump() const {
  isl_constraint_dump(get());
}


isl::constraint constraint::alloc_equality(isl::local_space ls) {
  auto res = isl_constraint_alloc_equality(ls.release());
  return manage(res);
}

isl::constraint constraint::alloc_inequality(isl::local_space ls) {
  auto res = isl_constraint_alloc_inequality(ls.release());
  return manage(res);
}

int constraint::cmp_last_non_zero(const isl::constraint &c2) const {
  auto res = isl_constraint_cmp_last_non_zero(get(), c2.get());
  return res;
}

isl::aff constraint::get_aff() const {
  auto res = isl_constraint_get_aff(get());
  return manage(res);
}

isl::aff constraint::get_bound(isl::dim type, int pos) const {
  auto res = isl_constraint_get_bound(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::val constraint::get_coefficient_val(isl::dim type, int pos) const {
  auto res = isl_constraint_get_coefficient_val(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::val constraint::get_constant_val() const {
  auto res = isl_constraint_get_constant_val(get());
  return manage(res);
}

std::string constraint::get_dim_name(isl::dim type, unsigned int pos) const {
  auto res = isl_constraint_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

isl::aff constraint::get_div(int pos) const {
  auto res = isl_constraint_get_div(get(), pos);
  return manage(res);
}

isl::local_space constraint::get_local_space() const {
  auto res = isl_constraint_get_local_space(get());
  return manage(res);
}

isl::space constraint::get_space() const {
  auto res = isl_constraint_get_space(get());
  return manage(res);
}

isl::boolean constraint::involves_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_constraint_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

int constraint::is_div_constraint() const {
  auto res = isl_constraint_is_div_constraint(get());
  return res;
}

isl::boolean constraint::is_lower_bound(isl::dim type, unsigned int pos) const {
  auto res = isl_constraint_is_lower_bound(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::boolean constraint::is_upper_bound(isl::dim type, unsigned int pos) const {
  auto res = isl_constraint_is_upper_bound(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

int constraint::plain_cmp(const isl::constraint &c2) const {
  auto res = isl_constraint_plain_cmp(get(), c2.get());
  return res;
}

isl::constraint constraint::set_coefficient_si(isl::dim type, int pos, int v) const {
  auto res = isl_constraint_set_coefficient_si(copy(), static_cast<enum isl_dim_type>(type), pos, v);
  return manage(res);
}

isl::constraint constraint::set_coefficient_val(isl::dim type, int pos, isl::val v) const {
  auto res = isl_constraint_set_coefficient_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

isl::constraint constraint::set_constant_si(int v) const {
  auto res = isl_constraint_set_constant_si(copy(), v);
  return manage(res);
}

isl::constraint constraint::set_constant_val(isl::val v) const {
  auto res = isl_constraint_set_constant_val(copy(), v.release());
  return manage(res);
}

// implementations for isl::constraint_list
isl::constraint_list manage(__isl_take isl_constraint_list *ptr) {
  return constraint_list(ptr);
}
isl::constraint_list give(__isl_take isl_constraint_list *ptr) {
  return manage(ptr);
}


constraint_list::constraint_list()
    : ptr(nullptr) {}

constraint_list::constraint_list(const isl::constraint_list &obj)
    : ptr(obj.copy()) {}
constraint_list::constraint_list(std::nullptr_t)
    : ptr(nullptr) {}


constraint_list::constraint_list(__isl_take isl_constraint_list *ptr)
    : ptr(ptr) {}


constraint_list &constraint_list::operator=(isl::constraint_list obj) {
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
__isl_keep isl_constraint_list *constraint_list::keep() const {
  return get();
}

__isl_give isl_constraint_list *constraint_list::take() {
  return release();
}

constraint_list::operator bool() const {
  return !is_null();
}

isl::ctx constraint_list::get_ctx() const {
  return isl::ctx(isl_constraint_list_get_ctx(ptr));
}



void constraint_list::dump() const {
  isl_constraint_list_dump(get());
}



// implementations for isl::id
isl::id manage(__isl_take isl_id *ptr) {
  return id(ptr);
}
isl::id give(__isl_take isl_id *ptr) {
  return manage(ptr);
}


id::id()
    : ptr(nullptr) {}

id::id(const isl::id &obj)
    : ptr(obj.copy()) {}
id::id(std::nullptr_t)
    : ptr(nullptr) {}


id::id(__isl_take isl_id *ptr)
    : ptr(ptr) {}


id &id::operator=(isl::id obj) {
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
__isl_keep isl_id *id::keep() const {
  return get();
}

__isl_give isl_id *id::take() {
  return release();
}

id::operator bool() const {
  return !is_null();
}

isl::ctx id::get_ctx() const {
  return isl::ctx(isl_id_get_ctx(ptr));
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


isl::id id::alloc(isl::ctx ctx, const std::string &name, void * user) {
  auto res = isl_id_alloc(ctx.release(), name.c_str(), user);
  return manage(res);
}

uint32_t id::get_hash() const {
  auto res = isl_id_get_hash(get());
  return res;
}

std::string id::get_name() const {
  auto res = isl_id_get_name(get());
  std::string tmp(res);
  return tmp;
}

void * id::get_user() const {
  auto res = isl_id_get_user(get());
  return res;
}

// implementations for isl::id_list
isl::id_list manage(__isl_take isl_id_list *ptr) {
  return id_list(ptr);
}
isl::id_list give(__isl_take isl_id_list *ptr) {
  return manage(ptr);
}


id_list::id_list()
    : ptr(nullptr) {}

id_list::id_list(const isl::id_list &obj)
    : ptr(obj.copy()) {}
id_list::id_list(std::nullptr_t)
    : ptr(nullptr) {}


id_list::id_list(__isl_take isl_id_list *ptr)
    : ptr(ptr) {}


id_list &id_list::operator=(isl::id_list obj) {
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
__isl_keep isl_id_list *id_list::keep() const {
  return get();
}

__isl_give isl_id_list *id_list::take() {
  return release();
}

id_list::operator bool() const {
  return !is_null();
}

isl::ctx id_list::get_ctx() const {
  return isl::ctx(isl_id_list_get_ctx(ptr));
}



void id_list::dump() const {
  isl_id_list_dump(get());
}



// implementations for isl::id_to_ast_expr
isl::id_to_ast_expr manage(__isl_take isl_id_to_ast_expr *ptr) {
  return id_to_ast_expr(ptr);
}
isl::id_to_ast_expr give(__isl_take isl_id_to_ast_expr *ptr) {
  return manage(ptr);
}


id_to_ast_expr::id_to_ast_expr()
    : ptr(nullptr) {}

id_to_ast_expr::id_to_ast_expr(const isl::id_to_ast_expr &obj)
    : ptr(obj.copy()) {}
id_to_ast_expr::id_to_ast_expr(std::nullptr_t)
    : ptr(nullptr) {}


id_to_ast_expr::id_to_ast_expr(__isl_take isl_id_to_ast_expr *ptr)
    : ptr(ptr) {}


id_to_ast_expr &id_to_ast_expr::operator=(isl::id_to_ast_expr obj) {
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
__isl_keep isl_id_to_ast_expr *id_to_ast_expr::keep() const {
  return get();
}

__isl_give isl_id_to_ast_expr *id_to_ast_expr::take() {
  return release();
}

id_to_ast_expr::operator bool() const {
  return !is_null();
}

isl::ctx id_to_ast_expr::get_ctx() const {
  return isl::ctx(isl_id_to_ast_expr_get_ctx(ptr));
}



void id_to_ast_expr::dump() const {
  isl_id_to_ast_expr_dump(get());
}


isl::id_to_ast_expr id_to_ast_expr::alloc(isl::ctx ctx, int min_size) {
  auto res = isl_id_to_ast_expr_alloc(ctx.release(), min_size);
  return manage(res);
}

isl::id_to_ast_expr id_to_ast_expr::drop(isl::id key) const {
  auto res = isl_id_to_ast_expr_drop(copy(), key.release());
  return manage(res);
}

isl::stat id_to_ast_expr::foreach(const std::function<isl::stat(isl::id, isl::ast_expr)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_id *arg_0, isl_ast_expr *arg_1, void *arg_2) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::id, isl::ast_expr)> **>(arg_2);
    stat ret = (*func)(isl::manage(arg_0), isl::manage(arg_1));
    return isl_stat(ret);
  };
  auto res = isl_id_to_ast_expr_foreach(get(), fn_lambda, &fn_p);
  return isl::stat(res);
}

isl::ast_expr id_to_ast_expr::get(isl::id key) const {
  auto res = isl_id_to_ast_expr_get(get(), key.release());
  return manage(res);
}

isl::boolean id_to_ast_expr::has(const isl::id &key) const {
  auto res = isl_id_to_ast_expr_has(get(), key.get());
  return manage(res);
}

isl::id_to_ast_expr id_to_ast_expr::set(isl::id key, isl::ast_expr val) const {
  auto res = isl_id_to_ast_expr_set(copy(), key.release(), val.release());
  return manage(res);
}

// implementations for isl::local_space
isl::local_space manage(__isl_take isl_local_space *ptr) {
  return local_space(ptr);
}
isl::local_space give(__isl_take isl_local_space *ptr) {
  return manage(ptr);
}


local_space::local_space()
    : ptr(nullptr) {}

local_space::local_space(const isl::local_space &obj)
    : ptr(obj.copy()) {}
local_space::local_space(std::nullptr_t)
    : ptr(nullptr) {}


local_space::local_space(__isl_take isl_local_space *ptr)
    : ptr(ptr) {}

local_space::local_space(isl::space dim) {
  auto res = isl_local_space_from_space(dim.release());
  ptr = res;
}

local_space &local_space::operator=(isl::local_space obj) {
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
__isl_keep isl_local_space *local_space::keep() const {
  return get();
}

__isl_give isl_local_space *local_space::take() {
  return release();
}

local_space::operator bool() const {
  return !is_null();
}

isl::ctx local_space::get_ctx() const {
  return isl::ctx(isl_local_space_get_ctx(ptr));
}



void local_space::dump() const {
  isl_local_space_dump(get());
}


isl::local_space local_space::add_dims(isl::dim type, unsigned int n) const {
  auto res = isl_local_space_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

int local_space::dim(isl::dim type) const {
  auto res = isl_local_space_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::local_space local_space::domain() const {
  auto res = isl_local_space_domain(copy());
  return manage(res);
}

isl::local_space local_space::drop_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_local_space_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

int local_space::find_dim_by_name(isl::dim type, const std::string &name) const {
  auto res = isl_local_space_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::local_space local_space::flatten_domain() const {
  auto res = isl_local_space_flatten_domain(copy());
  return manage(res);
}

isl::local_space local_space::flatten_range() const {
  auto res = isl_local_space_flatten_range(copy());
  return manage(res);
}

isl::local_space local_space::from_domain() const {
  auto res = isl_local_space_from_domain(copy());
  return manage(res);
}

isl::id local_space::get_dim_id(isl::dim type, unsigned int pos) const {
  auto res = isl_local_space_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

std::string local_space::get_dim_name(isl::dim type, unsigned int pos) const {
  auto res = isl_local_space_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

isl::aff local_space::get_div(int pos) const {
  auto res = isl_local_space_get_div(get(), pos);
  return manage(res);
}

isl::space local_space::get_space() const {
  auto res = isl_local_space_get_space(get());
  return manage(res);
}

isl::boolean local_space::has_dim_id(isl::dim type, unsigned int pos) const {
  auto res = isl_local_space_has_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::boolean local_space::has_dim_name(isl::dim type, unsigned int pos) const {
  auto res = isl_local_space_has_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::local_space local_space::insert_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_local_space_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::local_space local_space::intersect(isl::local_space ls2) const {
  auto res = isl_local_space_intersect(copy(), ls2.release());
  return manage(res);
}

isl::boolean local_space::is_equal(const isl::local_space &ls2) const {
  auto res = isl_local_space_is_equal(get(), ls2.get());
  return manage(res);
}

isl::boolean local_space::is_params() const {
  auto res = isl_local_space_is_params(get());
  return manage(res);
}

isl::boolean local_space::is_set() const {
  auto res = isl_local_space_is_set(get());
  return manage(res);
}

isl::local_space local_space::range() const {
  auto res = isl_local_space_range(copy());
  return manage(res);
}

isl::local_space local_space::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const {
  auto res = isl_local_space_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::local_space local_space::set_tuple_id(isl::dim type, isl::id id) const {
  auto res = isl_local_space_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::local_space local_space::wrap() const {
  auto res = isl_local_space_wrap(copy());
  return manage(res);
}

// implementations for isl::map
isl::map manage(__isl_take isl_map *ptr) {
  return map(ptr);
}
isl::map give(__isl_take isl_map *ptr) {
  return manage(ptr);
}


map::map()
    : ptr(nullptr) {}

map::map(const isl::map &obj)
    : ptr(obj.copy()) {}
map::map(std::nullptr_t)
    : ptr(nullptr) {}


map::map(__isl_take isl_map *ptr)
    : ptr(ptr) {}

map::map(isl::ctx ctx, const std::string &str) {
  auto res = isl_map_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}
map::map(isl::basic_map bmap) {
  auto res = isl_map_from_basic_map(bmap.release());
  ptr = res;
}

map &map::operator=(isl::map obj) {
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
__isl_keep isl_map *map::keep() const {
  return get();
}

__isl_give isl_map *map::take() {
  return release();
}

map::operator bool() const {
  return !is_null();
}

isl::ctx map::get_ctx() const {
  return isl::ctx(isl_map_get_ctx(ptr));
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


isl::map map::add_constraint(isl::constraint constraint) const {
  auto res = isl_map_add_constraint(copy(), constraint.release());
  return manage(res);
}

isl::map map::add_dims(isl::dim type, unsigned int n) const {
  auto res = isl_map_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::basic_map map::affine_hull() const {
  auto res = isl_map_affine_hull(copy());
  return manage(res);
}

isl::map map::align_params(isl::space model) const {
  auto res = isl_map_align_params(copy(), model.release());
  return manage(res);
}

isl::map map::apply_domain(isl::map map2) const {
  auto res = isl_map_apply_domain(copy(), map2.release());
  return manage(res);
}

isl::map map::apply_range(isl::map map2) const {
  auto res = isl_map_apply_range(copy(), map2.release());
  return manage(res);
}

isl::boolean map::can_curry() const {
  auto res = isl_map_can_curry(get());
  return manage(res);
}

isl::boolean map::can_range_curry() const {
  auto res = isl_map_can_range_curry(get());
  return manage(res);
}

isl::boolean map::can_uncurry() const {
  auto res = isl_map_can_uncurry(get());
  return manage(res);
}

isl::boolean map::can_zip() const {
  auto res = isl_map_can_zip(get());
  return manage(res);
}

isl::map map::coalesce() const {
  auto res = isl_map_coalesce(copy());
  return manage(res);
}

isl::map map::complement() const {
  auto res = isl_map_complement(copy());
  return manage(res);
}

isl::basic_map map::convex_hull() const {
  auto res = isl_map_convex_hull(copy());
  return manage(res);
}

isl::map map::curry() const {
  auto res = isl_map_curry(copy());
  return manage(res);
}

isl::set map::deltas() const {
  auto res = isl_map_deltas(copy());
  return manage(res);
}

isl::map map::deltas_map() const {
  auto res = isl_map_deltas_map(copy());
  return manage(res);
}

isl::map map::detect_equalities() const {
  auto res = isl_map_detect_equalities(copy());
  return manage(res);
}

unsigned int map::dim(isl::dim type) const {
  auto res = isl_map_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::pw_aff map::dim_max(int pos) const {
  auto res = isl_map_dim_max(copy(), pos);
  return manage(res);
}

isl::pw_aff map::dim_min(int pos) const {
  auto res = isl_map_dim_min(copy(), pos);
  return manage(res);
}

isl::set map::domain() const {
  auto res = isl_map_domain(copy());
  return manage(res);
}

isl::map map::domain_factor_domain() const {
  auto res = isl_map_domain_factor_domain(copy());
  return manage(res);
}

isl::map map::domain_factor_range() const {
  auto res = isl_map_domain_factor_range(copy());
  return manage(res);
}

isl::boolean map::domain_is_wrapping() const {
  auto res = isl_map_domain_is_wrapping(get());
  return manage(res);
}

isl::map map::domain_map() const {
  auto res = isl_map_domain_map(copy());
  return manage(res);
}

isl::map map::domain_product(isl::map map2) const {
  auto res = isl_map_domain_product(copy(), map2.release());
  return manage(res);
}

isl::map map::drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_map_drop_constraints_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::map map::drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_map_drop_constraints_not_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::map map::eliminate(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_map_eliminate(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::map map::empty(isl::space dim) {
  auto res = isl_map_empty(dim.release());
  return manage(res);
}

isl::map map::equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const {
  auto res = isl_map_equate(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

isl::map map::factor_domain() const {
  auto res = isl_map_factor_domain(copy());
  return manage(res);
}

isl::map map::factor_range() const {
  auto res = isl_map_factor_range(copy());
  return manage(res);
}

int map::find_dim_by_id(isl::dim type, const isl::id &id) const {
  auto res = isl_map_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int map::find_dim_by_name(isl::dim type, const std::string &name) const {
  auto res = isl_map_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::map map::fix_si(isl::dim type, unsigned int pos, int value) const {
  auto res = isl_map_fix_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::map map::fix_val(isl::dim type, unsigned int pos, isl::val v) const {
  auto res = isl_map_fix_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

isl::map map::fixed_power_val(isl::val exp) const {
  auto res = isl_map_fixed_power_val(copy(), exp.release());
  return manage(res);
}

isl::map map::flat_domain_product(isl::map map2) const {
  auto res = isl_map_flat_domain_product(copy(), map2.release());
  return manage(res);
}

isl::map map::flat_product(isl::map map2) const {
  auto res = isl_map_flat_product(copy(), map2.release());
  return manage(res);
}

isl::map map::flat_range_product(isl::map map2) const {
  auto res = isl_map_flat_range_product(copy(), map2.release());
  return manage(res);
}

isl::map map::flatten() const {
  auto res = isl_map_flatten(copy());
  return manage(res);
}

isl::map map::flatten_domain() const {
  auto res = isl_map_flatten_domain(copy());
  return manage(res);
}

isl::map map::flatten_range() const {
  auto res = isl_map_flatten_range(copy());
  return manage(res);
}

isl::map map::floordiv_val(isl::val d) const {
  auto res = isl_map_floordiv_val(copy(), d.release());
  return manage(res);
}

isl::stat map::foreach_basic_map(const std::function<isl::stat(isl::basic_map)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_basic_map *arg_0, void *arg_1) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::basic_map)> **>(arg_1);
    stat ret = (*func)(isl::manage(arg_0));
    return isl_stat(ret);
  };
  auto res = isl_map_foreach_basic_map(get(), fn_lambda, &fn_p);
  return isl::stat(res);
}

isl::map map::from_aff(isl::aff aff) {
  auto res = isl_map_from_aff(aff.release());
  return manage(res);
}

isl::map map::from_domain(isl::set set) {
  auto res = isl_map_from_domain(set.release());
  return manage(res);
}

isl::map map::from_domain_and_range(isl::set domain, isl::set range) {
  auto res = isl_map_from_domain_and_range(domain.release(), range.release());
  return manage(res);
}

isl::map map::from_multi_aff(isl::multi_aff maff) {
  auto res = isl_map_from_multi_aff(maff.release());
  return manage(res);
}

isl::map map::from_multi_pw_aff(isl::multi_pw_aff mpa) {
  auto res = isl_map_from_multi_pw_aff(mpa.release());
  return manage(res);
}

isl::map map::from_pw_aff(isl::pw_aff pwaff) {
  auto res = isl_map_from_pw_aff(pwaff.release());
  return manage(res);
}

isl::map map::from_pw_multi_aff(isl::pw_multi_aff pma) {
  auto res = isl_map_from_pw_multi_aff(pma.release());
  return manage(res);
}

isl::map map::from_range(isl::set set) {
  auto res = isl_map_from_range(set.release());
  return manage(res);
}

isl::map map::from_union_map(isl::union_map umap) {
  auto res = isl_map_from_union_map(umap.release());
  return manage(res);
}

isl::id map::get_dim_id(isl::dim type, unsigned int pos) const {
  auto res = isl_map_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

std::string map::get_dim_name(isl::dim type, unsigned int pos) const {
  auto res = isl_map_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

uint32_t map::get_hash() const {
  auto res = isl_map_get_hash(get());
  return res;
}

isl::space map::get_space() const {
  auto res = isl_map_get_space(get());
  return manage(res);
}

isl::id map::get_tuple_id(isl::dim type) const {
  auto res = isl_map_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

std::string map::get_tuple_name(isl::dim type) const {
  auto res = isl_map_get_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  std::string tmp(res);
  return tmp;
}

isl::map map::gist(isl::map context) const {
  auto res = isl_map_gist(copy(), context.release());
  return manage(res);
}

isl::map map::gist_basic_map(isl::basic_map context) const {
  auto res = isl_map_gist_basic_map(copy(), context.release());
  return manage(res);
}

isl::map map::gist_domain(isl::set context) const {
  auto res = isl_map_gist_domain(copy(), context.release());
  return manage(res);
}

isl::map map::gist_params(isl::set context) const {
  auto res = isl_map_gist_params(copy(), context.release());
  return manage(res);
}

isl::map map::gist_range(isl::set context) const {
  auto res = isl_map_gist_range(copy(), context.release());
  return manage(res);
}

isl::boolean map::has_dim_id(isl::dim type, unsigned int pos) const {
  auto res = isl_map_has_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::boolean map::has_dim_name(isl::dim type, unsigned int pos) const {
  auto res = isl_map_has_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::boolean map::has_equal_space(const isl::map &map2) const {
  auto res = isl_map_has_equal_space(get(), map2.get());
  return manage(res);
}

isl::boolean map::has_tuple_id(isl::dim type) const {
  auto res = isl_map_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::boolean map::has_tuple_name(isl::dim type) const {
  auto res = isl_map_has_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::map map::identity(isl::space dim) {
  auto res = isl_map_identity(dim.release());
  return manage(res);
}

isl::map map::insert_dims(isl::dim type, unsigned int pos, unsigned int n) const {
  auto res = isl_map_insert_dims(copy(), static_cast<enum isl_dim_type>(type), pos, n);
  return manage(res);
}

isl::map map::intersect(isl::map map2) const {
  auto res = isl_map_intersect(copy(), map2.release());
  return manage(res);
}

isl::map map::intersect_domain(isl::set set) const {
  auto res = isl_map_intersect_domain(copy(), set.release());
  return manage(res);
}

isl::map map::intersect_domain_factor_range(isl::map factor) const {
  auto res = isl_map_intersect_domain_factor_range(copy(), factor.release());
  return manage(res);
}

isl::map map::intersect_params(isl::set params) const {
  auto res = isl_map_intersect_params(copy(), params.release());
  return manage(res);
}

isl::map map::intersect_range(isl::set set) const {
  auto res = isl_map_intersect_range(copy(), set.release());
  return manage(res);
}

isl::map map::intersect_range_factor_range(isl::map factor) const {
  auto res = isl_map_intersect_range_factor_range(copy(), factor.release());
  return manage(res);
}

isl::boolean map::involves_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_map_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::boolean map::is_bijective() const {
  auto res = isl_map_is_bijective(get());
  return manage(res);
}

isl::boolean map::is_disjoint(const isl::map &map2) const {
  auto res = isl_map_is_disjoint(get(), map2.get());
  return manage(res);
}

isl::boolean map::is_empty() const {
  auto res = isl_map_is_empty(get());
  return manage(res);
}

isl::boolean map::is_equal(const isl::map &map2) const {
  auto res = isl_map_is_equal(get(), map2.get());
  return manage(res);
}

isl::boolean map::is_identity() const {
  auto res = isl_map_is_identity(get());
  return manage(res);
}

isl::boolean map::is_injective() const {
  auto res = isl_map_is_injective(get());
  return manage(res);
}

isl::boolean map::is_product() const {
  auto res = isl_map_is_product(get());
  return manage(res);
}

isl::boolean map::is_single_valued() const {
  auto res = isl_map_is_single_valued(get());
  return manage(res);
}

isl::boolean map::is_strict_subset(const isl::map &map2) const {
  auto res = isl_map_is_strict_subset(get(), map2.get());
  return manage(res);
}

isl::boolean map::is_subset(const isl::map &map2) const {
  auto res = isl_map_is_subset(get(), map2.get());
  return manage(res);
}

int map::is_translation() const {
  auto res = isl_map_is_translation(get());
  return res;
}

isl::map map::lex_ge(isl::space set_dim) {
  auto res = isl_map_lex_ge(set_dim.release());
  return manage(res);
}

isl::map map::lex_ge_first(isl::space dim, unsigned int n) {
  auto res = isl_map_lex_ge_first(dim.release(), n);
  return manage(res);
}

isl::map map::lex_ge_map(isl::map map2) const {
  auto res = isl_map_lex_ge_map(copy(), map2.release());
  return manage(res);
}

isl::map map::lex_gt(isl::space set_dim) {
  auto res = isl_map_lex_gt(set_dim.release());
  return manage(res);
}

isl::map map::lex_gt_first(isl::space dim, unsigned int n) {
  auto res = isl_map_lex_gt_first(dim.release(), n);
  return manage(res);
}

isl::map map::lex_gt_map(isl::map map2) const {
  auto res = isl_map_lex_gt_map(copy(), map2.release());
  return manage(res);
}

isl::map map::lex_le(isl::space set_dim) {
  auto res = isl_map_lex_le(set_dim.release());
  return manage(res);
}

isl::map map::lex_le_first(isl::space dim, unsigned int n) {
  auto res = isl_map_lex_le_first(dim.release(), n);
  return manage(res);
}

isl::map map::lex_le_map(isl::map map2) const {
  auto res = isl_map_lex_le_map(copy(), map2.release());
  return manage(res);
}

isl::map map::lex_lt(isl::space set_dim) {
  auto res = isl_map_lex_lt(set_dim.release());
  return manage(res);
}

isl::map map::lex_lt_first(isl::space dim, unsigned int n) {
  auto res = isl_map_lex_lt_first(dim.release(), n);
  return manage(res);
}

isl::map map::lex_lt_map(isl::map map2) const {
  auto res = isl_map_lex_lt_map(copy(), map2.release());
  return manage(res);
}

isl::map map::lexmax() const {
  auto res = isl_map_lexmax(copy());
  return manage(res);
}

isl::pw_multi_aff map::lexmax_pw_multi_aff() const {
  auto res = isl_map_lexmax_pw_multi_aff(copy());
  return manage(res);
}

isl::map map::lexmin() const {
  auto res = isl_map_lexmin(copy());
  return manage(res);
}

isl::pw_multi_aff map::lexmin_pw_multi_aff() const {
  auto res = isl_map_lexmin_pw_multi_aff(copy());
  return manage(res);
}

isl::map map::lower_bound_si(isl::dim type, unsigned int pos, int value) const {
  auto res = isl_map_lower_bound_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::map map::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const {
  auto res = isl_map_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl::map map::nat_universe(isl::space dim) {
  auto res = isl_map_nat_universe(dim.release());
  return manage(res);
}

isl::map map::neg() const {
  auto res = isl_map_neg(copy());
  return manage(res);
}

isl::map map::oppose(isl::dim type1, int pos1, isl::dim type2, int pos2) const {
  auto res = isl_map_oppose(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

isl::map map::order_ge(isl::dim type1, int pos1, isl::dim type2, int pos2) const {
  auto res = isl_map_order_ge(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

isl::map map::order_gt(isl::dim type1, int pos1, isl::dim type2, int pos2) const {
  auto res = isl_map_order_gt(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

isl::map map::order_le(isl::dim type1, int pos1, isl::dim type2, int pos2) const {
  auto res = isl_map_order_le(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

isl::map map::order_lt(isl::dim type1, int pos1, isl::dim type2, int pos2) const {
  auto res = isl_map_order_lt(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

isl::set map::params() const {
  auto res = isl_map_params(copy());
  return manage(res);
}

isl::val map::plain_get_val_if_fixed(isl::dim type, unsigned int pos) const {
  auto res = isl_map_plain_get_val_if_fixed(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::boolean map::plain_is_empty() const {
  auto res = isl_map_plain_is_empty(get());
  return manage(res);
}

isl::boolean map::plain_is_equal(const isl::map &map2) const {
  auto res = isl_map_plain_is_equal(get(), map2.get());
  return manage(res);
}

isl::boolean map::plain_is_injective() const {
  auto res = isl_map_plain_is_injective(get());
  return manage(res);
}

isl::boolean map::plain_is_single_valued() const {
  auto res = isl_map_plain_is_single_valued(get());
  return manage(res);
}

isl::boolean map::plain_is_universe() const {
  auto res = isl_map_plain_is_universe(get());
  return manage(res);
}

isl::basic_map map::plain_unshifted_simple_hull() const {
  auto res = isl_map_plain_unshifted_simple_hull(copy());
  return manage(res);
}

isl::basic_map map::polyhedral_hull() const {
  auto res = isl_map_polyhedral_hull(copy());
  return manage(res);
}

isl::map map::preimage_domain_multi_aff(isl::multi_aff ma) const {
  auto res = isl_map_preimage_domain_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::map map::preimage_domain_multi_pw_aff(isl::multi_pw_aff mpa) const {
  auto res = isl_map_preimage_domain_multi_pw_aff(copy(), mpa.release());
  return manage(res);
}

isl::map map::preimage_domain_pw_multi_aff(isl::pw_multi_aff pma) const {
  auto res = isl_map_preimage_domain_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::map map::preimage_range_multi_aff(isl::multi_aff ma) const {
  auto res = isl_map_preimage_range_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::map map::preimage_range_pw_multi_aff(isl::pw_multi_aff pma) const {
  auto res = isl_map_preimage_range_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::map map::product(isl::map map2) const {
  auto res = isl_map_product(copy(), map2.release());
  return manage(res);
}

isl::map map::project_out(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_map_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::set map::range() const {
  auto res = isl_map_range(copy());
  return manage(res);
}

isl::map map::range_curry() const {
  auto res = isl_map_range_curry(copy());
  return manage(res);
}

isl::map map::range_factor_domain() const {
  auto res = isl_map_range_factor_domain(copy());
  return manage(res);
}

isl::map map::range_factor_range() const {
  auto res = isl_map_range_factor_range(copy());
  return manage(res);
}

isl::boolean map::range_is_wrapping() const {
  auto res = isl_map_range_is_wrapping(get());
  return manage(res);
}

isl::map map::range_map() const {
  auto res = isl_map_range_map(copy());
  return manage(res);
}

isl::map map::range_product(isl::map map2) const {
  auto res = isl_map_range_product(copy(), map2.release());
  return manage(res);
}

isl::map map::remove_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_map_remove_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::map map::remove_divs() const {
  auto res = isl_map_remove_divs(copy());
  return manage(res);
}

isl::map map::remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_map_remove_divs_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::map map::remove_redundancies() const {
  auto res = isl_map_remove_redundancies(copy());
  return manage(res);
}

isl::map map::remove_unknown_divs() const {
  auto res = isl_map_remove_unknown_divs(copy());
  return manage(res);
}

isl::map map::reset_tuple_id(isl::dim type) const {
  auto res = isl_map_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::map map::reset_user() const {
  auto res = isl_map_reset_user(copy());
  return manage(res);
}

isl::map map::reverse() const {
  auto res = isl_map_reverse(copy());
  return manage(res);
}

isl::basic_map map::sample() const {
  auto res = isl_map_sample(copy());
  return manage(res);
}

isl::map map::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const {
  auto res = isl_map_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::map map::set_tuple_id(isl::dim type, isl::id id) const {
  auto res = isl_map_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::map map::set_tuple_name(isl::dim type, const std::string &s) const {
  auto res = isl_map_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

isl::basic_map map::simple_hull() const {
  auto res = isl_map_simple_hull(copy());
  return manage(res);
}

isl::map map::subtract(isl::map map2) const {
  auto res = isl_map_subtract(copy(), map2.release());
  return manage(res);
}

isl::map map::subtract_domain(isl::set dom) const {
  auto res = isl_map_subtract_domain(copy(), dom.release());
  return manage(res);
}

isl::map map::subtract_range(isl::set dom) const {
  auto res = isl_map_subtract_range(copy(), dom.release());
  return manage(res);
}

isl::map map::sum(isl::map map2) const {
  auto res = isl_map_sum(copy(), map2.release());
  return manage(res);
}

isl::map map::uncurry() const {
  auto res = isl_map_uncurry(copy());
  return manage(res);
}

isl::map map::unite(isl::map map2) const {
  auto res = isl_map_union(copy(), map2.release());
  return manage(res);
}

isl::map map::universe(isl::space dim) {
  auto res = isl_map_universe(dim.release());
  return manage(res);
}

isl::basic_map map::unshifted_simple_hull() const {
  auto res = isl_map_unshifted_simple_hull(copy());
  return manage(res);
}

isl::basic_map map::unshifted_simple_hull_from_map_list(isl::map_list list) const {
  auto res = isl_map_unshifted_simple_hull_from_map_list(copy(), list.release());
  return manage(res);
}

isl::map map::upper_bound_si(isl::dim type, unsigned int pos, int value) const {
  auto res = isl_map_upper_bound_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::set map::wrap() const {
  auto res = isl_map_wrap(copy());
  return manage(res);
}

isl::map map::zip() const {
  auto res = isl_map_zip(copy());
  return manage(res);
}

// implementations for isl::map_list
isl::map_list manage(__isl_take isl_map_list *ptr) {
  return map_list(ptr);
}
isl::map_list give(__isl_take isl_map_list *ptr) {
  return manage(ptr);
}


map_list::map_list()
    : ptr(nullptr) {}

map_list::map_list(const isl::map_list &obj)
    : ptr(obj.copy()) {}
map_list::map_list(std::nullptr_t)
    : ptr(nullptr) {}


map_list::map_list(__isl_take isl_map_list *ptr)
    : ptr(ptr) {}


map_list &map_list::operator=(isl::map_list obj) {
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
__isl_keep isl_map_list *map_list::keep() const {
  return get();
}

__isl_give isl_map_list *map_list::take() {
  return release();
}

map_list::operator bool() const {
  return !is_null();
}

isl::ctx map_list::get_ctx() const {
  return isl::ctx(isl_map_list_get_ctx(ptr));
}



void map_list::dump() const {
  isl_map_list_dump(get());
}



// implementations for isl::multi_aff
isl::multi_aff manage(__isl_take isl_multi_aff *ptr) {
  return multi_aff(ptr);
}
isl::multi_aff give(__isl_take isl_multi_aff *ptr) {
  return manage(ptr);
}


multi_aff::multi_aff()
    : ptr(nullptr) {}

multi_aff::multi_aff(const isl::multi_aff &obj)
    : ptr(obj.copy()) {}
multi_aff::multi_aff(std::nullptr_t)
    : ptr(nullptr) {}


multi_aff::multi_aff(__isl_take isl_multi_aff *ptr)
    : ptr(ptr) {}

multi_aff::multi_aff(isl::ctx ctx, const std::string &str) {
  auto res = isl_multi_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}
multi_aff::multi_aff(isl::aff aff) {
  auto res = isl_multi_aff_from_aff(aff.release());
  ptr = res;
}

multi_aff &multi_aff::operator=(isl::multi_aff obj) {
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
__isl_keep isl_multi_aff *multi_aff::keep() const {
  return get();
}

__isl_give isl_multi_aff *multi_aff::take() {
  return release();
}

multi_aff::operator bool() const {
  return !is_null();
}

isl::ctx multi_aff::get_ctx() const {
  return isl::ctx(isl_multi_aff_get_ctx(ptr));
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


isl::multi_aff multi_aff::add(isl::multi_aff multi2) const {
  auto res = isl_multi_aff_add(copy(), multi2.release());
  return manage(res);
}

isl::multi_aff multi_aff::add_dims(isl::dim type, unsigned int n) const {
  auto res = isl_multi_aff_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::multi_aff multi_aff::align_params(isl::space model) const {
  auto res = isl_multi_aff_align_params(copy(), model.release());
  return manage(res);
}

unsigned int multi_aff::dim(isl::dim type) const {
  auto res = isl_multi_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::multi_aff multi_aff::domain_map(isl::space space) {
  auto res = isl_multi_aff_domain_map(space.release());
  return manage(res);
}

isl::multi_aff multi_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_multi_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::multi_aff multi_aff::factor_range() const {
  auto res = isl_multi_aff_factor_range(copy());
  return manage(res);
}

int multi_aff::find_dim_by_id(isl::dim type, const isl::id &id) const {
  auto res = isl_multi_aff_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int multi_aff::find_dim_by_name(isl::dim type, const std::string &name) const {
  auto res = isl_multi_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::multi_aff multi_aff::flat_range_product(isl::multi_aff multi2) const {
  auto res = isl_multi_aff_flat_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_aff multi_aff::flatten_domain() const {
  auto res = isl_multi_aff_flatten_domain(copy());
  return manage(res);
}

isl::multi_aff multi_aff::flatten_range() const {
  auto res = isl_multi_aff_flatten_range(copy());
  return manage(res);
}

isl::multi_aff multi_aff::floor() const {
  auto res = isl_multi_aff_floor(copy());
  return manage(res);
}

isl::multi_aff multi_aff::from_aff_list(isl::space space, isl::aff_list list) {
  auto res = isl_multi_aff_from_aff_list(space.release(), list.release());
  return manage(res);
}

isl::multi_aff multi_aff::from_range() const {
  auto res = isl_multi_aff_from_range(copy());
  return manage(res);
}

isl::aff multi_aff::get_aff(int pos) const {
  auto res = isl_multi_aff_get_aff(get(), pos);
  return manage(res);
}

isl::id multi_aff::get_dim_id(isl::dim type, unsigned int pos) const {
  auto res = isl_multi_aff_get_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::space multi_aff::get_domain_space() const {
  auto res = isl_multi_aff_get_domain_space(get());
  return manage(res);
}

isl::space multi_aff::get_space() const {
  auto res = isl_multi_aff_get_space(get());
  return manage(res);
}

isl::id multi_aff::get_tuple_id(isl::dim type) const {
  auto res = isl_multi_aff_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

std::string multi_aff::get_tuple_name(isl::dim type) const {
  auto res = isl_multi_aff_get_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  std::string tmp(res);
  return tmp;
}

isl::multi_aff multi_aff::gist(isl::set context) const {
  auto res = isl_multi_aff_gist(copy(), context.release());
  return manage(res);
}

isl::multi_aff multi_aff::gist_params(isl::set context) const {
  auto res = isl_multi_aff_gist_params(copy(), context.release());
  return manage(res);
}

isl::boolean multi_aff::has_tuple_id(isl::dim type) const {
  auto res = isl_multi_aff_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::multi_aff multi_aff::identity(isl::space space) {
  auto res = isl_multi_aff_identity(space.release());
  return manage(res);
}

isl::multi_aff multi_aff::insert_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_multi_aff_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::boolean multi_aff::involves_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_multi_aff_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::boolean multi_aff::involves_nan() const {
  auto res = isl_multi_aff_involves_nan(get());
  return manage(res);
}

isl::set multi_aff::lex_ge_set(isl::multi_aff ma2) const {
  auto res = isl_multi_aff_lex_ge_set(copy(), ma2.release());
  return manage(res);
}

isl::set multi_aff::lex_gt_set(isl::multi_aff ma2) const {
  auto res = isl_multi_aff_lex_gt_set(copy(), ma2.release());
  return manage(res);
}

isl::set multi_aff::lex_le_set(isl::multi_aff ma2) const {
  auto res = isl_multi_aff_lex_le_set(copy(), ma2.release());
  return manage(res);
}

isl::set multi_aff::lex_lt_set(isl::multi_aff ma2) const {
  auto res = isl_multi_aff_lex_lt_set(copy(), ma2.release());
  return manage(res);
}

isl::multi_aff multi_aff::mod_multi_val(isl::multi_val mv) const {
  auto res = isl_multi_aff_mod_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_aff multi_aff::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const {
  auto res = isl_multi_aff_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl::multi_aff multi_aff::multi_val_on_space(isl::space space, isl::multi_val mv) {
  auto res = isl_multi_aff_multi_val_on_space(space.release(), mv.release());
  return manage(res);
}

isl::multi_aff multi_aff::neg() const {
  auto res = isl_multi_aff_neg(copy());
  return manage(res);
}

int multi_aff::plain_cmp(const isl::multi_aff &multi2) const {
  auto res = isl_multi_aff_plain_cmp(get(), multi2.get());
  return res;
}

isl::boolean multi_aff::plain_is_equal(const isl::multi_aff &multi2) const {
  auto res = isl_multi_aff_plain_is_equal(get(), multi2.get());
  return manage(res);
}

isl::multi_aff multi_aff::product(isl::multi_aff multi2) const {
  auto res = isl_multi_aff_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_aff multi_aff::project_out_map(isl::space space, isl::dim type, unsigned int first, unsigned int n) {
  auto res = isl_multi_aff_project_out_map(space.release(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::multi_aff multi_aff::pullback(isl::multi_aff ma2) const {
  auto res = isl_multi_aff_pullback_multi_aff(copy(), ma2.release());
  return manage(res);
}

isl::multi_aff multi_aff::range_factor_domain() const {
  auto res = isl_multi_aff_range_factor_domain(copy());
  return manage(res);
}

isl::multi_aff multi_aff::range_factor_range() const {
  auto res = isl_multi_aff_range_factor_range(copy());
  return manage(res);
}

isl::boolean multi_aff::range_is_wrapping() const {
  auto res = isl_multi_aff_range_is_wrapping(get());
  return manage(res);
}

isl::multi_aff multi_aff::range_map(isl::space space) {
  auto res = isl_multi_aff_range_map(space.release());
  return manage(res);
}

isl::multi_aff multi_aff::range_product(isl::multi_aff multi2) const {
  auto res = isl_multi_aff_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_aff multi_aff::range_splice(unsigned int pos, isl::multi_aff multi2) const {
  auto res = isl_multi_aff_range_splice(copy(), pos, multi2.release());
  return manage(res);
}

isl::multi_aff multi_aff::reset_tuple_id(isl::dim type) const {
  auto res = isl_multi_aff_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::multi_aff multi_aff::reset_user() const {
  auto res = isl_multi_aff_reset_user(copy());
  return manage(res);
}

isl::multi_aff multi_aff::scale_down_multi_val(isl::multi_val mv) const {
  auto res = isl_multi_aff_scale_down_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_aff multi_aff::scale_down_val(isl::val v) const {
  auto res = isl_multi_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::multi_aff multi_aff::scale_multi_val(isl::multi_val mv) const {
  auto res = isl_multi_aff_scale_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_aff multi_aff::scale_val(isl::val v) const {
  auto res = isl_multi_aff_scale_val(copy(), v.release());
  return manage(res);
}

isl::multi_aff multi_aff::set_aff(int pos, isl::aff el) const {
  auto res = isl_multi_aff_set_aff(copy(), pos, el.release());
  return manage(res);
}

isl::multi_aff multi_aff::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const {
  auto res = isl_multi_aff_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::multi_aff multi_aff::set_tuple_id(isl::dim type, isl::id id) const {
  auto res = isl_multi_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::multi_aff multi_aff::set_tuple_name(isl::dim type, const std::string &s) const {
  auto res = isl_multi_aff_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

isl::multi_aff multi_aff::splice(unsigned int in_pos, unsigned int out_pos, isl::multi_aff multi2) const {
  auto res = isl_multi_aff_splice(copy(), in_pos, out_pos, multi2.release());
  return manage(res);
}

isl::multi_aff multi_aff::sub(isl::multi_aff multi2) const {
  auto res = isl_multi_aff_sub(copy(), multi2.release());
  return manage(res);
}

isl::multi_aff multi_aff::zero(isl::space space) {
  auto res = isl_multi_aff_zero(space.release());
  return manage(res);
}

// implementations for isl::multi_pw_aff
isl::multi_pw_aff manage(__isl_take isl_multi_pw_aff *ptr) {
  return multi_pw_aff(ptr);
}
isl::multi_pw_aff give(__isl_take isl_multi_pw_aff *ptr) {
  return manage(ptr);
}


multi_pw_aff::multi_pw_aff()
    : ptr(nullptr) {}

multi_pw_aff::multi_pw_aff(const isl::multi_pw_aff &obj)
    : ptr(obj.copy()) {}
multi_pw_aff::multi_pw_aff(std::nullptr_t)
    : ptr(nullptr) {}


multi_pw_aff::multi_pw_aff(__isl_take isl_multi_pw_aff *ptr)
    : ptr(ptr) {}

multi_pw_aff::multi_pw_aff(isl::multi_aff ma) {
  auto res = isl_multi_pw_aff_from_multi_aff(ma.release());
  ptr = res;
}
multi_pw_aff::multi_pw_aff(isl::pw_aff pa) {
  auto res = isl_multi_pw_aff_from_pw_aff(pa.release());
  ptr = res;
}
multi_pw_aff::multi_pw_aff(isl::pw_multi_aff pma) {
  auto res = isl_multi_pw_aff_from_pw_multi_aff(pma.release());
  ptr = res;
}
multi_pw_aff::multi_pw_aff(isl::ctx ctx, const std::string &str) {
  auto res = isl_multi_pw_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

multi_pw_aff &multi_pw_aff::operator=(isl::multi_pw_aff obj) {
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
__isl_keep isl_multi_pw_aff *multi_pw_aff::keep() const {
  return get();
}

__isl_give isl_multi_pw_aff *multi_pw_aff::take() {
  return release();
}

multi_pw_aff::operator bool() const {
  return !is_null();
}

isl::ctx multi_pw_aff::get_ctx() const {
  return isl::ctx(isl_multi_pw_aff_get_ctx(ptr));
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


isl::multi_pw_aff multi_pw_aff::add(isl::multi_pw_aff multi2) const {
  auto res = isl_multi_pw_aff_add(copy(), multi2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::add_dims(isl::dim type, unsigned int n) const {
  auto res = isl_multi_pw_aff_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::align_params(isl::space model) const {
  auto res = isl_multi_pw_aff_align_params(copy(), model.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::coalesce() const {
  auto res = isl_multi_pw_aff_coalesce(copy());
  return manage(res);
}

unsigned int multi_pw_aff::dim(isl::dim type) const {
  auto res = isl_multi_pw_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::set multi_pw_aff::domain() const {
  auto res = isl_multi_pw_aff_domain(copy());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_multi_pw_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::map multi_pw_aff::eq_map(isl::multi_pw_aff mpa2) const {
  auto res = isl_multi_pw_aff_eq_map(copy(), mpa2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::factor_range() const {
  auto res = isl_multi_pw_aff_factor_range(copy());
  return manage(res);
}

int multi_pw_aff::find_dim_by_id(isl::dim type, const isl::id &id) const {
  auto res = isl_multi_pw_aff_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int multi_pw_aff::find_dim_by_name(isl::dim type, const std::string &name) const {
  auto res = isl_multi_pw_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::multi_pw_aff multi_pw_aff::flat_range_product(isl::multi_pw_aff multi2) const {
  auto res = isl_multi_pw_aff_flat_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::flatten_range() const {
  auto res = isl_multi_pw_aff_flatten_range(copy());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::from_pw_aff_list(isl::space space, isl::pw_aff_list list) {
  auto res = isl_multi_pw_aff_from_pw_aff_list(space.release(), list.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::from_range() const {
  auto res = isl_multi_pw_aff_from_range(copy());
  return manage(res);
}

isl::id multi_pw_aff::get_dim_id(isl::dim type, unsigned int pos) const {
  auto res = isl_multi_pw_aff_get_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::space multi_pw_aff::get_domain_space() const {
  auto res = isl_multi_pw_aff_get_domain_space(get());
  return manage(res);
}

uint32_t multi_pw_aff::get_hash() const {
  auto res = isl_multi_pw_aff_get_hash(get());
  return res;
}

isl::pw_aff multi_pw_aff::get_pw_aff(int pos) const {
  auto res = isl_multi_pw_aff_get_pw_aff(get(), pos);
  return manage(res);
}

isl::space multi_pw_aff::get_space() const {
  auto res = isl_multi_pw_aff_get_space(get());
  return manage(res);
}

isl::id multi_pw_aff::get_tuple_id(isl::dim type) const {
  auto res = isl_multi_pw_aff_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

std::string multi_pw_aff::get_tuple_name(isl::dim type) const {
  auto res = isl_multi_pw_aff_get_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  std::string tmp(res);
  return tmp;
}

isl::multi_pw_aff multi_pw_aff::gist(isl::set set) const {
  auto res = isl_multi_pw_aff_gist(copy(), set.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::gist_params(isl::set set) const {
  auto res = isl_multi_pw_aff_gist_params(copy(), set.release());
  return manage(res);
}

isl::boolean multi_pw_aff::has_tuple_id(isl::dim type) const {
  auto res = isl_multi_pw_aff_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::identity(isl::space space) {
  auto res = isl_multi_pw_aff_identity(space.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::insert_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_multi_pw_aff_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::intersect_domain(isl::set domain) const {
  auto res = isl_multi_pw_aff_intersect_domain(copy(), domain.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::intersect_params(isl::set set) const {
  auto res = isl_multi_pw_aff_intersect_params(copy(), set.release());
  return manage(res);
}

isl::boolean multi_pw_aff::involves_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_multi_pw_aff_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::boolean multi_pw_aff::involves_nan() const {
  auto res = isl_multi_pw_aff_involves_nan(get());
  return manage(res);
}

isl::boolean multi_pw_aff::is_cst() const {
  auto res = isl_multi_pw_aff_is_cst(get());
  return manage(res);
}

isl::boolean multi_pw_aff::is_equal(const isl::multi_pw_aff &mpa2) const {
  auto res = isl_multi_pw_aff_is_equal(get(), mpa2.get());
  return manage(res);
}

isl::map multi_pw_aff::lex_gt_map(isl::multi_pw_aff mpa2) const {
  auto res = isl_multi_pw_aff_lex_gt_map(copy(), mpa2.release());
  return manage(res);
}

isl::map multi_pw_aff::lex_lt_map(isl::multi_pw_aff mpa2) const {
  auto res = isl_multi_pw_aff_lex_lt_map(copy(), mpa2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::mod_multi_val(isl::multi_val mv) const {
  auto res = isl_multi_pw_aff_mod_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const {
  auto res = isl_multi_pw_aff_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::neg() const {
  auto res = isl_multi_pw_aff_neg(copy());
  return manage(res);
}

isl::boolean multi_pw_aff::plain_is_equal(const isl::multi_pw_aff &multi2) const {
  auto res = isl_multi_pw_aff_plain_is_equal(get(), multi2.get());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::product(isl::multi_pw_aff multi2) const {
  auto res = isl_multi_pw_aff_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::pullback(isl::multi_aff ma) const {
  auto res = isl_multi_pw_aff_pullback_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::pullback(isl::pw_multi_aff pma) const {
  auto res = isl_multi_pw_aff_pullback_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::pullback(isl::multi_pw_aff mpa2) const {
  auto res = isl_multi_pw_aff_pullback_multi_pw_aff(copy(), mpa2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::range_factor_domain() const {
  auto res = isl_multi_pw_aff_range_factor_domain(copy());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::range_factor_range() const {
  auto res = isl_multi_pw_aff_range_factor_range(copy());
  return manage(res);
}

isl::boolean multi_pw_aff::range_is_wrapping() const {
  auto res = isl_multi_pw_aff_range_is_wrapping(get());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::range_product(isl::multi_pw_aff multi2) const {
  auto res = isl_multi_pw_aff_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::range_splice(unsigned int pos, isl::multi_pw_aff multi2) const {
  auto res = isl_multi_pw_aff_range_splice(copy(), pos, multi2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::reset_tuple_id(isl::dim type) const {
  auto res = isl_multi_pw_aff_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::reset_user() const {
  auto res = isl_multi_pw_aff_reset_user(copy());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::scale_down_multi_val(isl::multi_val mv) const {
  auto res = isl_multi_pw_aff_scale_down_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::scale_down_val(isl::val v) const {
  auto res = isl_multi_pw_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::scale_multi_val(isl::multi_val mv) const {
  auto res = isl_multi_pw_aff_scale_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::scale_val(isl::val v) const {
  auto res = isl_multi_pw_aff_scale_val(copy(), v.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const {
  auto res = isl_multi_pw_aff_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::set_pw_aff(int pos, isl::pw_aff el) const {
  auto res = isl_multi_pw_aff_set_pw_aff(copy(), pos, el.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::set_tuple_id(isl::dim type, isl::id id) const {
  auto res = isl_multi_pw_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::set_tuple_name(isl::dim type, const std::string &s) const {
  auto res = isl_multi_pw_aff_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::splice(unsigned int in_pos, unsigned int out_pos, isl::multi_pw_aff multi2) const {
  auto res = isl_multi_pw_aff_splice(copy(), in_pos, out_pos, multi2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::sub(isl::multi_pw_aff multi2) const {
  auto res = isl_multi_pw_aff_sub(copy(), multi2.release());
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::zero(isl::space space) {
  auto res = isl_multi_pw_aff_zero(space.release());
  return manage(res);
}

// implementations for isl::multi_union_pw_aff
isl::multi_union_pw_aff manage(__isl_take isl_multi_union_pw_aff *ptr) {
  return multi_union_pw_aff(ptr);
}
isl::multi_union_pw_aff give(__isl_take isl_multi_union_pw_aff *ptr) {
  return manage(ptr);
}


multi_union_pw_aff::multi_union_pw_aff()
    : ptr(nullptr) {}

multi_union_pw_aff::multi_union_pw_aff(const isl::multi_union_pw_aff &obj)
    : ptr(obj.copy()) {}
multi_union_pw_aff::multi_union_pw_aff(std::nullptr_t)
    : ptr(nullptr) {}


multi_union_pw_aff::multi_union_pw_aff(__isl_take isl_multi_union_pw_aff *ptr)
    : ptr(ptr) {}

multi_union_pw_aff::multi_union_pw_aff(isl::union_pw_aff upa) {
  auto res = isl_multi_union_pw_aff_from_union_pw_aff(upa.release());
  ptr = res;
}
multi_union_pw_aff::multi_union_pw_aff(isl::multi_pw_aff mpa) {
  auto res = isl_multi_union_pw_aff_from_multi_pw_aff(mpa.release());
  ptr = res;
}
multi_union_pw_aff::multi_union_pw_aff(isl::union_pw_multi_aff upma) {
  auto res = isl_multi_union_pw_aff_from_union_pw_multi_aff(upma.release());
  ptr = res;
}
multi_union_pw_aff::multi_union_pw_aff(isl::ctx ctx, const std::string &str) {
  auto res = isl_multi_union_pw_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

multi_union_pw_aff &multi_union_pw_aff::operator=(isl::multi_union_pw_aff obj) {
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
__isl_keep isl_multi_union_pw_aff *multi_union_pw_aff::keep() const {
  return get();
}

__isl_give isl_multi_union_pw_aff *multi_union_pw_aff::take() {
  return release();
}

multi_union_pw_aff::operator bool() const {
  return !is_null();
}

isl::ctx multi_union_pw_aff::get_ctx() const {
  return isl::ctx(isl_multi_union_pw_aff_get_ctx(ptr));
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


isl::multi_union_pw_aff multi_union_pw_aff::add(isl::multi_union_pw_aff multi2) const {
  auto res = isl_multi_union_pw_aff_add(copy(), multi2.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::align_params(isl::space model) const {
  auto res = isl_multi_union_pw_aff_align_params(copy(), model.release());
  return manage(res);
}

isl::union_pw_aff multi_union_pw_aff::apply_aff(isl::aff aff) const {
  auto res = isl_multi_union_pw_aff_apply_aff(copy(), aff.release());
  return manage(res);
}

isl::union_pw_aff multi_union_pw_aff::apply_pw_aff(isl::pw_aff pa) const {
  auto res = isl_multi_union_pw_aff_apply_pw_aff(copy(), pa.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::apply_pw_multi_aff(isl::pw_multi_aff pma) const {
  auto res = isl_multi_union_pw_aff_apply_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::coalesce() const {
  auto res = isl_multi_union_pw_aff_coalesce(copy());
  return manage(res);
}

unsigned int multi_union_pw_aff::dim(isl::dim type) const {
  auto res = isl_multi_union_pw_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::union_set multi_union_pw_aff::domain() const {
  auto res = isl_multi_union_pw_aff_domain(copy());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_multi_union_pw_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::multi_pw_aff multi_union_pw_aff::extract_multi_pw_aff(isl::space space) const {
  auto res = isl_multi_union_pw_aff_extract_multi_pw_aff(get(), space.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::factor_range() const {
  auto res = isl_multi_union_pw_aff_factor_range(copy());
  return manage(res);
}

int multi_union_pw_aff::find_dim_by_id(isl::dim type, const isl::id &id) const {
  auto res = isl_multi_union_pw_aff_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int multi_union_pw_aff::find_dim_by_name(isl::dim type, const std::string &name) const {
  auto res = isl_multi_union_pw_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::multi_union_pw_aff multi_union_pw_aff::flat_range_product(isl::multi_union_pw_aff multi2) const {
  auto res = isl_multi_union_pw_aff_flat_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::flatten_range() const {
  auto res = isl_multi_union_pw_aff_flatten_range(copy());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::floor() const {
  auto res = isl_multi_union_pw_aff_floor(copy());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::from_multi_aff(isl::multi_aff ma) {
  auto res = isl_multi_union_pw_aff_from_multi_aff(ma.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::from_range() const {
  auto res = isl_multi_union_pw_aff_from_range(copy());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::from_union_map(isl::union_map umap) {
  auto res = isl_multi_union_pw_aff_from_union_map(umap.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::from_union_pw_aff_list(isl::space space, isl::union_pw_aff_list list) {
  auto res = isl_multi_union_pw_aff_from_union_pw_aff_list(space.release(), list.release());
  return manage(res);
}

isl::id multi_union_pw_aff::get_dim_id(isl::dim type, unsigned int pos) const {
  auto res = isl_multi_union_pw_aff_get_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::space multi_union_pw_aff::get_domain_space() const {
  auto res = isl_multi_union_pw_aff_get_domain_space(get());
  return manage(res);
}

isl::space multi_union_pw_aff::get_space() const {
  auto res = isl_multi_union_pw_aff_get_space(get());
  return manage(res);
}

isl::id multi_union_pw_aff::get_tuple_id(isl::dim type) const {
  auto res = isl_multi_union_pw_aff_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

std::string multi_union_pw_aff::get_tuple_name(isl::dim type) const {
  auto res = isl_multi_union_pw_aff_get_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  std::string tmp(res);
  return tmp;
}

isl::union_pw_aff multi_union_pw_aff::get_union_pw_aff(int pos) const {
  auto res = isl_multi_union_pw_aff_get_union_pw_aff(get(), pos);
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::gist(isl::union_set context) const {
  auto res = isl_multi_union_pw_aff_gist(copy(), context.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::gist_params(isl::set context) const {
  auto res = isl_multi_union_pw_aff_gist_params(copy(), context.release());
  return manage(res);
}

isl::boolean multi_union_pw_aff::has_tuple_id(isl::dim type) const {
  auto res = isl_multi_union_pw_aff_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::intersect_domain(isl::union_set uset) const {
  auto res = isl_multi_union_pw_aff_intersect_domain(copy(), uset.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::intersect_params(isl::set params) const {
  auto res = isl_multi_union_pw_aff_intersect_params(copy(), params.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::intersect_range(isl::set set) const {
  auto res = isl_multi_union_pw_aff_intersect_range(copy(), set.release());
  return manage(res);
}

isl::boolean multi_union_pw_aff::involves_nan() const {
  auto res = isl_multi_union_pw_aff_involves_nan(get());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::mod_multi_val(isl::multi_val mv) const {
  auto res = isl_multi_union_pw_aff_mod_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::multi_aff_on_domain(isl::union_set domain, isl::multi_aff ma) {
  auto res = isl_multi_union_pw_aff_multi_aff_on_domain(domain.release(), ma.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::multi_val_on_domain(isl::union_set domain, isl::multi_val mv) {
  auto res = isl_multi_union_pw_aff_multi_val_on_domain(domain.release(), mv.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::neg() const {
  auto res = isl_multi_union_pw_aff_neg(copy());
  return manage(res);
}

isl::boolean multi_union_pw_aff::plain_is_equal(const isl::multi_union_pw_aff &multi2) const {
  auto res = isl_multi_union_pw_aff_plain_is_equal(get(), multi2.get());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::pullback(isl::union_pw_multi_aff upma) const {
  auto res = isl_multi_union_pw_aff_pullback_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::range_factor_domain() const {
  auto res = isl_multi_union_pw_aff_range_factor_domain(copy());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::range_factor_range() const {
  auto res = isl_multi_union_pw_aff_range_factor_range(copy());
  return manage(res);
}

isl::boolean multi_union_pw_aff::range_is_wrapping() const {
  auto res = isl_multi_union_pw_aff_range_is_wrapping(get());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::range_product(isl::multi_union_pw_aff multi2) const {
  auto res = isl_multi_union_pw_aff_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::range_splice(unsigned int pos, isl::multi_union_pw_aff multi2) const {
  auto res = isl_multi_union_pw_aff_range_splice(copy(), pos, multi2.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::reset_tuple_id(isl::dim type) const {
  auto res = isl_multi_union_pw_aff_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::reset_user() const {
  auto res = isl_multi_union_pw_aff_reset_user(copy());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::scale_down_multi_val(isl::multi_val mv) const {
  auto res = isl_multi_union_pw_aff_scale_down_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::scale_down_val(isl::val v) const {
  auto res = isl_multi_union_pw_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::scale_multi_val(isl::multi_val mv) const {
  auto res = isl_multi_union_pw_aff_scale_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::scale_val(isl::val v) const {
  auto res = isl_multi_union_pw_aff_scale_val(copy(), v.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const {
  auto res = isl_multi_union_pw_aff_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::set_tuple_id(isl::dim type, isl::id id) const {
  auto res = isl_multi_union_pw_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::set_tuple_name(isl::dim type, const std::string &s) const {
  auto res = isl_multi_union_pw_aff_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::set_union_pw_aff(int pos, isl::union_pw_aff el) const {
  auto res = isl_multi_union_pw_aff_set_union_pw_aff(copy(), pos, el.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::sub(isl::multi_union_pw_aff multi2) const {
  auto res = isl_multi_union_pw_aff_sub(copy(), multi2.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::union_add(isl::multi_union_pw_aff mupa2) const {
  auto res = isl_multi_union_pw_aff_union_add(copy(), mupa2.release());
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::zero(isl::space space) {
  auto res = isl_multi_union_pw_aff_zero(space.release());
  return manage(res);
}

isl::union_set multi_union_pw_aff::zero_union_set() const {
  auto res = isl_multi_union_pw_aff_zero_union_set(copy());
  return manage(res);
}

// implementations for isl::multi_val
isl::multi_val manage(__isl_take isl_multi_val *ptr) {
  return multi_val(ptr);
}
isl::multi_val give(__isl_take isl_multi_val *ptr) {
  return manage(ptr);
}


multi_val::multi_val()
    : ptr(nullptr) {}

multi_val::multi_val(const isl::multi_val &obj)
    : ptr(obj.copy()) {}
multi_val::multi_val(std::nullptr_t)
    : ptr(nullptr) {}


multi_val::multi_val(__isl_take isl_multi_val *ptr)
    : ptr(ptr) {}

multi_val::multi_val(isl::ctx ctx, const std::string &str) {
  auto res = isl_multi_val_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

multi_val &multi_val::operator=(isl::multi_val obj) {
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
__isl_keep isl_multi_val *multi_val::keep() const {
  return get();
}

__isl_give isl_multi_val *multi_val::take() {
  return release();
}

multi_val::operator bool() const {
  return !is_null();
}

isl::ctx multi_val::get_ctx() const {
  return isl::ctx(isl_multi_val_get_ctx(ptr));
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


isl::multi_val multi_val::add(isl::multi_val multi2) const {
  auto res = isl_multi_val_add(copy(), multi2.release());
  return manage(res);
}

isl::multi_val multi_val::add_dims(isl::dim type, unsigned int n) const {
  auto res = isl_multi_val_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::multi_val multi_val::add_val(isl::val v) const {
  auto res = isl_multi_val_add_val(copy(), v.release());
  return manage(res);
}

isl::multi_val multi_val::align_params(isl::space model) const {
  auto res = isl_multi_val_align_params(copy(), model.release());
  return manage(res);
}

unsigned int multi_val::dim(isl::dim type) const {
  auto res = isl_multi_val_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::multi_val multi_val::drop_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_multi_val_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::multi_val multi_val::factor_range() const {
  auto res = isl_multi_val_factor_range(copy());
  return manage(res);
}

int multi_val::find_dim_by_id(isl::dim type, const isl::id &id) const {
  auto res = isl_multi_val_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int multi_val::find_dim_by_name(isl::dim type, const std::string &name) const {
  auto res = isl_multi_val_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::multi_val multi_val::flat_range_product(isl::multi_val multi2) const {
  auto res = isl_multi_val_flat_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_val multi_val::flatten_range() const {
  auto res = isl_multi_val_flatten_range(copy());
  return manage(res);
}

isl::multi_val multi_val::from_range() const {
  auto res = isl_multi_val_from_range(copy());
  return manage(res);
}

isl::multi_val multi_val::from_val_list(isl::space space, isl::val_list list) {
  auto res = isl_multi_val_from_val_list(space.release(), list.release());
  return manage(res);
}

isl::id multi_val::get_dim_id(isl::dim type, unsigned int pos) const {
  auto res = isl_multi_val_get_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::space multi_val::get_domain_space() const {
  auto res = isl_multi_val_get_domain_space(get());
  return manage(res);
}

isl::space multi_val::get_space() const {
  auto res = isl_multi_val_get_space(get());
  return manage(res);
}

isl::id multi_val::get_tuple_id(isl::dim type) const {
  auto res = isl_multi_val_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

std::string multi_val::get_tuple_name(isl::dim type) const {
  auto res = isl_multi_val_get_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  std::string tmp(res);
  return tmp;
}

isl::val multi_val::get_val(int pos) const {
  auto res = isl_multi_val_get_val(get(), pos);
  return manage(res);
}

isl::boolean multi_val::has_tuple_id(isl::dim type) const {
  auto res = isl_multi_val_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::multi_val multi_val::insert_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_multi_val_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::boolean multi_val::involves_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_multi_val_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::boolean multi_val::involves_nan() const {
  auto res = isl_multi_val_involves_nan(get());
  return manage(res);
}

isl::multi_val multi_val::mod_multi_val(isl::multi_val mv) const {
  auto res = isl_multi_val_mod_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_val multi_val::mod_val(isl::val v) const {
  auto res = isl_multi_val_mod_val(copy(), v.release());
  return manage(res);
}

isl::multi_val multi_val::neg() const {
  auto res = isl_multi_val_neg(copy());
  return manage(res);
}

isl::boolean multi_val::plain_is_equal(const isl::multi_val &multi2) const {
  auto res = isl_multi_val_plain_is_equal(get(), multi2.get());
  return manage(res);
}

isl::multi_val multi_val::product(isl::multi_val multi2) const {
  auto res = isl_multi_val_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_val multi_val::range_factor_domain() const {
  auto res = isl_multi_val_range_factor_domain(copy());
  return manage(res);
}

isl::multi_val multi_val::range_factor_range() const {
  auto res = isl_multi_val_range_factor_range(copy());
  return manage(res);
}

isl::boolean multi_val::range_is_wrapping() const {
  auto res = isl_multi_val_range_is_wrapping(get());
  return manage(res);
}

isl::multi_val multi_val::range_product(isl::multi_val multi2) const {
  auto res = isl_multi_val_range_product(copy(), multi2.release());
  return manage(res);
}

isl::multi_val multi_val::range_splice(unsigned int pos, isl::multi_val multi2) const {
  auto res = isl_multi_val_range_splice(copy(), pos, multi2.release());
  return manage(res);
}

isl::multi_val multi_val::reset_tuple_id(isl::dim type) const {
  auto res = isl_multi_val_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::multi_val multi_val::reset_user() const {
  auto res = isl_multi_val_reset_user(copy());
  return manage(res);
}

isl::multi_val multi_val::scale_down_multi_val(isl::multi_val mv) const {
  auto res = isl_multi_val_scale_down_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_val multi_val::scale_down_val(isl::val v) const {
  auto res = isl_multi_val_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::multi_val multi_val::scale_multi_val(isl::multi_val mv) const {
  auto res = isl_multi_val_scale_multi_val(copy(), mv.release());
  return manage(res);
}

isl::multi_val multi_val::scale_val(isl::val v) const {
  auto res = isl_multi_val_scale_val(copy(), v.release());
  return manage(res);
}

isl::multi_val multi_val::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const {
  auto res = isl_multi_val_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::multi_val multi_val::set_tuple_id(isl::dim type, isl::id id) const {
  auto res = isl_multi_val_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::multi_val multi_val::set_tuple_name(isl::dim type, const std::string &s) const {
  auto res = isl_multi_val_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

isl::multi_val multi_val::set_val(int pos, isl::val el) const {
  auto res = isl_multi_val_set_val(copy(), pos, el.release());
  return manage(res);
}

isl::multi_val multi_val::splice(unsigned int in_pos, unsigned int out_pos, isl::multi_val multi2) const {
  auto res = isl_multi_val_splice(copy(), in_pos, out_pos, multi2.release());
  return manage(res);
}

isl::multi_val multi_val::sub(isl::multi_val multi2) const {
  auto res = isl_multi_val_sub(copy(), multi2.release());
  return manage(res);
}

isl::multi_val multi_val::zero(isl::space space) {
  auto res = isl_multi_val_zero(space.release());
  return manage(res);
}

// implementations for isl::point
isl::point manage(__isl_take isl_point *ptr) {
  return point(ptr);
}
isl::point give(__isl_take isl_point *ptr) {
  return manage(ptr);
}


point::point()
    : ptr(nullptr) {}

point::point(const isl::point &obj)
    : ptr(obj.copy()) {}
point::point(std::nullptr_t)
    : ptr(nullptr) {}


point::point(__isl_take isl_point *ptr)
    : ptr(ptr) {}

point::point(isl::space dim) {
  auto res = isl_point_zero(dim.release());
  ptr = res;
}

point &point::operator=(isl::point obj) {
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
__isl_keep isl_point *point::keep() const {
  return get();
}

__isl_give isl_point *point::take() {
  return release();
}

point::operator bool() const {
  return !is_null();
}

isl::ctx point::get_ctx() const {
  return isl::ctx(isl_point_get_ctx(ptr));
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


isl::point point::add_ui(isl::dim type, int pos, unsigned int val) const {
  auto res = isl_point_add_ui(copy(), static_cast<enum isl_dim_type>(type), pos, val);
  return manage(res);
}

isl::val point::get_coordinate_val(isl::dim type, int pos) const {
  auto res = isl_point_get_coordinate_val(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::space point::get_space() const {
  auto res = isl_point_get_space(get());
  return manage(res);
}

isl::point point::set_coordinate_val(isl::dim type, int pos, isl::val v) const {
  auto res = isl_point_set_coordinate_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

isl::point point::sub_ui(isl::dim type, int pos, unsigned int val) const {
  auto res = isl_point_sub_ui(copy(), static_cast<enum isl_dim_type>(type), pos, val);
  return manage(res);
}

// implementations for isl::pw_aff
isl::pw_aff manage(__isl_take isl_pw_aff *ptr) {
  return pw_aff(ptr);
}
isl::pw_aff give(__isl_take isl_pw_aff *ptr) {
  return manage(ptr);
}


pw_aff::pw_aff()
    : ptr(nullptr) {}

pw_aff::pw_aff(const isl::pw_aff &obj)
    : ptr(obj.copy()) {}
pw_aff::pw_aff(std::nullptr_t)
    : ptr(nullptr) {}


pw_aff::pw_aff(__isl_take isl_pw_aff *ptr)
    : ptr(ptr) {}

pw_aff::pw_aff(isl::aff aff) {
  auto res = isl_pw_aff_from_aff(aff.release());
  ptr = res;
}
pw_aff::pw_aff(isl::local_space ls) {
  auto res = isl_pw_aff_zero_on_domain(ls.release());
  ptr = res;
}
pw_aff::pw_aff(isl::set domain, isl::val v) {
  auto res = isl_pw_aff_val_on_domain(domain.release(), v.release());
  ptr = res;
}
pw_aff::pw_aff(isl::ctx ctx, const std::string &str) {
  auto res = isl_pw_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

pw_aff &pw_aff::operator=(isl::pw_aff obj) {
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
__isl_keep isl_pw_aff *pw_aff::keep() const {
  return get();
}

__isl_give isl_pw_aff *pw_aff::take() {
  return release();
}

pw_aff::operator bool() const {
  return !is_null();
}

isl::ctx pw_aff::get_ctx() const {
  return isl::ctx(isl_pw_aff_get_ctx(ptr));
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


isl::pw_aff pw_aff::add(isl::pw_aff pwaff2) const {
  auto res = isl_pw_aff_add(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::add_dims(isl::dim type, unsigned int n) const {
  auto res = isl_pw_aff_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::pw_aff pw_aff::align_params(isl::space model) const {
  auto res = isl_pw_aff_align_params(copy(), model.release());
  return manage(res);
}

isl::pw_aff pw_aff::alloc(isl::set set, isl::aff aff) {
  auto res = isl_pw_aff_alloc(set.release(), aff.release());
  return manage(res);
}

isl::pw_aff pw_aff::ceil() const {
  auto res = isl_pw_aff_ceil(copy());
  return manage(res);
}

isl::pw_aff pw_aff::coalesce() const {
  auto res = isl_pw_aff_coalesce(copy());
  return manage(res);
}

isl::pw_aff pw_aff::cond(isl::pw_aff pwaff_true, isl::pw_aff pwaff_false) const {
  auto res = isl_pw_aff_cond(copy(), pwaff_true.release(), pwaff_false.release());
  return manage(res);
}

unsigned int pw_aff::dim(isl::dim type) const {
  auto res = isl_pw_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::pw_aff pw_aff::div(isl::pw_aff pa2) const {
  auto res = isl_pw_aff_div(copy(), pa2.release());
  return manage(res);
}

isl::set pw_aff::domain() const {
  auto res = isl_pw_aff_domain(copy());
  return manage(res);
}

isl::pw_aff pw_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_pw_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::pw_aff pw_aff::empty(isl::space dim) {
  auto res = isl_pw_aff_empty(dim.release());
  return manage(res);
}

isl::map pw_aff::eq_map(isl::pw_aff pa2) const {
  auto res = isl_pw_aff_eq_map(copy(), pa2.release());
  return manage(res);
}

isl::set pw_aff::eq_set(isl::pw_aff pwaff2) const {
  auto res = isl_pw_aff_eq_set(copy(), pwaff2.release());
  return manage(res);
}

int pw_aff::find_dim_by_name(isl::dim type, const std::string &name) const {
  auto res = isl_pw_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::pw_aff pw_aff::floor() const {
  auto res = isl_pw_aff_floor(copy());
  return manage(res);
}

isl::stat pw_aff::foreach_piece(const std::function<isl::stat(isl::set, isl::aff)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_set *arg_0, isl_aff *arg_1, void *arg_2) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::set, isl::aff)> **>(arg_2);
    stat ret = (*func)(isl::manage(arg_0), isl::manage(arg_1));
    return isl_stat(ret);
  };
  auto res = isl_pw_aff_foreach_piece(get(), fn_lambda, &fn_p);
  return isl::stat(res);
}

isl::pw_aff pw_aff::from_range() const {
  auto res = isl_pw_aff_from_range(copy());
  return manage(res);
}

isl::set pw_aff::ge_set(isl::pw_aff pwaff2) const {
  auto res = isl_pw_aff_ge_set(copy(), pwaff2.release());
  return manage(res);
}

isl::id pw_aff::get_dim_id(isl::dim type, unsigned int pos) const {
  auto res = isl_pw_aff_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

std::string pw_aff::get_dim_name(isl::dim type, unsigned int pos) const {
  auto res = isl_pw_aff_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

isl::space pw_aff::get_domain_space() const {
  auto res = isl_pw_aff_get_domain_space(get());
  return manage(res);
}

uint32_t pw_aff::get_hash() const {
  auto res = isl_pw_aff_get_hash(get());
  return res;
}

isl::space pw_aff::get_space() const {
  auto res = isl_pw_aff_get_space(get());
  return manage(res);
}

isl::id pw_aff::get_tuple_id(isl::dim type) const {
  auto res = isl_pw_aff_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::pw_aff pw_aff::gist(isl::set context) const {
  auto res = isl_pw_aff_gist(copy(), context.release());
  return manage(res);
}

isl::pw_aff pw_aff::gist_params(isl::set context) const {
  auto res = isl_pw_aff_gist_params(copy(), context.release());
  return manage(res);
}

isl::map pw_aff::gt_map(isl::pw_aff pa2) const {
  auto res = isl_pw_aff_gt_map(copy(), pa2.release());
  return manage(res);
}

isl::set pw_aff::gt_set(isl::pw_aff pwaff2) const {
  auto res = isl_pw_aff_gt_set(copy(), pwaff2.release());
  return manage(res);
}

isl::boolean pw_aff::has_dim_id(isl::dim type, unsigned int pos) const {
  auto res = isl_pw_aff_has_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::boolean pw_aff::has_tuple_id(isl::dim type) const {
  auto res = isl_pw_aff_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::pw_aff pw_aff::insert_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_pw_aff_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::pw_aff pw_aff::intersect_domain(isl::set set) const {
  auto res = isl_pw_aff_intersect_domain(copy(), set.release());
  return manage(res);
}

isl::pw_aff pw_aff::intersect_params(isl::set set) const {
  auto res = isl_pw_aff_intersect_params(copy(), set.release());
  return manage(res);
}

isl::boolean pw_aff::involves_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_pw_aff_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::boolean pw_aff::involves_nan() const {
  auto res = isl_pw_aff_involves_nan(get());
  return manage(res);
}

isl::boolean pw_aff::is_cst() const {
  auto res = isl_pw_aff_is_cst(get());
  return manage(res);
}

isl::boolean pw_aff::is_empty() const {
  auto res = isl_pw_aff_is_empty(get());
  return manage(res);
}

isl::boolean pw_aff::is_equal(const isl::pw_aff &pa2) const {
  auto res = isl_pw_aff_is_equal(get(), pa2.get());
  return manage(res);
}

isl::set pw_aff::le_set(isl::pw_aff pwaff2) const {
  auto res = isl_pw_aff_le_set(copy(), pwaff2.release());
  return manage(res);
}

isl::map pw_aff::lt_map(isl::pw_aff pa2) const {
  auto res = isl_pw_aff_lt_map(copy(), pa2.release());
  return manage(res);
}

isl::set pw_aff::lt_set(isl::pw_aff pwaff2) const {
  auto res = isl_pw_aff_lt_set(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::max(isl::pw_aff pwaff2) const {
  auto res = isl_pw_aff_max(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::min(isl::pw_aff pwaff2) const {
  auto res = isl_pw_aff_min(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::mod(isl::val mod) const {
  auto res = isl_pw_aff_mod_val(copy(), mod.release());
  return manage(res);
}

isl::pw_aff pw_aff::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const {
  auto res = isl_pw_aff_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl::pw_aff pw_aff::mul(isl::pw_aff pwaff2) const {
  auto res = isl_pw_aff_mul(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::nan_on_domain(isl::local_space ls) {
  auto res = isl_pw_aff_nan_on_domain(ls.release());
  return manage(res);
}

isl::set pw_aff::ne_set(isl::pw_aff pwaff2) const {
  auto res = isl_pw_aff_ne_set(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::neg() const {
  auto res = isl_pw_aff_neg(copy());
  return manage(res);
}

isl::set pw_aff::non_zero_set() const {
  auto res = isl_pw_aff_non_zero_set(copy());
  return manage(res);
}

isl::set pw_aff::nonneg_set() const {
  auto res = isl_pw_aff_nonneg_set(copy());
  return manage(res);
}

isl::set pw_aff::params() const {
  auto res = isl_pw_aff_params(copy());
  return manage(res);
}

int pw_aff::plain_cmp(const isl::pw_aff &pa2) const {
  auto res = isl_pw_aff_plain_cmp(get(), pa2.get());
  return res;
}

isl::boolean pw_aff::plain_is_equal(const isl::pw_aff &pwaff2) const {
  auto res = isl_pw_aff_plain_is_equal(get(), pwaff2.get());
  return manage(res);
}

isl::set pw_aff::pos_set() const {
  auto res = isl_pw_aff_pos_set(copy());
  return manage(res);
}

isl::pw_aff pw_aff::project_domain_on_params() const {
  auto res = isl_pw_aff_project_domain_on_params(copy());
  return manage(res);
}

isl::pw_aff pw_aff::pullback(isl::multi_aff ma) const {
  auto res = isl_pw_aff_pullback_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::pw_aff pw_aff::pullback(isl::pw_multi_aff pma) const {
  auto res = isl_pw_aff_pullback_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::pw_aff pw_aff::pullback(isl::multi_pw_aff mpa) const {
  auto res = isl_pw_aff_pullback_multi_pw_aff(copy(), mpa.release());
  return manage(res);
}

isl::pw_aff pw_aff::reset_tuple_id(isl::dim type) const {
  auto res = isl_pw_aff_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::pw_aff pw_aff::reset_user() const {
  auto res = isl_pw_aff_reset_user(copy());
  return manage(res);
}

isl::pw_aff pw_aff::scale(isl::val v) const {
  auto res = isl_pw_aff_scale_val(copy(), v.release());
  return manage(res);
}

isl::pw_aff pw_aff::scale_down(isl::val f) const {
  auto res = isl_pw_aff_scale_down_val(copy(), f.release());
  return manage(res);
}

isl::pw_aff pw_aff::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const {
  auto res = isl_pw_aff_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::pw_aff pw_aff::set_tuple_id(isl::dim type, isl::id id) const {
  auto res = isl_pw_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::pw_aff pw_aff::sub(isl::pw_aff pwaff2) const {
  auto res = isl_pw_aff_sub(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::subtract_domain(isl::set set) const {
  auto res = isl_pw_aff_subtract_domain(copy(), set.release());
  return manage(res);
}

isl::pw_aff pw_aff::tdiv_q(isl::pw_aff pa2) const {
  auto res = isl_pw_aff_tdiv_q(copy(), pa2.release());
  return manage(res);
}

isl::pw_aff pw_aff::tdiv_r(isl::pw_aff pa2) const {
  auto res = isl_pw_aff_tdiv_r(copy(), pa2.release());
  return manage(res);
}

isl::pw_aff pw_aff::union_add(isl::pw_aff pwaff2) const {
  auto res = isl_pw_aff_union_add(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::union_max(isl::pw_aff pwaff2) const {
  auto res = isl_pw_aff_union_max(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::union_min(isl::pw_aff pwaff2) const {
  auto res = isl_pw_aff_union_min(copy(), pwaff2.release());
  return manage(res);
}

isl::pw_aff pw_aff::var_on_domain(isl::local_space ls, isl::dim type, unsigned int pos) {
  auto res = isl_pw_aff_var_on_domain(ls.release(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::set pw_aff::zero_set() const {
  auto res = isl_pw_aff_zero_set(copy());
  return manage(res);
}

// implementations for isl::pw_aff_list
isl::pw_aff_list manage(__isl_take isl_pw_aff_list *ptr) {
  return pw_aff_list(ptr);
}
isl::pw_aff_list give(__isl_take isl_pw_aff_list *ptr) {
  return manage(ptr);
}


pw_aff_list::pw_aff_list()
    : ptr(nullptr) {}

pw_aff_list::pw_aff_list(const isl::pw_aff_list &obj)
    : ptr(obj.copy()) {}
pw_aff_list::pw_aff_list(std::nullptr_t)
    : ptr(nullptr) {}


pw_aff_list::pw_aff_list(__isl_take isl_pw_aff_list *ptr)
    : ptr(ptr) {}


pw_aff_list &pw_aff_list::operator=(isl::pw_aff_list obj) {
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
__isl_keep isl_pw_aff_list *pw_aff_list::keep() const {
  return get();
}

__isl_give isl_pw_aff_list *pw_aff_list::take() {
  return release();
}

pw_aff_list::operator bool() const {
  return !is_null();
}

isl::ctx pw_aff_list::get_ctx() const {
  return isl::ctx(isl_pw_aff_list_get_ctx(ptr));
}



void pw_aff_list::dump() const {
  isl_pw_aff_list_dump(get());
}



// implementations for isl::pw_multi_aff
isl::pw_multi_aff manage(__isl_take isl_pw_multi_aff *ptr) {
  return pw_multi_aff(ptr);
}
isl::pw_multi_aff give(__isl_take isl_pw_multi_aff *ptr) {
  return manage(ptr);
}


pw_multi_aff::pw_multi_aff()
    : ptr(nullptr) {}

pw_multi_aff::pw_multi_aff(const isl::pw_multi_aff &obj)
    : ptr(obj.copy()) {}
pw_multi_aff::pw_multi_aff(std::nullptr_t)
    : ptr(nullptr) {}


pw_multi_aff::pw_multi_aff(__isl_take isl_pw_multi_aff *ptr)
    : ptr(ptr) {}

pw_multi_aff::pw_multi_aff(isl::multi_aff ma) {
  auto res = isl_pw_multi_aff_from_multi_aff(ma.release());
  ptr = res;
}
pw_multi_aff::pw_multi_aff(isl::pw_aff pa) {
  auto res = isl_pw_multi_aff_from_pw_aff(pa.release());
  ptr = res;
}
pw_multi_aff::pw_multi_aff(isl::ctx ctx, const std::string &str) {
  auto res = isl_pw_multi_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

pw_multi_aff &pw_multi_aff::operator=(isl::pw_multi_aff obj) {
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
__isl_keep isl_pw_multi_aff *pw_multi_aff::keep() const {
  return get();
}

__isl_give isl_pw_multi_aff *pw_multi_aff::take() {
  return release();
}

pw_multi_aff::operator bool() const {
  return !is_null();
}

isl::ctx pw_multi_aff::get_ctx() const {
  return isl::ctx(isl_pw_multi_aff_get_ctx(ptr));
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


isl::pw_multi_aff pw_multi_aff::add(isl::pw_multi_aff pma2) const {
  auto res = isl_pw_multi_aff_add(copy(), pma2.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::align_params(isl::space model) const {
  auto res = isl_pw_multi_aff_align_params(copy(), model.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::alloc(isl::set set, isl::multi_aff maff) {
  auto res = isl_pw_multi_aff_alloc(set.release(), maff.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::coalesce() const {
  auto res = isl_pw_multi_aff_coalesce(copy());
  return manage(res);
}

unsigned int pw_multi_aff::dim(isl::dim type) const {
  auto res = isl_pw_multi_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::set pw_multi_aff::domain() const {
  auto res = isl_pw_multi_aff_domain(copy());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_pw_multi_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::empty(isl::space space) {
  auto res = isl_pw_multi_aff_empty(space.release());
  return manage(res);
}

int pw_multi_aff::find_dim_by_name(isl::dim type, const std::string &name) const {
  auto res = isl_pw_multi_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::pw_multi_aff pw_multi_aff::fix_si(isl::dim type, unsigned int pos, int value) const {
  auto res = isl_pw_multi_aff_fix_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::flat_range_product(isl::pw_multi_aff pma2) const {
  auto res = isl_pw_multi_aff_flat_range_product(copy(), pma2.release());
  return manage(res);
}

isl::stat pw_multi_aff::foreach_piece(const std::function<isl::stat(isl::set, isl::multi_aff)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_set *arg_0, isl_multi_aff *arg_1, void *arg_2) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::set, isl::multi_aff)> **>(arg_2);
    stat ret = (*func)(isl::manage(arg_0), isl::manage(arg_1));
    return isl_stat(ret);
  };
  auto res = isl_pw_multi_aff_foreach_piece(get(), fn_lambda, &fn_p);
  return isl::stat(res);
}

isl::pw_multi_aff pw_multi_aff::from_domain(isl::set set) {
  auto res = isl_pw_multi_aff_from_domain(set.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::from_map(isl::map map) {
  auto res = isl_pw_multi_aff_from_map(map.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::from_multi_pw_aff(isl::multi_pw_aff mpa) {
  auto res = isl_pw_multi_aff_from_multi_pw_aff(mpa.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::from_set(isl::set set) {
  auto res = isl_pw_multi_aff_from_set(set.release());
  return manage(res);
}

isl::id pw_multi_aff::get_dim_id(isl::dim type, unsigned int pos) const {
  auto res = isl_pw_multi_aff_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

std::string pw_multi_aff::get_dim_name(isl::dim type, unsigned int pos) const {
  auto res = isl_pw_multi_aff_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

isl::space pw_multi_aff::get_domain_space() const {
  auto res = isl_pw_multi_aff_get_domain_space(get());
  return manage(res);
}

isl::pw_aff pw_multi_aff::get_pw_aff(int pos) const {
  auto res = isl_pw_multi_aff_get_pw_aff(get(), pos);
  return manage(res);
}

isl::space pw_multi_aff::get_space() const {
  auto res = isl_pw_multi_aff_get_space(get());
  return manage(res);
}

isl::id pw_multi_aff::get_tuple_id(isl::dim type) const {
  auto res = isl_pw_multi_aff_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

std::string pw_multi_aff::get_tuple_name(isl::dim type) const {
  auto res = isl_pw_multi_aff_get_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  std::string tmp(res);
  return tmp;
}

isl::pw_multi_aff pw_multi_aff::gist(isl::set set) const {
  auto res = isl_pw_multi_aff_gist(copy(), set.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::gist_params(isl::set set) const {
  auto res = isl_pw_multi_aff_gist_params(copy(), set.release());
  return manage(res);
}

isl::boolean pw_multi_aff::has_tuple_id(isl::dim type) const {
  auto res = isl_pw_multi_aff_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::boolean pw_multi_aff::has_tuple_name(isl::dim type) const {
  auto res = isl_pw_multi_aff_has_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::identity(isl::space space) {
  auto res = isl_pw_multi_aff_identity(space.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::intersect_domain(isl::set set) const {
  auto res = isl_pw_multi_aff_intersect_domain(copy(), set.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::intersect_params(isl::set set) const {
  auto res = isl_pw_multi_aff_intersect_params(copy(), set.release());
  return manage(res);
}

isl::boolean pw_multi_aff::involves_nan() const {
  auto res = isl_pw_multi_aff_involves_nan(get());
  return manage(res);
}

isl::boolean pw_multi_aff::is_equal(const isl::pw_multi_aff &pma2) const {
  auto res = isl_pw_multi_aff_is_equal(get(), pma2.get());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::multi_val_on_domain(isl::set domain, isl::multi_val mv) {
  auto res = isl_pw_multi_aff_multi_val_on_domain(domain.release(), mv.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::neg() const {
  auto res = isl_pw_multi_aff_neg(copy());
  return manage(res);
}

isl::boolean pw_multi_aff::plain_is_equal(const isl::pw_multi_aff &pma2) const {
  auto res = isl_pw_multi_aff_plain_is_equal(get(), pma2.get());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::product(isl::pw_multi_aff pma2) const {
  auto res = isl_pw_multi_aff_product(copy(), pma2.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::project_domain_on_params() const {
  auto res = isl_pw_multi_aff_project_domain_on_params(copy());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::project_out_map(isl::space space, isl::dim type, unsigned int first, unsigned int n) {
  auto res = isl_pw_multi_aff_project_out_map(space.release(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::pullback(isl::multi_aff ma) const {
  auto res = isl_pw_multi_aff_pullback_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::pullback(isl::pw_multi_aff pma2) const {
  auto res = isl_pw_multi_aff_pullback_pw_multi_aff(copy(), pma2.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::range_map(isl::space space) {
  auto res = isl_pw_multi_aff_range_map(space.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::range_product(isl::pw_multi_aff pma2) const {
  auto res = isl_pw_multi_aff_range_product(copy(), pma2.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::reset_tuple_id(isl::dim type) const {
  auto res = isl_pw_multi_aff_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::reset_user() const {
  auto res = isl_pw_multi_aff_reset_user(copy());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::scale_down_val(isl::val v) const {
  auto res = isl_pw_multi_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::scale_multi_val(isl::multi_val mv) const {
  auto res = isl_pw_multi_aff_scale_multi_val(copy(), mv.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::scale_val(isl::val v) const {
  auto res = isl_pw_multi_aff_scale_val(copy(), v.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const {
  auto res = isl_pw_multi_aff_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::set_pw_aff(unsigned int pos, isl::pw_aff pa) const {
  auto res = isl_pw_multi_aff_set_pw_aff(copy(), pos, pa.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::set_tuple_id(isl::dim type, isl::id id) const {
  auto res = isl_pw_multi_aff_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::sub(isl::pw_multi_aff pma2) const {
  auto res = isl_pw_multi_aff_sub(copy(), pma2.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::subtract_domain(isl::set set) const {
  auto res = isl_pw_multi_aff_subtract_domain(copy(), set.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::union_add(isl::pw_multi_aff pma2) const {
  auto res = isl_pw_multi_aff_union_add(copy(), pma2.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::union_lexmax(isl::pw_multi_aff pma2) const {
  auto res = isl_pw_multi_aff_union_lexmax(copy(), pma2.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::union_lexmin(isl::pw_multi_aff pma2) const {
  auto res = isl_pw_multi_aff_union_lexmin(copy(), pma2.release());
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::zero(isl::space space) {
  auto res = isl_pw_multi_aff_zero(space.release());
  return manage(res);
}

// implementations for isl::pw_qpolynomial
isl::pw_qpolynomial manage(__isl_take isl_pw_qpolynomial *ptr) {
  return pw_qpolynomial(ptr);
}
isl::pw_qpolynomial give(__isl_take isl_pw_qpolynomial *ptr) {
  return manage(ptr);
}


pw_qpolynomial::pw_qpolynomial()
    : ptr(nullptr) {}

pw_qpolynomial::pw_qpolynomial(const isl::pw_qpolynomial &obj)
    : ptr(obj.copy()) {}
pw_qpolynomial::pw_qpolynomial(std::nullptr_t)
    : ptr(nullptr) {}


pw_qpolynomial::pw_qpolynomial(__isl_take isl_pw_qpolynomial *ptr)
    : ptr(ptr) {}

pw_qpolynomial::pw_qpolynomial(isl::ctx ctx, const std::string &str) {
  auto res = isl_pw_qpolynomial_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

pw_qpolynomial &pw_qpolynomial::operator=(isl::pw_qpolynomial obj) {
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
__isl_keep isl_pw_qpolynomial *pw_qpolynomial::keep() const {
  return get();
}

__isl_give isl_pw_qpolynomial *pw_qpolynomial::take() {
  return release();
}

pw_qpolynomial::operator bool() const {
  return !is_null();
}

isl::ctx pw_qpolynomial::get_ctx() const {
  return isl::ctx(isl_pw_qpolynomial_get_ctx(ptr));
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


isl::pw_qpolynomial pw_qpolynomial::add(isl::pw_qpolynomial pwqp2) const {
  auto res = isl_pw_qpolynomial_add(copy(), pwqp2.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::add_dims(isl::dim type, unsigned int n) const {
  auto res = isl_pw_qpolynomial_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::alloc(isl::set set, isl::qpolynomial qp) {
  auto res = isl_pw_qpolynomial_alloc(set.release(), qp.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::coalesce() const {
  auto res = isl_pw_qpolynomial_coalesce(copy());
  return manage(res);
}

unsigned int pw_qpolynomial::dim(isl::dim type) const {
  auto res = isl_pw_qpolynomial_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::set pw_qpolynomial::domain() const {
  auto res = isl_pw_qpolynomial_domain(copy());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::drop_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_pw_qpolynomial_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::val pw_qpolynomial::eval(isl::point pnt) const {
  auto res = isl_pw_qpolynomial_eval(copy(), pnt.release());
  return manage(res);
}

int pw_qpolynomial::find_dim_by_name(isl::dim type, const std::string &name) const {
  auto res = isl_pw_qpolynomial_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::pw_qpolynomial pw_qpolynomial::fix_val(isl::dim type, unsigned int n, isl::val v) const {
  auto res = isl_pw_qpolynomial_fix_val(copy(), static_cast<enum isl_dim_type>(type), n, v.release());
  return manage(res);
}

isl::stat pw_qpolynomial::foreach_piece(const std::function<isl::stat(isl::set, isl::qpolynomial)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_set *arg_0, isl_qpolynomial *arg_1, void *arg_2) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::set, isl::qpolynomial)> **>(arg_2);
    stat ret = (*func)(isl::manage(arg_0), isl::manage(arg_1));
    return isl_stat(ret);
  };
  auto res = isl_pw_qpolynomial_foreach_piece(get(), fn_lambda, &fn_p);
  return isl::stat(res);
}

isl::pw_qpolynomial pw_qpolynomial::from_pw_aff(isl::pw_aff pwaff) {
  auto res = isl_pw_qpolynomial_from_pw_aff(pwaff.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::from_qpolynomial(isl::qpolynomial qp) {
  auto res = isl_pw_qpolynomial_from_qpolynomial(qp.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::from_range() const {
  auto res = isl_pw_qpolynomial_from_range(copy());
  return manage(res);
}

isl::space pw_qpolynomial::get_domain_space() const {
  auto res = isl_pw_qpolynomial_get_domain_space(get());
  return manage(res);
}

isl::space pw_qpolynomial::get_space() const {
  auto res = isl_pw_qpolynomial_get_space(get());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::gist(isl::set context) const {
  auto res = isl_pw_qpolynomial_gist(copy(), context.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::gist_params(isl::set context) const {
  auto res = isl_pw_qpolynomial_gist_params(copy(), context.release());
  return manage(res);
}

isl::boolean pw_qpolynomial::has_equal_space(const isl::pw_qpolynomial &pwqp2) const {
  auto res = isl_pw_qpolynomial_has_equal_space(get(), pwqp2.get());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::insert_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_pw_qpolynomial_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::intersect_domain(isl::set set) const {
  auto res = isl_pw_qpolynomial_intersect_domain(copy(), set.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::intersect_params(isl::set set) const {
  auto res = isl_pw_qpolynomial_intersect_params(copy(), set.release());
  return manage(res);
}

isl::boolean pw_qpolynomial::involves_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_pw_qpolynomial_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::boolean pw_qpolynomial::involves_nan() const {
  auto res = isl_pw_qpolynomial_involves_nan(get());
  return manage(res);
}

isl::boolean pw_qpolynomial::is_zero() const {
  auto res = isl_pw_qpolynomial_is_zero(get());
  return manage(res);
}

isl::val pw_qpolynomial::max() const {
  auto res = isl_pw_qpolynomial_max(copy());
  return manage(res);
}

isl::val pw_qpolynomial::min() const {
  auto res = isl_pw_qpolynomial_min(copy());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const {
  auto res = isl_pw_qpolynomial_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::mul(isl::pw_qpolynomial pwqp2) const {
  auto res = isl_pw_qpolynomial_mul(copy(), pwqp2.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::neg() const {
  auto res = isl_pw_qpolynomial_neg(copy());
  return manage(res);
}

isl::boolean pw_qpolynomial::plain_is_equal(const isl::pw_qpolynomial &pwqp2) const {
  auto res = isl_pw_qpolynomial_plain_is_equal(get(), pwqp2.get());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::pow(unsigned int exponent) const {
  auto res = isl_pw_qpolynomial_pow(copy(), exponent);
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::project_domain_on_params() const {
  auto res = isl_pw_qpolynomial_project_domain_on_params(copy());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::reset_domain_space(isl::space dim) const {
  auto res = isl_pw_qpolynomial_reset_domain_space(copy(), dim.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::reset_user() const {
  auto res = isl_pw_qpolynomial_reset_user(copy());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::scale_down_val(isl::val v) const {
  auto res = isl_pw_qpolynomial_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::scale_val(isl::val v) const {
  auto res = isl_pw_qpolynomial_scale_val(copy(), v.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::split_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_pw_qpolynomial_split_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::split_periods(int max_periods) const {
  auto res = isl_pw_qpolynomial_split_periods(copy(), max_periods);
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::sub(isl::pw_qpolynomial pwqp2) const {
  auto res = isl_pw_qpolynomial_sub(copy(), pwqp2.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::subtract_domain(isl::set set) const {
  auto res = isl_pw_qpolynomial_subtract_domain(copy(), set.release());
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::to_polynomial(int sign) const {
  auto res = isl_pw_qpolynomial_to_polynomial(copy(), sign);
  return manage(res);
}

isl::pw_qpolynomial pw_qpolynomial::zero(isl::space dim) {
  auto res = isl_pw_qpolynomial_zero(dim.release());
  return manage(res);
}

// implementations for isl::qpolynomial
isl::qpolynomial manage(__isl_take isl_qpolynomial *ptr) {
  return qpolynomial(ptr);
}
isl::qpolynomial give(__isl_take isl_qpolynomial *ptr) {
  return manage(ptr);
}


qpolynomial::qpolynomial()
    : ptr(nullptr) {}

qpolynomial::qpolynomial(const isl::qpolynomial &obj)
    : ptr(obj.copy()) {}
qpolynomial::qpolynomial(std::nullptr_t)
    : ptr(nullptr) {}


qpolynomial::qpolynomial(__isl_take isl_qpolynomial *ptr)
    : ptr(ptr) {}


qpolynomial &qpolynomial::operator=(isl::qpolynomial obj) {
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
__isl_keep isl_qpolynomial *qpolynomial::keep() const {
  return get();
}

__isl_give isl_qpolynomial *qpolynomial::take() {
  return release();
}

qpolynomial::operator bool() const {
  return !is_null();
}

isl::ctx qpolynomial::get_ctx() const {
  return isl::ctx(isl_qpolynomial_get_ctx(ptr));
}



void qpolynomial::dump() const {
  isl_qpolynomial_dump(get());
}


isl::qpolynomial qpolynomial::add(isl::qpolynomial qp2) const {
  auto res = isl_qpolynomial_add(copy(), qp2.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::add_dims(isl::dim type, unsigned int n) const {
  auto res = isl_qpolynomial_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::qpolynomial qpolynomial::align_params(isl::space model) const {
  auto res = isl_qpolynomial_align_params(copy(), model.release());
  return manage(res);
}

isl::stat qpolynomial::as_polynomial_on_domain(const isl::basic_set &bset, const std::function<isl::stat(isl::basic_set, isl::qpolynomial)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_basic_set *arg_0, isl_qpolynomial *arg_1, void *arg_2) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::basic_set, isl::qpolynomial)> **>(arg_2);
    stat ret = (*func)(isl::manage(arg_0), isl::manage(arg_1));
    return isl_stat(ret);
  };
  auto res = isl_qpolynomial_as_polynomial_on_domain(get(), bset.get(), fn_lambda, &fn_p);
  return isl::stat(res);
}

unsigned int qpolynomial::dim(isl::dim type) const {
  auto res = isl_qpolynomial_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::qpolynomial qpolynomial::drop_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_qpolynomial_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::val qpolynomial::eval(isl::point pnt) const {
  auto res = isl_qpolynomial_eval(copy(), pnt.release());
  return manage(res);
}

isl::stat qpolynomial::foreach_term(const std::function<isl::stat(isl::term)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_term *arg_0, void *arg_1) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::term)> **>(arg_1);
    stat ret = (*func)(isl::manage(arg_0));
    return isl_stat(ret);
  };
  auto res = isl_qpolynomial_foreach_term(get(), fn_lambda, &fn_p);
  return isl::stat(res);
}

isl::qpolynomial qpolynomial::from_aff(isl::aff aff) {
  auto res = isl_qpolynomial_from_aff(aff.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::from_constraint(isl::constraint c, isl::dim type, unsigned int pos) {
  auto res = isl_qpolynomial_from_constraint(c.release(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::qpolynomial qpolynomial::from_term(isl::term term) {
  auto res = isl_qpolynomial_from_term(term.release());
  return manage(res);
}

isl::val qpolynomial::get_constant_val() const {
  auto res = isl_qpolynomial_get_constant_val(get());
  return manage(res);
}

isl::space qpolynomial::get_domain_space() const {
  auto res = isl_qpolynomial_get_domain_space(get());
  return manage(res);
}

isl::space qpolynomial::get_space() const {
  auto res = isl_qpolynomial_get_space(get());
  return manage(res);
}

isl::qpolynomial qpolynomial::gist(isl::set context) const {
  auto res = isl_qpolynomial_gist(copy(), context.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::gist_params(isl::set context) const {
  auto res = isl_qpolynomial_gist_params(copy(), context.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::homogenize() const {
  auto res = isl_qpolynomial_homogenize(copy());
  return manage(res);
}

isl::qpolynomial qpolynomial::infty_on_domain(isl::space dim) {
  auto res = isl_qpolynomial_infty_on_domain(dim.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::insert_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_qpolynomial_insert_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::boolean qpolynomial::involves_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_qpolynomial_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::boolean qpolynomial::is_infty() const {
  auto res = isl_qpolynomial_is_infty(get());
  return manage(res);
}

isl::boolean qpolynomial::is_nan() const {
  auto res = isl_qpolynomial_is_nan(get());
  return manage(res);
}

isl::boolean qpolynomial::is_neginfty() const {
  auto res = isl_qpolynomial_is_neginfty(get());
  return manage(res);
}

isl::boolean qpolynomial::is_zero() const {
  auto res = isl_qpolynomial_is_zero(get());
  return manage(res);
}

isl::qpolynomial qpolynomial::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const {
  auto res = isl_qpolynomial_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl::qpolynomial qpolynomial::mul(isl::qpolynomial qp2) const {
  auto res = isl_qpolynomial_mul(copy(), qp2.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::nan_on_domain(isl::space dim) {
  auto res = isl_qpolynomial_nan_on_domain(dim.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::neg() const {
  auto res = isl_qpolynomial_neg(copy());
  return manage(res);
}

isl::qpolynomial qpolynomial::neginfty_on_domain(isl::space dim) {
  auto res = isl_qpolynomial_neginfty_on_domain(dim.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::one_on_domain(isl::space dim) {
  auto res = isl_qpolynomial_one_on_domain(dim.release());
  return manage(res);
}

isl::boolean qpolynomial::plain_is_equal(const isl::qpolynomial &qp2) const {
  auto res = isl_qpolynomial_plain_is_equal(get(), qp2.get());
  return manage(res);
}

isl::qpolynomial qpolynomial::pow(unsigned int power) const {
  auto res = isl_qpolynomial_pow(copy(), power);
  return manage(res);
}

isl::qpolynomial qpolynomial::project_domain_on_params() const {
  auto res = isl_qpolynomial_project_domain_on_params(copy());
  return manage(res);
}

isl::qpolynomial qpolynomial::scale_down_val(isl::val v) const {
  auto res = isl_qpolynomial_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::scale_val(isl::val v) const {
  auto res = isl_qpolynomial_scale_val(copy(), v.release());
  return manage(res);
}

int qpolynomial::sgn() const {
  auto res = isl_qpolynomial_sgn(get());
  return res;
}

isl::qpolynomial qpolynomial::sub(isl::qpolynomial qp2) const {
  auto res = isl_qpolynomial_sub(copy(), qp2.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::val_on_domain(isl::space space, isl::val val) {
  auto res = isl_qpolynomial_val_on_domain(space.release(), val.release());
  return manage(res);
}

isl::qpolynomial qpolynomial::var_on_domain(isl::space dim, isl::dim type, unsigned int pos) {
  auto res = isl_qpolynomial_var_on_domain(dim.release(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::qpolynomial qpolynomial::zero_on_domain(isl::space dim) {
  auto res = isl_qpolynomial_zero_on_domain(dim.release());
  return manage(res);
}

// implementations for isl::schedule
isl::schedule manage(__isl_take isl_schedule *ptr) {
  return schedule(ptr);
}
isl::schedule give(__isl_take isl_schedule *ptr) {
  return manage(ptr);
}


schedule::schedule()
    : ptr(nullptr) {}

schedule::schedule(const isl::schedule &obj)
    : ptr(obj.copy()) {}
schedule::schedule(std::nullptr_t)
    : ptr(nullptr) {}


schedule::schedule(__isl_take isl_schedule *ptr)
    : ptr(ptr) {}

schedule::schedule(isl::ctx ctx, const std::string &str) {
  auto res = isl_schedule_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

schedule &schedule::operator=(isl::schedule obj) {
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
__isl_keep isl_schedule *schedule::keep() const {
  return get();
}

__isl_give isl_schedule *schedule::take() {
  return release();
}

schedule::operator bool() const {
  return !is_null();
}

isl::ctx schedule::get_ctx() const {
  return isl::ctx(isl_schedule_get_ctx(ptr));
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


isl::schedule schedule::align_params(isl::space space) const {
  auto res = isl_schedule_align_params(copy(), space.release());
  return manage(res);
}

isl::schedule schedule::empty(isl::space space) {
  auto res = isl_schedule_empty(space.release());
  return manage(res);
}

isl::schedule schedule::from_domain(isl::union_set domain) {
  auto res = isl_schedule_from_domain(domain.release());
  return manage(res);
}

isl::union_set schedule::get_domain() const {
  auto res = isl_schedule_get_domain(get());
  return manage(res);
}

isl::union_map schedule::get_map() const {
  auto res = isl_schedule_get_map(get());
  return manage(res);
}

isl::schedule_node schedule::get_root() const {
  auto res = isl_schedule_get_root(get());
  return manage(res);
}

isl::schedule schedule::gist_domain_params(isl::set context) const {
  auto res = isl_schedule_gist_domain_params(copy(), context.release());
  return manage(res);
}

isl::schedule schedule::insert_context(isl::set context) const {
  auto res = isl_schedule_insert_context(copy(), context.release());
  return manage(res);
}

isl::schedule schedule::insert_guard(isl::set guard) const {
  auto res = isl_schedule_insert_guard(copy(), guard.release());
  return manage(res);
}

isl::schedule schedule::insert_partial_schedule(isl::multi_union_pw_aff partial) const {
  auto res = isl_schedule_insert_partial_schedule(copy(), partial.release());
  return manage(res);
}

isl::schedule schedule::intersect_domain(isl::union_set domain) const {
  auto res = isl_schedule_intersect_domain(copy(), domain.release());
  return manage(res);
}

isl::boolean schedule::plain_is_equal(const isl::schedule &schedule2) const {
  auto res = isl_schedule_plain_is_equal(get(), schedule2.get());
  return manage(res);
}

isl::schedule schedule::pullback(isl::union_pw_multi_aff upma) const {
  auto res = isl_schedule_pullback_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

isl::schedule schedule::reset_user() const {
  auto res = isl_schedule_reset_user(copy());
  return manage(res);
}

isl::schedule schedule::sequence(isl::schedule schedule2) const {
  auto res = isl_schedule_sequence(copy(), schedule2.release());
  return manage(res);
}

isl::schedule schedule::set(isl::schedule schedule2) const {
  auto res = isl_schedule_set(copy(), schedule2.release());
  return manage(res);
}

// implementations for isl::schedule_constraints
isl::schedule_constraints manage(__isl_take isl_schedule_constraints *ptr) {
  return schedule_constraints(ptr);
}
isl::schedule_constraints give(__isl_take isl_schedule_constraints *ptr) {
  return manage(ptr);
}


schedule_constraints::schedule_constraints()
    : ptr(nullptr) {}

schedule_constraints::schedule_constraints(const isl::schedule_constraints &obj)
    : ptr(obj.copy()) {}
schedule_constraints::schedule_constraints(std::nullptr_t)
    : ptr(nullptr) {}


schedule_constraints::schedule_constraints(__isl_take isl_schedule_constraints *ptr)
    : ptr(ptr) {}

schedule_constraints::schedule_constraints(isl::ctx ctx, const std::string &str) {
  auto res = isl_schedule_constraints_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

schedule_constraints &schedule_constraints::operator=(isl::schedule_constraints obj) {
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
__isl_keep isl_schedule_constraints *schedule_constraints::keep() const {
  return get();
}

__isl_give isl_schedule_constraints *schedule_constraints::take() {
  return release();
}

schedule_constraints::operator bool() const {
  return !is_null();
}

isl::ctx schedule_constraints::get_ctx() const {
  return isl::ctx(isl_schedule_constraints_get_ctx(ptr));
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


isl::schedule_constraints schedule_constraints::apply(isl::union_map umap) const {
  auto res = isl_schedule_constraints_apply(copy(), umap.release());
  return manage(res);
}

isl::schedule schedule_constraints::compute_schedule() const {
  auto res = isl_schedule_constraints_compute_schedule(copy());
  return manage(res);
}

isl::union_map schedule_constraints::get_coincidence() const {
  auto res = isl_schedule_constraints_get_coincidence(get());
  return manage(res);
}

isl::union_map schedule_constraints::get_conditional_validity() const {
  auto res = isl_schedule_constraints_get_conditional_validity(get());
  return manage(res);
}

isl::union_map schedule_constraints::get_conditional_validity_condition() const {
  auto res = isl_schedule_constraints_get_conditional_validity_condition(get());
  return manage(res);
}

isl::set schedule_constraints::get_context() const {
  auto res = isl_schedule_constraints_get_context(get());
  return manage(res);
}

isl::union_set schedule_constraints::get_domain() const {
  auto res = isl_schedule_constraints_get_domain(get());
  return manage(res);
}

isl::union_map schedule_constraints::get_proximity() const {
  auto res = isl_schedule_constraints_get_proximity(get());
  return manage(res);
}

isl::union_map schedule_constraints::get_validity() const {
  auto res = isl_schedule_constraints_get_validity(get());
  return manage(res);
}

isl::schedule_constraints schedule_constraints::on_domain(isl::union_set domain) {
  auto res = isl_schedule_constraints_on_domain(domain.release());
  return manage(res);
}

isl::schedule_constraints schedule_constraints::set_coincidence(isl::union_map coincidence) const {
  auto res = isl_schedule_constraints_set_coincidence(copy(), coincidence.release());
  return manage(res);
}

isl::schedule_constraints schedule_constraints::set_conditional_validity(isl::union_map condition, isl::union_map validity) const {
  auto res = isl_schedule_constraints_set_conditional_validity(copy(), condition.release(), validity.release());
  return manage(res);
}

isl::schedule_constraints schedule_constraints::set_context(isl::set context) const {
  auto res = isl_schedule_constraints_set_context(copy(), context.release());
  return manage(res);
}

isl::schedule_constraints schedule_constraints::set_proximity(isl::union_map proximity) const {
  auto res = isl_schedule_constraints_set_proximity(copy(), proximity.release());
  return manage(res);
}

isl::schedule_constraints schedule_constraints::set_validity(isl::union_map validity) const {
  auto res = isl_schedule_constraints_set_validity(copy(), validity.release());
  return manage(res);
}

// implementations for isl::schedule_node
isl::schedule_node manage(__isl_take isl_schedule_node *ptr) {
  return schedule_node(ptr);
}
isl::schedule_node give(__isl_take isl_schedule_node *ptr) {
  return manage(ptr);
}


schedule_node::schedule_node()
    : ptr(nullptr) {}

schedule_node::schedule_node(const isl::schedule_node &obj)
    : ptr(obj.copy()) {}
schedule_node::schedule_node(std::nullptr_t)
    : ptr(nullptr) {}


schedule_node::schedule_node(__isl_take isl_schedule_node *ptr)
    : ptr(ptr) {}


schedule_node &schedule_node::operator=(isl::schedule_node obj) {
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
__isl_keep isl_schedule_node *schedule_node::keep() const {
  return get();
}

__isl_give isl_schedule_node *schedule_node::take() {
  return release();
}

schedule_node::operator bool() const {
  return !is_null();
}

isl::ctx schedule_node::get_ctx() const {
  return isl::ctx(isl_schedule_node_get_ctx(ptr));
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


isl::schedule_node schedule_node::align_params(isl::space space) const {
  auto res = isl_schedule_node_align_params(copy(), space.release());
  return manage(res);
}

isl::schedule_node schedule_node::ancestor(int generation) const {
  auto res = isl_schedule_node_ancestor(copy(), generation);
  return manage(res);
}

isl::boolean schedule_node::band_member_get_coincident(int pos) const {
  auto res = isl_schedule_node_band_member_get_coincident(get(), pos);
  return manage(res);
}

isl::schedule_node schedule_node::band_member_set_coincident(int pos, int coincident) const {
  auto res = isl_schedule_node_band_member_set_coincident(copy(), pos, coincident);
  return manage(res);
}

isl::schedule_node schedule_node::band_set_ast_build_options(isl::union_set options) const {
  auto res = isl_schedule_node_band_set_ast_build_options(copy(), options.release());
  return manage(res);
}

isl::schedule_node schedule_node::child(int pos) const {
  auto res = isl_schedule_node_child(copy(), pos);
  return manage(res);
}

isl::set schedule_node::context_get_context() const {
  auto res = isl_schedule_node_context_get_context(get());
  return manage(res);
}

isl::schedule_node schedule_node::cut() const {
  auto res = isl_schedule_node_cut(copy());
  return manage(res);
}

isl::union_set schedule_node::domain_get_domain() const {
  auto res = isl_schedule_node_domain_get_domain(get());
  return manage(res);
}

isl::union_pw_multi_aff schedule_node::expansion_get_contraction() const {
  auto res = isl_schedule_node_expansion_get_contraction(get());
  return manage(res);
}

isl::union_map schedule_node::expansion_get_expansion() const {
  auto res = isl_schedule_node_expansion_get_expansion(get());
  return manage(res);
}

isl::union_map schedule_node::extension_get_extension() const {
  auto res = isl_schedule_node_extension_get_extension(get());
  return manage(res);
}

isl::union_set schedule_node::filter_get_filter() const {
  auto res = isl_schedule_node_filter_get_filter(get());
  return manage(res);
}

isl::schedule_node schedule_node::first_child() const {
  auto res = isl_schedule_node_first_child(copy());
  return manage(res);
}

isl::stat schedule_node::foreach_ancestor_top_down(const std::function<isl::stat(isl::schedule_node)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_schedule_node *arg_0, void *arg_1) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::schedule_node)> **>(arg_1);
    stat ret = (*func)(isl::manage(arg_0));
    return isl_stat(ret);
  };
  auto res = isl_schedule_node_foreach_ancestor_top_down(get(), fn_lambda, &fn_p);
  return isl::stat(res);
}

isl::schedule_node schedule_node::from_domain(isl::union_set domain) {
  auto res = isl_schedule_node_from_domain(domain.release());
  return manage(res);
}

isl::schedule_node schedule_node::from_extension(isl::union_map extension) {
  auto res = isl_schedule_node_from_extension(extension.release());
  return manage(res);
}

int schedule_node::get_ancestor_child_position(const isl::schedule_node &ancestor) const {
  auto res = isl_schedule_node_get_ancestor_child_position(get(), ancestor.get());
  return res;
}

isl::schedule_node schedule_node::get_child(int pos) const {
  auto res = isl_schedule_node_get_child(get(), pos);
  return manage(res);
}

int schedule_node::get_child_position() const {
  auto res = isl_schedule_node_get_child_position(get());
  return res;
}

isl::union_set schedule_node::get_domain() const {
  auto res = isl_schedule_node_get_domain(get());
  return manage(res);
}

isl::multi_union_pw_aff schedule_node::get_prefix_schedule_multi_union_pw_aff() const {
  auto res = isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(get());
  return manage(res);
}

isl::union_map schedule_node::get_prefix_schedule_relation() const {
  auto res = isl_schedule_node_get_prefix_schedule_relation(get());
  return manage(res);
}

isl::union_map schedule_node::get_prefix_schedule_union_map() const {
  auto res = isl_schedule_node_get_prefix_schedule_union_map(get());
  return manage(res);
}

isl::union_pw_multi_aff schedule_node::get_prefix_schedule_union_pw_multi_aff() const {
  auto res = isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(get());
  return manage(res);
}

isl::schedule schedule_node::get_schedule() const {
  auto res = isl_schedule_node_get_schedule(get());
  return manage(res);
}

int schedule_node::get_schedule_depth() const {
  auto res = isl_schedule_node_get_schedule_depth(get());
  return res;
}

isl::schedule_node schedule_node::get_shared_ancestor(const isl::schedule_node &node2) const {
  auto res = isl_schedule_node_get_shared_ancestor(get(), node2.get());
  return manage(res);
}

isl::union_pw_multi_aff schedule_node::get_subtree_contraction() const {
  auto res = isl_schedule_node_get_subtree_contraction(get());
  return manage(res);
}

isl::union_map schedule_node::get_subtree_expansion() const {
  auto res = isl_schedule_node_get_subtree_expansion(get());
  return manage(res);
}

isl::union_map schedule_node::get_subtree_schedule_union_map() const {
  auto res = isl_schedule_node_get_subtree_schedule_union_map(get());
  return manage(res);
}

int schedule_node::get_tree_depth() const {
  auto res = isl_schedule_node_get_tree_depth(get());
  return res;
}

isl::union_set schedule_node::get_universe_domain() const {
  auto res = isl_schedule_node_get_universe_domain(get());
  return manage(res);
}

isl::schedule_node schedule_node::graft_after(isl::schedule_node graft) const {
  auto res = isl_schedule_node_graft_after(copy(), graft.release());
  return manage(res);
}

isl::schedule_node schedule_node::graft_before(isl::schedule_node graft) const {
  auto res = isl_schedule_node_graft_before(copy(), graft.release());
  return manage(res);
}

isl::schedule_node schedule_node::group(isl::id group_id) const {
  auto res = isl_schedule_node_group(copy(), group_id.release());
  return manage(res);
}

isl::set schedule_node::guard_get_guard() const {
  auto res = isl_schedule_node_guard_get_guard(get());
  return manage(res);
}

isl::boolean schedule_node::has_children() const {
  auto res = isl_schedule_node_has_children(get());
  return manage(res);
}

isl::boolean schedule_node::has_next_sibling() const {
  auto res = isl_schedule_node_has_next_sibling(get());
  return manage(res);
}

isl::boolean schedule_node::has_parent() const {
  auto res = isl_schedule_node_has_parent(get());
  return manage(res);
}

isl::boolean schedule_node::has_previous_sibling() const {
  auto res = isl_schedule_node_has_previous_sibling(get());
  return manage(res);
}

isl::schedule_node schedule_node::insert_context(isl::set context) const {
  auto res = isl_schedule_node_insert_context(copy(), context.release());
  return manage(res);
}

isl::schedule_node schedule_node::insert_filter(isl::union_set filter) const {
  auto res = isl_schedule_node_insert_filter(copy(), filter.release());
  return manage(res);
}

isl::schedule_node schedule_node::insert_guard(isl::set context) const {
  auto res = isl_schedule_node_insert_guard(copy(), context.release());
  return manage(res);
}

isl::schedule_node schedule_node::insert_mark(isl::id mark) const {
  auto res = isl_schedule_node_insert_mark(copy(), mark.release());
  return manage(res);
}

isl::schedule_node schedule_node::insert_partial_schedule(isl::multi_union_pw_aff schedule) const {
  auto res = isl_schedule_node_insert_partial_schedule(copy(), schedule.release());
  return manage(res);
}

isl::schedule_node schedule_node::insert_sequence(isl::union_set_list filters) const {
  auto res = isl_schedule_node_insert_sequence(copy(), filters.release());
  return manage(res);
}

isl::schedule_node schedule_node::insert_set(isl::union_set_list filters) const {
  auto res = isl_schedule_node_insert_set(copy(), filters.release());
  return manage(res);
}

isl::boolean schedule_node::is_equal(const isl::schedule_node &node2) const {
  auto res = isl_schedule_node_is_equal(get(), node2.get());
  return manage(res);
}

isl::boolean schedule_node::is_subtree_anchored() const {
  auto res = isl_schedule_node_is_subtree_anchored(get());
  return manage(res);
}

isl::id schedule_node::mark_get_id() const {
  auto res = isl_schedule_node_mark_get_id(get());
  return manage(res);
}

isl::schedule_node schedule_node::next_sibling() const {
  auto res = isl_schedule_node_next_sibling(copy());
  return manage(res);
}

isl::schedule_node schedule_node::order_after(isl::union_set filter) const {
  auto res = isl_schedule_node_order_after(copy(), filter.release());
  return manage(res);
}

isl::schedule_node schedule_node::order_before(isl::union_set filter) const {
  auto res = isl_schedule_node_order_before(copy(), filter.release());
  return manage(res);
}

isl::schedule_node schedule_node::parent() const {
  auto res = isl_schedule_node_parent(copy());
  return manage(res);
}

isl::schedule_node schedule_node::previous_sibling() const {
  auto res = isl_schedule_node_previous_sibling(copy());
  return manage(res);
}

isl::schedule_node schedule_node::reset_user() const {
  auto res = isl_schedule_node_reset_user(copy());
  return manage(res);
}

isl::schedule_node schedule_node::root() const {
  auto res = isl_schedule_node_root(copy());
  return manage(res);
}

isl::schedule_node schedule_node::sequence_splice_child(int pos) const {
  auto res = isl_schedule_node_sequence_splice_child(copy(), pos);
  return manage(res);
}

// implementations for isl::set
isl::set manage(__isl_take isl_set *ptr) {
  return set(ptr);
}
isl::set give(__isl_take isl_set *ptr) {
  return manage(ptr);
}


set::set()
    : ptr(nullptr) {}

set::set(const isl::set &obj)
    : ptr(obj.copy()) {}
set::set(std::nullptr_t)
    : ptr(nullptr) {}


set::set(__isl_take isl_set *ptr)
    : ptr(ptr) {}

set::set(isl::union_set uset) {
  auto res = isl_set_from_union_set(uset.release());
  ptr = res;
}
set::set(isl::ctx ctx, const std::string &str) {
  auto res = isl_set_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}
set::set(isl::basic_set bset) {
  auto res = isl_set_from_basic_set(bset.release());
  ptr = res;
}
set::set(isl::point pnt) {
  auto res = isl_set_from_point(pnt.release());
  ptr = res;
}

set &set::operator=(isl::set obj) {
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
__isl_keep isl_set *set::keep() const {
  return get();
}

__isl_give isl_set *set::take() {
  return release();
}

set::operator bool() const {
  return !is_null();
}

isl::ctx set::get_ctx() const {
  return isl::ctx(isl_set_get_ctx(ptr));
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


isl::set set::add_constraint(isl::constraint constraint) const {
  auto res = isl_set_add_constraint(copy(), constraint.release());
  return manage(res);
}

isl::set set::add_dims(isl::dim type, unsigned int n) const {
  auto res = isl_set_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::basic_set set::affine_hull() const {
  auto res = isl_set_affine_hull(copy());
  return manage(res);
}

isl::set set::align_params(isl::space model) const {
  auto res = isl_set_align_params(copy(), model.release());
  return manage(res);
}

isl::set set::apply(isl::map map) const {
  auto res = isl_set_apply(copy(), map.release());
  return manage(res);
}

isl::basic_set set::bounded_simple_hull() const {
  auto res = isl_set_bounded_simple_hull(copy());
  return manage(res);
}

isl::set set::box_from_points(isl::point pnt1, isl::point pnt2) {
  auto res = isl_set_box_from_points(pnt1.release(), pnt2.release());
  return manage(res);
}

isl::set set::coalesce() const {
  auto res = isl_set_coalesce(copy());
  return manage(res);
}

isl::basic_set set::coefficients() const {
  auto res = isl_set_coefficients(copy());
  return manage(res);
}

isl::set set::complement() const {
  auto res = isl_set_complement(copy());
  return manage(res);
}

isl::basic_set set::convex_hull() const {
  auto res = isl_set_convex_hull(copy());
  return manage(res);
}

isl::val set::count_val() const {
  auto res = isl_set_count_val(get());
  return manage(res);
}

isl::set set::detect_equalities() const {
  auto res = isl_set_detect_equalities(copy());
  return manage(res);
}

unsigned int set::dim(isl::dim type) const {
  auto res = isl_set_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::boolean set::dim_has_any_lower_bound(isl::dim type, unsigned int pos) const {
  auto res = isl_set_dim_has_any_lower_bound(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::boolean set::dim_has_any_upper_bound(isl::dim type, unsigned int pos) const {
  auto res = isl_set_dim_has_any_upper_bound(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::boolean set::dim_has_lower_bound(isl::dim type, unsigned int pos) const {
  auto res = isl_set_dim_has_lower_bound(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::boolean set::dim_has_upper_bound(isl::dim type, unsigned int pos) const {
  auto res = isl_set_dim_has_upper_bound(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::boolean set::dim_is_bounded(isl::dim type, unsigned int pos) const {
  auto res = isl_set_dim_is_bounded(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::pw_aff set::dim_max(int pos) const {
  auto res = isl_set_dim_max(copy(), pos);
  return manage(res);
}

isl::pw_aff set::dim_min(int pos) const {
  auto res = isl_set_dim_min(copy(), pos);
  return manage(res);
}

isl::set set::drop_constraints_involving_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_set_drop_constraints_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::set set::drop_constraints_not_involving_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_set_drop_constraints_not_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::set set::eliminate(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_set_eliminate(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::set set::empty(isl::space dim) {
  auto res = isl_set_empty(dim.release());
  return manage(res);
}

isl::set set::equate(isl::dim type1, int pos1, isl::dim type2, int pos2) const {
  auto res = isl_set_equate(copy(), static_cast<enum isl_dim_type>(type1), pos1, static_cast<enum isl_dim_type>(type2), pos2);
  return manage(res);
}

int set::find_dim_by_id(isl::dim type, const isl::id &id) const {
  auto res = isl_set_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int set::find_dim_by_name(isl::dim type, const std::string &name) const {
  auto res = isl_set_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::set set::fix_si(isl::dim type, unsigned int pos, int value) const {
  auto res = isl_set_fix_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::set set::fix_val(isl::dim type, unsigned int pos, isl::val v) const {
  auto res = isl_set_fix_val(copy(), static_cast<enum isl_dim_type>(type), pos, v.release());
  return manage(res);
}

isl::set set::flat_product(isl::set set2) const {
  auto res = isl_set_flat_product(copy(), set2.release());
  return manage(res);
}

isl::set set::flatten() const {
  auto res = isl_set_flatten(copy());
  return manage(res);
}

isl::map set::flatten_map() const {
  auto res = isl_set_flatten_map(copy());
  return manage(res);
}

int set::follows_at(const isl::set &set2, int pos) const {
  auto res = isl_set_follows_at(get(), set2.get(), pos);
  return res;
}

isl::stat set::foreach_basic_set(const std::function<isl::stat(isl::basic_set)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_basic_set *arg_0, void *arg_1) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::basic_set)> **>(arg_1);
    stat ret = (*func)(isl::manage(arg_0));
    return isl_stat(ret);
  };
  auto res = isl_set_foreach_basic_set(get(), fn_lambda, &fn_p);
  return isl::stat(res);
}

isl::stat set::foreach_point(const std::function<isl::stat(isl::point)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_point *arg_0, void *arg_1) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::point)> **>(arg_1);
    stat ret = (*func)(isl::manage(arg_0));
    return isl_stat(ret);
  };
  auto res = isl_set_foreach_point(get(), fn_lambda, &fn_p);
  return isl::stat(res);
}

isl::set set::from_multi_pw_aff(isl::multi_pw_aff mpa) {
  auto res = isl_set_from_multi_pw_aff(mpa.release());
  return manage(res);
}

isl::set set::from_params() const {
  auto res = isl_set_from_params(copy());
  return manage(res);
}

isl::set set::from_pw_aff(isl::pw_aff pwaff) {
  auto res = isl_set_from_pw_aff(pwaff.release());
  return manage(res);
}

isl::set set::from_pw_multi_aff(isl::pw_multi_aff pma) {
  auto res = isl_set_from_pw_multi_aff(pma.release());
  return manage(res);
}

isl::basic_set_list set::get_basic_set_list() const {
  auto res = isl_set_get_basic_set_list(get());
  return manage(res);
}

isl::id set::get_dim_id(isl::dim type, unsigned int pos) const {
  auto res = isl_set_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

std::string set::get_dim_name(isl::dim type, unsigned int pos) const {
  auto res = isl_set_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

isl::space set::get_space() const {
  auto res = isl_set_get_space(get());
  return manage(res);
}

isl::id set::get_tuple_id() const {
  auto res = isl_set_get_tuple_id(get());
  return manage(res);
}

std::string set::get_tuple_name() const {
  auto res = isl_set_get_tuple_name(get());
  std::string tmp(res);
  return tmp;
}

isl::set set::gist(isl::set context) const {
  auto res = isl_set_gist(copy(), context.release());
  return manage(res);
}

isl::set set::gist_basic_set(isl::basic_set context) const {
  auto res = isl_set_gist_basic_set(copy(), context.release());
  return manage(res);
}

isl::set set::gist_params(isl::set context) const {
  auto res = isl_set_gist_params(copy(), context.release());
  return manage(res);
}

isl::boolean set::has_dim_id(isl::dim type, unsigned int pos) const {
  auto res = isl_set_has_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::boolean set::has_dim_name(isl::dim type, unsigned int pos) const {
  auto res = isl_set_has_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::boolean set::has_equal_space(const isl::set &set2) const {
  auto res = isl_set_has_equal_space(get(), set2.get());
  return manage(res);
}

isl::boolean set::has_tuple_id() const {
  auto res = isl_set_has_tuple_id(get());
  return manage(res);
}

isl::boolean set::has_tuple_name() const {
  auto res = isl_set_has_tuple_name(get());
  return manage(res);
}

isl::map set::identity() const {
  auto res = isl_set_identity(copy());
  return manage(res);
}

isl::pw_aff set::indicator_function() const {
  auto res = isl_set_indicator_function(copy());
  return manage(res);
}

isl::set set::insert_dims(isl::dim type, unsigned int pos, unsigned int n) const {
  auto res = isl_set_insert_dims(copy(), static_cast<enum isl_dim_type>(type), pos, n);
  return manage(res);
}

isl::set set::intersect(isl::set set2) const {
  auto res = isl_set_intersect(copy(), set2.release());
  return manage(res);
}

isl::set set::intersect_params(isl::set params) const {
  auto res = isl_set_intersect_params(copy(), params.release());
  return manage(res);
}

isl::boolean set::involves_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_set_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::boolean set::is_bounded() const {
  auto res = isl_set_is_bounded(get());
  return manage(res);
}

isl::boolean set::is_box() const {
  auto res = isl_set_is_box(get());
  return manage(res);
}

isl::boolean set::is_disjoint(const isl::set &set2) const {
  auto res = isl_set_is_disjoint(get(), set2.get());
  return manage(res);
}

isl::boolean set::is_empty() const {
  auto res = isl_set_is_empty(get());
  return manage(res);
}

isl::boolean set::is_equal(const isl::set &set2) const {
  auto res = isl_set_is_equal(get(), set2.get());
  return manage(res);
}

isl::boolean set::is_params() const {
  auto res = isl_set_is_params(get());
  return manage(res);
}

isl::boolean set::is_singleton() const {
  auto res = isl_set_is_singleton(get());
  return manage(res);
}

isl::boolean set::is_strict_subset(const isl::set &set2) const {
  auto res = isl_set_is_strict_subset(get(), set2.get());
  return manage(res);
}

isl::boolean set::is_subset(const isl::set &set2) const {
  auto res = isl_set_is_subset(get(), set2.get());
  return manage(res);
}

isl::boolean set::is_wrapping() const {
  auto res = isl_set_is_wrapping(get());
  return manage(res);
}

isl::map set::lex_ge_set(isl::set set2) const {
  auto res = isl_set_lex_ge_set(copy(), set2.release());
  return manage(res);
}

isl::map set::lex_gt_set(isl::set set2) const {
  auto res = isl_set_lex_gt_set(copy(), set2.release());
  return manage(res);
}

isl::map set::lex_le_set(isl::set set2) const {
  auto res = isl_set_lex_le_set(copy(), set2.release());
  return manage(res);
}

isl::map set::lex_lt_set(isl::set set2) const {
  auto res = isl_set_lex_lt_set(copy(), set2.release());
  return manage(res);
}

isl::set set::lexmax() const {
  auto res = isl_set_lexmax(copy());
  return manage(res);
}

isl::pw_multi_aff set::lexmax_pw_multi_aff() const {
  auto res = isl_set_lexmax_pw_multi_aff(copy());
  return manage(res);
}

isl::set set::lexmin() const {
  auto res = isl_set_lexmin(copy());
  return manage(res);
}

isl::pw_multi_aff set::lexmin_pw_multi_aff() const {
  auto res = isl_set_lexmin_pw_multi_aff(copy());
  return manage(res);
}

isl::set set::lower_bound_si(isl::dim type, unsigned int pos, int value) const {
  auto res = isl_set_lower_bound_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::set set::lower_bound_val(isl::dim type, unsigned int pos, isl::val value) const {
  auto res = isl_set_lower_bound_val(copy(), static_cast<enum isl_dim_type>(type), pos, value.release());
  return manage(res);
}

isl::val set::max_val(const isl::aff &obj) const {
  auto res = isl_set_max_val(get(), obj.get());
  return manage(res);
}

isl::val set::min_val(const isl::aff &obj) const {
  auto res = isl_set_min_val(get(), obj.get());
  return manage(res);
}

isl::set set::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const {
  auto res = isl_set_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl::set set::nat_universe(isl::space dim) {
  auto res = isl_set_nat_universe(dim.release());
  return manage(res);
}

isl::set set::neg() const {
  auto res = isl_set_neg(copy());
  return manage(res);
}

isl::set set::params() const {
  auto res = isl_set_params(copy());
  return manage(res);
}

int set::plain_cmp(const isl::set &set2) const {
  auto res = isl_set_plain_cmp(get(), set2.get());
  return res;
}

isl::val set::plain_get_val_if_fixed(isl::dim type, unsigned int pos) const {
  auto res = isl_set_plain_get_val_if_fixed(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::boolean set::plain_is_disjoint(const isl::set &set2) const {
  auto res = isl_set_plain_is_disjoint(get(), set2.get());
  return manage(res);
}

isl::boolean set::plain_is_empty() const {
  auto res = isl_set_plain_is_empty(get());
  return manage(res);
}

isl::boolean set::plain_is_equal(const isl::set &set2) const {
  auto res = isl_set_plain_is_equal(get(), set2.get());
  return manage(res);
}

isl::boolean set::plain_is_universe() const {
  auto res = isl_set_plain_is_universe(get());
  return manage(res);
}

isl::basic_set set::plain_unshifted_simple_hull() const {
  auto res = isl_set_plain_unshifted_simple_hull(copy());
  return manage(res);
}

isl::basic_set set::polyhedral_hull() const {
  auto res = isl_set_polyhedral_hull(copy());
  return manage(res);
}

isl::set set::preimage_multi_aff(isl::multi_aff ma) const {
  auto res = isl_set_preimage_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::set set::preimage_multi_pw_aff(isl::multi_pw_aff mpa) const {
  auto res = isl_set_preimage_multi_pw_aff(copy(), mpa.release());
  return manage(res);
}

isl::set set::preimage_pw_multi_aff(isl::pw_multi_aff pma) const {
  auto res = isl_set_preimage_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::set set::product(isl::set set2) const {
  auto res = isl_set_product(copy(), set2.release());
  return manage(res);
}

isl::map set::project_onto_map(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_set_project_onto_map(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::set set::project_out(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_set_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::set set::remove_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_set_remove_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::set set::remove_divs() const {
  auto res = isl_set_remove_divs(copy());
  return manage(res);
}

isl::set set::remove_divs_involving_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_set_remove_divs_involving_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::set set::remove_redundancies() const {
  auto res = isl_set_remove_redundancies(copy());
  return manage(res);
}

isl::set set::remove_unknown_divs() const {
  auto res = isl_set_remove_unknown_divs(copy());
  return manage(res);
}

isl::set set::reset_space(isl::space dim) const {
  auto res = isl_set_reset_space(copy(), dim.release());
  return manage(res);
}

isl::set set::reset_tuple_id() const {
  auto res = isl_set_reset_tuple_id(copy());
  return manage(res);
}

isl::set set::reset_user() const {
  auto res = isl_set_reset_user(copy());
  return manage(res);
}

isl::basic_set set::sample() const {
  auto res = isl_set_sample(copy());
  return manage(res);
}

isl::point set::sample_point() const {
  auto res = isl_set_sample_point(copy());
  return manage(res);
}

isl::set set::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const {
  auto res = isl_set_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::set set::set_tuple_id(isl::id id) const {
  auto res = isl_set_set_tuple_id(copy(), id.release());
  return manage(res);
}

isl::set set::set_tuple_name(const std::string &s) const {
  auto res = isl_set_set_tuple_name(copy(), s.c_str());
  return manage(res);
}

isl::basic_set set::simple_hull() const {
  auto res = isl_set_simple_hull(copy());
  return manage(res);
}

int set::size() const {
  auto res = isl_set_size(get());
  return res;
}

isl::basic_set set::solutions() const {
  auto res = isl_set_solutions(copy());
  return manage(res);
}

isl::set set::split_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_set_split_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::set set::subtract(isl::set set2) const {
  auto res = isl_set_subtract(copy(), set2.release());
  return manage(res);
}

isl::set set::sum(isl::set set2) const {
  auto res = isl_set_sum(copy(), set2.release());
  return manage(res);
}

isl::set set::unite(isl::set set2) const {
  auto res = isl_set_union(copy(), set2.release());
  return manage(res);
}

isl::set set::universe(isl::space dim) {
  auto res = isl_set_universe(dim.release());
  return manage(res);
}

isl::basic_set set::unshifted_simple_hull() const {
  auto res = isl_set_unshifted_simple_hull(copy());
  return manage(res);
}

isl::basic_set set::unshifted_simple_hull_from_set_list(isl::set_list list) const {
  auto res = isl_set_unshifted_simple_hull_from_set_list(copy(), list.release());
  return manage(res);
}

isl::map set::unwrap() const {
  auto res = isl_set_unwrap(copy());
  return manage(res);
}

isl::set set::upper_bound_si(isl::dim type, unsigned int pos, int value) const {
  auto res = isl_set_upper_bound_si(copy(), static_cast<enum isl_dim_type>(type), pos, value);
  return manage(res);
}

isl::set set::upper_bound_val(isl::dim type, unsigned int pos, isl::val value) const {
  auto res = isl_set_upper_bound_val(copy(), static_cast<enum isl_dim_type>(type), pos, value.release());
  return manage(res);
}

isl::map set::wrapped_domain_map() const {
  auto res = isl_set_wrapped_domain_map(copy());
  return manage(res);
}

// implementations for isl::set_list
isl::set_list manage(__isl_take isl_set_list *ptr) {
  return set_list(ptr);
}
isl::set_list give(__isl_take isl_set_list *ptr) {
  return manage(ptr);
}


set_list::set_list()
    : ptr(nullptr) {}

set_list::set_list(const isl::set_list &obj)
    : ptr(obj.copy()) {}
set_list::set_list(std::nullptr_t)
    : ptr(nullptr) {}


set_list::set_list(__isl_take isl_set_list *ptr)
    : ptr(ptr) {}


set_list &set_list::operator=(isl::set_list obj) {
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
__isl_keep isl_set_list *set_list::keep() const {
  return get();
}

__isl_give isl_set_list *set_list::take() {
  return release();
}

set_list::operator bool() const {
  return !is_null();
}

isl::ctx set_list::get_ctx() const {
  return isl::ctx(isl_set_list_get_ctx(ptr));
}



void set_list::dump() const {
  isl_set_list_dump(get());
}



// implementations for isl::space
isl::space manage(__isl_take isl_space *ptr) {
  return space(ptr);
}
isl::space give(__isl_take isl_space *ptr) {
  return manage(ptr);
}


space::space()
    : ptr(nullptr) {}

space::space(const isl::space &obj)
    : ptr(obj.copy()) {}
space::space(std::nullptr_t)
    : ptr(nullptr) {}


space::space(__isl_take isl_space *ptr)
    : ptr(ptr) {}

space::space(isl::ctx ctx, unsigned int nparam, unsigned int n_in, unsigned int n_out) {
  auto res = isl_space_alloc(ctx.release(), nparam, n_in, n_out);
  ptr = res;
}
space::space(isl::ctx ctx, unsigned int nparam, unsigned int dim) {
  auto res = isl_space_set_alloc(ctx.release(), nparam, dim);
  ptr = res;
}

space &space::operator=(isl::space obj) {
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
__isl_keep isl_space *space::keep() const {
  return get();
}

__isl_give isl_space *space::take() {
  return release();
}

space::operator bool() const {
  return !is_null();
}

isl::ctx space::get_ctx() const {
  return isl::ctx(isl_space_get_ctx(ptr));
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


isl::space space::add_dims(isl::dim type, unsigned int n) const {
  auto res = isl_space_add_dims(copy(), static_cast<enum isl_dim_type>(type), n);
  return manage(res);
}

isl::space space::align_params(isl::space dim2) const {
  auto res = isl_space_align_params(copy(), dim2.release());
  return manage(res);
}

isl::boolean space::can_curry() const {
  auto res = isl_space_can_curry(get());
  return manage(res);
}

isl::boolean space::can_range_curry() const {
  auto res = isl_space_can_range_curry(get());
  return manage(res);
}

isl::boolean space::can_uncurry() const {
  auto res = isl_space_can_uncurry(get());
  return manage(res);
}

isl::boolean space::can_zip() const {
  auto res = isl_space_can_zip(get());
  return manage(res);
}

isl::space space::curry() const {
  auto res = isl_space_curry(copy());
  return manage(res);
}

unsigned int space::dim(isl::dim type) const {
  auto res = isl_space_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::space space::domain() const {
  auto res = isl_space_domain(copy());
  return manage(res);
}

isl::space space::domain_factor_domain() const {
  auto res = isl_space_domain_factor_domain(copy());
  return manage(res);
}

isl::space space::domain_factor_range() const {
  auto res = isl_space_domain_factor_range(copy());
  return manage(res);
}

isl::boolean space::domain_is_wrapping() const {
  auto res = isl_space_domain_is_wrapping(get());
  return manage(res);
}

isl::space space::domain_map() const {
  auto res = isl_space_domain_map(copy());
  return manage(res);
}

isl::space space::domain_product(isl::space right) const {
  auto res = isl_space_domain_product(copy(), right.release());
  return manage(res);
}

isl::space space::drop_dims(isl::dim type, unsigned int first, unsigned int num) const {
  auto res = isl_space_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, num);
  return manage(res);
}

isl::space space::factor_domain() const {
  auto res = isl_space_factor_domain(copy());
  return manage(res);
}

isl::space space::factor_range() const {
  auto res = isl_space_factor_range(copy());
  return manage(res);
}

int space::find_dim_by_id(isl::dim type, const isl::id &id) const {
  auto res = isl_space_find_dim_by_id(get(), static_cast<enum isl_dim_type>(type), id.get());
  return res;
}

int space::find_dim_by_name(isl::dim type, const std::string &name) const {
  auto res = isl_space_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::space space::from_domain() const {
  auto res = isl_space_from_domain(copy());
  return manage(res);
}

isl::space space::from_range() const {
  auto res = isl_space_from_range(copy());
  return manage(res);
}

isl::id space::get_dim_id(isl::dim type, unsigned int pos) const {
  auto res = isl_space_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

std::string space::get_dim_name(isl::dim type, unsigned int pos) const {
  auto res = isl_space_get_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  std::string tmp(res);
  return tmp;
}

isl::id space::get_tuple_id(isl::dim type) const {
  auto res = isl_space_get_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

std::string space::get_tuple_name(isl::dim type) const {
  auto res = isl_space_get_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  std::string tmp(res);
  return tmp;
}

isl::boolean space::has_dim_id(isl::dim type, unsigned int pos) const {
  auto res = isl_space_has_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::boolean space::has_dim_name(isl::dim type, unsigned int pos) const {
  auto res = isl_space_has_dim_name(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

isl::boolean space::has_equal_params(const isl::space &space2) const {
  auto res = isl_space_has_equal_params(get(), space2.get());
  return manage(res);
}

isl::boolean space::has_equal_tuples(const isl::space &space2) const {
  auto res = isl_space_has_equal_tuples(get(), space2.get());
  return manage(res);
}

isl::boolean space::has_tuple_id(isl::dim type) const {
  auto res = isl_space_has_tuple_id(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::boolean space::has_tuple_name(isl::dim type) const {
  auto res = isl_space_has_tuple_name(get(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::space space::insert_dims(isl::dim type, unsigned int pos, unsigned int n) const {
  auto res = isl_space_insert_dims(copy(), static_cast<enum isl_dim_type>(type), pos, n);
  return manage(res);
}

isl::boolean space::is_domain(const isl::space &space2) const {
  auto res = isl_space_is_domain(get(), space2.get());
  return manage(res);
}

isl::boolean space::is_equal(const isl::space &space2) const {
  auto res = isl_space_is_equal(get(), space2.get());
  return manage(res);
}

isl::boolean space::is_map() const {
  auto res = isl_space_is_map(get());
  return manage(res);
}

isl::boolean space::is_params() const {
  auto res = isl_space_is_params(get());
  return manage(res);
}

isl::boolean space::is_product() const {
  auto res = isl_space_is_product(get());
  return manage(res);
}

isl::boolean space::is_range(const isl::space &space2) const {
  auto res = isl_space_is_range(get(), space2.get());
  return manage(res);
}

isl::boolean space::is_set() const {
  auto res = isl_space_is_set(get());
  return manage(res);
}

isl::boolean space::is_wrapping() const {
  auto res = isl_space_is_wrapping(get());
  return manage(res);
}

isl::space space::join(isl::space right) const {
  auto res = isl_space_join(copy(), right.release());
  return manage(res);
}

isl::space space::map_from_domain_and_range(isl::space range) const {
  auto res = isl_space_map_from_domain_and_range(copy(), range.release());
  return manage(res);
}

isl::space space::map_from_set() const {
  auto res = isl_space_map_from_set(copy());
  return manage(res);
}

isl::space space::move_dims(isl::dim dst_type, unsigned int dst_pos, isl::dim src_type, unsigned int src_pos, unsigned int n) const {
  auto res = isl_space_move_dims(copy(), static_cast<enum isl_dim_type>(dst_type), dst_pos, static_cast<enum isl_dim_type>(src_type), src_pos, n);
  return manage(res);
}

isl::space space::params() const {
  auto res = isl_space_params(copy());
  return manage(res);
}

isl::space space::params_alloc(isl::ctx ctx, unsigned int nparam) {
  auto res = isl_space_params_alloc(ctx.release(), nparam);
  return manage(res);
}

isl::space space::product(isl::space right) const {
  auto res = isl_space_product(copy(), right.release());
  return manage(res);
}

isl::space space::range() const {
  auto res = isl_space_range(copy());
  return manage(res);
}

isl::space space::range_curry() const {
  auto res = isl_space_range_curry(copy());
  return manage(res);
}

isl::space space::range_factor_domain() const {
  auto res = isl_space_range_factor_domain(copy());
  return manage(res);
}

isl::space space::range_factor_range() const {
  auto res = isl_space_range_factor_range(copy());
  return manage(res);
}

isl::boolean space::range_is_wrapping() const {
  auto res = isl_space_range_is_wrapping(get());
  return manage(res);
}

isl::space space::range_map() const {
  auto res = isl_space_range_map(copy());
  return manage(res);
}

isl::space space::range_product(isl::space right) const {
  auto res = isl_space_range_product(copy(), right.release());
  return manage(res);
}

isl::space space::reset_tuple_id(isl::dim type) const {
  auto res = isl_space_reset_tuple_id(copy(), static_cast<enum isl_dim_type>(type));
  return manage(res);
}

isl::space space::reset_user() const {
  auto res = isl_space_reset_user(copy());
  return manage(res);
}

isl::space space::reverse() const {
  auto res = isl_space_reverse(copy());
  return manage(res);
}

isl::space space::set_dim_id(isl::dim type, unsigned int pos, isl::id id) const {
  auto res = isl_space_set_dim_id(copy(), static_cast<enum isl_dim_type>(type), pos, id.release());
  return manage(res);
}

isl::space space::set_from_params() const {
  auto res = isl_space_set_from_params(copy());
  return manage(res);
}

isl::space space::set_tuple_id(isl::dim type, isl::id id) const {
  auto res = isl_space_set_tuple_id(copy(), static_cast<enum isl_dim_type>(type), id.release());
  return manage(res);
}

isl::space space::set_tuple_name(isl::dim type, const std::string &s) const {
  auto res = isl_space_set_tuple_name(copy(), static_cast<enum isl_dim_type>(type), s.c_str());
  return manage(res);
}

isl::boolean space::tuple_is_equal(isl::dim type1, const isl::space &space2, isl::dim type2) const {
  auto res = isl_space_tuple_is_equal(get(), static_cast<enum isl_dim_type>(type1), space2.get(), static_cast<enum isl_dim_type>(type2));
  return manage(res);
}

isl::space space::uncurry() const {
  auto res = isl_space_uncurry(copy());
  return manage(res);
}

isl::space space::unwrap() const {
  auto res = isl_space_unwrap(copy());
  return manage(res);
}

isl::space space::wrap() const {
  auto res = isl_space_wrap(copy());
  return manage(res);
}

isl::space space::zip() const {
  auto res = isl_space_zip(copy());
  return manage(res);
}

// implementations for isl::term
isl::term manage(__isl_take isl_term *ptr) {
  return term(ptr);
}
isl::term give(__isl_take isl_term *ptr) {
  return manage(ptr);
}


term::term()
    : ptr(nullptr) {}

term::term(const isl::term &obj)
    : ptr(obj.copy()) {}
term::term(std::nullptr_t)
    : ptr(nullptr) {}


term::term(__isl_take isl_term *ptr)
    : ptr(ptr) {}


term &term::operator=(isl::term obj) {
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
__isl_keep isl_term *term::keep() const {
  return get();
}

__isl_give isl_term *term::take() {
  return release();
}

term::operator bool() const {
  return !is_null();
}

isl::ctx term::get_ctx() const {
  return isl::ctx(isl_term_get_ctx(ptr));
}




unsigned int term::dim(isl::dim type) const {
  auto res = isl_term_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::val term::get_coefficient_val() const {
  auto res = isl_term_get_coefficient_val(get());
  return manage(res);
}

isl::aff term::get_div(unsigned int pos) const {
  auto res = isl_term_get_div(get(), pos);
  return manage(res);
}

int term::get_exp(isl::dim type, unsigned int pos) const {
  auto res = isl_term_get_exp(get(), static_cast<enum isl_dim_type>(type), pos);
  return res;
}

// implementations for isl::union_access_info
isl::union_access_info manage(__isl_take isl_union_access_info *ptr) {
  return union_access_info(ptr);
}
isl::union_access_info give(__isl_take isl_union_access_info *ptr) {
  return manage(ptr);
}


union_access_info::union_access_info()
    : ptr(nullptr) {}

union_access_info::union_access_info(const isl::union_access_info &obj)
    : ptr(obj.copy()) {}
union_access_info::union_access_info(std::nullptr_t)
    : ptr(nullptr) {}


union_access_info::union_access_info(__isl_take isl_union_access_info *ptr)
    : ptr(ptr) {}

union_access_info::union_access_info(isl::union_map sink) {
  auto res = isl_union_access_info_from_sink(sink.release());
  ptr = res;
}

union_access_info &union_access_info::operator=(isl::union_access_info obj) {
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
__isl_keep isl_union_access_info *union_access_info::keep() const {
  return get();
}

__isl_give isl_union_access_info *union_access_info::take() {
  return release();
}

union_access_info::operator bool() const {
  return !is_null();
}

isl::ctx union_access_info::get_ctx() const {
  return isl::ctx(isl_union_access_info_get_ctx(ptr));
}


std::string union_access_info::to_str() const {
  char *Tmp = isl_union_access_info_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}



isl::union_flow union_access_info::compute_flow() const {
  auto res = isl_union_access_info_compute_flow(copy());
  return manage(res);
}

isl::union_access_info union_access_info::set_kill(isl::union_map kill) const {
  auto res = isl_union_access_info_set_kill(copy(), kill.release());
  return manage(res);
}

isl::union_access_info union_access_info::set_may_source(isl::union_map may_source) const {
  auto res = isl_union_access_info_set_may_source(copy(), may_source.release());
  return manage(res);
}

isl::union_access_info union_access_info::set_must_source(isl::union_map must_source) const {
  auto res = isl_union_access_info_set_must_source(copy(), must_source.release());
  return manage(res);
}

isl::union_access_info union_access_info::set_schedule(isl::schedule schedule) const {
  auto res = isl_union_access_info_set_schedule(copy(), schedule.release());
  return manage(res);
}

isl::union_access_info union_access_info::set_schedule_map(isl::union_map schedule_map) const {
  auto res = isl_union_access_info_set_schedule_map(copy(), schedule_map.release());
  return manage(res);
}

// implementations for isl::union_flow
isl::union_flow manage(__isl_take isl_union_flow *ptr) {
  return union_flow(ptr);
}
isl::union_flow give(__isl_take isl_union_flow *ptr) {
  return manage(ptr);
}


union_flow::union_flow()
    : ptr(nullptr) {}

union_flow::union_flow(const isl::union_flow &obj)
    : ptr(obj.copy()) {}
union_flow::union_flow(std::nullptr_t)
    : ptr(nullptr) {}


union_flow::union_flow(__isl_take isl_union_flow *ptr)
    : ptr(ptr) {}


union_flow &union_flow::operator=(isl::union_flow obj) {
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
__isl_keep isl_union_flow *union_flow::keep() const {
  return get();
}

__isl_give isl_union_flow *union_flow::take() {
  return release();
}

union_flow::operator bool() const {
  return !is_null();
}

isl::ctx union_flow::get_ctx() const {
  return isl::ctx(isl_union_flow_get_ctx(ptr));
}


std::string union_flow::to_str() const {
  char *Tmp = isl_union_flow_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}



isl::union_map union_flow::get_full_may_dependence() const {
  auto res = isl_union_flow_get_full_may_dependence(get());
  return manage(res);
}

isl::union_map union_flow::get_full_must_dependence() const {
  auto res = isl_union_flow_get_full_must_dependence(get());
  return manage(res);
}

isl::union_map union_flow::get_may_dependence() const {
  auto res = isl_union_flow_get_may_dependence(get());
  return manage(res);
}

isl::union_map union_flow::get_may_no_source() const {
  auto res = isl_union_flow_get_may_no_source(get());
  return manage(res);
}

isl::union_map union_flow::get_must_dependence() const {
  auto res = isl_union_flow_get_must_dependence(get());
  return manage(res);
}

isl::union_map union_flow::get_must_no_source() const {
  auto res = isl_union_flow_get_must_no_source(get());
  return manage(res);
}

// implementations for isl::union_map
isl::union_map manage(__isl_take isl_union_map *ptr) {
  return union_map(ptr);
}
isl::union_map give(__isl_take isl_union_map *ptr) {
  return manage(ptr);
}


union_map::union_map()
    : ptr(nullptr) {}

union_map::union_map(const isl::union_map &obj)
    : ptr(obj.copy()) {}
union_map::union_map(std::nullptr_t)
    : ptr(nullptr) {}


union_map::union_map(__isl_take isl_union_map *ptr)
    : ptr(ptr) {}

union_map::union_map(isl::union_pw_aff upa) {
  auto res = isl_union_map_from_union_pw_aff(upa.release());
  ptr = res;
}
union_map::union_map(isl::basic_map bmap) {
  auto res = isl_union_map_from_basic_map(bmap.release());
  ptr = res;
}
union_map::union_map(isl::map map) {
  auto res = isl_union_map_from_map(map.release());
  ptr = res;
}
union_map::union_map(isl::ctx ctx, const std::string &str) {
  auto res = isl_union_map_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

union_map &union_map::operator=(isl::union_map obj) {
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
__isl_keep isl_union_map *union_map::keep() const {
  return get();
}

__isl_give isl_union_map *union_map::take() {
  return release();
}

union_map::operator bool() const {
  return !is_null();
}

isl::ctx union_map::get_ctx() const {
  return isl::ctx(isl_union_map_get_ctx(ptr));
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


isl::union_map union_map::add_map(isl::map map) const {
  auto res = isl_union_map_add_map(copy(), map.release());
  return manage(res);
}

isl::union_map union_map::affine_hull() const {
  auto res = isl_union_map_affine_hull(copy());
  return manage(res);
}

isl::union_map union_map::align_params(isl::space model) const {
  auto res = isl_union_map_align_params(copy(), model.release());
  return manage(res);
}

isl::union_map union_map::apply_domain(isl::union_map umap2) const {
  auto res = isl_union_map_apply_domain(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::apply_range(isl::union_map umap2) const {
  auto res = isl_union_map_apply_range(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::coalesce() const {
  auto res = isl_union_map_coalesce(copy());
  return manage(res);
}

isl::boolean union_map::contains(const isl::space &space) const {
  auto res = isl_union_map_contains(get(), space.get());
  return manage(res);
}

isl::union_map union_map::curry() const {
  auto res = isl_union_map_curry(copy());
  return manage(res);
}

isl::union_set union_map::deltas() const {
  auto res = isl_union_map_deltas(copy());
  return manage(res);
}

isl::union_map union_map::deltas_map() const {
  auto res = isl_union_map_deltas_map(copy());
  return manage(res);
}

isl::union_map union_map::detect_equalities() const {
  auto res = isl_union_map_detect_equalities(copy());
  return manage(res);
}

unsigned int union_map::dim(isl::dim type) const {
  auto res = isl_union_map_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::union_set union_map::domain() const {
  auto res = isl_union_map_domain(copy());
  return manage(res);
}

isl::union_map union_map::domain_factor_domain() const {
  auto res = isl_union_map_domain_factor_domain(copy());
  return manage(res);
}

isl::union_map union_map::domain_factor_range() const {
  auto res = isl_union_map_domain_factor_range(copy());
  return manage(res);
}

isl::union_map union_map::domain_map() const {
  auto res = isl_union_map_domain_map(copy());
  return manage(res);
}

isl::union_pw_multi_aff union_map::domain_map_union_pw_multi_aff() const {
  auto res = isl_union_map_domain_map_union_pw_multi_aff(copy());
  return manage(res);
}

isl::union_map union_map::domain_product(isl::union_map umap2) const {
  auto res = isl_union_map_domain_product(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::empty(isl::space dim) {
  auto res = isl_union_map_empty(dim.release());
  return manage(res);
}

isl::union_map union_map::eq_at_multi_union_pw_aff(isl::multi_union_pw_aff mupa) const {
  auto res = isl_union_map_eq_at_multi_union_pw_aff(copy(), mupa.release());
  return manage(res);
}

isl::map union_map::extract_map(isl::space dim) const {
  auto res = isl_union_map_extract_map(get(), dim.release());
  return manage(res);
}

isl::union_map union_map::factor_domain() const {
  auto res = isl_union_map_factor_domain(copy());
  return manage(res);
}

isl::union_map union_map::factor_range() const {
  auto res = isl_union_map_factor_range(copy());
  return manage(res);
}

int union_map::find_dim_by_name(isl::dim type, const std::string &name) const {
  auto res = isl_union_map_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::union_map union_map::fixed_power(isl::val exp) const {
  auto res = isl_union_map_fixed_power_val(copy(), exp.release());
  return manage(res);
}

isl::union_map union_map::flat_domain_product(isl::union_map umap2) const {
  auto res = isl_union_map_flat_domain_product(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::flat_range_product(isl::union_map umap2) const {
  auto res = isl_union_map_flat_range_product(copy(), umap2.release());
  return manage(res);
}

isl::stat union_map::foreach_map(const std::function<isl::stat(isl::map)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_map *arg_0, void *arg_1) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::map)> **>(arg_1);
    stat ret = (*func)(isl::manage(arg_0));
    return isl_stat(ret);
  };
  auto res = isl_union_map_foreach_map(get(), fn_lambda, &fn_p);
  return isl::stat(res);
}

isl::union_map union_map::from(isl::union_pw_multi_aff upma) {
  auto res = isl_union_map_from_union_pw_multi_aff(upma.release());
  return manage(res);
}

isl::union_map union_map::from(isl::multi_union_pw_aff mupa) {
  auto res = isl_union_map_from_multi_union_pw_aff(mupa.release());
  return manage(res);
}

isl::union_map union_map::from_domain(isl::union_set uset) {
  auto res = isl_union_map_from_domain(uset.release());
  return manage(res);
}

isl::union_map union_map::from_domain_and_range(isl::union_set domain, isl::union_set range) {
  auto res = isl_union_map_from_domain_and_range(domain.release(), range.release());
  return manage(res);
}

isl::union_map union_map::from_range(isl::union_set uset) {
  auto res = isl_union_map_from_range(uset.release());
  return manage(res);
}

isl::id union_map::get_dim_id(isl::dim type, unsigned int pos) const {
  auto res = isl_union_map_get_dim_id(get(), static_cast<enum isl_dim_type>(type), pos);
  return manage(res);
}

uint32_t union_map::get_hash() const {
  auto res = isl_union_map_get_hash(get());
  return res;
}

isl::space union_map::get_space() const {
  auto res = isl_union_map_get_space(get());
  return manage(res);
}

isl::union_map union_map::gist(isl::union_map context) const {
  auto res = isl_union_map_gist(copy(), context.release());
  return manage(res);
}

isl::union_map union_map::gist_domain(isl::union_set uset) const {
  auto res = isl_union_map_gist_domain(copy(), uset.release());
  return manage(res);
}

isl::union_map union_map::gist_params(isl::set set) const {
  auto res = isl_union_map_gist_params(copy(), set.release());
  return manage(res);
}

isl::union_map union_map::gist_range(isl::union_set uset) const {
  auto res = isl_union_map_gist_range(copy(), uset.release());
  return manage(res);
}

isl::union_map union_map::intersect(isl::union_map umap2) const {
  auto res = isl_union_map_intersect(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::intersect_domain(isl::union_set uset) const {
  auto res = isl_union_map_intersect_domain(copy(), uset.release());
  return manage(res);
}

isl::union_map union_map::intersect_params(isl::set set) const {
  auto res = isl_union_map_intersect_params(copy(), set.release());
  return manage(res);
}

isl::union_map union_map::intersect_range(isl::union_set uset) const {
  auto res = isl_union_map_intersect_range(copy(), uset.release());
  return manage(res);
}

isl::union_map union_map::intersect_range_factor_range(isl::union_map factor) const {
  auto res = isl_union_map_intersect_range_factor_range(copy(), factor.release());
  return manage(res);
}

isl::boolean union_map::involves_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_union_map_involves_dims(get(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::boolean union_map::is_bijective() const {
  auto res = isl_union_map_is_bijective(get());
  return manage(res);
}

isl::boolean union_map::is_disjoint(const isl::union_map &umap2) const {
  auto res = isl_union_map_is_disjoint(get(), umap2.get());
  return manage(res);
}

isl::boolean union_map::is_empty() const {
  auto res = isl_union_map_is_empty(get());
  return manage(res);
}

isl::boolean union_map::is_equal(const isl::union_map &umap2) const {
  auto res = isl_union_map_is_equal(get(), umap2.get());
  return manage(res);
}

isl::boolean union_map::is_identity() const {
  auto res = isl_union_map_is_identity(get());
  return manage(res);
}

isl::boolean union_map::is_injective() const {
  auto res = isl_union_map_is_injective(get());
  return manage(res);
}

isl::boolean union_map::is_single_valued() const {
  auto res = isl_union_map_is_single_valued(get());
  return manage(res);
}

isl::boolean union_map::is_strict_subset(const isl::union_map &umap2) const {
  auto res = isl_union_map_is_strict_subset(get(), umap2.get());
  return manage(res);
}

isl::boolean union_map::is_subset(const isl::union_map &umap2) const {
  auto res = isl_union_map_is_subset(get(), umap2.get());
  return manage(res);
}

isl::union_map union_map::lex_ge_union_map(isl::union_map umap2) const {
  auto res = isl_union_map_lex_ge_union_map(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::lex_gt_at_multi_union_pw_aff(isl::multi_union_pw_aff mupa) const {
  auto res = isl_union_map_lex_gt_at_multi_union_pw_aff(copy(), mupa.release());
  return manage(res);
}

isl::union_map union_map::lex_gt_union_map(isl::union_map umap2) const {
  auto res = isl_union_map_lex_gt_union_map(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::lex_le_union_map(isl::union_map umap2) const {
  auto res = isl_union_map_lex_le_union_map(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::lex_lt_at_multi_union_pw_aff(isl::multi_union_pw_aff mupa) const {
  auto res = isl_union_map_lex_lt_at_multi_union_pw_aff(copy(), mupa.release());
  return manage(res);
}

isl::union_map union_map::lex_lt_union_map(isl::union_map umap2) const {
  auto res = isl_union_map_lex_lt_union_map(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::lexmax() const {
  auto res = isl_union_map_lexmax(copy());
  return manage(res);
}

isl::union_map union_map::lexmin() const {
  auto res = isl_union_map_lexmin(copy());
  return manage(res);
}

isl::set union_map::params() const {
  auto res = isl_union_map_params(copy());
  return manage(res);
}

isl::boolean union_map::plain_is_injective() const {
  auto res = isl_union_map_plain_is_injective(get());
  return manage(res);
}

isl::union_map union_map::polyhedral_hull() const {
  auto res = isl_union_map_polyhedral_hull(copy());
  return manage(res);
}

isl::union_map union_map::preimage_domain_multi_aff(isl::multi_aff ma) const {
  auto res = isl_union_map_preimage_domain_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::union_map union_map::preimage_domain_multi_pw_aff(isl::multi_pw_aff mpa) const {
  auto res = isl_union_map_preimage_domain_multi_pw_aff(copy(), mpa.release());
  return manage(res);
}

isl::union_map union_map::preimage_domain_pw_multi_aff(isl::pw_multi_aff pma) const {
  auto res = isl_union_map_preimage_domain_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::union_map union_map::preimage_domain_union_pw_multi_aff(isl::union_pw_multi_aff upma) const {
  auto res = isl_union_map_preimage_domain_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

isl::union_map union_map::preimage_range_multi_aff(isl::multi_aff ma) const {
  auto res = isl_union_map_preimage_range_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::union_map union_map::preimage_range_pw_multi_aff(isl::pw_multi_aff pma) const {
  auto res = isl_union_map_preimage_range_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::union_map union_map::preimage_range_union_pw_multi_aff(isl::union_pw_multi_aff upma) const {
  auto res = isl_union_map_preimage_range_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

isl::union_map union_map::product(isl::union_map umap2) const {
  auto res = isl_union_map_product(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::project_out(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_union_map_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::union_set union_map::range() const {
  auto res = isl_union_map_range(copy());
  return manage(res);
}

isl::union_map union_map::range_curry() const {
  auto res = isl_union_map_range_curry(copy());
  return manage(res);
}

isl::union_map union_map::range_factor_domain() const {
  auto res = isl_union_map_range_factor_domain(copy());
  return manage(res);
}

isl::union_map union_map::range_factor_range() const {
  auto res = isl_union_map_range_factor_range(copy());
  return manage(res);
}

isl::union_map union_map::range_map() const {
  auto res = isl_union_map_range_map(copy());
  return manage(res);
}

isl::union_map union_map::range_product(isl::union_map umap2) const {
  auto res = isl_union_map_range_product(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::remove_divs() const {
  auto res = isl_union_map_remove_divs(copy());
  return manage(res);
}

isl::union_map union_map::remove_redundancies() const {
  auto res = isl_union_map_remove_redundancies(copy());
  return manage(res);
}

isl::union_map union_map::reset_user() const {
  auto res = isl_union_map_reset_user(copy());
  return manage(res);
}

isl::union_map union_map::reverse() const {
  auto res = isl_union_map_reverse(copy());
  return manage(res);
}

isl::basic_map union_map::sample() const {
  auto res = isl_union_map_sample(copy());
  return manage(res);
}

isl::union_map union_map::simple_hull() const {
  auto res = isl_union_map_simple_hull(copy());
  return manage(res);
}

isl::union_map union_map::subtract(isl::union_map umap2) const {
  auto res = isl_union_map_subtract(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::subtract_domain(isl::union_set dom) const {
  auto res = isl_union_map_subtract_domain(copy(), dom.release());
  return manage(res);
}

isl::union_map union_map::subtract_range(isl::union_set dom) const {
  auto res = isl_union_map_subtract_range(copy(), dom.release());
  return manage(res);
}

isl::union_map union_map::uncurry() const {
  auto res = isl_union_map_uncurry(copy());
  return manage(res);
}

isl::union_map union_map::unite(isl::union_map umap2) const {
  auto res = isl_union_map_union(copy(), umap2.release());
  return manage(res);
}

isl::union_map union_map::universe() const {
  auto res = isl_union_map_universe(copy());
  return manage(res);
}

isl::union_set union_map::wrap() const {
  auto res = isl_union_map_wrap(copy());
  return manage(res);
}

isl::union_map union_map::zip() const {
  auto res = isl_union_map_zip(copy());
  return manage(res);
}

// implementations for isl::union_map_list
isl::union_map_list manage(__isl_take isl_union_map_list *ptr) {
  return union_map_list(ptr);
}
isl::union_map_list give(__isl_take isl_union_map_list *ptr) {
  return manage(ptr);
}


union_map_list::union_map_list()
    : ptr(nullptr) {}

union_map_list::union_map_list(const isl::union_map_list &obj)
    : ptr(obj.copy()) {}
union_map_list::union_map_list(std::nullptr_t)
    : ptr(nullptr) {}


union_map_list::union_map_list(__isl_take isl_union_map_list *ptr)
    : ptr(ptr) {}


union_map_list &union_map_list::operator=(isl::union_map_list obj) {
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
__isl_keep isl_union_map_list *union_map_list::keep() const {
  return get();
}

__isl_give isl_union_map_list *union_map_list::take() {
  return release();
}

union_map_list::operator bool() const {
  return !is_null();
}

isl::ctx union_map_list::get_ctx() const {
  return isl::ctx(isl_union_map_list_get_ctx(ptr));
}



void union_map_list::dump() const {
  isl_union_map_list_dump(get());
}



// implementations for isl::union_pw_aff
isl::union_pw_aff manage(__isl_take isl_union_pw_aff *ptr) {
  return union_pw_aff(ptr);
}
isl::union_pw_aff give(__isl_take isl_union_pw_aff *ptr) {
  return manage(ptr);
}


union_pw_aff::union_pw_aff()
    : ptr(nullptr) {}

union_pw_aff::union_pw_aff(const isl::union_pw_aff &obj)
    : ptr(obj.copy()) {}
union_pw_aff::union_pw_aff(std::nullptr_t)
    : ptr(nullptr) {}


union_pw_aff::union_pw_aff(__isl_take isl_union_pw_aff *ptr)
    : ptr(ptr) {}

union_pw_aff::union_pw_aff(isl::pw_aff pa) {
  auto res = isl_union_pw_aff_from_pw_aff(pa.release());
  ptr = res;
}
union_pw_aff::union_pw_aff(isl::union_set domain, isl::val v) {
  auto res = isl_union_pw_aff_val_on_domain(domain.release(), v.release());
  ptr = res;
}
union_pw_aff::union_pw_aff(isl::ctx ctx, const std::string &str) {
  auto res = isl_union_pw_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

union_pw_aff &union_pw_aff::operator=(isl::union_pw_aff obj) {
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
__isl_keep isl_union_pw_aff *union_pw_aff::keep() const {
  return get();
}

__isl_give isl_union_pw_aff *union_pw_aff::take() {
  return release();
}

union_pw_aff::operator bool() const {
  return !is_null();
}

isl::ctx union_pw_aff::get_ctx() const {
  return isl::ctx(isl_union_pw_aff_get_ctx(ptr));
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


isl::union_pw_aff union_pw_aff::add(isl::union_pw_aff upa2) const {
  auto res = isl_union_pw_aff_add(copy(), upa2.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::add_pw_aff(isl::pw_aff pa) const {
  auto res = isl_union_pw_aff_add_pw_aff(copy(), pa.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::aff_on_domain(isl::union_set domain, isl::aff aff) {
  auto res = isl_union_pw_aff_aff_on_domain(domain.release(), aff.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::align_params(isl::space model) const {
  auto res = isl_union_pw_aff_align_params(copy(), model.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::coalesce() const {
  auto res = isl_union_pw_aff_coalesce(copy());
  return manage(res);
}

unsigned int union_pw_aff::dim(isl::dim type) const {
  auto res = isl_union_pw_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::union_set union_pw_aff::domain() const {
  auto res = isl_union_pw_aff_domain(copy());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_union_pw_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::union_pw_aff union_pw_aff::empty(isl::space space) {
  auto res = isl_union_pw_aff_empty(space.release());
  return manage(res);
}

isl::pw_aff union_pw_aff::extract_pw_aff(isl::space space) const {
  auto res = isl_union_pw_aff_extract_pw_aff(get(), space.release());
  return manage(res);
}

int union_pw_aff::find_dim_by_name(isl::dim type, const std::string &name) const {
  auto res = isl_union_pw_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::union_pw_aff union_pw_aff::floor() const {
  auto res = isl_union_pw_aff_floor(copy());
  return manage(res);
}

isl::stat union_pw_aff::foreach_pw_aff(const std::function<isl::stat(isl::pw_aff)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_pw_aff *arg_0, void *arg_1) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::pw_aff)> **>(arg_1);
    stat ret = (*func)(isl::manage(arg_0));
    return isl_stat(ret);
  };
  auto res = isl_union_pw_aff_foreach_pw_aff(get(), fn_lambda, &fn_p);
  return isl::stat(res);
}

isl::space union_pw_aff::get_space() const {
  auto res = isl_union_pw_aff_get_space(get());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::gist(isl::union_set context) const {
  auto res = isl_union_pw_aff_gist(copy(), context.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::gist_params(isl::set context) const {
  auto res = isl_union_pw_aff_gist_params(copy(), context.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::intersect_domain(isl::union_set uset) const {
  auto res = isl_union_pw_aff_intersect_domain(copy(), uset.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::intersect_params(isl::set set) const {
  auto res = isl_union_pw_aff_intersect_params(copy(), set.release());
  return manage(res);
}

isl::boolean union_pw_aff::involves_nan() const {
  auto res = isl_union_pw_aff_involves_nan(get());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::mod_val(isl::val f) const {
  auto res = isl_union_pw_aff_mod_val(copy(), f.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::neg() const {
  auto res = isl_union_pw_aff_neg(copy());
  return manage(res);
}

isl::boolean union_pw_aff::plain_is_equal(const isl::union_pw_aff &upa2) const {
  auto res = isl_union_pw_aff_plain_is_equal(get(), upa2.get());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::pullback(isl::union_pw_multi_aff upma) const {
  auto res = isl_union_pw_aff_pullback_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::reset_user() const {
  auto res = isl_union_pw_aff_reset_user(copy());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::scale_down_val(isl::val v) const {
  auto res = isl_union_pw_aff_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::scale_val(isl::val v) const {
  auto res = isl_union_pw_aff_scale_val(copy(), v.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::sub(isl::union_pw_aff upa2) const {
  auto res = isl_union_pw_aff_sub(copy(), upa2.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::subtract_domain(isl::union_set uset) const {
  auto res = isl_union_pw_aff_subtract_domain(copy(), uset.release());
  return manage(res);
}

isl::union_pw_aff union_pw_aff::union_add(isl::union_pw_aff upa2) const {
  auto res = isl_union_pw_aff_union_add(copy(), upa2.release());
  return manage(res);
}

isl::union_set union_pw_aff::zero_union_set() const {
  auto res = isl_union_pw_aff_zero_union_set(copy());
  return manage(res);
}

// implementations for isl::union_pw_aff_list
isl::union_pw_aff_list manage(__isl_take isl_union_pw_aff_list *ptr) {
  return union_pw_aff_list(ptr);
}
isl::union_pw_aff_list give(__isl_take isl_union_pw_aff_list *ptr) {
  return manage(ptr);
}


union_pw_aff_list::union_pw_aff_list()
    : ptr(nullptr) {}

union_pw_aff_list::union_pw_aff_list(const isl::union_pw_aff_list &obj)
    : ptr(obj.copy()) {}
union_pw_aff_list::union_pw_aff_list(std::nullptr_t)
    : ptr(nullptr) {}


union_pw_aff_list::union_pw_aff_list(__isl_take isl_union_pw_aff_list *ptr)
    : ptr(ptr) {}


union_pw_aff_list &union_pw_aff_list::operator=(isl::union_pw_aff_list obj) {
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
__isl_keep isl_union_pw_aff_list *union_pw_aff_list::keep() const {
  return get();
}

__isl_give isl_union_pw_aff_list *union_pw_aff_list::take() {
  return release();
}

union_pw_aff_list::operator bool() const {
  return !is_null();
}

isl::ctx union_pw_aff_list::get_ctx() const {
  return isl::ctx(isl_union_pw_aff_list_get_ctx(ptr));
}



void union_pw_aff_list::dump() const {
  isl_union_pw_aff_list_dump(get());
}



// implementations for isl::union_pw_multi_aff
isl::union_pw_multi_aff manage(__isl_take isl_union_pw_multi_aff *ptr) {
  return union_pw_multi_aff(ptr);
}
isl::union_pw_multi_aff give(__isl_take isl_union_pw_multi_aff *ptr) {
  return manage(ptr);
}


union_pw_multi_aff::union_pw_multi_aff()
    : ptr(nullptr) {}

union_pw_multi_aff::union_pw_multi_aff(const isl::union_pw_multi_aff &obj)
    : ptr(obj.copy()) {}
union_pw_multi_aff::union_pw_multi_aff(std::nullptr_t)
    : ptr(nullptr) {}


union_pw_multi_aff::union_pw_multi_aff(__isl_take isl_union_pw_multi_aff *ptr)
    : ptr(ptr) {}

union_pw_multi_aff::union_pw_multi_aff(isl::pw_multi_aff pma) {
  auto res = isl_union_pw_multi_aff_from_pw_multi_aff(pma.release());
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(isl::union_set uset) {
  auto res = isl_union_pw_multi_aff_from_domain(uset.release());
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(isl::union_map umap) {
  auto res = isl_union_pw_multi_aff_from_union_map(umap.release());
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(isl::ctx ctx, const std::string &str) {
  auto res = isl_union_pw_multi_aff_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(isl::union_pw_aff upa) {
  auto res = isl_union_pw_multi_aff_from_union_pw_aff(upa.release());
  ptr = res;
}

union_pw_multi_aff &union_pw_multi_aff::operator=(isl::union_pw_multi_aff obj) {
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
__isl_keep isl_union_pw_multi_aff *union_pw_multi_aff::keep() const {
  return get();
}

__isl_give isl_union_pw_multi_aff *union_pw_multi_aff::take() {
  return release();
}

union_pw_multi_aff::operator bool() const {
  return !is_null();
}

isl::ctx union_pw_multi_aff::get_ctx() const {
  return isl::ctx(isl_union_pw_multi_aff_get_ctx(ptr));
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


isl::union_pw_multi_aff union_pw_multi_aff::add(isl::union_pw_multi_aff upma2) const {
  auto res = isl_union_pw_multi_aff_add(copy(), upma2.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::add_pw_multi_aff(isl::pw_multi_aff pma) const {
  auto res = isl_union_pw_multi_aff_add_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::align_params(isl::space model) const {
  auto res = isl_union_pw_multi_aff_align_params(copy(), model.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::coalesce() const {
  auto res = isl_union_pw_multi_aff_coalesce(copy());
  return manage(res);
}

unsigned int union_pw_multi_aff::dim(isl::dim type) const {
  auto res = isl_union_pw_multi_aff_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::union_set union_pw_multi_aff::domain() const {
  auto res = isl_union_pw_multi_aff_domain(copy());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::drop_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_union_pw_multi_aff_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::empty(isl::space space) {
  auto res = isl_union_pw_multi_aff_empty(space.release());
  return manage(res);
}

isl::pw_multi_aff union_pw_multi_aff::extract_pw_multi_aff(isl::space space) const {
  auto res = isl_union_pw_multi_aff_extract_pw_multi_aff(get(), space.release());
  return manage(res);
}

int union_pw_multi_aff::find_dim_by_name(isl::dim type, const std::string &name) const {
  auto res = isl_union_pw_multi_aff_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::union_pw_multi_aff union_pw_multi_aff::flat_range_product(isl::union_pw_multi_aff upma2) const {
  auto res = isl_union_pw_multi_aff_flat_range_product(copy(), upma2.release());
  return manage(res);
}

isl::stat union_pw_multi_aff::foreach_pw_multi_aff(const std::function<isl::stat(isl::pw_multi_aff)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_pw_multi_aff *arg_0, void *arg_1) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::pw_multi_aff)> **>(arg_1);
    stat ret = (*func)(isl::manage(arg_0));
    return isl_stat(ret);
  };
  auto res = isl_union_pw_multi_aff_foreach_pw_multi_aff(get(), fn_lambda, &fn_p);
  return isl::stat(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::from_aff(isl::aff aff) {
  auto res = isl_union_pw_multi_aff_from_aff(aff.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::from_multi_union_pw_aff(isl::multi_union_pw_aff mupa) {
  auto res = isl_union_pw_multi_aff_from_multi_union_pw_aff(mupa.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::from_union_set(isl::union_set uset) {
  auto res = isl_union_pw_multi_aff_from_union_set(uset.release());
  return manage(res);
}

isl::space union_pw_multi_aff::get_space() const {
  auto res = isl_union_pw_multi_aff_get_space(get());
  return manage(res);
}

isl::union_pw_aff union_pw_multi_aff::get_union_pw_aff(int pos) const {
  auto res = isl_union_pw_multi_aff_get_union_pw_aff(get(), pos);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::gist(isl::union_set context) const {
  auto res = isl_union_pw_multi_aff_gist(copy(), context.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::gist_params(isl::set context) const {
  auto res = isl_union_pw_multi_aff_gist_params(copy(), context.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::intersect_domain(isl::union_set uset) const {
  auto res = isl_union_pw_multi_aff_intersect_domain(copy(), uset.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::intersect_params(isl::set set) const {
  auto res = isl_union_pw_multi_aff_intersect_params(copy(), set.release());
  return manage(res);
}

isl::boolean union_pw_multi_aff::involves_nan() const {
  auto res = isl_union_pw_multi_aff_involves_nan(get());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::multi_val_on_domain(isl::union_set domain, isl::multi_val mv) {
  auto res = isl_union_pw_multi_aff_multi_val_on_domain(domain.release(), mv.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::neg() const {
  auto res = isl_union_pw_multi_aff_neg(copy());
  return manage(res);
}

isl::boolean union_pw_multi_aff::plain_is_equal(const isl::union_pw_multi_aff &upma2) const {
  auto res = isl_union_pw_multi_aff_plain_is_equal(get(), upma2.get());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::pullback(isl::union_pw_multi_aff upma2) const {
  auto res = isl_union_pw_multi_aff_pullback_union_pw_multi_aff(copy(), upma2.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::reset_user() const {
  auto res = isl_union_pw_multi_aff_reset_user(copy());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::scale_down_val(isl::val val) const {
  auto res = isl_union_pw_multi_aff_scale_down_val(copy(), val.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::scale_multi_val(isl::multi_val mv) const {
  auto res = isl_union_pw_multi_aff_scale_multi_val(copy(), mv.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::scale_val(isl::val val) const {
  auto res = isl_union_pw_multi_aff_scale_val(copy(), val.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::sub(isl::union_pw_multi_aff upma2) const {
  auto res = isl_union_pw_multi_aff_sub(copy(), upma2.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::subtract_domain(isl::union_set uset) const {
  auto res = isl_union_pw_multi_aff_subtract_domain(copy(), uset.release());
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::union_add(isl::union_pw_multi_aff upma2) const {
  auto res = isl_union_pw_multi_aff_union_add(copy(), upma2.release());
  return manage(res);
}

// implementations for isl::union_pw_multi_aff_list
isl::union_pw_multi_aff_list manage(__isl_take isl_union_pw_multi_aff_list *ptr) {
  return union_pw_multi_aff_list(ptr);
}
isl::union_pw_multi_aff_list give(__isl_take isl_union_pw_multi_aff_list *ptr) {
  return manage(ptr);
}


union_pw_multi_aff_list::union_pw_multi_aff_list()
    : ptr(nullptr) {}

union_pw_multi_aff_list::union_pw_multi_aff_list(const isl::union_pw_multi_aff_list &obj)
    : ptr(obj.copy()) {}
union_pw_multi_aff_list::union_pw_multi_aff_list(std::nullptr_t)
    : ptr(nullptr) {}


union_pw_multi_aff_list::union_pw_multi_aff_list(__isl_take isl_union_pw_multi_aff_list *ptr)
    : ptr(ptr) {}


union_pw_multi_aff_list &union_pw_multi_aff_list::operator=(isl::union_pw_multi_aff_list obj) {
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
__isl_keep isl_union_pw_multi_aff_list *union_pw_multi_aff_list::keep() const {
  return get();
}

__isl_give isl_union_pw_multi_aff_list *union_pw_multi_aff_list::take() {
  return release();
}

union_pw_multi_aff_list::operator bool() const {
  return !is_null();
}

isl::ctx union_pw_multi_aff_list::get_ctx() const {
  return isl::ctx(isl_union_pw_multi_aff_list_get_ctx(ptr));
}



void union_pw_multi_aff_list::dump() const {
  isl_union_pw_multi_aff_list_dump(get());
}



// implementations for isl::union_pw_qpolynomial
isl::union_pw_qpolynomial manage(__isl_take isl_union_pw_qpolynomial *ptr) {
  return union_pw_qpolynomial(ptr);
}
isl::union_pw_qpolynomial give(__isl_take isl_union_pw_qpolynomial *ptr) {
  return manage(ptr);
}


union_pw_qpolynomial::union_pw_qpolynomial()
    : ptr(nullptr) {}

union_pw_qpolynomial::union_pw_qpolynomial(const isl::union_pw_qpolynomial &obj)
    : ptr(obj.copy()) {}
union_pw_qpolynomial::union_pw_qpolynomial(std::nullptr_t)
    : ptr(nullptr) {}


union_pw_qpolynomial::union_pw_qpolynomial(__isl_take isl_union_pw_qpolynomial *ptr)
    : ptr(ptr) {}

union_pw_qpolynomial::union_pw_qpolynomial(isl::ctx ctx, const std::string &str) {
  auto res = isl_union_pw_qpolynomial_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

union_pw_qpolynomial &union_pw_qpolynomial::operator=(isl::union_pw_qpolynomial obj) {
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
__isl_keep isl_union_pw_qpolynomial *union_pw_qpolynomial::keep() const {
  return get();
}

__isl_give isl_union_pw_qpolynomial *union_pw_qpolynomial::take() {
  return release();
}

union_pw_qpolynomial::operator bool() const {
  return !is_null();
}

isl::ctx union_pw_qpolynomial::get_ctx() const {
  return isl::ctx(isl_union_pw_qpolynomial_get_ctx(ptr));
}


std::string union_pw_qpolynomial::to_str() const {
  char *Tmp = isl_union_pw_qpolynomial_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}



isl::union_pw_qpolynomial union_pw_qpolynomial::add(isl::union_pw_qpolynomial upwqp2) const {
  auto res = isl_union_pw_qpolynomial_add(copy(), upwqp2.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::add_pw_qpolynomial(isl::pw_qpolynomial pwqp) const {
  auto res = isl_union_pw_qpolynomial_add_pw_qpolynomial(copy(), pwqp.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::align_params(isl::space model) const {
  auto res = isl_union_pw_qpolynomial_align_params(copy(), model.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::coalesce() const {
  auto res = isl_union_pw_qpolynomial_coalesce(copy());
  return manage(res);
}

unsigned int union_pw_qpolynomial::dim(isl::dim type) const {
  auto res = isl_union_pw_qpolynomial_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::union_set union_pw_qpolynomial::domain() const {
  auto res = isl_union_pw_qpolynomial_domain(copy());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::drop_dims(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_union_pw_qpolynomial_drop_dims(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::val union_pw_qpolynomial::eval(isl::point pnt) const {
  auto res = isl_union_pw_qpolynomial_eval(copy(), pnt.release());
  return manage(res);
}

isl::pw_qpolynomial union_pw_qpolynomial::extract_pw_qpolynomial(isl::space dim) const {
  auto res = isl_union_pw_qpolynomial_extract_pw_qpolynomial(get(), dim.release());
  return manage(res);
}

int union_pw_qpolynomial::find_dim_by_name(isl::dim type, const std::string &name) const {
  auto res = isl_union_pw_qpolynomial_find_dim_by_name(get(), static_cast<enum isl_dim_type>(type), name.c_str());
  return res;
}

isl::stat union_pw_qpolynomial::foreach_pw_qpolynomial(const std::function<isl::stat(isl::pw_qpolynomial)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_pw_qpolynomial *arg_0, void *arg_1) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::pw_qpolynomial)> **>(arg_1);
    stat ret = (*func)(isl::manage(arg_0));
    return isl_stat(ret);
  };
  auto res = isl_union_pw_qpolynomial_foreach_pw_qpolynomial(get(), fn_lambda, &fn_p);
  return isl::stat(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::from_pw_qpolynomial(isl::pw_qpolynomial pwqp) {
  auto res = isl_union_pw_qpolynomial_from_pw_qpolynomial(pwqp.release());
  return manage(res);
}

isl::space union_pw_qpolynomial::get_space() const {
  auto res = isl_union_pw_qpolynomial_get_space(get());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::gist(isl::union_set context) const {
  auto res = isl_union_pw_qpolynomial_gist(copy(), context.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::gist_params(isl::set context) const {
  auto res = isl_union_pw_qpolynomial_gist_params(copy(), context.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::intersect_domain(isl::union_set uset) const {
  auto res = isl_union_pw_qpolynomial_intersect_domain(copy(), uset.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::intersect_params(isl::set set) const {
  auto res = isl_union_pw_qpolynomial_intersect_params(copy(), set.release());
  return manage(res);
}

isl::boolean union_pw_qpolynomial::involves_nan() const {
  auto res = isl_union_pw_qpolynomial_involves_nan(get());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::mul(isl::union_pw_qpolynomial upwqp2) const {
  auto res = isl_union_pw_qpolynomial_mul(copy(), upwqp2.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::neg() const {
  auto res = isl_union_pw_qpolynomial_neg(copy());
  return manage(res);
}

isl::boolean union_pw_qpolynomial::plain_is_equal(const isl::union_pw_qpolynomial &upwqp2) const {
  auto res = isl_union_pw_qpolynomial_plain_is_equal(get(), upwqp2.get());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::reset_user() const {
  auto res = isl_union_pw_qpolynomial_reset_user(copy());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::scale_down_val(isl::val v) const {
  auto res = isl_union_pw_qpolynomial_scale_down_val(copy(), v.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::scale_val(isl::val v) const {
  auto res = isl_union_pw_qpolynomial_scale_val(copy(), v.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::sub(isl::union_pw_qpolynomial upwqp2) const {
  auto res = isl_union_pw_qpolynomial_sub(copy(), upwqp2.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::subtract_domain(isl::union_set uset) const {
  auto res = isl_union_pw_qpolynomial_subtract_domain(copy(), uset.release());
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::to_polynomial(int sign) const {
  auto res = isl_union_pw_qpolynomial_to_polynomial(copy(), sign);
  return manage(res);
}

isl::union_pw_qpolynomial union_pw_qpolynomial::zero(isl::space dim) {
  auto res = isl_union_pw_qpolynomial_zero(dim.release());
  return manage(res);
}

// implementations for isl::union_set
isl::union_set manage(__isl_take isl_union_set *ptr) {
  return union_set(ptr);
}
isl::union_set give(__isl_take isl_union_set *ptr) {
  return manage(ptr);
}


union_set::union_set()
    : ptr(nullptr) {}

union_set::union_set(const isl::union_set &obj)
    : ptr(obj.copy()) {}
union_set::union_set(std::nullptr_t)
    : ptr(nullptr) {}


union_set::union_set(__isl_take isl_union_set *ptr)
    : ptr(ptr) {}

union_set::union_set(isl::point pnt) {
  auto res = isl_union_set_from_point(pnt.release());
  ptr = res;
}
union_set::union_set(isl::ctx ctx, const std::string &str) {
  auto res = isl_union_set_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}
union_set::union_set(isl::basic_set bset) {
  auto res = isl_union_set_from_basic_set(bset.release());
  ptr = res;
}
union_set::union_set(isl::set set) {
  auto res = isl_union_set_from_set(set.release());
  ptr = res;
}

union_set &union_set::operator=(isl::union_set obj) {
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
__isl_keep isl_union_set *union_set::keep() const {
  return get();
}

__isl_give isl_union_set *union_set::take() {
  return release();
}

union_set::operator bool() const {
  return !is_null();
}

isl::ctx union_set::get_ctx() const {
  return isl::ctx(isl_union_set_get_ctx(ptr));
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


isl::union_set union_set::add_set(isl::set set) const {
  auto res = isl_union_set_add_set(copy(), set.release());
  return manage(res);
}

isl::union_set union_set::affine_hull() const {
  auto res = isl_union_set_affine_hull(copy());
  return manage(res);
}

isl::union_set union_set::align_params(isl::space model) const {
  auto res = isl_union_set_align_params(copy(), model.release());
  return manage(res);
}

isl::union_set union_set::apply(isl::union_map umap) const {
  auto res = isl_union_set_apply(copy(), umap.release());
  return manage(res);
}

isl::union_set union_set::coalesce() const {
  auto res = isl_union_set_coalesce(copy());
  return manage(res);
}

isl::union_set union_set::coefficients() const {
  auto res = isl_union_set_coefficients(copy());
  return manage(res);
}

isl::schedule union_set::compute_schedule(isl::union_map validity, isl::union_map proximity) const {
  auto res = isl_union_set_compute_schedule(copy(), validity.release(), proximity.release());
  return manage(res);
}

isl::boolean union_set::contains(const isl::space &space) const {
  auto res = isl_union_set_contains(get(), space.get());
  return manage(res);
}

isl::union_set union_set::detect_equalities() const {
  auto res = isl_union_set_detect_equalities(copy());
  return manage(res);
}

unsigned int union_set::dim(isl::dim type) const {
  auto res = isl_union_set_dim(get(), static_cast<enum isl_dim_type>(type));
  return res;
}

isl::union_set union_set::empty(isl::space dim) {
  auto res = isl_union_set_empty(dim.release());
  return manage(res);
}

isl::set union_set::extract_set(isl::space dim) const {
  auto res = isl_union_set_extract_set(get(), dim.release());
  return manage(res);
}

isl::stat union_set::foreach_point(const std::function<isl::stat(isl::point)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_point *arg_0, void *arg_1) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::point)> **>(arg_1);
    stat ret = (*func)(isl::manage(arg_0));
    return isl_stat(ret);
  };
  auto res = isl_union_set_foreach_point(get(), fn_lambda, &fn_p);
  return isl::stat(res);
}

isl::stat union_set::foreach_set(const std::function<isl::stat(isl::set)> &fn) const {
  auto fn_p = &fn;
  auto fn_lambda = [](isl_set *arg_0, void *arg_1) -> isl_stat {
    auto *func = *static_cast<const std::function<isl::stat(isl::set)> **>(arg_1);
    stat ret = (*func)(isl::manage(arg_0));
    return isl_stat(ret);
  };
  auto res = isl_union_set_foreach_set(get(), fn_lambda, &fn_p);
  return isl::stat(res);
}

isl::basic_set_list union_set::get_basic_set_list() const {
  auto res = isl_union_set_get_basic_set_list(get());
  return manage(res);
}

uint32_t union_set::get_hash() const {
  auto res = isl_union_set_get_hash(get());
  return res;
}

isl::space union_set::get_space() const {
  auto res = isl_union_set_get_space(get());
  return manage(res);
}

isl::union_set union_set::gist(isl::union_set context) const {
  auto res = isl_union_set_gist(copy(), context.release());
  return manage(res);
}

isl::union_set union_set::gist_params(isl::set set) const {
  auto res = isl_union_set_gist_params(copy(), set.release());
  return manage(res);
}

isl::union_map union_set::identity() const {
  auto res = isl_union_set_identity(copy());
  return manage(res);
}

isl::union_pw_multi_aff union_set::identity_union_pw_multi_aff() const {
  auto res = isl_union_set_identity_union_pw_multi_aff(copy());
  return manage(res);
}

isl::union_set union_set::intersect(isl::union_set uset2) const {
  auto res = isl_union_set_intersect(copy(), uset2.release());
  return manage(res);
}

isl::union_set union_set::intersect_params(isl::set set) const {
  auto res = isl_union_set_intersect_params(copy(), set.release());
  return manage(res);
}

isl::boolean union_set::is_disjoint(const isl::union_set &uset2) const {
  auto res = isl_union_set_is_disjoint(get(), uset2.get());
  return manage(res);
}

isl::boolean union_set::is_empty() const {
  auto res = isl_union_set_is_empty(get());
  return manage(res);
}

isl::boolean union_set::is_equal(const isl::union_set &uset2) const {
  auto res = isl_union_set_is_equal(get(), uset2.get());
  return manage(res);
}

isl::boolean union_set::is_params() const {
  auto res = isl_union_set_is_params(get());
  return manage(res);
}

isl::boolean union_set::is_strict_subset(const isl::union_set &uset2) const {
  auto res = isl_union_set_is_strict_subset(get(), uset2.get());
  return manage(res);
}

isl::boolean union_set::is_subset(const isl::union_set &uset2) const {
  auto res = isl_union_set_is_subset(get(), uset2.get());
  return manage(res);
}

isl::union_map union_set::lex_ge_union_set(isl::union_set uset2) const {
  auto res = isl_union_set_lex_ge_union_set(copy(), uset2.release());
  return manage(res);
}

isl::union_map union_set::lex_gt_union_set(isl::union_set uset2) const {
  auto res = isl_union_set_lex_gt_union_set(copy(), uset2.release());
  return manage(res);
}

isl::union_map union_set::lex_le_union_set(isl::union_set uset2) const {
  auto res = isl_union_set_lex_le_union_set(copy(), uset2.release());
  return manage(res);
}

isl::union_map union_set::lex_lt_union_set(isl::union_set uset2) const {
  auto res = isl_union_set_lex_lt_union_set(copy(), uset2.release());
  return manage(res);
}

isl::union_set union_set::lexmax() const {
  auto res = isl_union_set_lexmax(copy());
  return manage(res);
}

isl::union_set union_set::lexmin() const {
  auto res = isl_union_set_lexmin(copy());
  return manage(res);
}

isl::multi_val union_set::min_multi_union_pw_aff(const isl::multi_union_pw_aff &obj) const {
  auto res = isl_union_set_min_multi_union_pw_aff(get(), obj.get());
  return manage(res);
}

isl::set union_set::params() const {
  auto res = isl_union_set_params(copy());
  return manage(res);
}

isl::union_set union_set::polyhedral_hull() const {
  auto res = isl_union_set_polyhedral_hull(copy());
  return manage(res);
}

isl::union_set union_set::preimage_multi_aff(isl::multi_aff ma) const {
  auto res = isl_union_set_preimage_multi_aff(copy(), ma.release());
  return manage(res);
}

isl::union_set union_set::preimage_pw_multi_aff(isl::pw_multi_aff pma) const {
  auto res = isl_union_set_preimage_pw_multi_aff(copy(), pma.release());
  return manage(res);
}

isl::union_set union_set::preimage_union_pw_multi_aff(isl::union_pw_multi_aff upma) const {
  auto res = isl_union_set_preimage_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

isl::union_set union_set::product(isl::union_set uset2) const {
  auto res = isl_union_set_product(copy(), uset2.release());
  return manage(res);
}

isl::union_set union_set::project_out(isl::dim type, unsigned int first, unsigned int n) const {
  auto res = isl_union_set_project_out(copy(), static_cast<enum isl_dim_type>(type), first, n);
  return manage(res);
}

isl::union_set union_set::remove_divs() const {
  auto res = isl_union_set_remove_divs(copy());
  return manage(res);
}

isl::union_set union_set::remove_redundancies() const {
  auto res = isl_union_set_remove_redundancies(copy());
  return manage(res);
}

isl::union_set union_set::reset_user() const {
  auto res = isl_union_set_reset_user(copy());
  return manage(res);
}

isl::basic_set union_set::sample() const {
  auto res = isl_union_set_sample(copy());
  return manage(res);
}

isl::point union_set::sample_point() const {
  auto res = isl_union_set_sample_point(copy());
  return manage(res);
}

isl::union_set union_set::simple_hull() const {
  auto res = isl_union_set_simple_hull(copy());
  return manage(res);
}

isl::union_set union_set::solutions() const {
  auto res = isl_union_set_solutions(copy());
  return manage(res);
}

isl::union_set union_set::subtract(isl::union_set uset2) const {
  auto res = isl_union_set_subtract(copy(), uset2.release());
  return manage(res);
}

isl::union_set union_set::unite(isl::union_set uset2) const {
  auto res = isl_union_set_union(copy(), uset2.release());
  return manage(res);
}

isl::union_set union_set::universe() const {
  auto res = isl_union_set_universe(copy());
  return manage(res);
}

isl::union_map union_set::unwrap() const {
  auto res = isl_union_set_unwrap(copy());
  return manage(res);
}

isl::union_map union_set::wrapped_domain_map() const {
  auto res = isl_union_set_wrapped_domain_map(copy());
  return manage(res);
}

// implementations for isl::union_set_list
isl::union_set_list manage(__isl_take isl_union_set_list *ptr) {
  return union_set_list(ptr);
}
isl::union_set_list give(__isl_take isl_union_set_list *ptr) {
  return manage(ptr);
}


union_set_list::union_set_list()
    : ptr(nullptr) {}

union_set_list::union_set_list(const isl::union_set_list &obj)
    : ptr(obj.copy()) {}
union_set_list::union_set_list(std::nullptr_t)
    : ptr(nullptr) {}


union_set_list::union_set_list(__isl_take isl_union_set_list *ptr)
    : ptr(ptr) {}


union_set_list &union_set_list::operator=(isl::union_set_list obj) {
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
__isl_keep isl_union_set_list *union_set_list::keep() const {
  return get();
}

__isl_give isl_union_set_list *union_set_list::take() {
  return release();
}

union_set_list::operator bool() const {
  return !is_null();
}

isl::ctx union_set_list::get_ctx() const {
  return isl::ctx(isl_union_set_list_get_ctx(ptr));
}



void union_set_list::dump() const {
  isl_union_set_list_dump(get());
}



// implementations for isl::val
isl::val manage(__isl_take isl_val *ptr) {
  return val(ptr);
}
isl::val give(__isl_take isl_val *ptr) {
  return manage(ptr);
}


val::val()
    : ptr(nullptr) {}

val::val(const isl::val &obj)
    : ptr(obj.copy()) {}
val::val(std::nullptr_t)
    : ptr(nullptr) {}


val::val(__isl_take isl_val *ptr)
    : ptr(ptr) {}

val::val(isl::ctx ctx, long i) {
  auto res = isl_val_int_from_si(ctx.release(), i);
  ptr = res;
}
val::val(isl::ctx ctx, const std::string &str) {
  auto res = isl_val_read_from_str(ctx.release(), str.c_str());
  ptr = res;
}

val &val::operator=(isl::val obj) {
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
__isl_keep isl_val *val::keep() const {
  return get();
}

__isl_give isl_val *val::take() {
  return release();
}

val::operator bool() const {
  return !is_null();
}

isl::ctx val::get_ctx() const {
  return isl::ctx(isl_val_get_ctx(ptr));
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


isl::val val::two_exp() const {
  auto res = isl_val_2exp(copy());
  return manage(res);
}

isl::val val::abs() const {
  auto res = isl_val_abs(copy());
  return manage(res);
}

isl::boolean val::abs_eq(const isl::val &v2) const {
  auto res = isl_val_abs_eq(get(), v2.get());
  return manage(res);
}

isl::val val::add(isl::val v2) const {
  auto res = isl_val_add(copy(), v2.release());
  return manage(res);
}

isl::val val::add_ui(unsigned long v2) const {
  auto res = isl_val_add_ui(copy(), v2);
  return manage(res);
}

isl::val val::ceil() const {
  auto res = isl_val_ceil(copy());
  return manage(res);
}

int val::cmp_si(long i) const {
  auto res = isl_val_cmp_si(get(), i);
  return res;
}

isl::val val::div(isl::val v2) const {
  auto res = isl_val_div(copy(), v2.release());
  return manage(res);
}

isl::val val::div_ui(unsigned long v2) const {
  auto res = isl_val_div_ui(copy(), v2);
  return manage(res);
}

isl::boolean val::eq(const isl::val &v2) const {
  auto res = isl_val_eq(get(), v2.get());
  return manage(res);
}

isl::val val::floor() const {
  auto res = isl_val_floor(copy());
  return manage(res);
}

isl::val val::gcd(isl::val v2) const {
  auto res = isl_val_gcd(copy(), v2.release());
  return manage(res);
}

isl::boolean val::ge(const isl::val &v2) const {
  auto res = isl_val_ge(get(), v2.get());
  return manage(res);
}

uint32_t val::get_hash() const {
  auto res = isl_val_get_hash(get());
  return res;
}

long val::get_num_si() const {
  auto res = isl_val_get_num_si(get());
  return res;
}

isl::boolean val::gt(const isl::val &v2) const {
  auto res = isl_val_gt(get(), v2.get());
  return manage(res);
}

isl::val val::infty(isl::ctx ctx) {
  auto res = isl_val_infty(ctx.release());
  return manage(res);
}

isl::val val::int_from_ui(isl::ctx ctx, unsigned long u) {
  auto res = isl_val_int_from_ui(ctx.release(), u);
  return manage(res);
}

isl::val val::inv() const {
  auto res = isl_val_inv(copy());
  return manage(res);
}

isl::boolean val::is_divisible_by(const isl::val &v2) const {
  auto res = isl_val_is_divisible_by(get(), v2.get());
  return manage(res);
}

isl::boolean val::is_infty() const {
  auto res = isl_val_is_infty(get());
  return manage(res);
}

isl::boolean val::is_int() const {
  auto res = isl_val_is_int(get());
  return manage(res);
}

isl::boolean val::is_nan() const {
  auto res = isl_val_is_nan(get());
  return manage(res);
}

isl::boolean val::is_neg() const {
  auto res = isl_val_is_neg(get());
  return manage(res);
}

isl::boolean val::is_neginfty() const {
  auto res = isl_val_is_neginfty(get());
  return manage(res);
}

isl::boolean val::is_negone() const {
  auto res = isl_val_is_negone(get());
  return manage(res);
}

isl::boolean val::is_nonneg() const {
  auto res = isl_val_is_nonneg(get());
  return manage(res);
}

isl::boolean val::is_nonpos() const {
  auto res = isl_val_is_nonpos(get());
  return manage(res);
}

isl::boolean val::is_one() const {
  auto res = isl_val_is_one(get());
  return manage(res);
}

isl::boolean val::is_pos() const {
  auto res = isl_val_is_pos(get());
  return manage(res);
}

isl::boolean val::is_rat() const {
  auto res = isl_val_is_rat(get());
  return manage(res);
}

isl::boolean val::is_zero() const {
  auto res = isl_val_is_zero(get());
  return manage(res);
}

isl::boolean val::le(const isl::val &v2) const {
  auto res = isl_val_le(get(), v2.get());
  return manage(res);
}

isl::boolean val::lt(const isl::val &v2) const {
  auto res = isl_val_lt(get(), v2.get());
  return manage(res);
}

isl::val val::max(isl::val v2) const {
  auto res = isl_val_max(copy(), v2.release());
  return manage(res);
}

isl::val val::min(isl::val v2) const {
  auto res = isl_val_min(copy(), v2.release());
  return manage(res);
}

isl::val val::mod(isl::val v2) const {
  auto res = isl_val_mod(copy(), v2.release());
  return manage(res);
}

isl::val val::mul(isl::val v2) const {
  auto res = isl_val_mul(copy(), v2.release());
  return manage(res);
}

isl::val val::mul_ui(unsigned long v2) const {
  auto res = isl_val_mul_ui(copy(), v2);
  return manage(res);
}

isl::val val::nan(isl::ctx ctx) {
  auto res = isl_val_nan(ctx.release());
  return manage(res);
}

isl::boolean val::ne(const isl::val &v2) const {
  auto res = isl_val_ne(get(), v2.get());
  return manage(res);
}

isl::val val::neg() const {
  auto res = isl_val_neg(copy());
  return manage(res);
}

isl::val val::neginfty(isl::ctx ctx) {
  auto res = isl_val_neginfty(ctx.release());
  return manage(res);
}

isl::val val::negone(isl::ctx ctx) {
  auto res = isl_val_negone(ctx.release());
  return manage(res);
}

isl::val val::one(isl::ctx ctx) {
  auto res = isl_val_one(ctx.release());
  return manage(res);
}

isl::val val::set_si(long i) const {
  auto res = isl_val_set_si(copy(), i);
  return manage(res);
}

int val::sgn() const {
  auto res = isl_val_sgn(get());
  return res;
}

isl::val val::sub(isl::val v2) const {
  auto res = isl_val_sub(copy(), v2.release());
  return manage(res);
}

isl::val val::sub_ui(unsigned long v2) const {
  auto res = isl_val_sub_ui(copy(), v2);
  return manage(res);
}

isl::val val::trunc() const {
  auto res = isl_val_trunc(copy());
  return manage(res);
}

isl::val val::zero(isl::ctx ctx) {
  auto res = isl_val_zero(ctx.release());
  return manage(res);
}

// implementations for isl::val_list
isl::val_list manage(__isl_take isl_val_list *ptr) {
  return val_list(ptr);
}
isl::val_list give(__isl_take isl_val_list *ptr) {
  return manage(ptr);
}


val_list::val_list()
    : ptr(nullptr) {}

val_list::val_list(const isl::val_list &obj)
    : ptr(obj.copy()) {}
val_list::val_list(std::nullptr_t)
    : ptr(nullptr) {}


val_list::val_list(__isl_take isl_val_list *ptr)
    : ptr(ptr) {}


val_list &val_list::operator=(isl::val_list obj) {
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
__isl_keep isl_val_list *val_list::keep() const {
  return get();
}

__isl_give isl_val_list *val_list::take() {
  return release();
}

val_list::operator bool() const {
  return !is_null();
}

isl::ctx val_list::get_ctx() const {
  return isl::ctx(isl_val_list_get_ctx(ptr));
}



void val_list::dump() const {
  isl_val_list_dump(get());
}


} // namespace noexceptions
} // namespace isl

#endif /* ISL_CPP_NOEXCEPTIONS */
