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
#include <isl/map.h>
#include <isl/ilp.h>
#include <isl/union_set.h>
#include <isl/union_map.h>
#include <isl/flow.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/ast_build.h>

#include <stdio.h>
#include <stdlib.h>

#include <functional>
#include <string>

namespace isl {
namespace checked {

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
    ISLPP_ASSERT(checked, "IMPLEMENTATION ERROR: Unchecked state");
  }

  /* implicit */ boolean(bool val)
      : val(val ? isl_bool_true : isl_bool_false) {}

  bool is_error() const { checked = true; return val == isl_bool_error; }
  bool is_false() const { checked = true; return val == isl_bool_false; }
  bool is_true() const { checked = true; return val == isl_bool_true; }

  explicit operator bool() const {
    ISLPP_ASSERT(checked, "IMPLEMENTATION ERROR: Unchecked error state");
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
	constexpr stat(isl_stat val) : val(val) {}
public:
	static stat ok() {
		return stat(isl_stat_ok);
	}
	static stat error() {
		return stat(isl_stat_error);
	}
	stat() : val(isl_stat_error) {}
	~stat() {
		ISLPP_ASSERT(checked, "IMPLEMENTATION ERROR: Unchecked state");
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

}
} // namespace isl

namespace isl {

namespace checked {

// forward declarations
class aff;
class ast_build;
class ast_expr;
class ast_node;
class basic_map;
class basic_set;
class map;
class multi_aff;
class multi_pw_aff;
class multi_union_pw_aff;
class multi_val;
class point;
class pw_aff;
class pw_multi_aff;
class schedule;
class schedule_constraints;
class schedule_node;
class set;
class union_access_info;
class union_flow;
class union_map;
class union_pw_aff;
class union_pw_multi_aff;
class union_set;
class val;

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
  inline explicit aff(ctx ctx, const std::string &str);
  inline aff &operator=(aff obj);
  inline ~aff();
  inline __isl_give isl_aff *copy() const &;
  inline __isl_give isl_aff *copy() && = delete;
  inline __isl_keep isl_aff *get() const;
  inline __isl_give isl_aff *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

  inline aff add(aff aff2) const;
  inline aff ceil() const;
  inline aff div(aff aff2) const;
  inline set eq_set(aff aff2) const;
  inline aff floor() const;
  inline set ge_set(aff aff2) const;
  inline set gt_set(aff aff2) const;
  inline set le_set(aff aff2) const;
  inline set lt_set(aff aff2) const;
  inline aff mod(val mod) const;
  inline aff mul(aff aff2) const;
  inline set ne_set(aff aff2) const;
  inline aff neg() const;
  inline aff pullback(multi_aff ma) const;
  inline aff scale(val v) const;
  inline aff scale_down(val v) const;
  inline aff sub(aff aff2) const;
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
  inline explicit ast_build(ctx ctx);
  inline ast_build &operator=(ast_build obj);
  inline ~ast_build();
  inline __isl_give isl_ast_build *copy() const &;
  inline __isl_give isl_ast_build *copy() && = delete;
  inline __isl_keep isl_ast_build *get() const;
  inline __isl_give isl_ast_build *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

  inline ast_expr access_from(pw_multi_aff pma) const;
  inline ast_expr access_from(multi_pw_aff mpa) const;
  inline ast_expr call_from(pw_multi_aff pma) const;
  inline ast_expr call_from(multi_pw_aff mpa) const;
  inline ast_expr expr_from(set set) const;
  inline ast_expr expr_from(pw_aff pa) const;
  static inline ast_build from_context(set set);
  inline ast_node node_from_schedule_map(union_map schedule) const;
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
  inline ctx get_ctx() const;

  inline std::string to_C_str() const;
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
  inline ctx get_ctx() const;

  inline std::string to_C_str() const;
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
  inline explicit basic_map(ctx ctx, const std::string &str);
  inline basic_map &operator=(basic_map obj);
  inline ~basic_map();
  inline __isl_give isl_basic_map *copy() const &;
  inline __isl_give isl_basic_map *copy() && = delete;
  inline __isl_keep isl_basic_map *get() const;
  inline __isl_give isl_basic_map *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

  inline basic_map affine_hull() const;
  inline basic_map apply_domain(basic_map bmap2) const;
  inline basic_map apply_range(basic_map bmap2) const;
  inline basic_set deltas() const;
  inline basic_map detect_equalities() const;
  inline basic_map flatten() const;
  inline basic_map flatten_domain() const;
  inline basic_map flatten_range() const;
  inline basic_map gist(basic_map context) const;
  inline basic_map intersect(basic_map bmap2) const;
  inline basic_map intersect_domain(basic_set bset) const;
  inline basic_map intersect_range(basic_set bset) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const basic_map &bmap2) const;
  inline boolean is_subset(const basic_map &bmap2) const;
  inline map lexmax() const;
  inline map lexmin() const;
  inline basic_map reverse() const;
  inline basic_map sample() const;
  inline map unite(basic_map bmap2) const;
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
  inline explicit basic_set(ctx ctx, const std::string &str);
  inline /* implicit */ basic_set(point pnt);
  inline basic_set &operator=(basic_set obj);
  inline ~basic_set();
  inline __isl_give isl_basic_set *copy() const &;
  inline __isl_give isl_basic_set *copy() && = delete;
  inline __isl_keep isl_basic_set *get() const;
  inline __isl_give isl_basic_set *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

  inline basic_set affine_hull() const;
  inline basic_set apply(basic_map bmap) const;
  inline basic_set detect_equalities() const;
  inline val dim_max_val(int pos) const;
  inline basic_set flatten() const;
  inline basic_set gist(basic_set context) const;
  inline basic_set intersect(basic_set bset2) const;
  inline basic_set intersect_params(basic_set bset2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const basic_set &bset2) const;
  inline boolean is_subset(const basic_set &bset2) const;
  inline boolean is_wrapping() const;
  inline set lexmax() const;
  inline set lexmin() const;
  inline basic_set sample() const;
  inline point sample_point() const;
  inline set unite(basic_set bset2) const;
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
  inline explicit map(ctx ctx, const std::string &str);
  inline /* implicit */ map(basic_map bmap);
  inline map &operator=(map obj);
  inline ~map();
  inline __isl_give isl_map *copy() const &;
  inline __isl_give isl_map *copy() && = delete;
  inline __isl_keep isl_map *get() const;
  inline __isl_give isl_map *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

  inline basic_map affine_hull() const;
  inline map apply_domain(map map2) const;
  inline map apply_range(map map2) const;
  inline map coalesce() const;
  inline map complement() const;
  inline set deltas() const;
  inline map detect_equalities() const;
  inline map flatten() const;
  inline map flatten_domain() const;
  inline map flatten_range() const;
  inline stat foreach_basic_map(const std::function<stat(basic_map)> &fn) const;
  inline map gist(map context) const;
  inline map gist_domain(set context) const;
  inline map intersect(map map2) const;
  inline map intersect_domain(set set) const;
  inline map intersect_params(set params) const;
  inline map intersect_range(set set) const;
  inline boolean is_bijective() const;
  inline boolean is_disjoint(const map &map2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const map &map2) const;
  inline boolean is_injective() const;
  inline boolean is_single_valued() const;
  inline boolean is_strict_subset(const map &map2) const;
  inline boolean is_subset(const map &map2) const;
  inline map lexmax() const;
  inline map lexmin() const;
  inline basic_map polyhedral_hull() const;
  inline map reverse() const;
  inline basic_map sample() const;
  inline map subtract(map map2) const;
  inline map unite(map map2) const;
  inline basic_map unshifted_simple_hull() const;
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
  inline /* implicit */ multi_aff(aff aff);
  inline explicit multi_aff(ctx ctx, const std::string &str);
  inline multi_aff &operator=(multi_aff obj);
  inline ~multi_aff();
  inline __isl_give isl_multi_aff *copy() const &;
  inline __isl_give isl_multi_aff *copy() && = delete;
  inline __isl_keep isl_multi_aff *get() const;
  inline __isl_give isl_multi_aff *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

  inline multi_aff add(multi_aff multi2) const;
  inline multi_aff flat_range_product(multi_aff multi2) const;
  inline multi_aff product(multi_aff multi2) const;
  inline multi_aff pullback(multi_aff ma2) const;
  inline multi_aff range_product(multi_aff multi2) const;
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
  inline ctx get_ctx() const;

  inline multi_pw_aff add(multi_pw_aff multi2) const;
  inline multi_pw_aff flat_range_product(multi_pw_aff multi2) const;
  inline multi_pw_aff product(multi_pw_aff multi2) const;
  inline multi_pw_aff pullback(multi_aff ma) const;
  inline multi_pw_aff pullback(pw_multi_aff pma) const;
  inline multi_pw_aff pullback(multi_pw_aff mpa2) const;
  inline multi_pw_aff range_product(multi_pw_aff multi2) const;
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
  inline /* implicit */ multi_union_pw_aff(union_pw_aff upa);
  inline /* implicit */ multi_union_pw_aff(multi_pw_aff mpa);
  inline explicit multi_union_pw_aff(ctx ctx, const std::string &str);
  inline multi_union_pw_aff &operator=(multi_union_pw_aff obj);
  inline ~multi_union_pw_aff();
  inline __isl_give isl_multi_union_pw_aff *copy() const &;
  inline __isl_give isl_multi_union_pw_aff *copy() && = delete;
  inline __isl_keep isl_multi_union_pw_aff *get() const;
  inline __isl_give isl_multi_union_pw_aff *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

  inline multi_union_pw_aff add(multi_union_pw_aff multi2) const;
  inline multi_union_pw_aff flat_range_product(multi_union_pw_aff multi2) const;
  inline multi_union_pw_aff pullback(union_pw_multi_aff upma) const;
  inline multi_union_pw_aff range_product(multi_union_pw_aff multi2) const;
  inline multi_union_pw_aff union_add(multi_union_pw_aff mupa2) const;
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
  inline multi_val &operator=(multi_val obj);
  inline ~multi_val();
  inline __isl_give isl_multi_val *copy() const &;
  inline __isl_give isl_multi_val *copy() && = delete;
  inline __isl_keep isl_multi_val *get() const;
  inline __isl_give isl_multi_val *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

  inline multi_val add(multi_val multi2) const;
  inline multi_val flat_range_product(multi_val multi2) const;
  inline multi_val product(multi_val multi2) const;
  inline multi_val range_product(multi_val multi2) const;
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
  inline point &operator=(point obj);
  inline ~point();
  inline __isl_give isl_point *copy() const &;
  inline __isl_give isl_point *copy() && = delete;
  inline __isl_keep isl_point *get() const;
  inline __isl_give isl_point *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

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
  inline /* implicit */ pw_aff(aff aff);
  inline explicit pw_aff(ctx ctx, const std::string &str);
  inline pw_aff &operator=(pw_aff obj);
  inline ~pw_aff();
  inline __isl_give isl_pw_aff *copy() const &;
  inline __isl_give isl_pw_aff *copy() && = delete;
  inline __isl_keep isl_pw_aff *get() const;
  inline __isl_give isl_pw_aff *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

  inline pw_aff add(pw_aff pwaff2) const;
  inline pw_aff ceil() const;
  inline pw_aff cond(pw_aff pwaff_true, pw_aff pwaff_false) const;
  inline pw_aff div(pw_aff pa2) const;
  inline set eq_set(pw_aff pwaff2) const;
  inline pw_aff floor() const;
  inline set ge_set(pw_aff pwaff2) const;
  inline set gt_set(pw_aff pwaff2) const;
  inline set le_set(pw_aff pwaff2) const;
  inline set lt_set(pw_aff pwaff2) const;
  inline pw_aff max(pw_aff pwaff2) const;
  inline pw_aff min(pw_aff pwaff2) const;
  inline pw_aff mod(val mod) const;
  inline pw_aff mul(pw_aff pwaff2) const;
  inline set ne_set(pw_aff pwaff2) const;
  inline pw_aff neg() const;
  inline pw_aff pullback(multi_aff ma) const;
  inline pw_aff pullback(pw_multi_aff pma) const;
  inline pw_aff pullback(multi_pw_aff mpa) const;
  inline pw_aff scale(val v) const;
  inline pw_aff scale_down(val f) const;
  inline pw_aff sub(pw_aff pwaff2) const;
  inline pw_aff tdiv_q(pw_aff pa2) const;
  inline pw_aff tdiv_r(pw_aff pa2) const;
  inline pw_aff union_add(pw_aff pwaff2) const;
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
  inline ctx get_ctx() const;

  inline pw_multi_aff add(pw_multi_aff pma2) const;
  inline pw_multi_aff flat_range_product(pw_multi_aff pma2) const;
  inline pw_multi_aff product(pw_multi_aff pma2) const;
  inline pw_multi_aff pullback(multi_aff ma) const;
  inline pw_multi_aff pullback(pw_multi_aff pma2) const;
  inline pw_multi_aff range_product(pw_multi_aff pma2) const;
  inline pw_multi_aff union_add(pw_multi_aff pma2) const;
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
  inline explicit schedule(ctx ctx, const std::string &str);
  inline schedule &operator=(schedule obj);
  inline ~schedule();
  inline __isl_give isl_schedule *copy() const &;
  inline __isl_give isl_schedule *copy() && = delete;
  inline __isl_keep isl_schedule *get() const;
  inline __isl_give isl_schedule *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

  inline union_map get_map() const;
  inline schedule_node get_root() const;
  inline schedule pullback(union_pw_multi_aff upma) const;
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
  inline explicit schedule_constraints(ctx ctx, const std::string &str);
  inline schedule_constraints &operator=(schedule_constraints obj);
  inline ~schedule_constraints();
  inline __isl_give isl_schedule_constraints *copy() const &;
  inline __isl_give isl_schedule_constraints *copy() && = delete;
  inline __isl_keep isl_schedule_constraints *get() const;
  inline __isl_give isl_schedule_constraints *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

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
  inline schedule_node &operator=(schedule_node obj);
  inline ~schedule_node();
  inline __isl_give isl_schedule_node *copy() const &;
  inline __isl_give isl_schedule_node *copy() && = delete;
  inline __isl_keep isl_schedule_node *get() const;
  inline __isl_give isl_schedule_node *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

  inline boolean band_member_get_coincident(int pos) const;
  inline schedule_node band_member_set_coincident(int pos, int coincident) const;
  inline schedule_node child(int pos) const;
  inline multi_union_pw_aff get_prefix_schedule_multi_union_pw_aff() const;
  inline union_map get_prefix_schedule_union_map() const;
  inline union_pw_multi_aff get_prefix_schedule_union_pw_multi_aff() const;
  inline schedule get_schedule() const;
  inline schedule_node parent() const;
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
  inline explicit set(ctx ctx, const std::string &str);
  inline /* implicit */ set(basic_set bset);
  inline /* implicit */ set(point pnt);
  inline set &operator=(set obj);
  inline ~set();
  inline __isl_give isl_set *copy() const &;
  inline __isl_give isl_set *copy() && = delete;
  inline __isl_keep isl_set *get() const;
  inline __isl_give isl_set *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

  inline basic_set affine_hull() const;
  inline set apply(map map) const;
  inline set coalesce() const;
  inline set complement() const;
  inline set detect_equalities() const;
  inline set flatten() const;
  inline stat foreach_basic_set(const std::function<stat(basic_set)> &fn) const;
  inline val get_stride(int pos) const;
  inline set gist(set context) const;
  inline map identity() const;
  inline set intersect(set set2) const;
  inline set intersect_params(set params) const;
  inline boolean is_disjoint(const set &set2) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const set &set2) const;
  inline boolean is_strict_subset(const set &set2) const;
  inline boolean is_subset(const set &set2) const;
  inline boolean is_wrapping() const;
  inline set lexmax() const;
  inline set lexmin() const;
  inline val max_val(const aff &obj) const;
  inline val min_val(const aff &obj) const;
  inline basic_set polyhedral_hull() const;
  inline basic_set sample() const;
  inline point sample_point() const;
  inline set subtract(set set2) const;
  inline set unite(set set2) const;
  inline basic_set unshifted_simple_hull() const;
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
  inline explicit union_access_info(union_map sink);
  inline union_access_info &operator=(union_access_info obj);
  inline ~union_access_info();
  inline __isl_give isl_union_access_info *copy() const &;
  inline __isl_give isl_union_access_info *copy() && = delete;
  inline __isl_keep isl_union_access_info *get() const;
  inline __isl_give isl_union_access_info *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

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
  inline union_flow &operator=(union_flow obj);
  inline ~union_flow();
  inline __isl_give isl_union_flow *copy() const &;
  inline __isl_give isl_union_flow *copy() && = delete;
  inline __isl_keep isl_union_flow *get() const;
  inline __isl_give isl_union_flow *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

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
  inline ctx get_ctx() const;

  inline union_map affine_hull() const;
  inline union_map apply_domain(union_map umap2) const;
  inline union_map apply_range(union_map umap2) const;
  inline union_map coalesce() const;
  inline union_map compute_divs() const;
  inline union_set deltas() const;
  inline union_map detect_equalities() const;
  inline union_set domain() const;
  inline union_map domain_factor_domain() const;
  inline union_map domain_factor_range() const;
  inline union_map domain_map() const;
  inline union_pw_multi_aff domain_map_union_pw_multi_aff() const;
  inline union_map domain_product(union_map umap2) const;
  inline union_map eq_at(multi_union_pw_aff mupa) const;
  inline union_map factor_domain() const;
  inline union_map factor_range() const;
  inline union_map fixed_power(val exp) const;
  inline stat foreach_map(const std::function<stat(map)> &fn) const;
  static inline union_map from(union_pw_multi_aff upma);
  static inline union_map from(multi_union_pw_aff mupa);
  static inline union_map from_domain(union_set uset);
  static inline union_map from_domain_and_range(union_set domain, union_set range);
  static inline union_map from_range(union_set uset);
  inline union_map gist(union_map context) const;
  inline union_map gist_domain(union_set uset) const;
  inline union_map gist_params(set set) const;
  inline union_map gist_range(union_set uset) const;
  inline union_map intersect(union_map umap2) const;
  inline union_map intersect_domain(union_set uset) const;
  inline union_map intersect_params(set set) const;
  inline union_map intersect_range(union_set uset) const;
  inline boolean is_bijective() const;
  inline boolean is_empty() const;
  inline boolean is_equal(const union_map &umap2) const;
  inline boolean is_injective() const;
  inline boolean is_single_valued() const;
  inline boolean is_strict_subset(const union_map &umap2) const;
  inline boolean is_subset(const union_map &umap2) const;
  inline union_map lexmax() const;
  inline union_map lexmin() const;
  inline union_map polyhedral_hull() const;
  inline union_map product(union_map umap2) const;
  inline union_map project_out_all_params() const;
  inline union_set range() const;
  inline union_map range_factor_domain() const;
  inline union_map range_factor_range() const;
  inline union_map range_map() const;
  inline union_map range_product(union_map umap2) const;
  inline union_map reverse() const;
  inline union_map subtract(union_map umap2) const;
  inline union_map subtract_domain(union_set dom) const;
  inline union_map subtract_range(union_set dom) const;
  inline union_map unite(union_map umap2) const;
  inline union_set wrap() const;
  inline union_map zip() const;
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
  inline /* implicit */ union_pw_aff(pw_aff pa);
  inline explicit union_pw_aff(ctx ctx, const std::string &str);
  inline union_pw_aff &operator=(union_pw_aff obj);
  inline ~union_pw_aff();
  inline __isl_give isl_union_pw_aff *copy() const &;
  inline __isl_give isl_union_pw_aff *copy() && = delete;
  inline __isl_keep isl_union_pw_aff *get() const;
  inline __isl_give isl_union_pw_aff *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

  inline union_pw_aff add(union_pw_aff upa2) const;
  inline union_pw_aff pullback(union_pw_multi_aff upma) const;
  inline union_pw_aff union_add(union_pw_aff upa2) const;
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
  inline /* implicit */ union_pw_multi_aff(pw_multi_aff pma);
  inline explicit union_pw_multi_aff(ctx ctx, const std::string &str);
  inline /* implicit */ union_pw_multi_aff(union_pw_aff upa);
  inline union_pw_multi_aff &operator=(union_pw_multi_aff obj);
  inline ~union_pw_multi_aff();
  inline __isl_give isl_union_pw_multi_aff *copy() const &;
  inline __isl_give isl_union_pw_multi_aff *copy() && = delete;
  inline __isl_keep isl_union_pw_multi_aff *get() const;
  inline __isl_give isl_union_pw_multi_aff *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

  inline union_pw_multi_aff add(union_pw_multi_aff upma2) const;
  inline union_pw_multi_aff flat_range_product(union_pw_multi_aff upma2) const;
  inline union_pw_multi_aff pullback(union_pw_multi_aff upma2) const;
  inline union_pw_multi_aff union_add(union_pw_multi_aff upma2) const;
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
  inline ctx get_ctx() const;

  inline union_set affine_hull() const;
  inline union_set apply(union_map umap) const;
  inline union_set coalesce() const;
  inline union_set compute_divs() const;
  inline union_set detect_equalities() const;
  inline stat foreach_point(const std::function<stat(point)> &fn) const;
  inline stat foreach_set(const std::function<stat(set)> &fn) const;
  inline union_set gist(union_set context) const;
  inline union_set gist_params(set set) const;
  inline union_map identity() const;
  inline union_set intersect(union_set uset2) const;
  inline union_set intersect_params(set set) const;
  inline boolean is_empty() const;
  inline boolean is_equal(const union_set &uset2) const;
  inline boolean is_strict_subset(const union_set &uset2) const;
  inline boolean is_subset(const union_set &uset2) const;
  inline union_set lexmax() const;
  inline union_set lexmin() const;
  inline union_set polyhedral_hull() const;
  inline union_set preimage(multi_aff ma) const;
  inline union_set preimage(pw_multi_aff pma) const;
  inline union_set preimage(union_pw_multi_aff upma) const;
  inline point sample_point() const;
  inline union_set subtract(union_set uset2) const;
  inline union_set unite(union_set uset2) const;
  inline union_map unwrap() const;
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
  inline explicit val(ctx ctx, const std::string &str);
  inline explicit val(ctx ctx, long i);
  inline val &operator=(val obj);
  inline ~val();
  inline __isl_give isl_val *copy() const &;
  inline __isl_give isl_val *copy() && = delete;
  inline __isl_keep isl_val *get() const;
  inline __isl_give isl_val *release();
  inline bool is_null() const;
  inline ctx get_ctx() const;

  inline val abs() const;
  inline boolean abs_eq(const val &v2) const;
  inline val add(val v2) const;
  inline val ceil() const;
  inline int cmp_si(long i) const;
  inline val div(val v2) const;
  inline boolean eq(const val &v2) const;
  inline val floor() const;
  inline val gcd(val v2) const;
  inline boolean ge(const val &v2) const;
  inline boolean gt(const val &v2) const;
  static inline val infty(ctx ctx);
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
  static inline val nan(ctx ctx);
  inline boolean ne(const val &v2) const;
  inline val neg() const;
  static inline val neginfty(ctx ctx);
  static inline val negone(ctx ctx);
  static inline val one(ctx ctx);
  inline int sgn() const;
  inline val sub(val v2) const;
  inline val trunc() const;
  static inline val zero(ctx ctx);
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

ctx aff::get_ctx() const {
  return ctx(isl_aff_get_ctx(ptr));
}

aff aff::add(aff aff2) const
{
  auto res = isl_aff_add(copy(), aff2.release());
  return manage(res);
}

aff aff::ceil() const
{
  auto res = isl_aff_ceil(copy());
  return manage(res);
}

aff aff::div(aff aff2) const
{
  auto res = isl_aff_div(copy(), aff2.release());
  return manage(res);
}

set aff::eq_set(aff aff2) const
{
  auto res = isl_aff_eq_set(copy(), aff2.release());
  return manage(res);
}

aff aff::floor() const
{
  auto res = isl_aff_floor(copy());
  return manage(res);
}

set aff::ge_set(aff aff2) const
{
  auto res = isl_aff_ge_set(copy(), aff2.release());
  return manage(res);
}

set aff::gt_set(aff aff2) const
{
  auto res = isl_aff_gt_set(copy(), aff2.release());
  return manage(res);
}

set aff::le_set(aff aff2) const
{
  auto res = isl_aff_le_set(copy(), aff2.release());
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

aff aff::mul(aff aff2) const
{
  auto res = isl_aff_mul(copy(), aff2.release());
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

aff aff::pullback(multi_aff ma) const
{
  auto res = isl_aff_pullback_multi_aff(copy(), ma.release());
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

aff aff::sub(aff aff2) const
{
  auto res = isl_aff_sub(copy(), aff2.release());
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

ast_node ast_build::node_from_schedule_map(union_map schedule) const
{
  auto res = isl_ast_build_node_from_schedule_map(get(), schedule.release());
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

ctx ast_expr::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}

std::string ast_expr::to_C_str() const
{
  auto res = isl_ast_expr_to_C_str(get());
  std::string tmp(res);
  free(res);
  return tmp;
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

ctx ast_node::get_ctx() const {
  return ctx(isl_ast_node_get_ctx(ptr));
}

std::string ast_node::to_C_str() const
{
  auto res = isl_ast_node_to_C_str(get());
  std::string tmp(res);
  free(res);
  return tmp;
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

ctx basic_map::get_ctx() const {
  return ctx(isl_basic_map_get_ctx(ptr));
}

basic_map basic_map::affine_hull() const
{
  auto res = isl_basic_map_affine_hull(copy());
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

basic_set basic_map::deltas() const
{
  auto res = isl_basic_map_deltas(copy());
  return manage(res);
}

basic_map basic_map::detect_equalities() const
{
  auto res = isl_basic_map_detect_equalities(copy());
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

basic_map basic_map::gist(basic_map context) const
{
  auto res = isl_basic_map_gist(copy(), context.release());
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

boolean basic_map::is_subset(const basic_map &bmap2) const
{
  auto res = isl_basic_map_is_subset(get(), bmap2.get());
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

map basic_map::unite(basic_map bmap2) const
{
  auto res = isl_basic_map_union(copy(), bmap2.release());
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

ctx basic_set::get_ctx() const {
  return ctx(isl_basic_set_get_ctx(ptr));
}

basic_set basic_set::affine_hull() const
{
  auto res = isl_basic_set_affine_hull(copy());
  return manage(res);
}

basic_set basic_set::apply(basic_map bmap) const
{
  auto res = isl_basic_set_apply(copy(), bmap.release());
  return manage(res);
}

basic_set basic_set::detect_equalities() const
{
  auto res = isl_basic_set_detect_equalities(copy());
  return manage(res);
}

val basic_set::dim_max_val(int pos) const
{
  auto res = isl_basic_set_dim_max_val(copy(), pos);
  return manage(res);
}

basic_set basic_set::flatten() const
{
  auto res = isl_basic_set_flatten(copy());
  return manage(res);
}

basic_set basic_set::gist(basic_set context) const
{
  auto res = isl_basic_set_gist(copy(), context.release());
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

boolean basic_set::is_subset(const basic_set &bset2) const
{
  auto res = isl_basic_set_is_subset(get(), bset2.get());
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

set basic_set::unite(basic_set bset2) const
{
  auto res = isl_basic_set_union(copy(), bset2.release());
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

ctx map::get_ctx() const {
  return ctx(isl_map_get_ctx(ptr));
}

basic_map map::affine_hull() const
{
  auto res = isl_map_affine_hull(copy());
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

set map::deltas() const
{
  auto res = isl_map_deltas(copy());
  return manage(res);
}

map map::detect_equalities() const
{
  auto res = isl_map_detect_equalities(copy());
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

map map::gist(map context) const
{
  auto res = isl_map_gist(copy(), context.release());
  return manage(res);
}

map map::gist_domain(set context) const
{
  auto res = isl_map_gist_domain(copy(), context.release());
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

map map::lexmax() const
{
  auto res = isl_map_lexmax(copy());
  return manage(res);
}

map map::lexmin() const
{
  auto res = isl_map_lexmin(copy());
  return manage(res);
}

basic_map map::polyhedral_hull() const
{
  auto res = isl_map_polyhedral_hull(copy());
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

map map::subtract(map map2) const
{
  auto res = isl_map_subtract(copy(), map2.release());
  return manage(res);
}

map map::unite(map map2) const
{
  auto res = isl_map_union(copy(), map2.release());
  return manage(res);
}

basic_map map::unshifted_simple_hull() const
{
  auto res = isl_map_unshifted_simple_hull(copy());
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

ctx multi_aff::get_ctx() const {
  return ctx(isl_multi_aff_get_ctx(ptr));
}

multi_aff multi_aff::add(multi_aff multi2) const
{
  auto res = isl_multi_aff_add(copy(), multi2.release());
  return manage(res);
}

multi_aff multi_aff::flat_range_product(multi_aff multi2) const
{
  auto res = isl_multi_aff_flat_range_product(copy(), multi2.release());
  return manage(res);
}

multi_aff multi_aff::product(multi_aff multi2) const
{
  auto res = isl_multi_aff_product(copy(), multi2.release());
  return manage(res);
}

multi_aff multi_aff::pullback(multi_aff ma2) const
{
  auto res = isl_multi_aff_pullback_multi_aff(copy(), ma2.release());
  return manage(res);
}

multi_aff multi_aff::range_product(multi_aff multi2) const
{
  auto res = isl_multi_aff_range_product(copy(), multi2.release());
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

ctx multi_pw_aff::get_ctx() const {
  return ctx(isl_multi_pw_aff_get_ctx(ptr));
}

multi_pw_aff multi_pw_aff::add(multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_add(copy(), multi2.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::flat_range_product(multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_flat_range_product(copy(), multi2.release());
  return manage(res);
}

multi_pw_aff multi_pw_aff::product(multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_product(copy(), multi2.release());
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

multi_pw_aff multi_pw_aff::range_product(multi_pw_aff multi2) const
{
  auto res = isl_multi_pw_aff_range_product(copy(), multi2.release());
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

ctx multi_union_pw_aff::get_ctx() const {
  return ctx(isl_multi_union_pw_aff_get_ctx(ptr));
}

multi_union_pw_aff multi_union_pw_aff::add(multi_union_pw_aff multi2) const
{
  auto res = isl_multi_union_pw_aff_add(copy(), multi2.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::flat_range_product(multi_union_pw_aff multi2) const
{
  auto res = isl_multi_union_pw_aff_flat_range_product(copy(), multi2.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::pullback(union_pw_multi_aff upma) const
{
  auto res = isl_multi_union_pw_aff_pullback_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::range_product(multi_union_pw_aff multi2) const
{
  auto res = isl_multi_union_pw_aff_range_product(copy(), multi2.release());
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::union_add(multi_union_pw_aff mupa2) const
{
  auto res = isl_multi_union_pw_aff_union_add(copy(), mupa2.release());
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

ctx multi_val::get_ctx() const {
  return ctx(isl_multi_val_get_ctx(ptr));
}

multi_val multi_val::add(multi_val multi2) const
{
  auto res = isl_multi_val_add(copy(), multi2.release());
  return manage(res);
}

multi_val multi_val::flat_range_product(multi_val multi2) const
{
  auto res = isl_multi_val_flat_range_product(copy(), multi2.release());
  return manage(res);
}

multi_val multi_val::product(multi_val multi2) const
{
  auto res = isl_multi_val_product(copy(), multi2.release());
  return manage(res);
}

multi_val multi_val::range_product(multi_val multi2) const
{
  auto res = isl_multi_val_range_product(copy(), multi2.release());
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

ctx point::get_ctx() const {
  return ctx(isl_point_get_ctx(ptr));
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

pw_aff::pw_aff(aff aff)
{
  auto res = isl_pw_aff_from_aff(aff.release());
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

ctx pw_aff::get_ctx() const {
  return ctx(isl_pw_aff_get_ctx(ptr));
}

pw_aff pw_aff::add(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_add(copy(), pwaff2.release());
  return manage(res);
}

pw_aff pw_aff::ceil() const
{
  auto res = isl_pw_aff_ceil(copy());
  return manage(res);
}

pw_aff pw_aff::cond(pw_aff pwaff_true, pw_aff pwaff_false) const
{
  auto res = isl_pw_aff_cond(copy(), pwaff_true.release(), pwaff_false.release());
  return manage(res);
}

pw_aff pw_aff::div(pw_aff pa2) const
{
  auto res = isl_pw_aff_div(copy(), pa2.release());
  return manage(res);
}

set pw_aff::eq_set(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_eq_set(copy(), pwaff2.release());
  return manage(res);
}

pw_aff pw_aff::floor() const
{
  auto res = isl_pw_aff_floor(copy());
  return manage(res);
}

set pw_aff::ge_set(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_ge_set(copy(), pwaff2.release());
  return manage(res);
}

set pw_aff::gt_set(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_gt_set(copy(), pwaff2.release());
  return manage(res);
}

set pw_aff::le_set(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_le_set(copy(), pwaff2.release());
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

pw_aff pw_aff::mul(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_mul(copy(), pwaff2.release());
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

pw_aff pw_aff::sub(pw_aff pwaff2) const
{
  auto res = isl_pw_aff_sub(copy(), pwaff2.release());
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

ctx pw_multi_aff::get_ctx() const {
  return ctx(isl_pw_multi_aff_get_ctx(ptr));
}

pw_multi_aff pw_multi_aff::add(pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_add(copy(), pma2.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::flat_range_product(pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_flat_range_product(copy(), pma2.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::product(pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_product(copy(), pma2.release());
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

pw_multi_aff pw_multi_aff::range_product(pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_range_product(copy(), pma2.release());
  return manage(res);
}

pw_multi_aff pw_multi_aff::union_add(pw_multi_aff pma2) const
{
  auto res = isl_pw_multi_aff_union_add(copy(), pma2.release());
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

ctx schedule::get_ctx() const {
  return ctx(isl_schedule_get_ctx(ptr));
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

schedule schedule::pullback(union_pw_multi_aff upma) const
{
  auto res = isl_schedule_pullback_union_pw_multi_aff(copy(), upma.release());
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

ctx schedule_constraints::get_ctx() const {
  return ctx(isl_schedule_constraints_get_ctx(ptr));
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

ctx schedule_node::get_ctx() const {
  return ctx(isl_schedule_node_get_ctx(ptr));
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

schedule_node schedule_node::child(int pos) const
{
  auto res = isl_schedule_node_child(copy(), pos);
  return manage(res);
}

multi_union_pw_aff schedule_node::get_prefix_schedule_multi_union_pw_aff() const
{
  auto res = isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(get());
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

schedule_node schedule_node::parent() const
{
  auto res = isl_schedule_node_parent(copy());
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

ctx set::get_ctx() const {
  return ctx(isl_set_get_ctx(ptr));
}

basic_set set::affine_hull() const
{
  auto res = isl_set_affine_hull(copy());
  return manage(res);
}

set set::apply(map map) const
{
  auto res = isl_set_apply(copy(), map.release());
  return manage(res);
}

set set::coalesce() const
{
  auto res = isl_set_coalesce(copy());
  return manage(res);
}

set set::complement() const
{
  auto res = isl_set_complement(copy());
  return manage(res);
}

set set::detect_equalities() const
{
  auto res = isl_set_detect_equalities(copy());
  return manage(res);
}

set set::flatten() const
{
  auto res = isl_set_flatten(copy());
  return manage(res);
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

val set::get_stride(int pos) const
{
  auto res = isl_set_get_stride(get(), pos);
  return manage(res);
}

set set::gist(set context) const
{
  auto res = isl_set_gist(copy(), context.release());
  return manage(res);
}

map set::identity() const
{
  auto res = isl_set_identity(copy());
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

set set::lexmax() const
{
  auto res = isl_set_lexmax(copy());
  return manage(res);
}

set set::lexmin() const
{
  auto res = isl_set_lexmin(copy());
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

basic_set set::polyhedral_hull() const
{
  auto res = isl_set_polyhedral_hull(copy());
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

set set::subtract(set set2) const
{
  auto res = isl_set_subtract(copy(), set2.release());
  return manage(res);
}

set set::unite(set set2) const
{
  auto res = isl_set_union(copy(), set2.release());
  return manage(res);
}

basic_set set::unshifted_simple_hull() const
{
  auto res = isl_set_unshifted_simple_hull(copy());
  return manage(res);
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

ctx union_access_info::get_ctx() const {
  return ctx(isl_union_access_info_get_ctx(ptr));
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

ctx union_flow::get_ctx() const {
  return ctx(isl_union_flow_get_ctx(ptr));
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

union_map::union_map(__isl_take isl_union_map *ptr)
    : ptr(ptr) {}

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

ctx union_map::get_ctx() const {
  return ctx(isl_union_map_get_ctx(ptr));
}

union_map union_map::affine_hull() const
{
  auto res = isl_union_map_affine_hull(copy());
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

union_map union_map::compute_divs() const
{
  auto res = isl_union_map_compute_divs(copy());
  return manage(res);
}

union_set union_map::deltas() const
{
  auto res = isl_union_map_deltas(copy());
  return manage(res);
}

union_map union_map::detect_equalities() const
{
  auto res = isl_union_map_detect_equalities(copy());
  return manage(res);
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

union_map union_map::eq_at(multi_union_pw_aff mupa) const
{
  auto res = isl_union_map_eq_at_multi_union_pw_aff(copy(), mupa.release());
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

union_map union_map::fixed_power(val exp) const
{
  auto res = isl_union_map_fixed_power_val(copy(), exp.release());
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

union_map union_map::from(union_pw_multi_aff upma)
{
  auto res = isl_union_map_from_union_pw_multi_aff(upma.release());
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

boolean union_map::is_bijective() const
{
  auto res = isl_union_map_is_bijective(get());
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

union_map union_map::polyhedral_hull() const
{
  auto res = isl_union_map_polyhedral_hull(copy());
  return manage(res);
}

union_map union_map::product(union_map umap2) const
{
  auto res = isl_union_map_product(copy(), umap2.release());
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

union_map union_map::reverse() const
{
  auto res = isl_union_map_reverse(copy());
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

union_map union_map::unite(union_map umap2) const
{
  auto res = isl_union_map_union(copy(), umap2.release());
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

union_pw_aff::union_pw_aff(pw_aff pa)
{
  auto res = isl_union_pw_aff_from_pw_aff(pa.release());
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

ctx union_pw_aff::get_ctx() const {
  return ctx(isl_union_pw_aff_get_ctx(ptr));
}

union_pw_aff union_pw_aff::add(union_pw_aff upa2) const
{
  auto res = isl_union_pw_aff_add(copy(), upa2.release());
  return manage(res);
}

union_pw_aff union_pw_aff::pullback(union_pw_multi_aff upma) const
{
  auto res = isl_union_pw_aff_pullback_union_pw_multi_aff(copy(), upma.release());
  return manage(res);
}

union_pw_aff union_pw_aff::union_add(union_pw_aff upa2) const
{
  auto res = isl_union_pw_aff_union_add(copy(), upa2.release());
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

union_pw_multi_aff::union_pw_multi_aff(pw_multi_aff pma)
{
  auto res = isl_union_pw_multi_aff_from_pw_multi_aff(pma.release());
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

ctx union_pw_multi_aff::get_ctx() const {
  return ctx(isl_union_pw_multi_aff_get_ctx(ptr));
}

union_pw_multi_aff union_pw_multi_aff::add(union_pw_multi_aff upma2) const
{
  auto res = isl_union_pw_multi_aff_add(copy(), upma2.release());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::flat_range_product(union_pw_multi_aff upma2) const
{
  auto res = isl_union_pw_multi_aff_flat_range_product(copy(), upma2.release());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::pullback(union_pw_multi_aff upma2) const
{
  auto res = isl_union_pw_multi_aff_pullback_union_pw_multi_aff(copy(), upma2.release());
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::union_add(union_pw_multi_aff upma2) const
{
  auto res = isl_union_pw_multi_aff_union_add(copy(), upma2.release());
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

ctx union_set::get_ctx() const {
  return ctx(isl_union_set_get_ctx(ptr));
}

union_set union_set::affine_hull() const
{
  auto res = isl_union_set_affine_hull(copy());
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

union_set union_set::compute_divs() const
{
  auto res = isl_union_set_compute_divs(copy());
  return manage(res);
}

union_set union_set::detect_equalities() const
{
  auto res = isl_union_set_detect_equalities(copy());
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

point union_set::sample_point() const
{
  auto res = isl_union_set_sample_point(copy());
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

union_map union_set::unwrap() const
{
  auto res = isl_union_set_unwrap(copy());
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

ctx val::get_ctx() const {
  return ctx(isl_val_get_ctx(ptr));
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

boolean val::gt(const val &v2) const
{
  auto res = isl_val_gt(get(), v2.get());
  return manage(res);
}

val val::infty(ctx ctx)
{
  auto res = isl_val_infty(ctx.release());
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
} // namespace checked
} // namespace isl

#endif /* ISL_CPP_CHECKED */
