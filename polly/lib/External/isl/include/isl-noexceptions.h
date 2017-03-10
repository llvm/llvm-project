/// These are automatically generated C++ bindings for isl.
///
///
/// isl is a library for computing with integer sets and maps described by
/// Presburger formula. On top of this, isl provides various tools for
/// Polyhedral compilation ranging from dependence analysis over scheduling
/// to AST generation.
///
///
/// WARNING: Even though these bindings have been throughly tested and the
///          design has been reviewed by various members of the isl community,
///          we do not yet provide any stability guarantees for this interface.
///          We do not expect any larger changes to the interface, but want to
///          reserve the freedom to improve the bindings based on insights that
///          only become visible after shipping these bindings with isl itself.

#ifndef ISL_CPP_NOEXCEPTIONS
#define ISL_CPP_NOEXCEPTIONS

#include <isl/aff.h>
#include <isl/ast_build.h>
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

#include <functional>
#include <string>

namespace isl {

inline namespace noexceptions {

// forward declarations
class aff;
class ast_build;
class ast_expr;
class ast_node;
class basic_map;
class basic_set;
class id;
class local_space;
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
class space;
class union_access_info;
class union_flow;
class union_map;
class union_pw_aff;
class union_pw_multi_aff;
class union_set;
class val;

class ctx {
  isl_ctx *ptr;

public:
  ctx(isl_ctx *ctx)
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

// declarations for isl::aff
inline aff manage(__isl_take isl_aff *ptr);

inline aff give(__isl_take isl_aff *ptr);

class aff {
  friend inline aff manage(__isl_take isl_aff *ptr);

  isl_aff *ptr = nullptr;

  inline explicit aff(__isl_take isl_aff *ptr);

public:
  inline /* implicit */ aff();
  inline /* implicit */ aff(const aff &obj);
  inline /* implicit */ aff(std::nullptr_t);
  inline aff &operator=(aff obj);
  inline ~aff();
  inline __isl_give isl_aff *copy() const &;
  inline __isl_give isl_aff *copy() && = delete;
  inline __isl_keep isl_aff *get() const;
  inline __isl_give isl_aff *release();
  inline __isl_keep isl_aff *keep() const;
  inline __isl_give isl_aff *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::ast_build
inline ast_build manage(__isl_take isl_ast_build *ptr);

inline ast_build give(__isl_take isl_ast_build *ptr);

class ast_build {
  friend inline ast_build manage(__isl_take isl_ast_build *ptr);

  isl_ast_build *ptr = nullptr;

  inline explicit ast_build(__isl_take isl_ast_build *ptr);

public:
  inline /* implicit */ ast_build();
  inline /* implicit */ ast_build(const ast_build &obj);
  inline /* implicit */ ast_build(std::nullptr_t);
  inline ast_build &operator=(ast_build obj);
  inline ~ast_build();
  inline __isl_give isl_ast_build *copy() const &;
  inline __isl_give isl_ast_build *copy() && = delete;
  inline __isl_keep isl_ast_build *get() const;
  inline __isl_give isl_ast_build *release();
  inline __isl_keep isl_ast_build *keep() const;
  inline __isl_give isl_ast_build *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
};

// declarations for isl::ast_expr
inline ast_expr manage(__isl_take isl_ast_expr *ptr);

inline ast_expr give(__isl_take isl_ast_expr *ptr);

class ast_expr {
  friend inline ast_expr manage(__isl_take isl_ast_expr *ptr);

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
  inline __isl_keep isl_ast_expr *keep() const;
  inline __isl_give isl_ast_expr *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::ast_node
inline ast_node manage(__isl_take isl_ast_node *ptr);

inline ast_node give(__isl_take isl_ast_node *ptr);

class ast_node {
  friend inline ast_node manage(__isl_take isl_ast_node *ptr);

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
  inline __isl_keep isl_ast_node *keep() const;
  inline __isl_give isl_ast_node *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::basic_map
inline basic_map manage(__isl_take isl_basic_map *ptr);

inline basic_map give(__isl_take isl_basic_map *ptr);

class basic_map {
  friend inline basic_map manage(__isl_take isl_basic_map *ptr);

  isl_basic_map *ptr = nullptr;

  inline explicit basic_map(__isl_take isl_basic_map *ptr);

public:
  inline /* implicit */ basic_map();
  inline /* implicit */ basic_map(const basic_map &obj);
  inline /* implicit */ basic_map(std::nullptr_t);
  inline basic_map &operator=(basic_map obj);
  inline ~basic_map();
  inline __isl_give isl_basic_map *copy() const &;
  inline __isl_give isl_basic_map *copy() && = delete;
  inline __isl_keep isl_basic_map *get() const;
  inline __isl_give isl_basic_map *release();
  inline __isl_keep isl_basic_map *keep() const;
  inline __isl_give isl_basic_map *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::basic_set
inline basic_set manage(__isl_take isl_basic_set *ptr);

inline basic_set give(__isl_take isl_basic_set *ptr);

class basic_set {
  friend inline basic_set manage(__isl_take isl_basic_set *ptr);

  isl_basic_set *ptr = nullptr;

  inline explicit basic_set(__isl_take isl_basic_set *ptr);

public:
  inline /* implicit */ basic_set();
  inline /* implicit */ basic_set(const basic_set &obj);
  inline /* implicit */ basic_set(std::nullptr_t);
  inline basic_set &operator=(basic_set obj);
  inline ~basic_set();
  inline __isl_give isl_basic_set *copy() const &;
  inline __isl_give isl_basic_set *copy() && = delete;
  inline __isl_keep isl_basic_set *get() const;
  inline __isl_give isl_basic_set *release();
  inline __isl_keep isl_basic_set *keep() const;
  inline __isl_give isl_basic_set *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::id
inline id manage(__isl_take isl_id *ptr);

inline id give(__isl_take isl_id *ptr);

class id {
  friend inline id manage(__isl_take isl_id *ptr);

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
  inline __isl_keep isl_id *keep() const;
  inline __isl_give isl_id *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::local_space
inline local_space manage(__isl_take isl_local_space *ptr);

inline local_space give(__isl_take isl_local_space *ptr);

class local_space {
  friend inline local_space manage(__isl_take isl_local_space *ptr);

  isl_local_space *ptr = nullptr;

  inline explicit local_space(__isl_take isl_local_space *ptr);

public:
  inline /* implicit */ local_space();
  inline /* implicit */ local_space(const local_space &obj);
  inline /* implicit */ local_space(std::nullptr_t);
  inline local_space &operator=(local_space obj);
  inline ~local_space();
  inline __isl_give isl_local_space *copy() const &;
  inline __isl_give isl_local_space *copy() && = delete;
  inline __isl_keep isl_local_space *get() const;
  inline __isl_give isl_local_space *release();
  inline __isl_keep isl_local_space *keep() const;
  inline __isl_give isl_local_space *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
};

// declarations for isl::map
inline map manage(__isl_take isl_map *ptr);

inline map give(__isl_take isl_map *ptr);

class map {
  friend inline map manage(__isl_take isl_map *ptr);

  isl_map *ptr = nullptr;

  inline explicit map(__isl_take isl_map *ptr);

public:
  inline /* implicit */ map();
  inline /* implicit */ map(const map &obj);
  inline /* implicit */ map(std::nullptr_t);
  inline map &operator=(map obj);
  inline ~map();
  inline __isl_give isl_map *copy() const &;
  inline __isl_give isl_map *copy() && = delete;
  inline __isl_keep isl_map *get() const;
  inline __isl_give isl_map *release();
  inline __isl_keep isl_map *keep() const;
  inline __isl_give isl_map *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::multi_aff
inline multi_aff manage(__isl_take isl_multi_aff *ptr);

inline multi_aff give(__isl_take isl_multi_aff *ptr);

class multi_aff {
  friend inline multi_aff manage(__isl_take isl_multi_aff *ptr);

  isl_multi_aff *ptr = nullptr;

  inline explicit multi_aff(__isl_take isl_multi_aff *ptr);

public:
  inline /* implicit */ multi_aff();
  inline /* implicit */ multi_aff(const multi_aff &obj);
  inline /* implicit */ multi_aff(std::nullptr_t);
  inline multi_aff &operator=(multi_aff obj);
  inline ~multi_aff();
  inline __isl_give isl_multi_aff *copy() const &;
  inline __isl_give isl_multi_aff *copy() && = delete;
  inline __isl_keep isl_multi_aff *get() const;
  inline __isl_give isl_multi_aff *release();
  inline __isl_keep isl_multi_aff *keep() const;
  inline __isl_give isl_multi_aff *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::multi_pw_aff
inline multi_pw_aff manage(__isl_take isl_multi_pw_aff *ptr);

inline multi_pw_aff give(__isl_take isl_multi_pw_aff *ptr);

class multi_pw_aff {
  friend inline multi_pw_aff manage(__isl_take isl_multi_pw_aff *ptr);

  isl_multi_pw_aff *ptr = nullptr;

  inline explicit multi_pw_aff(__isl_take isl_multi_pw_aff *ptr);

public:
  inline /* implicit */ multi_pw_aff();
  inline /* implicit */ multi_pw_aff(const multi_pw_aff &obj);
  inline /* implicit */ multi_pw_aff(std::nullptr_t);
  inline multi_pw_aff &operator=(multi_pw_aff obj);
  inline ~multi_pw_aff();
  inline __isl_give isl_multi_pw_aff *copy() const &;
  inline __isl_give isl_multi_pw_aff *copy() && = delete;
  inline __isl_keep isl_multi_pw_aff *get() const;
  inline __isl_give isl_multi_pw_aff *release();
  inline __isl_keep isl_multi_pw_aff *keep() const;
  inline __isl_give isl_multi_pw_aff *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::multi_union_pw_aff
inline multi_union_pw_aff manage(__isl_take isl_multi_union_pw_aff *ptr);

inline multi_union_pw_aff give(__isl_take isl_multi_union_pw_aff *ptr);

class multi_union_pw_aff {
  friend inline multi_union_pw_aff manage(__isl_take isl_multi_union_pw_aff *ptr);

  isl_multi_union_pw_aff *ptr = nullptr;

  inline explicit multi_union_pw_aff(__isl_take isl_multi_union_pw_aff *ptr);

public:
  inline /* implicit */ multi_union_pw_aff();
  inline /* implicit */ multi_union_pw_aff(const multi_union_pw_aff &obj);
  inline /* implicit */ multi_union_pw_aff(std::nullptr_t);
  inline multi_union_pw_aff &operator=(multi_union_pw_aff obj);
  inline ~multi_union_pw_aff();
  inline __isl_give isl_multi_union_pw_aff *copy() const &;
  inline __isl_give isl_multi_union_pw_aff *copy() && = delete;
  inline __isl_keep isl_multi_union_pw_aff *get() const;
  inline __isl_give isl_multi_union_pw_aff *release();
  inline __isl_keep isl_multi_union_pw_aff *keep() const;
  inline __isl_give isl_multi_union_pw_aff *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::multi_val
inline multi_val manage(__isl_take isl_multi_val *ptr);

inline multi_val give(__isl_take isl_multi_val *ptr);

class multi_val {
  friend inline multi_val manage(__isl_take isl_multi_val *ptr);

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
  inline __isl_keep isl_multi_val *keep() const;
  inline __isl_give isl_multi_val *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::point
inline point manage(__isl_take isl_point *ptr);

inline point give(__isl_take isl_point *ptr);

class point {
  friend inline point manage(__isl_take isl_point *ptr);

  isl_point *ptr = nullptr;

  inline explicit point(__isl_take isl_point *ptr);

public:
  inline /* implicit */ point();
  inline /* implicit */ point(const point &obj);
  inline /* implicit */ point(std::nullptr_t);
  inline point &operator=(point obj);
  inline ~point();
  inline __isl_give isl_point *copy() const &;
  inline __isl_give isl_point *copy() && = delete;
  inline __isl_keep isl_point *get() const;
  inline __isl_give isl_point *release();
  inline __isl_keep isl_point *keep() const;
  inline __isl_give isl_point *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::pw_aff
inline pw_aff manage(__isl_take isl_pw_aff *ptr);

inline pw_aff give(__isl_take isl_pw_aff *ptr);

class pw_aff {
  friend inline pw_aff manage(__isl_take isl_pw_aff *ptr);

  isl_pw_aff *ptr = nullptr;

  inline explicit pw_aff(__isl_take isl_pw_aff *ptr);

public:
  inline /* implicit */ pw_aff();
  inline /* implicit */ pw_aff(const pw_aff &obj);
  inline /* implicit */ pw_aff(std::nullptr_t);
  inline pw_aff &operator=(pw_aff obj);
  inline ~pw_aff();
  inline __isl_give isl_pw_aff *copy() const &;
  inline __isl_give isl_pw_aff *copy() && = delete;
  inline __isl_keep isl_pw_aff *get() const;
  inline __isl_give isl_pw_aff *release();
  inline __isl_keep isl_pw_aff *keep() const;
  inline __isl_give isl_pw_aff *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::pw_multi_aff
inline pw_multi_aff manage(__isl_take isl_pw_multi_aff *ptr);

inline pw_multi_aff give(__isl_take isl_pw_multi_aff *ptr);

class pw_multi_aff {
  friend inline pw_multi_aff manage(__isl_take isl_pw_multi_aff *ptr);

  isl_pw_multi_aff *ptr = nullptr;

  inline explicit pw_multi_aff(__isl_take isl_pw_multi_aff *ptr);

public:
  inline /* implicit */ pw_multi_aff();
  inline /* implicit */ pw_multi_aff(const pw_multi_aff &obj);
  inline /* implicit */ pw_multi_aff(std::nullptr_t);
  inline pw_multi_aff &operator=(pw_multi_aff obj);
  inline ~pw_multi_aff();
  inline __isl_give isl_pw_multi_aff *copy() const &;
  inline __isl_give isl_pw_multi_aff *copy() && = delete;
  inline __isl_keep isl_pw_multi_aff *get() const;
  inline __isl_give isl_pw_multi_aff *release();
  inline __isl_keep isl_pw_multi_aff *keep() const;
  inline __isl_give isl_pw_multi_aff *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::schedule
inline schedule manage(__isl_take isl_schedule *ptr);

inline schedule give(__isl_take isl_schedule *ptr);

class schedule {
  friend inline schedule manage(__isl_take isl_schedule *ptr);

  isl_schedule *ptr = nullptr;

  inline explicit schedule(__isl_take isl_schedule *ptr);

public:
  inline /* implicit */ schedule();
  inline /* implicit */ schedule(const schedule &obj);
  inline /* implicit */ schedule(std::nullptr_t);
  inline schedule &operator=(schedule obj);
  inline ~schedule();
  inline __isl_give isl_schedule *copy() const &;
  inline __isl_give isl_schedule *copy() && = delete;
  inline __isl_keep isl_schedule *get() const;
  inline __isl_give isl_schedule *release();
  inline __isl_keep isl_schedule *keep() const;
  inline __isl_give isl_schedule *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::schedule_constraints
inline schedule_constraints manage(__isl_take isl_schedule_constraints *ptr);

inline schedule_constraints give(__isl_take isl_schedule_constraints *ptr);

class schedule_constraints {
  friend inline schedule_constraints manage(__isl_take isl_schedule_constraints *ptr);

  isl_schedule_constraints *ptr = nullptr;

  inline explicit schedule_constraints(__isl_take isl_schedule_constraints *ptr);

public:
  inline /* implicit */ schedule_constraints();
  inline /* implicit */ schedule_constraints(const schedule_constraints &obj);
  inline /* implicit */ schedule_constraints(std::nullptr_t);
  inline schedule_constraints &operator=(schedule_constraints obj);
  inline ~schedule_constraints();
  inline __isl_give isl_schedule_constraints *copy() const &;
  inline __isl_give isl_schedule_constraints *copy() && = delete;
  inline __isl_keep isl_schedule_constraints *get() const;
  inline __isl_give isl_schedule_constraints *release();
  inline __isl_keep isl_schedule_constraints *keep() const;
  inline __isl_give isl_schedule_constraints *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::schedule_node
inline schedule_node manage(__isl_take isl_schedule_node *ptr);

inline schedule_node give(__isl_take isl_schedule_node *ptr);

class schedule_node {
  friend inline schedule_node manage(__isl_take isl_schedule_node *ptr);

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
  inline __isl_keep isl_schedule_node *keep() const;
  inline __isl_give isl_schedule_node *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::set
inline set manage(__isl_take isl_set *ptr);

inline set give(__isl_take isl_set *ptr);

class set {
  friend inline set manage(__isl_take isl_set *ptr);

  isl_set *ptr = nullptr;

  inline explicit set(__isl_take isl_set *ptr);

public:
  inline /* implicit */ set();
  inline /* implicit */ set(const set &obj);
  inline /* implicit */ set(std::nullptr_t);
  inline set &operator=(set obj);
  inline ~set();
  inline __isl_give isl_set *copy() const &;
  inline __isl_give isl_set *copy() && = delete;
  inline __isl_keep isl_set *get() const;
  inline __isl_give isl_set *release();
  inline __isl_keep isl_set *keep() const;
  inline __isl_give isl_set *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::space
inline space manage(__isl_take isl_space *ptr);

inline space give(__isl_take isl_space *ptr);

class space {
  friend inline space manage(__isl_take isl_space *ptr);

  isl_space *ptr = nullptr;

  inline explicit space(__isl_take isl_space *ptr);

public:
  inline /* implicit */ space();
  inline /* implicit */ space(const space &obj);
  inline /* implicit */ space(std::nullptr_t);
  inline space &operator=(space obj);
  inline ~space();
  inline __isl_give isl_space *copy() const &;
  inline __isl_give isl_space *copy() && = delete;
  inline __isl_keep isl_space *get() const;
  inline __isl_give isl_space *release();
  inline __isl_keep isl_space *keep() const;
  inline __isl_give isl_space *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::union_access_info
inline union_access_info manage(__isl_take isl_union_access_info *ptr);

inline union_access_info give(__isl_take isl_union_access_info *ptr);

class union_access_info {
  friend inline union_access_info manage(__isl_take isl_union_access_info *ptr);

  isl_union_access_info *ptr = nullptr;

  inline explicit union_access_info(__isl_take isl_union_access_info *ptr);

public:
  inline /* implicit */ union_access_info();
  inline /* implicit */ union_access_info(const union_access_info &obj);
  inline /* implicit */ union_access_info(std::nullptr_t);
  inline union_access_info &operator=(union_access_info obj);
  inline ~union_access_info();
  inline __isl_give isl_union_access_info *copy() const &;
  inline __isl_give isl_union_access_info *copy() && = delete;
  inline __isl_keep isl_union_access_info *get() const;
  inline __isl_give isl_union_access_info *release();
  inline __isl_keep isl_union_access_info *keep() const;
  inline __isl_give isl_union_access_info *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::union_flow
inline union_flow manage(__isl_take isl_union_flow *ptr);

inline union_flow give(__isl_take isl_union_flow *ptr);

class union_flow {
  friend inline union_flow manage(__isl_take isl_union_flow *ptr);

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
  inline __isl_keep isl_union_flow *keep() const;
  inline __isl_give isl_union_flow *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::union_map
inline union_map manage(__isl_take isl_union_map *ptr);

inline union_map give(__isl_take isl_union_map *ptr);

class union_map {
  friend inline union_map manage(__isl_take isl_union_map *ptr);

  isl_union_map *ptr = nullptr;

  inline explicit union_map(__isl_take isl_union_map *ptr);

public:
  inline /* implicit */ union_map();
  inline /* implicit */ union_map(const union_map &obj);
  inline /* implicit */ union_map(std::nullptr_t);
  inline union_map &operator=(union_map obj);
  inline ~union_map();
  inline __isl_give isl_union_map *copy() const &;
  inline __isl_give isl_union_map *copy() && = delete;
  inline __isl_keep isl_union_map *get() const;
  inline __isl_give isl_union_map *release();
  inline __isl_keep isl_union_map *keep() const;
  inline __isl_give isl_union_map *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::union_pw_aff
inline union_pw_aff manage(__isl_take isl_union_pw_aff *ptr);

inline union_pw_aff give(__isl_take isl_union_pw_aff *ptr);

class union_pw_aff {
  friend inline union_pw_aff manage(__isl_take isl_union_pw_aff *ptr);

  isl_union_pw_aff *ptr = nullptr;

  inline explicit union_pw_aff(__isl_take isl_union_pw_aff *ptr);

public:
  inline /* implicit */ union_pw_aff();
  inline /* implicit */ union_pw_aff(const union_pw_aff &obj);
  inline /* implicit */ union_pw_aff(std::nullptr_t);
  inline union_pw_aff &operator=(union_pw_aff obj);
  inline ~union_pw_aff();
  inline __isl_give isl_union_pw_aff *copy() const &;
  inline __isl_give isl_union_pw_aff *copy() && = delete;
  inline __isl_keep isl_union_pw_aff *get() const;
  inline __isl_give isl_union_pw_aff *release();
  inline __isl_keep isl_union_pw_aff *keep() const;
  inline __isl_give isl_union_pw_aff *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::union_pw_multi_aff
inline union_pw_multi_aff manage(__isl_take isl_union_pw_multi_aff *ptr);

inline union_pw_multi_aff give(__isl_take isl_union_pw_multi_aff *ptr);

class union_pw_multi_aff {
  friend inline union_pw_multi_aff manage(__isl_take isl_union_pw_multi_aff *ptr);

  isl_union_pw_multi_aff *ptr = nullptr;

  inline explicit union_pw_multi_aff(__isl_take isl_union_pw_multi_aff *ptr);

public:
  inline /* implicit */ union_pw_multi_aff();
  inline /* implicit */ union_pw_multi_aff(const union_pw_multi_aff &obj);
  inline /* implicit */ union_pw_multi_aff(std::nullptr_t);
  inline union_pw_multi_aff &operator=(union_pw_multi_aff obj);
  inline ~union_pw_multi_aff();
  inline __isl_give isl_union_pw_multi_aff *copy() const &;
  inline __isl_give isl_union_pw_multi_aff *copy() && = delete;
  inline __isl_keep isl_union_pw_multi_aff *get() const;
  inline __isl_give isl_union_pw_multi_aff *release();
  inline __isl_keep isl_union_pw_multi_aff *keep() const;
  inline __isl_give isl_union_pw_multi_aff *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::union_set
inline union_set manage(__isl_take isl_union_set *ptr);

inline union_set give(__isl_take isl_union_set *ptr);

class union_set {
  friend inline union_set manage(__isl_take isl_union_set *ptr);

  isl_union_set *ptr = nullptr;

  inline explicit union_set(__isl_take isl_union_set *ptr);

public:
  inline /* implicit */ union_set();
  inline /* implicit */ union_set(const union_set &obj);
  inline /* implicit */ union_set(std::nullptr_t);
  inline union_set &operator=(union_set obj);
  inline ~union_set();
  inline __isl_give isl_union_set *copy() const &;
  inline __isl_give isl_union_set *copy() && = delete;
  inline __isl_keep isl_union_set *get() const;
  inline __isl_give isl_union_set *release();
  inline __isl_keep isl_union_set *keep() const;
  inline __isl_give isl_union_set *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// declarations for isl::val
inline val manage(__isl_take isl_val *ptr);

inline val give(__isl_take isl_val *ptr);

class val {
  friend inline val manage(__isl_take isl_val *ptr);

  isl_val *ptr = nullptr;

  inline explicit val(__isl_take isl_val *ptr);

public:
  inline /* implicit */ val();
  inline /* implicit */ val(const val &obj);
  inline /* implicit */ val(std::nullptr_t);
  inline val &operator=(val obj);
  inline ~val();
  inline __isl_give isl_val *copy() const &;
  inline __isl_give isl_val *copy() && = delete;
  inline __isl_keep isl_val *get() const;
  inline __isl_give isl_val *release();
  inline __isl_keep isl_val *keep() const;
  inline __isl_give isl_val *take();
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline bool is_null() const;
  inline std::string to_str() const;
};

// implementations for isl::aff
aff manage(__isl_take isl_aff *ptr) {
  return aff(ptr);
}

aff give(__isl_take isl_aff *ptr) {
  return manage(ptr);
}

aff::aff()
    : ptr(nullptr) {}

aff::aff(const aff &obj)
    : ptr(obj.copy()) {}

aff::aff(std::nullptr_t)
    : ptr(nullptr) {}

aff::aff(__isl_take isl_aff *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_aff *aff::keep() const {
  return get();
}

__isl_give isl_aff *aff::take() {
  return release();
}

aff::operator bool() const {
  return !is_null();
}

ctx aff::get_ctx() const {
  return ctx(isl_aff_get_ctx(ptr));
}

bool aff::is_null() const {
  return ptr == nullptr;
}

std::string aff::to_str() const {
  char *Tmp = isl_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const aff &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::ast_build
ast_build manage(__isl_take isl_ast_build *ptr) {
  return ast_build(ptr);
}

ast_build give(__isl_take isl_ast_build *ptr) {
  return manage(ptr);
}

ast_build::ast_build()
    : ptr(nullptr) {}

ast_build::ast_build(const ast_build &obj)
    : ptr(obj.copy()) {}

ast_build::ast_build(std::nullptr_t)
    : ptr(nullptr) {}

ast_build::ast_build(__isl_take isl_ast_build *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_ast_build *ast_build::keep() const {
  return get();
}

__isl_give isl_ast_build *ast_build::take() {
  return release();
}

ast_build::operator bool() const {
  return !is_null();
}

ctx ast_build::get_ctx() const {
  return ctx(isl_ast_build_get_ctx(ptr));
}

bool ast_build::is_null() const {
  return ptr == nullptr;
}

// implementations for isl::ast_expr
ast_expr manage(__isl_take isl_ast_expr *ptr) {
  return ast_expr(ptr);
}

ast_expr give(__isl_take isl_ast_expr *ptr) {
  return manage(ptr);
}

ast_expr::ast_expr()
    : ptr(nullptr) {}

ast_expr::ast_expr(const ast_expr &obj)
    : ptr(obj.copy()) {}

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

__isl_keep isl_ast_expr *ast_expr::keep() const {
  return get();
}

__isl_give isl_ast_expr *ast_expr::take() {
  return release();
}

ast_expr::operator bool() const {
  return !is_null();
}

ctx ast_expr::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}

bool ast_expr::is_null() const {
  return ptr == nullptr;
}

std::string ast_expr::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const ast_expr &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::ast_node
ast_node manage(__isl_take isl_ast_node *ptr) {
  return ast_node(ptr);
}

ast_node give(__isl_take isl_ast_node *ptr) {
  return manage(ptr);
}

ast_node::ast_node()
    : ptr(nullptr) {}

ast_node::ast_node(const ast_node &obj)
    : ptr(obj.copy()) {}

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

__isl_keep isl_ast_node *ast_node::keep() const {
  return get();
}

__isl_give isl_ast_node *ast_node::take() {
  return release();
}

ast_node::operator bool() const {
  return !is_null();
}

ctx ast_node::get_ctx() const {
  return ctx(isl_ast_node_get_ctx(ptr));
}

bool ast_node::is_null() const {
  return ptr == nullptr;
}

std::string ast_node::to_str() const {
  char *Tmp = isl_ast_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const ast_node &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::basic_map
basic_map manage(__isl_take isl_basic_map *ptr) {
  return basic_map(ptr);
}

basic_map give(__isl_take isl_basic_map *ptr) {
  return manage(ptr);
}

basic_map::basic_map()
    : ptr(nullptr) {}

basic_map::basic_map(const basic_map &obj)
    : ptr(obj.copy()) {}

basic_map::basic_map(std::nullptr_t)
    : ptr(nullptr) {}

basic_map::basic_map(__isl_take isl_basic_map *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_basic_map *basic_map::keep() const {
  return get();
}

__isl_give isl_basic_map *basic_map::take() {
  return release();
}

basic_map::operator bool() const {
  return !is_null();
}

ctx basic_map::get_ctx() const {
  return ctx(isl_basic_map_get_ctx(ptr));
}

bool basic_map::is_null() const {
  return ptr == nullptr;
}

std::string basic_map::to_str() const {
  char *Tmp = isl_basic_map_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const basic_map &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::basic_set
basic_set manage(__isl_take isl_basic_set *ptr) {
  return basic_set(ptr);
}

basic_set give(__isl_take isl_basic_set *ptr) {
  return manage(ptr);
}

basic_set::basic_set()
    : ptr(nullptr) {}

basic_set::basic_set(const basic_set &obj)
    : ptr(obj.copy()) {}

basic_set::basic_set(std::nullptr_t)
    : ptr(nullptr) {}

basic_set::basic_set(__isl_take isl_basic_set *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_basic_set *basic_set::keep() const {
  return get();
}

__isl_give isl_basic_set *basic_set::take() {
  return release();
}

basic_set::operator bool() const {
  return !is_null();
}

ctx basic_set::get_ctx() const {
  return ctx(isl_basic_set_get_ctx(ptr));
}

bool basic_set::is_null() const {
  return ptr == nullptr;
}

std::string basic_set::to_str() const {
  char *Tmp = isl_basic_set_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const basic_set &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::id
id manage(__isl_take isl_id *ptr) {
  return id(ptr);
}

id give(__isl_take isl_id *ptr) {
  return manage(ptr);
}

id::id()
    : ptr(nullptr) {}

id::id(const id &obj)
    : ptr(obj.copy()) {}

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

__isl_keep isl_id *id::keep() const {
  return get();
}

__isl_give isl_id *id::take() {
  return release();
}

id::operator bool() const {
  return !is_null();
}

ctx id::get_ctx() const {
  return ctx(isl_id_get_ctx(ptr));
}

bool id::is_null() const {
  return ptr == nullptr;
}

std::string id::to_str() const {
  char *Tmp = isl_id_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const id &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::local_space
local_space manage(__isl_take isl_local_space *ptr) {
  return local_space(ptr);
}

local_space give(__isl_take isl_local_space *ptr) {
  return manage(ptr);
}

local_space::local_space()
    : ptr(nullptr) {}

local_space::local_space(const local_space &obj)
    : ptr(obj.copy()) {}

local_space::local_space(std::nullptr_t)
    : ptr(nullptr) {}

local_space::local_space(__isl_take isl_local_space *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_local_space *local_space::keep() const {
  return get();
}

__isl_give isl_local_space *local_space::take() {
  return release();
}

local_space::operator bool() const {
  return !is_null();
}

ctx local_space::get_ctx() const {
  return ctx(isl_local_space_get_ctx(ptr));
}

bool local_space::is_null() const {
  return ptr == nullptr;
}

// implementations for isl::map
map manage(__isl_take isl_map *ptr) {
  return map(ptr);
}

map give(__isl_take isl_map *ptr) {
  return manage(ptr);
}

map::map()
    : ptr(nullptr) {}

map::map(const map &obj)
    : ptr(obj.copy()) {}

map::map(std::nullptr_t)
    : ptr(nullptr) {}

map::map(__isl_take isl_map *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_map *map::keep() const {
  return get();
}

__isl_give isl_map *map::take() {
  return release();
}

map::operator bool() const {
  return !is_null();
}

ctx map::get_ctx() const {
  return ctx(isl_map_get_ctx(ptr));
}

bool map::is_null() const {
  return ptr == nullptr;
}

std::string map::to_str() const {
  char *Tmp = isl_map_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const map &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::multi_aff
multi_aff manage(__isl_take isl_multi_aff *ptr) {
  return multi_aff(ptr);
}

multi_aff give(__isl_take isl_multi_aff *ptr) {
  return manage(ptr);
}

multi_aff::multi_aff()
    : ptr(nullptr) {}

multi_aff::multi_aff(const multi_aff &obj)
    : ptr(obj.copy()) {}

multi_aff::multi_aff(std::nullptr_t)
    : ptr(nullptr) {}

multi_aff::multi_aff(__isl_take isl_multi_aff *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_multi_aff *multi_aff::keep() const {
  return get();
}

__isl_give isl_multi_aff *multi_aff::take() {
  return release();
}

multi_aff::operator bool() const {
  return !is_null();
}

ctx multi_aff::get_ctx() const {
  return ctx(isl_multi_aff_get_ctx(ptr));
}

bool multi_aff::is_null() const {
  return ptr == nullptr;
}

std::string multi_aff::to_str() const {
  char *Tmp = isl_multi_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const multi_aff &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::multi_pw_aff
multi_pw_aff manage(__isl_take isl_multi_pw_aff *ptr) {
  return multi_pw_aff(ptr);
}

multi_pw_aff give(__isl_take isl_multi_pw_aff *ptr) {
  return manage(ptr);
}

multi_pw_aff::multi_pw_aff()
    : ptr(nullptr) {}

multi_pw_aff::multi_pw_aff(const multi_pw_aff &obj)
    : ptr(obj.copy()) {}

multi_pw_aff::multi_pw_aff(std::nullptr_t)
    : ptr(nullptr) {}

multi_pw_aff::multi_pw_aff(__isl_take isl_multi_pw_aff *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_multi_pw_aff *multi_pw_aff::keep() const {
  return get();
}

__isl_give isl_multi_pw_aff *multi_pw_aff::take() {
  return release();
}

multi_pw_aff::operator bool() const {
  return !is_null();
}

ctx multi_pw_aff::get_ctx() const {
  return ctx(isl_multi_pw_aff_get_ctx(ptr));
}

bool multi_pw_aff::is_null() const {
  return ptr == nullptr;
}

std::string multi_pw_aff::to_str() const {
  char *Tmp = isl_multi_pw_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const multi_pw_aff &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::multi_union_pw_aff
multi_union_pw_aff manage(__isl_take isl_multi_union_pw_aff *ptr) {
  return multi_union_pw_aff(ptr);
}

multi_union_pw_aff give(__isl_take isl_multi_union_pw_aff *ptr) {
  return manage(ptr);
}

multi_union_pw_aff::multi_union_pw_aff()
    : ptr(nullptr) {}

multi_union_pw_aff::multi_union_pw_aff(const multi_union_pw_aff &obj)
    : ptr(obj.copy()) {}

multi_union_pw_aff::multi_union_pw_aff(std::nullptr_t)
    : ptr(nullptr) {}

multi_union_pw_aff::multi_union_pw_aff(__isl_take isl_multi_union_pw_aff *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_multi_union_pw_aff *multi_union_pw_aff::keep() const {
  return get();
}

__isl_give isl_multi_union_pw_aff *multi_union_pw_aff::take() {
  return release();
}

multi_union_pw_aff::operator bool() const {
  return !is_null();
}

ctx multi_union_pw_aff::get_ctx() const {
  return ctx(isl_multi_union_pw_aff_get_ctx(ptr));
}

bool multi_union_pw_aff::is_null() const {
  return ptr == nullptr;
}

std::string multi_union_pw_aff::to_str() const {
  char *Tmp = isl_multi_union_pw_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const multi_union_pw_aff &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::multi_val
multi_val manage(__isl_take isl_multi_val *ptr) {
  return multi_val(ptr);
}

multi_val give(__isl_take isl_multi_val *ptr) {
  return manage(ptr);
}

multi_val::multi_val()
    : ptr(nullptr) {}

multi_val::multi_val(const multi_val &obj)
    : ptr(obj.copy()) {}

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

__isl_keep isl_multi_val *multi_val::keep() const {
  return get();
}

__isl_give isl_multi_val *multi_val::take() {
  return release();
}

multi_val::operator bool() const {
  return !is_null();
}

ctx multi_val::get_ctx() const {
  return ctx(isl_multi_val_get_ctx(ptr));
}

bool multi_val::is_null() const {
  return ptr == nullptr;
}

std::string multi_val::to_str() const {
  char *Tmp = isl_multi_val_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const multi_val &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::point
point manage(__isl_take isl_point *ptr) {
  return point(ptr);
}

point give(__isl_take isl_point *ptr) {
  return manage(ptr);
}

point::point()
    : ptr(nullptr) {}

point::point(const point &obj)
    : ptr(obj.copy()) {}

point::point(std::nullptr_t)
    : ptr(nullptr) {}

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

__isl_keep isl_point *point::keep() const {
  return get();
}

__isl_give isl_point *point::take() {
  return release();
}

point::operator bool() const {
  return !is_null();
}

ctx point::get_ctx() const {
  return ctx(isl_point_get_ctx(ptr));
}

bool point::is_null() const {
  return ptr == nullptr;
}

std::string point::to_str() const {
  char *Tmp = isl_point_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const point &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::pw_aff
pw_aff manage(__isl_take isl_pw_aff *ptr) {
  return pw_aff(ptr);
}

pw_aff give(__isl_take isl_pw_aff *ptr) {
  return manage(ptr);
}

pw_aff::pw_aff()
    : ptr(nullptr) {}

pw_aff::pw_aff(const pw_aff &obj)
    : ptr(obj.copy()) {}

pw_aff::pw_aff(std::nullptr_t)
    : ptr(nullptr) {}

pw_aff::pw_aff(__isl_take isl_pw_aff *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_pw_aff *pw_aff::keep() const {
  return get();
}

__isl_give isl_pw_aff *pw_aff::take() {
  return release();
}

pw_aff::operator bool() const {
  return !is_null();
}

ctx pw_aff::get_ctx() const {
  return ctx(isl_pw_aff_get_ctx(ptr));
}

bool pw_aff::is_null() const {
  return ptr == nullptr;
}

std::string pw_aff::to_str() const {
  char *Tmp = isl_pw_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const pw_aff &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::pw_multi_aff
pw_multi_aff manage(__isl_take isl_pw_multi_aff *ptr) {
  return pw_multi_aff(ptr);
}

pw_multi_aff give(__isl_take isl_pw_multi_aff *ptr) {
  return manage(ptr);
}

pw_multi_aff::pw_multi_aff()
    : ptr(nullptr) {}

pw_multi_aff::pw_multi_aff(const pw_multi_aff &obj)
    : ptr(obj.copy()) {}

pw_multi_aff::pw_multi_aff(std::nullptr_t)
    : ptr(nullptr) {}

pw_multi_aff::pw_multi_aff(__isl_take isl_pw_multi_aff *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_pw_multi_aff *pw_multi_aff::keep() const {
  return get();
}

__isl_give isl_pw_multi_aff *pw_multi_aff::take() {
  return release();
}

pw_multi_aff::operator bool() const {
  return !is_null();
}

ctx pw_multi_aff::get_ctx() const {
  return ctx(isl_pw_multi_aff_get_ctx(ptr));
}

bool pw_multi_aff::is_null() const {
  return ptr == nullptr;
}

std::string pw_multi_aff::to_str() const {
  char *Tmp = isl_pw_multi_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const pw_multi_aff &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::schedule
schedule manage(__isl_take isl_schedule *ptr) {
  return schedule(ptr);
}

schedule give(__isl_take isl_schedule *ptr) {
  return manage(ptr);
}

schedule::schedule()
    : ptr(nullptr) {}

schedule::schedule(const schedule &obj)
    : ptr(obj.copy()) {}

schedule::schedule(std::nullptr_t)
    : ptr(nullptr) {}

schedule::schedule(__isl_take isl_schedule *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_schedule *schedule::keep() const {
  return get();
}

__isl_give isl_schedule *schedule::take() {
  return release();
}

schedule::operator bool() const {
  return !is_null();
}

ctx schedule::get_ctx() const {
  return ctx(isl_schedule_get_ctx(ptr));
}

bool schedule::is_null() const {
  return ptr == nullptr;
}

std::string schedule::to_str() const {
  char *Tmp = isl_schedule_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const schedule &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::schedule_constraints
schedule_constraints manage(__isl_take isl_schedule_constraints *ptr) {
  return schedule_constraints(ptr);
}

schedule_constraints give(__isl_take isl_schedule_constraints *ptr) {
  return manage(ptr);
}

schedule_constraints::schedule_constraints()
    : ptr(nullptr) {}

schedule_constraints::schedule_constraints(const schedule_constraints &obj)
    : ptr(obj.copy()) {}

schedule_constraints::schedule_constraints(std::nullptr_t)
    : ptr(nullptr) {}

schedule_constraints::schedule_constraints(__isl_take isl_schedule_constraints *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_schedule_constraints *schedule_constraints::keep() const {
  return get();
}

__isl_give isl_schedule_constraints *schedule_constraints::take() {
  return release();
}

schedule_constraints::operator bool() const {
  return !is_null();
}

ctx schedule_constraints::get_ctx() const {
  return ctx(isl_schedule_constraints_get_ctx(ptr));
}

bool schedule_constraints::is_null() const {
  return ptr == nullptr;
}

std::string schedule_constraints::to_str() const {
  char *Tmp = isl_schedule_constraints_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const schedule_constraints &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::schedule_node
schedule_node manage(__isl_take isl_schedule_node *ptr) {
  return schedule_node(ptr);
}

schedule_node give(__isl_take isl_schedule_node *ptr) {
  return manage(ptr);
}

schedule_node::schedule_node()
    : ptr(nullptr) {}

schedule_node::schedule_node(const schedule_node &obj)
    : ptr(obj.copy()) {}

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

__isl_keep isl_schedule_node *schedule_node::keep() const {
  return get();
}

__isl_give isl_schedule_node *schedule_node::take() {
  return release();
}

schedule_node::operator bool() const {
  return !is_null();
}

ctx schedule_node::get_ctx() const {
  return ctx(isl_schedule_node_get_ctx(ptr));
}

bool schedule_node::is_null() const {
  return ptr == nullptr;
}

std::string schedule_node::to_str() const {
  char *Tmp = isl_schedule_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const schedule_node &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::set
set manage(__isl_take isl_set *ptr) {
  return set(ptr);
}

set give(__isl_take isl_set *ptr) {
  return manage(ptr);
}

set::set()
    : ptr(nullptr) {}

set::set(const set &obj)
    : ptr(obj.copy()) {}

set::set(std::nullptr_t)
    : ptr(nullptr) {}

set::set(__isl_take isl_set *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_set *set::keep() const {
  return get();
}

__isl_give isl_set *set::take() {
  return release();
}

set::operator bool() const {
  return !is_null();
}

ctx set::get_ctx() const {
  return ctx(isl_set_get_ctx(ptr));
}

bool set::is_null() const {
  return ptr == nullptr;
}

std::string set::to_str() const {
  char *Tmp = isl_set_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const set &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::space
space manage(__isl_take isl_space *ptr) {
  return space(ptr);
}

space give(__isl_take isl_space *ptr) {
  return manage(ptr);
}

space::space()
    : ptr(nullptr) {}

space::space(const space &obj)
    : ptr(obj.copy()) {}

space::space(std::nullptr_t)
    : ptr(nullptr) {}

space::space(__isl_take isl_space *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_space *space::keep() const {
  return get();
}

__isl_give isl_space *space::take() {
  return release();
}

space::operator bool() const {
  return !is_null();
}

ctx space::get_ctx() const {
  return ctx(isl_space_get_ctx(ptr));
}

bool space::is_null() const {
  return ptr == nullptr;
}

std::string space::to_str() const {
  char *Tmp = isl_space_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const space &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::union_access_info
union_access_info manage(__isl_take isl_union_access_info *ptr) {
  return union_access_info(ptr);
}

union_access_info give(__isl_take isl_union_access_info *ptr) {
  return manage(ptr);
}

union_access_info::union_access_info()
    : ptr(nullptr) {}

union_access_info::union_access_info(const union_access_info &obj)
    : ptr(obj.copy()) {}

union_access_info::union_access_info(std::nullptr_t)
    : ptr(nullptr) {}

union_access_info::union_access_info(__isl_take isl_union_access_info *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_union_access_info *union_access_info::keep() const {
  return get();
}

__isl_give isl_union_access_info *union_access_info::take() {
  return release();
}

union_access_info::operator bool() const {
  return !is_null();
}

ctx union_access_info::get_ctx() const {
  return ctx(isl_union_access_info_get_ctx(ptr));
}

bool union_access_info::is_null() const {
  return ptr == nullptr;
}

std::string union_access_info::to_str() const {
  char *Tmp = isl_union_access_info_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const union_access_info &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::union_flow
union_flow manage(__isl_take isl_union_flow *ptr) {
  return union_flow(ptr);
}

union_flow give(__isl_take isl_union_flow *ptr) {
  return manage(ptr);
}

union_flow::union_flow()
    : ptr(nullptr) {}

union_flow::union_flow(const union_flow &obj)
    : ptr(obj.copy()) {}

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

__isl_keep isl_union_flow *union_flow::keep() const {
  return get();
}

__isl_give isl_union_flow *union_flow::take() {
  return release();
}

union_flow::operator bool() const {
  return !is_null();
}

ctx union_flow::get_ctx() const {
  return ctx(isl_union_flow_get_ctx(ptr));
}

bool union_flow::is_null() const {
  return ptr == nullptr;
}

std::string union_flow::to_str() const {
  char *Tmp = isl_union_flow_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const union_flow &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::union_map
union_map manage(__isl_take isl_union_map *ptr) {
  return union_map(ptr);
}

union_map give(__isl_take isl_union_map *ptr) {
  return manage(ptr);
}

union_map::union_map()
    : ptr(nullptr) {}

union_map::union_map(const union_map &obj)
    : ptr(obj.copy()) {}

union_map::union_map(std::nullptr_t)
    : ptr(nullptr) {}

union_map::union_map(__isl_take isl_union_map *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_union_map *union_map::keep() const {
  return get();
}

__isl_give isl_union_map *union_map::take() {
  return release();
}

union_map::operator bool() const {
  return !is_null();
}

ctx union_map::get_ctx() const {
  return ctx(isl_union_map_get_ctx(ptr));
}

bool union_map::is_null() const {
  return ptr == nullptr;
}

std::string union_map::to_str() const {
  char *Tmp = isl_union_map_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const union_map &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::union_pw_aff
union_pw_aff manage(__isl_take isl_union_pw_aff *ptr) {
  return union_pw_aff(ptr);
}

union_pw_aff give(__isl_take isl_union_pw_aff *ptr) {
  return manage(ptr);
}

union_pw_aff::union_pw_aff()
    : ptr(nullptr) {}

union_pw_aff::union_pw_aff(const union_pw_aff &obj)
    : ptr(obj.copy()) {}

union_pw_aff::union_pw_aff(std::nullptr_t)
    : ptr(nullptr) {}

union_pw_aff::union_pw_aff(__isl_take isl_union_pw_aff *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_union_pw_aff *union_pw_aff::keep() const {
  return get();
}

__isl_give isl_union_pw_aff *union_pw_aff::take() {
  return release();
}

union_pw_aff::operator bool() const {
  return !is_null();
}

ctx union_pw_aff::get_ctx() const {
  return ctx(isl_union_pw_aff_get_ctx(ptr));
}

bool union_pw_aff::is_null() const {
  return ptr == nullptr;
}

std::string union_pw_aff::to_str() const {
  char *Tmp = isl_union_pw_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const union_pw_aff &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::union_pw_multi_aff
union_pw_multi_aff manage(__isl_take isl_union_pw_multi_aff *ptr) {
  return union_pw_multi_aff(ptr);
}

union_pw_multi_aff give(__isl_take isl_union_pw_multi_aff *ptr) {
  return manage(ptr);
}

union_pw_multi_aff::union_pw_multi_aff()
    : ptr(nullptr) {}

union_pw_multi_aff::union_pw_multi_aff(const union_pw_multi_aff &obj)
    : ptr(obj.copy()) {}

union_pw_multi_aff::union_pw_multi_aff(std::nullptr_t)
    : ptr(nullptr) {}

union_pw_multi_aff::union_pw_multi_aff(__isl_take isl_union_pw_multi_aff *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_union_pw_multi_aff *union_pw_multi_aff::keep() const {
  return get();
}

__isl_give isl_union_pw_multi_aff *union_pw_multi_aff::take() {
  return release();
}

union_pw_multi_aff::operator bool() const {
  return !is_null();
}

ctx union_pw_multi_aff::get_ctx() const {
  return ctx(isl_union_pw_multi_aff_get_ctx(ptr));
}

bool union_pw_multi_aff::is_null() const {
  return ptr == nullptr;
}

std::string union_pw_multi_aff::to_str() const {
  char *Tmp = isl_union_pw_multi_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const union_pw_multi_aff &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::union_set
union_set manage(__isl_take isl_union_set *ptr) {
  return union_set(ptr);
}

union_set give(__isl_take isl_union_set *ptr) {
  return manage(ptr);
}

union_set::union_set()
    : ptr(nullptr) {}

union_set::union_set(const union_set &obj)
    : ptr(obj.copy()) {}

union_set::union_set(std::nullptr_t)
    : ptr(nullptr) {}

union_set::union_set(__isl_take isl_union_set *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_union_set *union_set::keep() const {
  return get();
}

__isl_give isl_union_set *union_set::take() {
  return release();
}

union_set::operator bool() const {
  return !is_null();
}

ctx union_set::get_ctx() const {
  return ctx(isl_union_set_get_ctx(ptr));
}

bool union_set::is_null() const {
  return ptr == nullptr;
}

std::string union_set::to_str() const {
  char *Tmp = isl_union_set_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const union_set &Obj) {
  OS << Obj.to_str();
  return OS;
}

// implementations for isl::val
val manage(__isl_take isl_val *ptr) {
  return val(ptr);
}

val give(__isl_take isl_val *ptr) {
  return manage(ptr);
}

val::val()
    : ptr(nullptr) {}

val::val(const val &obj)
    : ptr(obj.copy()) {}

val::val(std::nullptr_t)
    : ptr(nullptr) {}

val::val(__isl_take isl_val *ptr)
    : ptr(ptr) {}

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

__isl_keep isl_val *val::keep() const {
  return get();
}

__isl_give isl_val *val::take() {
  return release();
}

val::operator bool() const {
  return !is_null();
}

ctx val::get_ctx() const {
  return ctx(isl_val_get_ctx(ptr));
}

bool val::is_null() const {
  return ptr == nullptr;
}

std::string val::to_str() const {
  char *Tmp = isl_val_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const val &Obj) {
  OS << Obj.to_str();
  return OS;
}

} // namespace noexceptions
} // namespace isl

#endif /* ISL_CPP_NOEXCEPTIONS */
