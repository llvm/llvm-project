/// These are automatically generated C++ bindings for isl.
///
/// isl is a library for computing with integer sets and maps described by
/// Presburger formulas. On top of this, isl provides various tools for
/// polyhedral compilation, ranging from dependence analysis over scheduling
/// to AST generation.

#ifndef ISL_CPP
#define ISL_CPP

#include <isl/id.h>
#include <isl/space.h>
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
#include <isl/fixed_box.h>

#include <isl/ctx.h>
#include <isl/options.h>

#include <functional>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>

/* ISL_USE_EXCEPTIONS should be defined to 1 if exceptions are available.
 * gcc and clang define __cpp_exceptions; MSVC and xlC define _CPPUNWIND.
 * Older versions of gcc (e.g., 4.9) only define __EXCEPTIONS.
 * If exceptions are not available, any error condition will result
 * in an abort.
 */
#ifndef ISL_USE_EXCEPTIONS
#if defined(__cpp_exceptions) || defined(_CPPUNWIND) || defined(__EXCEPTIONS)
#define ISL_USE_EXCEPTIONS	1
#else
#define ISL_USE_EXCEPTIONS	0
#endif
#endif

namespace isl {

class ctx {
	isl_ctx *ptr;
public:
	/* implicit */ ctx(isl_ctx *ctx) : ptr(ctx) {}
	isl_ctx *release() {
		auto tmp = ptr;
		ptr = nullptr;
		return tmp;
	}
	isl_ctx *get() {
		return ptr;
	}
};

/* Macros hiding try/catch.
 * If exceptions are not available, then no exceptions will be thrown and
 * there is nothing to catch.
 */
#if ISL_USE_EXCEPTIONS
#define ISL_CPP_TRY		try
#define ISL_CPP_CATCH_ALL	catch (...)
#else
#define ISL_CPP_TRY		if (1)
#define ISL_CPP_CATCH_ALL	if (0)
#endif

#if ISL_USE_EXCEPTIONS

/* Class capturing isl errors.
 *
 * The what() return value is stored in a reference counted string
 * to ensure that the copy constructor and the assignment operator
 * do not throw any exceptions.
 */
class exception : public std::exception {
	std::shared_ptr<std::string> what_str;

protected:
	inline exception(const char *what_arg, const char *msg,
		const char *file, int line);
public:
	exception() {}
	exception(const char *what_arg) {
		what_str = std::make_shared<std::string>(what_arg);
	}
	static inline void throw_error(enum isl_error error, const char *msg,
		const char *file, int line);
	virtual const char *what() const noexcept {
		return what_str->c_str();
	}

	/* Default behavior on error conditions that occur inside isl calls
	 * performed from inside the bindings.
	 * In the case exceptions are available, isl should continue
	 * without printing a warning since the warning message
	 * will be included in the exception thrown from inside the bindings.
	 */
	static constexpr auto on_error = ISL_ON_ERROR_CONTINUE;
	/* Wrapper for throwing an exception with the given message.
	 */
	static void throw_invalid(const char *msg, const char *file, int line) {
		throw_error(isl_error_invalid, msg, file, line);
	}
	static inline void throw_last_error(ctx ctx);
};

/* Create an exception of a type described by "what_arg", with
 * error message "msg" in line "line" of file "file".
 *
 * Create a string holding the what() return value that
 * corresponds to what isl would have printed.
 * If no error message or no error file was set, then use "what_arg" instead.
 */
exception::exception(const char *what_arg, const char *msg, const char *file,
	int line)
{
	if (!msg || !file)
		what_str = std::make_shared<std::string>(what_arg);
	else
		what_str = std::make_shared<std::string>(std::string(file) +
				    ":" + std::to_string(line) + ": " + msg);
}

class exception_abort : public exception {
	friend exception;
	exception_abort(const char *msg, const char *file, int line) :
		exception("execution aborted", msg, file, line) {}
};

class exception_alloc : public exception {
	friend exception;
	exception_alloc(const char *msg, const char *file, int line) :
		exception("memory allocation failure", msg, file, line) {}
};

class exception_unknown : public exception {
	friend exception;
	exception_unknown(const char *msg, const char *file, int line) :
		exception("unknown failure", msg, file, line) {}
};

class exception_internal : public exception {
	friend exception;
	exception_internal(const char *msg, const char *file, int line) :
		exception("internal error", msg, file, line) {}
};

class exception_invalid : public exception {
	friend exception;
	exception_invalid(const char *msg, const char *file, int line) :
		exception("invalid argument", msg, file, line) {}
};

class exception_quota : public exception {
	friend exception;
	exception_quota(const char *msg, const char *file, int line) :
		exception("quota exceeded", msg, file, line) {}
};

class exception_unsupported : public exception {
	friend exception;
	exception_unsupported(const char *msg, const char *file, int line) :
		exception("unsupported operation", msg, file, line) {}
};

/* Throw an exception of the class that corresponds to "error", with
 * error message "msg" in line "line" of file "file".
 *
 * isl_error_none is treated as an invalid error type.
 */
void exception::throw_error(enum isl_error error, const char *msg,
	const char *file, int line)
{
	switch (error) {
	case isl_error_none:
		break;
	case isl_error_abort: throw exception_abort(msg, file, line);
	case isl_error_alloc: throw exception_alloc(msg, file, line);
	case isl_error_unknown: throw exception_unknown(msg, file, line);
	case isl_error_internal: throw exception_internal(msg, file, line);
	case isl_error_invalid: throw exception_invalid(msg, file, line);
	case isl_error_quota: throw exception_quota(msg, file, line);
	case isl_error_unsupported:
				throw exception_unsupported(msg, file, line);
	}

	throw exception_invalid("invalid error type", file, line);
}

/* Throw an exception corresponding to the last error on "ctx" and
 * reset the error.
 *
 * If "ctx" is NULL or if it is not in an error state at the start,
 * then an invalid argument exception is thrown.
 */
void exception::throw_last_error(ctx ctx)
{
	enum isl_error error;
	const char *msg, *file;
	int line;

	error = isl_ctx_last_error(ctx.get());
	msg = isl_ctx_last_error_msg(ctx.get());
	file = isl_ctx_last_error_file(ctx.get());
	line = isl_ctx_last_error_line(ctx.get());
	isl_ctx_reset_error(ctx.get());

	throw_error(error, msg, file, line);
}

#else

#include <stdio.h>
#include <stdlib.h>

class exception {
public:
	/* Default behavior on error conditions that occur inside isl calls
	 * performed from inside the bindings.
	 * In the case exceptions are not available, isl should abort.
	 */
	static constexpr auto on_error = ISL_ON_ERROR_ABORT;
	/* Wrapper for throwing an exception with the given message.
	 * In the case exceptions are not available, print an error and abort.
	 */
	static void throw_invalid(const char *msg, const char *file, int line) {
		fprintf(stderr, "%s:%d: %s\n", file, line, msg);
		abort();
	}
	/* Throw an exception corresponding to the last
	 * error on "ctx".
	 * isl should already abort when an error condition occurs,
	 * so this function should never be called.
	 */
	static void throw_last_error(ctx ctx) {
		abort();
	}
};

#endif

/* Helper class for setting the on_error and resetting the option
 * to the original value when leaving the scope.
 */
class options_scoped_set_on_error {
	isl_ctx *ctx;
	int saved_on_error;
public:
	options_scoped_set_on_error(class ctx ctx, int on_error) {
		this->ctx = ctx.get();
		saved_on_error = isl_options_get_on_error(this->ctx);
		isl_options_set_on_error(this->ctx, on_error);
	}
	~options_scoped_set_on_error() {
		isl_options_set_on_error(ctx, saved_on_error);
	}
};

} // namespace isl

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
class basic_set;
class fixed_box;
class id;
class id_list;
class map;
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
  inline aff &operator=(aff obj);
  inline ~aff();
  inline __isl_give isl_aff *copy() const &;
  inline __isl_give isl_aff *copy() && = delete;
  inline __isl_keep isl_aff *get() const;
  inline __isl_give isl_aff *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::aff add(isl::aff aff2) const;
  inline isl::aff add_constant(isl::val v) const;
  inline isl::aff add_constant(long v) const;
  inline isl::basic_set bind(isl::id id) const;
  inline isl::basic_set bind(const std::string &id) const;
  inline isl::aff ceil() const;
  inline isl::aff div(isl::aff aff2) const;
  inline isl::set eq_set(isl::aff aff2) const;
  inline isl::val eval(isl::point pnt) const;
  inline isl::aff floor() const;
  inline isl::set ge_set(isl::aff aff2) const;
  inline isl::val constant_val() const;
  inline isl::val get_constant_val() const;
  inline isl::aff gist(isl::set context) const;
  inline isl::set gt_set(isl::aff aff2) const;
  inline bool is_cst() const;
  inline isl::set le_set(isl::aff aff2) const;
  inline isl::set lt_set(isl::aff aff2) const;
  inline isl::aff mod(isl::val mod) const;
  inline isl::aff mod(long mod) const;
  inline isl::aff mul(isl::aff aff2) const;
  inline isl::set ne_set(isl::aff aff2) const;
  inline isl::aff neg() const;
  inline isl::aff pullback(isl::multi_aff ma) const;
  inline isl::aff scale(isl::val v) const;
  inline isl::aff scale(long v) const;
  inline isl::aff scale_down(isl::val v) const;
  inline isl::aff scale_down(long v) const;
  inline isl::aff sub(isl::aff aff2) const;
  inline isl::aff unbind_params_insert_domain(isl::multi_id domain) const;
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
  inline aff_list &operator=(aff_list obj);
  inline ~aff_list();
  inline __isl_give isl_aff_list *copy() const &;
  inline __isl_give isl_aff_list *copy() && = delete;
  inline __isl_keep isl_aff_list *get() const;
  inline __isl_give isl_aff_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::aff_list add(isl::aff el) const;
  inline isl::aff_list clear() const;
  inline isl::aff_list concat(isl::aff_list list2) const;
  inline isl::aff_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(isl::aff)> &fn) const;
  inline isl::aff at(int index) const;
  inline isl::aff get_at(int index) const;
  inline isl::aff_list insert(unsigned int pos, isl::aff el) const;
  inline unsigned size() const;
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
    std::exception_ptr eptr;
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
  inline isl::union_map schedule() const;
  inline isl::union_map get_schedule() const;
  inline isl::ast_node node_from(isl::schedule schedule) const;
  inline isl::ast_node node_from_schedule_map(isl::union_map schedule) const;
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
  inline bool isa_type(T subtype) const;
public:
  template <class T> inline bool isa() const;
  template <class T> inline T as() const;
  inline isl::ctx ctx() const;

  inline std::string to_C_str() const;
};

// declarations for isl::ast_expr_id

class ast_expr_id : public ast_expr {
  template <class T>
  friend bool ast_expr::isa() const;
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
  friend bool ast_expr::isa() const;
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
  friend bool ast_expr::isa() const;
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
  inline bool isa_type(T subtype) const;
public:
  template <class T> inline bool isa() const;
  template <class T> inline T as() const;
  inline isl::ctx ctx() const;

  inline isl::ast_expr arg(int pos) const;
  inline isl::ast_expr get_arg(int pos) const;
  inline unsigned n_arg() const;
  inline unsigned get_n_arg() const;
};

// declarations for isl::ast_expr_op_access

class ast_expr_op_access : public ast_expr_op {
  template <class T>
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  friend bool ast_expr_op::isa() const;
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
  inline bool isa_type(T subtype) const;
public:
  template <class T> inline bool isa() const;
  template <class T> inline T as() const;
  inline isl::ctx ctx() const;

  inline std::string to_C_str() const;
};

// declarations for isl::ast_node_block

class ast_node_block : public ast_node {
  template <class T>
  friend bool ast_node::isa() const;
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
  friend bool ast_node::isa() const;
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
  inline isl::ast_expr iterator() const;
  inline isl::ast_expr get_iterator() const;
  inline bool is_degenerate() const;
};

// declarations for isl::ast_node_if

class ast_node_if : public ast_node {
  template <class T>
  friend bool ast_node::isa() const;
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
  inline isl::ast_node then_node() const;
  inline isl::ast_node get_then_node() const;
  inline bool has_else_node() const;
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
  inline isl::ast_node_list clear() const;
  inline isl::ast_node_list concat(isl::ast_node_list list2) const;
  inline isl::ast_node_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(isl::ast_node)> &fn) const;
  inline isl::ast_node at(int index) const;
  inline isl::ast_node get_at(int index) const;
  inline isl::ast_node_list insert(unsigned int pos, isl::ast_node el) const;
  inline unsigned size() const;
};

// declarations for isl::ast_node_mark

class ast_node_mark : public ast_node {
  template <class T>
  friend bool ast_node::isa() const;
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
  friend bool ast_node::isa() const;
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

  inline isl::basic_map affine_hull() const;
  inline isl::basic_map apply_domain(isl::basic_map bmap2) const;
  inline isl::basic_map apply_range(isl::basic_map bmap2) const;
  inline isl::basic_set deltas() const;
  inline isl::basic_map detect_equalities() const;
  inline isl::basic_map flatten() const;
  inline isl::basic_map flatten_domain() const;
  inline isl::basic_map flatten_range() const;
  inline isl::basic_map gist(isl::basic_map context) const;
  inline isl::basic_map intersect(isl::basic_map bmap2) const;
  inline isl::basic_map intersect_domain(isl::basic_set bset) const;
  inline isl::basic_map intersect_range(isl::basic_set bset) const;
  inline bool is_empty() const;
  inline bool is_equal(const isl::basic_map &bmap2) const;
  inline bool is_subset(const isl::basic_map &bmap2) const;
  inline isl::map lexmax() const;
  inline isl::map lexmin() const;
  inline isl::basic_map reverse() const;
  inline isl::basic_map sample() const;
  inline isl::map unite(isl::basic_map bmap2) const;
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

  inline isl::basic_set affine_hull() const;
  inline isl::basic_set apply(isl::basic_map bmap) const;
  inline isl::basic_set detect_equalities() const;
  inline isl::val dim_max_val(int pos) const;
  inline isl::basic_set flatten() const;
  inline isl::basic_set gist(isl::basic_set context) const;
  inline isl::basic_set intersect(isl::basic_set bset2) const;
  inline isl::basic_set intersect_params(isl::basic_set bset2) const;
  inline bool is_empty() const;
  inline bool is_equal(const isl::basic_set &bset2) const;
  inline bool is_subset(const isl::basic_set &bset2) const;
  inline bool is_wrapping() const;
  inline isl::set lexmax() const;
  inline isl::set lexmin() const;
  inline isl::basic_set params() const;
  inline isl::basic_set sample() const;
  inline isl::point sample_point() const;
  inline isl::set unite(isl::basic_set bset2) const;
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

  inline isl::multi_aff offset() const;
  inline isl::multi_aff get_offset() const;
  inline isl::multi_val size() const;
  inline isl::multi_val get_size() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline bool is_valid() const;
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

  inline std::string name() const;
  inline std::string get_name() const;
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
  inline isl::id_list clear() const;
  inline isl::id_list concat(isl::id_list list2) const;
  inline isl::id_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(isl::id)> &fn) const;
  inline isl::id at(int index) const;
  inline isl::id get_at(int index) const;
  inline isl::id_list insert(unsigned int pos, isl::id el) const;
  inline isl::id_list insert(unsigned int pos, const std::string &el) const;
  inline unsigned size() const;
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

  inline isl::basic_map affine_hull() const;
  inline isl::map apply_domain(isl::map map2) const;
  inline isl::map apply_range(isl::map map2) const;
  inline isl::set bind_domain(isl::multi_id tuple) const;
  inline isl::set bind_range(isl::multi_id tuple) const;
  inline isl::map coalesce() const;
  inline isl::map complement() const;
  inline isl::map curry() const;
  inline isl::set deltas() const;
  inline isl::map detect_equalities() const;
  inline isl::set domain() const;
  inline isl::map domain_factor_domain() const;
  inline isl::map domain_factor_range() const;
  inline isl::map domain_product(isl::map map2) const;
  static inline isl::map empty(isl::space space);
  inline isl::map eq_at(isl::multi_pw_aff mpa) const;
  inline isl::map factor_domain() const;
  inline isl::map factor_range() const;
  inline isl::map flatten() const;
  inline isl::map flatten_domain() const;
  inline isl::map flatten_range() const;
  inline void foreach_basic_map(const std::function<void(isl::basic_map)> &fn) const;
  inline isl::fixed_box range_simple_fixed_box_hull() const;
  inline isl::fixed_box get_range_simple_fixed_box_hull() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::map gist(isl::map context) const;
  inline isl::map gist_domain(isl::set context) const;
  inline isl::map intersect(isl::map map2) const;
  inline isl::map intersect_domain(isl::set set) const;
  inline isl::map intersect_domain_factor_domain(isl::map factor) const;
  inline isl::map intersect_domain_factor_range(isl::map factor) const;
  inline isl::map intersect_params(isl::set params) const;
  inline isl::map intersect_range(isl::set set) const;
  inline isl::map intersect_range_factor_domain(isl::map factor) const;
  inline isl::map intersect_range_factor_range(isl::map factor) const;
  inline bool is_bijective() const;
  inline bool is_disjoint(const isl::map &map2) const;
  inline bool is_empty() const;
  inline bool is_equal(const isl::map &map2) const;
  inline bool is_injective() const;
  inline bool is_single_valued() const;
  inline bool is_strict_subset(const isl::map &map2) const;
  inline bool is_subset(const isl::map &map2) const;
  inline isl::map lex_ge_at(isl::multi_pw_aff mpa) const;
  inline isl::map lex_gt_at(isl::multi_pw_aff mpa) const;
  inline isl::map lex_le_at(isl::multi_pw_aff mpa) const;
  inline isl::map lex_lt_at(isl::multi_pw_aff mpa) const;
  inline isl::map lexmax() const;
  inline isl::pw_multi_aff lexmax_pw_multi_aff() const;
  inline isl::map lexmin() const;
  inline isl::pw_multi_aff lexmin_pw_multi_aff() const;
  inline isl::map lower_bound(isl::multi_pw_aff lower) const;
  inline isl::multi_pw_aff max_multi_pw_aff() const;
  inline isl::multi_pw_aff min_multi_pw_aff() const;
  inline isl::basic_map polyhedral_hull() const;
  inline isl::map preimage_domain(isl::multi_aff ma) const;
  inline isl::map preimage_domain(isl::multi_pw_aff mpa) const;
  inline isl::map preimage_domain(isl::pw_multi_aff pma) const;
  inline isl::map preimage_range(isl::multi_aff ma) const;
  inline isl::map preimage_range(isl::pw_multi_aff pma) const;
  inline isl::map product(isl::map map2) const;
  inline isl::map project_out_all_params() const;
  inline isl::set range() const;
  inline isl::map range_factor_domain() const;
  inline isl::map range_factor_range() const;
  inline isl::map range_product(isl::map map2) const;
  inline isl::map range_reverse() const;
  inline isl::map reverse() const;
  inline isl::basic_map sample() const;
  inline isl::map subtract(isl::map map2) const;
  inline isl::map uncurry() const;
  inline isl::map unite(isl::map map2) const;
  static inline isl::map universe(isl::space space);
  inline isl::basic_map unshifted_simple_hull() const;
  inline isl::map upper_bound(isl::multi_pw_aff upper) const;
  inline isl::set wrap() const;
  inline isl::map zip() const;
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
  inline isl::multi_aff add_constant(isl::multi_val mv) const;
  inline isl::multi_aff add_constant(isl::val v) const;
  inline isl::multi_aff add_constant(long v) const;
  inline isl::basic_set bind(isl::multi_id tuple) const;
  inline isl::multi_aff bind_domain(isl::multi_id tuple) const;
  inline isl::multi_aff bind_domain_wrapped_domain(isl::multi_id tuple) const;
  static inline isl::multi_aff domain_map(isl::space space);
  inline isl::multi_aff flat_range_product(isl::multi_aff multi2) const;
  inline isl::multi_aff floor() const;
  inline isl::aff at(int pos) const;
  inline isl::aff get_at(int pos) const;
  inline isl::multi_val constant_multi_val() const;
  inline isl::multi_val get_constant_multi_val() const;
  inline isl::aff_list list() const;
  inline isl::aff_list get_list() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::multi_aff gist(isl::set context) const;
  inline isl::multi_aff identity() const;
  static inline isl::multi_aff identity_on_domain(isl::space space);
  inline isl::multi_aff insert_domain(isl::space domain) const;
  inline bool involves_locals() const;
  inline bool involves_nan() const;
  inline isl::multi_aff neg() const;
  inline bool plain_is_equal(const isl::multi_aff &multi2) const;
  inline isl::multi_aff product(isl::multi_aff multi2) const;
  inline isl::multi_aff pullback(isl::multi_aff ma2) const;
  static inline isl::multi_aff range_map(isl::space space);
  inline isl::multi_aff range_product(isl::multi_aff multi2) const;
  inline isl::multi_aff scale(isl::multi_val mv) const;
  inline isl::multi_aff scale(isl::val v) const;
  inline isl::multi_aff scale(long v) const;
  inline isl::multi_aff scale_down(isl::multi_val mv) const;
  inline isl::multi_aff scale_down(isl::val v) const;
  inline isl::multi_aff scale_down(long v) const;
  inline isl::multi_aff set_at(int pos, isl::aff el) const;
  inline unsigned size() const;
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

  inline isl::multi_id flat_range_product(isl::multi_id multi2) const;
  inline isl::id at(int pos) const;
  inline isl::id get_at(int pos) const;
  inline isl::id_list list() const;
  inline isl::id_list get_list() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline bool plain_is_equal(const isl::multi_id &multi2) const;
  inline isl::multi_id range_product(isl::multi_id multi2) const;
  inline isl::multi_id set_at(int pos, isl::id el) const;
  inline isl::multi_id set_at(int pos, const std::string &el) const;
  inline unsigned size() const;
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
  inline isl::multi_pw_aff add_constant(isl::multi_val mv) const;
  inline isl::multi_pw_aff add_constant(isl::val v) const;
  inline isl::multi_pw_aff add_constant(long v) const;
  inline isl::set bind(isl::multi_id tuple) const;
  inline isl::multi_pw_aff bind_domain(isl::multi_id tuple) const;
  inline isl::multi_pw_aff bind_domain_wrapped_domain(isl::multi_id tuple) const;
  inline isl::multi_pw_aff coalesce() const;
  inline isl::set domain() const;
  inline isl::multi_pw_aff flat_range_product(isl::multi_pw_aff multi2) const;
  inline isl::pw_aff at(int pos) const;
  inline isl::pw_aff get_at(int pos) const;
  inline isl::pw_aff_list list() const;
  inline isl::pw_aff_list get_list() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::multi_pw_aff gist(isl::set set) const;
  inline isl::multi_pw_aff identity() const;
  static inline isl::multi_pw_aff identity_on_domain(isl::space space);
  inline isl::multi_pw_aff insert_domain(isl::space domain) const;
  inline isl::multi_pw_aff intersect_domain(isl::set domain) const;
  inline isl::multi_pw_aff intersect_params(isl::set set) const;
  inline bool involves_nan() const;
  inline bool involves_param(const isl::id &id) const;
  inline bool involves_param(const std::string &id) const;
  inline bool involves_param(const isl::id_list &list) const;
  inline isl::multi_pw_aff max(isl::multi_pw_aff multi2) const;
  inline isl::multi_val max_multi_val() const;
  inline isl::multi_pw_aff min(isl::multi_pw_aff multi2) const;
  inline isl::multi_val min_multi_val() const;
  inline isl::multi_pw_aff neg() const;
  inline bool plain_is_equal(const isl::multi_pw_aff &multi2) const;
  inline isl::multi_pw_aff product(isl::multi_pw_aff multi2) const;
  inline isl::multi_pw_aff pullback(isl::multi_aff ma) const;
  inline isl::multi_pw_aff pullback(isl::multi_pw_aff mpa2) const;
  inline isl::multi_pw_aff pullback(isl::pw_multi_aff pma) const;
  inline isl::multi_pw_aff range_product(isl::multi_pw_aff multi2) const;
  inline isl::multi_pw_aff scale(isl::multi_val mv) const;
  inline isl::multi_pw_aff scale(isl::val v) const;
  inline isl::multi_pw_aff scale(long v) const;
  inline isl::multi_pw_aff scale_down(isl::multi_val mv) const;
  inline isl::multi_pw_aff scale_down(isl::val v) const;
  inline isl::multi_pw_aff scale_down(long v) const;
  inline isl::multi_pw_aff set_at(int pos, isl::pw_aff el) const;
  inline unsigned size() const;
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

protected:
  isl_multi_union_pw_aff *ptr = nullptr;

  inline explicit multi_union_pw_aff(__isl_take isl_multi_union_pw_aff *ptr);

public:
  inline /* implicit */ multi_union_pw_aff();
  inline /* implicit */ multi_union_pw_aff(const multi_union_pw_aff &obj);
  inline /* implicit */ multi_union_pw_aff(isl::multi_pw_aff mpa);
  inline /* implicit */ multi_union_pw_aff(isl::union_pw_aff upa);
  inline explicit multi_union_pw_aff(isl::space space, isl::union_pw_aff_list list);
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
  inline isl::union_set bind(isl::multi_id tuple) const;
  inline isl::multi_union_pw_aff coalesce() const;
  inline isl::union_set domain() const;
  inline isl::multi_union_pw_aff flat_range_product(isl::multi_union_pw_aff multi2) const;
  inline isl::union_pw_aff at(int pos) const;
  inline isl::union_pw_aff get_at(int pos) const;
  inline isl::union_pw_aff_list list() const;
  inline isl::union_pw_aff_list get_list() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::multi_union_pw_aff gist(isl::union_set context) const;
  inline isl::multi_union_pw_aff intersect_domain(isl::union_set uset) const;
  inline isl::multi_union_pw_aff intersect_params(isl::set params) const;
  inline bool involves_nan() const;
  inline isl::multi_union_pw_aff neg() const;
  inline bool plain_is_equal(const isl::multi_union_pw_aff &multi2) const;
  inline isl::multi_union_pw_aff pullback(isl::union_pw_multi_aff upma) const;
  inline isl::multi_union_pw_aff range_product(isl::multi_union_pw_aff multi2) const;
  inline isl::multi_union_pw_aff scale(isl::multi_val mv) const;
  inline isl::multi_union_pw_aff scale(isl::val v) const;
  inline isl::multi_union_pw_aff scale(long v) const;
  inline isl::multi_union_pw_aff scale_down(isl::multi_val mv) const;
  inline isl::multi_union_pw_aff scale_down(isl::val v) const;
  inline isl::multi_union_pw_aff scale_down(long v) const;
  inline isl::multi_union_pw_aff set_at(int pos, isl::union_pw_aff el) const;
  inline unsigned size() const;
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
  inline isl::multi_val flat_range_product(isl::multi_val multi2) const;
  inline isl::val at(int pos) const;
  inline isl::val get_at(int pos) const;
  inline isl::val_list list() const;
  inline isl::val_list get_list() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline bool involves_nan() const;
  inline isl::multi_val max(isl::multi_val multi2) const;
  inline isl::multi_val min(isl::multi_val multi2) const;
  inline isl::multi_val neg() const;
  inline bool plain_is_equal(const isl::multi_val &multi2) const;
  inline isl::multi_val product(isl::multi_val multi2) const;
  inline isl::multi_val range_product(isl::multi_val multi2) const;
  inline isl::multi_val scale(isl::multi_val mv) const;
  inline isl::multi_val scale(isl::val v) const;
  inline isl::multi_val scale(long v) const;
  inline isl::multi_val scale_down(isl::multi_val mv) const;
  inline isl::multi_val scale_down(isl::val v) const;
  inline isl::multi_val scale_down(long v) const;
  inline isl::multi_val set_at(int pos, isl::val el) const;
  inline isl::multi_val set_at(int pos, long el) const;
  inline unsigned size() const;
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
  inline point &operator=(point obj);
  inline ~point();
  inline __isl_give isl_point *copy() const &;
  inline __isl_give isl_point *copy() && = delete;
  inline __isl_keep isl_point *get() const;
  inline __isl_give isl_point *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::multi_val multi_val() const;
  inline isl::multi_val get_multi_val() const;
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
  inline pw_aff &operator=(pw_aff obj);
  inline ~pw_aff();
  inline __isl_give isl_pw_aff *copy() const &;
  inline __isl_give isl_pw_aff *copy() && = delete;
  inline __isl_keep isl_pw_aff *get() const;
  inline __isl_give isl_pw_aff *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::pw_aff add(isl::pw_aff pwaff2) const;
  inline isl::pw_aff add_constant(isl::val v) const;
  inline isl::pw_aff add_constant(long v) const;
  inline isl::aff as_aff() const;
  inline isl::set bind(isl::id id) const;
  inline isl::set bind(const std::string &id) const;
  inline isl::pw_aff bind_domain(isl::multi_id tuple) const;
  inline isl::pw_aff bind_domain_wrapped_domain(isl::multi_id tuple) const;
  inline isl::pw_aff ceil() const;
  inline isl::pw_aff coalesce() const;
  inline isl::pw_aff cond(isl::pw_aff pwaff_true, isl::pw_aff pwaff_false) const;
  inline isl::pw_aff div(isl::pw_aff pa2) const;
  inline isl::set domain() const;
  inline isl::set eq_set(isl::pw_aff pwaff2) const;
  inline isl::val eval(isl::point pnt) const;
  inline isl::pw_aff floor() const;
  inline isl::set ge_set(isl::pw_aff pwaff2) const;
  inline isl::pw_aff gist(isl::set context) const;
  inline isl::set gt_set(isl::pw_aff pwaff2) const;
  inline isl::pw_aff insert_domain(isl::space domain) const;
  inline isl::pw_aff intersect_domain(isl::set set) const;
  inline isl::pw_aff intersect_params(isl::set set) const;
  inline bool isa_aff() const;
  inline isl::set le_set(isl::pw_aff pwaff2) const;
  inline isl::set lt_set(isl::pw_aff pwaff2) const;
  inline isl::pw_aff max(isl::pw_aff pwaff2) const;
  inline isl::pw_aff min(isl::pw_aff pwaff2) const;
  inline isl::pw_aff mod(isl::val mod) const;
  inline isl::pw_aff mod(long mod) const;
  inline isl::pw_aff mul(isl::pw_aff pwaff2) const;
  inline isl::set ne_set(isl::pw_aff pwaff2) const;
  inline isl::pw_aff neg() const;
  static inline isl::pw_aff param_on_domain(isl::set domain, isl::id id);
  inline isl::pw_aff pullback(isl::multi_aff ma) const;
  inline isl::pw_aff pullback(isl::multi_pw_aff mpa) const;
  inline isl::pw_aff pullback(isl::pw_multi_aff pma) const;
  inline isl::pw_aff scale(isl::val v) const;
  inline isl::pw_aff scale(long v) const;
  inline isl::pw_aff scale_down(isl::val f) const;
  inline isl::pw_aff scale_down(long f) const;
  inline isl::pw_aff sub(isl::pw_aff pwaff2) const;
  inline isl::pw_aff subtract_domain(isl::set set) const;
  inline isl::pw_aff tdiv_q(isl::pw_aff pa2) const;
  inline isl::pw_aff tdiv_r(isl::pw_aff pa2) const;
  inline isl::pw_aff union_add(isl::pw_aff pwaff2) const;
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
  inline pw_aff_list &operator=(pw_aff_list obj);
  inline ~pw_aff_list();
  inline __isl_give isl_pw_aff_list *copy() const &;
  inline __isl_give isl_pw_aff_list *copy() && = delete;
  inline __isl_keep isl_pw_aff_list *get() const;
  inline __isl_give isl_pw_aff_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::pw_aff_list add(isl::pw_aff el) const;
  inline isl::pw_aff_list clear() const;
  inline isl::pw_aff_list concat(isl::pw_aff_list list2) const;
  inline isl::pw_aff_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(isl::pw_aff)> &fn) const;
  inline isl::pw_aff at(int index) const;
  inline isl::pw_aff get_at(int index) const;
  inline isl::pw_aff_list insert(unsigned int pos, isl::pw_aff el) const;
  inline unsigned size() const;
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

  inline isl::pw_multi_aff add(isl::pw_multi_aff pma2) const;
  inline isl::pw_multi_aff add_constant(isl::multi_val mv) const;
  inline isl::pw_multi_aff add_constant(isl::val v) const;
  inline isl::pw_multi_aff add_constant(long v) const;
  inline isl::multi_aff as_multi_aff() const;
  inline isl::pw_multi_aff bind_domain(isl::multi_id tuple) const;
  inline isl::pw_multi_aff bind_domain_wrapped_domain(isl::multi_id tuple) const;
  inline isl::pw_multi_aff coalesce() const;
  inline isl::set domain() const;
  static inline isl::pw_multi_aff domain_map(isl::space space);
  inline isl::pw_multi_aff flat_range_product(isl::pw_multi_aff pma2) const;
  inline void foreach_piece(const std::function<void(isl::set, isl::multi_aff)> &fn) const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::pw_multi_aff gist(isl::set set) const;
  static inline isl::pw_multi_aff identity_on_domain(isl::space space);
  inline isl::pw_multi_aff insert_domain(isl::space domain) const;
  inline isl::pw_multi_aff intersect_domain(isl::set set) const;
  inline isl::pw_multi_aff intersect_params(isl::set set) const;
  inline bool involves_locals() const;
  inline bool isa_multi_aff() const;
  inline isl::multi_val max_multi_val() const;
  inline isl::multi_val min_multi_val() const;
  inline unsigned n_piece() const;
  inline isl::pw_multi_aff preimage_domain_wrapped_domain(isl::pw_multi_aff pma2) const;
  inline isl::pw_multi_aff product(isl::pw_multi_aff pma2) const;
  inline isl::pw_multi_aff pullback(isl::multi_aff ma) const;
  inline isl::pw_multi_aff pullback(isl::pw_multi_aff pma2) const;
  inline isl::pw_multi_aff range_factor_domain() const;
  inline isl::pw_multi_aff range_factor_range() const;
  static inline isl::pw_multi_aff range_map(isl::space space);
  inline isl::pw_multi_aff range_product(isl::pw_multi_aff pma2) const;
  inline isl::pw_multi_aff scale(isl::val v) const;
  inline isl::pw_multi_aff scale(long v) const;
  inline isl::pw_multi_aff scale_down(isl::val v) const;
  inline isl::pw_multi_aff scale_down(long v) const;
  inline isl::pw_multi_aff sub(isl::pw_multi_aff pma2) const;
  inline isl::pw_multi_aff subtract_domain(isl::set set) const;
  inline isl::pw_multi_aff union_add(isl::pw_multi_aff pma2) const;
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
  inline pw_multi_aff_list &operator=(pw_multi_aff_list obj);
  inline ~pw_multi_aff_list();
  inline __isl_give isl_pw_multi_aff_list *copy() const &;
  inline __isl_give isl_pw_multi_aff_list *copy() && = delete;
  inline __isl_keep isl_pw_multi_aff_list *get() const;
  inline __isl_give isl_pw_multi_aff_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::pw_multi_aff_list add(isl::pw_multi_aff el) const;
  inline isl::pw_multi_aff_list clear() const;
  inline isl::pw_multi_aff_list concat(isl::pw_multi_aff_list list2) const;
  inline isl::pw_multi_aff_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(isl::pw_multi_aff)> &fn) const;
  inline isl::pw_multi_aff at(int index) const;
  inline isl::pw_multi_aff get_at(int index) const;
  inline isl::pw_multi_aff_list insert(unsigned int pos, isl::pw_multi_aff el) const;
  inline unsigned size() const;
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

  static inline isl::schedule from_domain(isl::union_set domain);
  inline isl::union_set domain() const;
  inline isl::union_set get_domain() const;
  inline isl::union_map map() const;
  inline isl::union_map get_map() const;
  inline isl::schedule_node root() const;
  inline isl::schedule_node get_root() const;
  inline isl::schedule pullback(isl::union_pw_multi_aff upma) const;
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

  inline isl::schedule compute_schedule() const;
  inline isl::union_map coincidence() const;
  inline isl::union_map get_coincidence() const;
  inline isl::union_map conditional_validity() const;
  inline isl::union_map get_conditional_validity() const;
  inline isl::union_map conditional_validity_condition() const;
  inline isl::union_map get_conditional_validity_condition() const;
  inline isl::set context() const;
  inline isl::set get_context() const;
  inline isl::union_set domain() const;
  inline isl::union_set get_domain() const;
  inline isl::union_map proximity() const;
  inline isl::union_map get_proximity() const;
  inline isl::union_map validity() const;
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
  inline bool isa_type(T subtype) const;
public:
  template <class T> inline bool isa() const;
  template <class T> inline T as() const;
  inline isl::ctx ctx() const;

  inline isl::schedule_node ancestor(int generation) const;
  inline isl::schedule_node child(int pos) const;
  inline bool every_descendant(const std::function<bool(isl::schedule_node)> &test) const;
  inline isl::schedule_node first_child() const;
  inline void foreach_ancestor_top_down(const std::function<void(isl::schedule_node)> &fn) const;
  inline void foreach_descendant_top_down(const std::function<bool(isl::schedule_node)> &fn) const;
  static inline isl::schedule_node from_domain(isl::union_set domain);
  static inline isl::schedule_node from_extension(isl::union_map extension);
  inline unsigned ancestor_child_position(const isl::schedule_node &ancestor) const;
  inline unsigned get_ancestor_child_position(const isl::schedule_node &ancestor) const;
  inline unsigned child_position() const;
  inline unsigned get_child_position() const;
  inline isl::multi_union_pw_aff prefix_schedule_multi_union_pw_aff() const;
  inline isl::multi_union_pw_aff get_prefix_schedule_multi_union_pw_aff() const;
  inline isl::union_map prefix_schedule_union_map() const;
  inline isl::union_map get_prefix_schedule_union_map() const;
  inline isl::union_pw_multi_aff prefix_schedule_union_pw_multi_aff() const;
  inline isl::union_pw_multi_aff get_prefix_schedule_union_pw_multi_aff() const;
  inline isl::schedule schedule() const;
  inline isl::schedule get_schedule() const;
  inline isl::schedule_node shared_ancestor(const isl::schedule_node &node2) const;
  inline isl::schedule_node get_shared_ancestor(const isl::schedule_node &node2) const;
  inline unsigned tree_depth() const;
  inline unsigned get_tree_depth() const;
  inline isl::schedule_node graft_after(isl::schedule_node graft) const;
  inline isl::schedule_node graft_before(isl::schedule_node graft) const;
  inline bool has_children() const;
  inline bool has_next_sibling() const;
  inline bool has_parent() const;
  inline bool has_previous_sibling() const;
  inline isl::schedule_node insert_context(isl::set context) const;
  inline isl::schedule_node insert_filter(isl::union_set filter) const;
  inline isl::schedule_node insert_guard(isl::set context) const;
  inline isl::schedule_node insert_mark(isl::id mark) const;
  inline isl::schedule_node insert_mark(const std::string &mark) const;
  inline isl::schedule_node insert_partial_schedule(isl::multi_union_pw_aff schedule) const;
  inline isl::schedule_node insert_sequence(isl::union_set_list filters) const;
  inline isl::schedule_node insert_set(isl::union_set_list filters) const;
  inline bool is_equal(const isl::schedule_node &node2) const;
  inline bool is_subtree_anchored() const;
  inline isl::schedule_node map_descendant_bottom_up(const std::function<isl::schedule_node(isl::schedule_node)> &fn) const;
  inline unsigned n_children() const;
  inline isl::schedule_node next_sibling() const;
  inline isl::schedule_node order_after(isl::union_set filter) const;
  inline isl::schedule_node order_before(isl::union_set filter) const;
  inline isl::schedule_node parent() const;
  inline isl::schedule_node previous_sibling() const;
  inline isl::schedule_node root() const;
};

// declarations for isl::schedule_node_band

class schedule_node_band : public schedule_node {
  template <class T>
  friend bool schedule_node::isa() const;
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
  inline isl::multi_union_pw_aff partial_schedule() const;
  inline isl::multi_union_pw_aff get_partial_schedule() const;
  inline bool permutable() const;
  inline bool get_permutable() const;
  inline bool member_get_coincident(int pos) const;
  inline schedule_node_band member_set_coincident(int pos, int coincident) const;
  inline schedule_node_band mod(isl::multi_val mv) const;
  inline unsigned n_member() const;
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
  friend bool schedule_node::isa() const;
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
  friend bool schedule_node::isa() const;
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
  friend bool schedule_node::isa() const;
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
  friend bool schedule_node::isa() const;
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
  friend bool schedule_node::isa() const;
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
  friend bool schedule_node::isa() const;
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
  friend bool schedule_node::isa() const;
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
  friend bool schedule_node::isa() const;
  friend schedule_node_mark schedule_node::as<schedule_node_mark>() const;
  static const auto type = isl_schedule_node_mark;

protected:
  inline explicit schedule_node_mark(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_mark();
  inline /* implicit */ schedule_node_mark(const schedule_node_mark &obj);
  inline schedule_node_mark &operator=(schedule_node_mark obj);
  inline isl::ctx ctx() const;

};

// declarations for isl::schedule_node_sequence

class schedule_node_sequence : public schedule_node {
  template <class T>
  friend bool schedule_node::isa() const;
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
  friend bool schedule_node::isa() const;
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
  inline explicit set(isl::ctx ctx, const std::string &str);
  inline set &operator=(set obj);
  inline ~set();
  inline __isl_give isl_set *copy() const &;
  inline __isl_give isl_set *copy() && = delete;
  inline __isl_keep isl_set *get() const;
  inline __isl_give isl_set *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::basic_set affine_hull() const;
  inline isl::set apply(isl::map map) const;
  inline isl::set bind(isl::multi_id tuple) const;
  inline isl::set coalesce() const;
  inline isl::set complement() const;
  inline isl::set detect_equalities() const;
  inline isl::val dim_max_val(int pos) const;
  inline isl::val dim_min_val(int pos) const;
  static inline isl::set empty(isl::space space);
  inline isl::set flatten() const;
  inline void foreach_basic_set(const std::function<void(isl::basic_set)> &fn) const;
  inline void foreach_point(const std::function<void(isl::point)> &fn) const;
  inline isl::multi_val plain_multi_val_if_fixed() const;
  inline isl::multi_val get_plain_multi_val_if_fixed() const;
  inline isl::fixed_box simple_fixed_box_hull() const;
  inline isl::fixed_box get_simple_fixed_box_hull() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::val stride(int pos) const;
  inline isl::val get_stride(int pos) const;
  inline isl::set gist(isl::set context) const;
  inline isl::map identity() const;
  inline isl::pw_aff indicator_function() const;
  inline isl::map insert_domain(isl::space domain) const;
  inline isl::set intersect(isl::set set2) const;
  inline isl::set intersect_params(isl::set params) const;
  inline bool involves_locals() const;
  inline bool is_disjoint(const isl::set &set2) const;
  inline bool is_empty() const;
  inline bool is_equal(const isl::set &set2) const;
  inline bool is_singleton() const;
  inline bool is_strict_subset(const isl::set &set2) const;
  inline bool is_subset(const isl::set &set2) const;
  inline bool is_wrapping() const;
  inline isl::set lexmax() const;
  inline isl::pw_multi_aff lexmax_pw_multi_aff() const;
  inline isl::set lexmin() const;
  inline isl::pw_multi_aff lexmin_pw_multi_aff() const;
  inline isl::set lower_bound(isl::multi_pw_aff lower) const;
  inline isl::set lower_bound(isl::multi_val lower) const;
  inline isl::multi_pw_aff max_multi_pw_aff() const;
  inline isl::val max_val(const isl::aff &obj) const;
  inline isl::multi_pw_aff min_multi_pw_aff() const;
  inline isl::val min_val(const isl::aff &obj) const;
  inline isl::set params() const;
  inline isl::basic_set polyhedral_hull() const;
  inline isl::set preimage(isl::multi_aff ma) const;
  inline isl::set preimage(isl::multi_pw_aff mpa) const;
  inline isl::set preimage(isl::pw_multi_aff pma) const;
  inline isl::set product(isl::set set2) const;
  inline isl::set project_out_all_params() const;
  inline isl::set project_out_param(isl::id id) const;
  inline isl::set project_out_param(const std::string &id) const;
  inline isl::set project_out_param(isl::id_list list) const;
  inline isl::basic_set sample() const;
  inline isl::point sample_point() const;
  inline isl::set subtract(isl::set set2) const;
  inline isl::map translation() const;
  inline isl::set unbind_params(isl::multi_id tuple) const;
  inline isl::map unbind_params_insert_domain(isl::multi_id domain) const;
  inline isl::set unite(isl::set set2) const;
  static inline isl::set universe(isl::space space);
  inline isl::basic_set unshifted_simple_hull() const;
  inline isl::map unwrap() const;
  inline isl::set upper_bound(isl::multi_pw_aff upper) const;
  inline isl::set upper_bound(isl::multi_val upper) const;
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
  inline space &operator=(space obj);
  inline ~space();
  inline __isl_give isl_space *copy() const &;
  inline __isl_give isl_space *copy() && = delete;
  inline __isl_keep isl_space *get() const;
  inline __isl_give isl_space *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::space add_named_tuple(isl::id tuple_id, unsigned int dim) const;
  inline isl::space add_named_tuple(const std::string &tuple_id, unsigned int dim) const;
  inline isl::space add_unnamed_tuple(unsigned int dim) const;
  inline isl::space curry() const;
  inline isl::space domain() const;
  inline isl::space flatten_domain() const;
  inline isl::space flatten_range() const;
  inline bool is_equal(const isl::space &space2) const;
  inline bool is_wrapping() const;
  inline isl::space map_from_set() const;
  inline isl::space params() const;
  inline isl::space product(isl::space right) const;
  inline isl::space range() const;
  inline isl::space range_reverse() const;
  inline isl::space reverse() const;
  inline isl::space uncurry() const;
  static inline isl::space unit(isl::ctx ctx);
  inline isl::space unwrap() const;
  inline isl::space wrap() const;
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
  inline bool every_map(const std::function<bool(isl::map)> &test) const;
  inline isl::map extract_map(isl::space space) const;
  inline isl::union_map factor_domain() const;
  inline isl::union_map factor_range() const;
  inline isl::union_map fixed_power(isl::val exp) const;
  inline isl::union_map fixed_power(long exp) const;
  inline void foreach_map(const std::function<void(isl::map)> &fn) const;
  static inline isl::union_map from(isl::multi_union_pw_aff mupa);
  static inline isl::union_map from(isl::union_pw_multi_aff upma);
  static inline isl::union_map from_domain(isl::union_set uset);
  static inline isl::union_map from_domain_and_range(isl::union_set domain, isl::union_set range);
  static inline isl::union_map from_range(isl::union_set uset);
  inline isl::space space() const;
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
  inline bool is_bijective() const;
  inline bool is_disjoint(const isl::union_map &umap2) const;
  inline bool is_empty() const;
  inline bool is_equal(const isl::union_map &umap2) const;
  inline bool is_injective() const;
  inline bool is_single_valued() const;
  inline bool is_strict_subset(const isl::union_map &umap2) const;
  inline bool is_subset(const isl::union_map &umap2) const;
  inline bool isa_map() const;
  inline isl::union_map lexmax() const;
  inline isl::union_map lexmin() const;
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
  inline union_pw_aff &operator=(union_pw_aff obj);
  inline ~union_pw_aff();
  inline __isl_give isl_union_pw_aff *copy() const &;
  inline __isl_give isl_union_pw_aff *copy() && = delete;
  inline __isl_keep isl_union_pw_aff *get() const;
  inline __isl_give isl_union_pw_aff *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::union_pw_aff add(isl::union_pw_aff upa2) const;
  inline isl::union_set bind(isl::id id) const;
  inline isl::union_set bind(const std::string &id) const;
  inline isl::union_pw_aff coalesce() const;
  inline isl::union_set domain() const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::union_pw_aff gist(isl::union_set context) const;
  inline isl::union_pw_aff intersect_domain(isl::space space) const;
  inline isl::union_pw_aff intersect_domain(isl::union_set uset) const;
  inline isl::union_pw_aff intersect_domain_wrapped_domain(isl::union_set uset) const;
  inline isl::union_pw_aff intersect_domain_wrapped_range(isl::union_set uset) const;
  inline isl::union_pw_aff intersect_params(isl::set set) const;
  inline isl::union_pw_aff pullback(isl::union_pw_multi_aff upma) const;
  inline isl::union_pw_aff sub(isl::union_pw_aff upa2) const;
  inline isl::union_pw_aff subtract_domain(isl::space space) const;
  inline isl::union_pw_aff subtract_domain(isl::union_set uset) const;
  inline isl::union_pw_aff union_add(isl::union_pw_aff upa2) const;
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
  inline union_pw_aff_list &operator=(union_pw_aff_list obj);
  inline ~union_pw_aff_list();
  inline __isl_give isl_union_pw_aff_list *copy() const &;
  inline __isl_give isl_union_pw_aff_list *copy() && = delete;
  inline __isl_keep isl_union_pw_aff_list *get() const;
  inline __isl_give isl_union_pw_aff_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::union_pw_aff_list add(isl::union_pw_aff el) const;
  inline isl::union_pw_aff_list clear() const;
  inline isl::union_pw_aff_list concat(isl::union_pw_aff_list list2) const;
  inline isl::union_pw_aff_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(isl::union_pw_aff)> &fn) const;
  inline isl::union_pw_aff at(int index) const;
  inline isl::union_pw_aff get_at(int index) const;
  inline isl::union_pw_aff_list insert(unsigned int pos, isl::union_pw_aff el) const;
  inline unsigned size() const;
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
  inline /* implicit */ union_pw_multi_aff(isl::multi_aff ma);
  inline /* implicit */ union_pw_multi_aff(isl::pw_multi_aff pma);
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
  inline isl::union_pw_multi_aff apply(isl::union_pw_multi_aff upma2) const;
  inline isl::pw_multi_aff as_pw_multi_aff() const;
  inline isl::union_pw_multi_aff coalesce() const;
  inline isl::union_set domain() const;
  static inline isl::union_pw_multi_aff empty(isl::ctx ctx);
  inline isl::pw_multi_aff extract_pw_multi_aff(isl::space space) const;
  inline isl::union_pw_multi_aff flat_range_product(isl::union_pw_multi_aff upma2) const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::union_pw_multi_aff gist(isl::union_set context) const;
  inline isl::union_pw_multi_aff intersect_domain(isl::space space) const;
  inline isl::union_pw_multi_aff intersect_domain(isl::union_set uset) const;
  inline isl::union_pw_multi_aff intersect_domain_wrapped_domain(isl::union_set uset) const;
  inline isl::union_pw_multi_aff intersect_domain_wrapped_range(isl::union_set uset) const;
  inline isl::union_pw_multi_aff intersect_params(isl::set set) const;
  inline bool involves_locals() const;
  inline bool isa_pw_multi_aff() const;
  inline bool plain_is_empty() const;
  inline isl::union_pw_multi_aff preimage_domain_wrapped_domain(isl::union_pw_multi_aff upma2) const;
  inline isl::union_pw_multi_aff pullback(isl::union_pw_multi_aff upma2) const;
  inline isl::union_pw_multi_aff range_factor_domain() const;
  inline isl::union_pw_multi_aff range_factor_range() const;
  inline isl::union_pw_multi_aff range_product(isl::union_pw_multi_aff upma2) const;
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
  inline isl::union_set coalesce() const;
  inline isl::union_set compute_divs() const;
  inline isl::union_set detect_equalities() const;
  static inline isl::union_set empty(isl::ctx ctx);
  inline bool every_set(const std::function<bool(isl::set)> &test) const;
  inline isl::set extract_set(isl::space space) const;
  inline void foreach_point(const std::function<void(isl::point)> &fn) const;
  inline void foreach_set(const std::function<void(isl::set)> &fn) const;
  inline isl::space space() const;
  inline isl::space get_space() const;
  inline isl::union_set gist(isl::union_set context) const;
  inline isl::union_set gist_params(isl::set set) const;
  inline isl::union_map identity() const;
  inline isl::union_set intersect(isl::union_set uset2) const;
  inline isl::union_set intersect_params(isl::set set) const;
  inline bool is_disjoint(const isl::union_set &uset2) const;
  inline bool is_empty() const;
  inline bool is_equal(const isl::union_set &uset2) const;
  inline bool is_strict_subset(const isl::union_set &uset2) const;
  inline bool is_subset(const isl::union_set &uset2) const;
  inline bool isa_set() const;
  inline isl::union_set lexmax() const;
  inline isl::union_set lexmin() const;
  inline isl::union_set polyhedral_hull() const;
  inline isl::union_set preimage(isl::multi_aff ma) const;
  inline isl::union_set preimage(isl::pw_multi_aff pma) const;
  inline isl::union_set preimage(isl::union_pw_multi_aff upma) const;
  inline isl::point sample_point() const;
  inline isl::union_set subtract(isl::union_set uset2) const;
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
  inline union_set_list &operator=(union_set_list obj);
  inline ~union_set_list();
  inline __isl_give isl_union_set_list *copy() const &;
  inline __isl_give isl_union_set_list *copy() && = delete;
  inline __isl_keep isl_union_set_list *get() const;
  inline __isl_give isl_union_set_list *release();
  inline bool is_null() const;
  inline isl::ctx ctx() const;

  inline isl::union_set_list add(isl::union_set el) const;
  inline isl::union_set_list clear() const;
  inline isl::union_set_list concat(isl::union_set_list list2) const;
  inline isl::union_set_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(isl::union_set)> &fn) const;
  inline isl::union_set at(int index) const;
  inline isl::union_set get_at(int index) const;
  inline isl::union_set_list insert(unsigned int pos, isl::union_set el) const;
  inline unsigned size() const;
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
  inline bool abs_eq(const isl::val &v2) const;
  inline bool abs_eq(long v2) const;
  inline isl::val add(isl::val v2) const;
  inline isl::val add(long v2) const;
  inline isl::val ceil() const;
  inline int cmp_si(long i) const;
  inline isl::val div(isl::val v2) const;
  inline isl::val div(long v2) const;
  inline bool eq(const isl::val &v2) const;
  inline bool eq(long v2) const;
  inline isl::val floor() const;
  inline isl::val gcd(isl::val v2) const;
  inline isl::val gcd(long v2) const;
  inline bool ge(const isl::val &v2) const;
  inline bool ge(long v2) const;
  inline long den_si() const;
  inline long get_den_si() const;
  inline long num_si() const;
  inline long get_num_si() const;
  inline bool gt(const isl::val &v2) const;
  inline bool gt(long v2) const;
  static inline isl::val infty(isl::ctx ctx);
  inline isl::val inv() const;
  inline bool is_divisible_by(const isl::val &v2) const;
  inline bool is_divisible_by(long v2) const;
  inline bool is_infty() const;
  inline bool is_int() const;
  inline bool is_nan() const;
  inline bool is_neg() const;
  inline bool is_neginfty() const;
  inline bool is_negone() const;
  inline bool is_nonneg() const;
  inline bool is_nonpos() const;
  inline bool is_one() const;
  inline bool is_pos() const;
  inline bool is_rat() const;
  inline bool is_zero() const;
  inline bool le(const isl::val &v2) const;
  inline bool le(long v2) const;
  inline bool lt(const isl::val &v2) const;
  inline bool lt(long v2) const;
  inline isl::val max(isl::val v2) const;
  inline isl::val max(long v2) const;
  inline isl::val min(isl::val v2) const;
  inline isl::val min(long v2) const;
  inline isl::val mod(isl::val v2) const;
  inline isl::val mod(long v2) const;
  inline isl::val mul(isl::val v2) const;
  inline isl::val mul(long v2) const;
  static inline isl::val nan(isl::ctx ctx);
  inline bool ne(const isl::val &v2) const;
  inline bool ne(long v2) const;
  inline isl::val neg() const;
  static inline isl::val neginfty(isl::ctx ctx);
  static inline isl::val negone(isl::ctx ctx);
  static inline isl::val one(isl::ctx ctx);
  inline isl::val pow2() const;
  inline int sgn() const;
  inline isl::val sub(isl::val v2) const;
  inline isl::val sub(long v2) const;
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
  inline isl::val_list clear() const;
  inline isl::val_list concat(isl::val_list list2) const;
  inline isl::val_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(isl::val)> &fn) const;
  inline isl::val at(int index) const;
  inline isl::val get_at(int index) const;
  inline isl::val_list insert(unsigned int pos, isl::val el) const;
  inline isl::val_list insert(unsigned int pos, long el) const;
  inline unsigned size() const;
};

// implementations for isl::aff
aff manage(__isl_take isl_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return aff(ptr);
}
aff manage_copy(__isl_keep isl_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_aff_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_aff_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return aff(ptr);
}

aff::aff()
    : ptr(nullptr) {}

aff::aff(const aff &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_aff_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

aff::aff(__isl_take isl_aff *ptr)
    : ptr(ptr) {}

aff::aff(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_add(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff aff::add_constant(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_add_constant_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff aff::add_constant(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->add_constant(isl::val(ctx(), v));
}

isl::basic_set aff::bind(isl::id id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_bind_id(copy(), id.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_set aff::bind(const std::string &id) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->bind(isl::id(ctx(), id));
}

isl::aff aff::ceil() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_ceil(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff aff::div(isl::aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_div(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set aff::eq_set(isl::aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_eq_set(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val aff::eval(isl::point pnt) const
{
  if (!ptr || pnt.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_eval(copy(), pnt.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff aff::floor() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_floor(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set aff::ge_set(isl::aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_ge_set(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val aff::constant_val() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_get_constant_val(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val aff::get_constant_val() const
{
  return constant_val();
}

isl::aff aff::gist(isl::set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set aff::gt_set(isl::aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_gt_set(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool aff::is_cst() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_is_cst(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::set aff::le_set(isl::aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_le_set(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set aff::lt_set(isl::aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_lt_set(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff aff::mod(isl::val mod) const
{
  if (!ptr || mod.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_mod_val(copy(), mod.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff aff::mod(long mod) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->mod(isl::val(ctx(), mod));
}

isl::aff aff::mul(isl::aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_mul(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set aff::ne_set(isl::aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_ne_set(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff aff::neg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_neg(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff aff::pullback(isl::multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_pullback_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff aff::scale(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_scale_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff aff::scale(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->scale(isl::val(ctx(), v));
}

isl::aff aff::scale_down(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_scale_down_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff aff::scale_down(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->scale_down(isl::val(ctx(), v));
}

isl::aff aff::sub(isl::aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_sub(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff aff::unbind_params_insert_domain(isl::multi_id domain) const
{
  if (!ptr || domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_unbind_params_insert_domain(copy(), domain.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff aff::zero_on_domain(isl::space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_zero_on_domain_space(space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const aff &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_aff_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_aff_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::aff_list
aff_list manage(__isl_take isl_aff_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return aff_list(ptr);
}
aff_list manage_copy(__isl_keep isl_aff_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_aff_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_aff_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return aff_list(ptr);
}

aff_list::aff_list()
    : ptr(nullptr) {}

aff_list::aff_list(const aff_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_aff_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

aff_list::aff_list(__isl_take isl_aff_list *ptr)
    : ptr(ptr) {}

aff_list::aff_list(isl::ctx ctx, int n)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

aff_list::aff_list(isl::aff el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = el.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_list_from_aff(el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff_list aff_list::clear() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_list_clear(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff_list aff_list::concat(isl::aff_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff_list aff_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

void aff_list::foreach(const std::function<void(isl::aff)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct fn_data {
    std::function<void(isl::aff)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_aff_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return;
}

isl::aff aff_list::at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff aff_list::get_at(int index) const
{
  return at(index);
}

isl::aff_list aff_list::insert(unsigned int pos, isl::aff el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_list_insert(copy(), pos, el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

unsigned aff_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_aff_list_size(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

inline std::ostream &operator<<(std::ostream &os, const aff_list &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_aff_list_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_aff_list_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_build
ast_build manage(__isl_take isl_ast_build *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return ast_build(ptr);
}
ast_build manage_copy(__isl_keep isl_ast_build *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_build_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_ast_build_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return ast_build(ptr);
}

ast_build::ast_build()
    : ptr(nullptr) {}

ast_build::ast_build(const ast_build &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_build_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  copy_callbacks(obj);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

ast_build::ast_build(__isl_take isl_ast_build *ptr)
    : ptr(ptr) {}

ast_build::ast_build(isl::ctx ctx)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_build_alloc(ctx.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
    exception::throw_invalid("cannot release object with persistent callbacks", __FILE__, __LINE__);
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
  ISL_CPP_TRY {
    auto ret = (data->func)(manage(arg_0), manage_copy(arg_1));
    return ret.release();
  } ISL_CPP_CATCH_ALL {
    data->eptr = std::current_exception();
    return NULL;
  }
}

void ast_build::set_at_each_domain_data(const std::function<isl::ast_node(isl::ast_node, isl::ast_build)> &fn)
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_build_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  at_each_domain_data = std::make_shared<struct at_each_domain_data>();
  at_each_domain_data->func = fn;
  ptr = isl_ast_build_set_at_each_domain(ptr, &at_each_domain, at_each_domain_data.get());
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

isl::ast_build ast_build::set_at_each_domain(const std::function<isl::ast_node(isl::ast_node, isl::ast_build)> &fn) const
{
  auto copy = *this;
  copy.set_at_each_domain_data(fn);
  return copy;
}

isl::ast_expr ast_build::access_from(isl::multi_pw_aff mpa) const
{
  if (!ptr || mpa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_build_access_from_multi_pw_aff(get(), mpa.release());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_expr ast_build::access_from(isl::pw_multi_aff pma) const
{
  if (!ptr || pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_build_access_from_pw_multi_aff(get(), pma.release());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_expr ast_build::call_from(isl::multi_pw_aff mpa) const
{
  if (!ptr || mpa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_build_call_from_multi_pw_aff(get(), mpa.release());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_expr ast_build::call_from(isl::pw_multi_aff pma) const
{
  if (!ptr || pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_build_call_from_pw_multi_aff(get(), pma.release());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_expr ast_build::expr_from(isl::pw_aff pa) const
{
  if (!ptr || pa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_build_expr_from_pw_aff(get(), pa.release());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_expr ast_build::expr_from(isl::set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_build_expr_from_set(get(), set.release());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_build ast_build::from_context(isl::set set)
{
  if (set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = set.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_build_from_context(set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map ast_build::schedule() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_build_get_schedule(get());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map ast_build::get_schedule() const
{
  return schedule();
}

isl::ast_node ast_build::node_from(isl::schedule schedule) const
{
  if (!ptr || schedule.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_build_node_from_schedule(get(), schedule.release());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_node ast_build::node_from_schedule_map(isl::union_map schedule) const
{
  if (!ptr || schedule.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_build_node_from_schedule_map(get(), schedule.release());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

// implementations for isl::ast_expr
ast_expr manage(__isl_take isl_ast_expr *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return ast_expr(ptr);
}
ast_expr manage_copy(__isl_keep isl_ast_expr *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_ast_expr_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return ast_expr(ptr);
}

ast_expr::ast_expr()
    : ptr(nullptr) {}

ast_expr::ast_expr(const ast_expr &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
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
bool ast_expr::isa_type(T subtype) const
{
  if (is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return isl_ast_expr_get_type(get()) == subtype;
}
template <class T>
bool ast_expr::isa() const
{
  return isa_type<decltype(T::type)>(T::type);
}
template <class T>
T ast_expr::as() const
{
 if (!isa<T>())
    exception::throw_invalid("not an object of the requested subtype", __FILE__, __LINE__);
  return T(copy());
}

isl::ctx ast_expr::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

std::string ast_expr::to_C_str() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_expr_to_C_str(get());
  std::string tmp(res);
  free(res);
  return tmp;
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_expr_id_get_id(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::id ast_expr_id::get_id() const
{
  return id();
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_id &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_expr_int_get_val(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val ast_expr_int::get_val() const
{
  return val();
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_int &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
bool ast_expr_op::isa_type(T subtype) const
{
  if (is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return isl_ast_expr_op_get_type(get()) == subtype;
}
template <class T>
bool ast_expr_op::isa() const
{
  return isa_type<decltype(T::type)>(T::type);
}
template <class T>
T ast_expr_op::as() const
{
 if (!isa<T>())
    exception::throw_invalid("not an object of the requested subtype", __FILE__, __LINE__);
  return T(copy());
}

isl::ctx ast_expr_op::ctx() const {
  return isl::ctx(isl_ast_expr_get_ctx(ptr));
}

isl::ast_expr ast_expr_op::arg(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_expr_op_get_arg(get(), pos);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_expr ast_expr_op::get_arg(int pos) const
{
  return arg(pos);
}

unsigned ast_expr_op::n_arg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_expr_op_get_n_arg(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

unsigned ast_expr_op::get_n_arg() const
{
  return n_arg();
}

inline std::ostream &operator<<(std::ostream &os, const ast_expr_op &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_expr_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_expr_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_node
ast_node manage(__isl_take isl_ast_node *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return ast_node(ptr);
}
ast_node manage_copy(__isl_keep isl_ast_node *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_node_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_ast_node_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return ast_node(ptr);
}

ast_node::ast_node()
    : ptr(nullptr) {}

ast_node::ast_node(const ast_node &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_node_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
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
bool ast_node::isa_type(T subtype) const
{
  if (is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return isl_ast_node_get_type(get()) == subtype;
}
template <class T>
bool ast_node::isa() const
{
  return isa_type<decltype(T::type)>(T::type);
}
template <class T>
T ast_node::as() const
{
 if (!isa<T>())
    exception::throw_invalid("not an object of the requested subtype", __FILE__, __LINE__);
  return T(copy());
}

isl::ctx ast_node::ctx() const {
  return isl::ctx(isl_ast_node_get_ctx(ptr));
}

std::string ast_node::to_C_str() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_to_C_str(get());
  std::string tmp(res);
  free(res);
  return tmp;
}

inline std::ostream &operator<<(std::ostream &os, const ast_node &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_node_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_node_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_block_get_children(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_node_list ast_node_block::get_children() const
{
  return children();
}

inline std::ostream &operator<<(std::ostream &os, const ast_node_block &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_node_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_node_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_for_get_body(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_node ast_node_for::get_body() const
{
  return body();
}

isl::ast_expr ast_node_for::cond() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_for_get_cond(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_expr ast_node_for::get_cond() const
{
  return cond();
}

isl::ast_expr ast_node_for::inc() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_for_get_inc(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_expr ast_node_for::get_inc() const
{
  return inc();
}

isl::ast_expr ast_node_for::init() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_for_get_init(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_expr ast_node_for::get_init() const
{
  return init();
}

isl::ast_expr ast_node_for::iterator() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_for_get_iterator(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_expr ast_node_for::get_iterator() const
{
  return iterator();
}

bool ast_node_for::is_degenerate() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_for_is_degenerate(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

inline std::ostream &operator<<(std::ostream &os, const ast_node_for &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_node_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_node_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_if_get_cond(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_expr ast_node_if::get_cond() const
{
  return cond();
}

isl::ast_node ast_node_if::else_node() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_if_get_else_node(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_node ast_node_if::get_else_node() const
{
  return else_node();
}

isl::ast_node ast_node_if::then_node() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_if_get_then_node(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_node ast_node_if::get_then_node() const
{
  return then_node();
}

bool ast_node_if::has_else_node() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_if_has_else_node(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

inline std::ostream &operator<<(std::ostream &os, const ast_node_if &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_node_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_node_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::ast_node_list
ast_node_list manage(__isl_take isl_ast_node_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return ast_node_list(ptr);
}
ast_node_list manage_copy(__isl_keep isl_ast_node_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_node_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_ast_node_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return ast_node_list(ptr);
}

ast_node_list::ast_node_list()
    : ptr(nullptr) {}

ast_node_list::ast_node_list(const ast_node_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_node_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

ast_node_list::ast_node_list(__isl_take isl_ast_node_list *ptr)
    : ptr(ptr) {}

ast_node_list::ast_node_list(isl::ctx ctx, int n)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

ast_node_list::ast_node_list(isl::ast_node el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = el.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_list_from_ast_node(el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_node_list ast_node_list::clear() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_list_clear(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_node_list ast_node_list::concat(isl::ast_node_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_node_list ast_node_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

void ast_node_list::foreach(const std::function<void(isl::ast_node)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct fn_data {
    std::function<void(isl::ast_node)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_ast_node *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_ast_node_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return;
}

isl::ast_node ast_node_list::at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_node ast_node_list::get_at(int index) const
{
  return at(index);
}

isl::ast_node_list ast_node_list::insert(unsigned int pos, isl::ast_node el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_list_insert(copy(), pos, el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

unsigned ast_node_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_list_size(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

inline std::ostream &operator<<(std::ostream &os, const ast_node_list &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_node_list_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_node_list_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_mark_get_id(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::id ast_node_mark::get_id() const
{
  return id();
}

isl::ast_node ast_node_mark::node() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_mark_get_node(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_node ast_node_mark::get_node() const
{
  return node();
}

inline std::ostream &operator<<(std::ostream &os, const ast_node_mark &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_node_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_node_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_ast_node_user_get_expr(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::ast_expr ast_node_user::get_expr() const
{
  return expr();
}

inline std::ostream &operator<<(std::ostream &os, const ast_node_user &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_ast_node_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_ast_node_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::basic_map
basic_map manage(__isl_take isl_basic_map *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return basic_map(ptr);
}
basic_map manage_copy(__isl_keep isl_basic_map *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_basic_map_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_basic_map_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return basic_map(ptr);
}

basic_map::basic_map()
    : ptr(nullptr) {}

basic_map::basic_map(const basic_map &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_basic_map_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

basic_map::basic_map(__isl_take isl_basic_map *ptr)
    : ptr(ptr) {}

basic_map::basic_map(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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

isl::basic_map basic_map::affine_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_affine_hull(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_map basic_map::apply_domain(isl::basic_map bmap2) const
{
  if (!ptr || bmap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_apply_domain(copy(), bmap2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_map basic_map::apply_range(isl::basic_map bmap2) const
{
  if (!ptr || bmap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_apply_range(copy(), bmap2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_set basic_map::deltas() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_deltas(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_map basic_map::detect_equalities() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_detect_equalities(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_map basic_map::flatten() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_flatten(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_map basic_map::flatten_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_flatten_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_map basic_map::flatten_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_flatten_range(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_map basic_map::gist(isl::basic_map context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_map basic_map::intersect(isl::basic_map bmap2) const
{
  if (!ptr || bmap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_intersect(copy(), bmap2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_map basic_map::intersect_domain(isl::basic_set bset) const
{
  if (!ptr || bset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_intersect_domain(copy(), bset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_map basic_map::intersect_range(isl::basic_set bset) const
{
  if (!ptr || bset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_intersect_range(copy(), bset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool basic_map::is_empty() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_is_empty(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool basic_map::is_equal(const isl::basic_map &bmap2) const
{
  if (!ptr || bmap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_is_equal(get(), bmap2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool basic_map::is_subset(const isl::basic_map &bmap2) const
{
  if (!ptr || bmap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_is_subset(get(), bmap2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::map basic_map::lexmax() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_lexmax(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map basic_map::lexmin() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_lexmin(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_map basic_map::reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_reverse(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_map basic_map::sample() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_sample(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map basic_map::unite(isl::basic_map bmap2) const
{
  if (!ptr || bmap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_map_union(copy(), bmap2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const basic_map &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_basic_map_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_basic_map_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::basic_set
basic_set manage(__isl_take isl_basic_set *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return basic_set(ptr);
}
basic_set manage_copy(__isl_keep isl_basic_set *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_basic_set_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_basic_set_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return basic_set(ptr);
}

basic_set::basic_set()
    : ptr(nullptr) {}

basic_set::basic_set(const basic_set &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_basic_set_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

basic_set::basic_set(__isl_take isl_basic_set *ptr)
    : ptr(ptr) {}

basic_set::basic_set(isl::point pnt)
{
  if (pnt.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = pnt.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_from_point(pnt.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

basic_set::basic_set(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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

isl::basic_set basic_set::affine_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_affine_hull(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_set basic_set::apply(isl::basic_map bmap) const
{
  if (!ptr || bmap.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_apply(copy(), bmap.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_set basic_set::detect_equalities() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_detect_equalities(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val basic_set::dim_max_val(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_dim_max_val(copy(), pos);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_set basic_set::flatten() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_flatten(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_set basic_set::gist(isl::basic_set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_set basic_set::intersect(isl::basic_set bset2) const
{
  if (!ptr || bset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_intersect(copy(), bset2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_set basic_set::intersect_params(isl::basic_set bset2) const
{
  if (!ptr || bset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_intersect_params(copy(), bset2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool basic_set::is_empty() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_is_empty(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool basic_set::is_equal(const isl::basic_set &bset2) const
{
  if (!ptr || bset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_is_equal(get(), bset2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool basic_set::is_subset(const isl::basic_set &bset2) const
{
  if (!ptr || bset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_is_subset(get(), bset2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool basic_set::is_wrapping() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_is_wrapping(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::set basic_set::lexmax() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_lexmax(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set basic_set::lexmin() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_lexmin(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_set basic_set::params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_params(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_set basic_set::sample() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_sample(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::point basic_set::sample_point() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_sample_point(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set basic_set::unite(isl::basic_set bset2) const
{
  if (!ptr || bset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_basic_set_union(copy(), bset2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const basic_set &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_basic_set_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_basic_set_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::fixed_box
fixed_box manage(__isl_take isl_fixed_box *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return fixed_box(ptr);
}
fixed_box manage_copy(__isl_keep isl_fixed_box *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_fixed_box_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_fixed_box_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return fixed_box(ptr);
}

fixed_box::fixed_box()
    : ptr(nullptr) {}

fixed_box::fixed_box(const fixed_box &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_fixed_box_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
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

isl::multi_aff fixed_box::offset() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_fixed_box_get_offset(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff fixed_box::get_offset() const
{
  return offset();
}

isl::multi_val fixed_box::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_fixed_box_get_size(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val fixed_box::get_size() const
{
  return size();
}

isl::space fixed_box::space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_fixed_box_get_space(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space fixed_box::get_space() const
{
  return space();
}

bool fixed_box::is_valid() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_fixed_box_is_valid(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

inline std::ostream &operator<<(std::ostream &os, const fixed_box &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_fixed_box_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_fixed_box_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::id
id manage(__isl_take isl_id *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return id(ptr);
}
id manage_copy(__isl_keep isl_id *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_id_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_id_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return id(ptr);
}

id::id()
    : ptr(nullptr) {}

id::id(const id &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_id_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

id::id(__isl_take isl_id *ptr)
    : ptr(ptr) {}

id::id(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_id_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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

std::string id::name() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_id_get_name(get());
  std::string tmp(res);
  return tmp;
}

std::string id::get_name() const
{
  return name();
}

inline std::ostream &operator<<(std::ostream &os, const id &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_id_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_id_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::id_list
id_list manage(__isl_take isl_id_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return id_list(ptr);
}
id_list manage_copy(__isl_keep isl_id_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_id_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_id_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return id_list(ptr);
}

id_list::id_list()
    : ptr(nullptr) {}

id_list::id_list(const id_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_id_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

id_list::id_list(__isl_take isl_id_list *ptr)
    : ptr(ptr) {}

id_list::id_list(isl::ctx ctx, int n)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_id_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

id_list::id_list(isl::id el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = el.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_id_list_from_id(el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_id_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::id_list id_list::add(const std::string &el) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->add(isl::id(ctx(), el));
}

isl::id_list id_list::clear() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_id_list_clear(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::id_list id_list::concat(isl::id_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_id_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::id_list id_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_id_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

void id_list::foreach(const std::function<void(isl::id)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct fn_data {
    std::function<void(isl::id)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_id *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_id_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return;
}

isl::id id_list::at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_id_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::id id_list::get_at(int index) const
{
  return at(index);
}

isl::id_list id_list::insert(unsigned int pos, isl::id el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_id_list_insert(copy(), pos, el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::id_list id_list::insert(unsigned int pos, const std::string &el) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->insert(pos, isl::id(ctx(), el));
}

unsigned id_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_id_list_size(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

inline std::ostream &operator<<(std::ostream &os, const id_list &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_id_list_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_id_list_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::map
map manage(__isl_take isl_map *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return map(ptr);
}
map manage_copy(__isl_keep isl_map *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_map_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_map_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return map(ptr);
}

map::map()
    : ptr(nullptr) {}

map::map(const map &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_map_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

map::map(__isl_take isl_map *ptr)
    : ptr(ptr) {}

map::map(isl::basic_map bmap)
{
  if (bmap.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = bmap.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_from_basic_map(bmap.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

map::map(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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

isl::basic_map map::affine_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_affine_hull(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::apply_domain(isl::map map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_apply_domain(copy(), map2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::apply_range(isl::map map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_apply_range(copy(), map2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set map::bind_domain(isl::multi_id tuple) const
{
  if (!ptr || tuple.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_bind_domain(copy(), tuple.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set map::bind_range(isl::multi_id tuple) const
{
  if (!ptr || tuple.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_bind_range(copy(), tuple.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::coalesce() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_coalesce(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::complement() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_complement(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::curry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_curry(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set map::deltas() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_deltas(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::detect_equalities() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_detect_equalities(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set map::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::domain_factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_domain_factor_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::domain_factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_domain_factor_range(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::domain_product(isl::map map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_domain_product(copy(), map2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::empty(isl::space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_empty(space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::eq_at(isl::multi_pw_aff mpa) const
{
  if (!ptr || mpa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_eq_at_multi_pw_aff(copy(), mpa.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_factor_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_factor_range(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::flatten() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_flatten(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::flatten_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_flatten_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::flatten_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_flatten_range(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

void map::foreach_basic_map(const std::function<void(isl::basic_map)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct fn_data {
    std::function<void(isl::basic_map)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_basic_map *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_map_foreach_basic_map(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return;
}

isl::fixed_box map::range_simple_fixed_box_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_get_range_simple_fixed_box_hull(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::fixed_box map::get_range_simple_fixed_box_hull() const
{
  return range_simple_fixed_box_hull();
}

isl::space map::space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_get_space(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space map::get_space() const
{
  return space();
}

isl::map map::gist(isl::map context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::gist_domain(isl::set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_gist_domain(copy(), context.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::intersect(isl::map map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_intersect(copy(), map2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::intersect_domain(isl::set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_intersect_domain(copy(), set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::intersect_domain_factor_domain(isl::map factor) const
{
  if (!ptr || factor.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_intersect_domain_factor_domain(copy(), factor.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::intersect_domain_factor_range(isl::map factor) const
{
  if (!ptr || factor.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_intersect_domain_factor_range(copy(), factor.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::intersect_params(isl::set params) const
{
  if (!ptr || params.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_intersect_params(copy(), params.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::intersect_range(isl::set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_intersect_range(copy(), set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::intersect_range_factor_domain(isl::map factor) const
{
  if (!ptr || factor.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_intersect_range_factor_domain(copy(), factor.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::intersect_range_factor_range(isl::map factor) const
{
  if (!ptr || factor.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_intersect_range_factor_range(copy(), factor.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool map::is_bijective() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_is_bijective(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool map::is_disjoint(const isl::map &map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_is_disjoint(get(), map2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool map::is_empty() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_is_empty(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool map::is_equal(const isl::map &map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_is_equal(get(), map2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool map::is_injective() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_is_injective(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool map::is_single_valued() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_is_single_valued(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool map::is_strict_subset(const isl::map &map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_is_strict_subset(get(), map2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool map::is_subset(const isl::map &map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_is_subset(get(), map2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::map map::lex_ge_at(isl::multi_pw_aff mpa) const
{
  if (!ptr || mpa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_lex_ge_at_multi_pw_aff(copy(), mpa.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::lex_gt_at(isl::multi_pw_aff mpa) const
{
  if (!ptr || mpa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_lex_gt_at_multi_pw_aff(copy(), mpa.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::lex_le_at(isl::multi_pw_aff mpa) const
{
  if (!ptr || mpa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_lex_le_at_multi_pw_aff(copy(), mpa.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::lex_lt_at(isl::multi_pw_aff mpa) const
{
  if (!ptr || mpa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_lex_lt_at_multi_pw_aff(copy(), mpa.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::lexmax() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_lexmax(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff map::lexmax_pw_multi_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_lexmax_pw_multi_aff(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::lexmin() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_lexmin(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff map::lexmin_pw_multi_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_lexmin_pw_multi_aff(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::lower_bound(isl::multi_pw_aff lower) const
{
  if (!ptr || lower.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_lower_bound_multi_pw_aff(copy(), lower.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff map::max_multi_pw_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_max_multi_pw_aff(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff map::min_multi_pw_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_min_multi_pw_aff(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_map map::polyhedral_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_polyhedral_hull(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::preimage_domain(isl::multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_preimage_domain_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::preimage_domain(isl::multi_pw_aff mpa) const
{
  if (!ptr || mpa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_preimage_domain_multi_pw_aff(copy(), mpa.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::preimage_domain(isl::pw_multi_aff pma) const
{
  if (!ptr || pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_preimage_domain_pw_multi_aff(copy(), pma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::preimage_range(isl::multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_preimage_range_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::preimage_range(isl::pw_multi_aff pma) const
{
  if (!ptr || pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_preimage_range_pw_multi_aff(copy(), pma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::product(isl::map map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_product(copy(), map2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::project_out_all_params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_project_out_all_params(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set map::range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_range(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::range_factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_range_factor_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::range_factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_range_factor_range(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::range_product(isl::map map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_range_product(copy(), map2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::range_reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_range_reverse(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_reverse(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_map map::sample() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_sample(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::subtract(isl::map map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_subtract(copy(), map2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::uncurry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_uncurry(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::unite(isl::map map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_union(copy(), map2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::universe(isl::space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_universe(space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_map map::unshifted_simple_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_unshifted_simple_hull(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::upper_bound(isl::multi_pw_aff upper) const
{
  if (!ptr || upper.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_upper_bound_multi_pw_aff(copy(), upper.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set map::wrap() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_wrap(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map map::zip() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_map_zip(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const map &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_map_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_map_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::multi_aff
multi_aff manage(__isl_take isl_multi_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return multi_aff(ptr);
}
multi_aff manage_copy(__isl_keep isl_multi_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_multi_aff_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_multi_aff_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return multi_aff(ptr);
}

multi_aff::multi_aff()
    : ptr(nullptr) {}

multi_aff::multi_aff(const multi_aff &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_multi_aff_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

multi_aff::multi_aff(__isl_take isl_multi_aff *ptr)
    : ptr(ptr) {}

multi_aff::multi_aff(isl::aff aff)
{
  if (aff.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = aff.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_from_aff(aff.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

multi_aff::multi_aff(isl::space space, isl::aff_list list)
{
  if (space.is_null() || list.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_from_aff_list(space.release(), list.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

multi_aff::multi_aff(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_add(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::add_constant(isl::multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_add_constant_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::add_constant(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_add_constant_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::add_constant(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->add_constant(isl::val(ctx(), v));
}

isl::basic_set multi_aff::bind(isl::multi_id tuple) const
{
  if (!ptr || tuple.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_bind(copy(), tuple.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::bind_domain(isl::multi_id tuple) const
{
  if (!ptr || tuple.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_bind_domain(copy(), tuple.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::bind_domain_wrapped_domain(isl::multi_id tuple) const
{
  if (!ptr || tuple.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_bind_domain_wrapped_domain(copy(), tuple.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::domain_map(isl::space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_domain_map(space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::flat_range_product(isl::multi_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_flat_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::floor() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_floor(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff multi_aff::at(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_get_at(get(), pos);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff multi_aff::get_at(int pos) const
{
  return at(pos);
}

isl::multi_val multi_aff::constant_multi_val() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_get_constant_multi_val(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val multi_aff::get_constant_multi_val() const
{
  return constant_multi_val();
}

isl::aff_list multi_aff::list() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_get_list(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::aff_list multi_aff::get_list() const
{
  return list();
}

isl::space multi_aff::space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_get_space(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space multi_aff::get_space() const
{
  return space();
}

isl::multi_aff multi_aff::gist(isl::set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::identity() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_identity_multi_aff(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::identity_on_domain(isl::space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_identity_on_domain_space(space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::insert_domain(isl::space domain) const
{
  if (!ptr || domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_insert_domain(copy(), domain.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool multi_aff::involves_locals() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_involves_locals(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool multi_aff::involves_nan() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_involves_nan(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::multi_aff multi_aff::neg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_neg(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool multi_aff::plain_is_equal(const isl::multi_aff &multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_plain_is_equal(get(), multi2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::multi_aff multi_aff::product(isl::multi_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::pullback(isl::multi_aff ma2) const
{
  if (!ptr || ma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_pullback_multi_aff(copy(), ma2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::range_map(isl::space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_range_map(space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::range_product(isl::multi_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::scale(isl::multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_scale_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::scale(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_scale_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::scale(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->scale(isl::val(ctx(), v));
}

isl::multi_aff multi_aff::scale_down(isl::multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_scale_down_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::scale_down(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_scale_down_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::scale_down(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->scale_down(isl::val(ctx(), v));
}

isl::multi_aff multi_aff::set_at(int pos, isl::aff el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_set_at(copy(), pos, el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

unsigned multi_aff::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_size(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::multi_aff multi_aff::sub(isl::multi_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_sub(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::unbind_params_insert_domain(isl::multi_id domain) const
{
  if (!ptr || domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_unbind_params_insert_domain(copy(), domain.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_aff multi_aff::zero(isl::space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_aff_zero(space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const multi_aff &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_multi_aff_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_multi_aff_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::multi_id
multi_id manage(__isl_take isl_multi_id *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return multi_id(ptr);
}
multi_id manage_copy(__isl_keep isl_multi_id *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_multi_id_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_multi_id_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return multi_id(ptr);
}

multi_id::multi_id()
    : ptr(nullptr) {}

multi_id::multi_id(const multi_id &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_multi_id_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

multi_id::multi_id(__isl_take isl_multi_id *ptr)
    : ptr(ptr) {}

multi_id::multi_id(isl::space space, isl::id_list list)
{
  if (space.is_null() || list.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_id_from_id_list(space.release(), list.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

multi_id::multi_id(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_id_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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

isl::multi_id multi_id::flat_range_product(isl::multi_id multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_id_flat_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::id multi_id::at(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_id_get_at(get(), pos);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::id multi_id::get_at(int pos) const
{
  return at(pos);
}

isl::id_list multi_id::list() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_id_get_list(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::id_list multi_id::get_list() const
{
  return list();
}

isl::space multi_id::space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_id_get_space(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space multi_id::get_space() const
{
  return space();
}

bool multi_id::plain_is_equal(const isl::multi_id &multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_id_plain_is_equal(get(), multi2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::multi_id multi_id::range_product(isl::multi_id multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_id_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_id multi_id::set_at(int pos, isl::id el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_id_set_at(copy(), pos, el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_id multi_id::set_at(int pos, const std::string &el) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->set_at(pos, isl::id(ctx(), el));
}

unsigned multi_id::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_id_size(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

inline std::ostream &operator<<(std::ostream &os, const multi_id &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_multi_id_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_multi_id_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::multi_pw_aff
multi_pw_aff manage(__isl_take isl_multi_pw_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return multi_pw_aff(ptr);
}
multi_pw_aff manage_copy(__isl_keep isl_multi_pw_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_multi_pw_aff_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_multi_pw_aff_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return multi_pw_aff(ptr);
}

multi_pw_aff::multi_pw_aff()
    : ptr(nullptr) {}

multi_pw_aff::multi_pw_aff(const multi_pw_aff &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_multi_pw_aff_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

multi_pw_aff::multi_pw_aff(__isl_take isl_multi_pw_aff *ptr)
    : ptr(ptr) {}

multi_pw_aff::multi_pw_aff(isl::aff aff)
{
  if (aff.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = aff.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_from_aff(aff.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

multi_pw_aff::multi_pw_aff(isl::multi_aff ma)
{
  if (ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ma.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_from_multi_aff(ma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

multi_pw_aff::multi_pw_aff(isl::pw_aff pa)
{
  if (pa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = pa.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_from_pw_aff(pa.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

multi_pw_aff::multi_pw_aff(isl::space space, isl::pw_aff_list list)
{
  if (space.is_null() || list.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_from_pw_aff_list(space.release(), list.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

multi_pw_aff::multi_pw_aff(isl::pw_multi_aff pma)
{
  if (pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = pma.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_from_pw_multi_aff(pma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

multi_pw_aff::multi_pw_aff(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_add(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::add_constant(isl::multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_add_constant_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::add_constant(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_add_constant_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::add_constant(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->add_constant(isl::val(ctx(), v));
}

isl::set multi_pw_aff::bind(isl::multi_id tuple) const
{
  if (!ptr || tuple.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_bind(copy(), tuple.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::bind_domain(isl::multi_id tuple) const
{
  if (!ptr || tuple.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_bind_domain(copy(), tuple.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::bind_domain_wrapped_domain(isl::multi_id tuple) const
{
  if (!ptr || tuple.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_bind_domain_wrapped_domain(copy(), tuple.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::coalesce() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_coalesce(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set multi_pw_aff::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::flat_range_product(isl::multi_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_flat_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff multi_pw_aff::at(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_get_at(get(), pos);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff multi_pw_aff::get_at(int pos) const
{
  return at(pos);
}

isl::pw_aff_list multi_pw_aff::list() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_get_list(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff_list multi_pw_aff::get_list() const
{
  return list();
}

isl::space multi_pw_aff::space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_get_space(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space multi_pw_aff::get_space() const
{
  return space();
}

isl::multi_pw_aff multi_pw_aff::gist(isl::set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_gist(copy(), set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::identity() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_identity_multi_pw_aff(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::identity_on_domain(isl::space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_identity_on_domain_space(space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::insert_domain(isl::space domain) const
{
  if (!ptr || domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_insert_domain(copy(), domain.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::intersect_domain(isl::set domain) const
{
  if (!ptr || domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_intersect_domain(copy(), domain.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::intersect_params(isl::set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_intersect_params(copy(), set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool multi_pw_aff::involves_nan() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_involves_nan(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool multi_pw_aff::involves_param(const isl::id &id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_involves_param_id(get(), id.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool multi_pw_aff::involves_param(const std::string &id) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->involves_param(isl::id(ctx(), id));
}

bool multi_pw_aff::involves_param(const isl::id_list &list) const
{
  if (!ptr || list.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_involves_param_id_list(get(), list.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::multi_pw_aff multi_pw_aff::max(isl::multi_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_max(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val multi_pw_aff::max_multi_val() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_max_multi_val(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::min(isl::multi_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_min(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val multi_pw_aff::min_multi_val() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_min_multi_val(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::neg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_neg(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool multi_pw_aff::plain_is_equal(const isl::multi_pw_aff &multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_plain_is_equal(get(), multi2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::multi_pw_aff multi_pw_aff::product(isl::multi_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::pullback(isl::multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_pullback_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::pullback(isl::multi_pw_aff mpa2) const
{
  if (!ptr || mpa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_pullback_multi_pw_aff(copy(), mpa2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::pullback(isl::pw_multi_aff pma) const
{
  if (!ptr || pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_pullback_pw_multi_aff(copy(), pma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::range_product(isl::multi_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::scale(isl::multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_scale_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::scale(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_scale_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::scale(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->scale(isl::val(ctx(), v));
}

isl::multi_pw_aff multi_pw_aff::scale_down(isl::multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_scale_down_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::scale_down(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_scale_down_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::scale_down(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->scale_down(isl::val(ctx(), v));
}

isl::multi_pw_aff multi_pw_aff::set_at(int pos, isl::pw_aff el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_set_at(copy(), pos, el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

unsigned multi_pw_aff::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_size(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::multi_pw_aff multi_pw_aff::sub(isl::multi_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_sub(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::unbind_params_insert_domain(isl::multi_id domain) const
{
  if (!ptr || domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_unbind_params_insert_domain(copy(), domain.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::union_add(isl::multi_pw_aff mpa2) const
{
  if (!ptr || mpa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_union_add(copy(), mpa2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff multi_pw_aff::zero(isl::space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_pw_aff_zero(space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const multi_pw_aff &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_multi_pw_aff_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_multi_pw_aff_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::multi_union_pw_aff
multi_union_pw_aff manage(__isl_take isl_multi_union_pw_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return multi_union_pw_aff(ptr);
}
multi_union_pw_aff manage_copy(__isl_keep isl_multi_union_pw_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_multi_union_pw_aff_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_multi_union_pw_aff_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return multi_union_pw_aff(ptr);
}

multi_union_pw_aff::multi_union_pw_aff()
    : ptr(nullptr) {}

multi_union_pw_aff::multi_union_pw_aff(const multi_union_pw_aff &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_multi_union_pw_aff_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

multi_union_pw_aff::multi_union_pw_aff(__isl_take isl_multi_union_pw_aff *ptr)
    : ptr(ptr) {}

multi_union_pw_aff::multi_union_pw_aff(isl::multi_pw_aff mpa)
{
  if (mpa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = mpa.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_from_multi_pw_aff(mpa.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

multi_union_pw_aff::multi_union_pw_aff(isl::union_pw_aff upa)
{
  if (upa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = upa.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_from_union_pw_aff(upa.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

multi_union_pw_aff::multi_union_pw_aff(isl::space space, isl::union_pw_aff_list list)
{
  if (space.is_null() || list.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_from_union_pw_aff_list(space.release(), list.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

multi_union_pw_aff::multi_union_pw_aff(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_add(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set multi_union_pw_aff::bind(isl::multi_id tuple) const
{
  if (!ptr || tuple.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_bind(copy(), tuple.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::coalesce() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_coalesce(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set multi_union_pw_aff::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::flat_range_product(isl::multi_union_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_flat_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_aff multi_union_pw_aff::at(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_get_at(get(), pos);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_aff multi_union_pw_aff::get_at(int pos) const
{
  return at(pos);
}

isl::union_pw_aff_list multi_union_pw_aff::list() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_get_list(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_aff_list multi_union_pw_aff::get_list() const
{
  return list();
}

isl::space multi_union_pw_aff::space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_get_space(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space multi_union_pw_aff::get_space() const
{
  return space();
}

isl::multi_union_pw_aff multi_union_pw_aff::gist(isl::union_set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::intersect_domain(isl::union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_intersect_domain(copy(), uset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::intersect_params(isl::set params) const
{
  if (!ptr || params.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_intersect_params(copy(), params.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool multi_union_pw_aff::involves_nan() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_involves_nan(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::multi_union_pw_aff multi_union_pw_aff::neg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_neg(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool multi_union_pw_aff::plain_is_equal(const isl::multi_union_pw_aff &multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_plain_is_equal(get(), multi2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::multi_union_pw_aff multi_union_pw_aff::pullback(isl::union_pw_multi_aff upma) const
{
  if (!ptr || upma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_pullback_union_pw_multi_aff(copy(), upma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::range_product(isl::multi_union_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::scale(isl::multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_scale_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::scale(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_scale_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::scale(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->scale(isl::val(ctx(), v));
}

isl::multi_union_pw_aff multi_union_pw_aff::scale_down(isl::multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_scale_down_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::scale_down(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_scale_down_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::scale_down(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->scale_down(isl::val(ctx(), v));
}

isl::multi_union_pw_aff multi_union_pw_aff::set_at(int pos, isl::union_pw_aff el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_set_at(copy(), pos, el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

unsigned multi_union_pw_aff::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_size(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::multi_union_pw_aff multi_union_pw_aff::sub(isl::multi_union_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_sub(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::union_add(isl::multi_union_pw_aff mupa2) const
{
  if (!ptr || mupa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_union_add(copy(), mupa2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_union_pw_aff multi_union_pw_aff::zero(isl::space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_zero(space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const multi_union_pw_aff &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_multi_union_pw_aff_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_multi_union_pw_aff_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::multi_val
multi_val manage(__isl_take isl_multi_val *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return multi_val(ptr);
}
multi_val manage_copy(__isl_keep isl_multi_val *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_multi_val_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_multi_val_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return multi_val(ptr);
}

multi_val::multi_val()
    : ptr(nullptr) {}

multi_val::multi_val(const multi_val &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_multi_val_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

multi_val::multi_val(__isl_take isl_multi_val *ptr)
    : ptr(ptr) {}

multi_val::multi_val(isl::space space, isl::val_list list)
{
  if (space.is_null() || list.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_from_val_list(space.release(), list.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

multi_val::multi_val(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_add(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val multi_val::add(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_add_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val multi_val::add(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->add(isl::val(ctx(), v));
}

isl::multi_val multi_val::flat_range_product(isl::multi_val multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_flat_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val multi_val::at(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_get_at(get(), pos);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val multi_val::get_at(int pos) const
{
  return at(pos);
}

isl::val_list multi_val::list() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_get_list(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val_list multi_val::get_list() const
{
  return list();
}

isl::space multi_val::space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_get_space(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space multi_val::get_space() const
{
  return space();
}

bool multi_val::involves_nan() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_involves_nan(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::multi_val multi_val::max(isl::multi_val multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_max(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val multi_val::min(isl::multi_val multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_min(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val multi_val::neg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_neg(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool multi_val::plain_is_equal(const isl::multi_val &multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_plain_is_equal(get(), multi2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::multi_val multi_val::product(isl::multi_val multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val multi_val::range_product(isl::multi_val multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val multi_val::scale(isl::multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_scale_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val multi_val::scale(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_scale_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val multi_val::scale(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->scale(isl::val(ctx(), v));
}

isl::multi_val multi_val::scale_down(isl::multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_scale_down_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val multi_val::scale_down(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_scale_down_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val multi_val::scale_down(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->scale_down(isl::val(ctx(), v));
}

isl::multi_val multi_val::set_at(int pos, isl::val el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_set_at(copy(), pos, el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val multi_val::set_at(int pos, long el) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->set_at(pos, isl::val(ctx(), el));
}

unsigned multi_val::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_size(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::multi_val multi_val::sub(isl::multi_val multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_sub(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val multi_val::zero(isl::space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_multi_val_zero(space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const multi_val &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_multi_val_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_multi_val_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::point
point manage(__isl_take isl_point *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return point(ptr);
}
point manage_copy(__isl_keep isl_point *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_point_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_point_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return point(ptr);
}

point::point()
    : ptr(nullptr) {}

point::point(const point &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_point_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
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

isl::ctx point::ctx() const {
  return isl::ctx(isl_point_get_ctx(ptr));
}

isl::multi_val point::multi_val() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_point_get_multi_val(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val point::get_multi_val() const
{
  return multi_val();
}

inline std::ostream &operator<<(std::ostream &os, const point &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_point_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_point_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::pw_aff
pw_aff manage(__isl_take isl_pw_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return pw_aff(ptr);
}
pw_aff manage_copy(__isl_keep isl_pw_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_pw_aff_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_pw_aff_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return pw_aff(ptr);
}

pw_aff::pw_aff()
    : ptr(nullptr) {}

pw_aff::pw_aff(const pw_aff &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_pw_aff_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

pw_aff::pw_aff(__isl_take isl_pw_aff *ptr)
    : ptr(ptr) {}

pw_aff::pw_aff(isl::aff aff)
{
  if (aff.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = aff.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_from_aff(aff.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

pw_aff::pw_aff(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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

isl::pw_aff pw_aff::add(isl::pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_add(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::add_constant(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_add_constant_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::add_constant(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->add_constant(isl::val(ctx(), v));
}

isl::aff pw_aff::as_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_as_aff(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set pw_aff::bind(isl::id id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_bind_id(copy(), id.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set pw_aff::bind(const std::string &id) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->bind(isl::id(ctx(), id));
}

isl::pw_aff pw_aff::bind_domain(isl::multi_id tuple) const
{
  if (!ptr || tuple.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_bind_domain(copy(), tuple.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::bind_domain_wrapped_domain(isl::multi_id tuple) const
{
  if (!ptr || tuple.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_bind_domain_wrapped_domain(copy(), tuple.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::ceil() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_ceil(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::coalesce() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_coalesce(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::cond(isl::pw_aff pwaff_true, isl::pw_aff pwaff_false) const
{
  if (!ptr || pwaff_true.is_null() || pwaff_false.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_cond(copy(), pwaff_true.release(), pwaff_false.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::div(isl::pw_aff pa2) const
{
  if (!ptr || pa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_div(copy(), pa2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set pw_aff::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set pw_aff::eq_set(isl::pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_eq_set(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val pw_aff::eval(isl::point pnt) const
{
  if (!ptr || pnt.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_eval(copy(), pnt.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::floor() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_floor(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set pw_aff::ge_set(isl::pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_ge_set(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::gist(isl::set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set pw_aff::gt_set(isl::pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_gt_set(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::insert_domain(isl::space domain) const
{
  if (!ptr || domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_insert_domain(copy(), domain.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::intersect_domain(isl::set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_intersect_domain(copy(), set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::intersect_params(isl::set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_intersect_params(copy(), set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool pw_aff::isa_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_isa_aff(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::set pw_aff::le_set(isl::pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_le_set(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set pw_aff::lt_set(isl::pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_lt_set(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::max(isl::pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_max(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::min(isl::pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_min(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::mod(isl::val mod) const
{
  if (!ptr || mod.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_mod_val(copy(), mod.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::mod(long mod) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->mod(isl::val(ctx(), mod));
}

isl::pw_aff pw_aff::mul(isl::pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_mul(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set pw_aff::ne_set(isl::pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_ne_set(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::neg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_neg(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::param_on_domain(isl::set domain, isl::id id)
{
  if (domain.is_null() || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = domain.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_param_on_domain_id(domain.release(), id.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::pullback(isl::multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_pullback_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::pullback(isl::multi_pw_aff mpa) const
{
  if (!ptr || mpa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_pullback_multi_pw_aff(copy(), mpa.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::pullback(isl::pw_multi_aff pma) const
{
  if (!ptr || pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_pullback_pw_multi_aff(copy(), pma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::scale(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_scale_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::scale(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->scale(isl::val(ctx(), v));
}

isl::pw_aff pw_aff::scale_down(isl::val f) const
{
  if (!ptr || f.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_scale_down_val(copy(), f.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::scale_down(long f) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->scale_down(isl::val(ctx(), f));
}

isl::pw_aff pw_aff::sub(isl::pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_sub(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::subtract_domain(isl::set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_subtract_domain(copy(), set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::tdiv_q(isl::pw_aff pa2) const
{
  if (!ptr || pa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_tdiv_q(copy(), pa2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::tdiv_r(isl::pw_aff pa2) const
{
  if (!ptr || pa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_tdiv_r(copy(), pa2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff::union_add(isl::pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_union_add(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const pw_aff &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_pw_aff_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_pw_aff_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::pw_aff_list
pw_aff_list manage(__isl_take isl_pw_aff_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return pw_aff_list(ptr);
}
pw_aff_list manage_copy(__isl_keep isl_pw_aff_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_pw_aff_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_pw_aff_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return pw_aff_list(ptr);
}

pw_aff_list::pw_aff_list()
    : ptr(nullptr) {}

pw_aff_list::pw_aff_list(const pw_aff_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_pw_aff_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

pw_aff_list::pw_aff_list(__isl_take isl_pw_aff_list *ptr)
    : ptr(ptr) {}

pw_aff_list::pw_aff_list(isl::ctx ctx, int n)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

pw_aff_list::pw_aff_list(isl::pw_aff el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = el.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_list_from_pw_aff(el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff_list pw_aff_list::clear() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_list_clear(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff_list pw_aff_list::concat(isl::pw_aff_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff_list pw_aff_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

void pw_aff_list::foreach(const std::function<void(isl::pw_aff)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct fn_data {
    std::function<void(isl::pw_aff)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_pw_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_pw_aff_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return;
}

isl::pw_aff pw_aff_list::at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff pw_aff_list::get_at(int index) const
{
  return at(index);
}

isl::pw_aff_list pw_aff_list::insert(unsigned int pos, isl::pw_aff el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_list_insert(copy(), pos, el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

unsigned pw_aff_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_aff_list_size(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

inline std::ostream &operator<<(std::ostream &os, const pw_aff_list &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_pw_aff_list_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_pw_aff_list_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::pw_multi_aff
pw_multi_aff manage(__isl_take isl_pw_multi_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return pw_multi_aff(ptr);
}
pw_multi_aff manage_copy(__isl_keep isl_pw_multi_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_pw_multi_aff_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_pw_multi_aff_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return pw_multi_aff(ptr);
}

pw_multi_aff::pw_multi_aff()
    : ptr(nullptr) {}

pw_multi_aff::pw_multi_aff(const pw_multi_aff &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_pw_multi_aff_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

pw_multi_aff::pw_multi_aff(__isl_take isl_pw_multi_aff *ptr)
    : ptr(ptr) {}

pw_multi_aff::pw_multi_aff(isl::multi_aff ma)
{
  if (ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ma.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_from_multi_aff(ma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

pw_multi_aff::pw_multi_aff(isl::pw_aff pa)
{
  if (pa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = pa.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_from_pw_aff(pa.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

pw_multi_aff::pw_multi_aff(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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

isl::pw_multi_aff pw_multi_aff::add(isl::pw_multi_aff pma2) const
{
  if (!ptr || pma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_add(copy(), pma2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::add_constant(isl::multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_add_constant_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::add_constant(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_add_constant_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::add_constant(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->add_constant(isl::val(ctx(), v));
}

isl::multi_aff pw_multi_aff::as_multi_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_as_multi_aff(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::bind_domain(isl::multi_id tuple) const
{
  if (!ptr || tuple.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_bind_domain(copy(), tuple.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::bind_domain_wrapped_domain(isl::multi_id tuple) const
{
  if (!ptr || tuple.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_bind_domain_wrapped_domain(copy(), tuple.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::coalesce() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_coalesce(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set pw_multi_aff::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::domain_map(isl::space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_domain_map(space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::flat_range_product(isl::pw_multi_aff pma2) const
{
  if (!ptr || pma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_flat_range_product(copy(), pma2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

void pw_multi_aff::foreach_piece(const std::function<void(isl::set, isl::multi_aff)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct fn_data {
    std::function<void(isl::set, isl::multi_aff)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_set *arg_0, isl_multi_aff *arg_1, void *arg_2) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_2);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0), manage(arg_1));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_pw_multi_aff_foreach_piece(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return;
}

isl::space pw_multi_aff::space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_get_space(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space pw_multi_aff::get_space() const
{
  return space();
}

isl::pw_multi_aff pw_multi_aff::gist(isl::set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_gist(copy(), set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::identity_on_domain(isl::space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_identity_on_domain_space(space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::insert_domain(isl::space domain) const
{
  if (!ptr || domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_insert_domain(copy(), domain.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::intersect_domain(isl::set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_intersect_domain(copy(), set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::intersect_params(isl::set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_intersect_params(copy(), set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool pw_multi_aff::involves_locals() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_involves_locals(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool pw_multi_aff::isa_multi_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_isa_multi_aff(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::multi_val pw_multi_aff::max_multi_val() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_max_multi_val(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val pw_multi_aff::min_multi_val() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_min_multi_val(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

unsigned pw_multi_aff::n_piece() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_n_piece(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::pw_multi_aff pw_multi_aff::preimage_domain_wrapped_domain(isl::pw_multi_aff pma2) const
{
  if (!ptr || pma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_preimage_domain_wrapped_domain_pw_multi_aff(copy(), pma2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::product(isl::pw_multi_aff pma2) const
{
  if (!ptr || pma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_product(copy(), pma2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::pullback(isl::multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_pullback_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::pullback(isl::pw_multi_aff pma2) const
{
  if (!ptr || pma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_pullback_pw_multi_aff(copy(), pma2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::range_factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_range_factor_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::range_factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_range_factor_range(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::range_map(isl::space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_range_map(space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::range_product(isl::pw_multi_aff pma2) const
{
  if (!ptr || pma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_range_product(copy(), pma2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::scale(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_scale_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::scale(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->scale(isl::val(ctx(), v));
}

isl::pw_multi_aff pw_multi_aff::scale_down(isl::val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_scale_down_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::scale_down(long v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->scale_down(isl::val(ctx(), v));
}

isl::pw_multi_aff pw_multi_aff::sub(isl::pw_multi_aff pma2) const
{
  if (!ptr || pma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_sub(copy(), pma2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::subtract_domain(isl::set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_subtract_domain(copy(), set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::union_add(isl::pw_multi_aff pma2) const
{
  if (!ptr || pma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_union_add(copy(), pma2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff::zero(isl::space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_zero(space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const pw_multi_aff &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_pw_multi_aff_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_pw_multi_aff_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::pw_multi_aff_list
pw_multi_aff_list manage(__isl_take isl_pw_multi_aff_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return pw_multi_aff_list(ptr);
}
pw_multi_aff_list manage_copy(__isl_keep isl_pw_multi_aff_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_pw_multi_aff_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_pw_multi_aff_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return pw_multi_aff_list(ptr);
}

pw_multi_aff_list::pw_multi_aff_list()
    : ptr(nullptr) {}

pw_multi_aff_list::pw_multi_aff_list(const pw_multi_aff_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_pw_multi_aff_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

pw_multi_aff_list::pw_multi_aff_list(__isl_take isl_pw_multi_aff_list *ptr)
    : ptr(ptr) {}

pw_multi_aff_list::pw_multi_aff_list(isl::ctx ctx, int n)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

pw_multi_aff_list::pw_multi_aff_list(isl::pw_multi_aff el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = el.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_list_from_pw_multi_aff(el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff_list pw_multi_aff_list::clear() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_list_clear(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff_list pw_multi_aff_list::concat(isl::pw_multi_aff_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff_list pw_multi_aff_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

void pw_multi_aff_list::foreach(const std::function<void(isl::pw_multi_aff)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct fn_data {
    std::function<void(isl::pw_multi_aff)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_pw_multi_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_pw_multi_aff_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return;
}

isl::pw_multi_aff pw_multi_aff_list::at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff pw_multi_aff_list::get_at(int index) const
{
  return at(index);
}

isl::pw_multi_aff_list pw_multi_aff_list::insert(unsigned int pos, isl::pw_multi_aff el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_list_insert(copy(), pos, el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

unsigned pw_multi_aff_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_pw_multi_aff_list_size(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

inline std::ostream &operator<<(std::ostream &os, const pw_multi_aff_list &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_pw_multi_aff_list_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_pw_multi_aff_list_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::schedule
schedule manage(__isl_take isl_schedule *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return schedule(ptr);
}
schedule manage_copy(__isl_keep isl_schedule *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_schedule_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return schedule(ptr);
}

schedule::schedule()
    : ptr(nullptr) {}

schedule::schedule(const schedule &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

schedule::schedule(__isl_take isl_schedule *ptr)
    : ptr(ptr) {}

schedule::schedule(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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

isl::schedule schedule::from_domain(isl::union_set domain)
{
  if (domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = domain.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_from_domain(domain.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set schedule::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_get_domain(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set schedule::get_domain() const
{
  return domain();
}

isl::union_map schedule::map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_get_map(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map schedule::get_map() const
{
  return map();
}

isl::schedule_node schedule::root() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_get_root(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_node schedule::get_root() const
{
  return root();
}

isl::schedule schedule::pullback(isl::union_pw_multi_aff upma) const
{
  if (!ptr || upma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_pullback_union_pw_multi_aff(copy(), upma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const schedule &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_schedule_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::schedule_constraints
schedule_constraints manage(__isl_take isl_schedule_constraints *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return schedule_constraints(ptr);
}
schedule_constraints manage_copy(__isl_keep isl_schedule_constraints *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_constraints_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_schedule_constraints_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return schedule_constraints(ptr);
}

schedule_constraints::schedule_constraints()
    : ptr(nullptr) {}

schedule_constraints::schedule_constraints(const schedule_constraints &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_constraints_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

schedule_constraints::schedule_constraints(__isl_take isl_schedule_constraints *ptr)
    : ptr(ptr) {}

schedule_constraints::schedule_constraints(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_constraints_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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

isl::schedule schedule_constraints::compute_schedule() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_constraints_compute_schedule(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map schedule_constraints::coincidence() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_constraints_get_coincidence(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map schedule_constraints::get_coincidence() const
{
  return coincidence();
}

isl::union_map schedule_constraints::conditional_validity() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_constraints_get_conditional_validity(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map schedule_constraints::get_conditional_validity() const
{
  return conditional_validity();
}

isl::union_map schedule_constraints::conditional_validity_condition() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_constraints_get_conditional_validity_condition(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map schedule_constraints::get_conditional_validity_condition() const
{
  return conditional_validity_condition();
}

isl::set schedule_constraints::context() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_constraints_get_context(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set schedule_constraints::get_context() const
{
  return context();
}

isl::union_set schedule_constraints::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_constraints_get_domain(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set schedule_constraints::get_domain() const
{
  return domain();
}

isl::union_map schedule_constraints::proximity() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_constraints_get_proximity(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map schedule_constraints::get_proximity() const
{
  return proximity();
}

isl::union_map schedule_constraints::validity() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_constraints_get_validity(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map schedule_constraints::get_validity() const
{
  return validity();
}

isl::schedule_constraints schedule_constraints::on_domain(isl::union_set domain)
{
  if (domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = domain.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_constraints_on_domain(domain.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_constraints schedule_constraints::set_coincidence(isl::union_map coincidence) const
{
  if (!ptr || coincidence.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_constraints_set_coincidence(copy(), coincidence.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_constraints schedule_constraints::set_conditional_validity(isl::union_map condition, isl::union_map validity) const
{
  if (!ptr || condition.is_null() || validity.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_constraints_set_conditional_validity(copy(), condition.release(), validity.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_constraints schedule_constraints::set_context(isl::set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_constraints_set_context(copy(), context.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_constraints schedule_constraints::set_proximity(isl::union_map proximity) const
{
  if (!ptr || proximity.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_constraints_set_proximity(copy(), proximity.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_constraints schedule_constraints::set_validity(isl::union_map validity) const
{
  if (!ptr || validity.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_constraints_set_validity(copy(), validity.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const schedule_constraints &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_constraints_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_schedule_constraints_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::schedule_node
schedule_node manage(__isl_take isl_schedule_node *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return schedule_node(ptr);
}
schedule_node manage_copy(__isl_keep isl_schedule_node *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_node_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_schedule_node_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return schedule_node(ptr);
}

schedule_node::schedule_node()
    : ptr(nullptr) {}

schedule_node::schedule_node(const schedule_node &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_node_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
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
bool schedule_node::isa_type(T subtype) const
{
  if (is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return isl_schedule_node_get_type(get()) == subtype;
}
template <class T>
bool schedule_node::isa() const
{
  return isa_type<decltype(T::type)>(T::type);
}
template <class T>
T schedule_node::as() const
{
 if (!isa<T>())
    exception::throw_invalid("not an object of the requested subtype", __FILE__, __LINE__);
  return T(copy());
}

isl::ctx schedule_node::ctx() const {
  return isl::ctx(isl_schedule_node_get_ctx(ptr));
}

isl::schedule_node schedule_node::ancestor(int generation) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_ancestor(copy(), generation);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_node schedule_node::child(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_child(copy(), pos);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool schedule_node::every_descendant(const std::function<bool(isl::schedule_node)> &test) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct test_data {
    std::function<bool(isl::schedule_node)> func;
    std::exception_ptr eptr;
  } test_data = { test };
  auto test_lambda = [](isl_schedule_node *arg_0, void *arg_1) -> isl_bool {
    auto *data = static_cast<struct test_data *>(arg_1);
    ISL_CPP_TRY {
      auto ret = (data->func)(manage_copy(arg_0));
      return ret ? isl_bool_true : isl_bool_false;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_bool_error;
    }
  };
  auto res = isl_schedule_node_every_descendant(get(), test_lambda, &test_data);
  if (test_data.eptr)
    std::rethrow_exception(test_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::schedule_node schedule_node::first_child() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_first_child(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

void schedule_node::foreach_ancestor_top_down(const std::function<void(isl::schedule_node)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct fn_data {
    std::function<void(isl::schedule_node)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_schedule_node *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage_copy(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_schedule_node_foreach_ancestor_top_down(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return;
}

void schedule_node::foreach_descendant_top_down(const std::function<bool(isl::schedule_node)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct fn_data {
    std::function<bool(isl::schedule_node)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_schedule_node *arg_0, void *arg_1) -> isl_bool {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      auto ret = (data->func)(manage_copy(arg_0));
      return ret ? isl_bool_true : isl_bool_false;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_bool_error;
    }
  };
  auto res = isl_schedule_node_foreach_descendant_top_down(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return;
}

isl::schedule_node schedule_node::from_domain(isl::union_set domain)
{
  if (domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = domain.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_from_domain(domain.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_node schedule_node::from_extension(isl::union_map extension)
{
  if (extension.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = extension.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_from_extension(extension.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

unsigned schedule_node::ancestor_child_position(const isl::schedule_node &ancestor) const
{
  if (!ptr || ancestor.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_get_ancestor_child_position(get(), ancestor.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

unsigned schedule_node::get_ancestor_child_position(const isl::schedule_node &ancestor) const
{
  return ancestor_child_position(ancestor);
}

unsigned schedule_node::child_position() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_get_child_position(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

unsigned schedule_node::get_child_position() const
{
  return child_position();
}

isl::multi_union_pw_aff schedule_node::prefix_schedule_multi_union_pw_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_union_pw_aff schedule_node::get_prefix_schedule_multi_union_pw_aff() const
{
  return prefix_schedule_multi_union_pw_aff();
}

isl::union_map schedule_node::prefix_schedule_union_map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_get_prefix_schedule_union_map(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map schedule_node::get_prefix_schedule_union_map() const
{
  return prefix_schedule_union_map();
}

isl::union_pw_multi_aff schedule_node::prefix_schedule_union_pw_multi_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff schedule_node::get_prefix_schedule_union_pw_multi_aff() const
{
  return prefix_schedule_union_pw_multi_aff();
}

isl::schedule schedule_node::schedule() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_get_schedule(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule schedule_node::get_schedule() const
{
  return schedule();
}

isl::schedule_node schedule_node::shared_ancestor(const isl::schedule_node &node2) const
{
  if (!ptr || node2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_get_shared_ancestor(get(), node2.get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_node schedule_node::get_shared_ancestor(const isl::schedule_node &node2) const
{
  return shared_ancestor(node2);
}

unsigned schedule_node::tree_depth() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_get_tree_depth(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

unsigned schedule_node::get_tree_depth() const
{
  return tree_depth();
}

isl::schedule_node schedule_node::graft_after(isl::schedule_node graft) const
{
  if (!ptr || graft.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_graft_after(copy(), graft.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_node schedule_node::graft_before(isl::schedule_node graft) const
{
  if (!ptr || graft.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_graft_before(copy(), graft.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool schedule_node::has_children() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_has_children(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool schedule_node::has_next_sibling() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_has_next_sibling(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool schedule_node::has_parent() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_has_parent(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool schedule_node::has_previous_sibling() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_has_previous_sibling(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::schedule_node schedule_node::insert_context(isl::set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_insert_context(copy(), context.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_node schedule_node::insert_filter(isl::union_set filter) const
{
  if (!ptr || filter.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_insert_filter(copy(), filter.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_node schedule_node::insert_guard(isl::set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_insert_guard(copy(), context.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_node schedule_node::insert_mark(isl::id mark) const
{
  if (!ptr || mark.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_insert_mark(copy(), mark.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_node schedule_node::insert_mark(const std::string &mark) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->insert_mark(isl::id(ctx(), mark));
}

isl::schedule_node schedule_node::insert_partial_schedule(isl::multi_union_pw_aff schedule) const
{
  if (!ptr || schedule.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_insert_partial_schedule(copy(), schedule.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_node schedule_node::insert_sequence(isl::union_set_list filters) const
{
  if (!ptr || filters.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_insert_sequence(copy(), filters.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_node schedule_node::insert_set(isl::union_set_list filters) const
{
  if (!ptr || filters.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_insert_set(copy(), filters.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool schedule_node::is_equal(const isl::schedule_node &node2) const
{
  if (!ptr || node2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_is_equal(get(), node2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool schedule_node::is_subtree_anchored() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_is_subtree_anchored(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::schedule_node schedule_node::map_descendant_bottom_up(const std::function<isl::schedule_node(isl::schedule_node)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct fn_data {
    std::function<isl::schedule_node(isl::schedule_node)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_schedule_node *arg_0, void *arg_1) -> isl_schedule_node * {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      auto ret = (data->func)(manage(arg_0));
      return ret.release();
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return NULL;
    }
  };
  auto res = isl_schedule_node_map_descendant_bottom_up(copy(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

unsigned schedule_node::n_children() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_n_children(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::schedule_node schedule_node::next_sibling() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_next_sibling(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_node schedule_node::order_after(isl::union_set filter) const
{
  if (!ptr || filter.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_order_after(copy(), filter.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_node schedule_node::order_before(isl::union_set filter) const
{
  if (!ptr || filter.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_order_before(copy(), filter.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_node schedule_node::parent() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_parent(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_node schedule_node::previous_sibling() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_previous_sibling(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::schedule_node schedule_node::root() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_root(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_node_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_get_ast_build_options(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set schedule_node_band::get_ast_build_options() const
{
  return ast_build_options();
}

isl::set schedule_node_band::ast_isolate_option() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_get_ast_isolate_option(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set schedule_node_band::get_ast_isolate_option() const
{
  return ast_isolate_option();
}

isl::multi_union_pw_aff schedule_node_band::partial_schedule() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_get_partial_schedule(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_union_pw_aff schedule_node_band::get_partial_schedule() const
{
  return partial_schedule();
}

bool schedule_node_band::permutable() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_get_permutable(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool schedule_node_band::get_permutable() const
{
  return permutable();
}

bool schedule_node_band::member_get_coincident(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_member_get_coincident(get(), pos);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

schedule_node_band schedule_node_band::member_set_coincident(int pos, int coincident) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_member_set_coincident(copy(), pos, coincident);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::mod(isl::multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_mod(copy(), mv.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res).as<schedule_node_band>();
}

unsigned schedule_node_band::n_member() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_n_member(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

schedule_node_band schedule_node_band::scale(isl::multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_scale(copy(), mv.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::scale_down(isl::multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_scale_down(copy(), mv.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::set_ast_build_options(isl::union_set options) const
{
  if (!ptr || options.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_set_ast_build_options(copy(), options.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::set_permutable(int permutable) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_set_permutable(copy(), permutable);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::shift(isl::multi_union_pw_aff shift) const
{
  if (!ptr || shift.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_shift(copy(), shift.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::split(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_split(copy(), pos);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::tile(isl::multi_val sizes) const
{
  if (!ptr || sizes.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_tile(copy(), sizes.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::member_set_ast_loop_default(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_member_set_ast_loop_type(copy(), pos, isl_ast_loop_default);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::member_set_ast_loop_atomic(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_member_set_ast_loop_type(copy(), pos, isl_ast_loop_atomic);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::member_set_ast_loop_unroll(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_member_set_ast_loop_type(copy(), pos, isl_ast_loop_unroll);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::member_set_ast_loop_separate(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_band_member_set_ast_loop_type(copy(), pos, isl_ast_loop_separate);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res).as<schedule_node_band>();
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node_band &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_node_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_context_get_context(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set schedule_node_context::get_context() const
{
  return context();
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node_context &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_node_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_domain_get_domain(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set schedule_node_domain::get_domain() const
{
  return domain();
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node_domain &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_node_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_expansion_get_contraction(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff schedule_node_expansion::get_contraction() const
{
  return contraction();
}

isl::union_map schedule_node_expansion::expansion() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_expansion_get_expansion(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map schedule_node_expansion::get_expansion() const
{
  return expansion();
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node_expansion &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_node_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_extension_get_extension(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map schedule_node_extension::get_extension() const
{
  return extension();
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node_extension &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_node_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_filter_get_filter(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set schedule_node_filter::get_filter() const
{
  return filter();
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node_filter &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_node_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_schedule_node_guard_get_guard(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set schedule_node_guard::get_guard() const
{
  return guard();
}

inline std::ostream &operator<<(std::ostream &os, const schedule_node_guard &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_node_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_node_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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

inline std::ostream &operator<<(std::ostream &os, const schedule_node_mark &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_node_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_node_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
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
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_schedule_node_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_schedule_node_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::set
set manage(__isl_take isl_set *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return set(ptr);
}
set manage_copy(__isl_keep isl_set *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_set_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_set_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return set(ptr);
}

set::set()
    : ptr(nullptr) {}

set::set(const set &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_set_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

set::set(__isl_take isl_set *ptr)
    : ptr(ptr) {}

set::set(isl::basic_set bset)
{
  if (bset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = bset.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_from_basic_set(bset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

set::set(isl::point pnt)
{
  if (pnt.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = pnt.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_from_point(pnt.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

set::set(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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

isl::basic_set set::affine_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_affine_hull(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::apply(isl::map map) const
{
  if (!ptr || map.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_apply(copy(), map.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::bind(isl::multi_id tuple) const
{
  if (!ptr || tuple.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_bind(copy(), tuple.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::coalesce() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_coalesce(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::complement() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_complement(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::detect_equalities() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_detect_equalities(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val set::dim_max_val(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_dim_max_val(copy(), pos);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val set::dim_min_val(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_dim_min_val(copy(), pos);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::empty(isl::space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_empty(space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::flatten() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_flatten(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

void set::foreach_basic_set(const std::function<void(isl::basic_set)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct fn_data {
    std::function<void(isl::basic_set)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_basic_set *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_set_foreach_basic_set(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return;
}

void set::foreach_point(const std::function<void(isl::point)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct fn_data {
    std::function<void(isl::point)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_point *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_set_foreach_point(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return;
}

isl::multi_val set::plain_multi_val_if_fixed() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_get_plain_multi_val_if_fixed(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_val set::get_plain_multi_val_if_fixed() const
{
  return plain_multi_val_if_fixed();
}

isl::fixed_box set::simple_fixed_box_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_get_simple_fixed_box_hull(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::fixed_box set::get_simple_fixed_box_hull() const
{
  return simple_fixed_box_hull();
}

isl::space set::space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_get_space(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space set::get_space() const
{
  return space();
}

isl::val set::stride(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_get_stride(get(), pos);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val set::get_stride(int pos) const
{
  return stride(pos);
}

isl::set set::gist(isl::set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map set::identity() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_identity(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_aff set::indicator_function() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_indicator_function(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map set::insert_domain(isl::space domain) const
{
  if (!ptr || domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_insert_domain(copy(), domain.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::intersect(isl::set set2) const
{
  if (!ptr || set2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_intersect(copy(), set2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::intersect_params(isl::set params) const
{
  if (!ptr || params.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_intersect_params(copy(), params.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool set::involves_locals() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_involves_locals(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool set::is_disjoint(const isl::set &set2) const
{
  if (!ptr || set2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_is_disjoint(get(), set2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool set::is_empty() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_is_empty(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool set::is_equal(const isl::set &set2) const
{
  if (!ptr || set2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_is_equal(get(), set2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool set::is_singleton() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_is_singleton(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool set::is_strict_subset(const isl::set &set2) const
{
  if (!ptr || set2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_is_strict_subset(get(), set2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool set::is_subset(const isl::set &set2) const
{
  if (!ptr || set2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_is_subset(get(), set2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool set::is_wrapping() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_is_wrapping(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::set set::lexmax() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_lexmax(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff set::lexmax_pw_multi_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_lexmax_pw_multi_aff(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::lexmin() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_lexmin(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff set::lexmin_pw_multi_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_lexmin_pw_multi_aff(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::lower_bound(isl::multi_pw_aff lower) const
{
  if (!ptr || lower.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_lower_bound_multi_pw_aff(copy(), lower.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::lower_bound(isl::multi_val lower) const
{
  if (!ptr || lower.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_lower_bound_multi_val(copy(), lower.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff set::max_multi_pw_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_max_multi_pw_aff(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val set::max_val(const isl::aff &obj) const
{
  if (!ptr || obj.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_max_val(get(), obj.get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::multi_pw_aff set::min_multi_pw_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_min_multi_pw_aff(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val set::min_val(const isl::aff &obj) const
{
  if (!ptr || obj.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_min_val(get(), obj.get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_params(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_set set::polyhedral_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_polyhedral_hull(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::preimage(isl::multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_preimage_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::preimage(isl::multi_pw_aff mpa) const
{
  if (!ptr || mpa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_preimage_multi_pw_aff(copy(), mpa.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::preimage(isl::pw_multi_aff pma) const
{
  if (!ptr || pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_preimage_pw_multi_aff(copy(), pma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::product(isl::set set2) const
{
  if (!ptr || set2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_product(copy(), set2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::project_out_all_params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_project_out_all_params(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::project_out_param(isl::id id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_project_out_param_id(copy(), id.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::project_out_param(const std::string &id) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->project_out_param(isl::id(ctx(), id));
}

isl::set set::project_out_param(isl::id_list list) const
{
  if (!ptr || list.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_project_out_param_id_list(copy(), list.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_set set::sample() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_sample(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::point set::sample_point() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_sample_point(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::subtract(isl::set set2) const
{
  if (!ptr || set2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_subtract(copy(), set2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map set::translation() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_translation(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::unbind_params(isl::multi_id tuple) const
{
  if (!ptr || tuple.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_unbind_params(copy(), tuple.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map set::unbind_params_insert_domain(isl::multi_id domain) const
{
  if (!ptr || domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_unbind_params_insert_domain(copy(), domain.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::unite(isl::set set2) const
{
  if (!ptr || set2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_union(copy(), set2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::universe(isl::space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = space.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_universe(space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::basic_set set::unshifted_simple_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_unshifted_simple_hull(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::map set::unwrap() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_unwrap(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::upper_bound(isl::multi_pw_aff upper) const
{
  if (!ptr || upper.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_upper_bound_multi_pw_aff(copy(), upper.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::set set::upper_bound(isl::multi_val upper) const
{
  if (!ptr || upper.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_set_upper_bound_multi_val(copy(), upper.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const set &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_set_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_set_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::space
space manage(__isl_take isl_space *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return space(ptr);
}
space manage_copy(__isl_keep isl_space *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_space_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_space_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return space(ptr);
}

space::space()
    : ptr(nullptr) {}

space::space(const space &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_space_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

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

bool space::is_null() const {
  return ptr == nullptr;
}

isl::ctx space::ctx() const {
  return isl::ctx(isl_space_get_ctx(ptr));
}

isl::space space::add_named_tuple(isl::id tuple_id, unsigned int dim) const
{
  if (!ptr || tuple_id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_space_add_named_tuple_id_ui(copy(), tuple_id.release(), dim);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space space::add_named_tuple(const std::string &tuple_id, unsigned int dim) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->add_named_tuple(isl::id(ctx(), tuple_id), dim);
}

isl::space space::add_unnamed_tuple(unsigned int dim) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_space_add_unnamed_tuple_ui(copy(), dim);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space space::curry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_space_curry(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space space::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_space_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space space::flatten_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_space_flatten_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space space::flatten_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_space_flatten_range(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool space::is_equal(const isl::space &space2) const
{
  if (!ptr || space2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_space_is_equal(get(), space2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool space::is_wrapping() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_space_is_wrapping(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::space space::map_from_set() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_space_map_from_set(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space space::params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_space_params(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space space::product(isl::space right) const
{
  if (!ptr || right.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_space_product(copy(), right.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space space::range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_space_range(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space space::range_reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_space_range_reverse(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space space::reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_space_reverse(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space space::uncurry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_space_uncurry(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space space::unit(isl::ctx ctx)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_space_unit(ctx.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space space::unwrap() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_space_unwrap(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space space::wrap() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_space_wrap(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const space &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_space_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_space_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::union_access_info
union_access_info manage(__isl_take isl_union_access_info *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return union_access_info(ptr);
}
union_access_info manage_copy(__isl_keep isl_union_access_info *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_access_info_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_union_access_info_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return union_access_info(ptr);
}

union_access_info::union_access_info()
    : ptr(nullptr) {}

union_access_info::union_access_info(const union_access_info &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_access_info_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

union_access_info::union_access_info(__isl_take isl_union_access_info *ptr)
    : ptr(ptr) {}

union_access_info::union_access_info(isl::union_map sink)
{
  if (sink.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = sink.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_access_info_from_sink(sink.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_access_info_compute_flow(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_access_info union_access_info::set_kill(isl::union_map kill) const
{
  if (!ptr || kill.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_access_info_set_kill(copy(), kill.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_access_info union_access_info::set_may_source(isl::union_map may_source) const
{
  if (!ptr || may_source.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_access_info_set_may_source(copy(), may_source.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_access_info union_access_info::set_must_source(isl::union_map must_source) const
{
  if (!ptr || must_source.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_access_info_set_must_source(copy(), must_source.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_access_info union_access_info::set_schedule(isl::schedule schedule) const
{
  if (!ptr || schedule.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_access_info_set_schedule(copy(), schedule.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_access_info union_access_info::set_schedule_map(isl::union_map schedule_map) const
{
  if (!ptr || schedule_map.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_access_info_set_schedule_map(copy(), schedule_map.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const union_access_info &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_access_info_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_union_access_info_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::union_flow
union_flow manage(__isl_take isl_union_flow *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return union_flow(ptr);
}
union_flow manage_copy(__isl_keep isl_union_flow *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_flow_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_union_flow_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return union_flow(ptr);
}

union_flow::union_flow()
    : ptr(nullptr) {}

union_flow::union_flow(const union_flow &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_flow_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_flow_get_full_may_dependence(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_flow::get_full_may_dependence() const
{
  return full_may_dependence();
}

isl::union_map union_flow::full_must_dependence() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_flow_get_full_must_dependence(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_flow::get_full_must_dependence() const
{
  return full_must_dependence();
}

isl::union_map union_flow::may_dependence() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_flow_get_may_dependence(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_flow::get_may_dependence() const
{
  return may_dependence();
}

isl::union_map union_flow::may_no_source() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_flow_get_may_no_source(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_flow::get_may_no_source() const
{
  return may_no_source();
}

isl::union_map union_flow::must_dependence() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_flow_get_must_dependence(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_flow::get_must_dependence() const
{
  return must_dependence();
}

isl::union_map union_flow::must_no_source() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_flow_get_must_no_source(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_flow::get_must_no_source() const
{
  return must_no_source();
}

inline std::ostream &operator<<(std::ostream &os, const union_flow &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_flow_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_union_flow_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::union_map
union_map manage(__isl_take isl_union_map *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return union_map(ptr);
}
union_map manage_copy(__isl_keep isl_union_map *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_map_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_union_map_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return union_map(ptr);
}

union_map::union_map()
    : ptr(nullptr) {}

union_map::union_map(const union_map &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_map_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

union_map::union_map(__isl_take isl_union_map *ptr)
    : ptr(ptr) {}

union_map::union_map(isl::basic_map bmap)
{
  if (bmap.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = bmap.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_from_basic_map(bmap.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

union_map::union_map(isl::map map)
{
  if (map.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = map.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_from_map(map.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

union_map::union_map(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_affine_hull(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::apply_domain(isl::union_map umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_apply_domain(copy(), umap2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::apply_range(isl::union_map umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_apply_range(copy(), umap2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_map::bind_range(isl::multi_id tuple) const
{
  if (!ptr || tuple.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_bind_range(copy(), tuple.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::coalesce() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_coalesce(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::compute_divs() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_compute_divs(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::curry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_curry(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_map::deltas() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_deltas(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::detect_equalities() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_detect_equalities(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_map::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::domain_factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_domain_factor_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::domain_factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_domain_factor_range(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::domain_map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_domain_map(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff union_map::domain_map_union_pw_multi_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_domain_map_union_pw_multi_aff(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::domain_product(isl::union_map umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_domain_product(copy(), umap2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::empty(isl::ctx ctx)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_empty_ctx(ctx.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::eq_at(isl::multi_union_pw_aff mupa) const
{
  if (!ptr || mupa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_eq_at_multi_union_pw_aff(copy(), mupa.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool union_map::every_map(const std::function<bool(isl::map)> &test) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct test_data {
    std::function<bool(isl::map)> func;
    std::exception_ptr eptr;
  } test_data = { test };
  auto test_lambda = [](isl_map *arg_0, void *arg_1) -> isl_bool {
    auto *data = static_cast<struct test_data *>(arg_1);
    ISL_CPP_TRY {
      auto ret = (data->func)(manage_copy(arg_0));
      return ret ? isl_bool_true : isl_bool_false;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_bool_error;
    }
  };
  auto res = isl_union_map_every_map(get(), test_lambda, &test_data);
  if (test_data.eptr)
    std::rethrow_exception(test_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::map union_map::extract_map(isl::space space) const
{
  if (!ptr || space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_extract_map(get(), space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_factor_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_factor_range(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::fixed_power(isl::val exp) const
{
  if (!ptr || exp.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_fixed_power_val(copy(), exp.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::fixed_power(long exp) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->fixed_power(isl::val(ctx(), exp));
}

void union_map::foreach_map(const std::function<void(isl::map)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct fn_data {
    std::function<void(isl::map)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_map *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_union_map_foreach_map(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return;
}

isl::union_map union_map::from(isl::multi_union_pw_aff mupa)
{
  if (mupa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = mupa.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_from_multi_union_pw_aff(mupa.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::from(isl::union_pw_multi_aff upma)
{
  if (upma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = upma.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_from_union_pw_multi_aff(upma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::from_domain(isl::union_set uset)
{
  if (uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = uset.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_from_domain(uset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::from_domain_and_range(isl::union_set domain, isl::union_set range)
{
  if (domain.is_null() || range.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = domain.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_from_domain_and_range(domain.release(), range.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::from_range(isl::union_set uset)
{
  if (uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = uset.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_from_range(uset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space union_map::space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_get_space(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space union_map::get_space() const
{
  return space();
}

isl::union_map union_map::gist(isl::union_map context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::gist_domain(isl::union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_gist_domain(copy(), uset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::gist_params(isl::set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_gist_params(copy(), set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::gist_range(isl::union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_gist_range(copy(), uset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::intersect(isl::union_map umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_intersect(copy(), umap2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::intersect_domain(isl::space space) const
{
  if (!ptr || space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_intersect_domain_space(copy(), space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::intersect_domain(isl::union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_intersect_domain_union_set(copy(), uset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::intersect_domain_factor_domain(isl::union_map factor) const
{
  if (!ptr || factor.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_intersect_domain_factor_domain(copy(), factor.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::intersect_domain_factor_range(isl::union_map factor) const
{
  if (!ptr || factor.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_intersect_domain_factor_range(copy(), factor.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::intersect_params(isl::set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_intersect_params(copy(), set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::intersect_range(isl::space space) const
{
  if (!ptr || space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_intersect_range_space(copy(), space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::intersect_range(isl::union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_intersect_range_union_set(copy(), uset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::intersect_range_factor_domain(isl::union_map factor) const
{
  if (!ptr || factor.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_intersect_range_factor_domain(copy(), factor.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::intersect_range_factor_range(isl::union_map factor) const
{
  if (!ptr || factor.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_intersect_range_factor_range(copy(), factor.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool union_map::is_bijective() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_is_bijective(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool union_map::is_disjoint(const isl::union_map &umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_is_disjoint(get(), umap2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool union_map::is_empty() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_is_empty(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool union_map::is_equal(const isl::union_map &umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_is_equal(get(), umap2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool union_map::is_injective() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_is_injective(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool union_map::is_single_valued() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_is_single_valued(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool union_map::is_strict_subset(const isl::union_map &umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_is_strict_subset(get(), umap2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool union_map::is_subset(const isl::union_map &umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_is_subset(get(), umap2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool union_map::isa_map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_isa_map(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::union_map union_map::lexmax() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_lexmax(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::lexmin() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_lexmin(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::polyhedral_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_polyhedral_hull(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::preimage_domain(isl::multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_preimage_domain_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::preimage_domain(isl::multi_pw_aff mpa) const
{
  if (!ptr || mpa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_preimage_domain_multi_pw_aff(copy(), mpa.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::preimage_domain(isl::pw_multi_aff pma) const
{
  if (!ptr || pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_preimage_domain_pw_multi_aff(copy(), pma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::preimage_domain(isl::union_pw_multi_aff upma) const
{
  if (!ptr || upma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_preimage_domain_union_pw_multi_aff(copy(), upma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::preimage_range(isl::multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_preimage_range_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::preimage_range(isl::pw_multi_aff pma) const
{
  if (!ptr || pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_preimage_range_pw_multi_aff(copy(), pma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::preimage_range(isl::union_pw_multi_aff upma) const
{
  if (!ptr || upma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_preimage_range_union_pw_multi_aff(copy(), upma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::product(isl::union_map umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_product(copy(), umap2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::project_out_all_params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_project_out_all_params(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_map::range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_range(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::range_factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_range_factor_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::range_factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_range_factor_range(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::range_map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_range_map(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::range_product(isl::union_map umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_range_product(copy(), umap2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::range_reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_range_reverse(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_reverse(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::subtract(isl::union_map umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_subtract(copy(), umap2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::subtract_domain(isl::union_set dom) const
{
  if (!ptr || dom.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_subtract_domain(copy(), dom.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::subtract_range(isl::union_set dom) const
{
  if (!ptr || dom.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_subtract_range(copy(), dom.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::uncurry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_uncurry(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::unite(isl::union_map umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_union(copy(), umap2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::universe() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_universe(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_map::wrap() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_wrap(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_map::zip() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_map_zip(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const union_map &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_map_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_union_map_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::union_pw_aff
union_pw_aff manage(__isl_take isl_union_pw_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return union_pw_aff(ptr);
}
union_pw_aff manage_copy(__isl_keep isl_union_pw_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_pw_aff_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_union_pw_aff_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return union_pw_aff(ptr);
}

union_pw_aff::union_pw_aff()
    : ptr(nullptr) {}

union_pw_aff::union_pw_aff(const union_pw_aff &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_pw_aff_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

union_pw_aff::union_pw_aff(__isl_take isl_union_pw_aff *ptr)
    : ptr(ptr) {}

union_pw_aff::union_pw_aff(isl::aff aff)
{
  if (aff.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = aff.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_from_aff(aff.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

union_pw_aff::union_pw_aff(isl::pw_aff pa)
{
  if (pa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = pa.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_from_pw_aff(pa.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

union_pw_aff::union_pw_aff(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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

isl::union_pw_aff union_pw_aff::add(isl::union_pw_aff upa2) const
{
  if (!ptr || upa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_add(copy(), upa2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_pw_aff::bind(isl::id id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_bind_id(copy(), id.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_pw_aff::bind(const std::string &id) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->bind(isl::id(ctx(), id));
}

isl::union_pw_aff union_pw_aff::coalesce() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_coalesce(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_pw_aff::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space union_pw_aff::space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_get_space(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space union_pw_aff::get_space() const
{
  return space();
}

isl::union_pw_aff union_pw_aff::gist(isl::union_set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_aff union_pw_aff::intersect_domain(isl::space space) const
{
  if (!ptr || space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_intersect_domain_space(copy(), space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_aff union_pw_aff::intersect_domain(isl::union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_intersect_domain_union_set(copy(), uset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_aff union_pw_aff::intersect_domain_wrapped_domain(isl::union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_intersect_domain_wrapped_domain(copy(), uset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_aff union_pw_aff::intersect_domain_wrapped_range(isl::union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_intersect_domain_wrapped_range(copy(), uset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_aff union_pw_aff::intersect_params(isl::set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_intersect_params(copy(), set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_aff union_pw_aff::pullback(isl::union_pw_multi_aff upma) const
{
  if (!ptr || upma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_pullback_union_pw_multi_aff(copy(), upma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_aff union_pw_aff::sub(isl::union_pw_aff upa2) const
{
  if (!ptr || upa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_sub(copy(), upa2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_aff union_pw_aff::subtract_domain(isl::space space) const
{
  if (!ptr || space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_subtract_domain_space(copy(), space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_aff union_pw_aff::subtract_domain(isl::union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_subtract_domain_union_set(copy(), uset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_aff union_pw_aff::union_add(isl::union_pw_aff upa2) const
{
  if (!ptr || upa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_union_add(copy(), upa2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const union_pw_aff &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_pw_aff_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_union_pw_aff_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::union_pw_aff_list
union_pw_aff_list manage(__isl_take isl_union_pw_aff_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return union_pw_aff_list(ptr);
}
union_pw_aff_list manage_copy(__isl_keep isl_union_pw_aff_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_pw_aff_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_union_pw_aff_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return union_pw_aff_list(ptr);
}

union_pw_aff_list::union_pw_aff_list()
    : ptr(nullptr) {}

union_pw_aff_list::union_pw_aff_list(const union_pw_aff_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_pw_aff_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

union_pw_aff_list::union_pw_aff_list(__isl_take isl_union_pw_aff_list *ptr)
    : ptr(ptr) {}

union_pw_aff_list::union_pw_aff_list(isl::ctx ctx, int n)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

union_pw_aff_list::union_pw_aff_list(isl::union_pw_aff el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = el.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_list_from_union_pw_aff(el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_aff_list union_pw_aff_list::clear() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_list_clear(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_aff_list union_pw_aff_list::concat(isl::union_pw_aff_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_aff_list union_pw_aff_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

void union_pw_aff_list::foreach(const std::function<void(isl::union_pw_aff)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct fn_data {
    std::function<void(isl::union_pw_aff)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_union_pw_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_union_pw_aff_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return;
}

isl::union_pw_aff union_pw_aff_list::at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_aff union_pw_aff_list::get_at(int index) const
{
  return at(index);
}

isl::union_pw_aff_list union_pw_aff_list::insert(unsigned int pos, isl::union_pw_aff el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_list_insert(copy(), pos, el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

unsigned union_pw_aff_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_aff_list_size(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

inline std::ostream &operator<<(std::ostream &os, const union_pw_aff_list &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_pw_aff_list_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_union_pw_aff_list_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::union_pw_multi_aff
union_pw_multi_aff manage(__isl_take isl_union_pw_multi_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return union_pw_multi_aff(ptr);
}
union_pw_multi_aff manage_copy(__isl_keep isl_union_pw_multi_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_pw_multi_aff_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_union_pw_multi_aff_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return union_pw_multi_aff(ptr);
}

union_pw_multi_aff::union_pw_multi_aff()
    : ptr(nullptr) {}

union_pw_multi_aff::union_pw_multi_aff(const union_pw_multi_aff &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_pw_multi_aff_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

union_pw_multi_aff::union_pw_multi_aff(__isl_take isl_union_pw_multi_aff *ptr)
    : ptr(ptr) {}

union_pw_multi_aff::union_pw_multi_aff(isl::multi_aff ma)
{
  if (ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ma.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_from_multi_aff(ma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

union_pw_multi_aff::union_pw_multi_aff(isl::pw_multi_aff pma)
{
  if (pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = pma.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_from_pw_multi_aff(pma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

union_pw_multi_aff::union_pw_multi_aff(isl::union_pw_aff upa)
{
  if (upa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = upa.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_from_union_pw_aff(upa.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

union_pw_multi_aff::union_pw_multi_aff(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr || upma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_add(copy(), upma2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::apply(isl::union_pw_multi_aff upma2) const
{
  if (!ptr || upma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_apply_union_pw_multi_aff(copy(), upma2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff union_pw_multi_aff::as_pw_multi_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_as_pw_multi_aff(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::coalesce() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_coalesce(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_pw_multi_aff::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::empty(isl::ctx ctx)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_empty_ctx(ctx.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::pw_multi_aff union_pw_multi_aff::extract_pw_multi_aff(isl::space space) const
{
  if (!ptr || space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_extract_pw_multi_aff(get(), space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::flat_range_product(isl::union_pw_multi_aff upma2) const
{
  if (!ptr || upma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_flat_range_product(copy(), upma2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space union_pw_multi_aff::space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_get_space(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space union_pw_multi_aff::get_space() const
{
  return space();
}

isl::union_pw_multi_aff union_pw_multi_aff::gist(isl::union_set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::intersect_domain(isl::space space) const
{
  if (!ptr || space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_intersect_domain_space(copy(), space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::intersect_domain(isl::union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_intersect_domain_union_set(copy(), uset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::intersect_domain_wrapped_domain(isl::union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_intersect_domain_wrapped_domain(copy(), uset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::intersect_domain_wrapped_range(isl::union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_intersect_domain_wrapped_range(copy(), uset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::intersect_params(isl::set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_intersect_params(copy(), set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool union_pw_multi_aff::involves_locals() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_involves_locals(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool union_pw_multi_aff::isa_pw_multi_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_isa_pw_multi_aff(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool union_pw_multi_aff::plain_is_empty() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_plain_is_empty(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::union_pw_multi_aff union_pw_multi_aff::preimage_domain_wrapped_domain(isl::union_pw_multi_aff upma2) const
{
  if (!ptr || upma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_preimage_domain_wrapped_domain_union_pw_multi_aff(copy(), upma2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::pullback(isl::union_pw_multi_aff upma2) const
{
  if (!ptr || upma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_pullback_union_pw_multi_aff(copy(), upma2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::range_factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_range_factor_domain(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::range_factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_range_factor_range(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::range_product(isl::union_pw_multi_aff upma2) const
{
  if (!ptr || upma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_range_product(copy(), upma2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::sub(isl::union_pw_multi_aff upma2) const
{
  if (!ptr || upma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_sub(copy(), upma2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::subtract_domain(isl::space space) const
{
  if (!ptr || space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_subtract_domain_space(copy(), space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::subtract_domain(isl::union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_subtract_domain_union_set(copy(), uset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_pw_multi_aff union_pw_multi_aff::union_add(isl::union_pw_multi_aff upma2) const
{
  if (!ptr || upma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_union_add(copy(), upma2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const union_pw_multi_aff &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_pw_multi_aff_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_union_pw_multi_aff_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::union_set
union_set manage(__isl_take isl_union_set *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return union_set(ptr);
}
union_set manage_copy(__isl_keep isl_union_set *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_set_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_union_set_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return union_set(ptr);
}

union_set::union_set()
    : ptr(nullptr) {}

union_set::union_set(const union_set &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_set_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

union_set::union_set(__isl_take isl_union_set *ptr)
    : ptr(ptr) {}

union_set::union_set(isl::basic_set bset)
{
  if (bset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = bset.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_from_basic_set(bset.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

union_set::union_set(isl::point pnt)
{
  if (pnt.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = pnt.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_from_point(pnt.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

union_set::union_set(isl::set set)
{
  if (set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = set.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_from_set(set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

union_set::union_set(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_affine_hull(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_set::apply(isl::union_map umap) const
{
  if (!ptr || umap.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_apply(copy(), umap.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_set::coalesce() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_coalesce(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_set::compute_divs() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_compute_divs(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_set::detect_equalities() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_detect_equalities(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_set::empty(isl::ctx ctx)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_empty_ctx(ctx.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool union_set::every_set(const std::function<bool(isl::set)> &test) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct test_data {
    std::function<bool(isl::set)> func;
    std::exception_ptr eptr;
  } test_data = { test };
  auto test_lambda = [](isl_set *arg_0, void *arg_1) -> isl_bool {
    auto *data = static_cast<struct test_data *>(arg_1);
    ISL_CPP_TRY {
      auto ret = (data->func)(manage_copy(arg_0));
      return ret ? isl_bool_true : isl_bool_false;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_bool_error;
    }
  };
  auto res = isl_union_set_every_set(get(), test_lambda, &test_data);
  if (test_data.eptr)
    std::rethrow_exception(test_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::set union_set::extract_set(isl::space space) const
{
  if (!ptr || space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_extract_set(get(), space.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

void union_set::foreach_point(const std::function<void(isl::point)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct fn_data {
    std::function<void(isl::point)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_point *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_union_set_foreach_point(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return;
}

void union_set::foreach_set(const std::function<void(isl::set)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct fn_data {
    std::function<void(isl::set)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_set *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_union_set_foreach_set(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return;
}

isl::space union_set::space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_get_space(get());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::space union_set::get_space() const
{
  return space();
}

isl::union_set union_set::gist(isl::union_set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_set::gist_params(isl::set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_gist_params(copy(), set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_set::identity() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_identity(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_set::intersect(isl::union_set uset2) const
{
  if (!ptr || uset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_intersect(copy(), uset2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_set::intersect_params(isl::set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_intersect_params(copy(), set.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool union_set::is_disjoint(const isl::union_set &uset2) const
{
  if (!ptr || uset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_is_disjoint(get(), uset2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool union_set::is_empty() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_is_empty(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool union_set::is_equal(const isl::union_set &uset2) const
{
  if (!ptr || uset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_is_equal(get(), uset2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool union_set::is_strict_subset(const isl::union_set &uset2) const
{
  if (!ptr || uset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_is_strict_subset(get(), uset2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool union_set::is_subset(const isl::union_set &uset2) const
{
  if (!ptr || uset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_is_subset(get(), uset2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool union_set::isa_set() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_isa_set(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

isl::union_set union_set::lexmax() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_lexmax(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_set::lexmin() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_lexmin(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_set::polyhedral_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_polyhedral_hull(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_set::preimage(isl::multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_preimage_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_set::preimage(isl::pw_multi_aff pma) const
{
  if (!ptr || pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_preimage_pw_multi_aff(copy(), pma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_set::preimage(isl::union_pw_multi_aff upma) const
{
  if (!ptr || upma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_preimage_union_pw_multi_aff(copy(), upma.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::point union_set::sample_point() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_sample_point(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_set::subtract(isl::union_set uset2) const
{
  if (!ptr || uset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_subtract(copy(), uset2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_set::unite(isl::union_set uset2) const
{
  if (!ptr || uset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_union(copy(), uset2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_set::universe() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_universe(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_map union_set::unwrap() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_unwrap(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const union_set &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_set_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_union_set_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::union_set_list
union_set_list manage(__isl_take isl_union_set_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return union_set_list(ptr);
}
union_set_list manage_copy(__isl_keep isl_union_set_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_set_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_union_set_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return union_set_list(ptr);
}

union_set_list::union_set_list()
    : ptr(nullptr) {}

union_set_list::union_set_list(const union_set_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_set_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

union_set_list::union_set_list(__isl_take isl_union_set_list *ptr)
    : ptr(ptr) {}

union_set_list::union_set_list(isl::ctx ctx, int n)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

union_set_list::union_set_list(isl::union_set el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = el.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_list_from_union_set(el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set_list union_set_list::clear() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_list_clear(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set_list union_set_list::concat(isl::union_set_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set_list union_set_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

void union_set_list::foreach(const std::function<void(isl::union_set)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct fn_data {
    std::function<void(isl::union_set)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_union_set *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_union_set_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return;
}

isl::union_set union_set_list::at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::union_set union_set_list::get_at(int index) const
{
  return at(index);
}

isl::union_set_list union_set_list::insert(unsigned int pos, isl::union_set el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_list_insert(copy(), pos, el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

unsigned union_set_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_union_set_list_size(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

inline std::ostream &operator<<(std::ostream &os, const union_set_list &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_union_set_list_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_union_set_list_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::val
val manage(__isl_take isl_val *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return val(ptr);
}
val manage_copy(__isl_keep isl_val *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_val_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_val_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return val(ptr);
}

val::val()
    : ptr(nullptr) {}

val::val(const val &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_val_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

val::val(__isl_take isl_val *ptr)
    : ptr(ptr) {}

val::val(isl::ctx ctx, long i)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_int_from_si(ctx.release(), i);
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

val::val(isl::ctx ctx, const std::string &str)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_abs(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool val::abs_eq(const isl::val &v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_abs_eq(get(), v2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::abs_eq(long v2) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->abs_eq(isl::val(ctx(), v2));
}

isl::val val::add(isl::val v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_add(copy(), v2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val val::add(long v2) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->add(isl::val(ctx(), v2));
}

isl::val val::ceil() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_ceil(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

int val::cmp_si(long i) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_cmp_si(get(), i);
  return res;
}

isl::val val::div(isl::val v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_div(copy(), v2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val val::div(long v2) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->div(isl::val(ctx(), v2));
}

bool val::eq(const isl::val &v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_eq(get(), v2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::eq(long v2) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->eq(isl::val(ctx(), v2));
}

isl::val val::floor() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_floor(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val val::gcd(isl::val v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_gcd(copy(), v2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val val::gcd(long v2) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->gcd(isl::val(ctx(), v2));
}

bool val::ge(const isl::val &v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_ge(get(), v2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::ge(long v2) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->ge(isl::val(ctx(), v2));
}

long val::den_si() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_get_den_si(get());
  return res;
}

long val::get_den_si() const
{
  return den_si();
}

long val::num_si() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_get_num_si(get());
  return res;
}

long val::get_num_si() const
{
  return num_si();
}

bool val::gt(const isl::val &v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_gt(get(), v2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::gt(long v2) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->gt(isl::val(ctx(), v2));
}

isl::val val::infty(isl::ctx ctx)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_infty(ctx.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val val::inv() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_inv(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool val::is_divisible_by(const isl::val &v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_is_divisible_by(get(), v2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::is_divisible_by(long v2) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->is_divisible_by(isl::val(ctx(), v2));
}

bool val::is_infty() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_is_infty(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::is_int() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_is_int(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::is_nan() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_is_nan(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::is_neg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_is_neg(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::is_neginfty() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_is_neginfty(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::is_negone() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_is_negone(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::is_nonneg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_is_nonneg(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::is_nonpos() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_is_nonpos(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::is_one() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_is_one(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::is_pos() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_is_pos(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::is_rat() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_is_rat(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::is_zero() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_is_zero(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::le(const isl::val &v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_le(get(), v2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::le(long v2) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->le(isl::val(ctx(), v2));
}

bool val::lt(const isl::val &v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_lt(get(), v2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::lt(long v2) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->lt(isl::val(ctx(), v2));
}

isl::val val::max(isl::val v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_max(copy(), v2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val val::max(long v2) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->max(isl::val(ctx(), v2));
}

isl::val val::min(isl::val v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_min(copy(), v2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val val::min(long v2) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->min(isl::val(ctx(), v2));
}

isl::val val::mod(isl::val v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_mod(copy(), v2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val val::mod(long v2) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->mod(isl::val(ctx(), v2));
}

isl::val val::mul(isl::val v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_mul(copy(), v2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val val::mul(long v2) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->mul(isl::val(ctx(), v2));
}

isl::val val::nan(isl::ctx ctx)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_nan(ctx.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

bool val::ne(const isl::val &v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_ne(get(), v2.get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

bool val::ne(long v2) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->ne(isl::val(ctx(), v2));
}

isl::val val::neg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_neg(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val val::neginfty(isl::ctx ctx)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_neginfty(ctx.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val val::negone(isl::ctx ctx)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_negone(ctx.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val val::one(isl::ctx ctx)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_one(ctx.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val val::pow2() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_pow2(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

int val::sgn() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_sgn(get());
  return res;
}

isl::val val::sub(isl::val v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_sub(copy(), v2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val val::sub(long v2) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->sub(isl::val(ctx(), v2));
}

isl::val val::trunc() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_trunc(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val val::zero(isl::ctx ctx)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_zero(ctx.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

inline std::ostream &operator<<(std::ostream &os, const val &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_val_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_val_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}

// implementations for isl::val_list
val_list manage(__isl_take isl_val_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return val_list(ptr);
}
val_list manage_copy(__isl_keep isl_val_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_val_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = isl_val_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(saved_ctx);
  return val_list(ptr);
}

val_list::val_list()
    : ptr(nullptr) {}

val_list::val_list(const val_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_val_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  ptr = obj.copy();
  if (!ptr)
    exception::throw_last_error(saved_ctx);
}

val_list::val_list(__isl_take isl_val_list *ptr)
    : ptr(ptr) {}

val_list::val_list(isl::ctx ctx, int n)
{
  auto saved_ctx = ctx;
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(saved_ctx);
  ptr = res;
}

val_list::val_list(isl::val el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = el.ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_list_from_val(el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
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
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val_list val_list::add(long el) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->add(isl::val(ctx(), el));
}

isl::val_list val_list::clear() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_list_clear(copy());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val_list val_list::concat(isl::val_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val_list val_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

void val_list::foreach(const std::function<void(isl::val)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  struct fn_data {
    std::function<void(isl::val)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_val *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_val_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return;
}

isl::val val_list::at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val val_list::get_at(int index) const
{
  return at(index);
}

isl::val_list val_list::insert(unsigned int pos, isl::val el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_list_insert(copy(), pos, el.release());
  if (!res)
    exception::throw_last_error(saved_ctx);
  return manage(res);
}

isl::val_list val_list::insert(unsigned int pos, long el) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return this->insert(pos, isl::val(ctx(), el));
}

unsigned val_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = ctx();
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  auto res = isl_val_list_size(get());
  if (res < 0)
    exception::throw_last_error(saved_ctx);
  return res;
}

inline std::ostream &operator<<(std::ostream &os, const val_list &obj)
{
  if (!obj.get())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto saved_ctx = isl_val_list_get_ctx(obj.get());
  options_scoped_set_on_error saved_on_error(saved_ctx, exception::on_error);
  char *str = isl_val_list_to_str(obj.get());
  if (!str)
    exception::throw_last_error(saved_ctx);
  os << str;
  free(str);
  return os;
}
} // namespace isl

#endif /* ISL_CPP */
