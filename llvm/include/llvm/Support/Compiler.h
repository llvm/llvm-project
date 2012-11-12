//===-- llvm/Support/Compiler.h - Compiler abstraction support --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines several macros, based on the current compiler.  This allows
// use of compiler-specific features in a way that remains portable.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_COMPILER_H
#define LLVM_SUPPORT_COMPILER_H

#ifndef __has_feature
# define __has_feature(x) 0
#endif

/// LLVM_HAS_RVALUE_REFERENCES - Does the compiler provide r-value references?
/// This implies that <utility> provides the one-argument std::move;  it
/// does not imply the existence of any other C++ library features.
#if (__has_feature(cxx_rvalue_references)   \
     || defined(__GXX_EXPERIMENTAL_CXX0X__) \
     || (defined(_MSC_VER) && _MSC_VER >= 1600))
#define LLVM_USE_RVALUE_REFERENCES 1
#else
#define LLVM_USE_RVALUE_REFERENCES 0
#endif

/// llvm_move - Expands to ::std::move if the compiler supports
/// r-value references; otherwise, expands to the argument.
#if LLVM_USE_RVALUE_REFERENCES
#define llvm_move(value) (::std::move(value))
#else
#define llvm_move(value) (value)
#endif

/// LLVM_DELETED_FUNCTION - Expands to = delete if the compiler supports it.
/// Use to mark functions as uncallable. Member functions with this should
/// be declared private so that some behavior is kept in C++03 mode.
///
/// class DontCopy {
/// private:
///   DontCopy(const DontCopy&) LLVM_DELETED_FUNCTION;
///   DontCopy &operator =(const DontCopy&) LLVM_DELETED_FUNCTION;
/// public:
///   ...
/// };
#if (__has_feature(cxx_deleted_functions) \
     || defined(__GXX_EXPERIMENTAL_CXX0X__))
     // No version of MSVC currently supports this.
#define LLVM_DELETED_FUNCTION = delete
#else
#define LLVM_DELETED_FUNCTION
#endif

/// LLVM_FINAL - Expands to 'final' if the compiler supports it.
/// Use to mark classes or virtual methods as final.
#if (__has_feature(cxx_override_control))
#define LLVM_FINAL final
#else
#define LLVM_FINAL
#endif

/// LLVM_OVERRIDE - Expands to 'override' if the compiler supports it.
/// Use to mark virtual methods as overriding a base class method.
#if (__has_feature(cxx_override_control))
#define LLVM_OVERRIDE override
#else
#define LLVM_OVERRIDE
#endif

/// LLVM_LIBRARY_VISIBILITY - If a class marked with this attribute is linked
/// into a shared library, then the class should be private to the library and
/// not accessible from outside it.  Can also be used to mark variables and
/// functions, making them private to any shared library they are linked into.
#if (__GNUC__ >= 4) && !defined(__MINGW32__) && !defined(__CYGWIN__)
#define LLVM_LIBRARY_VISIBILITY __attribute__ ((visibility("hidden")))
#else
#define LLVM_LIBRARY_VISIBILITY
#endif

#if (__GNUC__ >= 4 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#define LLVM_ATTRIBUTE_USED __attribute__((__used__))
#else
#define LLVM_ATTRIBUTE_USED
#endif

// Some compilers warn about unused functions. When a function is sometimes
// used or not depending on build settings (e.g. a function only called from
// within "assert"), this attribute can be used to suppress such warnings.
//
// However, it shouldn't be used for unused *variables*, as those have a much
// more portable solution:
//   (void)unused_var_name;
// Prefer cast-to-void wherever it is sufficient.
#if (__GNUC__ >= 4 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#define LLVM_ATTRIBUTE_UNUSED __attribute__((__unused__))
#else
#define LLVM_ATTRIBUTE_UNUSED
#endif

#if (__GNUC__ >= 4) && !defined(__MINGW32__) && !defined(__CYGWIN__)
#define LLVM_ATTRIBUTE_WEAK __attribute__((__weak__))
#else
#define LLVM_ATTRIBUTE_WEAK
#endif

#ifdef __GNUC__ // aka 'CONST' but following LLVM Conventions.
#define LLVM_READNONE __attribute__((__const__))
#else
#define LLVM_READNONE
#endif

#ifdef __GNUC__  // aka 'PURE' but following LLVM Conventions.
#define LLVM_READONLY __attribute__((__pure__))
#else
#define LLVM_READONLY
#endif

#if (__GNUC__ >= 4)
#define LLVM_LIKELY(EXPR) __builtin_expect((bool)(EXPR), true)
#define LLVM_UNLIKELY(EXPR) __builtin_expect((bool)(EXPR), false)
#else
#define LLVM_LIKELY(EXPR) (EXPR)
#define LLVM_UNLIKELY(EXPR) (EXPR)
#endif


// C++ doesn't support 'extern template' of template specializations.  GCC does,
// but requires __extension__ before it.  In the header, use this:
//   EXTERN_TEMPLATE_INSTANTIATION(class foo<bar>);
// in the .cpp file, use this:
//   TEMPLATE_INSTANTIATION(class foo<bar>);
#ifdef __GNUC__
#define EXTERN_TEMPLATE_INSTANTIATION(X) __extension__ extern template X
#define TEMPLATE_INSTANTIATION(X) template X
#else
#define EXTERN_TEMPLATE_INSTANTIATION(X)
#define TEMPLATE_INSTANTIATION(X)
#endif

// LLVM_ATTRIBUTE_NOINLINE - On compilers where we have a directive to do so,
// mark a method "not for inlining".
#if (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#define LLVM_ATTRIBUTE_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define LLVM_ATTRIBUTE_NOINLINE __declspec(noinline)
#else
#define LLVM_ATTRIBUTE_NOINLINE
#endif

// LLVM_ATTRIBUTE_ALWAYS_INLINE - On compilers where we have a directive to do
// so, mark a method "always inline" because it is performance sensitive. GCC
// 3.4 supported this but is buggy in various cases and produces unimplemented
// errors, just use it in GCC 4.0 and later.
#if __GNUC__ > 3
#define LLVM_ATTRIBUTE_ALWAYS_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define LLVM_ATTRIBUTE_ALWAYS_INLINE __forceinline
#else
#define LLVM_ATTRIBUTE_ALWAYS_INLINE
#endif


#ifdef __GNUC__
#define LLVM_ATTRIBUTE_NORETURN __attribute__((noreturn))
#elif defined(_MSC_VER)
#define LLVM_ATTRIBUTE_NORETURN __declspec(noreturn)
#else
#define LLVM_ATTRIBUTE_NORETURN
#endif

// LLVM_EXTENSION - Support compilers where we have a keyword to suppress
// pedantic diagnostics.
#ifdef __GNUC__
#define LLVM_EXTENSION __extension__
#else
#define LLVM_EXTENSION
#endif

// LLVM_ATTRIBUTE_DEPRECATED(decl, "message")
#if __has_feature(attribute_deprecated_with_message)
# define LLVM_ATTRIBUTE_DEPRECATED(decl, message) \
  decl __attribute__((deprecated(message)))
#elif defined(__GNUC__)
# define LLVM_ATTRIBUTE_DEPRECATED(decl, message) \
  decl __attribute__((deprecated))
#elif defined(_MSC_VER)
# define LLVM_ATTRIBUTE_DEPRECATED(decl, message) \
  __declspec(deprecated(message)) decl
#else
# define LLVM_ATTRIBUTE_DEPRECATED(decl, message) \
  decl
#endif

// LLVM_BUILTIN_UNREACHABLE - On compilers which support it, expands
// to an expression which states that it is undefined behavior for the
// compiler to reach this point.  Otherwise is not defined.
#if defined(__clang__) || (__GNUC__ > 4) \
 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5)
# define LLVM_BUILTIN_UNREACHABLE __builtin_unreachable()
#endif

// LLVM_BUILTIN_TRAP - On compilers which support it, expands to an expression
// which causes the program to exit abnormally.
#if defined(__clang__) || (__GNUC__ > 4) \
 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3)
# define LLVM_BUILTIN_TRAP __builtin_trap()
#else
# define LLVM_BUILTIN_TRAP *(volatile int*)0x11 = 0
#endif

#endif
