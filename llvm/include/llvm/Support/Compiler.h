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

// The VISIBILITY_HIDDEN macro, used for marking classes with the GCC-specific
// visibility("hidden") attribute.
#if (__GNUC__ >= 4) && !defined(__MINGW32__) && !defined(__CYGWIN__)
#define VISIBILITY_HIDDEN __attribute__ ((visibility("hidden")))
#else
#define VISIBILITY_HIDDEN
#endif

#if (__GNUC__ >= 4 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#define ATTRIBUTE_USED __attribute__((__used__))
#else
#define ATTRIBUTE_USED
#endif

#if (__GNUC__ >= 4)
#define BUILTIN_EXPECT(EXPR, VALUE) __builtin_expect((EXPR), (VALUE))
#else
#define BUILTIN_EXPECT(EXPR, VALUE) (EXPR)
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

// DISABLE_INLINE - On compilers where we have a directive to do so, mark a
// method "not for inlining".
#if (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#define DISABLE_INLINE __attribute__((noinline))
#else
#define DISABLE_INLINE
#endif

#ifdef __GNUC__
#define NORETURN __attribute__((noreturn))
#else
#define NORETURN
#endif

#endif
