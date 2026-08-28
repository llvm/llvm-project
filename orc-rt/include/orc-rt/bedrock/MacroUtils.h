//===- MacroUtils.h - General-purpose preprocessor helpers ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Small preprocessor-utility macros not specific to any particular orc-rt
// subsystem.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_MACROUTILS_H
#define ORC_RT_MACROUTILS_H

#define ORC_RT_DETAIL_DEPAREN_HELPER(...) __VA_ARGS__

/// Strip a single layer of outer parentheses from a token sequence.
///
/// Useful for passing a parenthesized comma-separated list (e.g. a list of
/// types) as a single argument to a function-like macro, then unwrapping it
/// at the use site:
///
///     #define MY_MACRO(Types) some_template<ORC_RT_DEPAREN(Types)>
///     MY_MACRO((int, double, char))
///         -> some_template<int, double, char>
///
#define ORC_RT_DEPAREN(X) ORC_RT_DETAIL_DEPAREN_HELPER X

#endif // ORC_RT_MACROUTILS_H
