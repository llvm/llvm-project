//===- OpAsmSupport.h - OpAsm Interface Utilities ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines various classes and utilites for
// OpAsm{Dialect,Type,Attr,Op}Interface
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_OPASMSUPPORT_H_
#define MLIR_IR_OPASMSUPPORT_H_

#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"

namespace mlir {

//===--------------------------------------------------------------------===//
// Utilities used by OpAsm{Dialect,Op,Type,Attr}Interface.
//===--------------------------------------------------------------------===//

/// A functor used to set the name of the result. See 'getAsmResultNames' below
/// for more details.
using OpAsmSetNameFn = function_ref<void(StringRef)>;

/// A functor used to set the name of the start of a result group of an
/// operation. See 'getAsmResultNames' below for more details.
using OpAsmSetValueNameFn = function_ref<void(Value, StringRef)>;

/// A functor used to set the name of blocks in regions directly nested under
/// an operation.
using OpAsmSetBlockNameFn = function_ref<void(Block *, StringRef)>;

/// Holds the result of `OpAsm{Dialect,Attr,Type}Interface::getAlias` hook call.
enum class OpAsmAliasResult {
  /// The object (type or attribute) is not supported by the hook
  /// and an alias was not provided.
  NoAlias,
  /// An alias was provided, but it might be overriden by other hook.
  OverridableAlias,
  /// An alias was provided and it should be used
  /// (no other hooks will be checked).
  FinalAlias
};

} // namespace mlir

#endif // MLIR_IR_OPASMSUPPORT_H_
