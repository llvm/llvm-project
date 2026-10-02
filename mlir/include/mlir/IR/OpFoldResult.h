//===- OpFoldResult.h - Results of an operation fold ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines OpFoldResult, the result of a fold of an operation with
// exactly one result.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_OPFOLDRESULT_H
#define MLIR_IR_OPFOLDRESULT_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/Compiler.h"

namespace mlir {

/// This class represents a single result from folding an operation.
class OpFoldResult : public PointerUnion<Attribute, Value> {
  using PointerUnion<Attribute, Value>::PointerUnion;

public:
  LLVM_DUMP_METHOD void dump() const;

  MLIRContext *getContext() const {
    PointerUnion pu = *this;
    return isa<Attribute>(pu) ? cast<Attribute>(pu).getContext()
                              : cast<Value>(pu).getContext();
  }
};

// Temporarily exit the MLIR namespace to add casting support as later code in
// this uses it. The CastInfo must come after the OpFoldResult definition and
// before any cast function calls depending on CastInfo.

} // namespace mlir

namespace llvm {

// Allow llvm::cast style functions.
template <typename To>
struct CastInfo<To, mlir::OpFoldResult>
    : public CastInfo<To, mlir::OpFoldResult::PointerUnion> {};

template <typename To>
struct CastInfo<To, const mlir::OpFoldResult>
    : public CastInfo<To, const mlir::OpFoldResult::PointerUnion> {};

} // namespace llvm

namespace mlir {

/// Allow printing to a stream.
raw_ostream &operator<<(raw_ostream &os, OpFoldResult ofr);

} // namespace mlir

#endif // MLIR_IR_OPFOLDRESULT_H
