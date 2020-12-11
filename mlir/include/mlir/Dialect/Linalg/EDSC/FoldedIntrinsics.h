//===- FoldedIntrinsics.h - MLIR EDSC Intrinsics for Linalg -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_LINALG_EDSC_FOLDEDINTRINSICS_H_
#define MLIR_DIALECT_LINALG_EDSC_FOLDEDINTRINSICS_H_

#include "mlir/Dialect/Linalg/EDSC/Builders.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"

#include "mlir/Transforms/FoldUtils.h"

namespace mlir {
namespace edsc {
namespace intrinsics {

template <typename Op>
struct FoldedValueBuilder {
  // Builder-based
  template <typename... Args>
  FoldedValueBuilder(OperationFolder *folder, Args... args) {
    value = folder ? folder->create<Op>(ScopedContext::getBuilderRef(),
                                        ScopedContext::getLocation(), args...)
                   : ScopedContext::getBuilderRef().create<Op>(
                         ScopedContext::getLocation(), args...);
  }

  operator Value() { return value; }
  Value value;
};

using folded_std_constant_index = FoldedValueBuilder<ConstantIndexOp>;
using folded_std_constant_float = FoldedValueBuilder<ConstantFloatOp>;
using folded_std_constant_int = FoldedValueBuilder<ConstantIntOp>;
using folded_std_constant = FoldedValueBuilder<ConstantOp>;
using folded_std_dim = FoldedValueBuilder<DimOp>;
using folded_std_muli = FoldedValueBuilder<MulIOp>;
using folded_std_addi = FoldedValueBuilder<AddIOp>;
using folded_std_addf = FoldedValueBuilder<AddFOp>;
using folded_std_alloc = FoldedValueBuilder<AllocOp>;
using folded_std_constant = FoldedValueBuilder<ConstantOp>;
using folded_std_constant_float = FoldedValueBuilder<ConstantFloatOp>;
using folded_std_constant_index = FoldedValueBuilder<ConstantIndexOp>;
using folded_std_constant_int = FoldedValueBuilder<ConstantIntOp>;
using folded_std_dim = FoldedValueBuilder<DimOp>;
using folded_std_extract_element = FoldedValueBuilder<ExtractElementOp>;
using folded_std_index_cast = FoldedValueBuilder<IndexCastOp>;
using folded_std_muli = FoldedValueBuilder<MulIOp>;
using folded_std_mulf = FoldedValueBuilder<MulFOp>;
using folded_std_memref_cast = FoldedValueBuilder<MemRefCastOp>;
using folded_std_select = FoldedValueBuilder<SelectOp>;
using folded_std_load = FoldedValueBuilder<LoadOp>;
using folded_std_subi = FoldedValueBuilder<SubIOp>;
using folded_std_sub_view = FoldedValueBuilder<SubViewOp>;
using folded_std_tanh = FoldedValueBuilder<TanhOp>;
using folded_std_tensor_load = FoldedValueBuilder<TensorLoadOp>;
using folded_std_view = FoldedValueBuilder<ViewOp>;
using folded_std_zero_extendi = FoldedValueBuilder<ZeroExtendIOp>;
using folded_std_sign_extendi = FoldedValueBuilder<SignExtendIOp>;
} // namespace intrinsics
} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_EDSC_FOLDEDINTRINSICS_H_
