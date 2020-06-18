//===- Intrinsics.h - MLIR EDSC Intrinsics for StandardOps ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_STANDARDOPS_EDSC_INTRINSICS_H_
#define MLIR_DIALECT_STANDARDOPS_EDSC_INTRINSICS_H_

#include "mlir/Dialect/StandardOps/EDSC/Builders.h"

namespace mlir {
namespace edsc {
namespace intrinsics {

using std_addi = ValueBuilder<AddIOp>;
using std_addf = ValueBuilder<AddFOp>;
using std_alloc = ValueBuilder<AllocOp>;
using std_alloca = ValueBuilder<AllocaOp>;
using std_call = OperationBuilder<CallOp>;
using std_create_complex = ValueBuilder<CreateComplexOp>;
using std_constant = ValueBuilder<ConstantOp>;
using std_constant_float = ValueBuilder<ConstantFloatOp>;
using std_constant_index = ValueBuilder<ConstantIndexOp>;
using std_constant_int = ValueBuilder<ConstantIntOp>;
using std_dealloc = OperationBuilder<DeallocOp>;
using std_dim = ValueBuilder<DimOp>;
using std_extract_element = ValueBuilder<ExtractElementOp>;
using std_im = ValueBuilder<ImOp>;
using std_index_cast = ValueBuilder<IndexCastOp>;
using std_muli = ValueBuilder<MulIOp>;
using std_mulf = ValueBuilder<MulFOp>;
using std_memref_cast = ValueBuilder<MemRefCastOp>;
using std_re = ValueBuilder<ReOp>;
using std_ret = OperationBuilder<ReturnOp>;
using std_select = ValueBuilder<SelectOp>;
using std_load = ValueBuilder<LoadOp>;
using std_splat = ValueBuilder<SplatOp>;
using std_store = OperationBuilder<StoreOp>;
using std_subi = ValueBuilder<SubIOp>;
using std_sub_view = ValueBuilder<SubViewOp>;
using std_tanh = ValueBuilder<TanhOp>;
using std_tensor_load = ValueBuilder<TensorLoadOp>;
using std_tensor_store = OperationBuilder<TensorStoreOp>;
using std_view = ValueBuilder<ViewOp>;
using std_zero_extendi = ValueBuilder<ZeroExtendIOp>;
using std_sign_extendi = ValueBuilder<SignExtendIOp>;

/// Branches into the mlir::Block* captured by BlockHandle `b` with `operands`.
///
/// Prerequisites:
///   All Handles have already captured previously constructed IR objects.
BranchOp std_br(BlockHandle bh, ValueRange operands);
BranchOp std_br(Block *block, ValueRange operands);

/// Creates a new mlir::Block* and branches to it from the current block.
/// Argument types are specified by `operands`.
/// Captures the new block in `bh` and the actual `operands` in `captures`. To
/// insert the new mlir::Block*, a local ScopedContext is constructed and
/// released to the current block. The branch operation is then added to the
/// new block.
///
/// Prerequisites:
///   `b` has not yet captured an mlir::Block*.
///   No `captures` have captured any mlir::Value.
///   All `operands` have already captured an mlir::Value
///   captures.size() == operands.size()
///   captures and operands are pairwise of the same type.
BranchOp std_br(BlockHandle *bh, ArrayRef<Type> types,
                MutableArrayRef<Value> captures, ValueRange operands);

/// Branches into the mlir::Block* captured by BlockHandle `trueBranch` with
/// `trueOperands` if `cond` evaluates to `true` (resp. `falseBranch` and
/// `falseOperand` if `cond` evaluates to `false`).
///
/// Prerequisites:
///   All Handles have captured previously constructed IR objects.
CondBranchOp std_cond_br(Value cond, BlockHandle trueBranch,
                         ValueRange trueOperands, BlockHandle falseBranch,
                         ValueRange falseOperands);
CondBranchOp std_cond_br(Value cond, Block *trueBranch, ValueRange trueOperands,
                         Block *falseBranch, ValueRange falseOperands);

/// Eagerly creates new mlir::Block* with argument types specified by
/// `trueOperands`/`falseOperands`.
/// Captures the new blocks in `trueBranch`/`falseBranch` and the arguments in
/// `trueCaptures/falseCaptures`.
/// To insert the new mlir::Block*, a local ScopedContext is constructed and
/// released. The branch operation is then added in the original location and
/// targeting the eagerly constructed blocks.
///
/// Prerequisites:
///   `trueBranch`/`falseBranch` has not yet captured an mlir::Block*.
///   No `trueCaptures`/`falseCaptures` have captured any mlir::Value.
///   All `trueOperands`/`trueOperands` have already captured an mlir::Value
///   `trueCaptures`.size() == `trueOperands`.size()
///   `falseCaptures`.size() == `falseOperands`.size()
///   `trueCaptures` and `trueOperands` are pairwise of the same type
///   `falseCaptures` and `falseOperands` are pairwise of the same type.
CondBranchOp
std_cond_br(Value cond, BlockHandle *trueBranch, ArrayRef<Type> trueTypes,
            MutableArrayRef<Value> trueCaptures, ValueRange trueOperands,
            BlockHandle *falseBranch, ArrayRef<Type> falseTypes,
            MutableArrayRef<Value> falseCaptures, ValueRange falseOperands);

/// Provide an index notation around sdt_load and std_store.
using StdIndexedValue =
    TemplatedIndexedValue<intrinsics::std_load, intrinsics::std_store>;
} // namespace intrinsics
} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_STANDARDOPS_EDSC_INTRINSICS_H_
