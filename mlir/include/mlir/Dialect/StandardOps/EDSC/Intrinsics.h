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
using std_divis = ValueBuilder<SignedDivIOp>;
using std_diviu = ValueBuilder<UnsignedDivIOp>;
using std_dim = ValueBuilder<DimOp>;
using std_extract_element = ValueBuilder<ExtractElementOp>;
using std_fpext = ValueBuilder<FPExtOp>;
using std_fptrunc = ValueBuilder<FPTruncOp>;
using std_im = ValueBuilder<ImOp>;
using std_index_cast = ValueBuilder<IndexCastOp>;
using std_muli = ValueBuilder<MulIOp>;
using std_mulf = ValueBuilder<MulFOp>;
using std_memref_cast = ValueBuilder<MemRefCastOp>;
using std_re = ValueBuilder<ReOp>;
using std_ret = OperationBuilder<ReturnOp>;
using std_rsqrt = ValueBuilder<RsqrtOp>;
using std_select = ValueBuilder<SelectOp>;
using std_load = ValueBuilder<LoadOp>;
using std_splat = ValueBuilder<SplatOp>;
using std_store = OperationBuilder<StoreOp>;
using std_subf = ValueBuilder<SubFOp>;
using std_subi = ValueBuilder<SubIOp>;
using std_sub_view = ValueBuilder<SubViewOp>;
using std_tanh = ValueBuilder<TanhOp>;
using std_tensor_load = ValueBuilder<TensorLoadOp>;
using std_tensor_store = OperationBuilder<TensorStoreOp>;
using std_view = ValueBuilder<ViewOp>;
using std_zero_extendi = ValueBuilder<ZeroExtendIOp>;
using std_sign_extendi = ValueBuilder<SignExtendIOp>;

/// Branches into `block` with `operands`.
BranchOp std_br(Block *block, ValueRange operands);

/// Branches into `trueBranch` with `trueOperands` if `cond` evaluates to `true`
/// or to `falseBranch` and `falseOperand` if `cond` evaluates to `false`.
CondBranchOp std_cond_br(Value cond, Block *trueBranch, ValueRange trueOperands,
                         Block *falseBranch, ValueRange falseOperands);

/// Provide an index notation around sdt_load and std_store.
using StdIndexedValue =
    TemplatedIndexedValue<intrinsics::std_load, intrinsics::std_store>;
} // namespace intrinsics
} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_STANDARDOPS_EDSC_INTRINSICS_H_
