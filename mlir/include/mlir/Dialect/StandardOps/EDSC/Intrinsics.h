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
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace edsc {
namespace intrinsics {

using std_addi = ValueBuilder<AddIOp>;
using std_addf = ValueBuilder<AddFOp>;
using std_call = OperationBuilder<CallOp>;
using std_constant = ValueBuilder<ConstantOp>;
using std_constant_float = ValueBuilder<ConstantFloatOp>;
using std_constant_index = ValueBuilder<ConstantIndexOp>;
using std_constant_int = ValueBuilder<ConstantIntOp>;
using std_divis = ValueBuilder<SignedDivIOp>;
using std_diviu = ValueBuilder<UnsignedDivIOp>;
using std_fpext = ValueBuilder<FPExtOp>;
using std_fptrunc = ValueBuilder<FPTruncOp>;
using std_index_cast = ValueBuilder<IndexCastOp>;
using std_muli = ValueBuilder<MulIOp>;
using std_mulf = ValueBuilder<MulFOp>;
using std_ret = OperationBuilder<ReturnOp>;
using std_select = ValueBuilder<SelectOp>;
using std_sign_extendi = ValueBuilder<SignExtendIOp>;
using std_splat = ValueBuilder<SplatOp>;
using std_subf = ValueBuilder<SubFOp>;
using std_subi = ValueBuilder<SubIOp>;
using std_zero_extendi = ValueBuilder<ZeroExtendIOp>;
using tensor_extract = ValueBuilder<tensor::ExtractOp>;

template <int N>
struct SExtiValueBuilder : public ValueBuilder<SignExtendIOp> {
  using ValueBuilder<SignExtendIOp>::ValueBuilder;
  template <typename... Args>
  SExtiValueBuilder(Args... args)
      : ValueBuilder<SignExtendIOp>(ScopedContext::getBuilderRef().getI32Type(),
                                    args...) {}
};

using std_sexti32 = SExtiValueBuilder<32>;

template <CmpFPredicate Predicate>
struct CmpFValueBuilder : public ValueBuilder<CmpFOp> {
  using ValueBuilder<CmpFOp>::ValueBuilder;
  template <typename... Args>
  CmpFValueBuilder(Args... args) : ValueBuilder<CmpFOp>(Predicate, args...) {}
};

using std_cmpf_ogt = CmpFValueBuilder<CmpFPredicate::OGT>;
using std_cmpf_olt = CmpFValueBuilder<CmpFPredicate::OLT>;

template <CmpIPredicate Predicate>
struct CmpIValueBuilder : public ValueBuilder<CmpIOp> {
  using ValueBuilder<CmpIOp>::ValueBuilder;
  template <typename... Args>
  CmpIValueBuilder(Args... args) : ValueBuilder<CmpIOp>(Predicate, args...) {}
};

using std_cmpi_sgt = CmpIValueBuilder<CmpIPredicate::sgt>;

/// Branches into `block` with `operands`.
BranchOp std_br(Block *block, ValueRange operands);

/// Branches into `trueBranch` with `trueOperands` if `cond` evaluates to `true`
/// or to `falseBranch` and `falseOperand` if `cond` evaluates to `false`.
CondBranchOp std_cond_br(Value cond, Block *trueBranch, ValueRange trueOperands,
                         Block *falseBranch, ValueRange falseOperands);
} // namespace intrinsics
} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_STANDARDOPS_EDSC_INTRINSICS_H_
