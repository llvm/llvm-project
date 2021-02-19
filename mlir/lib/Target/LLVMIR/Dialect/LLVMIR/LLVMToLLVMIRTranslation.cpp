//===- LLVMToLLVMIRTranslation.cpp - Translate LLVM dialect to LLVM IR ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVM dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/MatrixBuilder.h"
#include "llvm/IR/Operator.h"

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::getLLVMConstant;

#include "mlir/Dialect/LLVMIR/LLVMConversionEnumsToLLVM.inc"

/// Convert MLIR integer comparison predicate to LLVM IR comparison predicate.
static llvm::CmpInst::Predicate getLLVMCmpPredicate(ICmpPredicate p) {
  switch (p) {
  case LLVM::ICmpPredicate::eq:
    return llvm::CmpInst::Predicate::ICMP_EQ;
  case LLVM::ICmpPredicate::ne:
    return llvm::CmpInst::Predicate::ICMP_NE;
  case LLVM::ICmpPredicate::slt:
    return llvm::CmpInst::Predicate::ICMP_SLT;
  case LLVM::ICmpPredicate::sle:
    return llvm::CmpInst::Predicate::ICMP_SLE;
  case LLVM::ICmpPredicate::sgt:
    return llvm::CmpInst::Predicate::ICMP_SGT;
  case LLVM::ICmpPredicate::sge:
    return llvm::CmpInst::Predicate::ICMP_SGE;
  case LLVM::ICmpPredicate::ult:
    return llvm::CmpInst::Predicate::ICMP_ULT;
  case LLVM::ICmpPredicate::ule:
    return llvm::CmpInst::Predicate::ICMP_ULE;
  case LLVM::ICmpPredicate::ugt:
    return llvm::CmpInst::Predicate::ICMP_UGT;
  case LLVM::ICmpPredicate::uge:
    return llvm::CmpInst::Predicate::ICMP_UGE;
  }
  llvm_unreachable("incorrect comparison predicate");
}

static llvm::CmpInst::Predicate getLLVMCmpPredicate(FCmpPredicate p) {
  switch (p) {
  case LLVM::FCmpPredicate::_false:
    return llvm::CmpInst::Predicate::FCMP_FALSE;
  case LLVM::FCmpPredicate::oeq:
    return llvm::CmpInst::Predicate::FCMP_OEQ;
  case LLVM::FCmpPredicate::ogt:
    return llvm::CmpInst::Predicate::FCMP_OGT;
  case LLVM::FCmpPredicate::oge:
    return llvm::CmpInst::Predicate::FCMP_OGE;
  case LLVM::FCmpPredicate::olt:
    return llvm::CmpInst::Predicate::FCMP_OLT;
  case LLVM::FCmpPredicate::ole:
    return llvm::CmpInst::Predicate::FCMP_OLE;
  case LLVM::FCmpPredicate::one:
    return llvm::CmpInst::Predicate::FCMP_ONE;
  case LLVM::FCmpPredicate::ord:
    return llvm::CmpInst::Predicate::FCMP_ORD;
  case LLVM::FCmpPredicate::ueq:
    return llvm::CmpInst::Predicate::FCMP_UEQ;
  case LLVM::FCmpPredicate::ugt:
    return llvm::CmpInst::Predicate::FCMP_UGT;
  case LLVM::FCmpPredicate::uge:
    return llvm::CmpInst::Predicate::FCMP_UGE;
  case LLVM::FCmpPredicate::ult:
    return llvm::CmpInst::Predicate::FCMP_ULT;
  case LLVM::FCmpPredicate::ule:
    return llvm::CmpInst::Predicate::FCMP_ULE;
  case LLVM::FCmpPredicate::une:
    return llvm::CmpInst::Predicate::FCMP_UNE;
  case LLVM::FCmpPredicate::uno:
    return llvm::CmpInst::Predicate::FCMP_UNO;
  case LLVM::FCmpPredicate::_true:
    return llvm::CmpInst::Predicate::FCMP_TRUE;
  }
  llvm_unreachable("incorrect comparison predicate");
}

static llvm::AtomicRMWInst::BinOp getLLVMAtomicBinOp(AtomicBinOp op) {
  switch (op) {
  case LLVM::AtomicBinOp::xchg:
    return llvm::AtomicRMWInst::BinOp::Xchg;
  case LLVM::AtomicBinOp::add:
    return llvm::AtomicRMWInst::BinOp::Add;
  case LLVM::AtomicBinOp::sub:
    return llvm::AtomicRMWInst::BinOp::Sub;
  case LLVM::AtomicBinOp::_and:
    return llvm::AtomicRMWInst::BinOp::And;
  case LLVM::AtomicBinOp::nand:
    return llvm::AtomicRMWInst::BinOp::Nand;
  case LLVM::AtomicBinOp::_or:
    return llvm::AtomicRMWInst::BinOp::Or;
  case LLVM::AtomicBinOp::_xor:
    return llvm::AtomicRMWInst::BinOp::Xor;
  case LLVM::AtomicBinOp::max:
    return llvm::AtomicRMWInst::BinOp::Max;
  case LLVM::AtomicBinOp::min:
    return llvm::AtomicRMWInst::BinOp::Min;
  case LLVM::AtomicBinOp::umax:
    return llvm::AtomicRMWInst::BinOp::UMax;
  case LLVM::AtomicBinOp::umin:
    return llvm::AtomicRMWInst::BinOp::UMin;
  case LLVM::AtomicBinOp::fadd:
    return llvm::AtomicRMWInst::BinOp::FAdd;
  case LLVM::AtomicBinOp::fsub:
    return llvm::AtomicRMWInst::BinOp::FSub;
  }
  llvm_unreachable("incorrect atomic binary operator");
}

static llvm::AtomicOrdering getLLVMAtomicOrdering(AtomicOrdering ordering) {
  switch (ordering) {
  case LLVM::AtomicOrdering::not_atomic:
    return llvm::AtomicOrdering::NotAtomic;
  case LLVM::AtomicOrdering::unordered:
    return llvm::AtomicOrdering::Unordered;
  case LLVM::AtomicOrdering::monotonic:
    return llvm::AtomicOrdering::Monotonic;
  case LLVM::AtomicOrdering::acquire:
    return llvm::AtomicOrdering::Acquire;
  case LLVM::AtomicOrdering::release:
    return llvm::AtomicOrdering::Release;
  case LLVM::AtomicOrdering::acq_rel:
    return llvm::AtomicOrdering::AcquireRelease;
  case LLVM::AtomicOrdering::seq_cst:
    return llvm::AtomicOrdering::SequentiallyConsistent;
  }
  llvm_unreachable("incorrect atomic ordering");
}

static llvm::FastMathFlags getFastmathFlags(FastmathFlagsInterface &op) {
  using llvmFMF = llvm::FastMathFlags;
  using FuncT = void (llvmFMF::*)(bool);
  const std::pair<FastmathFlags, FuncT> handlers[] = {
      // clang-format off
      {FastmathFlags::nnan,     &llvmFMF::setNoNaNs},
      {FastmathFlags::ninf,     &llvmFMF::setNoInfs},
      {FastmathFlags::nsz,      &llvmFMF::setNoSignedZeros},
      {FastmathFlags::arcp,     &llvmFMF::setAllowReciprocal},
      {FastmathFlags::contract, &llvmFMF::setAllowContract},
      {FastmathFlags::afn,      &llvmFMF::setApproxFunc},
      {FastmathFlags::reassoc,  &llvmFMF::setAllowReassoc},
      {FastmathFlags::fast,     &llvmFMF::setFast},
      // clang-format on
  };
  llvm::FastMathFlags ret;
  auto fmf = op.fastmathFlags();
  for (auto it : handlers)
    if (bitEnumContains(fmf, it.first))
      (ret.*(it.second))(true);
  return ret;
}

static LogicalResult
convertOperationImpl(Operation &opInst, llvm::IRBuilderBase &builder,
                     LLVM::ModuleTranslation &moduleTranslation) {
  auto extractPosition = [](ArrayAttr attr) {
    SmallVector<unsigned, 4> position;
    position.reserve(attr.size());
    for (Attribute v : attr)
      position.push_back(v.cast<IntegerAttr>().getValue().getZExtValue());
    return position;
  };

  llvm::IRBuilder<>::FastMathFlagGuard fmfGuard(builder);
  if (auto fmf = dyn_cast<FastmathFlagsInterface>(opInst))
    builder.setFastMathFlags(getFastmathFlags(fmf));

#include "mlir/Dialect/LLVMIR/LLVMConversions.inc"

  // Emit function calls.  If the "callee" attribute is present, this is a
  // direct function call and we also need to look up the remapped function
  // itself.  Otherwise, this is an indirect call and the callee is the first
  // operand, look it up as a normal value.  Return the llvm::Value representing
  // the function result, which may be of llvm::VoidTy type.
  auto convertCall = [&](Operation &op) -> llvm::Value * {
    auto operands = moduleTranslation.lookupValues(op.getOperands());
    ArrayRef<llvm::Value *> operandsRef(operands);
    if (auto attr = op.getAttrOfType<FlatSymbolRefAttr>("callee"))
      return builder.CreateCall(
          moduleTranslation.lookupFunction(attr.getValue()), operandsRef);
    auto *calleePtrType =
        cast<llvm::PointerType>(operandsRef.front()->getType());
    auto *calleeType =
        cast<llvm::FunctionType>(calleePtrType->getElementType());
    return builder.CreateCall(calleeType, operandsRef.front(),
                              operandsRef.drop_front());
  };

  // Emit calls.  If the called function has a result, remap the corresponding
  // value.  Note that LLVM IR dialect CallOp has either 0 or 1 result.
  if (isa<LLVM::CallOp>(opInst)) {
    llvm::Value *result = convertCall(opInst);
    if (opInst.getNumResults() != 0) {
      moduleTranslation.mapValue(opInst.getResult(0), result);
      return success();
    }
    // Check that LLVM call returns void for 0-result functions.
    return success(result->getType()->isVoidTy());
  }

  if (auto inlineAsmOp = dyn_cast<LLVM::InlineAsmOp>(opInst)) {
    // TODO: refactor function type creation which usually occurs in std-LLVM
    // conversion.
    SmallVector<Type, 8> operandTypes;
    operandTypes.reserve(inlineAsmOp.operands().size());
    for (auto t : inlineAsmOp.operands().getTypes())
      operandTypes.push_back(t);

    Type resultType;
    if (inlineAsmOp.getNumResults() == 0) {
      resultType = LLVM::LLVMVoidType::get(&moduleTranslation.getContext());
    } else {
      assert(inlineAsmOp.getNumResults() == 1);
      resultType = inlineAsmOp.getResultTypes()[0];
    }
    auto ft = LLVM::LLVMFunctionType::get(resultType, operandTypes);
    llvm::InlineAsm *inlineAsmInst =
        inlineAsmOp.asm_dialect().hasValue()
            ? llvm::InlineAsm::get(
                  static_cast<llvm::FunctionType *>(
                      moduleTranslation.convertType(ft)),
                  inlineAsmOp.asm_string(), inlineAsmOp.constraints(),
                  inlineAsmOp.has_side_effects(), inlineAsmOp.is_align_stack(),
                  convertAsmDialectToLLVM(*inlineAsmOp.asm_dialect()))
            : llvm::InlineAsm::get(
                  static_cast<llvm::FunctionType *>(
                      moduleTranslation.convertType(ft)),
                  inlineAsmOp.asm_string(), inlineAsmOp.constraints(),
                  inlineAsmOp.has_side_effects(), inlineAsmOp.is_align_stack());
    llvm::Value *result = builder.CreateCall(
        inlineAsmInst, moduleTranslation.lookupValues(inlineAsmOp.operands()));
    if (opInst.getNumResults() != 0)
      moduleTranslation.mapValue(opInst.getResult(0), result);
    return success();
  }

  if (auto invOp = dyn_cast<LLVM::InvokeOp>(opInst)) {
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    ArrayRef<llvm::Value *> operandsRef(operands);
    if (auto attr = opInst.getAttrOfType<FlatSymbolRefAttr>("callee")) {
      builder.CreateInvoke(moduleTranslation.lookupFunction(attr.getValue()),
                           moduleTranslation.lookupBlock(invOp.getSuccessor(0)),
                           moduleTranslation.lookupBlock(invOp.getSuccessor(1)),
                           operandsRef);
    } else {
      auto *calleePtrType =
          cast<llvm::PointerType>(operandsRef.front()->getType());
      auto *calleeType =
          cast<llvm::FunctionType>(calleePtrType->getElementType());
      builder.CreateInvoke(calleeType, operandsRef.front(),
                           moduleTranslation.lookupBlock(invOp.getSuccessor(0)),
                           moduleTranslation.lookupBlock(invOp.getSuccessor(1)),
                           operandsRef.drop_front());
    }
    return success();
  }

  if (auto lpOp = dyn_cast<LLVM::LandingpadOp>(opInst)) {
    llvm::Type *ty = moduleTranslation.convertType(lpOp.getType());
    llvm::LandingPadInst *lpi =
        builder.CreateLandingPad(ty, lpOp.getNumOperands());

    // Add clauses
    for (llvm::Value *operand :
         moduleTranslation.lookupValues(lpOp.getOperands())) {
      // All operands should be constant - checked by verifier
      if (auto *constOperand = dyn_cast<llvm::Constant>(operand))
        lpi->addClause(constOperand);
    }
    moduleTranslation.mapValue(lpOp.getResult(), lpi);
    return success();
  }

  // Emit branches.  We need to look up the remapped blocks and ignore the block
  // arguments that were transformed into PHI nodes.
  if (auto brOp = dyn_cast<LLVM::BrOp>(opInst)) {
    llvm::BranchInst *branch =
        builder.CreateBr(moduleTranslation.lookupBlock(brOp.getSuccessor()));
    moduleTranslation.mapBranch(&opInst, branch);
    return success();
  }
  if (auto condbrOp = dyn_cast<LLVM::CondBrOp>(opInst)) {
    auto weights = condbrOp.branch_weights();
    llvm::MDNode *branchWeights = nullptr;
    if (weights) {
      // Map weight attributes to LLVM metadata.
      auto trueWeight =
          weights.getValue().getValue(0).cast<IntegerAttr>().getInt();
      auto falseWeight =
          weights.getValue().getValue(1).cast<IntegerAttr>().getInt();
      branchWeights =
          llvm::MDBuilder(moduleTranslation.getLLVMContext())
              .createBranchWeights(static_cast<uint32_t>(trueWeight),
                                   static_cast<uint32_t>(falseWeight));
    }
    llvm::BranchInst *branch = builder.CreateCondBr(
        moduleTranslation.lookupValue(condbrOp.getOperand(0)),
        moduleTranslation.lookupBlock(condbrOp.getSuccessor(0)),
        moduleTranslation.lookupBlock(condbrOp.getSuccessor(1)), branchWeights);
    moduleTranslation.mapBranch(&opInst, branch);
    return success();
  }
  if (auto switchOp = dyn_cast<LLVM::SwitchOp>(opInst)) {
    llvm::MDNode *branchWeights = nullptr;
    if (auto weights = switchOp.branch_weights()) {
      llvm::SmallVector<uint32_t> weightValues;
      weightValues.reserve(weights->size());
      for (llvm::APInt weight : weights->cast<DenseIntElementsAttr>())
        weightValues.push_back(weight.getLimitedValue());
      branchWeights = llvm::MDBuilder(moduleTranslation.getLLVMContext())
                          .createBranchWeights(weightValues);
    }

    llvm::SwitchInst *switchInst = builder.CreateSwitch(
        moduleTranslation.lookupValue(switchOp.value()),
        moduleTranslation.lookupBlock(switchOp.defaultDestination()),
        switchOp.caseDestinations().size(), branchWeights);

    auto *ty = llvm::cast<llvm::IntegerType>(
        moduleTranslation.convertType(switchOp.value().getType()));
    for (auto i :
         llvm::zip(switchOp.case_values()->cast<DenseIntElementsAttr>(),
                   switchOp.caseDestinations()))
      switchInst->addCase(
          llvm::ConstantInt::get(ty, std::get<0>(i).getLimitedValue()),
          moduleTranslation.lookupBlock(std::get<1>(i)));

    moduleTranslation.mapBranch(&opInst, switchInst);
    return success();
  }

  // Emit addressof.  We need to look up the global value referenced by the
  // operation and store it in the MLIR-to-LLVM value mapping.  This does not
  // emit any LLVM instruction.
  if (auto addressOfOp = dyn_cast<LLVM::AddressOfOp>(opInst)) {
    LLVM::GlobalOp global = addressOfOp.getGlobal();
    LLVM::LLVMFuncOp function = addressOfOp.getFunction();

    // The verifier should not have allowed this.
    assert((global || function) &&
           "referencing an undefined global or function");

    moduleTranslation.mapValue(
        addressOfOp.getResult(),
        global ? moduleTranslation.lookupGlobal(global)
               : moduleTranslation.lookupFunction(function.getName()));
    return success();
  }

  return failure();
}

LogicalResult mlir::LLVMDialectLLVMIRTranslationInterface::convertOperation(
    Operation *op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) const {
  return convertOperationImpl(*op, builder, moduleTranslation);
}
