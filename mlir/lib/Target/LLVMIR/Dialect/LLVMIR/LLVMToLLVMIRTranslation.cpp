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
using mlir::LLVM::detail::createIntrinsicCall;
using mlir::LLVM::detail::getLLVMConstant;

#include "mlir/Dialect/LLVMIR/LLVMConversionEnumsToLLVM.inc"

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
      // clang-format on
  };
  llvm::FastMathFlags ret;
  ::mlir::LLVM::FastmathFlags fmfMlir = op.getFastmathAttr().getValue();
  for (auto it : handlers)
    if (bitEnumContainsAll(fmfMlir, it.first))
      (ret.*(it.second))(true);
  return ret;
}

/// Convert the value of a DenseI64ArrayAttr to a vector of unsigned indices.
static SmallVector<unsigned> extractPosition(ArrayRef<int64_t> indices) {
  SmallVector<unsigned> position;
  llvm::append_range(position, indices);
  return position;
}

/// Convert an LLVM type to a string for printing in diagnostics.
static std::string diagStr(const llvm::Type *type) {
  std::string str;
  llvm::raw_string_ostream os(str);
  type->print(os);
  return os.str();
}

/// Get the declaration of an overloaded llvm intrinsic. First we get the
/// overloaded argument types and/or result type from the CallIntrinsicOp, and
/// then use those to get the correct declaration of the overloaded intrinsic.
static FailureOr<llvm::Function *>
getOverloadedDeclaration(CallIntrinsicOp op, llvm::Intrinsic::ID id,
                         llvm::Module *module,
                         LLVM::ModuleTranslation &moduleTranslation) {
  SmallVector<llvm::Type *, 8> allArgTys;
  for (Type type : op->getOperandTypes())
    allArgTys.push_back(moduleTranslation.convertType(type));

  llvm::Type *resTy;
  if (op.getNumResults() == 0)
    resTy = llvm::Type::getVoidTy(module->getContext());
  else
    resTy = moduleTranslation.convertType(op.getResult(0).getType());

  // ATM we do not support variadic intrinsics.
  llvm::FunctionType *ft = llvm::FunctionType::get(resTy, allArgTys, false);

  SmallVector<llvm::Intrinsic::IITDescriptor, 8> table;
  getIntrinsicInfoTableEntries(id, table);
  ArrayRef<llvm::Intrinsic::IITDescriptor> tableRef = table;

  SmallVector<llvm::Type *, 8> overloadedArgTys;
  if (llvm::Intrinsic::matchIntrinsicSignature(ft, tableRef,
                                               overloadedArgTys) !=
      llvm::Intrinsic::MatchIntrinsicTypesResult::MatchIntrinsicTypes_Match) {
    return mlir::emitError(op.getLoc(), "call intrinsic signature ")
           << diagStr(ft) << " to overloaded intrinsic " << op.getIntrinAttr()
           << " does not match any of the overloads";
  }

  ArrayRef<llvm::Type *> overloadedArgTysRef = overloadedArgTys;
  return llvm::Intrinsic::getDeclaration(module, id, overloadedArgTysRef);
}

/// Builder for LLVM_CallIntrinsicOp
static LogicalResult
convertCallLLVMIntrinsicOp(CallIntrinsicOp op, llvm::IRBuilderBase &builder,
                           LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::Intrinsic::ID id =
      llvm::Function::lookupIntrinsicID(op.getIntrinAttr());
  if (!id)
    return mlir::emitError(op.getLoc(), "could not find LLVM intrinsic: ")
           << op.getIntrinAttr();

  llvm::Function *fn = nullptr;
  if (llvm::Intrinsic::isOverloaded(id)) {
    auto fnOrFailure =
        getOverloadedDeclaration(op, id, module, moduleTranslation);
    if (failed(fnOrFailure))
      return failure();
    fn = *fnOrFailure;
  } else {
    fn = llvm::Intrinsic::getDeclaration(module, id, {});
  }

  // Check the result type of the call.
  const llvm::Type *intrinType =
      op.getNumResults() == 0
          ? llvm::Type::getVoidTy(module->getContext())
          : moduleTranslation.convertType(op.getResultTypes().front());
  if (intrinType != fn->getReturnType()) {
    return mlir::emitError(op.getLoc(), "intrinsic call returns ")
           << diagStr(intrinType) << " but " << op.getIntrinAttr()
           << " actually returns " << diagStr(fn->getReturnType());
  }

  // Check the argument types of the call. If the function is variadic, check
  // the subrange of required arguments.
  if (!fn->getFunctionType()->isVarArg() &&
      op.getNumOperands() != fn->arg_size()) {
    return mlir::emitError(op.getLoc(), "intrinsic call has ")
           << op.getNumOperands() << " operands but " << op.getIntrinAttr()
           << " expects " << fn->arg_size();
  }
  if (fn->getFunctionType()->isVarArg() &&
      op.getNumOperands() < fn->arg_size()) {
    return mlir::emitError(op.getLoc(), "intrinsic call has ")
           << op.getNumOperands() << " operands but variadic "
           << op.getIntrinAttr() << " expects at least " << fn->arg_size();
  }
  // Check the arguments up to the number the function requires.
  for (unsigned i = 0, e = fn->arg_size(); i != e; ++i) {
    const llvm::Type *expected = fn->getArg(i)->getType();
    const llvm::Type *actual =
        moduleTranslation.convertType(op.getOperandTypes()[i]);
    if (actual != expected) {
      return mlir::emitError(op.getLoc(), "intrinsic call operand #")
             << i << " has type " << diagStr(actual) << " but "
             << op.getIntrinAttr() << " expects " << diagStr(expected);
    }
  }

  FastmathFlagsInterface itf = op;
  builder.setFastMathFlags(getFastmathFlags(itf));

  auto *inst =
      builder.CreateCall(fn, moduleTranslation.lookupValues(op.getOperands()));
  if (op.getNumResults() == 1)
    moduleTranslation.mapValue(op->getResults().front()) = inst;
  return success();
}

static LogicalResult
convertOperationImpl(Operation &opInst, llvm::IRBuilderBase &builder,
                     LLVM::ModuleTranslation &moduleTranslation) {

  llvm::IRBuilder<>::FastMathFlagGuard fmfGuard(builder);
  if (auto fmf = dyn_cast<FastmathFlagsInterface>(opInst))
    builder.setFastMathFlags(getFastmathFlags(fmf));

#include "mlir/Dialect/LLVMIR/LLVMConversions.inc"
#include "mlir/Dialect/LLVMIR/LLVMIntrinsicConversions.inc"

  // Helper function to reconstruct the function type for an indirect call given
  // the result and argument types. The function cannot reconstruct the type of
  // variadic functions since the call operation does not carry enough
  // information to distinguish normal and variadic arguments. Supporting
  // indirect variadic calls requires an additional type attribute on the call
  // operation that stores the LLVM function type of the callee.
  // TODO: Support indirect calls to variadic function pointers.
  auto getCalleeFunctionType = [&](TypeRange resultTypes, ValueRange args) {
    Type resultType = resultTypes.empty()
                          ? LLVMVoidType::get(opInst.getContext())
                          : resultTypes.front();
    return llvm::cast<llvm::FunctionType>(moduleTranslation.convertType(
        LLVMFunctionType::get(opInst.getContext(), resultType,
                              llvm::to_vector(args.getTypes()), false)));
  };

  // Emit function calls.  If the "callee" attribute is present, this is a
  // direct function call and we also need to look up the remapped function
  // itself.  Otherwise, this is an indirect call and the callee is the first
  // operand, look it up as a normal value.
  if (auto callOp = dyn_cast<LLVM::CallOp>(opInst)) {
    auto operands = moduleTranslation.lookupValues(callOp.getOperands());
    ArrayRef<llvm::Value *> operandsRef(operands);
    llvm::CallInst *call;
    if (auto attr = callOp.getCalleeAttr()) {
      call = builder.CreateCall(
          moduleTranslation.lookupFunction(attr.getValue()), operandsRef);
    } else {
      call = builder.CreateCall(getCalleeFunctionType(callOp.getResultTypes(),
                                                      callOp.getArgOperands()),
                                operandsRef.front(), operandsRef.drop_front());
    }
    moduleTranslation.setAccessGroupsMetadata(callOp, call);
    moduleTranslation.setAliasScopeMetadata(callOp, call);
    moduleTranslation.setTBAAMetadata(callOp, call);
    // If the called function has a result, remap the corresponding value.  Note
    // that LLVM IR dialect CallOp has either 0 or 1 result.
    if (opInst.getNumResults() != 0)
      moduleTranslation.mapValue(opInst.getResult(0), call);
    // Check that LLVM call returns void for 0-result functions.
    else if (!call->getType()->isVoidTy())
      return failure();
    moduleTranslation.mapCall(callOp, call);
    return success();
  }

  if (auto inlineAsmOp = dyn_cast<LLVM::InlineAsmOp>(opInst)) {
    // TODO: refactor function type creation which usually occurs in std-LLVM
    // conversion.
    SmallVector<Type, 8> operandTypes;
    llvm::append_range(operandTypes, inlineAsmOp.getOperands().getTypes());

    Type resultType;
    if (inlineAsmOp.getNumResults() == 0) {
      resultType = LLVM::LLVMVoidType::get(&moduleTranslation.getContext());
    } else {
      assert(inlineAsmOp.getNumResults() == 1);
      resultType = inlineAsmOp.getResultTypes()[0];
    }
    auto ft = LLVM::LLVMFunctionType::get(resultType, operandTypes);
    llvm::InlineAsm *inlineAsmInst =
        inlineAsmOp.getAsmDialect()
            ? llvm::InlineAsm::get(
                  static_cast<llvm::FunctionType *>(
                      moduleTranslation.convertType(ft)),
                  inlineAsmOp.getAsmString(), inlineAsmOp.getConstraints(),
                  inlineAsmOp.getHasSideEffects(),
                  inlineAsmOp.getIsAlignStack(),
                  convertAsmDialectToLLVM(*inlineAsmOp.getAsmDialect()))
            : llvm::InlineAsm::get(static_cast<llvm::FunctionType *>(
                                       moduleTranslation.convertType(ft)),
                                   inlineAsmOp.getAsmString(),
                                   inlineAsmOp.getConstraints(),
                                   inlineAsmOp.getHasSideEffects(),
                                   inlineAsmOp.getIsAlignStack());
    llvm::CallInst *inst = builder.CreateCall(
        inlineAsmInst,
        moduleTranslation.lookupValues(inlineAsmOp.getOperands()));
    if (auto maybeOperandAttrs = inlineAsmOp.getOperandAttrs()) {
      llvm::AttributeList attrList;
      for (const auto &it : llvm::enumerate(*maybeOperandAttrs)) {
        Attribute attr = it.value();
        if (!attr)
          continue;
        DictionaryAttr dAttr = cast<DictionaryAttr>(attr);
        TypeAttr tAttr =
            cast<TypeAttr>(dAttr.get(InlineAsmOp::getElementTypeAttrName()));
        llvm::AttrBuilder b(moduleTranslation.getLLVMContext());
        llvm::Type *ty = moduleTranslation.convertType(tAttr.getValue());
        b.addTypeAttr(llvm::Attribute::ElementType, ty);
        // shift to account for the returned value (this is always 1 aggregate
        // value in LLVM).
        int shift = (opInst.getNumResults() > 0) ? 1 : 0;
        attrList = attrList.addAttributesAtIndex(
            moduleTranslation.getLLVMContext(), it.index() + shift, b);
      }
      inst->setAttributes(attrList);
    }

    if (opInst.getNumResults() != 0)
      moduleTranslation.mapValue(opInst.getResult(0), inst);
    return success();
  }

  if (auto invOp = dyn_cast<LLVM::InvokeOp>(opInst)) {
    auto operands = moduleTranslation.lookupValues(invOp.getCalleeOperands());
    ArrayRef<llvm::Value *> operandsRef(operands);
    llvm::Instruction *result;
    if (auto attr = opInst.getAttrOfType<FlatSymbolRefAttr>("callee")) {
      result = builder.CreateInvoke(
          moduleTranslation.lookupFunction(attr.getValue()),
          moduleTranslation.lookupBlock(invOp.getSuccessor(0)),
          moduleTranslation.lookupBlock(invOp.getSuccessor(1)), operandsRef);
    } else {
      result = builder.CreateInvoke(
          getCalleeFunctionType(invOp.getResultTypes(), invOp.getArgOperands()),
          operandsRef.front(),
          moduleTranslation.lookupBlock(invOp.getSuccessor(0)),
          moduleTranslation.lookupBlock(invOp.getSuccessor(1)),
          operandsRef.drop_front());
    }
    moduleTranslation.mapBranch(invOp, result);
    // InvokeOp can only have 0 or 1 result
    if (invOp->getNumResults() != 0) {
      moduleTranslation.mapValue(opInst.getResult(0), result);
      return success();
    }
    return success(result->getType()->isVoidTy());
  }

  if (auto lpOp = dyn_cast<LLVM::LandingpadOp>(opInst)) {
    llvm::Type *ty = moduleTranslation.convertType(lpOp.getType());
    llvm::LandingPadInst *lpi =
        builder.CreateLandingPad(ty, lpOp.getNumOperands());
    lpi->setCleanup(lpOp.getCleanup());

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

  // Emit branches.  We need to look up the remapped blocks and ignore the
  // block arguments that were transformed into PHI nodes.
  if (auto brOp = dyn_cast<LLVM::BrOp>(opInst)) {
    llvm::BranchInst *branch =
        builder.CreateBr(moduleTranslation.lookupBlock(brOp.getSuccessor()));
    moduleTranslation.mapBranch(&opInst, branch);
    moduleTranslation.setLoopMetadata(&opInst, branch);
    return success();
  }
  if (auto condbrOp = dyn_cast<LLVM::CondBrOp>(opInst)) {
    llvm::BranchInst *branch = builder.CreateCondBr(
        moduleTranslation.lookupValue(condbrOp.getOperand(0)),
        moduleTranslation.lookupBlock(condbrOp.getSuccessor(0)),
        moduleTranslation.lookupBlock(condbrOp.getSuccessor(1)));
    moduleTranslation.mapBranch(&opInst, branch);
    moduleTranslation.setLoopMetadata(&opInst, branch);
    return success();
  }
  if (auto switchOp = dyn_cast<LLVM::SwitchOp>(opInst)) {
    llvm::SwitchInst *switchInst = builder.CreateSwitch(
        moduleTranslation.lookupValue(switchOp.getValue()),
        moduleTranslation.lookupBlock(switchOp.getDefaultDestination()),
        switchOp.getCaseDestinations().size());

    // Handle switch with zero cases.
    if (!switchOp.getCaseValues())
      return success();

    auto *ty = llvm::cast<llvm::IntegerType>(
        moduleTranslation.convertType(switchOp.getValue().getType()));
    for (auto i :
         llvm::zip(llvm::cast<DenseIntElementsAttr>(*switchOp.getCaseValues()),
                   switchOp.getCaseDestinations()))
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
    LLVM::GlobalOp global =
        addressOfOp.getGlobal(moduleTranslation.symbolTable());
    LLVM::LLVMFuncOp function =
        addressOfOp.getFunction(moduleTranslation.symbolTable());

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

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the LLVM dialect to LLVM IR.
class LLVMDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    return convertOperationImpl(*op, builder, moduleTranslation);
  }
};
} // namespace

void mlir::registerLLVMDialectTranslation(DialectRegistry &registry) {
  registry.insert<LLVM::LLVMDialect>();
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    dialect->addInterfaces<LLVMDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerLLVMDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerLLVMDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
