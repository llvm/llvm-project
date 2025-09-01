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
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/MatrixBuilder.h"

using namespace mlir;
using namespace mlir::LLVM;
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
  return str;
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
  return llvm::Intrinsic::getOrInsertDeclaration(module, id,
                                                 overloadedArgTysRef);
}

static llvm::OperandBundleDef
convertOperandBundle(OperandRange bundleOperands, StringRef bundleTag,
                     LLVM::ModuleTranslation &moduleTranslation) {
  std::vector<llvm::Value *> operands;
  operands.reserve(bundleOperands.size());
  for (Value bundleArg : bundleOperands)
    operands.push_back(moduleTranslation.lookupValue(bundleArg));
  return llvm::OperandBundleDef(bundleTag.str(), std::move(operands));
}

static SmallVector<llvm::OperandBundleDef>
convertOperandBundles(OperandRangeRange bundleOperands, ArrayAttr bundleTags,
                      LLVM::ModuleTranslation &moduleTranslation) {
  SmallVector<llvm::OperandBundleDef> bundles;
  bundles.reserve(bundleOperands.size());

  for (auto [operands, tagAttr] : llvm::zip_equal(bundleOperands, bundleTags)) {
    StringRef tag = cast<StringAttr>(tagAttr).getValue();
    bundles.push_back(convertOperandBundle(operands, tag, moduleTranslation));
  }
  return bundles;
}

static SmallVector<llvm::OperandBundleDef>
convertOperandBundles(OperandRangeRange bundleOperands,
                      std::optional<ArrayAttr> bundleTags,
                      LLVM::ModuleTranslation &moduleTranslation) {
  if (!bundleTags)
    return {};
  return convertOperandBundles(bundleOperands, *bundleTags, moduleTranslation);
}

/// Builder for LLVM_CallIntrinsicOp
static LogicalResult
convertCallLLVMIntrinsicOp(CallIntrinsicOp op, llvm::IRBuilderBase &builder,
                           LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::Intrinsic::ID id =
      llvm::Intrinsic::lookupIntrinsicID(op.getIntrinAttr());
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
    fn = llvm::Intrinsic::getOrInsertDeclaration(module, id, {});
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
      op.getArgs().size() != fn->arg_size()) {
    return mlir::emitError(op.getLoc(), "intrinsic call has ")
           << op.getArgs().size() << " operands but " << op.getIntrinAttr()
           << " expects " << fn->arg_size();
  }
  if (fn->getFunctionType()->isVarArg() &&
      op.getArgs().size() < fn->arg_size()) {
    return mlir::emitError(op.getLoc(), "intrinsic call has ")
           << op.getArgs().size() << " operands but variadic "
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

  auto *inst = builder.CreateCall(
      fn, moduleTranslation.lookupValues(op.getArgs()),
      convertOperandBundles(op.getOpBundleOperands(), op.getOpBundleTags(),
                            moduleTranslation));

  if (failed(moduleTranslation.convertArgAndResultAttrs(op, inst)))
    return failure();

  if (op.getNumResults() == 1)
    moduleTranslation.mapValue(op->getResults().front()) = inst;
  return success();
}

static void convertLinkerOptionsOp(ArrayAttr options,
                                   llvm::IRBuilderBase &builder,
                                   LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Module *llvmModule = moduleTranslation.getLLVMModule();
  llvm::LLVMContext &context = llvmModule->getContext();
  llvm::NamedMDNode *linkerMDNode =
      llvmModule->getOrInsertNamedMetadata("llvm.linker.options");
  SmallVector<llvm::Metadata *> MDNodes;
  MDNodes.reserve(options.size());
  for (auto s : options.getAsRange<StringAttr>()) {
    auto *MDNode = llvm::MDString::get(context, s.getValue());
    MDNodes.push_back(MDNode);
  }

  auto *listMDNode = llvm::MDTuple::get(context, MDNodes);
  linkerMDNode->addOperand(listMDNode);
}

static llvm::Metadata *
convertModuleFlagValue(StringRef key, ArrayAttr arrayAttr,
                       llvm::IRBuilderBase &builder,
                       LLVM::ModuleTranslation &moduleTranslation) {
  llvm::LLVMContext &context = builder.getContext();
  llvm::MDBuilder mdb(context);
  SmallVector<llvm::Metadata *> nodes;

  if (key == LLVMDialect::getModuleFlagKeyCGProfileName()) {
    for (auto entry : arrayAttr.getAsRange<ModuleFlagCGProfileEntryAttr>()) {
      llvm::Metadata *fromMetadata =
          entry.getFrom()
              ? llvm::ValueAsMetadata::get(moduleTranslation.lookupFunction(
                    entry.getFrom().getValue()))
              : nullptr;
      llvm::Metadata *toMetadata =
          entry.getTo()
              ? llvm::ValueAsMetadata::get(
                    moduleTranslation.lookupFunction(entry.getTo().getValue()))
              : nullptr;

      llvm::Metadata *vals[] = {
          fromMetadata, toMetadata,
          mdb.createConstant(llvm::ConstantInt::get(
              llvm::Type::getInt64Ty(context), entry.getCount()))};
      nodes.push_back(llvm::MDNode::get(context, vals));
    }
    return llvm::MDTuple::getDistinct(context, nodes);
  }
  return nullptr;
}

static llvm::Metadata *convertModuleFlagProfileSummaryAttr(
    StringRef key, ModuleFlagProfileSummaryAttr summaryAttr,
    llvm::IRBuilderBase &builder, LLVM::ModuleTranslation &moduleTranslation) {
  llvm::LLVMContext &context = builder.getContext();
  llvm::MDBuilder mdb(context);

  auto getIntTuple = [&](StringRef key, uint64_t val) -> llvm::MDTuple * {
    SmallVector<llvm::Metadata *> tupleNodes{
        mdb.createString(key), mdb.createConstant(llvm::ConstantInt::get(
                                   llvm::Type::getInt64Ty(context), val))};
    return llvm::MDTuple::get(context, tupleNodes);
  };

  SmallVector<llvm::Metadata *> fmtNode{
      mdb.createString("ProfileFormat"),
      mdb.createString(
          stringifyProfileSummaryFormatKind(summaryAttr.getFormat()))};

  SmallVector<llvm::Metadata *> vals = {
      llvm::MDTuple::get(context, fmtNode),
      getIntTuple("TotalCount", summaryAttr.getTotalCount()),
      getIntTuple("MaxCount", summaryAttr.getMaxCount()),
      getIntTuple("MaxInternalCount", summaryAttr.getMaxInternalCount()),
      getIntTuple("MaxFunctionCount", summaryAttr.getMaxFunctionCount()),
      getIntTuple("NumCounts", summaryAttr.getNumCounts()),
      getIntTuple("NumFunctions", summaryAttr.getNumFunctions()),
  };

  if (summaryAttr.getIsPartialProfile())
    vals.push_back(
        getIntTuple("IsPartialProfile", *summaryAttr.getIsPartialProfile()));

  if (summaryAttr.getPartialProfileRatio()) {
    SmallVector<llvm::Metadata *> tupleNodes{
        mdb.createString("PartialProfileRatio"),
        mdb.createConstant(llvm::ConstantFP::get(
            llvm::Type::getDoubleTy(context),
            summaryAttr.getPartialProfileRatio().getValue()))};
    vals.push_back(llvm::MDTuple::get(context, tupleNodes));
  }

  SmallVector<llvm::Metadata *> detailedEntries;
  llvm::Type *llvmInt64Type = llvm::Type::getInt64Ty(context);
  for (ModuleFlagProfileSummaryDetailedAttr detailedEntry :
       summaryAttr.getDetailedSummary()) {
    SmallVector<llvm::Metadata *> tupleNodes{
        mdb.createConstant(
            llvm::ConstantInt::get(llvmInt64Type, detailedEntry.getCutOff())),
        mdb.createConstant(
            llvm::ConstantInt::get(llvmInt64Type, detailedEntry.getMinCount())),
        mdb.createConstant(llvm::ConstantInt::get(
            llvmInt64Type, detailedEntry.getNumCounts()))};
    detailedEntries.push_back(llvm::MDTuple::get(context, tupleNodes));
  }
  SmallVector<llvm::Metadata *> detailedSummary{
      mdb.createString("DetailedSummary"),
      llvm::MDTuple::get(context, detailedEntries)};
  vals.push_back(llvm::MDTuple::get(context, detailedSummary));

  return llvm::MDNode::get(context, vals);
}

static void convertModuleFlagsOp(ArrayAttr flags, llvm::IRBuilderBase &builder,
                                 LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Module *llvmModule = moduleTranslation.getLLVMModule();
  for (auto flagAttr : flags.getAsRange<ModuleFlagAttr>()) {
    llvm::Metadata *valueMetadata =
        llvm::TypeSwitch<Attribute, llvm::Metadata *>(flagAttr.getValue())
            .Case<StringAttr>([&](auto strAttr) {
              return llvm::MDString::get(builder.getContext(),
                                         strAttr.getValue());
            })
            .Case<IntegerAttr>([&](auto intAttr) {
              return llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                  llvm::Type::getInt32Ty(builder.getContext()),
                  intAttr.getInt()));
            })
            .Case<ArrayAttr>([&](auto arrayAttr) {
              return convertModuleFlagValue(flagAttr.getKey().getValue(),
                                            arrayAttr, builder,
                                            moduleTranslation);
            })
            .Case([&](ModuleFlagProfileSummaryAttr summaryAttr) {
              return convertModuleFlagProfileSummaryAttr(
                  flagAttr.getKey().getValue(), summaryAttr, builder,
                  moduleTranslation);
            })
            .Default([](auto) { return nullptr; });

    assert(valueMetadata && "expected valid metadata");
    llvmModule->addModuleFlag(
        convertModFlagBehaviorToLLVM(flagAttr.getBehavior()),
        flagAttr.getKey().getValue(), valueMetadata);
  }
}

static llvm::DILocalScope *
getLocalScopeFromLoc(llvm::IRBuilderBase &builder, Location loc,
                     LLVM::ModuleTranslation &moduleTranslation) {
  if (auto scopeLoc = loc->findInstanceOf<FusedLocWith<LLVM::DIScopeAttr>>())
    if (auto *localScope = llvm::dyn_cast<llvm::DILocalScope>(
            moduleTranslation.translateDebugInfo(scopeLoc.getMetadata())))
      return localScope;
  return builder.GetInsertBlock()->getParent()->getSubprogram();
}

static LogicalResult
convertOperationImpl(Operation &opInst, llvm::IRBuilderBase &builder,
                     LLVM::ModuleTranslation &moduleTranslation) {

  llvm::IRBuilder<>::FastMathFlagGuard fmfGuard(builder);
  if (auto fmf = dyn_cast<FastmathFlagsInterface>(opInst))
    builder.setFastMathFlags(getFastmathFlags(fmf));

#include "mlir/Dialect/LLVMIR/LLVMConversions.inc"
#include "mlir/Dialect/LLVMIR/LLVMIntrinsicConversions.inc"

  // Emit function calls.  If the "callee" attribute is present, this is a
  // direct function call and we also need to look up the remapped function
  // itself.  Otherwise, this is an indirect call and the callee is the first
  // operand, look it up as a normal value.
  if (auto callOp = dyn_cast<LLVM::CallOp>(opInst)) {
    auto operands = moduleTranslation.lookupValues(callOp.getCalleeOperands());
    SmallVector<llvm::OperandBundleDef> opBundles =
        convertOperandBundles(callOp.getOpBundleOperands(),
                              callOp.getOpBundleTags(), moduleTranslation);
    ArrayRef<llvm::Value *> operandsRef(operands);
    llvm::CallInst *call;
    if (auto attr = callOp.getCalleeAttr()) {
      if (llvm::Function *function =
              moduleTranslation.lookupFunction(attr.getValue())) {
        call = builder.CreateCall(function, operandsRef, opBundles);
      } else {
        Operation *moduleOp = parentLLVMModule(&opInst);
        Operation *ifuncOp =
            moduleTranslation.symbolTable().lookupSymbolIn(moduleOp, attr);
        llvm::GlobalValue *ifunc = moduleTranslation.lookupIFunc(ifuncOp);
        llvm::FunctionType *calleeType = llvm::cast<llvm::FunctionType>(
            moduleTranslation.convertType(callOp.getCalleeFunctionType()));
        call = builder.CreateCall(calleeType, ifunc, operandsRef, opBundles);
      }
    } else {
      llvm::FunctionType *calleeType = llvm::cast<llvm::FunctionType>(
          moduleTranslation.convertType(callOp.getCalleeFunctionType()));
      call = builder.CreateCall(calleeType, operandsRef.front(),
                                operandsRef.drop_front(), opBundles);
    }
    call->setCallingConv(convertCConvToLLVM(callOp.getCConv()));
    call->setTailCallKind(convertTailCallKindToLLVM(callOp.getTailCallKind()));
    if (callOp.getConvergentAttr())
      call->addFnAttr(llvm::Attribute::Convergent);
    if (callOp.getNoUnwindAttr())
      call->addFnAttr(llvm::Attribute::NoUnwind);
    if (callOp.getWillReturnAttr())
      call->addFnAttr(llvm::Attribute::WillReturn);
    if (callOp.getNoInlineAttr())
      call->addFnAttr(llvm::Attribute::NoInline);
    if (callOp.getAlwaysInlineAttr())
      call->addFnAttr(llvm::Attribute::AlwaysInline);
    if (callOp.getInlineHintAttr())
      call->addFnAttr(llvm::Attribute::InlineHint);

    if (failed(moduleTranslation.convertArgAndResultAttrs(callOp, call)))
      return failure();

    if (MemoryEffectsAttr memAttr = callOp.getMemoryEffectsAttr()) {
      llvm::MemoryEffects memEffects =
          llvm::MemoryEffects(llvm::MemoryEffects::Location::ArgMem,
                              convertModRefInfoToLLVM(memAttr.getArgMem())) |
          llvm::MemoryEffects(
              llvm::MemoryEffects::Location::InaccessibleMem,
              convertModRefInfoToLLVM(memAttr.getInaccessibleMem())) |
          llvm::MemoryEffects(llvm::MemoryEffects::Location::Other,
                              convertModRefInfoToLLVM(memAttr.getOther()));
      call->setMemoryEffects(memEffects);
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
    inst->setTailCallKind(convertTailCallKindToLLVM(
        inlineAsmOp.getTailCallKindAttr().getTailCallKind()));
    if (auto maybeOperandAttrs = inlineAsmOp.getOperandAttrs()) {
      llvm::AttributeList attrList;
      for (const auto &it : llvm::enumerate(*maybeOperandAttrs)) {
        Attribute attr = it.value();
        if (!attr)
          continue;
        DictionaryAttr dAttr = cast<DictionaryAttr>(attr);
        if (dAttr.empty())
          continue;
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
    SmallVector<llvm::OperandBundleDef> opBundles =
        convertOperandBundles(invOp.getOpBundleOperands(),
                              invOp.getOpBundleTags(), moduleTranslation);
    ArrayRef<llvm::Value *> operandsRef(operands);
    llvm::InvokeInst *result;
    if (auto attr = opInst.getAttrOfType<FlatSymbolRefAttr>("callee")) {
      result = builder.CreateInvoke(
          moduleTranslation.lookupFunction(attr.getValue()),
          moduleTranslation.lookupBlock(invOp.getSuccessor(0)),
          moduleTranslation.lookupBlock(invOp.getSuccessor(1)), operandsRef,
          opBundles);
    } else {
      llvm::FunctionType *calleeType = llvm::cast<llvm::FunctionType>(
          moduleTranslation.convertType(invOp.getCalleeFunctionType()));
      result = builder.CreateInvoke(
          calleeType, operandsRef.front(),
          moduleTranslation.lookupBlock(invOp.getSuccessor(0)),
          moduleTranslation.lookupBlock(invOp.getSuccessor(1)),
          operandsRef.drop_front(), opBundles);
    }
    result->setCallingConv(convertCConvToLLVM(invOp.getCConv()));
    if (failed(moduleTranslation.convertArgAndResultAttrs(invOp, result)))
      return failure();
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
  if (auto indBrOp = dyn_cast<LLVM::IndirectBrOp>(opInst)) {
    llvm::IndirectBrInst *indBr = builder.CreateIndirectBr(
        moduleTranslation.lookupValue(indBrOp.getAddr()),
        indBrOp->getNumSuccessors());
    for (auto *succ : indBrOp.getSuccessors())
      indBr->addDestination(moduleTranslation.lookupBlock(succ));
    moduleTranslation.mapBranch(&opInst, indBr);
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
    LLVM::AliasOp alias = addressOfOp.getAlias(moduleTranslation.symbolTable());
    LLVM::IFuncOp ifunc = addressOfOp.getIFunc(moduleTranslation.symbolTable());

    // The verifier should not have allowed this.
    assert((global || function || alias || ifunc) &&
           "referencing an undefined global, function, alias, or ifunc");

    llvm::Value *llvmValue = nullptr;
    if (global)
      llvmValue = moduleTranslation.lookupGlobal(global);
    else if (alias)
      llvmValue = moduleTranslation.lookupAlias(alias);
    else if (function)
      llvmValue = moduleTranslation.lookupFunction(function.getName());
    else
      llvmValue = moduleTranslation.lookupIFunc(ifunc);

    moduleTranslation.mapValue(addressOfOp.getResult(), llvmValue);
    return success();
  }

  // Emit dso_local_equivalent. We need to look up the global value referenced
  // by the operation and store it in the MLIR-to-LLVM value mapping.
  if (auto dsoLocalEquivalentOp =
          dyn_cast<LLVM::DSOLocalEquivalentOp>(opInst)) {
    LLVM::LLVMFuncOp function =
        dsoLocalEquivalentOp.getFunction(moduleTranslation.symbolTable());
    LLVM::AliasOp alias =
        dsoLocalEquivalentOp.getAlias(moduleTranslation.symbolTable());

    // The verifier should not have allowed this.
    assert((function || alias) &&
           "referencing an undefined function, or alias");

    llvm::Value *llvmValue = nullptr;
    if (alias)
      llvmValue = moduleTranslation.lookupAlias(alias);
    else
      llvmValue = moduleTranslation.lookupFunction(function.getName());

    moduleTranslation.mapValue(
        dsoLocalEquivalentOp.getResult(),
        llvm::DSOLocalEquivalent::get(cast<llvm::GlobalValue>(llvmValue)));
    return success();
  }

  // Emit blockaddress. We first need to find the LLVM block referenced by this
  // operation and then create a LLVM block address for it.
  if (auto blockAddressOp = dyn_cast<LLVM::BlockAddressOp>(opInst)) {
    BlockAddressAttr blockAddressAttr = blockAddressOp.getBlockAddr();
    llvm::BasicBlock *llvmBlock =
        moduleTranslation.lookupBlockAddress(blockAddressAttr);

    llvm::Value *llvmValue = nullptr;
    StringRef fnName = blockAddressAttr.getFunction().getValue();
    if (llvmBlock) {
      llvm::Function *llvmFn = moduleTranslation.lookupFunction(fnName);
      llvmValue = llvm::BlockAddress::get(llvmFn, llvmBlock);
    } else {
      // The matching LLVM block is not yet emitted, a placeholder is created
      // in its place. When the LLVM block is emitted later in translation,
      // the llvmValue is replaced with the actual llvm::BlockAddress.
      // A GlobalVariable is chosen as placeholder because in general LLVM
      // constants are uniqued and are not proper for RAUW, since that could
      // harm unrelated uses of the constant.
      llvmValue = new llvm::GlobalVariable(
          *moduleTranslation.getLLVMModule(),
          llvm::PointerType::getUnqual(moduleTranslation.getLLVMContext()),
          /*isConstant=*/true, llvm::GlobalValue::LinkageTypes::ExternalLinkage,
          /*Initializer=*/nullptr,
          Twine("__mlir_block_address_")
              .concat(Twine(fnName))
              .concat(Twine((uint64_t)blockAddressOp.getOperation())));
      moduleTranslation.mapUnresolvedBlockAddress(blockAddressOp, llvmValue);
    }

    moduleTranslation.mapValue(blockAddressOp.getResult(), llvmValue);
    return success();
  }

  // Emit block label. If this label is seen before BlockAddressOp is
  // translated, go ahead and already map it.
  if (auto blockTagOp = dyn_cast<LLVM::BlockTagOp>(opInst)) {
    auto funcOp = blockTagOp->getParentOfType<LLVMFuncOp>();
    BlockAddressAttr blockAddressAttr = BlockAddressAttr::get(
        &moduleTranslation.getContext(),
        FlatSymbolRefAttr::get(&moduleTranslation.getContext(),
                               funcOp.getName()),
        blockTagOp.getTag());
    moduleTranslation.mapBlockAddress(blockAddressAttr,
                                      builder.GetInsertBlock());
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
