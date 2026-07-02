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
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/FPEnv.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/MatrixBuilder.h"
#include "llvm/IR/MemoryModelRelaxationAnnotations.h"
#include "llvm/Support/LogicalResult.h"

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

//===----------------------------------------------------------------------===//
// Constrained floating-point lowering (the `#llvm.fenv` attribute).
//===----------------------------------------------------------------------===//

namespace {
/// Scoped guard that configures the IRBuilder's constrained floating-point
/// state, mirroring clang's `CodeGenFunction::CGFPOptionsRAII`. While the state
/// is enabled, the IRBuilder automatically lowers ordinary floating-point
/// operations (`CreateFAdd`, `CreateFCmp`, `CreateFPExt`, ...) to the matching
/// `llvm.experimental.constrained.*` intrinsics. The previous state is restored
/// on destruction.
class ConstrainedFPStateRAII {
public:
  explicit ConstrainedFPStateRAII(llvm::IRBuilderBase &builder)
      : builder(builder), oldIsConstrained(builder.getIsFPConstrained()),
        oldExcept(builder.getDefaultConstrainedExcept()),
        oldRounding(builder.getDefaultConstrainedRounding()) {}

  ~ConstrainedFPStateRAII() {
    builder.setIsFPConstrained(oldIsConstrained);
    builder.setDefaultConstrainedExcept(oldExcept);
    builder.setDefaultConstrainedRounding(oldRounding);
  }

  void enable(llvm::RoundingMode rounding, llvm::fp::ExceptionBehavior except) {
    builder.setIsFPConstrained(true);
    builder.setDefaultConstrainedRounding(rounding);
    builder.setDefaultConstrainedExcept(except);
  }

private:
  llvm::IRBuilderBase &builder;
  bool oldIsConstrained;
  llvm::fp::ExceptionBehavior oldExcept;
  llvm::RoundingMode oldRounding;
};
} // namespace

static llvm::RoundingMode
getConstrainedRoundingMode(LLVM::FPEnvConstrainedOpInterface fenvOp) {
  switch (fenvOp.getFenvRoundingMode()) {
  case LLVM::FPRoundingMode::Dynamic:
    return llvm::RoundingMode::Dynamic;
  case LLVM::FPRoundingMode::ToNearest:
    return llvm::RoundingMode::NearestTiesToEven;
  case LLVM::FPRoundingMode::Downward:
    return llvm::RoundingMode::TowardNegative;
  case LLVM::FPRoundingMode::Upward:
    return llvm::RoundingMode::TowardPositive;
  case LLVM::FPRoundingMode::UpwardZero:
    return llvm::RoundingMode::TowardZero;
  case LLVM::FPRoundingMode::ToNearestAway:
    return llvm::RoundingMode::NearestTiesToAway;
  }
  llvm_unreachable("unknown LLVM::FPRoundingMode");
}

static llvm::fp::ExceptionBehavior
getConstrainedExceptionBehavior(LLVM::FPEnvConstrainedOpInterface fenvOp) {
  switch (fenvOp.getFenvExceptionMode()) {
  case LLVM::FPExceptionMode::Masked:
    return llvm::fp::ebIgnore;
  case LLVM::FPExceptionMode::Unmasked:
  case LLVM::FPExceptionMode::Unknown:
    return fenvOp.getFenvStrictExcept() ? llvm::fp::ebStrict
                                        : llvm::fp::ebMayTrap;
  }
  llvm_unreachable("unknown LLVM::FPExceptionMode");
}

/// Maps a non-constrained LLVM intrinsic to its constrained counterpart, or
/// `not_intrinsic` if none exists. Used both for the math intrinsic dialect
/// operations and for `llvm.call_intrinsic`. `ConstrainedOps.def` is the single
/// source of truth for this mapping.
static constexpr llvm::Intrinsic::ID
getConstrainedIntrinsicFor(llvm::Intrinsic::ID base) {
  switch (base) {
#define DAG_FUNCTION(NAME, NARG, ROUND, INTRINSIC, DAGN)                       \
  case llvm::Intrinsic::NAME:                                                  \
    return llvm::Intrinsic::INTRINSIC;
#define FUNCTION(NAME, NARG, ROUND, INTRINSIC)                                 \
  case llvm::Intrinsic::NAME:                                                  \
    return llvm::Intrinsic::INTRINSIC;
#include "llvm/IR/ConstrainedOps.def"
  default:
    return llvm::Intrinsic::not_intrinsic;
  }
}

/// Emits a constrained floating-point call for a function-style operation: the
/// math intrinsic dialect operations and `llvm.call_intrinsic`. The original
/// floating-point operands are taken from \p fpOperands and the result is
/// mapped to \p result.
static LogicalResult
emitConstrainedFPCall(Operation *op, llvm::Intrinsic::ID constrainedID,
                      ValueRange fpOperands, Value result,
                      llvm::IRBuilderBase &builder,
                      LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Module *mod = builder.GetInsertBlock()->getModule();
  llvm::LLVMContext &ctx = mod->getContext();

  SmallVector<llvm::Value *> args = moduleTranslation.lookupValues(fpOperands);
  llvm::Type *resultType = moduleTranslation.convertType(result.getType());

  // Reconstruct the constrained intrinsic signature so the correct overloaded
  // declaration can be resolved. Constrained intrinsics take one (exception
  // behavior) or two (rounding mode and exception behavior) trailing metadata
  // arguments in addition to the original floating-point arguments.
  SmallVector<llvm::Type *> signatureArgTypes;
  signatureArgTypes.reserve(args.size() + 2);
  for (llvm::Value *arg : args)
    signatureArgTypes.push_back(arg->getType());
  unsigned numMetadataArgs =
      llvm::Intrinsic::hasConstrainedFPRoundingModeOperand(constrainedID) ? 2
                                                                          : 1;
  llvm::Type *metadataType = llvm::Type::getMetadataTy(ctx);
  for (unsigned i = 0; i < numMetadataArgs; ++i)
    signatureArgTypes.push_back(metadataType);

  llvm::FunctionType *signature = llvm::FunctionType::get(
      resultType, signatureArgTypes, /*isVarArg=*/false);

  std::string errorMsg;
  llvm::raw_string_ostream errorOS(errorMsg);
  SmallVector<llvm::Type *> overloadedTypes;
  if (!llvm::Intrinsic::isSignatureValid(constrainedID, signature,
                                         overloadedTypes, errorOS)) {
    return op->emitError("could not resolve constrained intrinsic for the "
                         "'fenv' attribute: ")
           << errorMsg;
  }

  llvm::Function *callee = llvm::Intrinsic::getOrInsertDeclaration(
      mod, constrainedID, overloadedTypes);
  // The rounding mode and exception behavior come from the IRBuilder's
  // constrained floating-point state, configured by ConstrainedFPStateRAII.
  llvm::Value *call = builder.CreateConstrainedFPCall(callee, args, "");
  moduleTranslation.mapValue(result, call);
  return success();
}

static bool isSignalingPredicate(LLVM::FCmpPredicate predicate) {
  switch (predicate) {
  case LLVM::FCmpPredicate::oeq:
  case LLVM::FCmpPredicate::one:
  case LLVM::FCmpPredicate::ueq:
  case LLVM::FCmpPredicate::une:
  case LLVM::FCmpPredicate::ugt:
  case LLVM::FCmpPredicate::uge:
  case LLVM::FCmpPredicate::ult:
  case LLVM::FCmpPredicate::ule:
  case LLVM::FCmpPredicate::uno:
    return false;
  default:
    return true;
  }
}

/// Emits the comparison for an `llvm.fcmp` carrying a `#llvm.fenv` attribute.
///
/// - `masked` exceptions: a plain `fcmp` instruction is emitted, even if other
///   `fenv` fields (such as rounding) are set, since comparisons are unaffected
///   by rounding and raise no exception.
/// - `unmasked`/`unknown` exceptions with `strict_snan` and a non-equality
///   predicate: a signaling comparison (`experimental.constrained.fcmps`).
/// - `unmasked`/`unknown` exceptions otherwise: a quiet comparison
///   (`experimental.constrained.fcmp`).
static LogicalResult
emitConstrainedFCmp(LLVM::FCmpOp fcmpOp, llvm::IRBuilderBase &builder,
                    LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Value *lhs = moduleTranslation.lookupValue(fcmpOp.getLhs());
  llvm::Value *rhs = moduleTranslation.lookupValue(fcmpOp.getRhs());
  llvm::CmpInst::Predicate predicate =
      convertFCmpPredicateToLLVM(fcmpOp.getPredicate());

  llvm::Value *result;
  if (fcmpOp.getFenvExceptionMode() == LLVM::FPExceptionMode::Masked) {
    // Exceptions are masked: emit an ordinary comparison. The IRBuilder's
    // constrained state may be enabled because of unrelated `fenv` fields, so
    // temporarily disable it to force an unconstrained `fcmp`.
    bool wasConstrained = builder.getIsFPConstrained();
    builder.setIsFPConstrained(false);
    result = builder.CreateFCmp(predicate, lhs, rhs);
    builder.setIsFPConstrained(wasConstrained);
  } else if (fcmpOp.getFenvStrictSNaN() &&
             isSignalingPredicate(fcmpOp.getPredicate())) {
    result = builder.CreateFCmpS(predicate, lhs, rhs);
  } else {
    result = builder.CreateFCmp(predicate, lhs, rhs);
  }
  moduleTranslation.mapValue(fcmpOp.getRes(), result);
  return success();
}

/// Emits the constrained floating-point form of a math intrinsic dialect
/// operation (sqrt, sin, fma, ...) carrying a `#llvm.fenv` attribute. These are
/// declared with `LLVM_{Unary,BinarySameArgs,TernarySameArgs}IntrOpF<"func">`.
/// \p base is the operation's unconstrained LLVM intrinsic. It is known
/// statically in the generated translation code (the same ID the default
/// builder passes to `createIntrinsicCall`) and is mapped to its constrained
/// counterpart via `ConstrainedOps.def`. Because `getConstrainedIntrinsicFor`
/// is `constexpr` and \p base is a compile-time constant at each call site,
/// that mapping folds to the exact `experimental_constrained_*` intrinsic. The
/// rounding mode and exception behavior are taken from the IRBuilder's
/// constrained state (see ConstrainedFPStateRAII).
///
/// This is invoked from the generated translation code
/// (LLVMIntrinsicConversions .inc) for the math intrinsic operations. See the
/// `llvmBuilder` overrides on the `LLVM_*SameArgsIntrOpF`/`LLVM_UnaryIntrOpF`
/// base classes.
static LogicalResult
emitConstrainedFPIntrinsic(Operation *op, llvm::Intrinsic::ID base,
                           llvm::IRBuilderBase &builder,
                           LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Intrinsic::ID constrainedID = getConstrainedIntrinsicFor(base);
  if (!constrainedID)
    return op->emitError("no constrained intrinsic is available for '")
           << op->getName().getStringRef() << "' carrying a 'fenv' attribute";

  return emitConstrainedFPCall(op, constrainedID, op->getOperands(),
                               op->getResult(0), builder, moduleTranslation);
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

  std::string errorMsg;
  llvm::raw_string_ostream errorOS(errorMsg);
  SmallVector<llvm::Type *, 8> overloadedTys;
  if (!llvm::Intrinsic::isSignatureValid(id, ft, overloadedTys, errorOS)) {
    return mlir::emitError(op.getLoc(), "call intrinsic signature ")
           << diagStr(ft) << " to overloaded intrinsic " << op.getIntrinAttr()
           << " does not match any of the overloads: " << errorMsg;
  }

  return llvm::Intrinsic::getOrInsertDeclaration(module, id, overloadedTys);
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
  // A non-default `#llvm.fenv` attribute selects the constrained variant of the
  // named intrinsic. The IRBuilder's constrained floating-point state
  // (configured by ConstrainedFPStateRAII) supplies the rounding mode and
  // exception behavior.
  if (op.getFenvAttr()) {
    llvm::Intrinsic::ID base =
        llvm::Intrinsic::lookupIntrinsicID(op.getIntrin());
    if (!base)
      return op.emitError("could not find LLVM intrinsic: ") << op.getIntrin();
    llvm::Intrinsic::ID constrainedID = getConstrainedIntrinsicFor(base);
    if (!constrainedID)
      return op.emitError("no constrained intrinsic is available for '")
             << op.getIntrin() << "' carrying a 'fenv' attribute";
    if (op.getNumResults() != 1)
      return op.emitError("constrained lowering of 'llvm.call_intrinsic' "
                          "requires exactly one result");
    return emitConstrainedFPCall(op, constrainedID, op.getArgs(),
                                 op.getResult(0), builder, moduleTranslation);
  }

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

/// Recursively converts an MLIR metadata attribute to an LLVM metadata node.
static llvm::Metadata *
convertMetadataAttr(Attribute attr, llvm::IRBuilderBase &builder,
                    LLVM::ModuleTranslation &moduleTranslation) {
  return llvm::TypeSwitch<Attribute, llvm::Metadata *>(attr)
      .Case<LLVM::MDStringAttr>([&](auto a) -> llvm::Metadata * {
        return llvm::MDString::get(builder.getContext(),
                                   a.getValue().getValue());
      })
      .Case<LLVM::MDConstantAttr>([&](auto a) -> llvm::Metadata * {
        IntegerAttr intAttr = llvm::dyn_cast<IntegerAttr>(a.getValue());
        if (!intAttr)
          return nullptr;
        return llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
            llvm::Type::getIntNTy(builder.getContext(),
                                  intAttr.getType().getIntOrFloatBitWidth()),
            intAttr.getValue()));
      })
      .Case<LLVM::MDFuncAttr>([&](auto a) -> llvm::Metadata * {
        if (llvm::Function *fn =
                moduleTranslation.lookupFunction(a.getName().getValue()))
          return llvm::ValueAsMetadata::get(fn);
        return nullptr;
      })
      .Case<LLVM::MDNodeAttr>([&](auto a) -> llvm::Metadata * {
        SmallVector<llvm::Metadata *> operands;
        for (Attribute op : a.getOperands())
          operands.push_back(
              convertMetadataAttr(op, builder, moduleTranslation));
        return llvm::MDNode::get(builder.getContext(), operands);
      })
      .Default([](auto) -> llvm::Metadata * { return nullptr; });
}

static void convertNamedMetadataOp(StringRef metadataName, ArrayAttr nodes,
                                   llvm::IRBuilderBase &builder,
                                   LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Module *llvmModule = moduleTranslation.getLLVMModule();
  llvm::NamedMDNode *namedMD =
      llvmModule->getOrInsertNamedMetadata(metadataName);
  for (Attribute nodeAttr : nodes) {
    llvm::Metadata *md =
        convertMetadataAttr(nodeAttr, builder, moduleTranslation);
    if (auto *mdNode = llvm::dyn_cast_or_null<llvm::MDNode>(md))
      namedMD->addOperand(mdNode);
  }
}

static void convertLinkerOptionsOp(ArrayAttr options,
                                   llvm::IRBuilderBase &builder,
                                   LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Module *llvmModule = moduleTranslation.getLLVMModule();
  llvm::LLVMContext &context = llvmModule->getContext();
  llvm::NamedMDNode *linkerMDNode =
      llvmModule->getOrInsertNamedMetadata("llvm.linker.options");
  SmallVector<llvm::Metadata *> mdNodes;
  mdNodes.reserve(options.size());
  for (auto s : options.getAsRange<StringAttr>()) {
    auto *mdNode = llvm::MDString::get(context, s.getValue());
    mdNodes.push_back(mdNode);
  }

  auto *listMDNode = llvm::MDTuple::get(context, mdNodes);
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
      auto getFuncMetadata = [&](FlatSymbolRefAttr sym) -> llvm::Metadata * {
        if (!sym)
          return nullptr;
        if (llvm::Function *fn =
                moduleTranslation.lookupFunction(sym.getValue()))
          return llvm::ValueAsMetadata::get(fn);
        return nullptr;
      };
      llvm::Metadata *fromMetadata = getFuncMetadata(entry.getFrom());
      llvm::Metadata *toMetadata = getFuncMetadata(entry.getTo());

      llvm::Metadata *vals[] = {
          fromMetadata, toMetadata,
          mdb.createConstant(llvm::ConstantInt::get(
              llvm::Type::getInt64Ty(context), entry.getCount()))};
      nodes.push_back(llvm::MDNode::get(context, vals));
    }
    return llvm::MDTuple::getDistinct(context, nodes);
  }
  // Handle ArrayAttr of StringAttrs (e.g. "riscv-isa") by converting back to
  // an MDTuple of MDStrings for a lossless round-trip.
  if (llvm::all_of(arrayAttr, [](Attribute a) { return isa<StringAttr>(a); })) {
    assert(!arrayAttr.empty() &&
           "empty string-array is invalid per ModuleFlagAttr::verify");
    for (StringAttr strAttr : arrayAttr.getAsRange<StringAttr>())
      nodes.push_back(llvm::MDString::get(context, strAttr.getValue()));
    return llvm::MDTuple::get(context, nodes);
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
  auto convertIntegerAttr = [&](IntegerAttr intAttr) -> llvm::Metadata * {
    return llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
        llvm::Type::getInt32Ty(builder.getContext()), intAttr.getInt()));
  };
  for (auto flagAttr : flags.getAsRange<ModuleFlagAttrInterface>()) {
    llvm::Metadata *valueMetadata =
        llvm::TypeSwitch<Attribute, llvm::Metadata *>(
            flagAttr.getModuleFlagValue())
            .Case([&](StringAttr strAttr) {
              return llvm::MDString::get(builder.getContext(),
                                         strAttr.getValue());
            })
            .Case([&](IntegerAttr intAttr) {
              return convertIntegerAttr(intAttr);
            })
            .Case([&](IntrinsicIntegerAttrInterface intAttr) {
              return convertIntegerAttr(intAttr.getIntegerAttr());
            })
            .Case([&](ArrayAttr arrayAttr) {
              return convertModuleFlagValue(
                  flagAttr.getModuleFlagKey().getValue(), arrayAttr, builder,
                  moduleTranslation);
            })
            .Case([&](ModuleFlagProfileSummaryAttr summaryAttr) {
              return convertModuleFlagProfileSummaryAttr(
                  flagAttr.getModuleFlagKey().getValue(), summaryAttr, builder,
                  moduleTranslation);
            })
            .Default([](auto) { return nullptr; });

    assert(valueMetadata && "expected valid metadata");
    llvmModule->addModuleFlag(
        convertModFlagBehaviorToLLVM(flagAttr.getModuleFlagBehavior()),
        flagAttr.getModuleFlagKey().getValue(), valueMetadata);
  }
}

/// Looks up the GlobalValue and FunctionType for a callee symbol that is not a
/// regular LLVM function (i.e. an alias or ifunc). Returns the lowered
/// GlobalValue and FunctionType derived from \p calleeFuncType.
static std::pair<llvm::GlobalValue *, llvm::FunctionType *>
lookupNonFunctionSymbolCallee(FlatSymbolRefAttr attr, mlir::Type calleeFuncType,
                              Operation &opInst,
                              LLVM::ModuleTranslation &moduleTranslation) {
  Operation *moduleOp = parentLLVMModule(&opInst);
  Operation *calleeOp =
      moduleTranslation.symbolTable().lookupSymbolIn(moduleOp, attr);
  llvm::FunctionType *calleeType = llvm::cast<llvm::FunctionType>(
      moduleTranslation.convertType(calleeFuncType));
  llvm::GlobalValue *calleeGV;
  if (isa<LLVM::AliasOp>(calleeOp))
    calleeGV = moduleTranslation.lookupAlias(calleeOp);
  else
    calleeGV = moduleTranslation.lookupIFunc(calleeOp);
  return {calleeGV, calleeType};
}

static llvm::DILocalScope *
getLocalScopeFromLoc(llvm::IRBuilderBase &builder, Location loc,
                     LLVM::ModuleTranslation &moduleTranslation) {
  if (auto scopeLoc =
          loc->findInstanceOf<FusedLocWith<LLVM::DILocalScopeAttr>>())
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

  // Operations carrying an `#llvm.fenv` attribute are lowered to the matching
  // constrained floating-point intrinsic. Configure the IRBuilder's constrained
  // floating-point state (as clang's CGFPOptionsRAII does) so that ordinary
  // floating-point operations are emitted as the corresponding
  // `llvm.experimental.constrained.*` intrinsics automatically. The state is
  // restored when this guard goes out of scope. Operations the IRBuilder cannot
  // constrain automatically (`llvm.fcmp`, the math intrinsics, and
  // `llvm.call_intrinsic`) handle the constrained lowering themselves from
  // their generated `llvmBuilder` code (see emitConstrainedFCmp,
  // emitConstrainedFPIntrinsic, and convertCallLLVMIntrinsicOp).
  ConstrainedFPStateRAII constrainedFPState(builder);
  if (auto fenvOp = dyn_cast<LLVM::FPEnvConstrainedOpInterface>(opInst);
      fenvOp && fenvOp.getFenvAttr())
    constrainedFPState.enable(getConstrainedRoundingMode(fenvOp),
                              getConstrainedExceptionBehavior(fenvOp));

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
        auto [calleeGV, calleeType] = lookupNonFunctionSymbolCallee(
            attr, callOp.getCalleeFunctionType(), opInst, moduleTranslation);
        call = builder.CreateCall(calleeType, calleeGV, operandsRef, opBundles);
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
    if (callOp.getNoreturnAttr())
      call->addFnAttr(llvm::Attribute::NoReturn);
    if (callOp.getOptsizeAttr())
      call->addFnAttr(llvm::Attribute::OptimizeForSize);
    if (callOp.getMinsizeAttr())
      call->addFnAttr(llvm::Attribute::MinSize);
    if (callOp.getSaveRegParamsAttr())
      call->addFnAttr(llvm::Attribute::get(moduleTranslation.getLLVMContext(),
                                           "save-reg-params"));
    if (callOp.getBuiltinAttr())
      call->addFnAttr(llvm::Attribute::Builtin);
    if (callOp.getNobuiltinAttr())
      call->addFnAttr(llvm::Attribute::NoBuiltin);
    if (callOp.getReturnsTwiceAttr())
      call->addFnAttr(llvm::Attribute::ReturnsTwice);
    if (callOp.getColdAttr())
      call->addFnAttr(llvm::Attribute::Cold);
    if (callOp.getHotAttr())
      call->addFnAttr(llvm::Attribute::Hot);
    if (callOp.getNoduplicateAttr())
      call->addFnAttr(llvm::Attribute::NoDuplicate);
    if (callOp.getNoInlineAttr())
      call->addFnAttr(llvm::Attribute::NoInline);
    if (callOp.getAlwaysInlineAttr())
      call->addFnAttr(llvm::Attribute::AlwaysInline);
    if (callOp.getInlineHintAttr())
      call->addFnAttr(llvm::Attribute::InlineHint);
    if (callOp.getNoCallerSavedRegistersAttr())
      call->addFnAttr(llvm::Attribute::get(moduleTranslation.getLLVMContext(),
                                           "no_caller_saved_registers"));
    if (callOp.getNocallbackAttr())
      call->addFnAttr(llvm::Attribute::NoCallback);
    if (StringAttr modFormat = callOp.getModularFormatAttr())
      call->addFnAttr(llvm::Attribute::get(moduleTranslation.getLLVMContext(),
                                           "modular-format",
                                           modFormat.getValue()));
    if (StringAttr zcsr = callOp.getZeroCallUsedRegsAttr())
      call->addFnAttr(llvm::Attribute::get(moduleTranslation.getLLVMContext(),
                                           "zero-call-used-regs",
                                           zcsr.getValue()));
    if (StringAttr trapFunc = callOp.getTrapFuncNameAttr())
      call->addFnAttr(llvm::Attribute::get(moduleTranslation.getLLVMContext(),
                                           "trap-func-name",
                                           trapFunc.getValue()));

    if (ArrayAttr noBuiltins = callOp.getNobuiltinsAttr()) {
      if (noBuiltins.empty())
        call->addFnAttr(llvm::Attribute::get(moduleTranslation.getLLVMContext(),
                                             "no-builtins"));

      moduleTranslation.convertFunctionAttrCollection(
          noBuiltins, call, ModuleTranslation::convertNoBuiltin);
    }

    moduleTranslation.convertFunctionAttrCollection(
        callOp.getDefaultFuncAttrsAttr(), call,
        ModuleTranslation::convertDefaultFuncAttr);

    if (llvm::Attribute attr =
            moduleTranslation.convertAllocsizeAttr(callOp.getAllocsizeAttr());
        attr.isValid())
      call->addFnAttr(attr);

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
                              convertModRefInfoToLLVM(memAttr.getOther())) |
          llvm::MemoryEffects(llvm::MemoryEffects::Location::ErrnoMem,
                              convertModRefInfoToLLVM(memAttr.getErrnoMem())) |
          llvm::MemoryEffects(
              llvm::MemoryEffects::Location::TargetMem0,
              convertModRefInfoToLLVM(memAttr.getTargetMem0())) |
          llvm::MemoryEffects(llvm::MemoryEffects::Location::TargetMem1,
                              convertModRefInfoToLLVM(memAttr.getTargetMem1()));
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
      if (llvm::Function *function =
              moduleTranslation.lookupFunction(attr.getValue())) {
        result = builder.CreateInvoke(
            function, moduleTranslation.lookupBlock(invOp.getSuccessor(0)),
            moduleTranslation.lookupBlock(invOp.getSuccessor(1)), operandsRef,
            opBundles);
      } else {
        auto [calleeGV, calleeType] = lookupNonFunctionSymbolCallee(
            attr, invOp.getCalleeFunctionType(), opInst, moduleTranslation);
        result = builder.CreateInvoke(
            calleeType, calleeGV,
            moduleTranslation.lookupBlock(invOp.getSuccessor(0)),
            moduleTranslation.lookupBlock(invOp.getSuccessor(1)), operandsRef,
            opBundles);
      }
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
    llvm::UncondBrInst *branch =
        builder.CreateBr(moduleTranslation.lookupBlock(brOp.getSuccessor()));
    moduleTranslation.mapBranch(&opInst, branch);
    moduleTranslation.setLoopMetadata(&opInst, branch);
    return success();
  }
  if (auto condbrOp = dyn_cast<LLVM::CondBrOp>(opInst)) {
    llvm::CondBrInst *branch = builder.CreateCondBr(
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

static LogicalResult
amendOperationImpl(Operation &op, ArrayRef<llvm::Instruction *> instructions,
                   NamedAttribute attribute,
                   LLVM::ModuleTranslation &moduleTranslation) {
  StringRef name = attribute.getName();
  if (name == LLVMDialect::getMmraAttrName()) {
    SmallVector<llvm::MMRAMetadata::TagT> tags;
    if (auto oneTag = dyn_cast<LLVM::MMRATagAttr>(attribute.getValue())) {
      tags.emplace_back(oneTag.getPrefix(), oneTag.getSuffix());
    } else if (auto manyTags = dyn_cast<ArrayAttr>(attribute.getValue())) {
      for (Attribute attr : manyTags) {
        auto tag = dyn_cast<MMRATagAttr>(attr);
        if (!tag)
          return op.emitOpError(
              "MMRA annotations array contains value that isn't an MMRA tag");
        tags.emplace_back(tag.getPrefix(), tag.getSuffix());
      }
    } else {
      return op.emitOpError(
          "llvm.mmra is something other than an MMRA tag or an array of them");
    }
    llvm::MDTuple *mmraMd =
        llvm::MMRAMetadata::getMD(moduleTranslation.getLLVMContext(), tags);
    if (!mmraMd) {
      // Empty list, canonicalizes to nothing
      return success();
    }
    for (llvm::Instruction *inst : instructions)
      inst->setMetadata(llvm::LLVMContext::MD_mmra, mmraMd);
    return success();
  }
  return success();
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

  /// Handle some metadata that is represented as a discardable attribute.
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    return amendOperationImpl(*op, instructions, attribute, moduleTranslation);
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
