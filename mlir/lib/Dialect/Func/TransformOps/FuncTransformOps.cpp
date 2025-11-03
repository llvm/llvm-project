//===- FuncTransformOps.cpp - Implementation of CF transform ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Utils/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Apply...ConversionPatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyFuncToLLVMConversionPatternsOp::populatePatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  populateFuncToLLVMConversionPatterns(
      static_cast<LLVMTypeConverter &>(typeConverter), patterns);
}

LogicalResult
transform::ApplyFuncToLLVMConversionPatternsOp::verifyTypeConverter(
    transform::TypeConverterBuilderOpInterface builder) {
  if (builder.getTypeConverterType() != "LLVMTypeConverter")
    return emitOpError("expected LLVMTypeConverter");
  return success();
}

//===----------------------------------------------------------------------===//
// CastAndCallOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::CastAndCallOp::apply(transform::TransformRewriter &rewriter,
                                transform::TransformResults &results,
                                transform::TransformState &state) {
  SmallVector<Value> inputs;
  if (getInputs())
    llvm::append_range(inputs, state.getPayloadValues(getInputs()));

  SetVector<Value> outputs;
  if (getOutputs()) {
    outputs.insert_range(state.getPayloadValues(getOutputs()));

    // Verify that the set of output values to be replaced is unique.
    if (outputs.size() !=
        llvm::range_size(state.getPayloadValues(getOutputs()))) {
      return emitSilenceableFailure(getLoc())
             << "cast and call output values must be unique";
    }
  }

  // Get the insertion point for the call.
  auto insertionOps = state.getPayloadOps(getInsertionPoint());
  if (!llvm::hasSingleElement(insertionOps)) {
    return emitSilenceableFailure(getLoc())
           << "Only one op can be specified as an insertion point";
  }
  bool insertAfter = getInsertAfter();
  Operation *insertionPoint = *insertionOps.begin();

  // Check that all inputs dominate the insertion point, and the insertion
  // point dominates all users of the outputs.
  DominanceInfo dom(insertionPoint);
  for (Value output : outputs) {
    for (Operation *user : output.getUsers()) {
      // If we are inserting after the insertion point operation, the
      // insertion point operation must properly dominate the user. Otherwise
      // basic dominance is enough.
      bool doesDominate = insertAfter
                              ? dom.properlyDominates(insertionPoint, user)
                              : dom.dominates(insertionPoint, user);
      if (!doesDominate) {
        return emitDefiniteFailure()
               << "User " << user << " is not dominated by insertion point "
               << insertionPoint;
      }
    }
  }

  for (Value input : inputs) {
    // If we are inserting before the insertion point operation, the
    // input must properly dominate the insertion point operation. Otherwise
    // basic dominance is enough.
    bool doesDominate = insertAfter
                            ? dom.dominates(input, insertionPoint)
                            : dom.properlyDominates(input, insertionPoint);
    if (!doesDominate) {
      return emitDefiniteFailure()
             << "input " << input << " does not dominate insertion point "
             << insertionPoint;
    }
  }

  // Get the function to call. This can either be specified by symbol or as a
  // transform handle.
  func::FuncOp targetFunction = nullptr;
  if (getFunctionName()) {
    targetFunction = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        insertionPoint, *getFunctionName());
    if (!targetFunction) {
      return emitDefiniteFailure()
             << "unresolved symbol " << *getFunctionName();
    }
  } else if (getFunction()) {
    auto payloadOps = state.getPayloadOps(getFunction());
    if (!llvm::hasSingleElement(payloadOps)) {
      return emitDefiniteFailure() << "requires a single function to call";
    }
    targetFunction = dyn_cast<func::FuncOp>(*payloadOps.begin());
    if (!targetFunction) {
      return emitDefiniteFailure() << "invalid non-function callee";
    }
  } else {
    llvm_unreachable("Invalid CastAndCall op without a function to call");
    return emitDefiniteFailure();
  }

  // Verify that the function argument and result lengths match the inputs and
  // outputs given to this op.
  if (targetFunction.getNumArguments() != inputs.size()) {
    return emitSilenceableFailure(targetFunction.getLoc())
           << "mismatch between number of function arguments "
           << targetFunction.getNumArguments() << " and number of inputs "
           << inputs.size();
  }
  if (targetFunction.getNumResults() != outputs.size()) {
    return emitSilenceableFailure(targetFunction.getLoc())
           << "mismatch between number of function results "
           << targetFunction->getNumResults() << " and number of outputs "
           << outputs.size();
  }

  // Gather all specified converters.
  mlir::TypeConverter converter;
  if (!getRegion().empty()) {
    for (Operation &op : getRegion().front()) {
      cast<transform::TypeConverterBuilderOpInterface>(&op)
          .populateTypeMaterializations(converter);
    }
  }

  if (insertAfter)
    rewriter.setInsertionPointAfter(insertionPoint);
  else
    rewriter.setInsertionPoint(insertionPoint);

  for (auto [input, type] :
       llvm::zip_equal(inputs, targetFunction.getArgumentTypes())) {
    if (input.getType() != type) {
      Value newInput = converter.materializeSourceConversion(
          rewriter, input.getLoc(), type, input);
      if (!newInput) {
        return emitDefiniteFailure() << "Failed to materialize conversion of "
                                     << input << " to type " << type;
      }
      input = newInput;
    }
  }

  auto callOp = func::CallOp::create(rewriter, insertionPoint->getLoc(),
                                     targetFunction, inputs);

  // Cast the call results back to the expected types. If any conversions fail
  // this is a definite failure as the call has been constructed at this point.
  for (auto [output, newOutput] :
       llvm::zip_equal(outputs, callOp.getResults())) {
    Value convertedOutput = newOutput;
    if (output.getType() != newOutput.getType()) {
      convertedOutput = converter.materializeTargetConversion(
          rewriter, output.getLoc(), output.getType(), newOutput);
      if (!convertedOutput) {
        return emitDefiniteFailure()
               << "Failed to materialize conversion of " << newOutput
               << " to type " << output.getType();
      }
    }
    rewriter.replaceAllUsesExcept(output, convertedOutput, callOp);
  }
  results.set(cast<OpResult>(getResult()), {callOp});
  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::CastAndCallOp::verify() {
  if (!getRegion().empty()) {
    for (Operation &op : getRegion().front()) {
      if (!isa<transform::TypeConverterBuilderOpInterface>(&op)) {
        InFlightDiagnostic diag = emitOpError()
                                  << "expected children ops to implement "
                                     "TypeConverterBuilderOpInterface";
        diag.attachNote(op.getLoc()) << "op without interface";
        return diag;
      }
    }
  }
  if (!getFunction() && !getFunctionName()) {
    return emitOpError() << "expected a function handle or name to call";
  }
  if (getFunction() && getFunctionName()) {
    return emitOpError() << "function handle and name are mutually exclusive";
  }
  return success();
}

void transform::CastAndCallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getInsertionPointMutable(), effects);
  if (getInputs())
    transform::onlyReadsHandle(getInputsMutable(), effects);
  if (getOutputs())
    transform::onlyReadsHandle(getOutputsMutable(), effects);
  if (getFunction())
    transform::onlyReadsHandle(getFunctionMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ReplaceFuncSignatureOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ReplaceFuncSignatureOp::apply(transform::TransformRewriter &rewriter,
                                         transform::TransformResults &results,
                                         transform::TransformState &state) {
  auto payloadOps = state.getPayloadOps(getModule());
  if (!llvm::hasSingleElement(payloadOps))
    return emitDefiniteFailure() << "requires a single module to operate on";

  auto targetModuleOp = dyn_cast<ModuleOp>(*payloadOps.begin());
  if (!targetModuleOp)
    return emitSilenceableFailure(getLoc())
           << "target is expected to be module operation";

  func::FuncOp funcOp =
      targetModuleOp.lookupSymbol<func::FuncOp>(getFunctionName());
  if (!funcOp)
    return emitSilenceableFailure(getLoc())
           << "function with name '" << getFunctionName() << "' not found";

  unsigned numArgs = funcOp.getNumArguments();
  unsigned numResults = funcOp.getNumResults();
  // Check that the number of arguments and results matches the
  // interchange sizes.
  if (numArgs != getArgsInterchange().size())
    return emitSilenceableFailure(getLoc())
           << "function with name '" << getFunctionName() << "' has " << numArgs
           << " arguments, but " << getArgsInterchange().size()
           << " args interchange were given";

  if (numResults != getResultsInterchange().size())
    return emitSilenceableFailure(getLoc())
           << "function with name '" << getFunctionName() << "' has "
           << numResults << " results, but " << getResultsInterchange().size()
           << " results interchange were given";

  // Check that the args and results interchanges are unique.
  SetVector<unsigned> argsInterchange, resultsInterchange;
  argsInterchange.insert_range(getArgsInterchange());
  resultsInterchange.insert_range(getResultsInterchange());
  if (argsInterchange.size() != getArgsInterchange().size())
    return emitSilenceableFailure(getLoc())
           << "args interchange must be unique";

  if (resultsInterchange.size() != getResultsInterchange().size())
    return emitSilenceableFailure(getLoc())
           << "results interchange must be unique";

  // Check that the args and results interchange indices are in bounds.
  for (unsigned index : argsInterchange) {
    if (index >= numArgs) {
      return emitSilenceableFailure(getLoc())
             << "args interchange index " << index
             << " is out of bounds for function with name '"
             << getFunctionName() << "' with " << numArgs << " arguments";
    }
  }
  for (unsigned index : resultsInterchange) {
    if (index >= numResults) {
      return emitSilenceableFailure(getLoc())
             << "results interchange index " << index
             << " is out of bounds for function with name '"
             << getFunctionName() << "' with " << numResults << " results";
    }
  }

  llvm::SmallVector<int> oldArgToNewArg(argsInterchange.size());
  for (auto [newArgIdx, oldArgIdx] : llvm::enumerate(argsInterchange))
    oldArgToNewArg[oldArgIdx] = newArgIdx;

  llvm::SmallVector<int> oldResToNewRes(resultsInterchange.size());
  for (auto [newResIdx, oldResIdx] : llvm::enumerate(resultsInterchange))
    oldResToNewRes[oldResIdx] = newResIdx;

  FailureOr<func::FuncOp> newFuncOpOrFailure = func::replaceFuncWithNewMapping(
      rewriter, funcOp, oldArgToNewArg, oldResToNewRes);
  if (failed(newFuncOpOrFailure))
    return emitSilenceableFailure(getLoc())
           << "failed to replace function signature '" << getFunctionName()
           << "' with new order";

  if (getAdjustFuncCalls()) {
    SmallVector<func::CallOp> callOps;
    targetModuleOp.walk([&](func::CallOp callOp) {
      if (callOp.getCallee() == getFunctionName().getRootReference().getValue())
        callOps.push_back(callOp);
    });

    for (func::CallOp callOp : callOps)
      func::replaceCallOpWithNewMapping(rewriter, callOp, oldArgToNewArg,
                                        oldResToNewRes);
  }

  results.set(cast<OpResult>(getTransformedModule()), {targetModuleOp});
  results.set(cast<OpResult>(getTransformedFunction()), {*newFuncOpOrFailure});

  return DiagnosedSilenceableFailure::success();
}

void transform::ReplaceFuncSignatureOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getModuleMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// DeduplicateFuncArgsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::DeduplicateFuncArgsOp::apply(transform::TransformRewriter &rewriter,
                                        transform::TransformResults &results,
                                        transform::TransformState &state) {
  auto payloadOps = state.getPayloadOps(getModule());
  if (!llvm::hasSingleElement(payloadOps))
    return emitDefiniteFailure() << "requires a single module to operate on";

  auto targetModuleOp = dyn_cast<ModuleOp>(*payloadOps.begin());
  if (!targetModuleOp)
    return emitSilenceableFailure(getLoc())
           << "target is expected to be module operation";

  func::FuncOp funcOp =
      targetModuleOp.lookupSymbol<func::FuncOp>(getFunctionName());
  if (!funcOp)
    return emitSilenceableFailure(getLoc())
           << "function with name '" << getFunctionName() << "' is not found";

  auto transformationResult =
      func::deduplicateArgsOfFuncOp(rewriter, funcOp, targetModuleOp);
  if (failed(transformationResult))
    return emitSilenceableFailure(getLoc())
           << "failed to deduplicate function arguments of function "
           << funcOp.getName();

  auto [newFuncOp, newCallOp] = *transformationResult;

  results.set(cast<OpResult>(getTransformedModule()), {targetModuleOp});
  results.set(cast<OpResult>(getTransformedFunction()), {newFuncOp});

  return DiagnosedSilenceableFailure::success();
}

void transform::DeduplicateFuncArgsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getModuleMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class FuncTransformDialectExtension
    : public transform::TransformDialectExtension<
          FuncTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuncTransformDialectExtension)

  using Base::Base;

  void init() {
    declareGeneratedDialect<LLVM::LLVMDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.cpp.inc"

void mlir::func::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<FuncTransformDialectExtension>();
}
