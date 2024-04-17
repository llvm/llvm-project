//===- FuncTransformOps.cpp - Implementation of CF transform ops ---===//
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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

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
    for (auto output : state.getPayloadValues(getOutputs()))
      outputs.insert(output);

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

  auto callOp = rewriter.create<func::CallOp>(insertionPoint->getLoc(),
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
  transform::onlyReadsHandle(getInsertionPoint(), effects);
  if (getInputs())
    transform::onlyReadsHandle(getInputs(), effects);
  if (getOutputs())
    transform::onlyReadsHandle(getOutputs(), effects);
  if (getFunction())
    transform::onlyReadsHandle(getFunction(), effects);
  transform::producesHandle(getResult(), effects);
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
