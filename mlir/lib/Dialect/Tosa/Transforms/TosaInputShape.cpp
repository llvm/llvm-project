//===- TosaInputShape.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass that overrides the dynamic input shapes of function arguments to
// specified static shapes. If a specified static shape conflicts with the
// static dimensions in an original input shape, an error is reported.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSAINPUTSHAPE
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {

typedef std::pair<size_t, SmallVector<int64_t>> IdxAndShape;

FailureOr<IdxAndShape> parseInputShape(Location loc, StringRef input) {
  if (!input.consume_front("arg")) {
    emitError(loc) << "expected prefix 'arg' at the start of " << input;
    return failure();
  }

  const size_t colonPos = input.find(':');
  if (colonPos == StringRef::npos) {
    emitError(loc) << "expected ':' after argument index in '" << input << "'";
    return failure();
  }

  const StringRef indexStr = input.substr(0, colonPos);
  input = input.substr(colonPos + 1);

  size_t index;
  if (indexStr.getAsInteger(10, index) || index < 0) {
    emitError(loc) << "invalid argument index, got " << indexStr;
    return failure();
  }

  SmallVector<int64_t> shape;
  while (!input.empty()) {
    const size_t xPos = input.find("x");
    StringRef dimStr;
    if (xPos == StringRef::npos) {
      dimStr = input;
      input = "";
    } else {
      dimStr = input.substr(0, xPos);
      input = input.substr(xPos + 1);
    }

    int64_t dimVal;
    if (dimStr.getAsInteger(10, dimVal) || dimVal <= 0) {
      return failure();
    }
    shape.push_back(dimVal);
  }

  const auto idxAndShape = std::make_pair(index, shape);
  return {idxAndShape};
}

// Parse input shape arguments from command line input. Returns parsed
// static shapes and an optional error message.
// For example:
//   "args=arg0:5x10,arg8:3x9" => {{{0, {5, 10}}, {8, {3, 9}}}, ""}
//   "args=arg0:" => {{}, "error message"}
FailureOr<SmallVector<IdxAndShape>>
parseInputShapes(Location loc, const std::vector<std::string> &args) {
  SmallVector<IdxAndShape> inputShapes;
  for (const std::string &arg : args) {
    const auto maybeInputShape = parseInputShape(loc, arg);
    if (failed(maybeInputShape))
      return failure();
    inputShapes.push_back(maybeInputShape.value());
  }
  return inputShapes;
}

struct TosaInputShape : public tosa::impl::TosaInputShapeBase<TosaInputShape> {
public:
  TosaInputShape() = default;

  explicit TosaInputShape(std::vector<std::string> args) : TosaInputShape() {
    this->args = args;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    const Location unknownLoc = UnknownLoc::get(context);
    const auto maybeArgsParsed = parseInputShapes(unknownLoc, args);
    if (failed(maybeArgsParsed))
      return;
    const SmallVector<IdxAndShape> argsParsed = maybeArgsParsed.value();
    func::FuncOp func = getOperation();

    const auto getUpdatedTensorType =
        [&](size_t argIdx, ArrayRef<Type> argTypes,
            ArrayRef<int64_t> requestedShape) -> FailureOr<Type> {
      const size_t numInputs = argTypes.size();
      if (argIdx >= numInputs)
        return func.emitError()
               << "provided arg index " << argIdx
               << " is larger than number of inputs " << numInputs << ".";

      auto tensorType = dyn_cast<TensorType>(argTypes[argIdx]);
      if (!tensorType)
        return func.emitError()
               << "expected tensor type, got " << argTypes[argIdx];

      const ArrayRef<int64_t> originalShape = tensorType.getShape();
      if (failed(verifyCompatibleShape(originalShape, requestedShape)))
        return func.emitError()
               << "arg" << argIdx
               << " has incompatible shape with requested input shape ("
               << requestedShape << "), got " << tensorType;
      return tensorType.cloneWith(requestedShape, tensorType.getElementType());
    };

    // Update argument shapes in the entry block
    Block &entryBlock = func.getBody().front();
    const SmallVector<Type> argTypes(entryBlock.getArgumentTypes());
    for (const auto &[argIdx, shape] : argsParsed) {
      FailureOr<Type> newTensorType =
          getUpdatedTensorType(argIdx, argTypes, shape);
      if (failed(newTensorType))
        return signalPassFailure();

      entryBlock.getArgument(argIdx).setType(newTensorType.value());
    }

    // Get new func argument types
    const FunctionType oldFunctionType = func.getFunctionType();
    const ArrayRef<Type> oldInputTypes = oldFunctionType.getInputs();
    SmallVector<Type> newInputs(oldInputTypes.begin(), oldInputTypes.end());
    for (const auto &[argIdx, shape] : argsParsed) {
      FailureOr<Type> newTensorType =
          getUpdatedTensorType(argIdx, oldInputTypes, shape);
      if (failed(newTensorType))
        return signalPassFailure();

      newInputs[argIdx] = newTensorType.value();
    }

    // Update function signature
    Block &lastBlock = func.getBody().back();
    const Operation *terminator = lastBlock.getTerminator();
    SmallVector<Type> newResults;
    if (auto returnOp = dyn_cast_or_null<func::ReturnOp>(terminator)) {
      const auto types = returnOp.getOperandTypes();
      newResults.assign(types.begin(), types.end());
    } else {
      const auto types = oldFunctionType.getResults();
      newResults.assign(types.begin(), types.end());
    }
    const FunctionType newFunctionType =
        oldFunctionType.clone(newInputs, newResults);
    func.setFunctionType(newFunctionType);
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::tosa::createTosaInputShapePass(std::vector<std::string> args) {
  return std::make_unique<TosaInputShape>(args);
}
