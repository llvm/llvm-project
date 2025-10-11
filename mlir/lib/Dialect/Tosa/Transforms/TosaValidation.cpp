//===- TosaValidation.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Validate if TOSA dialect input matchs with the specification for given
// requirements.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TargetEnv.h"
#include "mlir/Dialect/Tosa/IR/TosaProfileCompliance.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"

#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringExtras.h"

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSAVALIDATION
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {

static LogicalResult
checkConstantOperands(Operation *op, ArrayRef<unsigned int> operandIndices) {
  for (const auto index : operandIndices) {
    Attribute attr;
    if (!matchPattern(op->getOperand(index), m_Constant(&attr))) {
      return op->emitOpError("expected compile time resolvable constant, but "
                             "got variable value for operand #")
             << index;
    }
  }
  return success();
}

static LogicalResult checkConstantOperandMul(Operation *op,
                                             const TargetEnv &env) {
  if (!env.allows(Extension::dynamic) && isa<tosa::MulOp>(op)) {
    // Check 'shift'
    return checkConstantOperands(op, {2});
  }
  return success();
}

static LogicalResult checkConstantOperandTable(Operation *op,
                                               const TargetEnv &env) {
  if (!env.allows(Extension::dynamic) && isa<tosa::TableOp>(op)) {
    // Check 'table'
    return checkConstantOperands(op, {1});
  }
  return success();
}

static LogicalResult checkConstantOperandPad(Operation *op,
                                             const TargetEnv &env) {
  if (auto padOp = dyn_cast<tosa::PadOp>(op)) {
    // Assume this op is zero-padding if padConst is not presented
    if (!env.allows(Extension::dynamic) && padOp.getPadConst())
      // Check 'pad_const'
      // Note: 'padding' (operand 1) is not checked as it is a tosa.shape type
      return checkConstantOperands(op, {2});
  }
  return success();
}

static LogicalResult checkConstantOperandRescale(Operation *op,
                                                 const TargetEnv &env) {
  if (!env.allows(Extension::dynamic) && isa<tosa::RescaleOp>(op)) {
    // Check 'multiplier', 'shift', 'input_zp' and 'output_zp'
    return checkConstantOperands(op, {1, 2, 3, 4});
  }
  return success();
}

template <typename T>
static LogicalResult checkConstantOperandConvOps(Operation *op,
                                                 const TargetEnv &env) {
  if (!env.allows(Extension::dynamic) && isa<T>(op)) {
    // Check 'input_zp' and 'weight_zp'
    return checkConstantOperands(op, {3, 4});
  }
  return success();
}

static LogicalResult checkConstantOperandMatMul(Operation *op,
                                                const TargetEnv &env) {
  if (!env.allows(Extension::dynamic) && isa<tosa::MatMulOp>(op)) {
    // Check 'A_zp' and 'B_zp'
    return checkConstantOperands(op, {2, 3});
  }
  return success();
}

static LogicalResult checkConstantOperandAvgPool2d(Operation *op,
                                                   const TargetEnv &env) {
  if (!env.allows(Extension::dynamic) && isa<tosa::AvgPool2dOp>(op)) {
    // Check 'input_zp' and 'output_zp'
    return checkConstantOperands(op, {1, 2});
  }
  return success();
}

static LogicalResult checkConstantOperandNegate(Operation *op,
                                                const TargetEnv &env) {
  if (!env.allows(Extension::dynamic) && isa<tosa::NegateOp>(op)) {
    // Check 'input1_zp' and 'output_zp'
    return checkConstantOperands(op, {1, 2});
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TOSA Validation Pass.
//===----------------------------------------------------------------------===//

struct TosaValidation : public tosa::impl::TosaValidationBase<TosaValidation> {
public:
  explicit TosaValidation() { populateConstantOperandChecks(); }

  explicit TosaValidation(const TosaValidationOptions &options)
      : TosaValidation() {
    this->strictOpSpecAlignment = options.strictOpSpecAlignment;
    this->allowInvalidOpDatatypeCombinations =
        options.allowInvalidOpDatatypeCombinations;
  }
  void runOnOperation() final;

  LogicalResult applyConstantOperandCheck(Operation *op) {
    for (auto &checker : constCheckers) {
      if (failed(checker(op, targetEnv)))
        return failure();
    }
    return success();
  }

  LogicalResult applyLevelCheck(Operation *op);
  LogicalResult applyAttributeCheck(Operation *op);

  // check variable read/write data types against variable declarations
  LogicalResult applyVariableCheck(Operation *op);

  // check error if conditions
  LogicalResult applyErrorIfCheck(Operation *op);

private:
  void populateConstantOperandChecks() {
    constCheckers.emplace_back(checkConstantOperandMul);
    constCheckers.emplace_back(checkConstantOperandTable);
    constCheckers.emplace_back(checkConstantOperandPad);
    constCheckers.emplace_back(checkConstantOperandRescale);
    constCheckers.emplace_back(checkConstantOperandConvOps<tosa::Conv2DOp>);
    constCheckers.emplace_back(checkConstantOperandConvOps<tosa::Conv3DOp>);
    constCheckers.emplace_back(
        checkConstantOperandConvOps<tosa::DepthwiseConv2DOp>);
    constCheckers.emplace_back(
        checkConstantOperandConvOps<tosa::TransposeConv2DOp>);
    constCheckers.emplace_back(checkConstantOperandMatMul);
    constCheckers.emplace_back(checkConstantOperandAvgPool2d);
    constCheckers.emplace_back(checkConstantOperandNegate);
  }

  LogicalResult levelCheckKernel(Operation *op, int32_t v,
                                 const StringRef checkDesc) {
    if (v > targetEnv.getLevel().MAX_KERNEL)
      return op->emitOpError() << "failed level check: " << checkDesc;
    return success();
  }

  LogicalResult levelCheckStride(Operation *op, int32_t v,
                                 const StringRef checkDesc) {
    if (v > targetEnv.getLevel().MAX_STRIDE)
      return op->emitOpError() << "failed level check: " << checkDesc;
    return success();
  }

  LogicalResult levelCheckScale(Operation *op, int32_t v,
                                const StringRef checkDesc) {
    if (v > targetEnv.getLevel().MAX_SCALE)
      return op->emitOpError() << "failed level check: " << checkDesc;
    return success();
  }

  LogicalResult levelCheckListSize(Operation *op, int32_t v,
                                   const StringRef checkDesc) {
    if (v > targetEnv.getLevel().MAX_TENSOR_LIST_SIZE)
      return op->emitOpError()
             << "failed level check for MAX_TENSOR_LIST_SIZE: " << checkDesc;
    return success();
  }

  // Perform the Level Rank check on the tensor type.
  LogicalResult levelCheckRank(Operation *op, const Type typeToCheck,
                               const StringRef operandOrResult,
                               int32_t highest_rank) {
    if (ShapedType type = dyn_cast<ShapedType>(typeToCheck)) {
      if (!type.hasRank())
        return op->emitOpError() << "failed level check: unranked tensor";
      if (type.getRank() > highest_rank)
        return op->emitOpError() << "failed level check: " << operandOrResult
                                 << " rank(shape) <= MAX_RANK";
    }
    return success();
  }

  // Perform the Level Rank check on the tensor value.
  LogicalResult levelCheckRank(Operation *op, const Value &v,
                               const StringRef operandOrResult,
                               int32_t highest_rank) {
    return levelCheckRank(op, v.getType(), operandOrResult, highest_rank);
  }

  // Perform the Level tensor size check on the tensor type.
  LogicalResult levelCheckSize(Operation *op, const Type &typeToCheck,
                               const StringRef operandOrResult);

  // Perform the Level tensor size check on the tensor value.
  LogicalResult levelCheckSize(Operation *op, const Value &v,
                               const StringRef operandOrResult) {
    return levelCheckSize(op, v.getType(), operandOrResult);
  }

  // Level check sizes of all operands and results of the operation.
  template <typename T>
  LogicalResult levelCheckSizes(T tosaOp) {
    auto op = tosaOp.getOperation();
    for (auto v : op->getOperands()) {
      if (failed(levelCheckSize(op, v, "operand")))
        return failure();
    }

    for (auto v : op->getResults()) {
      if (failed(levelCheckSize(op, v, "result")))
        return failure();
    }
    return success();
  }

  // Level check ranks of all operands, attribute and results of the operation.
  template <typename T>
  LogicalResult levelCheckRanks(T tosaOp) {
    auto op = tosaOp.getOperation();
    const TosaLevel tosaLevel = targetEnv.getLevel();
    for (auto v : op->getOperands()) {
      if (failed(levelCheckRank(op, v, "operand", tosaLevel.MAX_RANK)))
        return failure();
    }

    for (auto v : op->getResults()) {
      if (failed(levelCheckRank(op, v, "result", tosaLevel.MAX_RANK)))
        return failure();
    }
    return success();
  }

  // Level check ranks and sizes.
  LogicalResult levelCheckRanksAndSizes(Operation *op);

  // Pool Op: level check kernel/stride/pad values
  template <typename T>
  LogicalResult levelCheckPool(Operation *op) {
    if (auto poolOp = dyn_cast<T>(op)) {
      for (auto k : poolOp.getKernel()) {
        if (failed(levelCheckKernel(op, k, "kernel <= MAX_KERNEL"))) {
          return failure();
        }
      }
      for (auto s : poolOp.getStride()) {
        if (failed(levelCheckStride(op, s, "stride <= MAX_STRIDE"))) {
          return failure();
        }
      }
      for (auto p : poolOp.getPad()) {
        if (failed(levelCheckKernel(op, p, "pad <= MAX_KERNEL"))) {
          return failure();
        }
      }
    }
    return success();
  }

  // Conv Op: level check dilation/stride/pad values
  template <typename T>
  LogicalResult levelCheckConv(Operation *op) {
    if (auto convOp = dyn_cast<T>(op)) {

      for (auto k : convOp.getDilation()) {
        if (failed(levelCheckKernel(op, k, "dilation <= MAX_KERNEL"))) {
          return failure();
        }
      }
      for (auto p : convOp.getPad()) {
        if (failed(levelCheckKernel(op, p, "pad <= MAX_KERNEL"))) {
          return failure();
        }
      }
      for (auto s : convOp.getStride()) {
        if (failed(levelCheckStride(op, s, "stride <= MAX_STRIDE"))) {
          return failure();
        }
      }
      auto dilation = convOp.getDilation();
      if (ShapedType weightType =
              dyn_cast<ShapedType>(op->getOperand(1).getType())) {
        auto shape = weightType.getShape();
        if (isa<tosa::Conv2DOp>(op)) {
          assert(shape.size() == 4);
          assert(dilation.size() == 2);
          if (failed(levelCheckKernel(op, dilation[0] * shape[1],
                                      "dilation_y * KH <= MAX_KERNEL)")) ||
              failed(levelCheckKernel(op, dilation[1] * shape[2],
                                      "dilation_x * KW <= MAX_KERNEL)")))
            return failure();
        } else if (isa<tosa::Conv3DOp>(op)) {
          assert(shape.size() == 5);
          assert(dilation.size() == 3);
          if (failed(levelCheckKernel(op, dilation[0] * shape[1],
                                      "dilation_d * KD <= MAX_KERNEL)")) ||
              failed(levelCheckKernel(op, dilation[1] * shape[2],
                                      "dilation_y * KH <= MAX_KERNEL)")) ||
              failed(levelCheckKernel(op, dilation[2] * shape[3],
                                      "dilation_x * KW <= MAX_KERNEL)")))
            return failure();
        } else if (isa<tosa::DepthwiseConv2DOp>(op)) {
          assert(shape.size() == 4);
          assert(dilation.size() == 2);
          if (failed(levelCheckKernel(op, dilation[0] * shape[0],
                                      "dilation_y * KH <= MAX_KERNEL)")) ||
              failed(levelCheckKernel(op, dilation[1] * shape[1],
                                      "dilation_x * KW <= MAX_KERNEL)")))
            return failure();
        }
      }
    }
    return success();
  }

  // FFT op: level check H, W in input shape [N,H,W]
  template <typename T>
  LogicalResult levelCheckFFT(Operation *op) {
    if (isa<T>(op)) {
      for (auto v : op->getOperands()) {
        if (ShapedType type = dyn_cast<ShapedType>(v.getType())) {
          auto shape = type.getShape();
          assert(shape.size() == 3);
          if (failed(levelCheckKernel(op, shape[1], "H <= MAX_KERNEL")) ||
              failed(levelCheckKernel(op, shape[2], "W <= MAX_KERNEL"))) {
            return failure();
          }
        }
      }
    }
    return success();
  }

  // TransposeConv2d op: level check kH/kW, outpad, and stride
  LogicalResult levelCheckTransposeConv2d(Operation *op) {
    if (auto transpose = dyn_cast<tosa::TransposeConv2DOp>(op)) {
      if (ShapedType filterType =
              dyn_cast<ShapedType>(transpose.getWeight().getType())) {
        auto shape = filterType.getShape();
        assert(shape.size() == 4);
        // level check kernel sizes for kH and KW
        if (failed(levelCheckKernel(op, shape[1], "KH <= MAX_KERNEL")) ||
            failed(levelCheckKernel(op, shape[2], "KW <= MAX_KERNEL"))) {
          return failure();
        }
      }
      for (auto p : transpose.getOutPad()) {
        if (failed(levelCheckKernel(op, p, "pad <= MAX_KERNEL"))) {
          return failure();
        }
      }
      for (auto s : transpose.getStride()) {
        if (failed(levelCheckStride(op, s, "stride <= MAX_STRIDE"))) {
          return failure();
        }
      }
    }
    return success();
  }

  // Resize op: level check max scales
  LogicalResult levelCheckResize(Operation *op) {
    if (auto resize = dyn_cast<tosa::ResizeOp>(op)) {
      SmallVector<int64_t> scale;
      if (!tosa::getConstShapeValues(resize.getScale().getDefiningOp(),
                                     scale)) {
        return failure();
      }
      const int64_t scaleYN = scale[0];
      const int64_t scaleYD = scale[1];
      const int64_t scaleXN = scale[2];
      const int64_t scaleXD = scale[3];
      if (failed(levelCheckScale(op, scaleYN / scaleYD,
                                 "scale_y_n/scale_y_d <= MAX_SCALE")) ||
          failed(levelCheckScale(op, scaleXN / scaleXD,
                                 "scale_x_n/scale_x_d <= MAX_SCALE"))) {
        return failure();
      }
    }
    return success();
  }

  // Recursively perform a bottom-up search to determine the maximum nesting
  // depth, starting from a specific operation and continuing up to the function
  // or module scope. Tosa nesting_depth starts at 0 and increments by one each
  // time a new nested `region` is encountered.
  static void getMaxNestedDepth(Operation *op, int32_t &depth) {
    if (isa<mlir::func::FuncOp>(op) || isa<ModuleOp>(op))
      return;

    op = op->getParentOp();
    if (!op)
      return;

    depth++;
    getMaxNestedDepth(op, depth);
  }

  LogicalResult levelCheckMaxNesting(Operation *op) {
    int32_t maxNestedDepth = 0;
    getMaxNestedDepth(op, maxNestedDepth);

    if (maxNestedDepth >= targetEnv.getLevel().MAX_NESTING) {
      op->emitOpError() << "failed level check: " << maxNestedDepth
                        << " >= MAX_NESTING";
      return failure();
    }
    return success();
  }

  LogicalResult levelCheckListSize(Operation *op) {
    if (auto concat = dyn_cast<tosa::ConcatOp>(op)) {
      return levelCheckListSize(op, concat.getInput1().size(), "input1");
    }
    if (auto custom = dyn_cast<tosa::CustomOp>(op)) {
      if (failed(levelCheckListSize(op, custom.getInputList().size(),
                                    "input_list")) ||
          failed(levelCheckListSize(op, custom.getOutputList().size(),
                                    "output_list"))) {
        return failure();
      }
    }
    if (auto condIf = dyn_cast<tosa::IfOp>(op)) {
      if (failed(
              levelCheckListSize(op, condIf.getInputList().size(), "inputs")) ||
          failed(levelCheckListSize(op, condIf.getOutputList().size(),
                                    "outputs"))) {
        return failure();
      }
    }
    if (auto w = dyn_cast<tosa::WhileOp>(op)) {
      if (failed(levelCheckListSize(op, w.getInputList().size(), "inputs")) ||
          failed(levelCheckListSize(op, w.getOutputList().size(), "outputs"))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult attributeCheckRescale(Operation *op) {
    if (auto rescale = dyn_cast<tosa::RescaleOp>(op)) {
      if (rescale.getRoundingMode() == RoundingMode::DOUBLE_ROUND &&
          !targetEnv.allows(Extension::doubleround)) {
        op->emitOpError()
            << "failed attribute check: rounding_mode = DOUBLE_ROUND "
            << "requires extension [doubleround]";
        return failure();
      }
      if (rescale.getRoundingMode() == RoundingMode::INEXACT_ROUND &&
          !targetEnv.allows(Extension::inexactround)) {
        op->emitOpError()
            << "failed attribute check: rounding_mode = INEXACT_ROUND "
            << "requires extension [inexactround]";
        return failure();
      }
    }
    return success();
  }

  LogicalResult CheckVariable(Operation *op);
  LogicalResult CheckVariableReadOrWrite(Operation *op);
  bool isValidElementType(Type type, const bool allowUnsigned = false);

  SmallVector<
      std::function<LogicalResult(Operation *, const tosa::TargetEnv &)>>
      constCheckers;
  DenseMap<StringAttr, mlir::Type> variablesMap;
  TosaProfileCompliance profileComp;
  tosa::TargetEnv targetEnv;
};

template <>
LogicalResult TosaValidation::levelCheckRanks(tosa::ArgMaxOp tosaOp) {
  auto *op = tosaOp.getOperation();
  if (failed(levelCheckRank(op, tosaOp.getInput(), "operand",
                            targetEnv.getLevel().MAX_RANK)))
    return failure();

  // rank(output) = rank(input) - 1
  if (failed(levelCheckRank(op, tosaOp.getOutput(), "result",
                            targetEnv.getLevel().MAX_RANK - 1)))
    return failure();

  return success();
}

template <>
LogicalResult TosaValidation::levelCheckRanks(tosa::IfOp tosaOp) {
  auto *op = tosaOp.getOperation();

  // Only the condition input has rank limitation.
  if (failed(levelCheckRank(op, tosaOp.getCondition(), "operand",
                            targetEnv.getLevel().MAX_RANK)))
    return failure();

  return success();
}

template <>
LogicalResult TosaValidation::levelCheckRanks(tosa::VariableOp tosaOp) {
  auto *op = tosaOp.getOperation();
  auto variableType = getVariableType(tosaOp);
  if (failed(levelCheckRank(op, variableType, "variable type",
                            targetEnv.getLevel().MAX_RANK)))
    return failure();

  return success();
}

template <>
LogicalResult TosaValidation::levelCheckSizes(tosa::VariableOp tosaOp) {
  auto *op = tosaOp.getOperation();
  auto variableType = getVariableType(tosaOp);
  if (failed(levelCheckSize(op, variableType, "variable type")))
    return failure();

  return success();
}

LogicalResult TosaValidation::levelCheckRanksAndSizes(Operation *op) {
#define CHECK_RANKS_AND_SIZES(tosaOp)                                          \
  if (isa<tosa::tosaOp##Op>(op)) {                                             \
    if (failed(levelCheckRanks(cast<tosa::tosaOp##Op>(op))))                   \
      return failure();                                                        \
    if (failed(levelCheckSizes(cast<tosa::tosaOp##Op>(op))))                   \
      return failure();                                                        \
  }

#define CHECK_SIZES(tosaOp)                                                    \
  if (isa<tosa::tosaOp##Op>(op)) {                                             \
    if (failed(levelCheckSizes(cast<tosa::tosaOp##Op>(op))))                   \
      return failure();                                                        \
  }

  // Tensor Operators
  CHECK_RANKS_AND_SIZES(ArgMax);
  // Activation Functions
  CHECK_RANKS_AND_SIZES(Clamp);
  CHECK_RANKS_AND_SIZES(Erf);
  CHECK_RANKS_AND_SIZES(Sigmoid);
  CHECK_RANKS_AND_SIZES(Tanh);
  // Elementwise Binary Operators
  CHECK_RANKS_AND_SIZES(Add);
  CHECK_RANKS_AND_SIZES(ArithmeticRightShift);
  CHECK_RANKS_AND_SIZES(BitwiseAnd);
  CHECK_RANKS_AND_SIZES(BitwiseOr);
  CHECK_RANKS_AND_SIZES(BitwiseXor);
  CHECK_RANKS_AND_SIZES(IntDiv);
  CHECK_RANKS_AND_SIZES(LogicalAnd);
  CHECK_RANKS_AND_SIZES(LogicalLeftShift);
  CHECK_RANKS_AND_SIZES(LogicalRightShift);
  CHECK_RANKS_AND_SIZES(LogicalOr);
  CHECK_RANKS_AND_SIZES(LogicalXor);
  CHECK_RANKS_AND_SIZES(Maximum);
  CHECK_RANKS_AND_SIZES(Minimum);
  CHECK_RANKS_AND_SIZES(Mul);
  CHECK_RANKS_AND_SIZES(Pow);
  CHECK_RANKS_AND_SIZES(Sub);
  CHECK_RANKS_AND_SIZES(Table);
  // Elementwise Unary Operators
  CHECK_RANKS_AND_SIZES(Abs);
  CHECK_RANKS_AND_SIZES(BitwiseNot);
  CHECK_RANKS_AND_SIZES(Ceil);
  CHECK_RANKS_AND_SIZES(Clz);
  CHECK_RANKS_AND_SIZES(Cos);
  CHECK_RANKS_AND_SIZES(Exp);
  CHECK_RANKS_AND_SIZES(Floor);
  CHECK_RANKS_AND_SIZES(Log);
  CHECK_RANKS_AND_SIZES(LogicalNot);
  CHECK_RANKS_AND_SIZES(Negate);
  CHECK_RANKS_AND_SIZES(Reciprocal);
  CHECK_RANKS_AND_SIZES(Rsqrt);
  CHECK_RANKS_AND_SIZES(Sin);
  // Elementwise Ternary Operators
  CHECK_RANKS_AND_SIZES(Select);
  // Comparison Operators
  CHECK_RANKS_AND_SIZES(Equal);
  CHECK_RANKS_AND_SIZES(Greater);
  CHECK_RANKS_AND_SIZES(GreaterEqual);
  // Reduction Operators
  CHECK_RANKS_AND_SIZES(ReduceAll);
  CHECK_RANKS_AND_SIZES(ReduceAny);
  CHECK_RANKS_AND_SIZES(ReduceMax);
  CHECK_RANKS_AND_SIZES(ReduceMin);
  CHECK_RANKS_AND_SIZES(ReduceProduct);
  CHECK_RANKS_AND_SIZES(ReduceSum);
  // Data Layout Operators
  CHECK_RANKS_AND_SIZES(Concat);
  CHECK_RANKS_AND_SIZES(Pad);
  CHECK_RANKS_AND_SIZES(Reshape);
  CHECK_RANKS_AND_SIZES(Reverse);
  CHECK_RANKS_AND_SIZES(Slice);
  CHECK_RANKS_AND_SIZES(Tile);
  CHECK_RANKS_AND_SIZES(Transpose);
  // Type Conversion
  CHECK_RANKS_AND_SIZES(Cast);
  CHECK_RANKS_AND_SIZES(Rescale);
  // Control Flow Operators
  CHECK_RANKS_AND_SIZES(If);
  // Variable Operators
  CHECK_RANKS_AND_SIZES(Variable);
  CHECK_RANKS_AND_SIZES(VariableWrite);
  CHECK_RANKS_AND_SIZES(VariableRead);
  // Data Nodes
  CHECK_RANKS_AND_SIZES(Const);
  CHECK_RANKS_AND_SIZES(Identity);

  // For the following operators, check whether the size of each tensor
  // operand is valid in a given Level.

  // Tensor Operators
  CHECK_SIZES(AvgPool2d);
  CHECK_SIZES(Conv2D);
  CHECK_SIZES(Conv3D);
  CHECK_SIZES(DepthwiseConv2D);
  CHECK_SIZES(TransposeConv2D);
  CHECK_SIZES(FFT2d);
  CHECK_SIZES(MatMul);
  CHECK_SIZES(MaxPool2d);
  CHECK_SIZES(RFFT2d);
  // Scatter/Gather Operators
  CHECK_SIZES(Gather);
  CHECK_SIZES(Scatter);
  // Image Operators
  CHECK_SIZES(Resize);
  // Custom Operators
  CHECK_SIZES(Custom);
  // Control Flow Operators
  CHECK_SIZES(While);
  // Shape Operators
  CHECK_SIZES(ConstShape);

#undef CHECK_RANKS_AND_SIZES
#undef CHECK_SIZES
  return success();
}

// Perform the Level tensor size check on the tensor type.
LogicalResult TosaValidation::levelCheckSize(Operation *op,
                                             const Type &typeToCheck,
                                             const StringRef operandOrResult) {
  if (ShapedType type = dyn_cast<ShapedType>(typeToCheck)) {
    if (!type.hasRank())
      return op->emitOpError() << "failed level check: unranked tensor";
    auto shape = type.getShape();
    for (auto dim : shape) {
      if (mlir::ShapedType::isDynamic(dim))
        return op->emitOpError() << "failed level check: " << operandOrResult
                                 << " shape dimension cannot be dynamic";
    }

    int64_t element_bits = type.getElementTypeBitWidth();
    int64_t element_bytes = std::max(INT64_C(1), element_bits / 8);
    int64_t size = element_bytes * type.getNumElements();

    // According to 1.11. Tensor Definitions of Tosa spec, the value of
    // tensor_size_t is 1 << MAX_LOG2_SIZE) - 1 where MAX_LOG2_SIZE is
    // defined in 1.7. Levels.
    // For each tensor, the number of tensor elements multiplied by the
    // element size in bytes must be representable as a tensor_size_t.
    const int64_t max_size =
        (INT64_C(1) << targetEnv.getLevel().MAX_LOG2_SIZE) - 1;
    if (size > max_size)
      return op->emitOpError()
             << "failed level check: " << operandOrResult
             << " tensor size (in bytes) <= (1 << MAX_LOG2_SIZE - 1)";
  }
  return success();
}

LogicalResult TosaValidation::applyLevelCheck(Operation *op) {
  if (targetEnv.getLevel() == TOSA_LEVEL_NONE) {
    // no need to do level checks
    return success();
  }

  // check rank and sizes early so later checks can assume shaped operands
  if (failed(levelCheckRanksAndSizes(op)))
    return failure();

  // additional level checks from spec 0.70
  if (failed(levelCheckPool<tosa::AvgPool2dOp>(op)) ||
      failed(levelCheckConv<tosa::Conv2DOp>(op)) ||
      failed(levelCheckConv<tosa::Conv3DOp>(op)) ||
      failed(levelCheckConv<tosa::DepthwiseConv2DOp>(op)) ||
      failed(levelCheckFFT<tosa::FFT2dOp>(op)) ||
      failed(levelCheckPool<tosa::MaxPool2dOp>(op)) ||
      failed(levelCheckFFT<tosa::RFFT2dOp>(op)) ||
      failed(levelCheckTransposeConv2d(op)) || failed(levelCheckResize(op))) {
    return failure();
  }

  // level check MAX_TENSOR_LIST_SIZE
  if (failed(levelCheckListSize(op))) {
    return failure();
  }

  if (isa<tosa::IfOp>(op) || isa<tosa::WhileOp>(op)) {
    if (failed(levelCheckMaxNesting(op))) {
      return failure();
    }
  }

  return success();
}

LogicalResult TosaValidation::applyAttributeCheck(Operation *op) {
  if (failed(attributeCheckRescale(op)))
    return failure();
  return success();
}

inline bool CompatibleTypes(const mlir::Type &type,
                            const mlir::Type &declaredType) {
  // for now, simply use type equality comparison
  return type == declaredType;
}

LogicalResult TosaValidation::CheckVariable(Operation *op) {
  if (auto variableOp = dyn_cast<mlir::tosa::VariableOp>(op)) {
    mlir::StringAttr nameAttr = variableOp.getNameAttr();

    if (variablesMap.count(nameAttr))
      return op->emitOpError() << "name has already been declared";

    auto elementType = variableOp.getType();
    DenseIntElementsAttr varShapeAttr = variableOp.getVarShape();
    SmallVector<int64_t> shape = to_vector(varShapeAttr.getValues<int64_t>());
    RankedTensorType variableType =
        RankedTensorType::get(ArrayRef<int64_t>(shape), elementType);

    variablesMap[nameAttr] = variableType;
  }

  return success();
}

LogicalResult TosaValidation::CheckVariableReadOrWrite(Operation *op) {
  if (isa<mlir::tosa::VariableReadOp>(op) ||
      isa<mlir::tosa::VariableWriteOp>(op)) {
    mlir::StringAttr nameAttr = cast<mlir::StringAttr>(op->getAttr("name"));
    if (!variablesMap.count(nameAttr))
      return op->emitOpError() << "name has not been declared";

    auto varType = variablesMap[nameAttr];

    for (auto v : op->getOperands()) {
      auto type = v.getType();
      if (!CompatibleTypes(type, varType))
        return op->emitOpError() << "operand type does not equal variable type";
    }

    for (auto v : op->getResults()) {
      auto type = v.getType();
      if (!CompatibleTypes(type, varType))
        return op->emitOpError() << "result type does not equal variable type";
    }
  }

  return success();
}

LogicalResult TosaValidation::applyVariableCheck(Operation *op) {
  if (failed(CheckVariable(op)) || failed(CheckVariableReadOrWrite(op)))
    return failure();
  return success();
}

LogicalResult checkErrorIfResize(Operation *op) {
  auto resize = dyn_cast<tosa::ResizeOp>(op);
  if (!resize)
    return success();

  const Value input = resize.getInput();
  const Value output = resize.getOutput();
  const RankedTensorType inputType =
      llvm::dyn_cast<RankedTensorType>(input.getType());
  const RankedTensorType outputType =
      llvm::dyn_cast<RankedTensorType>(output.getType());

  if (!inputType || !outputType)
    return op->emitOpError("expect ranked input/output tensor");

  // Ensure the image size is supported by GPU APIs and that for integer
  // implementations, position * stride does not overflow int32_t.
  if (inputType.hasStaticShape() && outputType.hasStaticShape()) {
    const SmallVector<int64_t, 4> sizes = {
        outputType.getDimSize(1), outputType.getDimSize(2),
        inputType.getDimSize(1), inputType.getDimSize(2)};
    const int64_t *maxDim = llvm::max_element(sizes);
    if (maxDim != sizes.end() && *maxDim >= 16384)
      return op->emitOpError(
                 "expect input/output height/width dims to be < 16384, ")
             << "got [OH, OW, IH, IW] = " << sizes;
  }

  SmallVector<int64_t> scale;
  if (!tosa::getConstShapeValues(resize.getScale().getDefiningOp(), scale))
    return failure();

  const int64_t scaleYN = scale[0];
  const int64_t scaleYD = scale[1];
  const int64_t scaleXN = scale[2];
  const int64_t scaleXD = scale[3];

  // Ensure scale values don't overflow int32 accumulator
  if (scaleYN > (1 << 11) || scaleXN > (1 << 11))
    return op->emitOpError(
               "expect all scale numerator values to be <= (1 << 11), "
               "got scale_y_n=")
           << scaleYN << ", scale_x_n=" << scaleXN;

  if (scaleYD >= 16 * scaleYN || scaleXD >= 16 * scaleXN)
    return op->emitOpError("expect a downscale ratio larger than 1/16, got y=")
           << scaleYN << "/" << scaleYD << ", x=" << scaleXN << "/" << scaleXD;

  SmallVector<int64_t> offset;
  SmallVector<int64_t> border;
  if (!tosa::getConstShapeValues(resize.getOffset().getDefiningOp(), offset) ||
      !tosa::getConstShapeValues(resize.getBorder().getDefiningOp(), border))
    return failure();

  const int64_t offsetY = offset[0];
  const int64_t offsetX = offset[1];
  // Set a consistent lower limit of 1/16 downscale to simplify
  // implementations
  if (offsetY < -scaleYN || offsetY >= 16 * scaleYN)
    return op->emitOpError(
               "expect offsetY / scaleYNumerator to be in range [-1, 16), got ")
           << offsetY << "/" << scaleYN;
  if (offsetX < -scaleXN || offsetX >= 16 * scaleXN)
    return op->emitOpError(
               "expect offsetX / scaleXNumerator to be in range [-1, 16), got ")
           << offsetX << "/" << scaleXN;

  const int64_t borderY = border[0];
  const int64_t borderX = border[1];
  if (borderY < -16 * scaleYN || borderY >= scaleYN)
    return op->emitOpError(
               "expect borderY / scaleYNumerator to be in range [-16, 1), got ")
           << borderY << "/" << scaleYN;
  if (borderX < -16 * scaleXN || borderX >= scaleXN)
    return op->emitOpError(
               "expect borderX / scaleXNumerator to be in range [-16, 1), got ")
           << borderX << "/" << scaleXN;

  // The following section of code is mostly duplicated with ResizeOp::verify().
  //
  // In TOSA specification, we do not support broadcast behavior.
  // However, there is a rewrite pattern to materialize broadcast ResizeOp.
  // It makes invalid TOSA ResizeOp into valid one. To avoid breaking
  // existing code, we keep the rewrite pattern untouched. So, we need
  // loose the checking in ResizeOp::verify() to support broadcast ResizeOp.
  //
  // Here is a strict checking to conform TOSA specification.
  // FIXME: Remove the duplicated checkings when broadcast ResizeOp is removed.
  auto idivCheck = [](const int64_t lhs,
                      const int64_t rhs) -> std::optional<int64_t> {
    if (lhs % rhs != 0)
      return std::nullopt;
    return lhs / rhs;
  };

  const int64_t oh = outputType.getDimSize(1);
  const int64_t ow = outputType.getDimSize(2);
  const int64_t ih = inputType.getDimSize(1);
  const int64_t iw = inputType.getDimSize(2);

  if (ih != ShapedType::kDynamic) {
    const std::optional<int64_t> calculatedOutHeightMinusOne =
        idivCheck((ih - 1) * scaleYN - offsetY + borderY, scaleYD);
    if (!calculatedOutHeightMinusOne.has_value())
      return op->emitOpError(
                 "expected (input_height - 1) * scale_y_n - offset_y + "
                 "border_y ")
             << "to be wholly divisible by scale_y_d, got ((" << ih
             << " - 1) * " << scaleYN << " - " << offsetY << " + " << borderY
             << ") / " << scaleYD;
    const int64_t calculatedOutHeight = calculatedOutHeightMinusOne.value() + 1;
    if (oh != ShapedType::kDynamic && calculatedOutHeight != oh)
      return op->emitOpError(
                 "calculated output height did not match expected: ")
             << "calculated=" << calculatedOutHeight << ", expected=" << oh;
  }

  if (iw != ShapedType::kDynamic) {
    const std::optional<int64_t> calculatedOutWidthMinusOne =
        idivCheck((iw - 1) * scaleXN - offsetX + borderX, scaleXD);
    if (!calculatedOutWidthMinusOne.has_value())
      return op->emitOpError(
                 "expected (input_width - 1) * scale_x_n - offset_x + "
                 "border_x ")
             << "to be wholly divisible by scale_x_d, got ((" << iw
             << " - 1) * " << scaleXN << " - " << offsetX << " + " << borderX
             << ") / " << scaleXD;
    const int64_t calculatedOutWidth = calculatedOutWidthMinusOne.value() + 1;
    if (ow != ShapedType::kDynamic && calculatedOutWidth != ow)
      return op->emitOpError("calculated output width did not match expected: ")
             << "calculated=" << calculatedOutWidth << ", expected=" << ow;
  }

  return success();
}

LogicalResult checkErrorIfMul(Operation *op) {
  auto mul = dyn_cast<tosa::MulOp>(op);
  if (!mul)
    return success();

  // REQUIRE(0 <= shift && shift <= 63);
  // REQUIRE(is_same<in_t,int32_t>() || shift == 0);
  ElementsAttr shift_elem;
  if (!matchPattern(mul.getShift(), m_Constant(&shift_elem)))
    return success();
  int32_t shift = shift_elem.getValues<IntegerAttr>()[0].getInt();
  auto inputElemType = getElementTypeOrSelf(mul.getInput1());
  if (inputElemType.isInteger(32)) {
    // 0 <= shift <= 63 for int32_t type
    if (shift < 0 || shift > 63)
      return op->emitOpError()
             << "requires 0 <= shift && shift <= 63, but got: " << shift;
  } else {
    // shift must be 0 for all other types
    if (shift != 0)
      return op->emitOpError()
             << "requires shift = 0 for all input data types that "
                "are not int32_t, but got: "
             << shift;
  }

  return success();
}

LogicalResult checkErrorIfTable(Operation *op) {
  auto table = dyn_cast<tosa::TableOp>(op);
  if (!table)
    return success();

  // REQUIRE(length(table) == TABLE_SIZE) where TABLE_SIZE is 256 or 513
  const auto inputElemType = getElementTypeOrSelf(table.getInput1().getType());
  const int tableSize = inputElemType.isInteger(8) ? 256 : 513;

  const ShapeAdaptor tableShape(table.getTable().getType());
  if (tableShape.hasStaticShape()) {
    const auto numElements = tableShape.getNumElements();
    if (numElements != tableSize)
      return op->emitOpError() << "requires table size of " << tableSize
                               << ", got " << numElements;
  }

  return success();
}

LogicalResult checkErrorIfRescale(Operation *op) {
  auto rescale = dyn_cast<tosa::RescaleOp>(op);
  if (!rescale)
    return success();

  auto inputType = llvm::dyn_cast<ShapedType>(rescale.getInput().getType());
  auto outputType = llvm::dyn_cast<ShapedType>(rescale.getOutput().getType());
  if (!inputType || !outputType || !inputType.getElementType().isInteger() ||
      !outputType.getElementType().isInteger())
    return success();

  auto inElemType = inputType.getElementType();
  auto outElemType = outputType.getElementType();
  auto inWidth = inElemType.getIntOrFloatBitWidth();
  auto outWidth = outElemType.getIntOrFloatBitWidth();

  bool inputUnsigned = rescale.getInputUnsigned();
  bool outputUnsigned = rescale.getOutputUnsigned();

  bool scale32 = rescale.getScale32();
  auto roundingMode = rescale.getRoundingMode();

  // ERROR_IF(scale32 && is_same<in_t,i48_t>())
  if (scale32 && inWidth == 48)
    return op->emitOpError() << "scale32 is not allowed with 48-bit input.";

  // ERROR_IF(!scale32 && (rounding_mode == DOUBLE_ROUND))
  if (!scale32 && roundingMode == RoundingMode::DOUBLE_ROUND)
    return op->emitOpError()
           << "DOUBLE_ROUND is only allowed with scale32=true.";

  // ERROR_IF(input_unsigned && output_unsigned)
  if (inputUnsigned && outputUnsigned)
    return op->emitOpError() << "input and output cannot be both unsigned.";

  // ERROR_IF(is_same<out_t,i32_t>() && input_unsigned)
  if (outWidth == 32 && inputUnsigned)
    return op->emitOpError()
           << "i32 output type is not allowed with unsigned input.";

  // ERROR_IF(is_same<in_t,i32_t>() && output_unsigned)
  if (inWidth == 32 && outputUnsigned)
    return op->emitOpError()
           << "i32 input type is not allowed with unsigned output.";

  // ERROR_IF(is_same<in_t,i48_t>() && output_unsigned)
  if (inWidth == 48 && outputUnsigned)
    return op->emitOpError()
           << "i48 input type is not allowed with unsigned output.";

  // ERROR_IF(is_same<in_t, i48_t> && input_unsigned)
  if (inWidth == 48 && inputUnsigned)
    return op->emitOpError() << "i48 input type cannot be unsigned.";

  // ERROR_IF(is_same<in_t, i32_t> && input_unsigned)
  if (inWidth == 32 && inputUnsigned)
    return op->emitOpError() << "i32 input type cannot be unsigned.";

  // ERROR_IF(is_same<out_t, i32_t> && output_unsigned)
  if (outWidth == 32 && outputUnsigned)
    return op->emitOpError() << "i32 output type cannot be unsigned.";

  return success();
}

LogicalResult checkErrorIfPad(Operation *op) {
  auto pad = dyn_cast<tosa::PadOp>(op);
  if (!pad)
    return success();

  DenseIntElementsAttr paddingAttr;
  if (!matchPattern(pad.getPadding(), m_Constant(&paddingAttr)))
    // Pad verifier will catch this
    return success();

  for (const APInt &val : paddingAttr.getValues<APInt>()) {
    if (val.getSExtValue() < 0)
      return op->emitOpError() << "padding value must all be non-negative, got "
                               << val.getSExtValue();
  }

  return success();
}

static bool isOpIsolatedWithinRegion(Operation *op, Region *region) {
  return llvm::all_of(op->getOperands(), [&](auto operand) {
    Region *operandRegion = operand.getParentRegion();
    return operandRegion && region->isAncestor(operandRegion);
  });
}

static LogicalResult isRegionIsolatedFromAbove(Region &regionToCheck) {
  bool noLiveInValue = true;
  regionToCheck.walk([&noLiveInValue, &regionToCheck](Operation *op) {
    if (!isOpIsolatedWithinRegion(op, &regionToCheck)) {
      noLiveInValue = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return noLiveInValue ? success() : failure();
}

LogicalResult checkIsolatedRegion(Operation *op, Region &regionToCheck,
                                  StringRef regionName) {
  if (succeeded(isRegionIsolatedFromAbove(regionToCheck)))
    return success();
  return op->emitOpError()
         << "is not conformant to the TOSA specification. It requires the '"
         << regionName << "' region is isolated from above.\n";
}

LogicalResult checkErrorIfCondIf(Operation *op) {
  auto ifOp = dyn_cast<tosa::IfOp>(op);
  if (!ifOp)
    return success();

  // Currently the dialect supports declaring cond_if operations that
  // have then/else regions that reference values from outside these
  // regions. According to the specification, all values used by the
  // then/else regions must be explicitly declared within the regions.
  // Therefore we must check that the then/else regions are
  // "isolated from above", in order to be conformant to the
  // specification.
  //
  // Note: the dialect currently supports two styles of syntax for
  // declaring "cond_if" operations. We'll refer to these as follows:
  //
  // Generic:
  // %0 = "tosa.cond_if"(%arg0, %arg1, %arg2) ({
  //   ^bb0(%arg3, %arg4):
  //     tosa.yield %arg3
  // },  {
  //   ^bb0(%arg3, %arg4):
  //     tosa.yield %arg4
  // })
  //
  // Simplified:
  // %0 = tosa.cond_if %arg2 (%arg3 = %arg0, %arg4 = %arg1) {
  //   ^bb0(%arg3, %arg4):
  //   tosa.yield %arg3
  // } else {
  //   ^bb0(%arg3, %arg4):
  //   tosa.yield %arg4
  // }

  if (failed(checkIsolatedRegion(op, ifOp.getThenGraph(), "then")) ||
      failed(checkIsolatedRegion(op, ifOp.getElseGraph(), "else")))
    return failure();
  return success();
}

LogicalResult checkErrorIfWhileLoop(Operation *op) {
  auto whileOp = dyn_cast<tosa::WhileOp>(op);
  if (!whileOp)
    return success();

  if (failed(checkIsolatedRegion(op, whileOp.getCondGraph(), "cond")) ||
      failed(checkIsolatedRegion(op, whileOp.getBodyGraph(), "body")))
    return failure();
  return success();
}

LogicalResult checkErrorIfScatter(Operation *op) {
  auto scatterOp = dyn_cast<tosa::ScatterOp>(op);
  if (!scatterOp)
    return success();

  // for constant indices, check that there are no duplicate values
  DenseIntElementsAttr indicesAttr;
  if (!matchPattern(scatterOp.getIndices(), m_Constant(&indicesAttr)))
    return success();

  auto const indicesType =
      dyn_cast<ShapedType>(scatterOp.getIndices().getType());
  if (!indicesType || !indicesType.hasRank()) {
    op->emitOpError("expect ranked indices tensor");
    return failure();
  }

  if (!hasUniqueConstantScatterIndices(indicesType, indicesAttr)) {
    op->emitOpError("indices values contain duplicates");
    return failure();
  }

  return success();
}

LogicalResult TosaValidation::applyErrorIfCheck(Operation *op) {
  if (failed(checkErrorIfResize(op)) || failed(checkErrorIfMul(op)) ||
      failed(checkErrorIfTable(op)) || failed(checkErrorIfRescale(op)) ||
      failed(checkErrorIfPad(op)) || failed(checkErrorIfCondIf(op)) ||
      failed(checkErrorIfWhileLoop(op)) || failed(checkErrorIfScatter(op)))
    return failure();
  return success();
}

bool TosaValidation::isValidElementType(Type type, const bool allowUnsigned) {
  if (isa<FloatType>(type)) {
    return isa<Float32Type, Float16Type, BFloat16Type, Float8E4M3FNType,
               Float8E5M2Type>(type);
  }
  if (auto intTy = dyn_cast<IntegerType>(type)) {
    if (intTy.isSignless()) {
      switch (intTy.getWidth()) {
      case 1:
      case 4:
      case 8:
      case 16:
      case 32:
      case 48:
        return true;
      }
    } else if (allowUnsigned && intTy.isUnsigned()) {
      switch (intTy.getWidth()) {
      case 8:
      case 16:
      case 32:
        return true;
      }
    }
  } else if (mlir::isa<tosa::shapeType>(type)) {
    return true;
  }
  return false;
}

void TosaValidation::runOnOperation() {
  TosaDialect *tosaDialect = getContext().getLoadedDialect<TosaDialect>();
  if (!tosaDialect)
    return;

  targetEnv = tosa::TargetEnv(lookupTargetEnvOrDefault(getOperation()));

  getOperation().walk([&](Operation *op) {
    if (op->getDialect() != tosaDialect)
      return;

    // validate operator element types:
    // - rescale operator is allowed to have ui8/ui16/ui32
    //   operands/results when strictOpSpecAlignment is false
    // - perform valid element type check at the beginning to
    //   protect rest of code against quantized element types
    const bool allowUnsigned =
        !strictOpSpecAlignment && isa<tosa::RescaleOp>(op);
    for (Value operand : op->getOperands()) {
      auto elementTy = getElementTypeOrSelf(operand);
      if (!isValidElementType(elementTy, allowUnsigned)) {
        op->emitOpError() << "is not profile-aligned: element type "
                          << elementTy << " is not legal";
        return signalPassFailure();
      }
    }
    for (Type resultTy : op->getResultTypes()) {
      auto elementTy = getElementTypeOrSelf(resultTy);
      if (!isValidElementType(elementTy, allowUnsigned)) {
        op->emitOpError() << "is not profile-aligned: element type "
                          << elementTy << " is not legal";
        return signalPassFailure();
      }
    }

    if (strictOpSpecAlignment &&
        failed(profileComp.checkProfile(op, targetEnv)))
      return signalPassFailure();

    if (strictOpSpecAlignment &&
        failed(profileComp.checkExtension(op, targetEnv)))
      return signalPassFailure();

    if (!allowInvalidOpDatatypeCombinations &&
        failed(profileComp.checkInvalid(op)))
      return signalPassFailure();

    // Some uses of TOSA rely on the constant operands of particular
    // operations.
    if (failed(applyConstantOperandCheck(op)))
      signalPassFailure();

    // do level checks
    if (failed(applyLevelCheck(op)))
      signalPassFailure();

    // check additional attribute restrictions
    if (failed(applyAttributeCheck(op)))
      signalPassFailure();

    // do variable type checks
    if (failed(applyVariableCheck(op)))
      signalPassFailure();

    // do error if checks
    if (strictOpSpecAlignment && failed(applyErrorIfCheck(op)))
      signalPassFailure();
  });
}
} // namespace
