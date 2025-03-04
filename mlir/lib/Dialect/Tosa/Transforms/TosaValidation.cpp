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
#include "mlir/Dialect/Tosa/Transforms/PassesEnums.cpp.inc"

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

static LogicalResult checkConstantOperandPad(Operation *op) {
  if (auto padOp = dyn_cast<tosa::PadOp>(op)) {
    DenseElementsAttr paddings;
    if (!matchPattern(padOp.getPadding(), m_Constant(&paddings)))
      return op->emitOpError("padding of pad is not constant");

    DenseElementsAttr padConst;
    // Assume this op is zero-padding if padConst is not presented.
    if (padOp.getPadConst() &&
        !matchPattern(padOp.getPadConst(), m_Constant(&padConst)))
      return op->emitOpError("pad_const of pad is not constant");
  }
  return success();
}

struct TosaLevel {
  int32_t MAX_RANK = 0;
  int32_t MAX_KERNEL = 0;
  int32_t MAX_STRIDE = 0;
  int32_t MAX_SCALE = 0;
  int32_t MAX_LOG2_SIZE = 0;
  int32_t MAX_NESTING = 0;
  int32_t MAX_TENSOR_LIST_SIZE = 0;

  bool operator==(const TosaLevel &rhs) {
    return MAX_RANK == rhs.MAX_RANK && MAX_KERNEL == rhs.MAX_KERNEL &&
           MAX_STRIDE == rhs.MAX_STRIDE && MAX_SCALE == rhs.MAX_SCALE &&
           MAX_LOG2_SIZE == rhs.MAX_LOG2_SIZE &&
           MAX_NESTING == rhs.MAX_NESTING &&
           MAX_TENSOR_LIST_SIZE == rhs.MAX_TENSOR_LIST_SIZE;
  }
};

static constexpr TosaLevel TOSA_LEVEL_EIGHTK = {6, 8192, 8192, 256, 31, 6, 64};
static constexpr TosaLevel TOSA_LEVEL_NONE = {32, 2147483647, 2147483647, 2048,
                                              63, 256,        256};

//===----------------------------------------------------------------------===//
// TOSA Validation Pass.
//===----------------------------------------------------------------------===//

struct TosaValidation : public tosa::impl::TosaValidationBase<TosaValidation> {
public:
  explicit TosaValidation() { populateConstantOperandChecks(); }

  explicit TosaValidation(const TosaValidationOptions &options)
      : TosaValidation() {
    this->profile = options.profile;
    this->extension = options.extension;
    this->strictOpSpecAlignment = options.strictOpSpecAlignment;
    this->level = options.level;
  }
  void runOnOperation() final;

  LogicalResult applyConstantOperandCheck(Operation *op) {
    for (auto &checker : constCheckers) {
      if (failed(checker(op)))
        return failure();
    }
    return success();
  }

  LogicalResult applyLevelCheck(Operation *op);

  // check variable read/write data types against variable declarations
  LogicalResult applyVariableCheck(Operation *op);

  // check error if conditions
  LogicalResult applyErrorIfCheck(Operation *op);

private:
  void populateConstantOperandChecks() {
    constCheckers.emplace_back(checkConstantOperandPad);
  }

  bool levelCheckKernel(Operation *op, int32_t v, const StringRef checkDesc) {
    if (v > tosaLevel.MAX_KERNEL) {
      op->emitOpError() << "failed level check: " << checkDesc;
      return false;
    }
    return true;
  }

  bool levelCheckStride(Operation *op, int32_t v, const StringRef checkDesc) {
    if (v > tosaLevel.MAX_STRIDE) {
      op->emitOpError() << "failed level check: " << checkDesc;
      return false;
    }
    return true;
  }

  bool levelCheckScale(Operation *op, int32_t v, const StringRef checkDesc) {
    if (v > tosaLevel.MAX_SCALE) {
      op->emitOpError() << "failed level check: " << checkDesc;
      return false;
    }
    return true;
  }

  bool levelCheckListSize(Operation *op, int32_t v, const StringRef checkDesc) {
    if (v > tosaLevel.MAX_TENSOR_LIST_SIZE) {
      op->emitOpError() << "failed level check for MAX_TENSOR_LIST_SIZE: "
                        << checkDesc;
      return false;
    }
    return true;
  }

  template <typename T>
  bool levelCheckRank(Operation *op, const T &v,
                      const StringRef operandOrResult, int32_t highest_rank) {
    if (ShapedType type = dyn_cast<ShapedType>(v.getType())) {
      if (!type.hasRank()) {
        op->emitOpError() << "failed level check: unranked tensor";
        return false;
      }
      if (type.getRank() > highest_rank) {
        op->emitOpError() << "failed level check: " << operandOrResult
                          << " rank(shape) <= MAX_RANK";
        return false;
      }
    }
    return true;
  }

  // Perform the Level tensor size check on the input tensor.
  bool levelCheckSize(Operation *op, const Value &v,
                      const StringRef operandOrResult);

  // Level check sizes of all operands and results of the operation.
  template <typename T>
  bool levelCheckSizes(T tosaOp) {
    auto op = tosaOp.getOperation();
    for (auto v : op->getOperands()) {
      if (!levelCheckSize(op, v, "operand"))
        return false;
    }

    for (auto v : op->getResults()) {
      if (!levelCheckSize(op, v, "result"))
        return false;
    }
    return true;
  }

  // Level check ranks of all operands, attribute and results of the operation.
  template <typename T>
  bool levelCheckRanks(T tosaOp) {
    auto op = tosaOp.getOperation();
    for (auto v : op->getOperands()) {
      if (!levelCheckRank(op, v, "operand", tosaLevel.MAX_RANK))
        return false;
    }

    if (!op->getAttrs().empty()) {
      for (NamedAttribute attr : op->getAttrs()) {
        if (auto elemAttr = dyn_cast<ElementsAttr>(attr.getValue())) {
          if (!levelCheckRank(op, elemAttr, "attribute", tosaLevel.MAX_RANK))
            return false;
        }
      }
    }

    for (auto v : op->getResults()) {
      if (!levelCheckRank(op, v, "result", tosaLevel.MAX_RANK))
        return false;
    }
    return true;
  }

  // Level check ranks and sizes.
  bool levelCheckRanksAndSizes(Operation *op);

  // Pool Op: level check kernel/stride/pad values
  template <typename T>
  bool levelCheckPool(Operation *op) {
    if (auto poolOp = dyn_cast<T>(op)) {
      for (auto k : poolOp.getKernel()) {
        if (!levelCheckKernel(op, k, "kernel <= MAX_KERNEL")) {
          return false;
        }
      }
      for (auto s : poolOp.getStride()) {
        if (!levelCheckStride(op, s, "stride <= MAX_STRIDE")) {
          return false;
        }
      }
      for (auto p : poolOp.getPad()) {
        if (!levelCheckKernel(op, p, "pad <= MAX_KERNEL")) {
          return false;
        }
      }
    }
    return true;
  }

  // Conv Op: level check dilation/stride/pad values
  template <typename T>
  bool levelCheckConv(Operation *op) {
    if (auto convOp = dyn_cast<T>(op)) {

      for (auto k : convOp.getDilation()) {
        if (!levelCheckKernel(op, k, "dilation <= MAX_KERNEL")) {
          return false;
        }
      }
      for (auto p : convOp.getPad()) {
        if (!levelCheckKernel(op, p, "pad <= MAX_KERNEL")) {
          return false;
        }
      }
      for (auto s : convOp.getStride()) {
        if (!levelCheckStride(op, s, "stride <= MAX_STRIDE")) {
          return false;
        }
      }
      auto dilation = convOp.getDilation();
      if (ShapedType weightType =
              dyn_cast<ShapedType>(op->getOperand(1).getType())) {
        auto shape = weightType.getShape();
        if (isa<tosa::Conv2DOp>(op)) {
          assert(shape.size() == 4);
          assert(dilation.size() == 2);
          if (!levelCheckKernel(op, dilation[0] * shape[1],
                                "dilation_y * KH <= MAX_KERNEL)") ||
              !levelCheckKernel(op, dilation[1] * shape[2],
                                "dilation_x * KW <= MAX_KERNEL)"))
            return false;
        } else if (isa<tosa::Conv3DOp>(op)) {
          assert(shape.size() == 5);
          assert(dilation.size() == 3);
          if (!levelCheckKernel(op, dilation[0] * shape[1],
                                "dilation_d * KD <= MAX_KERNEL)") ||
              !levelCheckKernel(op, dilation[1] * shape[2],
                                "dilation_y * KH <= MAX_KERNEL)") ||
              !levelCheckKernel(op, dilation[2] * shape[3],
                                "dilation_x * KW <= MAX_KERNEL)"))
            return false;
        } else if (isa<tosa::DepthwiseConv2DOp>(op)) {
          assert(shape.size() == 4);
          assert(dilation.size() == 2);
          if (!levelCheckKernel(op, dilation[0] * shape[0],
                                "dilation_y * KH <= MAX_KERNEL)") ||
              !levelCheckKernel(op, dilation[1] * shape[1],
                                "dilation_x * KW <= MAX_KERNEL)"))
            return false;
        }
      }
    }
    return true;
  }

  // FFT op: level check H, W in input shape [N,H,W]
  template <typename T>
  bool levelCheckFFT(Operation *op) {
    if (isa<T>(op)) {
      for (auto v : op->getOperands()) {
        if (ShapedType type = dyn_cast<ShapedType>(v.getType())) {
          auto shape = type.getShape();
          assert(shape.size() == 3);
          if (!levelCheckKernel(op, shape[1], "H <= MAX_KERNEL") ||
              !levelCheckKernel(op, shape[2], "W <= MAX_KERNEL")) {
            return false;
          }
        }
      }
    }
    return true;
  }

  // TransposeConv2d op: level check kH/kW, outpad, and stride
  bool levelCheckTransposeConv2d(Operation *op) {
    if (auto transpose = dyn_cast<tosa::TransposeConv2DOp>(op)) {
      if (ShapedType filterType =
              dyn_cast<ShapedType>(transpose.getWeight().getType())) {
        auto shape = filterType.getShape();
        assert(shape.size() == 4);
        // level check kernel sizes for kH and KW
        if (!levelCheckKernel(op, shape[1], "KH <= MAX_KERNEL") ||
            !levelCheckKernel(op, shape[2], "KW <= MAX_KERNEL")) {
          return false;
        }
      }
      for (auto p : transpose.getOutPad()) {
        if (!levelCheckKernel(op, p, "pad <= MAX_KERNEL")) {
          return false;
        }
      }
      for (auto s : transpose.getStride()) {
        if (!levelCheckStride(op, s, "stride <= MAX_STRIDE")) {
          return false;
        }
      }
    }
    return true;
  }

  // Resize op: level check max scales
  bool levelCheckResize(Operation *op) {
    if (auto resize = dyn_cast<tosa::ResizeOp>(op)) {
      SmallVector<int64_t> scale;
      if (!tosa::getConstShapeValue(resize.getScale().getDefiningOp(), scale)) {
        return false;
      }
      const int64_t scaleYN = scale[0];
      const int64_t scaleYD = scale[1];
      const int64_t scaleXN = scale[2];
      const int64_t scaleXD = scale[3];
      if (!levelCheckScale(op, scaleYN / scaleYD,
                           "scale_y_n/scale_y_d <= MAX_SCALE") ||
          !levelCheckScale(op, scaleXN / scaleXD,
                           "scale_x_n/scale_x_d <= MAX_SCALE")) {
        return false;
      }
    }
    return true;
  }

  bool levelCheckListSize(Operation *op) {
    if (auto concat = dyn_cast<tosa::ConcatOp>(op)) {
      return levelCheckListSize(op, concat.getInput1().size(), "input1");
    }
    if (auto custom = dyn_cast<tosa::CustomOp>(op)) {
      if (!levelCheckListSize(op, custom.getInputList().size(), "input_list") ||
          !levelCheckListSize(op, custom.getOutputList().size(),
                              "output_list")) {
        return false;
      }
    }
    if (auto condIf = dyn_cast<tosa::IfOp>(op)) {
      if (!levelCheckListSize(op, condIf.getInputs().size(), "inputs") ||
          !levelCheckListSize(op, condIf.getOutput().size(), "outputs")) {
        return false;
      }
    }
    if (auto w = dyn_cast<tosa::WhileOp>(op)) {
      if (!levelCheckListSize(op, w.getInputs().size(), "inputs") ||
          !levelCheckListSize(op, w.getOutput().size(), "outputs")) {
        return false;
      }
    }
    return true;
  }

  // configure profile and level values from pass options profileName and
  // levelName
  void configLevelAndProfile() {
    tosaLevel = TOSA_LEVEL_NONE;
    if (level == TosaLevelEnum::EightK) {
      tosaLevel = TOSA_LEVEL_EIGHTK;
    }

    if (!profile.empty()) {
      for (std::string &prof : profile) {
        auto profSymbol = symbolizeProfile(prof);
        if (profSymbol) {
          targetEnv.addProfile(profSymbol.value());
        } else {
          llvm::errs() << "unknown TOSA profile name passed in: " << prof
                       << ", supported profiles are `pro_int` and `pro_fp`\n";
          return signalPassFailure();
        }
      }
    }

    if (!extension.empty()) {
      for (std::string &ext : extension) {
        auto extSymbol = symbolizeExtension(ext);
        if (extSymbol) {
          targetEnv.addExtension(extSymbol.value());
        } else {
          llvm::errs() << "unknown TOSA extension name passed in: " << ext
                       << ", supported extension are int16, int4, bf16, "
                       << "fp8e4m3, fp8e5m2, fft, variable and controlflow\n";
          return signalPassFailure();
        }
      }
    }
  }

  bool CheckVariable(Operation *op);
  bool CheckVariableReadOrWrite(Operation *op);
  bool isValidElementType(Type type);

  SmallVector<std::function<LogicalResult(Operation *)>> constCheckers;
  TosaLevel tosaLevel;
  DenseMap<StringAttr, mlir::Type> variablesMap;
  TosaProfileCompliance profileComp;
  tosa::TargetEnv targetEnv;
};

template <>
bool TosaValidation::levelCheckRanks(tosa::ArgMaxOp tosaOp) {
  auto op = tosaOp.getOperation();
  if (!levelCheckRank(op, tosaOp.getInput(), "operand", tosaLevel.MAX_RANK))
    return false;

  // rank(output) = rank(input) - 1
  if (!levelCheckRank(op, tosaOp.getOutput(), "result", tosaLevel.MAX_RANK - 1))
    return false;

  return true;
}

template <>
bool TosaValidation::levelCheckRanks(tosa::IfOp tosaOp) {
  auto op = tosaOp.getOperation();

  // Only the condition input has rank limitation.
  if (!levelCheckRank(op, tosaOp.getCond(), "operand", tosaLevel.MAX_RANK))
    return false;

  return true;
}

bool TosaValidation::levelCheckRanksAndSizes(Operation *op) {
#define CHECK_RANKS_AND_SIZES(tosaOp)                                          \
  if (isa<tosa::tosaOp##Op>(op)) {                                             \
    if (!levelCheckRanks(cast<tosa::tosaOp##Op>(op)))                          \
      return false;                                                            \
    if (!levelCheckSizes(cast<tosa::tosaOp##Op>(op)))                          \
      return false;                                                            \
  }

#define CHECK_SIZES(tosaOp)                                                    \
  if (isa<tosa::tosaOp##Op>(op)) {                                             \
    if (!levelCheckSizes(cast<tosa::tosaOp##Op>(op)))                          \
      return false;                                                            \
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
  return true;
}

// Perform the Level tensor size check
bool TosaValidation::levelCheckSize(Operation *op, const Value &v,
                                    const StringRef operandOrResult) {
  if (ShapedType type = dyn_cast<ShapedType>(v.getType())) {
    if (!type.hasRank()) {
      op->emitOpError() << "failed level check: unranked tensor";
      return false;
    }
    auto shape = type.getShape();
    for (auto dim : shape) {
      if (mlir::ShapedType::isDynamic(dim)) {
        op->emitOpError() << "failed level check: " << operandOrResult
                          << " shape dimension cannot be dynamic";
        return false;
      }
    }

    int64_t element_bits = type.getElementTypeBitWidth();
    int64_t element_bytes = std::max(INT64_C(1), element_bits / 8);
    int64_t size = element_bytes * type.getNumElements();

    // According to 1.11. Tensor Definitions of Tosa spec, the value of
    // tensor_size_t is 1 << MAX_LOG2_SIZE) - 1 where MAX_LOG2_SIZE is
    // defined in 1.7. Levels.
    // For each tensor, the number of tensor elements multiplied by the
    // element size in bytes must be representable as a tensor_size_t.
    const int64_t max_size = (INT64_C(1) << tosaLevel.MAX_LOG2_SIZE) - 1;
    if (size > max_size) {
      op->emitOpError()
          << "failed level check: " << operandOrResult
          << " tensor size (in bytes) <= (1 << MAX_LOG2_SIZE - 1)";
      return false;
    }
  }
  return true;
}

LogicalResult TosaValidation::applyLevelCheck(Operation *op) {
  if (tosaLevel == TOSA_LEVEL_NONE) {
    // no need to do level checks
    return success();
  }

  // additional level checks from spec 0.70
  if (!levelCheckPool<tosa::AvgPool2dOp>(op) ||
      !levelCheckConv<tosa::Conv2DOp>(op) ||
      !levelCheckConv<tosa::Conv3DOp>(op) ||
      !levelCheckConv<tosa::DepthwiseConv2DOp>(op) ||
      !levelCheckFFT<tosa::FFT2dOp>(op) ||
      !levelCheckPool<tosa::MaxPool2dOp>(op) ||
      !levelCheckFFT<tosa::RFFT2dOp>(op) || !levelCheckTransposeConv2d(op) ||
      !levelCheckResize(op)) {
    return failure();
  }

  if (!levelCheckRanksAndSizes(op)) {
    return failure();
  }

  // level check MAX_TENSOR_LIST_SIZE
  if (!levelCheckListSize(op)) {
    return failure();
  }

  return success();
}

inline bool CompatibleTypes(const mlir::Type &type,
                            const mlir::Type &declaredType) {
  // for now, simply use type equality comparison
  return type == declaredType;
}

bool TosaValidation::CheckVariable(Operation *op) {
  if (isa<mlir::tosa::VariableOp>(op)) {
    auto nameAttr = cast<mlir::StringAttr>(op->getAttr("name"));

    if (variablesMap.count(nameAttr)) {
      op->emitOpError() << "name has already been declared";
      return false;
    }

    auto typeAttr = cast<mlir::TypeAttr>(op->getAttr("type"));
    mlir::Type type = typeAttr.getValue();

    variablesMap[nameAttr] = type;
  }

  return true;
}

bool TosaValidation::CheckVariableReadOrWrite(Operation *op) {
  if (isa<mlir::tosa::VariableReadOp>(op) ||
      isa<mlir::tosa::VariableWriteOp>(op)) {
    auto nameAttr = cast<mlir::StringAttr>(op->getAttr("name"));

    if (!variablesMap.count(nameAttr)) {
      op->emitOpError() << "name has not been declared";
      return false;
    }

    auto varType = variablesMap[nameAttr];

    for (auto v : op->getOperands()) {
      auto type = v.getType();
      if (!CompatibleTypes(type, varType)) {
        op->emitOpError() << "operand type does not equal variable type";
        return false;
      }
    }

    for (auto v : op->getResults()) {
      auto type = v.getType();
      if (!CompatibleTypes(type, varType)) {
        op->emitOpError() << "result type does not equal variable type";
        return false;
      }
    }
  }

  return true;
}

LogicalResult TosaValidation::applyVariableCheck(Operation *op) {
  if (!CheckVariable(op) || !CheckVariableReadOrWrite(op)) {
    return failure();
  }
  return success();
}

bool checkErrorIfResize(Operation *op) {
  auto resize = dyn_cast<tosa::ResizeOp>(op);
  if (!resize)
    return true;

  const Value input = resize.getInput();
  const Value output = resize.getOutput();
  const RankedTensorType inputType =
      llvm::dyn_cast<RankedTensorType>(input.getType());
  const RankedTensorType outputType =
      llvm::dyn_cast<RankedTensorType>(output.getType());

  if (!inputType || !outputType) {
    op->emitOpError("expect ranked input/output tensor");
    return false;
  }

  // Ensure the image size is supported by GPU APIs and that for integer
  // implementations, position * stride does not overflow int32_t.
  if (inputType.hasStaticShape() && outputType.hasStaticShape()) {
    const SmallVector<int64_t, 4> sizes = {
        outputType.getDimSize(1), outputType.getDimSize(2),
        inputType.getDimSize(1), inputType.getDimSize(2)};
    const int64_t *maxDim = llvm::max_element(sizes);
    if (maxDim != sizes.end() && *maxDim >= 16384) {
      op->emitOpError("expect input/output height/width dims to be < 16384, ")
          << "got [OH, OW, IH, IW] = " << sizes;
      return false;
    }
  }

  SmallVector<int64_t> scale;
  if (!tosa::getConstShapeValue(resize.getScale().getDefiningOp(), scale)) {
    return false;
  }

  const int64_t scaleYN = scale[0];
  const int64_t scaleYD = scale[1];
  const int64_t scaleXN = scale[2];
  const int64_t scaleXD = scale[3];

  // Ensure scale values don't overflow int32 accumulator
  if (scaleYN > (1 << 11) || scaleXN > (1 << 11)) {
    op->emitOpError("expect all scale numerator values to be <= (1 << 11), "
                    "got scale_y_n=")
        << scaleYN << ", scale_x_n=" << scaleXN;
    return false;
  }

  if (scaleYD >= 16 * scaleYN || scaleXD >= 16 * scaleXN) {
    op->emitOpError("expect a downscale ratio larger than 1/16, got y=")
        << scaleYN << "/" << scaleYD << ", x=" << scaleXN << "/" << scaleXD;
    return false;
  }

  SmallVector<int64_t> offset;
  SmallVector<int64_t> border;
  if (!tosa::getConstShapeValue(resize.getOffset().getDefiningOp(), offset) ||
      !tosa::getConstShapeValue(resize.getBorder().getDefiningOp(), border)) {
    return false;
  }

  const int64_t offsetY = offset[0];
  const int64_t offsetX = offset[1];
  // Set a consistent lower limit of 1/16 downscale to simplify
  // implementations
  if (offsetY < -scaleYN || offsetY >= 16 * scaleYN) {
    op->emitOpError(
        "expect offsetY / scaleYNumerator to be in range [-1, 16), got ")
        << offsetY << "/" << scaleYN;
    return false;
  }
  if (offsetX < -scaleXN || offsetX >= 16 * scaleXN) {
    op->emitOpError(
        "expect offsetX / scaleXNumerator to be in range [-1, 16), got ")
        << offsetX << "/" << scaleXN;
    return false;
  }

  const int64_t borderY = border[0];
  const int64_t borderX = border[1];
  if (borderY < -16 * scaleYN || borderY >= scaleYN) {
    op->emitOpError(
        "expect borderY / scaleYNumerator to be in range [-16, 1), got ")
        << borderY << "/" << scaleYN;
    return false;
  }
  if (borderX < -16 * scaleXN || borderX >= scaleXN) {
    op->emitOpError(
        "expect borderX / scaleXNumerator to be in range [-16, 1), got ")
        << borderX << "/" << scaleXN;
    return false;
  }

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
    if (!calculatedOutHeightMinusOne.has_value()) {
      op->emitOpError("expected (input_height - 1) * scale_y_n - offset_y + "
                      "border_y ")
          << "to be wholly divisible by scale_y_d, got ((" << ih << " - 1) * "
          << scaleYN << " - " << offsetY << " + " << borderY << ") / "
          << scaleYD;
      return false;
    }
    const int64_t calculatedOutHeight = calculatedOutHeightMinusOne.value() + 1;
    if (oh != ShapedType::kDynamic && calculatedOutHeight != oh) {
      op->emitOpError("calculated output height did not match expected: ")
          << "calculated=" << calculatedOutHeight << ", expected=" << oh;
      return false;
    }
  }

  if (iw != ShapedType::kDynamic) {
    const std::optional<int64_t> calculatedOutWidthMinusOne =
        idivCheck((iw - 1) * scaleXN - offsetX + borderX, scaleXD);
    if (!calculatedOutWidthMinusOne.has_value()) {
      op->emitOpError("expected (input_width - 1) * scale_x_n - offset_x + "
                      "border_x ")
          << "to be wholly divisible by scale_x_d, got ((" << iw << " - 1) * "
          << scaleXN << " - " << offsetX << " + " << borderX << ") / "
          << scaleXD;
      return false;
    }
    const int64_t calculatedOutWidth = calculatedOutWidthMinusOne.value() + 1;
    if (ow != ShapedType::kDynamic && calculatedOutWidth != ow) {
      op->emitOpError("calculated output width did not match expected: ")
          << "calculated=" << calculatedOutWidth << ", expected=" << ow;
      return false;
    }
  }

  return true;
}

LogicalResult TosaValidation::applyErrorIfCheck(Operation *op) {
  if (!checkErrorIfResize(op))
    return failure();
  return success();
}

bool TosaValidation::isValidElementType(Type type) {
  if (isa<FloatType>(type)) {
    return isa<Float32Type, Float16Type, BFloat16Type, Float8E4M3FNType,
               Float8E5M2Type>(type);
  } else if (auto intTy = dyn_cast<IntegerType>(type)) {
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
    }
  } else if (mlir::isa<tosa::shapeType>(type)) {
    return true;
  }
  return false;
}

void TosaValidation::runOnOperation() {
  configLevelAndProfile();

  TosaDialect *tosaDialect = getContext().getLoadedDialect<TosaDialect>();
  if (!tosaDialect)
    return;

  getOperation().walk([&](Operation *op) {
    if (op->getDialect() != tosaDialect)
      return;

    // Profile-Extension based validation should be performed at the beginning.
    if (strictOpSpecAlignment &&
        failed(profileComp.checkProfile(op, targetEnv)))
      return signalPassFailure();

    if (strictOpSpecAlignment &&
        failed(profileComp.checkExtension(op, targetEnv)))
      return signalPassFailure();

    for (Value operand : op->getOperands()) {
      auto elementTy = getElementTypeOrSelf(operand);
      if (!isValidElementType(elementTy)) {
        op->emitOpError() << "is not profile-aligned: element type "
                          << elementTy << " is not legal";
        return signalPassFailure();
      }
    }
    for (Type resultTy : op->getResultTypes()) {
      auto elementTy = getElementTypeOrSelf(resultTy);
      if (!isValidElementType(elementTy)) {
        op->emitOpError() << "is not profile-aligned: element type "
                          << elementTy << " is not legal";
        return signalPassFailure();
      }
    }

    // Some uses of TOSA rely on the constant operands of particular
    // operations.
    if (strictOpSpecAlignment && failed(applyConstantOperandCheck(op)))
      signalPassFailure();

    // do level checks
    if (failed(applyLevelCheck(op)))
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
