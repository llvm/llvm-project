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

#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/PassesEnums.cpp.inc"

#include <string>
#include <unordered_map>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

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

static LogicalResult checkConstantOperandTranspose(Operation *op) {
  if (auto transposeOp = dyn_cast<tosa::TransposeOp>(op)) {
    DenseElementsAttr perms;
    if (!matchPattern(transposeOp.getPerms(), m_Constant(&perms)))
      return op->emitOpError("perms of transpose is not constant");
  }
  return success();
}

static LogicalResult checkConstantOperandFullyConnected(Operation *op) {
  if (auto fcOp = dyn_cast<tosa::FullyConnectedOp>(op)) {
    DenseElementsAttr weight;
    if (!matchPattern(fcOp.getWeight(), m_Constant(&weight)))
      return op->emitOpError("weight of fully_connected is not constant");

    DenseElementsAttr bias;
    if (!matchPattern(fcOp.getBias(), m_Constant(&bias)))
      return op->emitOpError("bias of fully_connected is not constant");
  }
  return success();
}

struct TosaLevel {
  int32_t MAX_RANK = 0;
  int32_t MAX_KERNEL = 0;
  int32_t MAX_STRIDE = 0;
  int32_t MAX_SCALE = 0;

  // @todo: MAX_LOG2_SIZE value and checks

  bool operator==(const TosaLevel &rhs) {
    return MAX_RANK == rhs.MAX_RANK && MAX_KERNEL == rhs.MAX_KERNEL &&
           MAX_STRIDE == rhs.MAX_STRIDE && MAX_SCALE == rhs.MAX_SCALE;
  }
};

static constexpr TosaLevel TOSA_LEVEL_EIGHTK = {6, 8192, 8192, 256};
static constexpr TosaLevel TOSA_LEVEL_NONE = {0, 0, 0, 0};

//===----------------------------------------------------------------------===//
// TOSA Validation Pass.
//===----------------------------------------------------------------------===//

struct TosaValidation : public tosa::impl::TosaValidationBase<TosaValidation> {
public:
  explicit TosaValidation() { populateConstantOperandChecks(); }
  explicit TosaValidation(const TosaValidationOptions &options)
      : TosaValidation() {
    this->profile = options.profile;
    this->StrictOperationSpecAlignment = options.StrictOperationSpecAlignment;
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

private:
  void populateConstantOperandChecks() {
    constCheckers.emplace_back(checkConstantOperandPad);
    constCheckers.emplace_back(checkConstantOperandTranspose);
    constCheckers.emplace_back(checkConstantOperandFullyConnected);
  }

  bool levelCheckKernel(Operation *op, int32_t v,
                        const std::string &checkDesc) {
    if (v > tosaLevel.MAX_KERNEL) {
      op->emitOpError() << "failed level check: " << checkDesc;
      return false;
    }
    return true;
  }

  bool levelCheckStride(Operation *op, int32_t v,
                        const std::string &checkDesc) {
    if (v > tosaLevel.MAX_STRIDE) {
      op->emitOpError() << "failed level check: " << checkDesc;
      return false;
    }
    return true;
  }

  bool levelCheckScale(Operation *op, int32_t v, const std::string &checkDesc) {
    if (v > tosaLevel.MAX_SCALE) {
      op->emitOpError() << "failed level check: " << checkDesc;
      return false;
    }
    return true;
  }

  bool levelCheckRank(Operation *op, const Value &v,
                      const std::string &checkDesc) {
    if (ShapedType type = dyn_cast<ShapedType>(v.getType())) {
      if (type.getRank() > tosaLevel.MAX_RANK) {
        op->emitOpError() << "failed level check: " << checkDesc;
        return false;
      }
    }
    return true;
  }

  template <typename T>
  bool levelCheckRanksFor(Operation *op) {
    if (dyn_cast<T>(op)) {
      // level check ranks of all operands and results
      for (auto v : op->getOperands()) {
        if (!levelCheckRank(op, v, "operand rank(shape) <= MAX_RANK"))
          return false;
      }
      for (auto v : op->getResults()) {
        if (!levelCheckRank(op, v, "result rank(shape) <= MAX_RANK"))
          return false;
      }
    }
    return true;
  }

  bool levelCheckRanks(Operation *op) {
#define CHECK_RANKS_FOR(tosaOp)                                                \
  if (!levelCheckRanksFor<tosaOp##Op>(op))                                     \
    return false;

    // tensor operators:
    CHECK_RANKS_FOR(ArgMax);
    // all activation functions:
    CHECK_RANKS_FOR(Clamp);
    CHECK_RANKS_FOR(Sigmoid);
    CHECK_RANKS_FOR(Tanh);
    // all elementwise binary operators:
    CHECK_RANKS_FOR(Add);
    CHECK_RANKS_FOR(ArithmeticRightShift);
    CHECK_RANKS_FOR(BitwiseAnd);
    CHECK_RANKS_FOR(BitwiseOr);
    CHECK_RANKS_FOR(BitwiseXor);
    CHECK_RANKS_FOR(Div);
    CHECK_RANKS_FOR(LogicalAnd);
    CHECK_RANKS_FOR(LogicalLeftShift);
    CHECK_RANKS_FOR(LogicalRightShift);
    CHECK_RANKS_FOR(LogicalOr);
    CHECK_RANKS_FOR(LogicalXor);
    CHECK_RANKS_FOR(Maximum);
    CHECK_RANKS_FOR(Minimum);
    CHECK_RANKS_FOR(Mul);
    CHECK_RANKS_FOR(Pow);
    CHECK_RANKS_FOR(Sub);
    CHECK_RANKS_FOR(Table);
    // all elementwise unary operators:
    CHECK_RANKS_FOR(Abs);
    CHECK_RANKS_FOR(BitwiseNot);
    CHECK_RANKS_FOR(Ceil);
    CHECK_RANKS_FOR(Clz);
    CHECK_RANKS_FOR(Exp);
    CHECK_RANKS_FOR(Floor);
    CHECK_RANKS_FOR(Log);
    CHECK_RANKS_FOR(LogicalNot);
    CHECK_RANKS_FOR(Negate);
    CHECK_RANKS_FOR(Reciprocal);
    CHECK_RANKS_FOR(Rsqrt);
    // all elementwise ternary operators:
    CHECK_RANKS_FOR(Select);
    // all comparison operators:
    CHECK_RANKS_FOR(Equal);
    CHECK_RANKS_FOR(Greater);
    CHECK_RANKS_FOR(GreaterEqual);
    // all reduction operators:
    CHECK_RANKS_FOR(ReduceAll);
    CHECK_RANKS_FOR(ReduceAny);
    CHECK_RANKS_FOR(ReduceMax);
    CHECK_RANKS_FOR(ReduceMin);
    CHECK_RANKS_FOR(ReduceProd);
    CHECK_RANKS_FOR(ReduceSum);
    // all data layout operators:
    CHECK_RANKS_FOR(Concat);
    CHECK_RANKS_FOR(Pad);
    CHECK_RANKS_FOR(Reshape);
    CHECK_RANKS_FOR(Reverse);
    CHECK_RANKS_FOR(Slice);
    CHECK_RANKS_FOR(Tile);
    CHECK_RANKS_FOR(Transpose);
    // all type conversion operators:
    CHECK_RANKS_FOR(Cast);
    CHECK_RANKS_FOR(Rescale);
    // all data nodes operators:
    CHECK_RANKS_FOR(Const);
    CHECK_RANKS_FOR(Identity);

#undef CHECK_RANKS_FOR
    return true;
  }

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
              transpose.getFilter().getType().dyn_cast<ShapedType>()) {
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
      auto scale = resize.getScale();
      int16_t scaleYN = scale[0];
      int16_t scaleYD = scale[1];
      int16_t scaleXN = scale[2];
      int16_t scaleXD = scale[3];
      if (!levelCheckScale(op, scaleYN / scaleYD,
                           "scale_y_n/scale_y_d <= MAX_SCALE") ||
          !levelCheckScale(op, scaleXN / scaleXD,
                           "scale_x_n/scale_x_d <= MAX_SCALE")) {
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
  }

  bool CheckVariable(Operation *op);
  bool CheckVariableReadOrWrite(Operation *op);

  SmallVector<std::function<LogicalResult(Operation *)>> constCheckers;
  TosaLevel tosaLevel;
  DenseMap<StringAttr, mlir::Type> variablesMap;
};

LogicalResult TosaValidation::applyLevelCheck(Operation *op) {
  if (tosaLevel == TOSA_LEVEL_NONE) {
    // no need to do level checks
    return success();
  }

  if (!levelCheckRanks(op)) {
    return failure();
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

void TosaValidation::runOnOperation() {
  configLevelAndProfile();
  getOperation().walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if ((profile == TosaProfileEnum::BaseInference) &&
          isa<FloatType>(getElementTypeOrSelf(operand))) {
        return signalPassFailure();
      }
      if (getElementTypeOrSelf(operand).isF64()) {
        return signalPassFailure();
      }
    }

    // Some uses of TOSA rely on the constant operands of particular
    // operations.
    if (StrictOperationSpecAlignment && failed(applyConstantOperandCheck(op)))
      signalPassFailure();

    // do level checks
    if (failed(applyLevelCheck(op)))
      signalPassFailure();

    // do variable type checks
    if (failed(applyVariableCheck(op)))
      signalPassFailure();
  });
}
} // namespace
