//===- TosaInputShape.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Change input shape of function argument to specified shape.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSAINPUTSHAPE
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {

std::pair<std::vector<std::pair<size_t, std::vector<int64_t>>>, std::string>
parse_input_shapes(std::vector<std::string> args) {
  /**
   * This function returns two values: a vector of parsed arguments, and an
   * optional error message. Each arguments contains its argument number and the
   * shape. For example:
   * "args=arg0:5x10,arg8:3x9" => {{{0, {5, 10}}, {8, {3, 9}}}, ""}
   * "args=arg0:" => {{}, "error message"}
   */

  std::vector<std::pair<size_t, std::vector<int64_t>>> shapes;

  for (std::string arg : args) {
    if (arg.substr(0, 3) != "arg") {
      return {{}, "Arguments must start with 'arg'"};
    }

    char *endptr;
    size_t argnum = std::strtoul(&arg[3], &endptr, /*base=*/10);
    if (*endptr != ':') {
      return {{}, "Invalid argument name"};
    }
    std::string shape_str = endptr + 1;

    std::vector<int64_t> curr;
    while (!shape_str.empty()) {
      size_t dim = std::strtoul(shape_str.data(), &endptr, /*base=*/10);
      if ((*endptr != '\0' && *endptr != 'x') || shape_str == endptr) {
        return {{}, "Invalid input shape description"};
      }
      curr.push_back(dim);
      if (*endptr == '\0') {
        break;
      }
      shape_str = endptr + 1;
    }
    shapes.push_back({argnum, curr});
  }
  return {shapes, ""};
}

/// Pass that change function input shapes to specified static input shapes
struct TosaInputShape : public tosa::impl::TosaInputShapeBase<TosaInputShape> {
public:
  TosaInputShape() = default;
  explicit TosaInputShape(std::vector<std::string> args) : TosaInputShape() {
    this->args = args;
  }
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto [args_parsed, args_parse_err] = parse_input_shapes(args);

    if (!args_parse_err.empty()) {
      func.emitError() << args_parse_err;
      return;
    }

    for (auto &block : func.getBody()) {

      for (auto [argnum, shape] : args_parsed) {
        if (argnum >= block.getNumArguments()) {
          func.emitError() << "arg" << argnum << " doesn't exist.";
          return;
        }
        BlockArgument block_arg = block.getArgument(argnum);
        Type arg_type = block_arg.getType();
        TensorType tensor_type = cast<TensorType>(arg_type);
        if (failed(
                mlir::verifyCompatibleShape(tensor_type.getShape(), shape))) {
          func->emitError()
              << "arg" << argnum << " has incompatible shape with input shape.";
          return;
        }
        SmallVector<int64_t> new_shape(shape.begin(), shape.end());
        auto new_tensor_type =
            tensor_type.cloneWith(new_shape, tensor_type.getElementType());
        block_arg.setType(new_tensor_type);
      }

      bool found_func_op = false;

      for (Operation &op : block) {
        // Update result shape for func.func
        func::FuncOp funcOp = mlir::dyn_cast<func::FuncOp>(op.getParentOp());
        if (funcOp && !found_func_op) {
          FunctionType old_function_type = funcOp.getFunctionType();
          std::vector<Type> inputs = old_function_type.getInputs();

          for (auto [argnum, shape] : args_parsed) {
            if ((size_t)argnum >= inputs.size()) {
              func.emitError() << "arg" << argnum << " doesn't exist.";
              return;
            }
            auto tensor_type = cast<TensorType>(inputs[argnum]);

            if (failed(mlir::verifyCompatibleShape(tensor_type.getShape(),
                                                   shape))) {
              funcOp->emitError()
                  << "arg" << argnum
                  << " has incompatible shape with input shape.";
              return;
            }
            SmallVector<int64_t> new_shape(shape.begin(), shape.end());
            auto new_tensor_type =
                tensor_type.cloneWith(new_shape, tensor_type.getElementType());
            inputs[argnum] = cast<Type>(new_tensor_type);
          }

          FunctionType new_function_type = old_function_type.clone(
              TypeRange{ArrayRef(inputs)},
              TypeRange{old_function_type.getResults()});
          funcOp.setFunctionType(new_function_type);
          found_func_op = true;
        }
        // Update result shape of func.return
        func::ReturnOp returnOp = mlir::dyn_cast<func::ReturnOp>(op);
        if (returnOp) {
          func::FuncOp funcOp = dyn_cast<func::FuncOp>(op.getParentOp());
          if (funcOp) {
            FunctionType old_function_type = funcOp.getFunctionType();
            FunctionType new_function_type = old_function_type.clone(
                TypeRange{old_function_type.getInputs()},
                returnOp.getOperandTypes());
            funcOp.setFunctionType(new_function_type);
          }
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::tosa::createTosaInputShapePass(std::vector<std::string> args) {
  return std::make_unique<TosaInputShape>(args);
}
