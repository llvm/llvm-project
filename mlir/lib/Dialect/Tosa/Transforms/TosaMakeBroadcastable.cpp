//===- TosaMakeBroadcastable.cpp ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Insert reshape to binary op's input if needed to match rank
//
//===----------------------------------------------------------------------===//

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR//TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define PASS_NAME "tosa-make=broadcastable"
#define DEBUG_TYPE PASS_NAME

namespace mlir {

namespace tosa {

namespace {

class TosaMakeBroadcastable
    : public PassWrapper<TosaMakeBroadcastable, FunctionPass> {
public:
  explicit TosaMakeBroadcastable() {}
  void runOnFunction() override;
};

#define REPLACE_OP_LOGICAL(tosa_op, LHS_VALUE, RHS_VALUE)

#define REPLACE_OP(tosa_op, LHS_VALUE, RHS_VALUE)                              \
  {                                                                            \
    rewriter.replaceOpWithNewOp<tosa::tosa_op##Op>(op, output_type, LHS_VALUE, \
                                                   RHS_VALUE);                 \
  }

/* the legalization macro that reshapes lower rank input to output's shape
 * if lower=[a], target=[a, b, c], [a] reshaped into [a, 1, 1]
 * if lower=[b], target=[a, b, c], [b] should but NOT YET reshaped into [1, b,
 * 1] (TODO)
 * if lower=[c], target=[a, b, c], [c] reshaped into [1, 1, c]
 * if lower=[a, c], target=[a, b, c], [a, c] reshaped into [a, 1, c]
 * if lower=[a, b], target=[a, b, c], [a, b] reshaped into [a, b, 1]
 * if lower=[b, c], target=[a, b, c], [b, c] reshaped into [1, b, c]
 * if lower=[a], target=[a, a], [a] reshaped into [1, a] instead of [a, 1]
 * if lower=[a], target=[a, b, a], [a] reshaped into [1, 1, a]
 * if lower=[], target=[a, b, c], [] reshaped into [1, 1, 1] */

#define DECL_TOSACONVERT_OP(tosa_op)                                           \
  struct ConvertTosa##tosa_op##Op : public RewritePattern {                    \
    explicit ConvertTosa##tosa_op##Op(MLIRContext *context)                    \
        : RewritePattern(tosa::tosa_op##Op::getOperationName(), 1, context) {} \
    LogicalResult matchAndRewrite(Operation *op,                               \
                                  PatternRewriter &rewriter) const {           \
      auto tosa_binary_op = cast<tosa::tosa_op##Op>(op);                       \
                                                                               \
      auto lhs = tosa_binary_op.lhs();                                         \
      auto rhs = tosa_binary_op.rhs();                                         \
                                                                               \
      int64_t lhs_rank = lhs.getType().dyn_cast<RankedTensorType>().getRank(); \
      int64_t rhs_rank = rhs.getType().dyn_cast<RankedTensorType>().getRank(); \
                                                                               \
      auto output_type =                                                       \
          tosa_binary_op.getResult().getType().dyn_cast<RankedTensorType>();   \
                                                                               \
      int64_t higher_rank, lower_rank;                                         \
      Value higher_tensor_value, lower_tensor_value;                           \
      /* return if rank already match */                                       \
      if (lhs_rank == rhs_rank) {                                              \
        return failure();                                                      \
      } else if (lhs_rank > rhs_rank) {                                        \
        higher_rank = lhs_rank;                                                \
        lower_rank = rhs_rank;                                                 \
        higher_tensor_value = lhs;                                             \
        lower_tensor_value = rhs;                                              \
      } else {                                                                 \
        higher_rank = rhs_rank;                                                \
        lower_rank = lhs_rank;                                                 \
        higher_tensor_value = rhs;                                             \
        lower_tensor_value = lhs;                                              \
      }                                                                        \
                                                                               \
      ArrayRef<int64_t> higher_rank_shape = output_type.getShape();            \
      ArrayRef<int64_t> lower_rank_shape = lower_tensor_value.getType()        \
                                               .dyn_cast<RankedTensorType>()   \
                                               .getShape();                    \
                                                                               \
      SmallVector<int64_t, 8> reshape_output_shape;                            \
      reshape_output_shape.assign(higher_rank, 1);                             \
                                                                               \
      int64_t higher_left_index = 0;                                           \
      int64_t higher_right_index = higher_rank;                                \
      int64_t lower_left_index = 0;                                            \
      int64_t lower_right_index = lower_rank;                                  \
      int64_t higher_rank_dim, lower_rank_dim;                                 \
                                                                               \
      if (lower_right_index != 0 && higher_right_index != 0) {                 \
        while (true) {                                                         \
          higher_rank_dim = higher_rank_shape[higher_right_index - 1];         \
          lower_rank_dim = lower_rank_shape[lower_right_index - 1];            \
          if (higher_rank_dim == lower_rank_dim) {                             \
            reshape_output_shape[higher_right_index - 1] = higher_rank_dim;    \
                                                                               \
            if (higher_right_index > 0) {                                      \
              higher_right_index--;                                            \
            }                                                                  \
                                                                               \
            if (lower_right_index > 0) {                                       \
              lower_right_index--;                                             \
            }                                                                  \
                                                                               \
            if (higher_right_index == 0 || lower_right_index == 0) {           \
              break;                                                           \
            }                                                                  \
          } else {                                                             \
            break;                                                             \
          }                                                                    \
        }                                                                      \
        if (lower_right_index != 0 && higher_right_index != 0) {               \
          while (true) {                                                       \
            higher_rank_dim = higher_rank_shape[higher_left_index];            \
            lower_rank_dim = lower_rank_shape[lower_left_index];               \
            if (higher_rank_dim == lower_rank_dim) {                           \
              reshape_output_shape[higher_left_index] = higher_rank_dim;       \
                                                                               \
              if (higher_left_index < higher_right_index) {                    \
                higher_left_index++;                                           \
              }                                                                \
                                                                               \
              if (lower_left_index < lower_right_index) {                      \
                lower_left_index++;                                            \
              }                                                                \
                                                                               \
              if (higher_left_index == higher_right_index ||                   \
                  lower_left_index == lower_right_index) {                     \
                break;                                                         \
              }                                                                \
            } else {                                                           \
              break;                                                           \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
                                                                               \
      auto reshape_input_type =                                                \
          lower_tensor_value.getType().dyn_cast<RankedTensorType>();           \
      auto reshape_output_type =                                               \
          RankedTensorType::get(ArrayRef<int64_t>(reshape_output_shape),       \
                                reshape_input_type.getElementType());          \
                                                                               \
      auto reshape_lower = rewriter.create<tosa::ReshapeOp>(                   \
          op->getLoc(), reshape_output_type, lower_tensor_value,               \
          rewriter.getI64ArrayAttr(reshape_output_shape));                     \
                                                                               \
      if (lhs_rank > rhs_rank) {                                               \
        REPLACE_OP(tosa_op, higher_tensor_value, reshape_lower.getResult());   \
      } else {                                                                 \
        REPLACE_OP(tosa_op, reshape_lower.getResult(), higher_tensor_value);   \
      }                                                                        \
                                                                               \
      return success();                                                        \
    }                                                                          \
  };
DECL_TOSACONVERT_OP(Add)
DECL_TOSACONVERT_OP(Sub)
DECL_TOSACONVERT_OP(Mul)
DECL_TOSACONVERT_OP(LogicalLeftShift)
DECL_TOSACONVERT_OP(ArithmeticRightShift)
DECL_TOSACONVERT_OP(LogicalRightShift)
#undef DECL_TOSACONVERT_OP

#undef REPLACE_OP

void TosaMakeBroadcastable::runOnFunction() {
  OwningRewritePatternList patterns;
  auto *ctx = &getContext();
  auto func = getFunction();

  // Add the generated patterns to the list.
  patterns.insert<ConvertTosaAddOp>(ctx);
  patterns.insert<ConvertTosaSubOp>(ctx);
  patterns.insert<ConvertTosaMulOp>(ctx);
  patterns.insert<ConvertTosaLogicalLeftShiftOp>(ctx);
  patterns.insert<ConvertTosaArithmeticRightShiftOp>(ctx);
  patterns.insert<ConvertTosaLogicalRightShiftOp>(ctx);
  applyPatternsAndFoldGreedily(func, std::move(patterns));
}

} // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> CreateTosaMakeBroadcastablePass() {
  return std::make_unique<TosaMakeBroadcastable>();
}

static PassRegistration<TosaMakeBroadcastable>
    pass(PASS_NAME,
         "Perform broadcast on elementwise TosaOps to ensure same rank");

} // namespace tosa

} // namespace mlir
