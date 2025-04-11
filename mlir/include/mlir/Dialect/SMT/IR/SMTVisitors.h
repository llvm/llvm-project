//===- SMTVisitors.h - SMT Dialect Visitors ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines visitors that make it easier to work with the SMT IR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SMT_IR_SMTVISITORS_H
#define MLIR_DIALECT_SMT_IR_SMTVISITORS_H

#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace smt {

/// This helps visit SMT nodes.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class SMTOpVisitor {
public:
  ResultType dispatchSMTOpVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<
            // Constants
            BoolConstantOp, IntConstantOp, BVConstantOp,
            // Bit-vector arithmetic
            BVNegOp, BVAddOp, BVMulOp, BVURemOp, BVSRemOp, BVSModOp, BVShlOp,
            BVLShrOp, BVAShrOp, BVUDivOp, BVSDivOp,
            // Bit-vector bitwise
            BVNotOp, BVAndOp, BVOrOp, BVXOrOp,
            // Other bit-vector ops
            ConcatOp, ExtractOp, RepeatOp, BVCmpOp, BV2IntOp,
            // Int arithmetic
            IntAddOp, IntMulOp, IntSubOp, IntDivOp, IntModOp, IntCmpOp,
            Int2BVOp,
            // Core Ops
            EqOp, DistinctOp, IteOp,
            // Variable/symbol declaration
            DeclareFunOp, ApplyFuncOp,
            // solver interaction
            SolverOp, AssertOp, ResetOp, PushOp, PopOp, CheckOp, SetLogicOp,
            // Boolean logic
            NotOp, AndOp, OrOp, XOrOp, ImpliesOp,
            // Arrays
            ArrayStoreOp, ArraySelectOp, ArrayBroadcastOp,
            // Quantifiers
            ForallOp, ExistsOp, YieldOp>([&](auto expr) -> ResultType {
          return thisCast->visitSMTOp(expr, args...);
        })
        .Default([&](auto expr) -> ResultType {
          return thisCast->visitInvalidSMTOp(op, args...);
        });
  }

  /// This callback is invoked on any non-expression operations.
  ResultType visitInvalidSMTOp(Operation *op, ExtraArgs... args) {
    op->emitOpError("unknown SMT node");
    abort();
  }

  /// This callback is invoked on any SMT operations that are not
  /// handled by the concrete visitor.
  ResultType visitUnhandledSMTOp(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE, OPKIND)                                                 \
  ResultType visitSMTOp(OPTYPE op, ExtraArgs... args) {                        \
    return static_cast<ConcreteType *>(this)->visit##OPKIND##SMTOp(op,         \
                                                                   args...);   \
  }

  // Constants
  HANDLE(BoolConstantOp, Unhandled);
  HANDLE(IntConstantOp, Unhandled);
  HANDLE(BVConstantOp, Unhandled);

  // Bit-vector arithmetic
  HANDLE(BVNegOp, Unhandled);
  HANDLE(BVAddOp, Unhandled);
  HANDLE(BVMulOp, Unhandled);
  HANDLE(BVURemOp, Unhandled);
  HANDLE(BVSRemOp, Unhandled);
  HANDLE(BVSModOp, Unhandled);
  HANDLE(BVShlOp, Unhandled);
  HANDLE(BVLShrOp, Unhandled);
  HANDLE(BVAShrOp, Unhandled);
  HANDLE(BVUDivOp, Unhandled);
  HANDLE(BVSDivOp, Unhandled);

  // Bit-vector bitwise operations
  HANDLE(BVNotOp, Unhandled);
  HANDLE(BVAndOp, Unhandled);
  HANDLE(BVOrOp, Unhandled);
  HANDLE(BVXOrOp, Unhandled);

  // Other bit-vector operations
  HANDLE(ConcatOp, Unhandled);
  HANDLE(ExtractOp, Unhandled);
  HANDLE(RepeatOp, Unhandled);
  HANDLE(BVCmpOp, Unhandled);
  HANDLE(BV2IntOp, Unhandled);

  // Int arithmetic
  HANDLE(IntAddOp, Unhandled);
  HANDLE(IntMulOp, Unhandled);
  HANDLE(IntSubOp, Unhandled);
  HANDLE(IntDivOp, Unhandled);
  HANDLE(IntModOp, Unhandled);

  HANDLE(IntCmpOp, Unhandled);
  HANDLE(Int2BVOp, Unhandled);

  HANDLE(EqOp, Unhandled);
  HANDLE(DistinctOp, Unhandled);
  HANDLE(IteOp, Unhandled);

  HANDLE(DeclareFunOp, Unhandled);
  HANDLE(ApplyFuncOp, Unhandled);

  HANDLE(SolverOp, Unhandled);
  HANDLE(AssertOp, Unhandled);
  HANDLE(ResetOp, Unhandled);
  HANDLE(PushOp, Unhandled);
  HANDLE(PopOp, Unhandled);
  HANDLE(CheckOp, Unhandled);
  HANDLE(SetLogicOp, Unhandled);

  // Boolean logic operations
  HANDLE(NotOp, Unhandled);
  HANDLE(AndOp, Unhandled);
  HANDLE(OrOp, Unhandled);
  HANDLE(XOrOp, Unhandled);
  HANDLE(ImpliesOp, Unhandled);

  // Array operations
  HANDLE(ArrayStoreOp, Unhandled);
  HANDLE(ArraySelectOp, Unhandled);
  HANDLE(ArrayBroadcastOp, Unhandled);

  // Quantifier operations
  HANDLE(ForallOp, Unhandled);
  HANDLE(ExistsOp, Unhandled);
  HANDLE(YieldOp, Unhandled);

#undef HANDLE
};

/// This helps visit SMT types.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class SMTTypeVisitor {
public:
  ResultType dispatchSMTTypeVisitor(Type type, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Type, ResultType>(type)
        .template Case<BoolType, IntType, BitVectorType, ArrayType, SMTFuncType,
                       SortType>([&](auto expr) -> ResultType {
          return thisCast->visitSMTType(expr, args...);
        })
        .Default([&](auto expr) -> ResultType {
          return thisCast->visitInvalidSMTType(type, args...);
        });
  }

  /// This callback is invoked on any non-expression types.
  ResultType visitInvalidSMTType(Type type, ExtraArgs... args) { abort(); }

  /// This callback is invoked on any SMT type that are not
  /// handled by the concrete visitor.
  ResultType visitUnhandledSMTType(Type type, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(TYPE, KIND)                                                     \
  ResultType visitSMTType(TYPE op, ExtraArgs... args) {                        \
    return static_cast<ConcreteType *>(this)->visit##KIND##SMTType(op,         \
                                                                   args...);   \
  }

  HANDLE(BoolType, Unhandled);
  HANDLE(IntegerType, Unhandled);
  HANDLE(BitVectorType, Unhandled);
  HANDLE(ArrayType, Unhandled);
  HANDLE(SMTFuncType, Unhandled);
  HANDLE(SortType, Unhandled);

#undef HANDLE
};

} // namespace smt
} // namespace mlir

#endif // MLIR_DIALECT_SMT_IR_SMTVISITORS_H
