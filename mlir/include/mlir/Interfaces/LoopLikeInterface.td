//===- LoopLikeInterface.td - LoopLike interface -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for loop-like operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_LOOPLIKEINTERFACE
#define MLIR_INTERFACES_LOOPLIKEINTERFACE

include "mlir/IR/OpBase.td"

def LoopLikeOpInterface : OpInterface<"LoopLikeOpInterface"> {
  let description = [{
    Contains helper functions to query properties and perform transformations
    of a loop. Operations that implement this interface will be considered by
    loop-invariant code motion.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns true if the given value is defined outside of the loop.
        A sensible implementation could be to check whether the value's defining
        operation lies outside of the loops body region. If the loop uses
        explicit capture of dependencies, an implementation could check whether
        the value corresponds to a captured dependency.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isDefinedOutsideOfLoop",
      /*args=*/(ins "::mlir::Value ":$value),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return value.getParentRegion()->isProperAncestor(&$_op.getLoopBody());
      }]
    >,
    InterfaceMethod<[{
        Returns the region that makes up the body of the loop and should be
        inspected for loop-invariant operations.
      }],
      /*retTy=*/"::mlir::Region &",
      /*methodName=*/"getLoopBody"
    >,
    InterfaceMethod<[{
        Moves the given loop-invariant operation out of the loop.
      }],
      /*retTy=*/"void",
      /*methodName=*/"moveOutOfLoop",
      /*args=*/(ins "::mlir::Operation *":$op),
      /*methodBody=*/"",
      /*defaultImplementation=*/"op->moveBefore($_op);"
    >,
    InterfaceMethod<[{
        Promotes the loop body to its containing block if the loop is known to
        have a single iteration. Returns "success" if the promotion was
        successful.
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"promoteIfSingleIteration",
      /*args=*/(ins "::mlir::RewriterBase &":$rewriter),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::failure();
      }]
    >,
    InterfaceMethod<[{
        If there is a single induction variable return it, otherwise return
        std::nullopt.
      }],
      /*retTy=*/"::std::optional<::mlir::Value>",
      /*methodName=*/"getSingleInductionVar",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return std::nullopt;
      }]
    >,
    InterfaceMethod<[{
        Return the single lower bound value or attribute if it exists, otherwise
        return std::nullopt.
      }],
      /*retTy=*/"::std::optional<::mlir::OpFoldResult>",
      /*methodName=*/"getSingleLowerBound",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return std::nullopt;
      }]
    >,
    InterfaceMethod<[{
        Return the single step value or attribute if it exists, otherwise
        return std::nullopt.
      }],
      /*retTy=*/"::std::optional<::mlir::OpFoldResult>",
      /*methodName=*/"getSingleStep",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return std::nullopt;
      }]
    >,
    InterfaceMethod<[{
        Return the single upper bound value or attribute if it exists, otherwise
        return std::nullopt.
      }],
      /*retTy=*/"::std::optional<::mlir::OpFoldResult>",
      /*methodName=*/"getSingleUpperBound",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return std::nullopt;
      }]
    >,
  ];

  let extraClassDeclaration = [{
    /// Returns if a block is inside a loop (within the current function). This
    /// can either be because the block is nested inside a LoopLikeInterface, or
    /// because the control flow graph is cyclic
    static bool blockIsInLoop(Block *block);
  }];
}

#endif // MLIR_INTERFACES_LOOPLIKEINTERFACE
