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

    Loop-carried variables can be exposed through this interface. There are
    3 components to a loop-carried variable.
    - The "region iter_arg" is the block argument of the entry block that
      represents the loop-carried variable in each iteration.
    - The "init value" is an operand of the loop op that serves as the initial
      region iter_arg value for the first iteration (if any).
    - The "yielded" value is the value that is forwarded from one iteration to
      serve as the region iter_arg of the next iteration.

    If one of the respective interface methods is implemented, so must the other
    two. The interface verifier ensures that the number of types of the region
    iter_args, init values and yielded values match.

    Optionally, "loop results" can be exposed through this interface. These are
    the values that are returned from the loop op when there are no more
    iterations. The number and types of the loop results must match with the
    region iter_args. Note: Loop results are optional because some loops
    (e.g., `scf.while`) may produce results that do match 1-to-1 with the
    region iter_args.
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
        return !$_op->isAncestor(value.getParentRegion()->getParentOp());
      }]
    >,
    InterfaceMethod<[{
        Returns the regions that make up the body of the loop and should be
        inspected for loop-invariant operations.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::Region *>",
      /*methodName=*/"getLoopRegions"
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
    InterfaceMethod<[{
        Return the mutable "init" operands that are used as initialization
        values for the region "iter_args" of this loop.
      }],
      /*retTy=*/"::llvm::MutableArrayRef<::mlir::OpOperand>",
      /*methodName=*/"getInitsMutable",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return {};
      }]
    >,
    InterfaceMethod<[{
        Return the region "iter_args" (block arguments) that correspond to the
        "init" operands. If the op has multiple regions, return the
        corresponding block arguments of the entry region.
      }],
      /*retTy=*/"::mlir::Block::BlockArgListType",
      /*methodName=*/"getRegionIterArgs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::Block::BlockArgListType();
      }]
    >,
    InterfaceMethod<[{
        Return the mutable operand range of values that are yielded to the next
        iteration by the loop terminator.

        For loop operations that dont yield a value, this should return
        std::nullopt.
      }],
      /*retTy=*/"std::optional<::llvm::MutableArrayRef<::mlir::OpOperand>>",
      /*methodName=*/"getYieldedValuesMutable",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return std::nullopt;
      }]
    >,
    InterfaceMethod<[{
        Return the range of results that are return from this loop and
        correspond to the "init" operands.

        Note: This interface method is optional. If loop results are not
        exposed via this interface, "std::nullopt" should be returned.
        Otherwise, the number and types of results must match with the
        region iter_args, inits and yielded values that are exposed via this
        interface. If loop results are exposed but this loop op has no
        loop-carried variables, an empty result range (and not "std::nullopt")
        should be returned.
      }],
      /*retTy=*/"::std::optional<::mlir::ResultRange>",
      /*methodName=*/"getLoopResults",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::std::nullopt;
      }]
    >,
    InterfaceMethod<[{
        Append the specified additional "init" operands: replace this loop with
        a new loop that has the additional init operands. The loop body of
        this loop is moved over to the new loop.

        `newInitOperands` specifies the additional "init" operands.
        `newYieldValuesFn` is a function that returns the yielded values (which
        can be computed based on the additional region iter_args). If
        `replaceInitOperandUsesInLoop` is set, all uses of the additional init
        operands inside of this loop are replaced with the corresponding, newly
        added region iter_args.

        Note: Loops that do not support init/iter_args should return "failure".
      }],
      /*retTy=*/"::mlir::FailureOr<::mlir::LoopLikeOpInterface>",
      /*methodName=*/"replaceWithAdditionalYields",
      /*args=*/(ins "::mlir::RewriterBase &":$rewriter,
                    "::mlir::ValueRange":$newInitOperands,
                    "bool":$replaceInitOperandUsesInLoop,
                    "const ::mlir::NewYieldValuesFn &":$newYieldValuesFn),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::failure();
      }]
    >
  ];

  let extraClassDeclaration = [{
    /// Returns if a block is inside a loop (within the current function). This
    /// can either be because the block is nested inside a LoopLikeInterface, or
    /// because the control flow graph is cyclic
    static bool blockIsInLoop(Block *block);
  }];

  let extraSharedClassDeclaration = [{
    /// Append the specified additional "init" operands: replace this loop with
    /// a new loop that has the additional init operands. The loop body of this
    /// loop is moved over to the new loop.
    ///
    /// The newly added region iter_args are yielded from the loop.
    ::mlir::FailureOr<::mlir::LoopLikeOpInterface>
        replaceWithAdditionalIterOperands(::mlir::RewriterBase &rewriter,
                                          ::mlir::ValueRange newInitOperands,
                                          bool replaceInitOperandUsesInLoop) {
      return $_op.replaceWithAdditionalYields(
          rewriter, newInitOperands, replaceInitOperandUsesInLoop,
          [](OpBuilder &b, Location loc, ArrayRef<BlockArgument> newBBArgs) {
            return SmallVector<Value>(newBBArgs);
          });
    }

    /// Return the values that are yielded to the next iteration. If
    /// the loop doesnt yield any values return `{}`.
    ::mlir::ValueRange getYieldedValues() {
      auto mutableValues = $_op.getYieldedValuesMutable();
      if (!mutableValues || mutableValues->empty())
        return {};
      Operation *yieldOp = mutableValues->begin()->getOwner();
      unsigned firstOperandIndex = mutableValues->begin()->getOperandNumber();
      return OperandRange(
          yieldOp->operand_begin() + firstOperandIndex,
          yieldOp->operand_begin() + firstOperandIndex + mutableValues->size());
    }

    /// Return the "init" operands that are used as initialization values for
    /// the region "iter_args" of this loop.
    ::mlir::OperandRange getInits() {
      auto initsMutable = $_op.getInitsMutable();
      if (initsMutable.empty())
        return ::mlir::OperandRange($_op->operand_end(), $_op->operand_end());
      unsigned firstOperandIndex = initsMutable.begin()->getOperandNumber();
      return OperandRange(
          $_op->operand_begin() + firstOperandIndex,
          $_op->operand_begin() + firstOperandIndex + initsMutable.size());
    }

    /// Return the region iter_arg that corresponds to the given init operand.
    /// Return an "empty" block argument if the given operand is not an init
    /// operand of this loop op.
    BlockArgument getTiedLoopRegionIterArg(OpOperand *opOperand) {
      auto initsMutable = $_op.getInitsMutable();
      auto it = llvm::find(initsMutable, *opOperand);
      if (it == initsMutable.end())
        return {};
      return $_op.getRegionIterArgs()[std::distance(initsMutable.begin(), it)];
    }

    /// Return the region iter_arg that corresponds to the given loop result.
    /// Return an "empty" block argument if the given OpResult is not a loop
    /// result or if this op does not expose any loop results.
    BlockArgument getTiedLoopRegionIterArg(OpResult opResult) {
      auto loopResults = $_op.getLoopResults();
      if (!loopResults)
        return {};
      auto it = llvm::find(*loopResults, opResult);
      if (it == loopResults->end())
        return {};
      return $_op.getRegionIterArgs()[std::distance(loopResults->begin(), it)];
    }

    /// Return the init operand that corresponds to the given region iter_arg.
    /// Return "nullptr" if the given block argument is not a region iter_arg
    /// of this loop op.
    OpOperand *getTiedLoopInit(BlockArgument bbArg) {
      auto iterArgs = $_op.getRegionIterArgs();
      auto it = llvm::find(iterArgs, bbArg);
      if (it == iterArgs.end())
        return {};
      return &$_op.getInitsMutable()[std::distance(iterArgs.begin(), it)];
    }

    /// Return the init operand that corresponds to the given loop result.
    /// Return "nullptr" if the given OpResult is not a loop result or if this
    /// op does not expose any loop results.
    OpOperand *getTiedLoopInit(OpResult opResult) {
      auto loopResults = $_op.getLoopResults();
      if (!loopResults)
        return nullptr;
      auto it = llvm::find(*loopResults, opResult);
      if (it == loopResults->end())
        return nullptr;
      return &$_op.getInitsMutable()[std::distance(loopResults->begin(), it)];
    }

    /// Return the yielded value that corresponds to the given region iter_arg.
    /// Return "nullptr" if the given block argument is not a region iter_arg
    /// of this loop op or if there is no yield corresponding to this `bbArg`.
    OpOperand *getTiedLoopYieldedValue(BlockArgument bbArg) {
      auto iterArgs = $_op.getRegionIterArgs();
      auto it = llvm::find(iterArgs, bbArg);
      if (it == iterArgs.end())
        return {};
      std::optional<llvm::MutableArrayRef<::mlir::OpOperand>> yieldValues =
        $_op.getYieldedValuesMutable();
      if (!yieldValues)
        return {};
      return &yieldValues.value()[std::distance(iterArgs.begin(), it)];
    }

    /// Return the loop result that corresponds to the given init operand.
    /// Return an "empty" OpResult if the given operand is not an init operand
    /// of this loop op or if this op does not expose any loop results.
    OpResult getTiedLoopResult(OpOperand *opOperand) {
      auto loopResults = $_op.getLoopResults();
      if (!loopResults)
        return {};
      auto initsMutable = $_op.getInitsMutable();
      auto it = llvm::find(initsMutable, *opOperand);
      if (it == initsMutable.end())
        return {};
      return (*loopResults)[std::distance(initsMutable.begin(), it)];
    }

    /// Return the loop result that corresponds to the given region iter_arg.
    /// Return an "empty" OpResult if the given block argument is not a region
    /// iter_arg of this loop op or if this op does not expose any loop results.
    OpResult getTiedLoopResult(BlockArgument bbArg) {
      auto loopResults = $_op.getLoopResults();
      if (!loopResults)
        return {};
      auto iterArgs = $_op.getRegionIterArgs();
      auto it = llvm::find(iterArgs, bbArg);
      if (it == iterArgs.end())
        return {};
      return (*loopResults)[std::distance(iterArgs.begin(), it)];
    }
  }];

  let verifyWithRegions = 1;

  let verify = [{
    return detail::verifyLoopLikeOpInterface($_op);
  }];
}

#endif // MLIR_INTERFACES_LOOPLIKEINTERFACE
