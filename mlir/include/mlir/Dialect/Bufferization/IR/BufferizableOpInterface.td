//===-- BufferizableOpInterface.td - Bufferizable Ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUFFERIZABLE_OP_INTERFACE
#define BUFFERIZABLE_OP_INTERFACE

include "mlir/IR/OpBase.td"

def BufferizableOpInterface : OpInterface<"BufferizableOpInterface"> {
  let description = [{
    An op interface for One-Shot Bufferize. Ops that implement this interface
    interface can be analyzed and bufferized using One-Shot Bufferize.

    Note: All "bufferizesTo*" and "getAliasing*" interface methods must be
    implemented conservatively. If it is not statically known whether an
    OpOperand/OpResult bufferizes in a certain way (e.g., to a memory write),
    the worst case must be assumed (e.g., that it does). Similarly,
    "getAliasing*" interface methods may always return additional OpOperands or
    OpResults, but must not miss an OpOperand or OpResult that could potentially
    alias at runtime.
  }];
  let cppNamespace = "::mlir::bufferization";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Return `true` if the given OpResult may bufferize to a new buffer
          allocation. If it is statically unknown if the given OpResult
          bufferizes to a buffer allocation, `true` should be returned.
        }],
        /*retType=*/"bool",
        /*methodName=*/"bufferizesToAllocation",
        /*args=*/(ins "OpResult":$opResult),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return false;"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return `true` if the given OpOperand bufferizes to a memory read. This
          method will never be called on OpOperands that do not have a tensor
          type.

          Note: It is always safe to consider an OpOperand as a memory read,
          even if it does actually not read; however, this can introduce
          unnecessary out-of-place bufferization decisions. One-Shot Analysis
          considers OpOperands of unknown ops (that do not implement this
          interface) as reading OpOperands.
        }],
        /*retType=*/"bool",
        /*methodName=*/"bufferizesToMemoryRead",
        /*args=*/(ins "OpOperand &":$opOperand,
                      "const AnalysisState &":$state),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          // Does not have to be implemented for ops without tensor OpOperands.
          llvm_unreachable("bufferizesToMemoryRead not implemented");
         }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return `true` if the given OpOperand bufferizes to a memory write.

          This method will never be called on OpOperands that do not have a
          tensor type.

          This method will never be called on OpOperands that do not have an
          aliasing OpResult. Intuitively, it does not make sense for an
          OpOperand to bufferize to a memory write without returning an aliasing
          tensor, because the write would have no visible effect outside of the
          op.

          Note: It is always safe to consider an OpOperand as a memory write,
          even if it does actually not write; however, this can introduce
          unnecessary out-of-place bufferization decisions. One-Shot Analysis
          considers OpOperands of unknown ops (that do not implement this
          interface) as writing OpOperands.
        }],
        /*retType=*/"bool",
        /*methodName=*/"bufferizesToMemoryWrite",
        /*args=*/(ins "OpOperand &":$opOperand,
                      "const AnalysisState &":$state),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          // Does not have to be implemented for ops without tensor OpOperands.
          // Does not have to be implemented for OpOperands that do not have an
          // aliasing OpResult.
          llvm_unreachable("bufferizesToMemoryWrite not implemented");
         }]
      >,
      InterfaceMethod<
          /*desc=*/[{
            Return `true` if the given OpResult is a memory write. This is the
            case if in the following cases:

            * The corresponding aliasing OpOperand bufferizes to a memory write.
            * Or: There is no corresponding aliasing OpOperand.

            If the OpResult has multiple aliasing OpOperands, this method
            returns `true` if at least one of them bufferizes to a memory write.
          }],
          /*retType=*/"bool",
          /*methodName=*/"isMemoryWrite",
          /*args=*/(ins "OpResult":$opResult,
                        "const AnalysisState &":$state),
          /*methodBody=*/"",
          /*defaultImplementation=*/[{
            auto bufferizableOp =
                cast<BufferizableOpInterface>($_op.getOperation());
            SmallVector<OpOperand*> opOperands =
              bufferizableOp.getAliasingOpOperand(opResult, state);
            if (opOperands.empty())
              return true;
            return llvm::any_of(
                opOperands,
                [&](OpOperand *operand) {
                  return bufferizableOp.bufferizesToMemoryWrite(*operand,
                                                                state);
                });
          }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return `true` if the given OpOperand must bufferize in-place. Alias
          sets and inplace attributes will be set up accordingly before making
          any other bufferization decisions. This method will never be called on
          OpOperands that do not have a tensor type.
        }],
        /*retType=*/"bool",
        /*methodName=*/"mustBufferizeInPlace",
        /*args=*/(ins "OpOperand &":$opOperand,
                      "const AnalysisState &":$state),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return false;
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the OpResult that aliases with a given OpOperand when
          bufferized in-place. This method will never be called on OpOperands
          that do not have a tensor type.
        }],
        /*retType=*/"SmallVector<OpResult>",
        /*methodName=*/"getAliasingOpResult",
        /*args=*/(ins "OpOperand &":$opOperand,
                      "const AnalysisState &":$state),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          // Does not have to be implemented for ops without tensor OpOperands.
          llvm_unreachable("getAliasingOpResult not implemented");
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the OpOperands that alias with a given OpResult when
          bufferized in-place. This method will never be called on OpResults
          that do not have a tensor type.

          By default, this method is the inverse of `getAliasingOpResult`. Ops
          with a region that yield values may want to override this method to
          return the OpOperands that are yielded by the terminator.

          Note: This method can return multiple OpOperands, indicating that the
          given OpResult may at runtime alias with any of the OpOperands. This
          is useful for branches and for ops such as `arith.select`.
        }],
        /*retType=*/"SmallVector<OpOperand *>",
        /*methodName=*/"getAliasingOpOperand",
        /*args=*/(ins "OpResult":$opResult,
                      "const AnalysisState &":$state),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          assert(opResult.getType().isa<TensorType>() &&
                 "expected OpResult with tensor type");
          SmallVector<OpOperand *> result;
          auto bufferizableOp =
              cast<BufferizableOpInterface>($_op.getOperation());
          for (OpOperand &opOperand : $_op.getOperation()->getOpOperands()) {
            if (!opOperand.get().getType().isa<TensorType>())
              continue;
            SmallVector<OpResult> aliasingOpResults =
                bufferizableOp.getAliasingOpResult(opOperand, state);
            if (llvm::is_contained(aliasingOpResults, opResult))
              result.push_back(&opOperand);
          }
          return result;
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the buffer relation between the given OpResult and its aliasing
          OpOperands when bufferized in-place. Most OpOperands have an
          "equivalence" relation. This method will never be called on OpResults
          that do not have a tensor type. It will also never be called on
          OpResults that do not have at least one aliasing OpOperand.

          TODO: Support other relations such as "OpOperand is included in
          OpResult".
        }],
        /*retType=*/"BufferRelation",
        /*methodName=*/"bufferRelation",
        /*args=*/(ins "OpResult":$opResult,
                      "const AnalysisState &":$state),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          // Does not have to be implemented for ops without tensor OpResults
          // that have an aliasing OpOperand.
          llvm_unreachable("bufferRelation not implemented");
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Resolve all inplacability conflicts by inserting explicit
          `bufferization.alloc_tensor` ops. Examples of inplacability conflicts
          are read-after-write conflicts or writes into non-writable buffers.

          This method should rewrite the IR in such a way that for each tensor
          OpOperand t, buffer(t) can be directly used when during bufferization.
          The bufferization does no longer have to care about inplacability
          conflicts.

          This method can query analysis information from the given analysis
          state.
        }],
        /*retType=*/"LogicalResult",
        /*methodName=*/"resolveConflicts",
        /*args=*/(ins "RewriterBase &":$rewriter,
                      "const AnalysisState &":$state),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          auto bufferizableOp =
              cast<BufferizableOpInterface>($_op.getOperation());
          return bufferizableOp.resolveTensorOpOperandConflicts(
              rewriter, state);
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Bufferize this op, i.e., rewrite it into a memref-based equivalent.
          Buffers of tensor SSA values can be retrieved via `getBuffer`.
          Uses of tensor results of the existing tensor op can be replaced with
          `replaceOpWithBufferizedValues` or `replaceOpWithNewBufferizedOp`.
          These two functions automatically handle the tensor-to-memref type
          conversion.

          The implementation of this method must be consistent with the
          remaining methods, in particular `getAliasingOpOperand`. I.e., a
          tensor result `r` may only be replaced with:
          a) A buffer that aliases one of buffers in getAliasingOpOperand(r).
          b) Or: A newly allocated buffer.

          This method will never be called on ops that do not have at least one
          tensor operand/result.

          The return value of this method indicates whether there was an error
          while bufferizing this op (such as failing to create a new buffer
          allocation op). The bufferization driver immediately stops bufferizing
          the input IR and returns `failure` in that case. If this op is
          expected to survive bufferization, `success` should be returned
          (together with `allow-unknown-ops` enabled).
        }],
        /*retType=*/"LogicalResult",
        /*methodName=*/"bufferize",
        /*args=*/(ins "RewriterBase &":$rewriter,
                      "const BufferizationOptions &":$options),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          llvm_unreachable("bufferize not implemented");
          return failure();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return `true` if the given Value can be written to in-place. Value is
          either an OpResult of this operation or a BlockArgument of a block of
          this operation.

          Most OpResult buffers can be written to, but some ops such as
          ConstantOp may bufferize to non-writable (read-only) memory locations.
          Therefore, by default, this method returns `true` for OpResults. This
          method will never be called on OpResults that do not have a tensor
          type.

          Whether a BlockArgument can be written to or not depends on the
          operation. This method conservatively returns `false`. This method
          will never be called on BlockArguments that do not have a tensor type.
        }],
        /*retType=*/"bool",
        /*methodName=*/"isWritable",
        /*args=*/(ins "Value":$value,
                      "const AnalysisState &":$state),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return value.isa<OpResult>();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return `true` if the `uRead` and `uWrite` do not constitute a RaW
          conflict. If they are conflicting or if it is unknown whether they are
          conflicting, return `false`. This method will never be called with
          OpOperands that do not have a tensor type. At least one of the two
          given OpOperands belongs to this operation.

          This method can be implemented to specify custom RaW analysis rules.
          If this method returns `true` the given OpOperands are not considered
          to be conflicting and do not force out-of-place bufferization. (There
          may still be other conflicts that do.)
        }],
        /*retType=*/"bool",
        /*methodName=*/"isNotConflicting",
        /*args=*/(ins "OpOperand *":$uRead,
                      "OpOperand *":$uWrite,
                      "const AnalysisState &":$state),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return false;
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return `failure` if this op does not pass the analysis. This method
          is run during One-Shot Bufferize (after all post-analysis steps). If
          the op does not pass the analysis, bufferization is aborted.

          This method can be used to check expected invariants and limitations
          of the current bufferization implementation.
        }],
        /*retType=*/"LogicalResult",
        /*methodName=*/"verifyAnalysis",
        /*args=*/(ins "const AnalysisState &":$state),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return success();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the bufferized type of the given tensor block argument. The
          block argument is guaranteed to belong to a block of this op.
        }],
        /*retType=*/"BaseMemRefType",
        /*methodName=*/"getBufferType",
        /*args=*/(ins "BlockArgument":$bbArg,
                      "const BufferizationOptions &":$options),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          assert(bbArg.getOwner()->getParentOp() == $_op &&
                 "bbArg must belong to this op");
          auto tensorType = bbArg.getType().cast<TensorType>();
          return bufferization::getMemRefType(tensorType, options);
        }]
      >,
  ];

  let extraClassDeclaration = [{
    /// Resolve out-of-place tensor OpOperands with explicit allocations in the
    /// form of `bufferization.alloc_tensor` ops.
    LogicalResult resolveTensorOpOperandConflicts(
        RewriterBase &rewriter, const AnalysisState &state);

    /// Return `true` if the given OpOperand creates an alias but does neither
    /// read nor write. This implies that `bufferizesToMemoryRead` and
    /// `bufferizesToMemoryWrite` must return `false`. This method will never
    /// be called on OpOperands that do not have a tensor type.
    ///
    /// Examples of such ops are `tensor.extract_slice` and `tensor.cast`.
    bool bufferizesToAliasOnly(OpOperand &opOperand,
                               const AnalysisState &state) {
      auto bufferizableOp =
          cast<BufferizableOpInterface>(getOperation());
      return !bufferizableOp.bufferizesToMemoryRead(opOperand, state)
          && !bufferizableOp.bufferizesToMemoryWrite(opOperand, state)
          && !bufferizableOp.getAliasingOpResult(opOperand, state).empty();
    }

    // TODO: This attribute is deprecated. Use `bufferization.writable` or add
    // a new attribute in a different dialect.
    /// Attribute name used to mark region arguments that can be bufferized
    /// in-place during one-shot bufferization.
    constexpr const static ::llvm::StringLiteral
        kInplaceableAttrName = "linalg.inplaceable";
  }];
}

#endif  // BUFFERIZABLE_OP_INTERFACE
