//===- LinalgStructuredInterface.td- Linalg StructuredIfce -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the structured interface for Linalg ops.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_IR_STRUCTURED_OPS_INTERFACE
#define LINALG_IR_STRUCTURED_OPS_INTERFACE

include "mlir/Dialect/Linalg/IR/LinalgBase.td"

// The linalg 'LinalgStructuredInterface' provides access to the 'LinalgOp'
// interface.
def LinalgStructuredInterface : OpInterface<"LinalgOp"> {
  let methods = [
    //===------------------------------------------------------------------===//
    // Loop types handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      "Return the number of parallel loops within the current operation.",
      "unsigned", "getNumParallelLoops"
    >,
    InterfaceMethod<
      "Return the number of reduction loops within the current operation.",
      "unsigned", "getNumReductionLoops"
    >,
    InterfaceMethod<
      "Return the number of window loops within the current operation.",
      "unsigned", "getNumWindowLoops"
    >,
    InterfaceMethod<
      "Return the number of loops within the current operation.",
      "unsigned", "getNumLoops">,

    InterfaceMethod<
      [{Returns true if the current operation has only one loop and it's a
        reduction loop}],
      "bool", "hasSingleReductionLoop">,

    //===------------------------------------------------------------------===//
    // Input arguments handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      "Return the number of inputs from the current operation.",
      "unsigned", "getNumInputs"
    >,
    InterfaceMethod<"Return the input view at the given index.",
      "Value", "getInput", (ins "unsigned":$i)
    >,
    InterfaceMethod<[{
        Return the index of the given input value `v`, or `None` if the value is
        not an input.
      }],
      "llvm::Optional<unsigned>", "getIndexOfInput", (ins "Value":$v)
    >,
    InterfaceMethod<
      "Return the input operands from the current operation.",
      "Operation::operand_range", "getInputs"
    >,
    InterfaceMethod<[{
        Return the `i`-th input shaped type, irrespective of buffer or tensor
        type.
      }], "ShapedType", "getInputShapedType", (ins "unsigned":$i)>,
    InterfaceMethod<[{
        Return the subset of input operands that are of ranked tensor type.
      }], "SmallVector<RankedTensorType, 4>", "getInputTensorTypes">,

    //===------------------------------------------------------------------===//
    // Output arguments handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      "Return the number of outputs from the current operation.",
      "unsigned", "getNumOutputs"
    >,
    InterfaceMethod<"Return the output buffer at the given index.",
      "Value", "getOutputBuffer", (ins "unsigned":$i)
    >,
    InterfaceMethod<[{
        Return the index of the given buffer value, or `None` if the value is
        not part of the output buffers.
      }],
      "llvm::Optional<unsigned>", "getIndexOfOutputBuffer", (ins "Value":$view)
    >,
    InterfaceMethod<[{
        Return the type of the output buffer at the given index.
      }], "MemRefType", "getOutputBufferType", (ins "unsigned":$i)>,
    InterfaceMethod<[{
        Return the `i`-th output shaped type, irrespective of buffer or tensor
        type.
      }], "ShapedType", "getOutputShapedType", (ins "unsigned":$i)>,
    InterfaceMethod<[{
        Return the results that are of ranked tensor type.
      }], "SmallVector<RankedTensorType, 4>", "getOutputTensorTypes">,
    InterfaceMethod<
      "Return the output buffers (operands) from the current operation.",
      "Operation::operand_range", "getOutputBuffers"
    >,

    //===------------------------------------------------------------------===//
    // Input and Output arguments handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      "Return one single buffer at position `$i`.",
      "Value", "getBuffer", (ins "unsigned":$i)
    >,
    InterfaceMethod<
      "Return the number of inputs and outputs, irrespective of their buffer "
      "or tensor type.",
      "unsigned", "getNumInputsAndOutputs"
    >,
    InterfaceMethod<
      "Return the number of inputs, irrespective of their buffer or tensor "
      "type, and output buffers",
      "unsigned", "getNumInputsAndOutputBuffers"
    >,
    InterfaceMethod<
      "Return the range over inputs (irrespective of type) and output buffers.",
      "Operation::operand_range", "getInputsAndOutputBuffers"
    >,
    InterfaceMethod<
      "Return the shaped types for all the inputs and outputs",
      "SmallVector<ShapedType, 4>", "getInputOutputShapedTypes"
    >,

    //===------------------------------------------------------------------===//
    // Other interface methods.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      "Return the reference iterators for this named op (if any are "
      "specified). These reference iterators are used to specify the default "
      "behavior of the op. Typically this would be a static method but in "
      "order to allow rank-polymorphic ops, this needs to be per object "
      "instance. Named ops must define referenceIterators, even if empty for "
      "the 0-D case. Generic ops on the other hand have a None "
      "`referenceIterators`",
      "llvm::Optional<SmallVector<StringRef, 8>>", "referenceIterators"
    >,
    InterfaceMethod<
      "Return the reference indexing maps for this named op (if any are "
      "specified). Typically this would be a static method but in order to "
      "allow rank-polymorphic ops, this needs to be per object instance. Named "
      "ops must define referenceIterators, even if empty for the 0-D case. "
      "Generic ops on the other hand have a None `referenceIndexingMaps`",
      "llvm::Optional<SmallVector<AffineMap, 8>>", "referenceIndexingMaps"
    >,
    InterfaceMethod<
      "Return the iterator types attribute within the current operation.",
      "ArrayAttr", "iterator_types"
    >,
    InterfaceMethod<
      "Return the indexing maps attribute within the current operation.",
      "ArrayAttr", "indexing_maps"
    >,
    InterfaceMethod<
      "Return the indexing maps within the current operation.",
      "SmallVector<AffineMap, 4>", "getIndexingMaps"
    >,
    InterfaceMethod<"Return the input or output indexing map at index `i`.",
      "AffineMap", "getIndexingMap", (ins "unsigned":$i)
    >,
    InterfaceMethod<"Return the input indexing map at index `i`.",
      "AffineMap", "getInputIndexingMap", (ins "unsigned":$i)
    >,
    InterfaceMethod<"Return the output indexing map at index `i`.",
      "AffineMap", "getOutputIndexingMap", (ins "unsigned":$i)
    >,
    InterfaceMethod<[{
        Return whether the op has only MemRef input and outputs.
      }], "bool", "hasBufferSemantics">,
    InterfaceMethod<[{
        Return whether the op has only RankedTensor input and outputs.
      }], "bool", "hasTensorSemantics">,

    //===------------------------------------------------------------------===//
    // Other static interface methods.
    //===------------------------------------------------------------------===//
    StaticInterfaceMethod<[{
        Create an operation of the current type with the given location,
        operands, and attributes.
      }],
      "Operation *", "create",
      (ins "OpBuilder &":$builder, "Location":$loc,
           "ValueRange":$operands,
           "ArrayRef<NamedAttribute>":$attributes), [{
        return builder.create<ConcreteOp>(loc, ArrayRef<Type>{}, operands,
                                          attributes);
      }]
    >,
    InterfaceMethod<[{
        Clone the current operation with the given location and operands. This
        is used to abstract away the optional underlying region creation.
      }],
      "Operation *", "clone",
      (ins "OpBuilder &":$b, "Location":$loc, "ValueRange":$operands), [{
        BlockAndValueMapping map;
        unsigned numRegions = op.getOperation()->getNumRegions();
        Operation *res = create(b, loc, operands, op.getAttrs());
        assert(res->getNumRegions() == numRegions && "inconsistent # regions");
        for (unsigned ridx = 0; ridx < numRegions; ++ridx)
          op.getOperation()->getRegion(ridx).cloneInto(
            &res->getRegion(ridx), map);
        return res;
      }]
    >
  ];
}

#endif // LINALG_IR_STRUCTURED_OPS_INTERFACE
