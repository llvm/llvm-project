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
  let cppNamespace = "::mlir::linalg";
  let methods = [
    //===------------------------------------------------------------------===//
    // Loop types handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the number of parallel loops within the current operation.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumParallelLoops",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getNumIterators(getParallelIteratorTypeName(),
                               $_op.iterator_types());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of reduction loops within the current operation.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumReductionLoops",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getNumIterators(getReductionIteratorTypeName(),
                               $_op.iterator_types());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of window loops within the current operation.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumWindowLoops",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getNumIterators(getWindowIteratorTypeName(),
                               $_op.iterator_types());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the total number of loops within the current operation.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumLoops",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getNumIterators($_op.iterator_types());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the current operation has only one loop and it's a
        reduction loop.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasSingleReductionLoop",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto iters = $_op.iterator_types();
        return iters.size() == 1 &&
               getNumIterators(getReductionIteratorTypeName(), iters) == 1;
      }]>,
    //===------------------------------------------------------------------===//
    // Num input/output arguments handling.
    //===------------------------------------------------------------------===//
    // These special methods must be defined by each op that wants to implement
    // the LinalgStructuredInterface. For now, this is either:
    // - inherited statically by using the NInputs<unsigned> or
    //   NOutputs<unsigned> traits.
    // - derived from args_in/args_out attributes (for linalg.generic and
    //   linalg.indexed_generic ops).
    InterfaceMethod<
      /*desc=*/[{
        Return the number of inputs from the current operation.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumInputs"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of outputs from the current operation.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumOutputs"
    >,
    //===------------------------------------------------------------------===//
    // Input arguments handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the `i`-th input value.
        The `i^th` input argument is always the `i^th` operand regardless of
        whether we have tensors or buffers.
      }],
      /*retTy=*/"Value",
      /*methodName=*/"getInput",
      /*args=*/(ins "unsigned":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i < $_op.getNumInputs());
        return this->getOperation()->getOperand(i);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the index of the given input value `v`, or `None` if the value is
        not an input.
      }],
      /*retTy=*/"llvm::Optional<unsigned>",
      /*methodName=*/"getIndexOfInput",
      /*args=*/(ins "Value":$value),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto it = llvm::find(getInputs(), value);
        if (it != getInputs().end())
          return it - getInputs().begin();
        return llvm::None;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `i`-th input shaped type, irrespective of buffer or tensor
        type.
      }],
      /*retTy=*/"ShapedType",
      /*methodName=*/"getInputShapedType",
      /*args=*/(ins "unsigned":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getInput(i).getType().template cast<ShapedType>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the input operands from the current operation.
      }],
      /*retTy=*/"Operation::operand_range",
      /*methodName=*/"getInputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto range = this->getOperation()->getOperands();
        return {range.begin(), range.begin() + $_op.getNumInputs()};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of input operands that are of ranked tensor type.
      }],
      /*retTy=*/"SmallVector<RankedTensorType, 4>",
      /*methodName=*/"getInputTensorTypes" ,
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<RankedTensorType, 4> res;
        for (Type type : getInputs().getTypes())
          if (auto t = type.template dyn_cast<RankedTensorType>())
            res.push_back(t);
        return res;
      }]
    >,

    //===------------------------------------------------------------------===//
    // Output arguments handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the output buffer at the given index, asserts that this is a
        buffer operand and not a tensor result.
        The `i^th` output argument is an operand (resp. a return value) iff it
        is a value of buffer type (resp. a return value of tensor type).
      }],
      /*retTy=*/"Value",
      /*methodName=*/"getOutputBuffer",
      /*args=*/(ins "unsigned":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // Output buffers are passed as output buffer operands (side-effecting).
        // Output tensors are results.
        // The union of the 2 are all the outputs and we want to ensure i does
        // not overflow the buffer operands.
        assert(i + this->getOperation()->getNumResults() < $_op.getNumOutputs()
               && "overflowing output buffer index");
        return this->getOperation()->getOperand($_op.getNumInputs() + i);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the index of the given buffer value, or `None` if the value is
        not part of the output buffers.
      }],
      /*retTy=*/"llvm::Optional<unsigned>",
      /*methodName=*/"getIndexOfOutputBuffer",
      /*args=*/(ins "Value":$value),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto it = llvm::find(getOutputBuffers(), value);
        if (it != getOutputBuffers().end())
          return it - getOutputBuffers().begin();
        return llvm::None;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the type of the output buffer at the given index.
      }],
      /*retTy=*/"MemRefType",
      /*methodName=*/"getOutputBufferType",
      /*args=*/(ins "unsigned":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getOutputBuffer(i).getType().template cast<MemRefType>();
      }]>,
    InterfaceMethod<
      /*desc=*/[{
        Return the `i`-th output shaped type, irrespective of buffer or tensor
        type.
      }],
      /*retTy=*/"ShapedType",
      /*methodName=*/"getOutputShapedType",
      /*args=*/(ins "unsigned":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getShapedType(i + $_op.getNumInputs());
      }]>,
    InterfaceMethod<
      /*desc=*/[{
        Return the results that are of ranked tensor type.
      }],
      /*retTy=*/"SmallVector<RankedTensorType, 4>",
      /*methodName=*/"getOutputTensorTypes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<RankedTensorType, 4> res;
        for (Type type : this->getOperation()->getResults().getTypes())
          res.push_back(type.template cast<RankedTensorType>());
        return res;
      }]>,
    InterfaceMethod<
      /*desc=*/[{
        Return the output buffers (operands) from the current operation.
      }],
      /*retTy=*/"Operation::operand_range",
      /*methodName=*/"getOutputBuffers",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto range = this->getOperation()->getOperands();
        return {range.begin() + $_op.getNumInputs(),
                range.begin() + getNumInputsAndOutputBuffers()};
      }]
    >,

    //===------------------------------------------------------------------===//
    // Input and Output arguments handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return one single buffer at position `$i`.
      }],
      /*retTy=*/"Value",
      /*methodName=*/"getBuffer",
      /*args=*/(ins "unsigned":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i < getNumInputsAndOutputBuffers() && "overflowing buffers index");
        return this->getOperation()->getOperand(i);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of inputs and outputs, irrespective of their buffer or
        tensor type.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumInputsAndOutputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getNumInputs() + $_op.getNumOutputs();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of inputs, irrespective of their buffer or tensor type
        and output buffers
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumInputsAndOutputBuffers",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getNumInputs() + $_op.getNumOutputs() -
          this->getOperation()->getNumResults();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the range over inputs (irrespective of type) and output buffers.
      }],
      /*retTy=*/"Operation::operand_range",
      /*methodName=*/"getInputsAndOutputBuffers",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto range = this->getOperation()->getOperands();
        return {range.begin(), range.begin() + getNumInputsAndOutputBuffers()};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `i`-th shaped type, there are 3 cases:
          1. if `i < $_op.getNumInputs()` then return `getInputShapedType(i)`;
             otherwise
          2. if `i < getNumInputsAndOutputBuffers()` then return the
             `getOutputBufferType(i - $_op.getNumInputs())`; otherwise
          3. return the `i - getNumInputsAndOutputBuffers()` result type.
      }],
      /*retTy=*/"ShapedType",
      /*methodName=*/"getShapedType",
      /*args=*/(ins "unsigned":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if (i < $_op.getNumInputs())
          return getInputShapedType(i);
        if (i < getNumInputsAndOutputBuffers())
          return getOutputBufferType(i - $_op.getNumInputs());
        return getOutputTensorTypes()[i - getNumInputsAndOutputBuffers()];
      }]>,
    InterfaceMethod<
      /*desc=*/[{
        Return the shaped types for all the inputs and outputs
      }],
      /*retTy=*/"SmallVector<ShapedType, 4>",
      /*methodName=*/"getInputOutputShapedTypes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<Type, 4> inputOutputTypes(
            this->getOperation()->operand_type_begin(),
            this->getOperation()->operand_type_end());
        inputOutputTypes.append(this->getOperation()->result_type_begin(),
                                this->getOperation()->result_type_end());
        return llvm::to_vector<4>(
            llvm::map_range(inputOutputTypes, [](Type type) -> ShapedType {
              return type.cast<ShapedType>();
            }));
      }]
    >,

    //===------------------------------------------------------------------===//
    // Other interface methods.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the iterator types attribute within the current operation.
      }],
      /*retTy=*/"ArrayAttr",
      /*methodName=*/"iterator_types",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.iterator_types();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing maps attribute within the current operation.
      }],
      /*retTy=*/"ArrayAttr",
      /*methodName=*/"indexing_maps"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing maps within the current operation.
      }],
      /*retTy=*/"SmallVector<AffineMap, 4>",
      /*methodName=*/"getIndexingMaps",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::to_vector<4>(
            llvm::map_range($_op.indexing_maps(),
                            [](Attribute attr) -> AffineMap {
              return attr.cast<AffineMapAttr>().getValue();
            }));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the input or output indexing map at index `i`.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getIndexingMap",
      /*args=*/(ins "unsigned":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i < getNumInputsAndOutputs());
        return $_op.indexing_maps()
            .getValue()[i]
            .template cast<AffineMapAttr>()
            .getValue();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the input indexing map at index `i`.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getInputIndexingMap",
      /*args=*/(ins "unsigned":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i < $_op.getNumInputs());
        return $_op.indexing_maps()
            .getValue()[i]
            .template cast<AffineMapAttr>()
            .getValue();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the output indexing map at index `i`.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getOutputIndexingMap",
      /*args=*/(ins "unsigned":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i < $_op.getNumOutputs());
        return $_op.indexing_maps()
            .getValue()[i + $_op.getNumInputs()]
            .template cast<AffineMapAttr>()
            .getValue();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return whether the op has only MemRef input and outputs.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasBufferSemantics",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return this->getOperation()->getNumResults() == 0 &&
          llvm::all_of(getInputs(),
                       [](Value v) { return v.getType().isa<MemRefType>(); });
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return whether the op has only RankedTensor input and outputs.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasTensorSemantics",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto isTensorType = [](Value v) {
          return v.getType().isa<RankedTensorType>();
        };
        return llvm::all_of(getInputs(), isTensorType) &&
               llvm::all_of(this->getOperation()->getResults(), isTensorType);
      }]
    >,

    //===------------------------------------------------------------------===//
    // Other static interface methods.
    //===------------------------------------------------------------------===//
    StaticInterfaceMethod<
      /*desc=*/[{
        Create an operation of the current type with the given location,
        operands, and attributes.
      }],
      /*retTy=*/"Operation *",
      /*methodName=*/"create",
      (ins "OpBuilder &":$builder, "Location":$loc,
           "ValueRange":$operands,
           "ArrayRef<NamedAttribute>":$attributes), [{
        return builder.create<ConcreteOp>(loc, ArrayRef<Type>{}, operands,
                                          attributes);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Clone the current operation with the given location and operands. This
        is used to abstract away the optional underlying region creation.
      }],
      /*retTy=*/"Operation *",
      /*methodName=*/"clone",
      (ins "OpBuilder &":$b, "Location":$loc, "ValueRange":$operands), [{
        BlockAndValueMapping map;
        unsigned numRegions = $_op.getOperation()->getNumRegions();
        Operation *res = create(b, loc, operands, $_op.getAttrs());
        assert(res->getNumRegions() == numRegions && "inconsistent # regions");
        for (unsigned ridx = 0; ridx < numRegions; ++ridx)
          $_op.getOperation()->getRegion(ridx).cloneInto(
            &res->getRegion(ridx), map);
        return res;
      }]
    >
  ];
}

#endif // LINALG_IR_STRUCTURED_OPS_INTERFACE
