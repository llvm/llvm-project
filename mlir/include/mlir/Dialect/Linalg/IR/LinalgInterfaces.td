//===- LinalgInterfaces.td - Linalg Interfaces Declaration -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the structured interface sfor Linalg ops.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_IR_LINALGINTERFACES
#define LINALG_IR_LINALGINTERFACES

include "mlir/IR/OpBase.td"

// The 'LinalgContractionOpInterface' provides access to the
// 'ContractionOpInterface'.
def LinalgContractionOpInterface : OpInterface<"ContractionOpInterface"> {
  let description = [{
   A Linalg contraction is defined in general terms:
     1. Has 2 input and 1 output shapes.
     2. Has at least one reduction dimension.
     3. Has only projected permutation indexing maps.
     4. its body computes `u5(u1(c) + u2(u3(a) * u4(b)))` on some field
     (AddOpType, MulOpType), where u1, u2, u3, u4 and u5 represent scalar unary
     operations that may change the type (e.g. for mixed-precision).
   As a consequence, when vectorization of such an op occurs, the only special
   behavior is that the (unique) MulOpType is vectorized into a
   `vector.contract`. All other ops are handled in a generic fashion.
   In the future, we may wish to allow more input arguments and elementwise and
   constant operations that do not involve the reduction dimension(s).
  }];
  let cppNamespace = "::mlir::linalg";
  let verify = [{ return detail::verifyContractionInterface($_op); }];
  let methods = [
    InterfaceMethod<
    /*desc=*/"Returns the left-hand side operand.",
    /*retTy=*/"Value",
    /*methodName=*/"lhs",
    /*args=*/(ins),
    /*methodBody=*/[{
      return $_op.getOperation()->getOperand(0);
    }]>,
    InterfaceMethod<
    /*desc=*/"Returns the right-hand side operand.",
    /*retTy=*/"Value",
    /*methodName=*/"rhs",
    /*args=*/(ins),
    /*methodBody=*/[{
      return $_op.getOperation()->getOperand(1);
    }]>,
    InterfaceMethod<
    /*desc=*/[{
      Returns whether the given op has indexing maps that correspond to a
      row-major matmul operation.
    }],
    /*retTy=*/"bool",
    /*methodName=*/"isRowMajorMatmul",
    /*args=*/(ins),
    /*methodBody=*/[{
        return mlir::isRowMajorMatmul($_op.indexing_maps());
    }]>,
    InterfaceMethod<
    /*desc=*/[{
      Returns whether the given op has indexing maps that correspond to a
      column-major matmul operation.
    }],
    /*retTy=*/"bool",
    /*methodName=*/"isColumnMajorMatmul",
    /*args=*/(ins),
    /*methodBody=*/[{
        return mlir::isColumnMajorMatmul($_op.indexing_maps());
    }]>,
    InterfaceMethod<
    /*desc=*/[{
      Returns whether the given op has indexing maps that correspond to a
      row-major batch matmul operation.
    }],
    /*retTy=*/"bool",
    /*methodName=*/"isRowMajorBatchMatmul",
    /*args=*/(ins),
    /*methodBody=*/[{
        return mlir::isRowMajorBatchMatmul($_op.indexing_maps());
    }]>,
  ];
}

// The 'LinalgStructuredInterface' provides access to the 'LinalgOp' interface.
def LinalgStructuredInterface : OpInterface<"LinalgOp"> {
  let cppNamespace = "::mlir::linalg";
  let methods = [
    //===------------------------------------------------------------------===//
    // Loop types handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the number of induction variables in the basic block. This should
        always be 0 for index-free linalg ops. For IndexedGeneric, this must be
        equal to numLoops
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumPayloadInductionVariables",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/""
    >,
    //===------------------------------------------------------------------===//
    // Loop types handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the number of parallel loops.
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
        Return the dims that are parallel loops.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getParallelDims",
      /*args=*/(ins "SmallVectorImpl<AffineExpr> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getDimsOfType($_op, getParallelIteratorTypeName(), res);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of reduction loops.
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
        Return the dims that are reduction loops.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getReductionDims",
      /*args=*/(ins "SmallVectorImpl<AffineExpr> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getDimsOfType($_op, getReductionIteratorTypeName(), res);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of window loops.
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
        Return the dims that are window loops.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getWindowDims",
      /*args=*/(ins "SmallVectorImpl<AffineExpr> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getDimsOfType($_op.getOperation(), getWindowIteratorTypeName(), res);
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
    // `inputs` must be defined by each op that wants to implement the
    // LinalgStructuredInterface.
    InterfaceMethod<
      /*desc=*/[{
        Return the input shape operands.
      }],
      /*retTy=*/"ValueRange",
      /*methodName=*/"inputs",
      /*args=*/(ins)
    >,
    // These special methods rely on `inputs` and `outputs` being defined by
    // each op that wants to implement the LinalgStructuredInterface.
    InterfaceMethod<
      /*desc=*/[{
        Return the number of inputs.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumInputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.inputs().size();
      }]
    >,
    // `outputs` must be defined by each op that wants to implement the
    // LinalgStructuredInterface.
    InterfaceMethod<
      /*desc=*/[{
        Return the output shape operands.
      }],
      /*retTy=*/"ValueRange",
      /*methodName=*/"outputs",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of outputs.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumOutputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.outputs().size();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of inputs and outputs.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumInputsAndOutputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getNumInputs() + getNumOutputs();
      }]
    >,
    //===------------------------------------------------------------------===//
    // Input operands handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the input operands.
      }],
      /*retTy=*/"OpOperandVector",
      /*methodName=*/"getInputOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        OpOperandVector result;
        result.reserve(getNumInputs());
        llvm::transform(
          this->getOperation()->getOpOperands().take_front(getNumInputs()),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `i`-th input operand.
      }],
      /*retTy=*/"OpOperand*",
      /*methodName=*/"getInputOperand",
      /*args=*/(ins "int64_t":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i < getNumInputs());
        return getInputOperands()[i];
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of input operands that are of buffer type.
      }],
      /*retTy=*/"OpOperandVector",
      /*methodName=*/"getInputBufferOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        OpOperandVector result;
        result.reserve(getNumInputs());
        llvm::copy_if(getInputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<MemRefType>();
          });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of input operands that are of tensor type.
      }],
      /*retTy=*/"OpOperandVector",
      /*methodName=*/"getInputTensorOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        OpOperandVector result;
        result.reserve(getNumInputs());
        llvm::copy_if(getInputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<RankedTensorType>();
          });
        return result;
      }]
    >,
    //===------------------------------------------------------------------===//
    // Output operands handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the output operands.
      }],
      /*retTy=*/"OpOperandVector",
      /*methodName=*/"getOutputOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        OpOperandVector result;
        result.reserve(getNumOutputs());
        llvm::transform(
          this->getOperation()->getOpOperands()
            .drop_front(getNumInputs())
            .take_front(getNumOutputs()),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `i`-th output operand.
      }],
      /*retTy=*/"OpOperand*",
      /*methodName=*/"getOutputOperand",
      /*args=*/(ins "int64_t":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i < getNumOutputs());
        return getOutputOperands()[i];
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of output operands that are of buffer type.
      }],
      /*retTy=*/"OpOperandVector",
      /*methodName=*/"getOutputBufferOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        OpOperandVector result;
        result.reserve(getNumOutputs());
        llvm::copy_if(getOutputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<MemRefType>();
          });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of output operands that are of tensor type.
      }],
      /*retTy=*/"OpOperandVector",
      /*methodName=*/"getOutputTensorOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        OpOperandVector result;
        result.reserve(getNumOutputs());
        llvm::copy_if(getOutputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<RankedTensorType>();
          });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the types of the subset of output operands that are of buffer type.
      }],
      /*retTy=*/"SmallVector<MemRefType>",
      /*methodName=*/"getOutputBufferTypes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<MemRefType> result;
        result.reserve(getNumOutputs());
        llvm::transform(getOutputBufferOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperands) {
            return opOperands->get().getType().cast<MemRefType>();
          });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the types of the subset of output operands that are of tensor type.
      }],
      /*retTy=*/"SmallVector<RankedTensorType>",
      /*methodName=*/"getOutputTensorTypes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<RankedTensorType> result;
        result.reserve(getNumOutputs());
        llvm::transform(getOutputTensorOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperands) {
            return opOperands->get().getType().cast<RankedTensorType>();
          });
        return result;
      }]
    >,
    //===------------------------------------------------------------------===//
    // Input and Output arguments handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the range over input and output operands.
      }],
      /*retTy=*/"OpOperandVector",
      /*methodName=*/"getInputAndOutputOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        OpOperandVector result;
        result.reserve(getNumInputsAndOutputs());
        llvm::transform(
          this->getOperation()->getOpOperands()
            .take_front(getNumInputsAndOutputs()),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if the payload uses the value loaded from `opOperand`. This
        is useful to avoid loading from "write-only" memory that may be
        uninitialized, as well as properly cloning "read-write" operands.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"payloadUsesValueFromOperand",
      /*args=*/(ins "OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        unsigned bbArgNumber =
          $_op.getNumPayloadInductionVariables() + opOperand->getOperandNumber();
        // Safeguard against the named linalg ops that are manually defined and
        // that only support buffer semantics: we should not be there.
        // Such ops have an empty regionBuilder and are not constructed with a
        // region for now. In the future they are slated to disappear.
        assert(this->getOperation()->getNumRegions() == 1 && "unexpected "
               "missing region (calling `payloadUsesValueFromOperand` on "
               "manually defined named Linalg op?)");
        Block &block = this->getOperation()->getRegion(0).front();
        // Init tensors have uses.
        return !block.getArgument(bbArgNumber).use_empty();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if `opOperand` is an input tensor.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isInputTensor",
      /*args=*/(ins "OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if (!opOperand->get().getType().template isa<RankedTensorType>())
          return false;
        if (opOperand->getOperandNumber() < $_op.getNumInputs())
          return true;
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if `opOperand` is an output tensor.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isOutputTensor",
      /*args=*/(ins "OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if (!opOperand->get().getType().template isa<RankedTensorType>())
          return false;
        if (opOperand->getOperandNumber() >= $_op.getNumInputs())
          return true;
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if `opOperand` is an init tensor. This is true when it is
        an output tensor operand whose value is used in the payload region.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isInitTensor",
      /*args=*/(ins "OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if (!$_op.isOutputTensor(opOperand))
          return false;
        return payloadUsesValueFromOperand(opOperand);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `opOperand` rank or zero for scalars.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getRank",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        if (auto shapedType =
              opOperand->get().getType().template dyn_cast<ShapedType>())
          return shapedType.getRank();
        return 0;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `opOperand` shape or an empty vector for scalars.
      }],
      /*retTy=*/"ArrayRef<int64_t>",
      /*methodName=*/"getShape",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        if (auto shapedType =
              opOperand->get().getType().template dyn_cast<ShapedType>())
          return shapedType.getShape();
        return {};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the input or output indexing map for `opOperand`.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getTiedIndexingMap",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        return getIndexingMaps()[opOperand->getOperandNumber()];
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the result tied to `opOperand`.
      }],
      /*retTy=*/"OpResult",
      /*methodName=*/"getTiedOpResult",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        int64_t resultIndex = opOperand->getOperandNumber() - getNumInputs();
        assert(resultIndex >= 0 &&
               resultIndex < this->getOperation()->getNumResults() );
        return this->getOperation()->getResult(resultIndex);
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
        Return true if the indexing map is depending on the current op instance.
        This means that the indexing map is dynamically synthesized by using the
        op instance's concrete attributes, instead of being static for all
        instances of the same op kind.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasDynamicIndexingMaps",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return false; }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Verify all attributes used by indexing maps are valid.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"verifyIndexingMapRequiredAttributes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return success(); }]
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
      /*retTy=*/"SmallVector<AffineMap>",
      /*methodName=*/"getIndexingMaps",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto range = $_op.indexing_maps()
          .template getAsValueRange<AffineMapAttr>();
        return {range.begin(), range.end()};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if any of the operands has a dynamic shape.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasDynamicShape",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::any_of(getStaticShape(), ShapedType::isDynamic);
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
          llvm::all_of(getInputAndOutputOperands(),
            [](OpOperand *opOperand) {
              return opOperand->get().getType().template isa<MemRefType>();
            });
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
        return llvm::all_of(getInputAndOutputOperands(),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<RankedTensorType>();
          });
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the name registered for this op when lowering to an external
        library call.
      }],
      /*retTy=*/"std::string",
      /*methodName=*/"getLibraryCallName",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getLibraryCallName();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
         Return whether the op accesses the iteration indices.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasIndexSemantics",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/""
    >,
    //===------------------------------------------------------------------===//
    // Linalg generalization hooks.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Hook to provide a custom AffineMap used to compute all the operand
        subshapes given loop bounds. This is used to answer the question: "given
        an iteration space over the codomain, what are the subshapes of the
        operands involved in the computation".
        The default behavior is to just concatenate all the indexing maps.
        A custom AffineMap allows providing a map that can be used to
        compute subshapes even in cases where the concatenation of indexing maps
        (i.e. the data traversal order) is not a simple permutation of the loop
        traversal order. It is then possible to define ops with skewed data
        traversal order for which we can still easily compute hyperrectangular
        loop bounds and subviews.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getLoopsToShapesMap",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto r = $_op.indexing_maps().template getAsRange<AffineMapAttr>();
        auto maps = llvm::to_vector<8>(
            llvm::map_range(r, [](AffineMapAttr a) { return a.getValue(); }));
        return concatAffineMaps(maps);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Hook to provide a custom AffineMap used to construct the
        hyperrectangular loop iteration space given all the operand subshapes.
        This is used to answer the question:
        "Given a list of operand ranges, what is the subportion of the iteration
        space involved in the computation".
        This is the inverse problem of `getLoopsToShapesMap`.
        Return the empty AffineMap when such an AffineMap cannot be constructed.
        The default behavior is based on a very simple inference procedure that
        only works with permutation affine maps.
        A more advanced Tensor-Comprehension like inference is possible but has
        proven to be ambiguous in unfavorable case.
        A safer and more robust alternative is to allow each each op to define
        its own AffineMap.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getShapesToLoopsMap",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return inversePermutation(getLoopsToShapesMap());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the range of position in the result of the affine map
        computed by getLoopsToShapesMap() which correspond to the
        AffineExprs used to access the outputs of the operation.
      }],
      /*retTy=*/"std::pair<int64_t, int64_t>",
      /*methodName=*/"getResultsPositionInLoopsToShapeMap",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        int64_t inputRankSum = 0;
        int64_t outputRankSum = 0;
        for(OpOperand *input : getInputOperands())
          inputRankSum += getRank(input);
        for(OpOperand *output : getOutputOperands())
          outputRankSum += getRank(output);
        return {inputRankSum, inputRankSum + outputRankSum};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Like `getShape`, but only returns statically-known information, without
        generating any new IR. For each shape dimension, returns >=0 if that
        dimension is statically known, or ShapeType::kDynamicSize otherwise.
      }],
      /*retTy=*/"SmallVector<int64_t>",
      /*methodName=*/"getStaticShape",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<int64_t> res;
        for (OpOperand *opOperand : getInputAndOutputOperands())
          llvm::append_range(res, getShape(opOperand));
        return res;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the statically-known loop ranges. Composes
        `getShapesToLoopsMap()` with the result of `getStaticShape`.
        Returns None if `getShapesToLoopsMap()` fails. Returns
        ShapeType::kDynamicSize for non-statically-known loop ranges.
      }],
      /*retTy=*/"Optional<SmallVector<int64_t, 4>>",
      /*methodName=*/"getStaticLoopRanges",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<int64_t> viewSizes = getStaticShape();
        AffineMap invertedMap = getShapesToLoopsMap();
        if (!invertedMap)
          return {};
        return invertedMap.compose(viewSizes);
      }]
    >,
    //===------------------------------------------------------------------===//
    // Other static interface methods.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Clone the current operation with the given location and operands. This
        is used to abstract away the optional underlying region creation. This
        does not change the balance between input, output_buffer and
        init_tensors operands.
      }],
      /*retTy=*/"Operation *",
      /*methodName=*/"clone",
      (ins "OpBuilder &":$b, "Location":$loc, "TypeRange":$resultTypes,
           "ValueRange":$operands),
      [{
        BlockAndValueMapping bvm;
        OperationState state(
          loc, ConcreteOp::getOperationName(), operands, resultTypes,
          $_op->getAttrs());
        for (Region &r : $_op->getRegions())
          r.cloneInto(state.addRegion(), bvm);
        return b.createOperation(state);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Clone the current operation with the given location, operands
        and BlockAndValueMapping. This is used to abstract away the
        optional underlying region creation. This does not change the
        balance between input, output_buffer and init_tensors
        operands.
      }],
      /*retTy=*/"Operation *",
      /*methodName=*/"cloneWithMapper",
      (ins "OpBuilder &":$b, "Location":$loc, "TypeRange":$resultTypes,
           "ValueRange":$operands, "BlockAndValueMapping &":$bvm),
      [{
        OperationState state(
          loc, ConcreteOp::getOperationName(), operands, resultTypes,
          $_op->getAttrs());
        for (Region &r : $_op->getRegions())
          r.cloneInto(state.addRegion(), bvm);
        return b.createOperation(state);
      }]
    >,
    StaticInterfaceMethod<
      /*desc=*/[{
        Returns the region builder for constructing the body for linalg.generic.
        Returns a null function if this named op does not define a region
        builder.
      }],
      /*retTy=*/"std::function<void(ImplicitLocOpBuilder &, Block &, ValueRange)>",
      /*methodName=*/"getRegionBuilder",
      (ins),
      [{ return ConcreteOp::getRegionBuilder(); }]
    >,
    //===------------------------------------------------------------------===//
    // DEPRECATED METHODS
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return true if the payload uses the value loaded from `opOperand`. This
        is useful to avoid loading from "write-only" memory that may be
        uninitialized, as well as properly cloning "read-write" operands.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"payloadUsesValueFromOpOperand",
      /*args=*/(ins "OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        unsigned bbArgNumber =
          $_op.getNumPayloadInductionVariables() + opOperand->getOperandNumber();
        // Safeguard against the named linalg ops that are manually defined and
        // that only support buffer semantics: we should not be there.
        // Such ops have an empty regionBuilder and are not constructed with a
        // region for now. In the future they are slated to disappear.
        assert(this->getOperation()->getNumRegions() == 1 && "unexpected "
               "missing region (calling `payloadUsesValueFromOpOperand` on "
               "manually defined named Linalg op?)");
        Block &block = this->getOperation()->getRegion(0).front();
        // Init tensors have uses.
        return !block.getArgument(bbArgNumber).use_empty();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if the payload uses the value loaded from input operand
        `index`.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"payloadUsesValueFromInputOperandIndex",
      /*args=*/(ins "unsigned":$index),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return payloadUsesValueFromOpOperand(getInputOperand(index));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if the payload uses the value loaded from output operand
        `index`.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"payloadUsesValueFromOutputOperandIndex",
      /*args=*/(ins "unsigned":$index),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return payloadUsesValueFromOpOperand(getOutputOperand(index));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `i`-th input operand.
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
        Return the `i`-th input shaped type
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
        Return the range of input operands.
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
        Return the OpOperands for the input operands.
      }],
      /*retTy=*/" MutableArrayRef<OpOperand>",
      /*methodName=*/"getInputOpOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return this->getOperation()->getOpOperands().take_front(getNumInputs());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of input operands that are of buffer type.
      }],
      /*retTy=*/"SmallVector<Value, 4>",
      /*methodName=*/"getInputBuffers",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::to_vector<4>(llvm::make_filter_range(
          getInputs(), [](Value in){ return in.getType().template isa<MemRefType>(); }));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of input buffer operands.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumInputBuffers",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getInputBuffers().size();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `index`^th input buffer.
      }],
      /*retTy=*/"Value",
      /*methodName=*/"getInputBuffer",
      /*args=*/(ins "unsigned":$index),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(index < getNumInputBuffers());
        return getInputBuffers()[index];
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of input operands that are of buffer type.
      }],
      /*retTy=*/"SmallVector<OpOperand*, 4>",
      /*methodName=*/"getInputBuffersOpOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<OpOperand*, 4> res;
        res.reserve(getNumInputs());
        for (OpOperand &o : getInputOpOperands())
          if (o.get().getType().isa<MemRefType>())
            res.push_back(&o);
        return res;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of input operands that are of tensor type.
      }],
      /*retTy=*/"SmallVector<Value, 4>",
      /*methodName=*/"getInputTensors",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::to_vector<4>(llvm::make_filter_range(
          getInputs(),
          [](Value in){ return in.getType().template isa<RankedTensorType>(); }));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of op operands that are of tensor type.
      }],
      /*retTy=*/"SmallVector<OpOperand*, 4>",
      /*methodName=*/"getInputTensorsOpOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<OpOperand*, 4> res;
        res.reserve(getNumInputs());
        for (OpOperand &o : getInputOpOperands())
          if (o.get().getType().isa<RankedTensorType>())
            res.push_back(&o);
        return res;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `i`-th output operand.
      }],
      /*retTy=*/"Value",
      /*methodName=*/"getOutput",
      /*args=*/(ins "unsigned":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i < $_op.getNumOutputs());
        return this->getOperation()->getOperand(i + $_op.getNumInputs());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `i`-th output shaped type
      }],
      /*retTy=*/"ShapedType",
      /*methodName=*/"getOutputShapedType",
      /*args=*/(ins "unsigned":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getOutput(i).getType().template cast<ShapedType>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the range of output operands.
      }],
      /*retTy=*/"Operation::operand_range",
      /*methodName=*/"getOutputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto start =
          this->getOperation()->getOperands().begin() + $_op.getNumInputs();
        return {start, start + $_op.getNumOutputs()};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the OpOperands for the output operands.
      }],
      /*retTy=*/" MutableArrayRef<OpOperand>",
      /*methodName=*/"getOutputOpOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return this->getOperation()->getOpOperands().slice(
          getNumInputs(), getNumOutputs());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of output operands that are of buffer type.
      }],
      /*retTy=*/"SmallVector<Value, 4>",
      /*methodName=*/"getOutputBuffers",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::to_vector<4>(llvm::make_filter_range(
          getOutputs(), [](Value in){ return in.getType().template isa<MemRefType>(); }));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `index`^th output buffer.
      }],
      /*retTy=*/"Value",
      /*methodName=*/"getOutputBuffer",
      /*args=*/(ins "unsigned":$index),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(index < getNumOutputBuffers());
        return getOutputBuffers()[index];
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of output operands that are of buffer type.
      }],
      /*retTy=*/"SmallVector<OpOperand*, 4>",
      /*methodName=*/"getOutputBuffersOpOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<OpOperand*, 4> res;
        res.reserve(getNumOutputs());
        for (OpOperand &o : getOutputOpOperands())
          if (o.get().getType().isa<MemRefType>())
            res.push_back(&o);
        return res;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of output buffer operands.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumOutputBuffers",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getOutputBuffers().size();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of output operands that are of tensor type.
      }],
      /*retTy=*/"SmallVector<Value, 4>",
      /*methodName=*/"getOutputTensors",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::to_vector<4>(llvm::make_filter_range(
          getOutputs(),
          [](Value in){ return in.getType().template isa<RankedTensorType>(); }));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the subset of output operands that are of tensor type.
      }],
      /*retTy=*/"SmallVector<OpOperand*, 4>",
      /*methodName=*/"getOutputTensorsOpOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<OpOperand*, 4> res;
        res.reserve(getNumOutputs());
        for (OpOperand &o : getOutputOpOperands())
          if (o.get().getType().isa<RankedTensorType>())
            res.push_back(&o);
        return res;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of output tensor operands.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumOutputTensors",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getOutputTensors().size();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of input and output operands.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumShapedOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getNumInputs() + $_op.getNumOutputs();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `i`-th shaped operand value.
      }],
      /*retTy=*/"Value",
      /*methodName=*/"getShapedOperand",
      /*args=*/(ins "unsigned":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i < $_op.getNumShapedOperands());
        return this->getOperation()->getOperand(i);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the range over input and output operands.
      }],
      /*retTy=*/"Operation::operand_range",
      /*methodName=*/"getShapedOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto range = this->getOperation()->getOperands();
        return {range.begin(), range.begin() + getNumShapedOperands()};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the OpOperands for all the shaped operands.
      }],
      /*retTy=*/" MutableArrayRef<OpOperand>",
      /*methodName=*/"getShapedOpOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return this->getOperation()->getOpOperands().take_front(
          getNumShapedOperands());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the OpOperands for all the shaped operands.
      }],
      /*retTy=*/" OpOperand&",
      /*methodName=*/"getShapedOpOperand",
      /*args=*/(ins "unsigned":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return *(this->getShapedOpOperands().begin() + i);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the range over input and output operands.
      }],
      /*retTy=*/"SmallVector<ShapedType, 4>",
      /*methodName=*/"getShapedOperandTypes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::to_vector<4>(
          llvm::map_range(
            getShapedOperands(),
            [](Value v) { return v.getType().cast<ShapedType>(); }));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `i`-th shaped type
      }],
      /*retTy=*/"ShapedType",
      /*methodName=*/"getShapedType",
      /*args=*/(ins "unsigned":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getShapedOperand(i).getType().template cast<ShapedType>();
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
        assert(i < $_op.getNumShapedOperands());
        return getIndexingMaps()[i];
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
        return getIndexingMaps()[i];
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the input indexing maps.
      }],
      /*retTy=*/"SmallVector<AffineMap>",
      /*methodName=*/"getInputIndexingMaps",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto maps = $_op.getIndexingMaps();
        return SmallVector<AffineMap>{maps.begin(),
                                      maps.begin() + $_op.getNumInputs()};
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
        return getIndexingMaps()[i + $_op.getNumInputs()];
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the output indexing maps.
      }],
      /*retTy=*/"SmallVector<AffineMap>",
      /*methodName=*/"getOutputIndexingMaps",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto maps = $_op.getIndexingMaps();
        return SmallVector<AffineMap>{maps.begin() + $_op.getNumInputs(),
                                      maps.begin() + $_op.getNumShapedOperands()};
      }]
    >
  ];

  let extraClassDeclaration = [{
    /// Return the flat list of all operand dimension sizes in the order they
    /// appear in the operands.
    SmallVector<Value, 4> createFlatListOfOperandDims(OpBuilder &, Location);

    /// Return the flat list of all operands' static dimension sizes in the
    /// order they appear in the operands. All operand dimension sizes have to
    /// be statically known.
    SmallVector<int64_t, 4> createFlatListOfOperandStaticDims();

    /// Create the loop ranges to materialize the computation over the current
    /// operands. This is done by applying `getShapesToLoopsMap` to
    /// `createFlatListOfOperandDims`.
    SmallVector<Range, 4> createLoopRanges(OpBuilder &b, Location loc);

    /// Compute the static loop sizes necessary to vectorize the computation.
    /// This is done by applying `getShapesToLoopsMap` to
    /// `createFlatListOfOperandStaticDims`.
    SmallVector<int64_t, 4> computeStaticLoopSizes();

    /// Returns all the operands past the inputs, output_buffers and
    /// init_tensors operands. Asserts that these operands are value types to
    /// allow transformations like tiling to just use the values when cloning
    /// `linalgOp`.
    Operation::operand_range getAssumedNonShapedOperands() {
      Operation::operand_range res{
        getOperation()->getOperands().begin() + getNumInputsAndOutputs(),
        getOperation()->getOperands().end()};
      for (Type t : TypeRange{res}) {
        (void)t;
        assert((t.isSignlessIntOrIndexOrFloat() || t.template isa<VectorType>())
               &&"expected scalar or vector type");
      }
      return res;
    }

    /// Returns the value that expresses the shape of the output in terms of
    /// shape of the input operands where possible
    LogicalResult reifyReturnTypeShapesPerResultDim(OpBuilder &b,
        SmallVectorImpl<SmallVector<Value>> &reifiedReturnShapes);

    //========================================================================//
    // Helper functions to mutate the `operand_segment_sizes` attribute.
    // These are useful when cloning and changing operand types.
    //========================================================================//
    void setNumInputs(unsigned num) { setOperandSegmentAt(0, num); }
    void setNumOutputBuffers(unsigned num) { setOperandSegmentAt(1, num); }

    private:
    void setOperandSegmentAt(unsigned idx, unsigned val) {
      auto attr = (*this)->getAttr("operand_segment_sizes")
        .cast<DenseIntElementsAttr>();
      unsigned i = 0;
      auto newAttr = attr.mapValues(IntegerType::get(getContext(), 32),
        [&](const APInt &v) { return (i++ == idx) ? APInt(32, val) : v; });
      getOperation()->setAttr("operand_segment_sizes", newAttr);
    }
  }];

  let verify = [{ return detail::verifyStructuredOpInterface($_op); }];
}

#endif // LINALG_IR_LINALGINTERFACES
