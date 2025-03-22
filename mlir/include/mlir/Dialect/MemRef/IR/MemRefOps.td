//===- MemRefOps.td - MemRef op definitions ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MEMREF_OPS
#define MEMREF_OPS

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

/// A TypeAttr for memref types.
def MemRefTypeAttr
    : TypeAttrBase<"::mlir::MemRefType", "memref type attribute"> {
  let constBuilderCall = "::mlir::TypeAttr::get($0)";
}

class MemRef_Op<string mnemonic, list<Trait> traits = []>
    : Op<MemRef_Dialect, mnemonic, traits>;

// Base class for ops with static/dynamic offset, sizes and strides
// attributes/arguments.
class MemRef_OpWithOffsetSizesAndStrides<string mnemonic,
                                         list<Trait> traits = []>
    : MemRef_Op<mnemonic, traits> {
  code extraBaseClassDeclaration = [{
    /// Returns the dynamic sizes for this subview operation if specified.
    ::mlir::Operation::operand_range getDynamicSizes() { return getSizes(); }

    /// Return the list of Range (i.e. offset, size, stride). Each
    /// Range entry contains either the dynamic value or a ConstantIndexOp
    /// constructed with `b` at location `loc`.
    ::mlir::SmallVector<::mlir::Range, 8> getOrCreateRanges(
        ::mlir::OpBuilder &b, ::mlir::Location loc) {
      return ::mlir::getOrCreateRanges(*this, b, loc);
    }
  }];
}

//===----------------------------------------------------------------------===//
// AllocLikeOp
//===----------------------------------------------------------------------===//

// Base class for memref allocating ops: alloca and alloc.
//
//   %0 = alloclike(%m)[%s] : memref<8x?xf32, affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>>
//
class AllocLikeOp<string mnemonic,
                  Resource resource,
                  list<Trait> traits = []> :
    MemRef_Op<mnemonic,
    !listconcat([
      AttrSizedOperandSegments
    ], traits)> {

  let arguments = (ins Variadic<Index>:$dynamicSizes,
                       // The symbolic operands (the ones in square brackets)
                       // bind to the symbols of the memref's layout map.
                       Variadic<Index>:$symbolOperands,
                       ConfinedAttr<OptionalAttr<I64Attr>,
                                [IntMinValue<0>]>:$alignment);
  let results = (outs Res<AnyMemRef, "",
                          [MemAlloc<resource, 0, FullEffect>]>:$memref);

  let builders = [
    OpBuilder<(ins "MemRefType":$memrefType,
                  CArg<"IntegerAttr", "IntegerAttr()">:$alignment), [{
      return build($_builder, $_state, memrefType, {}, alignment);
    }]>,
    OpBuilder<(ins "MemRefType":$memrefType, "ValueRange":$dynamicSizes,
                  CArg<"IntegerAttr", "IntegerAttr()">:$alignment), [{
      return build($_builder, $_state, memrefType, dynamicSizes, {}, alignment);
    }]>,
    OpBuilder<(ins "MemRefType":$memrefType, "ValueRange":$dynamicSizes,
                  "ValueRange":$symbolOperands,
                  CArg<"IntegerAttr", "{}">:$alignment), [{
      $_state.types.push_back(memrefType);
      $_state.addOperands(dynamicSizes);
      $_state.addOperands(symbolOperands);
      $_state.addAttribute(getOperandSegmentSizeAttr(),
          $_builder.getDenseI32ArrayAttr({
              static_cast<int32_t>(dynamicSizes.size()),
              static_cast<int32_t>(symbolOperands.size())}));
      if (alignment)
        $_state.addAttribute(getAlignmentAttrStrName(), alignment);
    }]>,
    OpBuilder<(ins "ArrayRef<OpFoldResult>":$sizes, "Type":$elementType,
                   CArg<"Attribute", "{}">:$memorySpace), [{
      SmallVector<int64_t> staticShape;
      SmallVector<Value> dynamicSizes;
      dispatchIndexOpFoldResults(sizes, dynamicSizes, staticShape);
      MemRefLayoutAttrInterface layout;
      MemRefType memrefType = MemRefType::get(staticShape, elementType, layout,
                                              memorySpace);
      return build($_builder, $_state, memrefType, dynamicSizes);
    }]>
  ];

  let extraClassDeclaration = [{
    static StringRef getAlignmentAttrStrName() { return "alignment"; }

    MemRefType getType() { return ::llvm::cast<MemRefType>(getResult().getType()); }

    SmallVector<OpFoldResult> getMixedSizes() {
      SmallVector<OpFoldResult> result;
      unsigned ctr = 0;
      OpBuilder b(getContext());
      for (int64_t i = 0, e = getType().getRank(); i < e; ++i) {
        if (getType().isDynamicDim(i)) {
          result.push_back(getDynamicSizes()[ctr++]);
        } else {
          result.push_back(b.getIndexAttr(getType().getShape()[i]));
        }
      }
      return result;
    }
  }];

  let assemblyFormat = [{
    `(`$dynamicSizes`)` (`` `[` $symbolOperands^ `]`)? attr-dict `:` type($memref)
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AssumeAlignmentOp
//===----------------------------------------------------------------------===//

def AssumeAlignmentOp : MemRef_Op<"assume_alignment"> {
  let summary =
      "assertion that gives alignment information to the input memref";
  let description = [{
    The `assume_alignment` operation takes a memref and an integer of alignment
    value, and internally annotates the buffer with the given alignment. If
    the buffer isn't aligned to the given alignment, the behavior is undefined.

    This operation doesn't affect the semantics of a correct program. It's for
    optimization only, and the optimization is best-effort.
  }];
  let arguments = (ins AnyMemRef:$memref,
                       ConfinedAttr<I32Attr, [IntPositive]>:$alignment);
  let results = (outs);

  let assemblyFormat = "$memref `,` $alignment attr-dict `:` type($memref)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

def MemRef_AllocOp : AllocLikeOp<"alloc", DefaultResource, [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "memory allocation operation";
  let description = [{
    The `alloc` operation allocates a region of memory, as specified by its
    memref type.

    Example:

    ```mlir
    %0 = memref.alloc() : memref<8x64xf32, 1>
    ```

    The optional list of dimension operands are bound to the dynamic dimensions
    specified in its memref type. In the example below, the ssa value '%d' is
    bound to the second dimension of the memref (which is dynamic).

    ```mlir
    %0 = memref.alloc(%d) : memref<8x?xf32, 1>
    ```

    The optional list of symbol operands are bound to the symbols of the
    memrefs affine map. In the example below, the ssa value '%s' is bound to
    the symbol 's0' in the affine map specified in the allocs memref type.

    ```mlir
    %0 = memref.alloc()[%s] : memref<8x64xf32,
                              affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
    ```

    This operation returns a single ssa value of memref type, which can be used
    by subsequent load and store operations.

    The optional `alignment` attribute may be specified to ensure that the
    region of memory that will be indexed is aligned at the specified byte
    boundary.

    ```mlir
    %0 = memref.alloc()[%s] {alignment = 8} :
      memref<8x64xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
    ```
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReallocOp
//===----------------------------------------------------------------------===//


def MemRef_ReallocOp : MemRef_Op<"realloc"> {
  let summary = "memory reallocation operation";
  let description = [{
    The `realloc` operation changes the size of a memory region. The memory
    region is specified by a 1D source memref and the size of the new memory
    region is specified by a 1D result memref type and an optional dynamic Value
    of `Index` type. The source and the result memref must be in the same memory
    space and have the same element type.

    The operation may move the memory region to a new location. In this case,
    the content of the memory block is preserved up to the lesser of the new
    and old sizes. If the new size if larger, the value of the extended memory
    is undefined. This is consistent with the ISO C realloc.

    The operation returns an SSA value for the memref.

    Example:

    ```mlir
    %0 = memref.realloc %src : memref<64xf32> to memref<124xf32>
    ```

    The source memref may have a dynamic shape, in which case, the compiler will
    generate code to extract its size from the runtime data structure for the
    memref.

    ```mlir
    %1 = memref.realloc %src : memref<?xf32> to memref<124xf32>
    ```

    If the result memref has a dynamic shape, a result dimension operand is
    needed to spefify its dynamic dimension. In the example below, the ssa value
    '%d' specifies the unknown dimension of the result memref.

    ```mlir
    %2 = memref.realloc %src(%d) : memref<?xf32> to memref<?xf32>
    ```

    An optional `alignment` attribute may be specified to ensure that the
    region of memory that will be indexed is aligned at the specified byte
    boundary.  This is consistent with the fact that memref.alloc supports such
    an optional alignment attribute. Note that in ISO C standard, neither alloc
    nor realloc supports alignment, though there is aligned_alloc but not
    aligned_realloc.

    ```mlir
    %3 = memref.realloc %src {alignment = 8} : memref<64xf32> to memref<124xf32>
    ```

    Referencing the memref through the old SSA value after realloc is undefined
    behavior.

    ```mlir
    %new = memref.realloc %old : memref<64xf32> to memref<124xf32>
    %4 = memref.load %new[%index]   // ok
    %5 = memref.load %old[%index]   // undefined behavior
    ```
  }];

  // Note that we conceptually mark the operands as freeing the incoming
  // memref and allocating the outcoming memref, even though this may not
  // physically happen on each execution.

  let arguments = (ins Arg<MemRefRankOf<[AnyType], [1]>, "",
                                        [MemFreeAt<0, FullEffect>]>:$source,
                   Optional<Index>:$dynamicResultSize,
                   ConfinedAttr<OptionalAttr<I64Attr>,
                                [IntMinValue<0>]>:$alignment);

  let results = (outs Res<MemRefRankOf<[AnyType], [1]>, "",
                                       [MemAlloc<DefaultResource, 1,
                                                 FullEffect>]>);

  let builders = [
    OpBuilder<(ins "MemRefType":$resultType,
                  "Value":$source,
                  CArg<"Value", "Value()">:$dynamicResultSize), [{
      return build($_builder, $_state, resultType, source, dynamicResultSize,
                   IntegerAttr());
    }]>];

    let extraClassDeclaration = [{
    /// The result of a realloc is always a memref.
    MemRefType getType() { return ::llvm::cast<MemRefType>(getResult().getType()); }
  }];

  let assemblyFormat = [{
    $source (`(` $dynamicResultSize^ `)`)? attr-dict
    `:` type($source) `to` type(results)
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

def MemRef_AllocaOp : AllocLikeOp<"alloca", AutomaticAllocationScopeResource,[
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<PromotableAllocationOpInterface>,
    DeclareOpInterfaceMethods<DestructurableAllocationOpInterface>]> {
  let summary = "stack memory allocation operation";
  let description = [{
    The `alloca` operation allocates memory on the stack, to be automatically
    released when control transfers back from the region of its closest
    surrounding operation with an
    [`AutomaticAllocationScope`](../Traits.md/#automaticallocationscope) trait.
    The amount of memory allocated is specified by its memref and additional
    operands. For example:

    ```mlir
    %0 = memref.alloca() : memref<8x64xf32>
    ```

    The optional list of dimension operands are bound to the dynamic dimensions
    specified in its memref type. In the example below, the SSA value '%d' is
    bound to the second dimension of the memref (which is dynamic).

    ```mlir
    %0 = memref.alloca(%d) : memref<8x?xf32>
    ```

    The optional list of symbol operands are bound to the symbols of the
    memref's affine map. In the example below, the SSA value '%s' is bound to
    the symbol 's0' in the affine map specified in the allocs memref type.

    ```mlir
    %0 = memref.alloca()[%s] : memref<8x64xf32,
                               affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>>
    ```

    This operation returns a single SSA value of memref type, which can be used
    by subsequent load and store operations. An optional alignment attribute, if
    specified, guarantees alignment at least to that boundary. If not specified,
    an alignment on any convenient boundary compatible with the type will be
    chosen.
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AllocaScopeOp
//===----------------------------------------------------------------------===//

def MemRef_AllocaScopeOp : MemRef_Op<"alloca_scope",
      [AutomaticAllocationScope,
       DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       SingleBlockImplicitTerminator<"AllocaScopeReturnOp">,
       RecursiveMemoryEffects,
       NoRegionArguments]> {
  let summary = "explicitly delimited scope for stack allocation";
  let description = [{
    The `memref.alloca_scope` operation represents an explicitly-delimited
    scope for the alloca allocations. Any `memref.alloca` operations that are
    used within this scope are going to be cleaned up automatically once
    the control-flow exits the nested region. For example:

    ```mlir
    memref.alloca_scope {
      %myalloca = memref.alloca(): memref<4x3xf32>
      ...
    }
    ```

    Here, `%myalloca` memref is valid within the explicitly delimited scope
    and is automatically deallocated at the end of the given region. Conceptually,
    `memref.alloca_scope` is a passthrough operation with
    `AutomaticAllocationScope` that spans the body of the region within the operation.

    `memref.alloca_scope` may also return results that are defined in the nested
    region. To return a value, one should use `memref.alloca_scope.return`
    operation:

    ```mlir
    %result = memref.alloca_scope {
      ...
      memref.alloca_scope.return %value
    }
    ```

    If `memref.alloca_scope` returns no value, the `memref.alloca_scope.return ` can
    be left out, and will be inserted implicitly.
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// AllocaScopeReturnOp
//===----------------------------------------------------------------------===//

def MemRef_AllocaScopeReturnOp : MemRef_Op<"alloca_scope.return",
      [HasParent<"AllocaScopeOp">,
       Pure,
       ReturnLike,
       Terminator]> {
  let summary = "terminator for alloca_scope operation";
  let description = [{
    `memref.alloca_scope.return` operation returns zero or more SSA values
    from the region within `memref.alloca_scope`. If no values are returned,
    the return operation may be omitted. Otherwise, it has to be present
    to indicate which values are going to be returned. For example:

    ```mlir
    memref.alloca_scope.return %value
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /*nothing to do */ }]>];

  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def MemRef_CastOp : MemRef_Op<"cast", [
      DeclareOpInterfaceMethods<CastOpInterface>,
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
      MemRefsNormalizable,
      Pure,
      SameOperandsAndResultShape,
      ViewLikeOpInterface
    ]> {
  let summary = "memref cast operation";
  let description = [{
    The `memref.cast` operation converts a memref from one type to an equivalent
    type with a compatible shape. The source and destination types are
    compatible if:

    a. Both are ranked memref types with the same element type, address space,
    and rank and:
      1. Both have the same layout or both have compatible strided layouts.
      2. The individual sizes (resp. offset and strides in the case of strided
         memrefs) may convert constant dimensions to dynamic dimensions and
         vice-versa.

    If the cast converts any dimensions from an unknown to a known size, then it
    acts as an assertion that fails at runtime if the dynamic dimensions
    disagree with resultant destination size.

    Example:

    ```mlir
    // Assert that the input dynamic shape matches the destination static shape.
    %2 = memref.cast %1 : memref<?x?xf32> to memref<4x4xf32>
    // Erase static shape information, replacing it with dynamic information.
    %3 = memref.cast %1 : memref<4xf32> to memref<?xf32>

    // The same holds true for offsets and strides.

    // Assert that the input dynamic shape matches the destination static stride.
    %4 = memref.cast %1 : memref<12x4xf32, strided<[?, ?], offset: ?>> to
                          memref<12x4xf32, strided<[4, 1], offset: 5>>
    // Erase static offset and stride information, replacing it with
    // dynamic information.
    %5 = memref.cast %1 : memref<12x4xf32, strided<[4, 1], offset: 5>> to
                          memref<12x4xf32, strided<[?, ?], offset: ?>>
    ```

    b. Either or both memref types are unranked with the same element type, and
    address space.

    Example:

    ```mlir
    Cast to concrete shape.
        %4 = memref.cast %1 : memref<*xf32> to memref<4x?xf32>

    Erase rank information.
        %5 = memref.cast %1 : memref<4x?xf32> to memref<*xf32>
    ```
  }];

  let arguments = (ins AnyRankedOrUnrankedMemRef:$source);
  let results = (outs AnyRankedOrUnrankedMemRef:$dest);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";

  let extraClassDeclaration = [{
    /// Fold the given CastOp into consumer op.
    static bool canFoldIntoConsumerOp(CastOp castOp);

    Value getViewSource() { return getSource(); }
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

def CopyOp : MemRef_Op<"copy", [CopyOpInterface, SameOperandsElementType,
    SameOperandsShape]> {

  let description = [{
    Copies the data from the source to the destination memref.

    Usage:

    ```mlir
    memref.copy %arg0, %arg1 : memref<?xf32> to memref<?xf32>
    ```

    Source and destination are expected to have the same element type and shape.
    Otherwise, the result is undefined. They may have different layouts.
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "the memref to copy from",
                           [MemReadAt<0, FullEffect>]>:$source,
                       Arg<AnyRankedOrUnrankedMemRef, "the memref to copy to",
                           [MemWriteAt<0, FullEffect>]>:$target);

  let assemblyFormat = [{
    $source `,` $target attr-dict `:` type($source) `to` type($target)
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

def MemRef_DeallocOp : MemRef_Op<"dealloc", [MemRefsNormalizable]> {
  let summary = "memory deallocation operation";
  let description = [{
    The `dealloc` operation frees the region of memory referenced by a memref
    which was originally created by the `alloc` operation.
    The `dealloc` operation should not be called on memrefs which alias an
    alloc'd memref (e.g. memrefs returned by `view` operations).

    Example:

    ```mlir
    %0 = memref.alloc() : memref<8x64xf32, affine_map<(d0, d1) -> (d0, d1), 1>>
    memref.dealloc %0 : memref<8x64xf32,  affine_map<(d0, d1) -> (d0, d1), 1>>
    ```
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "",
                           [MemFreeAt<0, FullEffect>]>:$memref);

  let hasFolder = 1;
  let assemblyFormat = "$memref attr-dict `:` type($memref)";
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

def MemRef_DimOp : MemRef_Op<"dim", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    MemRefsNormalizable,
    ConditionallySpeculatable, NoMemoryEffect,
    ShapedDimOpInterface,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRangesFromOptional"]>]> {
  let summary = "dimension index operation";
  let description = [{
    The `dim` operation takes a memref and a dimension operand of type `index`.
    It returns the size of the requested dimension of the given memref.
    If the dimension index is out of bounds the behavior is undefined.

    The specified memref type is that of the first operand.

    Example:

    ```mlir
    // Always returns 4, can be constant folded:
    %c0 = arith.constant 0 : index
    %x = memref.dim %A, %c0 : memref<4 x ? x f32>

    // Returns the dynamic dimension of %A.
    %c1 = arith.constant 1 : index
    %y = memref.dim %A, %c1 : memref<4 x ? x f32>

    // Equivalent generic form:
    %x = "memref.dim"(%A, %c0) : (memref<4 x ? x f32>, index) -> index
    %y = "memref.dim"(%A, %c1) : (memref<4 x ? x f32>, index) -> index
    ```
  }];

  let arguments = (ins AnyNon0RankedOrUnrankedMemRef:$source,
                       Index:$index);
  let results = (outs Index:$result);

  let assemblyFormat = [{
    attr-dict $source `,` $index `:` type($source)
  }];

  let builders = [
    OpBuilder<(ins "Value":$source, "int64_t":$index)>,
  ];

  let extraClassDeclaration = [{
    /// Helper function to get the index as a simple integer if it is constant.
    std::optional<int64_t> getConstantIndex();

    /// Interface method of ShapedDimOpInterface: Return the source memref.
    Value getShapedValue() { return getSource(); }

    /// Interface method of ShapedDimOpInterface: Return the dimension.
    OpFoldResult getDimension() { return getIndex(); }

    /// Interface method for ConditionallySpeculatable.
    Speculation::Speculatability getSpeculatability();
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// DmaStartOp
//===----------------------------------------------------------------------===//

def MemRef_DmaStartOp : MemRef_Op<"dma_start"> {
  let summary = "non-blocking DMA operation that starts a transfer";
  let description = [{
    Syntax:

    ```
    operation ::= `memref.dma_start` ssa-use`[`ssa-use-list`]` `,`
                   ssa-use`[`ssa-use-list`]` `,` ssa-use `,`
                   ssa-use`[`ssa-use-list`]` (`,` ssa-use `,` ssa-use)?
                  `:` memref-type `,` memref-type `,` memref-type
    ```

    DmaStartOp starts a non-blocking DMA operation that transfers data from a
    source memref to a destination memref. The source and destination memref
    need not be of the same dimensionality, but need to have the same elemental
    type. The operands include the source and destination memref's each followed
    by its indices, size of the data transfer in terms of the number of elements
    (of the elemental type of the memref), a tag memref with its indices, and
    optionally at the end, a stride and a number_of_elements_per_stride
    arguments. The tag location is used by a DmaWaitOp to check for completion.
    The indices of the source memref, destination memref, and the tag memref
    have the same restrictions as any load/store. The optional stride arguments
    should be of 'index' type, and specify a stride for the slower memory space
    (memory space with a lower memory space id), transferring chunks of
    number_of_elements_per_stride every stride until %num_elements are
    transferred. Either both or no stride arguments should be specified. If the
    source and destination locations overlap the behavior of this operation is
    not defined.

    For example, a DmaStartOp operation that transfers 256 elements of a memref
    '%src' in memory space 0 at indices [%i, %j] to memref '%dst' in memory
    space 1 at indices [%k, %l], would be specified as follows:

    ```mlir
    %num_elements = arith.constant 256
    %idx = arith.constant 0 : index
    %tag = memref.alloc() : memref<1 x i32, affine_map<(d0) -> (d0)>, 4>
    dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%idx] :
      memref<40 x 128 x f32>, affine_map<(d0) -> (d0)>, 0>,
      memref<2 x 1024 x f32>, affine_map<(d0) -> (d0)>, 1>,
      memref<1 x i32>, affine_map<(d0) -> (d0)>, 2>
    ```

    If %stride and %num_elt_per_stride are specified, the DMA is expected to
    transfer %num_elt_per_stride elements every %stride elements apart from
    memory space 0 until %num_elements are transferred.

    ```mlir
    dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%idx], %stride,
              %num_elt_per_stride :
    ```

    * TODO: add additional operands to allow source and destination striding, and
    multiple stride levels.
    * TODO: Consider replacing src/dst memref indices with view memrefs.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins "Value":$srcMemRef, "ValueRange":$srcIndices,
                   "Value":$destMemRef, "ValueRange":$destIndices,
                   "Value":$numElements, "Value":$tagMemRef,
                   "ValueRange":$tagIndices, CArg<"Value", "{}">:$stride,
                   CArg<"Value", "{}">:$elementsPerStride)>
  ];

  let extraClassDeclaration = [{
    // Returns the source MemRefType for this DMA operation.
    Value getSrcMemRef() { return getOperand(0); }
    OpOperand &getSrcMemRefMutable() { return getOperation()->getOpOperand(0); }
    // Returns the rank (number of indices) of the source MemRefType.
    unsigned getSrcMemRefRank() {
      return ::llvm::cast<MemRefType>(getSrcMemRef().getType()).getRank();
    }
    // Returns the source memref indices for this DMA operation.
    operand_range getSrcIndices() {
      return {(*this)->operand_begin() + 1,
              (*this)->operand_begin() + 1 + getSrcMemRefRank()};
    }

    // Returns the destination MemRefType for this DMA operations.
    Value getDstMemRef() { return getOperand(1 + getSrcMemRefRank()); }
    OpOperand &getDstMemRefMutable() { return getOperation()->getOpOperand(1 + getSrcMemRefRank()); }
    // Returns the rank (number of indices) of the destination MemRefType.
    unsigned getDstMemRefRank() {
      return ::llvm::cast<MemRefType>(getDstMemRef().getType()).getRank();
    }
    unsigned getSrcMemorySpace() {
      return ::llvm::cast<MemRefType>(getSrcMemRef().getType()).getMemorySpaceAsInt();
    }
    unsigned getDstMemorySpace() {
      return ::llvm::cast<MemRefType>(getDstMemRef().getType()).getMemorySpaceAsInt();
    }

    // Returns the destination memref indices for this DMA operation.
    operand_range getDstIndices() {
      return {(*this)->operand_begin() + 1 + getSrcMemRefRank() + 1,
              (*this)->operand_begin() + 1 + getSrcMemRefRank() + 1 +
                  getDstMemRefRank()};
    }

    // Returns the number of elements being transferred by this DMA operation.
    Value getNumElements() {
      return getOperand(1 + getSrcMemRefRank() + 1 + getDstMemRefRank());
    }

    // Returns the Tag MemRef for this DMA operation.
    Value getTagMemRef() {
      return getOperand(1 + getSrcMemRefRank() + 1 + getDstMemRefRank() + 1);
    }
    OpOperand &getTagMemRefMutable() {
      return getOperation()->getOpOperand(1 + getSrcMemRefRank() + 1 + getDstMemRefRank() + 1);
    }

    // Returns the rank (number of indices) of the tag MemRefType.
    unsigned getTagMemRefRank() {
      return ::llvm::cast<MemRefType>(getTagMemRef().getType()).getRank();
    }

    // Returns the tag memref index for this DMA operation.
    operand_range getTagIndices() {
      unsigned tagIndexStartPos =
          1 + getSrcMemRefRank() + 1 + getDstMemRefRank() + 1 + 1;
      return {(*this)->operand_begin() + tagIndexStartPos,
              (*this)->operand_begin() + tagIndexStartPos + getTagMemRefRank()};
    }

    /// Returns true if this is a DMA from a faster memory space to a slower
    /// one.
    bool isDestMemorySpaceFaster() {
      return (getSrcMemorySpace() < getDstMemorySpace());
    }

    /// Returns true if this is a DMA from a slower memory space to a faster
    /// one.
    bool isSrcMemorySpaceFaster() {
      // Assumes that a lower number is for a slower memory space.
      return (getDstMemorySpace() < getSrcMemorySpace());
    }

    /// Given a DMA start operation, returns the operand position of either the
    /// source or destination memref depending on the one that is at the higher
    /// level of the memory hierarchy. Asserts failure if neither is true.
    unsigned getFasterMemPos() {
      assert(isSrcMemorySpaceFaster() || isDestMemorySpaceFaster());
      return isSrcMemorySpaceFaster() ? 0 : getSrcMemRefRank() + 1;
    }

    bool isStrided() {
      return getNumOperands() != 1 + getSrcMemRefRank() + 1 +
                                 getDstMemRefRank() + 1 + 1 +
                                 getTagMemRefRank();
    }

    Value getStride() {
      if (!isStrided())
        return nullptr;
      return getOperand(getNumOperands() - 1 - 1);
    }

    Value getNumElementsPerStride() {
      if (!isStrided())
        return nullptr;
      return getOperand(getNumOperands() - 1);
    }

    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getSrcMemRefMutable(),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getDstMemRefMutable(),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getTagMemRefMutable(),
                           SideEffects::DefaultResource::get());
    }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DmaWaitOp
//===----------------------------------------------------------------------===//

def MemRef_DmaWaitOp : MemRef_Op<"dma_wait"> {
  let summary = "blocking DMA operation that waits for transfer completion";
  let description = [{
   DmaWaitOp blocks until the completion of a DMA operation associated with the
   tag element '%tag[%index]'. %tag is a memref, and %index has to be an index
   with the same restrictions as any load/store index. %num_elements is the
   number of elements associated with the DMA operation.

   Example:

   ```mlir
    dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%index] :
      memref<2048 x f32>, affine_map<(d0) -> (d0)>, 0>,
      memref<256 x f32>, affine_map<(d0) -> (d0)>, 1>
      memref<1 x i32>, affine_map<(d0) -> (d0)>, 2>
    ...
    ...
    dma_wait %tag[%index], %num_elements : memref<1 x i32, affine_map<(d0) -> (d0)>, 2>
    ```
  }];
  let arguments = (ins AnyMemRef:$tagMemRef,
                       Variadic<Index>:$tagIndices,
                       Index:$numElements);
  let assemblyFormat = [{
    $tagMemRef `[` $tagIndices `]` `,` $numElements attr-dict `:` type($tagMemRef)
  }];
  let extraClassDeclaration = [{
    /// Returns the rank (number of indices) of the tag memref.
    unsigned getTagMemRefRank() {
      return ::llvm::cast<MemRefType>(getTagMemRef().getType()).getRank();
    }
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getTagMemRefMutable(),
                           SideEffects::DefaultResource::get());
    }
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExtractAlignedPointerAsIndexOp
//===----------------------------------------------------------------------===//

def MemRef_ExtractAlignedPointerAsIndexOp :
  MemRef_Op<"extract_aligned_pointer_as_index", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure,
    SameVariadicResultSize]> {
  let summary = "Extracts a memref's underlying aligned pointer as an index";
  let description = [{
    Extracts the underlying aligned pointer as an index.

    This operation is useful for lowering to lower-level dialects while still
    avoiding the need to define a pointer type in higher-level dialects such as
    the memref dialect.

    This operation is intended solely as step during lowering, it has no side
    effects. A reverse operation that creates a memref from an index interpreted
    as a pointer is explicitly discouraged.

    Example:

    ```
      %0 = memref.extract_aligned_pointer_as_index %arg : memref<4x4xf32> -> index
      %1 = arith.index_cast %0 : index to i64
      %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
      call @foo(%2) : (!llvm.ptr) ->()
    ```
  }];

  let arguments = (ins
    AnyRankedOrUnrankedMemRef:$source
  );
  let results = (outs Index:$aligned_pointer);

  let assemblyFormat = [{
    $source `:` type($source) `->` type(results) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ExtractStridedMetadataOp
//===----------------------------------------------------------------------===//

def MemRef_ExtractStridedMetadataOp : MemRef_Op<"extract_strided_metadata", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure,
    SameVariadicResultSize,
    ViewLikeOpInterface,
    InferTypeOpAdaptor]> {
  let summary = "Extracts a buffer base with offset and strides";
  let description = [{
    Extracts a base buffer, offset and strides. This op allows additional layers
    of transformations and foldings to be added as lowering progresses from
    higher-level dialect to lower-level dialects such as the LLVM dialect.

    The op requires a strided memref source operand. If the source operand is not
    a strided memref, then verification fails.

    This operation is also useful for completeness to the existing memref.dim op.
    While accessing strides, offsets and the base pointer independently is not
    available, this is useful for composing with its natural complement op:
    `memref.reinterpret_cast`.

    Intended Use Cases:

    The main use case is to expose the logic for manipulate memref metadata at a
    higher level than the LLVM dialect.
    This makes lowering more progressive and brings the following benefits:
      - not all users of MLIR want to lower to LLVM and the information to e.g.
        lower to library calls---like libxsmm---or to SPIR-V was not available.
      - foldings and canonicalizations can happen at a higher level in MLIR:
        before this op existed, lowering to LLVM would create large amounts of
        LLVMIR. Even when LLVM does a good job at folding the low-level IR from
        a performance perspective, it is unnecessarily opaque and inefficient to
        send unkempt IR to LLVM.

    Example:

    ```mlir
      %base, %offset, %sizes:2, %strides:2 =
        memref.extract_strided_metadata %memref :
          memref<10x?xf32>, index, index, index, index, index

      // After folding, the type of %m2 can be memref<10x?xf32> and further
      // folded to %memref.
      %m2 = memref.reinterpret_cast %base to
          offset: [%offset],
          sizes: [%sizes#0, %sizes#1],
          strides: [%strides#0, %strides#1]
        : memref<f32> to memref<?x?xf32, offset: ?, strides: [?, ?]>
    ```
  }];

  let arguments = (ins
    AnyStridedMemRef:$source
  );
  let results = (outs
    AnyStridedMemRefOfRank<0>:$base_buffer,
    Index:$offset,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides
  );

  let assemblyFormat = [{
    $source `:` type($source) `->` type(results) attr-dict
  }];

  let extraClassDeclaration = [{
    /// Return a vector of all the static or dynamic sizes of the op, while
    /// statically inferring the sizes of the dynamic sizes, when possible.
    /// This is best effort.
    /// E.g., if `getSizes` returns `[%dyn_size0, %dyn_size1]`, but the
    /// source memref type is `memref<2x8xi16>`, this method will
    /// return `[2, 8]`.
    /// Similarly if the resulting memref type is `memref<2x?xi16>`, but
    /// `%dyn_size1` can statically be pinned to a constant value, this
    /// constant value is returned instead of `%dyn_size`.
    SmallVector<OpFoldResult> getConstifiedMixedSizes();
    /// Similar to `getConstifiedMixedSizes` but for strides.
    SmallVector<OpFoldResult> getConstifiedMixedStrides();
    /// Similar to `getConstifiedMixedSizes` but for the offset.
    OpFoldResult getConstifiedMixedOffset();

    ::mlir::Value getViewSource() { return getSource(); }
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// GenericAtomicRMWOp
//===----------------------------------------------------------------------===//

def GenericAtomicRMWOp : MemRef_Op<"generic_atomic_rmw", [
      SingleBlockImplicitTerminator<"AtomicYieldOp">,
      TypesMatchWith<"result type matches element type of memref",
                     "memref", "result",
                     "::llvm::cast<MemRefType>($_self).getElementType()">
    ]> {
  let summary = "atomic read-modify-write operation with a region";
  let description = [{
    The `memref.generic_atomic_rmw` operation provides a way to perform a
    read-modify-write sequence that is free from data races. The memref operand
    represents the buffer that the read and write will be performed against, as
    accessed by the specified indices. The arity of the indices is the rank of
    the memref. The result represents the latest value that was stored. The
    region contains the code for the modification itself. The entry block has
    a single argument that represents the value stored in `memref[indices]`
    before the write is performed. No side-effecting ops are allowed in the
    body of `GenericAtomicRMWOp`.

    Example:

    ```mlir
    %x = memref.generic_atomic_rmw %I[%i] : memref<10xf32> {
      ^bb0(%current_value : f32):
        %c1 = arith.constant 1.0 : f32
        %inc = arith.addf %c1, %current_value : f32
        memref.atomic_yield %inc : f32
    }
    ```
  }];

  let arguments = (ins
      Arg<MemRefOf<[AnySignlessInteger, AnyFloat]>, "the reference to read from and write to", [MemRead, MemWrite]>:$memref,
      Variadic<Index>:$indices);

  let results = (outs
      AnyTypeOf<[AnySignlessInteger, AnyFloat]>:$result);

  let regions = (region AnyRegion:$atomic_body);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$memref, "ValueRange":$ivs)>];

  let extraClassDeclaration = [{
    // TODO: remove post migrating callers.
    Region &body() { return getRegion(); }

    // The value stored in memref[ivs].
    Value getCurrentValue() {
      return getRegion().getArgument(0);
    }
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getMemref().getType());
    }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def AtomicYieldOp : MemRef_Op<"atomic_yield", [
      HasParent<"GenericAtomicRMWOp">,
      Pure,
      Terminator
    ]> {
  let summary = "yield operation for GenericAtomicRMWOp";
  let description = [{
    "memref.atomic_yield" yields an SSA value from a
    GenericAtomicRMWOp region.
  }];

  let arguments = (ins AnyType:$result);
  let assemblyFormat = "$result attr-dict `:` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GetGlobalOp
//===----------------------------------------------------------------------===//

def MemRef_GetGlobalOp : MemRef_Op<"get_global",
    [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "get the memref pointing to a global variable";
  let description = [{
     The `memref.get_global` operation retrieves the memref pointing to a
     named global variable. If the global variable is marked constant, writing
     to the result memref (such as through a `memref.store` operation) is
     undefined.

     Example:

     ```mlir
     %x = memref.get_global @foo : memref<2xf32>
     ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs AnyStaticShapeMemRef:$result);
  let assemblyFormat = "$name `:` type($result) attr-dict";
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

def MemRef_GlobalOp : MemRef_Op<"global", [Symbol]> {
  let summary = "declare or define a global memref variable";
  let description = [{
    The `memref.global` operation declares or defines a named global memref
    variable. The backing memory for the variable is allocated statically and is
    described by the type of the variable (which should be a statically shaped
    memref type). The operation is a declaration if no `initial_value` is
    specified, else it is a definition. The `initial_value` can either be a unit
    attribute to represent a definition of an uninitialized global variable, or
    an elements attribute to represent the definition of a global variable with
    an initial value. The global variable can also be marked constant using the
    `constant` unit attribute. Writing to such constant global variables is
    undefined.

    The global variable can be accessed by using the `memref.get_global` to
    retrieve the memref for the global variable. Note that the memref
    for such global variable itself is immutable (i.e., memref.get_global for a
    given global variable will always return the same memref descriptor).

    Example:

    ```mlir
    // Private variable with an initial value.
    memref.global "private" @x : memref<2xf32> = dense<0.0,2.0>

    // Private variable with an initial value and an alignment (power of 2).
    memref.global "private" @x : memref<2xf32> = dense<0.0,2.0> {alignment = 64}

    // Declaration of an external variable.
    memref.global "private" @y : memref<4xi32>

    // Uninitialized externally visible variable.
    memref.global @z : memref<3xf16> = uninitialized

    // Externally visible constant variable.
    memref.global constant @c : memref<2xi32> = dense<1, 4>
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       MemRefTypeAttr:$type,
                       OptionalAttr<AnyAttr>:$initial_value,
                       UnitAttr:$constant,
                       OptionalAttr<I64Attr>:$alignment);

  let assemblyFormat = [{
       ($sym_visibility^)?
       (`constant` $constant^)?
       $sym_name `:`
       custom<GlobalMemrefOpTypeAndInitialValue>($type, $initial_value)
       attr-dict
  }];

  let extraClassDeclaration = [{
     bool isExternal() { return !getInitialValue(); }
     bool isUninitialized() {
       return !isExternal() && ::llvm::isa<UnitAttr>(*getInitialValue());
     }
     /// Returns the constant initial value if the memref.global is a constant,
     /// or null otherwise.
     ElementsAttr getConstantInitValue();
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def LoadOp : MemRef_Op<"load",
     [TypesMatchWith<"result type matches element type of 'memref'",
                     "memref", "result",
                     "::llvm::cast<MemRefType>($_self).getElementType()">,
      MemRefsNormalizable,
      DeclareOpInterfaceMethods<PromotableMemOpInterface>,
      DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>]> {
  let summary = "load operation";
  let description = [{
    The `load` op reads an element from a memref at the specified indices.

    The number of indices must match the rank of the memref. The indices must
    be in-bounds: `0 <= idx < dim_size`

    The single result of `memref.load` is a value with the same type as the
    element type of the memref.

    A set `nontemporal` attribute indicates that this load is not expected to
    be reused in the cache. For details, refer to the
    [https://llvm.org/docs/LangRef.html#load-instruction](LLVM load instruction).

    Example:

    ```mlir
    %0 = memref.load %A[%a, %b] : memref<8x?xi32, #layout, memspace0>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                           [MemRead]>:$memref,
                       Variadic<Index>:$indices,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$nontemporal);
  let results = (outs AnyType:$result);

  let extraClassDeclaration = [{
    Value getMemRef() { return getOperand(0); }
    void setMemRef(Value value) { setOperand(0, value); }
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getMemRef().getType());
    }
  }];

  let hasFolder = 1;
  let hasVerifier = 1;

  let assemblyFormat = "$memref `[` $indices `]` attr-dict `:` type($memref)";
}

//===----------------------------------------------------------------------===//
// MemorySpaceCastOp
//===----------------------------------------------------------------------===//
def MemRef_MemorySpaceCastOp : MemRef_Op<"memory_space_cast", [
      DeclareOpInterfaceMethods<CastOpInterface>,
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
      MemRefsNormalizable,
      Pure,
      SameOperandsAndResultElementType,
      SameOperandsAndResultShape,
      ViewLikeOpInterface
    ]> {
  let summary = "memref memory space cast operation";
  let description = [{
    This operation casts memref values between memory spaces.
    The input and result will be memrefs of the same types and shape that alias
    the same underlying memory, though, for some casts on some targets,
    the underlying values of the pointer stored in the memref may be affected
    by the cast.

    The input and result must have the same shape, element type, rank, and layout.

    If the source and target address spaces are the same, this operation is a noop.

    Example:

    ```mlir
    // Cast a GPU private memory attribution into a generic pointer
    %2 = memref.memory_space_cast %1 : memref<?xf32, 5> to memref<?xf32>
    // Cast a generic pointer to workgroup-local memory
    %4 = memref.memory_space_cast %3 : memref<5x4xi32> to memref<5x34xi32, 3>
    // Cast between two non-default memory spaces
    %6 = memref.memory_space_cast %5
      : memref<*xmemref<?xf32>, 5> to memref<*xmemref<?xf32>, 3>
    ```
  }];

  let arguments = (ins AnyRankedOrUnrankedMemRef:$source);
  let results = (outs AnyRankedOrUnrankedMemRef:$dest);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";

  let extraClassDeclaration = [{
    Value getViewSource() { return getSource(); }
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// PrefetchOp
//===----------------------------------------------------------------------===//

def MemRef_PrefetchOp : MemRef_Op<"prefetch"> {
  let summary = "prefetch operation";
  let description = [{
    The "prefetch" op prefetches data from a memref location described with
    subscript indices similar to memref.load, and with three attributes: a
    read/write specifier, a locality hint, and a cache type specifier as shown
    below:

    ```mlir
    memref.prefetch %0[%i, %j], read, locality<3>, data : memref<400x400xi32>
    ```

    The read/write specifier is either 'read' or 'write', the locality hint
    ranges from locality<0> (no locality) to locality<3> (extremely local keep
    in cache). The cache type specifier is either 'data' or 'instr'
    and specifies whether the prefetch is performed on data cache or on
    instruction cache.
  }];

  let arguments = (ins AnyMemRef:$memref, Variadic<Index>:$indices,
                       BoolAttr:$isWrite,
                       ConfinedAttr<I32Attr, [IntMinValue<0>,
                                          IntMaxValue<3>]>:$localityHint,
                       BoolAttr:$isDataCache);

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getMemref().getType());
    }
    static StringRef getLocalityHintAttrStrName() { return "localityHint"; }
    static StringRef getIsWriteAttrStrName() { return "isWrite"; }
    static StringRef getIsDataCacheAttrStrName() { return "isDataCache"; }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReinterpretCastOp
//===----------------------------------------------------------------------===//

def MemRef_ReinterpretCastOp
  : MemRef_OpWithOffsetSizesAndStrides<"reinterpret_cast", [
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
      AttrSizedOperandSegments,
      MemRefsNormalizable,
      Pure,
      OffsetSizeAndStrideOpInterface,
      ViewLikeOpInterface
    ]> {
  let summary = "memref reinterpret cast operation";
  let description = [{
    Modify offset, sizes and strides of an unranked/ranked memref.

    Example:
    ```mlir
    memref.reinterpret_cast %ranked to
      offset: [0],
      sizes: [%size0, 10],
      strides: [1, %stride1]
    : memref<?x?xf32> to memref<?x10xf32, strided<[1, ?], offset: 0>>

    memref.reinterpret_cast %unranked to
      offset: [%offset],
      sizes: [%size0, %size1],
      strides: [%stride0, %stride1]
    : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    ```

    This operation creates a new memref descriptor using the base of the
    source and applying the input arguments to the other metadata.
    In other words:
    ```mlir
    %dst = memref.reinterpret_cast %src to
      offset: [%offset],
      sizes: [%sizes],
      strides: [%strides]
    ```
    means that `%dst`'s descriptor will be:
    ```mlir
    %dst.base = %src.base
    %dst.aligned = %src.aligned
    %dst.offset = %offset
    %dst.sizes = %sizes
    %dst.strides = %strides
    ```
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "", []>:$source,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_sizes,
                       DenseI64ArrayAttr:$static_strides);
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $source `to` `offset` `` `:`
    custom<DynamicIndexList>($offsets, $static_offsets)
    `` `,` `sizes` `` `:`
    custom<DynamicIndexList>($sizes, $static_sizes)
    `` `,` `strides` `` `:`
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)
  }];

  let hasVerifier = 1;

  let builders = [
    // Build a ReinterpretCastOp with mixed static and dynamic entries.
    OpBuilder<(ins "MemRefType":$resultType, "Value":$source,
      "OpFoldResult":$offset, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a ReinterpretCastOp and infer the result type.
    OpBuilder<(ins "Value":$source, "OpFoldResult":$offset,
      "ArrayRef<OpFoldResult>":$sizes, "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a ReinterpretCastOp with static entries.
    OpBuilder<(ins "MemRefType":$resultType, "Value":$source,
      "int64_t":$offset, "ArrayRef<int64_t>":$sizes,
      "ArrayRef<int64_t>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a ReinterpretCastOp with dynamic entries.
    OpBuilder<(ins "MemRefType":$resultType, "Value":$source,
      "Value":$offset, "ValueRange":$sizes,
      "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    // The result of the op is always a ranked memref.
    MemRefType getType() { return getResult().getType(); }
    Value getViewSource() { return getSource(); }

    /// Return the rank of the result type.
    unsigned getResultRank() { return getType().getRank(); }

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned resultRank = getType().getRank();
      return {1, resultRank, resultRank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

    /// Return a vector of all the static or dynamic sizes of the op, while
    /// statically inferring the sizes of the dynamic sizes, when possible.
    /// This is best effort.
    /// E.g., if `getMixedSizes` returns `[2, %dyn_size]`, but the resulting
    /// memref type is `memref<2x8xi16>`, this method will return `[2, 8]`.
    /// Similarly if the resulting memref type is `memref<2x?xi16>`, but
    /// `%dyn_size` can statically be pinned to a constant value, this
    /// constant value is returned instead of `%dyn_size`.
    SmallVector<OpFoldResult> getConstifiedMixedSizes();
    /// Similar to `getConstifiedMixedSizes` but for strides.
    SmallVector<OpFoldResult> getConstifiedMixedStrides();
    /// Similar to `getConstifiedMixedSizes` but for the offset.
    OpFoldResult getConstifiedMixedOffset();
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// RankOp
//===----------------------------------------------------------------------===//

def MemRef_RankOp : MemRef_Op<"rank", [Pure]> {
  let summary = "rank operation";
  let description = [{
    The `memref.rank` operation takes a memref operand and returns its rank.

    Example:

    ```mlir
    %0 = memref.rank %arg0 : memref<*xf32>
    %1 = memref.rank %arg1 : memref<?x?xf32>
    ```
  }];

  let arguments = (ins AnyRankedOrUnrankedMemRef:$memref);
  let results = (outs Index);

  let hasFolder = 1;
  let assemblyFormat = "$memref attr-dict `:` type($memref)";
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

def MemRef_ReshapeOp: MemRef_Op<"reshape", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure,
    ViewLikeOpInterface]>  {
  let summary = "memref reshape operation";
  let description = [{
    The `reshape` operation converts a memref from one type to an
    equivalent type with a provided shape. The data is never copied or
    modified. The source and destination types are compatible if both have the
    same element type, same number of elements, address space and identity
    layout map. The following combinations are possible:

    a. Source type is ranked or unranked. Shape argument has static size.
    Result type is ranked.

    ```mlir
    // Reshape statically-shaped memref.
    %dst = memref.reshape %src(%shape)
             : (memref<4x1xf32>, memref<1xi32>) to memref<4xf32>
    %dst0 = memref.reshape %src(%shape0)
             : (memref<4x1xf32>, memref<2xi32>) to memref<2x2xf32>
    // Flatten unranked memref.
    %dst = memref.reshape %src(%shape)
             : (memref<*xf32>, memref<1xi32>) to memref<?xf32>
    ```

    b. Source type is ranked or unranked. Shape argument has dynamic size.
    Result type is unranked.

    ```mlir
    // Reshape dynamically-shaped 1D memref.
    %dst = memref.reshape %src(%shape)
             : (memref<?xf32>, memref<?xi32>) to memref<*xf32>
    // Reshape unranked memref.
    %dst = memref.reshape %src(%shape)
             : (memref<*xf32>, memref<?xi32>) to memref<*xf32>
    ```
  }];

  let arguments = (ins AnyRankedOrUnrankedMemRef:$source,
                       Arg<MemRefRankOf<[AnySignlessInteger, Index], [1]>,
                       "dynamically-sized shape", [MemRead]>:$shape);
  let results = (outs AnyRankedOrUnrankedMemRef:$result);

  let builders = [OpBuilder<
     (ins "MemRefType":$resultType, "Value":$operand, "Value":$shape), [{
       $_state.addOperands(operand);
       $_state.addOperands(shape);
       $_state.addTypes(resultType);
     }]>];

  let extraClassDeclaration = [{
    MemRefType getType() { return ::llvm::cast<MemRefType>(getResult().getType()); }
    Value getViewSource() { return getSource(); }
  }];

  let assemblyFormat = [{
    $source `(` $shape `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExpandShapeOp / CollapseShapeOp
//===----------------------------------------------------------------------===//

class MemRef_ReassociativeReshapeOp<string mnemonic, list<Trait> traits = []> :
    MemRef_Op<mnemonic, !listconcat(traits,
      [Pure, ViewLikeOpInterface])>,
    Results<(outs AnyStridedMemRef:$result)>{

  code commonExtraClassDeclaration = [{
    SmallVector<AffineMap, 4> getReassociationMaps();

    SmallVector<ReassociationExprs, 4> getReassociationExprs();

    SmallVector<ReassociationIndices, 4> getReassociationIndices() {
      SmallVector<ReassociationIndices, 4> reassociationIndices;
      for (auto attr : getReassociation())
        reassociationIndices.push_back(llvm::to_vector<2>(
            llvm::map_range(::llvm::cast<ArrayAttr>(attr), [&](Attribute indexAttr) {
              return ::llvm::cast<IntegerAttr>(indexAttr).getInt();
            })));
      return reassociationIndices;
    };

    MemRefType getSrcType() { return ::llvm::cast<MemRefType>(getSrc().getType()); }

    MemRefType getResultType() { return ::llvm::cast<MemRefType>(getResult().getType()); }

    Value getViewSource() { return getSrc(); }
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def MemRef_ExpandShapeOp : MemRef_ReassociativeReshapeOp<"expand_shape", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>]> {
  let summary = "operation to produce a memref with a higher rank.";
  let description = [{
    The `memref.expand_shape` op produces a new view with a higher rank whose
    sizes are a reassociation of the original `view`. The operation is limited
    to such reassociations, where a dimension is expanded into one or multiple
    contiguous dimensions. Such reassociations never require additional allocs
    or copies.

    A reassociation is defined as a grouping of dimensions and is represented
    with an array of DenseI64ArrayAttr attributes.

    Example:

    ```mlir
    %r = memref.expand_shape %0 [[0, 1], [2]] output_shape [%sz0, %sz1, 32]
        : memref<?x32xf32> into memref<?x?x32xf32>
    ```

    If an op can be statically proven to be invalid (e.g, an expansion from
    `memref<10xf32>` to `memref<2x6xf32>`), it is rejected by the verifier. If
    it cannot statically be proven invalid (e.g., the full example above; it is
    unclear whether the first source dimension is divisible by 5), the op is
    accepted by the verifier. However, if the op is in fact invalid at runtime,
    the behavior is undefined.

    The source memref can be zero-ranked. In that case, the reassociation
    indices must be empty and the result shape may only consist of unit
    dimensions.

    For simplicity, this op may not be used to cast dynamicity of dimension
    sizes and/or strides. I.e., if and only if a source dimension is dynamic,
    there must be a dynamic result dimension in the corresponding reassociation
    group. Same for strides.

    The representation for the output shape supports a partially-static
    specification via attributes specified through the `static_output_shape`
    argument.  A special sentinel value `ShapedType::kDynamic` encodes that the
    corresponding entry has a dynamic value.  There must be exactly as many SSA
    inputs in `output_shape` as there are `ShapedType::kDynamic` entries in
    `static_output_shape`.

    Note: This op currently assumes that the inner strides are of the
    source/result layout map are the faster-varying ones.
  }];

  let arguments = (ins AnyStridedMemRef:$src, IndexListArrayAttr:$reassociation,
                       Variadic<Index>:$output_shape,
                       DenseI64ArrayAttr:$static_output_shape);

  let assemblyFormat = [{
    $src $reassociation `output_shape`
    custom<DynamicIndexList>($output_shape, $static_output_shape) attr-dict `:`
    type($src) `into` type($result)
  }];

  let builders = [
    // Builders using ReassociationIndices.
    OpBuilder<(ins "Type":$resultType, "Value":$src,
      "ArrayRef<ReassociationIndices>":$reassociation,
      "ArrayRef<OpFoldResult>":$outputShape)>,

    // It will infer output shape using inferOutputShape() method.
    OpBuilder<(ins "Type":$resultType, "Value":$src,
      "ArrayRef<ReassociationIndices>":$reassociation)>,

    // Builder using ReassociationExprs.
    OpBuilder<(ins "Type":$resultType, "Value":$src,
      "ArrayRef<ReassociationExprs>":$reassociation),
    [{
      auto reassociationIndices =
          convertReassociationMapsToIndices(reassociation);
      build($_builder, $_state, resultType, src, reassociationIndices);
    }]>,

    OpBuilder<(ins "Type":$resultType, "Value":$src,
      "ArrayRef<ReassociationExprs>":$reassociation,
      "ArrayRef<OpFoldResult>":$outputShape),
    [{
      auto reassociationMaps =
          convertReassociationMapsToIndices(reassociation);
      build($_builder, $_state, resultType, src, reassociationMaps,
            outputShape);
    }]>,

    // Builder that infers the result layout map. The result shape must be
    // specified. Otherwise, the op may be ambiguous. The output shape for
    // the op will be inferred using the inferOutputShape() method.
    OpBuilder<(ins "ArrayRef<int64_t>":$resultShape, "Value":$src,
               "ArrayRef<ReassociationIndices>":$reassociation)>,

    // Builder that infers the result layout map. The result shape must be
    // specified. Otherwise, the op may be ambiguous.
    OpBuilder<(ins "ArrayRef<int64_t>":$resultShape, "Value":$src,
               "ArrayRef<ReassociationIndices>":$reassociation,
               "ArrayRef<OpFoldResult>":$outputShape)>
  ];

  let extraClassDeclaration = commonExtraClassDeclaration # [{
    static FailureOr<MemRefType> computeExpandedType(
        MemRefType srcType, ArrayRef<int64_t> resultShape,
        ArrayRef<ReassociationIndices> reassociation);

    // Infer the output shape for a memref.expand_shape when it is possible
    // to do so.
    static FailureOr<SmallVector<OpFoldResult>> inferOutputShape(
        OpBuilder &b, Location loc, MemRefType expandedType,
        ArrayRef<ReassociationIndices> reassociation,
        ArrayRef<OpFoldResult> inputShape);
  }];

  let hasVerifier = 1;
}

def MemRef_CollapseShapeOp : MemRef_ReassociativeReshapeOp<"collapse_shape", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "operation to produce a memref with a smaller rank.";
  let description = [{
    The `memref.collapse_shape` op produces a new view with a smaller rank
    whose sizes are a reassociation of the original `view`. The operation is
    limited to such reassociations, where subsequent, contiguous dimensions are
    collapsed into a single dimension. Such reassociations never require
    additional allocs or copies.

    Collapsing non-contiguous dimensions is undefined behavior. When a group of
    dimensions can be statically proven to be non-contiguous, collapses of such
    groups are rejected in the verifier on a best-effort basis. In the general
    case, collapses of dynamically-sized dims with dynamic strides cannot be
    proven to be contiguous or non-contiguous due to limitations in the memref
    type.

    A reassociation is defined as a continuous grouping of dimensions and is
    represented with an array of DenseI64ArrayAttr attribute.

    Note: Only the dimensions within a reassociation group must be contiguous.
    The remaining dimensions may be non-contiguous.

    The result memref type can be zero-ranked if the source memref type is
    statically shaped with all dimensions being unit extent. In such a case, the
    reassociation indices must be empty.

    Examples:

    ```mlir
    // Dimension collapse (i, j) -> i' and k -> k'
    %1 = memref.collapse_shape %0 [[0, 1], [2]] :
        memref<?x?x?xf32, stride_spec> into memref<?x?xf32, stride_spec_2>
    ```

    For simplicity, this op may not be used to cast dynamicity of dimension
    sizes and/or strides. I.e., a result dimension must be dynamic if and only
    if at least one dimension in the corresponding reassociation group is
    dynamic. Similarly, the stride of a result dimension must be dynamic if and
    only if the corresponding start dimension in the source type is dynamic.

    Note: This op currently assumes that the inner strides are of the
    source/result layout map are the faster-varying ones.
  }];

  let arguments = (ins AnyStridedMemRef:$src, IndexListArrayAttr:$reassociation);

  let assemblyFormat = [{
    $src $reassociation attr-dict `:` type($src) `into` type($result)
  }];

  let builders = [
    // Builders for a contracting reshape whose result type is computed from
    // `src` and `reassociation`.
    OpBuilder<(ins "Value":$src,
      "ArrayRef<ReassociationIndices>":$reassociation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins "Value":$src,
      "ArrayRef<ReassociationExprs>":$reassociation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      auto reassociationMaps =
          convertReassociationMapsToIndices(reassociation);
      build($_builder, $_state, src, reassociationMaps, attrs);
    }]>,

    // Builders for a reshape whose result type is passed explicitly.
    OpBuilder<(ins "Type":$resultType, "Value":$src,
      "ArrayRef<ReassociationIndices>":$reassociation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      $_state.addAttribute("reassociation",
                          getReassociationIndicesAttribute($_builder, reassociation));
      build($_builder, $_state, resultType, src, attrs);
    }]>,
    OpBuilder<(ins "Type":$resultType, "Value":$src,
      "ArrayRef<ReassociationExprs>":$reassociation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      auto reassociationMaps =
          convertReassociationMapsToIndices(reassociation);
      build($_builder, $_state, resultType, src, reassociationMaps, attrs);
    }]>
  ];

  let extraClassDeclaration = commonExtraClassDeclaration # [{
    /// Return `true` if this source MemRef type is guaranteed to be collapsible
    /// according to the given reassociation indices. In the presence of dynamic
    /// strides this is usually not the case.
    static bool isGuaranteedCollapsible(
        MemRefType srcType, ArrayRef<ReassociationIndices> reassociation);

    static MemRefType computeCollapsedType(
        MemRefType srcType, ArrayRef<ReassociationIndices> reassociation);
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def MemRef_StoreOp : MemRef_Op<"store",
     [TypesMatchWith<"type of 'value' matches element type of 'memref'",
                     "memref", "value",
                     "::llvm::cast<MemRefType>($_self).getElementType()">,
      MemRefsNormalizable,
      DeclareOpInterfaceMethods<PromotableMemOpInterface>,
      DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>]> {
  let summary = "store operation";
  let description = [{
    The `store` op stores an element into a memref at the specified indices.
    
    The number of indices must match the rank of the memref. The indices must
    be in-bounds: `0 <= idx < dim_size`

    A set `nontemporal` attribute indicates that this store is not expected to
    be reused in the cache. For details, refer to the
    [https://llvm.org/docs/LangRef.html#store-instruction](LLVM store instruction).

    Example:

    ```mlir
    memref.store %val, %A[%a, %b] : memref<8x?xi32, #layout, memspace0>
    ```
  }];

  let arguments = (ins AnyType:$value,
                       Arg<AnyMemRef, "the reference to store to",
                           [MemWrite]>:$memref,
                       Variadic<Index>:$indices,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$nontemporal);

  let builders = [
    OpBuilder<(ins "Value":$valueToStore, "Value":$memref), [{
      $_state.addOperands(valueToStore);
      $_state.addOperands(memref);
    }]>];

  let extraClassDeclaration = [{
      Value getValueToStore() { return getOperand(0); }

      Value getMemRef() { return getOperand(1); }
      void setMemRef(Value value) { setOperand(1, value); }
      MemRefType getMemRefType() {
        return ::llvm::cast<MemRefType>(getMemRef().getType());
      }
  }];

  let hasFolder = 1;
  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $memref `[` $indices `]` attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// SubViewOp
//===----------------------------------------------------------------------===//

def SubViewOp : MemRef_OpWithOffsetSizesAndStrides<"subview", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<ViewLikeOpInterface>,
    AttrSizedOperandSegments,
    OffsetSizeAndStrideOpInterface,
    Pure
  ]> {
  let summary = "memref subview operation";
  let description = [{
    The `subview` operation converts a memref type to a memref type which
    represents a reduced-size view of the original memref as specified by the
    operation's offsets, sizes and strides arguments.

    The `subview` operation supports the following arguments:

    * source: the "base" memref on which to create a "view" memref.
    * offsets: memref-rank number of offsets into the "base" memref at which to
               create the "view" memref.
    * sizes: memref-rank number of sizes which specify the sizes of the result
             "view" memref type.
    * strides: memref-rank number of strides that compose multiplicatively with
               the base memref strides in each dimension.

    The representation based on offsets, sizes and strides support a
    partially-static specification via attributes specified through the
    `static_offsets`, `static_sizes` and `static_strides` arguments. A special
    sentinel value `ShapedType::kDynamic` encodes that the corresponding entry
    has a dynamic value.

    A `subview` operation may additionally reduce the rank of the resulting
    view by removing dimensions that are statically known to be of size 1.

    The offsets, sizes and strides must be in-bounds with respect to the source
    memref. When possible, the static operation verifier will detect
    out-of-bounds subviews. Subviews that cannot be confirmed to be in-bounds
    or out-of-bounds based on compile-time information are valid. However,
    performing an out-of-bounds subview at runtime is undefined behavior.

    Example 1:

    ```mlir
    // Subview of static memref with identity layout at dynamic offsets, sizes
    // and strides.
    %1 = memref.subview %0[%off0, %off1][%sz0, %sz1][%str0, %str1]
        : memref<64x4xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    ```

    Example 2:

    ```mlir
    // Subview of static memref with strided layout at static offsets, sizes
    // and strides.
    %1 = memref.subview %0[4, 2][8, 2][3, 2]
        : memref<64x4xf32, strided<[7, 9], offset: 91>> to
          memref<8x2xf32, strided<[21, 18], offset: 137>>
    ```

    Example 4:

    ```mlir
    // Subview of dynamic memref with strided layout at dynamic offsets and
    // strides, but static sizes.
    %1 = memref.subview %0[%off0, %off1][4, 4][%str0, %str1]
        : memref<?x?xf32, strided<[?, ?], offset: ?>> to
          memref<4x4xf32, strided<[?, ?], offset: ?>>
    ```

    Example 5:

    ```mlir
    // Rank-reducing subviews.
    %1 = memref.subview %0[0, 0, 0][1, 16, 4][1, 1, 1]
        : memref<8x16x4xf32> to memref<16x4xf32>
    %3 = memref.subview %2[3, 4, 2][1, 6, 3][1, 1, 1]
        : memref<8x16x4xf32> to memref<6x3xf32, strided<[4, 1], offset: 210>>
    ```

    Example 6:
    
    ```mlir
    // Identity subview. The subview is the full source memref.
    %1 = memref.subview %0[0, 0, 0] [8, 16, 4] [1, 1, 1]
        : memref<8x16x4xf32> to memref<8x16x4xf32>
    ```

  }];

  let arguments = (ins AnyMemRef:$source,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_sizes,
                       DenseI64ArrayAttr:$static_strides);
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)
  }];

  let builders = [
    // Build a SubViewOp with mixed static and dynamic entries and inferred
    // result type.
    OpBuilder<(ins "Value":$source, "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$sizes, "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a SubViewOp with mixed static and dynamic entries and custom
    // result type. If the type passed is nullptr, it is inferred.
    OpBuilder<(ins "MemRefType":$resultType, "Value":$source,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a SubViewOp with static entries and custom result type. If the
    // type passed is nullptr, it is inferred.
    OpBuilder<(ins "Value":$source, "ArrayRef<int64_t>":$offsets,
      "ArrayRef<int64_t>":$sizes, "ArrayRef<int64_t>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a SubViewOp with static entries and inferred result type.
    OpBuilder<(ins "MemRefType":$resultType, "Value":$source,
      "ArrayRef<int64_t>":$offsets, "ArrayRef<int64_t>":$sizes,
      "ArrayRef<int64_t>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a SubViewOp with dynamic entries and custom result type. If the
    // type passed is nullptr, it is inferred.
    OpBuilder<(ins "Value":$source, "ValueRange":$offsets,
      "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a SubViewOp with dynamic entries and inferred result type.
    OpBuilder<(ins "MemRefType":$resultType, "Value":$source,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// Returns the type of the base memref operand.
    MemRefType getSourceType() {
      return ::llvm::cast<MemRefType>(getSource().getType());
    }

    /// The result of a subview is always a memref.
    MemRefType getType() { return ::llvm::cast<MemRefType>(getResult().getType()); }

    /// A subview result type can be fully inferred from the source type and the
    /// static representation of offsets, sizes and strides. Special sentinels
    /// encode the dynamic case.
    static MemRefType inferResultType(MemRefType sourceMemRefType,
                                      ArrayRef<int64_t> staticOffsets,
                                      ArrayRef<int64_t> staticSizes,
                                      ArrayRef<int64_t> staticStrides);
    static MemRefType inferResultType(MemRefType sourceMemRefType,
                                      ArrayRef<OpFoldResult> staticOffsets,
                                      ArrayRef<OpFoldResult> staticSizes,
                                      ArrayRef<OpFoldResult> staticStrides);

    /// A rank-reducing result type can be inferred from the desired result
    /// shape. Only the layout map is inferred.
    ///
    /// Note: The result shape cannot be inferred with just the result rank and
    /// and the desired sizes. In case there are more "ones" among the sizes
    /// than the difference in source/result rank, it is not clear which dims of
    /// size one should be dropped.
    static MemRefType inferRankReducedResultType(
        ArrayRef<int64_t> resultShape, MemRefType sourceMemRefType,
        ArrayRef<int64_t> staticOffsets,
        ArrayRef<int64_t> staticSizes,
        ArrayRef<int64_t> staticStrides);
    static MemRefType inferRankReducedResultType(
        ArrayRef<int64_t> resultShape, MemRefType sourceMemRefType,
        ArrayRef<OpFoldResult> staticOffsets,
        ArrayRef<OpFoldResult> staticSizes,
        ArrayRef<OpFoldResult> staticStrides);

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSourceType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

    /// Return the dimensions of the source type that are dropped when
    /// the result is rank-reduced.
    llvm::SmallBitVector getDroppedDims();

    /// Given a `value`, asserted to be of MemRefType, build a SubViewOp that
    /// results in a rank reduction to the desired memref shape and return the
    /// new value created.
    /// If the shape of `value` is already the `desiredShape`, just return
    /// `value`.
    /// If the shape of `value` cannot be rank-reduced to `desiredShape`, fail.
    static FailureOr<Value> rankReduceIfNeeded(
      OpBuilder &b, Location loc, Value value, ArrayRef<int64_t> desiredShape);
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

def MemRef_TransposeOp : MemRef_Op<"transpose", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure]>,
    Arguments<(ins AnyStridedMemRef:$in, AffineMapAttr:$permutation)>,
    Results<(outs AnyStridedMemRef)> {
  let summary = "`transpose` produces a new strided memref (metadata-only)";
  let description = [{
    The `transpose` op produces a strided memref whose sizes and strides
    are a permutation of the original `in` memref. This is purely a metadata
    transformation.

    Example:

    ```mlir
    %1 = memref.transpose %0 (i, j) -> (j, i) : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d1 * s0 + d0)>>
    ```
  }];

  let builders = [
    OpBuilder<(ins "Value":$in, "AffineMapAttr":$permutation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];

  let extraClassDeclaration = [{
    static StringRef getPermutationAttrStrName() { return "permutation"; }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ViewOp
//===----------------------------------------------------------------------===//

def MemRef_ViewOp : MemRef_Op<"view", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<ViewLikeOpInterface>,
    Pure]> {
  let summary = "memref view operation";
  let description = [{
    The "view" operation extracts an N-D contiguous memref with empty layout map
    with arbitrary element type from a 1-D contiguous memref with empty layout
    map of i8 element  type. The ViewOp supports the following arguments:

    * A single dynamic byte-shift operand must be specified which represents a
      a shift of the base 1-D memref pointer from which to create the resulting
      contiguous memref view with identity layout.
    * A dynamic size operand that must be specified for each dynamic dimension
      in the resulting view memref type.

    The "view" operation gives a structured indexing form to a flat 1-D buffer.
    Unlike "subview" it can perform a type change. The type change behavior
    requires the op to have special semantics because, e.g. a byte shift of 3
    cannot be represented as an offset on f64.
    For now, a "view" op:

    1. Only takes a contiguous source memref with 0 offset and empty layout.
    2. Must specify a byte_shift operand (in the future, a special integer
       attribute may be added to support the folded case).
    3. Returns a contiguous memref with 0 offset and empty layout.

    Example:

    ```mlir
    // Allocate a flat 1D/i8 memref.
    %0 = memref.alloc() : memref<2048xi8>

    // ViewOp with dynamic offset and static sizes.
    %1 = memref.view %0[%offset_1024][] : memref<2048xi8> to memref<64x4xf32>

    // ViewOp with dynamic offset and two dynamic size.
    %2 = memref.view %0[%offset_1024][%size0, %size1] :
      memref<2048xi8> to memref<?x4x?xf32>
    ```
  }];

  let arguments = (ins MemRefRankOf<[I8], [1]>:$source,
                       Index:$byte_shift,
                       Variadic<Index>:$sizes);
  let results = (outs AnyMemRef);

  let extraClassDeclaration = [{
    /// The result of a view is always a memref.
    MemRefType getType() { return ::llvm::cast<MemRefType>(getResult().getType()); }

    /// Returns the dynamic sizes for this view operation. This is redundant
    /// with `sizes` but needed in template implementations. More specifically:
    /// ```
    /// template <typename AnyMemRefDefOp>
    /// bool isMemRefSizeValidSymbol(AnyMemRefDefOp memrefDefOp, unsigned index,
    ///                              Region *region)
    /// ```
    operand_range getDynamicSizes() {
      return {getSizes().begin(), getSizes().end()};
    }
  }];

  let assemblyFormat = [{
    $source `[` $byte_shift `]` `` `[` $sizes `]` attr-dict
    `:` type($source) `to` type(results)
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AtomicRMWOp
//===----------------------------------------------------------------------===//

def AtomicRMWOp : MemRef_Op<"atomic_rmw", [
      AllTypesMatch<["value", "result"]>,
      TypesMatchWith<"value type matches element type of memref",
                     "memref", "value",
                     "::llvm::cast<MemRefType>($_self).getElementType()">
    ]> {
  let summary = "atomic read-modify-write operation";
  let description = [{
    The `memref.atomic_rmw` operation provides a way to perform a read-modify-write
    sequence that is free from data races. The kind enumeration specifies the
    modification to perform. The value operand represents the new value to be
    applied during the modification. The memref operand represents the buffer
    that the read and write will be performed against, as accessed by the
    specified indices. The arity of the indices is the rank of the memref. The
    result represents the latest value that was stored.

    Example:

    ```mlir
    %x = memref.atomic_rmw "addf" %value, %I[%i] : (f32, memref<10xf32>) -> f32
    ```
  }];

  let arguments = (ins
      AtomicRMWKindAttr:$kind,
      AnyTypeOf<[AnySignlessInteger, AnyFloat]>:$value,
      Arg<MemRefOf<[AnySignlessInteger, AnyFloat]>, "the reference to read from and write to", [MemRead, MemWrite]>:$memref,
      Variadic<Index>:$indices);
  let results = (outs AnyTypeOf<[AnySignlessInteger, AnyFloat]>:$result);

  let assemblyFormat = [{
    $kind $value `,` $memref `[` $indices `]` attr-dict `:` `(` type($value) `,`
    type($memref) `)` `->` type($result)
  }];

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getMemref().getType());
    }
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

#endif // MEMREF_OPS
