//===- MemRefOps.td - MemRef op definitions ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MEMREF_OPS
#define MEMREF_OPS

include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class MemRef_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<MemRef_Dialect, mnemonic, traits> {
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

//===----------------------------------------------------------------------===//
// AllocLikeOp
//===----------------------------------------------------------------------===//

// Base class for memref allocating ops: alloca and alloc.
//
//   %0 = alloclike(%m)[%s] : memref<8x?xf32, (d0, d1)[s0] -> ((d0 + s0), d1)>
//
class AllocLikeOp<string mnemonic,
                  Resource resource,
                  list<OpTrait> traits = []> :
    MemRef_Op<mnemonic,
    !listconcat([
      AttrSizedOperandSegments
    ], traits)> {

  let arguments = (ins Variadic<Index>:$dynamicSizes,
                   // The symbolic operands (the ones in square brackets) bind
                   // to the symbols of the memref's layout map.
                   Variadic<Index>:$symbolOperands,
                   Confined<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$alignment);
  let results = (outs Res<AnyMemRef, "", [MemAlloc<resource>]>:$memref);

  let builders = [
    OpBuilderDAG<(ins "MemRefType":$memrefType,
                  CArg<"IntegerAttr", "IntegerAttr()">:$alignment), [{
      return build($_builder, $_state, memrefType, {}, alignment);
    }]>,
    OpBuilderDAG<(ins "MemRefType":$memrefType, "ValueRange":$dynamicSizes,
                  CArg<"IntegerAttr", "IntegerAttr()">:$alignment), [{
      return build($_builder, $_state, memrefType, dynamicSizes, {}, alignment);
    }]>,
    OpBuilderDAG<(ins "MemRefType":$memrefType, "ValueRange":$dynamicSizes,
                  "ValueRange":$symbolOperands,
                  CArg<"IntegerAttr", "{}">:$alignment), [{
      $_state.types.push_back(memrefType);
      $_state.addOperands(dynamicSizes);
      $_state.addOperands(symbolOperands);
      $_state.addAttribute(getOperandSegmentSizeAttr(),
          $_builder.getI32VectorAttr({
              static_cast<int32_t>(dynamicSizes.size()),
              static_cast<int32_t>(symbolOperands.size())}));
      if (alignment)
        $_state.addAttribute(getAlignmentAttrName(), alignment);
    }]>];

  let extraClassDeclaration = [{
    static StringRef getAlignmentAttrName() { return "alignment"; }

    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

    /// Returns the dynamic sizes for this alloc operation if specified.
    operand_range getDynamicSizes() { return dynamicSizes(); }
  }];

  let assemblyFormat = [{
    `(`$dynamicSizes`)` (`` `[` $symbolOperands^ `]`)? attr-dict `:` type($memref)
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// BaseCastOp
//===----------------------------------------------------------------------===//

// Base class for standard cast operations. Requires single operand and result,
// but does not constrain them to specific types.
class CastOp<string mnemonic, list<OpTrait> traits = []> :
    MemRef_Op<mnemonic, traits # [
      NoSideEffect, SameOperandsAndResultShape,
      DeclareOpInterfaceMethods<CastOpInterface>
    ]> {
  let results = (outs AnyType);

  let builders = [
    OpBuilderDAG<(ins "Value":$source, "Type":$destType), [{
       impl::buildCastOp($_builder, $_state, source, destType);
    }]>
  ];

  let parser = [{
    return impl::parseCastOp(parser, result);
  }];
  let printer = [{
    return printStandardCastOp(this->getOperation(), p);
  }];

  // Cast operations are fully verified by its traits.
  let verifier = ?;
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

def MemRef_AllocOp : AllocLikeOp<"alloc", DefaultResource> {
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
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

def MemRef_AllocaOp : AllocLikeOp<"alloca", AutomaticAllocationScopeResource> {
  let summary = "stack memory allocation operation";
  let description = [{
    The `alloca` operation allocates memory on the stack, to be automatically
    released when control transfers back from the region of its closest
    surrounding operation with an
    [`AutomaticAllocationScope`](../Traits.md#automaticallocationscope) trait.
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
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def MemRef_CastOp : CastOp<"cast", [NoSideEffect]> {
  let summary = "memref cast operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `memref.cast` ssa-use `:` type `to` type
    ```

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
    %4 = memref.cast %1 : memref<12x4xf32, offset:?, strides: [?, ?]> to
                          memref<12x4xf32, offset:5, strides: [4, 1]>
    // Erase static offset and stride information, replacing it with
    // dynamic information.
    %5 = memref.cast %1 : memref<12x4xf32, offset:5, strides: [4, 1]> to
                          memref<12x4xf32, offset:?, strides: [?, ?]>
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
  let verifier = "return impl::verifyCastOp(*this, areCastCompatible);";
  let builders = [
    OpBuilderDAG<(ins "Value":$source, "Type":$destType), [{
       impl::buildCastOp($_builder, $_state, source, destType);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Fold the given CastOp into consumer op.
    static bool canFoldIntoConsumerOp(CastOp castOp);
  }];

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
    %0 = memref.alloc() : memref<8x64xf32, (d0, d1) -> (d0, d1), 1>
    memref.dealloc %0 : memref<8x64xf32, (d0, d1) -> (d0, d1), 1>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "", [MemFree]>:$memref);

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let assemblyFormat = "$memref attr-dict `:` type($memref)";
}

//===----------------------------------------------------------------------===//
// GetGlobalOp
//===----------------------------------------------------------------------===//

def MemRef_GetGlobalOp : MemRef_Op<"get_global",
    [NoSideEffect, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
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

  // `GetGlobalOp` is fully verified by its traits.
  let verifier = ?;
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

def MemRef_GlobalOp : MemRef_Op<"global", [Symbol]> {
  let summary = "declare or define a global memref variable";
  let description = [{
    The `memref.global` operation declares or defines a named global variable.
    The backing memory for the variable is allocated statically and is described
    by the type of the variable (which should be a statically shaped memref
    type). The operation is a declaration if no `inital_value` is specified,
    else it is a definition. The `initial_value` can either be a unit attribute
    to represent a definition of an uninitialized global variable, or an
    elements attribute to represent the definition of a global variable with an
    initial value. The global variable can also be marked constant using the
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

    // Declaration of an external variable.
    memref.global "private" @y : memref<4xi32>

    // Uninitialized externally visible variable.
    memref.global @z : memref<3xf16> = uninitialized

    // Externally visible constant variable.
    memref.global constant @c : memref<2xi32> = dense<1, 4>
    ```
  }];

  let arguments = (ins
      SymbolNameAttr:$sym_name,
      OptionalAttr<StrAttr>:$sym_visibility,
      TypeAttr:$type,
      OptionalAttr<AnyAttr>:$initial_value,
      UnitAttr:$constant
  );

  let assemblyFormat = [{
       ($sym_visibility^)?
       (`constant` $constant^)?
       $sym_name `:`
       custom<GlobalMemrefOpTypeAndInitialValue>($type, $initial_value)
       attr-dict
  }];

  let extraClassDeclaration = [{
     bool isExternal() { return !initial_value(); }
     bool isUninitialized() {
       return !isExternal() && initial_value().getValue().isa<UnitAttr>();
     }
  }];
}

//===----------------------------------------------------------------------===//
// PrefetchOp
//===----------------------------------------------------------------------===//

def MemRef_PrefetchOp : MemRef_Op<"prefetch"> {
  let summary = "prefetch operation";
  let description = [{
    The "prefetch" op prefetches data from a memref location described with
    subscript indices similar to std.load, and with three attributes: a
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
                   Confined<I32Attr, [IntMinValue<0>,
                     IntMaxValue<3>]>:$localityHint,
                   BoolAttr:$isDataCache);

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return memref().getType().cast<MemRefType>();
    }
    static StringRef getLocalityHintAttrName() { return "localityHint"; }
    static StringRef getIsWriteAttrName() { return "isWrite"; }
    static StringRef getIsDataCacheAttrName() { return "isDataCache"; }
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

def MemRef_ReshapeOp: MemRef_Op<"reshape", [
    ViewLikeOpInterface, NoSideEffect]>  {
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

  let arguments = (ins
    AnyRankedOrUnrankedMemRef:$source,
    MemRefRankOf<[AnySignlessInteger, Index], [1]>:$shape
  );
  let results = (outs AnyRankedOrUnrankedMemRef:$result);

  let builders = [OpBuilderDAG<
     (ins "MemRefType":$resultType, "Value":$operand, "Value":$shape), [{
       $_state.addOperands(operand);
       $_state.addOperands(shape);
       $_state.addTypes(resultType);
     }]>];

  let extraClassDeclaration = [{
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
    Value getViewSource() { return source(); }
  }];

  let assemblyFormat = [{
    $source `(` $shape `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def MemRef_StoreOp : MemRef_Op<"store",
     [TypesMatchWith<"type of 'value' matches element type of 'memref'",
                     "memref", "value",
                     "$_self.cast<MemRefType>().getElementType()">,
                     MemRefsNormalizable]> {
  let summary = "store operation";
  let description = [{
    Store a value to a memref location given by indices. The value stored should
    have the same type as the elemental type of the memref. The number of
    arguments provided within brackets need to match the rank of the memref.

    In an affine context, the indices of a store are restricted to SSA values
    bound to surrounding loop induction variables,
    [symbols](Affine.md#restrictions-on-dimensions-and-symbols), results of a
    [`constant` operation](#stdconstant-constantop), or the result of an
    [`affine.apply`](Affine.md#affineapply-affineapplyop) operation that can in
    turn take as arguments all of the aforementioned SSA values or the
    recursively result of such an `affine.apply` operation.

    Example:

    ```mlir
    memref.store %100, %A[%1, 1023] : memref<4x?xf32, #layout, memspace0>
    ```

    **Context:** The `load` and `store` operations are specifically crafted to
    fully resolve a reference to an element of a memref, and (in polyhedral
    `affine.if` and `affine.for` operations) the compiler can follow use-def
    chains (e.g. through [`affine.apply`](Affine.md#affineapply-affineapplyop)
    operations) to precisely analyze references at compile-time using polyhedral
    techniques. This is possible because of the
    [restrictions on dimensions and symbols](Affine.md#restrictions-on-dimensions-and-symbols)
    in these contexts.
  }];

  let arguments = (ins AnyType:$value,
                       Arg<AnyMemRef, "the reference to store to",
                           [MemWrite]>:$memref,
                       Variadic<Index>:$indices);

  let builders = [
    OpBuilderDAG<(ins "Value":$valueToStore, "Value":$memref), [{
      $_state.addOperands(valueToStore);
      $_state.addOperands(memref);
    }]>];

  let extraClassDeclaration = [{
      Value getValueToStore() { return getOperand(0); }

      Value getMemRef() { return getOperand(1); }
      void setMemRef(Value value) { setOperand(1, value); }
      MemRefType getMemRefType() {
        return getMemRef().getType().cast<MemRefType>();
      }

      operand_range getIndices() {
        return {operand_begin() + 2, operand_end()};
      }
  }];

  let hasFolder = 1;

  let assemblyFormat = [{
    $value `,` $memref `[` $indices `]` attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

def MemRef_TransposeOp : MemRef_Op<"transpose", [NoSideEffect]>,
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
    OpBuilderDAG<(ins "Value":$in, "AffineMapAttr":$permutation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];

  let extraClassDeclaration = [{
    static StringRef getPermutationAttrName() { return "permutation"; }
    ShapedType getShapedType() { return in().getType().cast<ShapedType>(); }
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ViewOp
//===----------------------------------------------------------------------===//

def MemRef_ViewOp : MemRef_Op<"view", [
    DeclareOpInterfaceMethods<ViewLikeOpInterface>, NoSideEffect]> {
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
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

    /// Returns the dynamic sizes for this view operation. This is redundant
    /// with `sizes` but needed in template implementations. More specifically:
    /// ```
    /// template <typename AnyMemRefDefOp>
    /// bool isMemRefSizeValidSymbol(AnyMemRefDefOp memrefDefOp, unsigned index,
    ///                              Region *region)
    /// ```
    operand_range getDynamicSizes() {
      return {sizes().begin(), sizes().end()};
    }
  }];

  let hasCanonicalizer = 1;
}

#endif // MEMREF_OPS
