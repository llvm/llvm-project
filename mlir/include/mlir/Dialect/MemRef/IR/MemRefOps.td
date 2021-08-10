//===- MemRefOps.td - MemRef op definitions ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MEMREF_OPS
#define MEMREF_OPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
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
                       Confined<I32Attr, [IntPositive]>:$alignment);
  let results = (outs);

  let assemblyFormat = "$memref `,` $alignment attr-dict `:` type($memref)";
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
}

//===----------------------------------------------------------------------===//
// AllocaScopeOp
//===----------------------------------------------------------------------===//

def MemRef_AllocaScopeOp : MemRef_Op<"alloca_scope",
      [AutomaticAllocationScope,
       DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       SingleBlockImplicitTerminator<"AllocaScopeReturnOp">,
       RecursiveSideEffects,
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
}

//===----------------------------------------------------------------------===//
// AllocaScopeReturnOp
//===----------------------------------------------------------------------===//

def MemRef_AllocaScopeReturnOp : MemRef_Op<"alloca_scope.return",
      [HasParent<"AllocaScopeOp">,
       NoSideEffect,
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

  let assemblyFormat =
      [{ attr-dict ($results^ `:` type($results))? }];

  // No custom verification needed.
  let verifier = ?;
}



//===----------------------------------------------------------------------===//
// BufferCastOp
//===----------------------------------------------------------------------===//

def MemRef_BufferCastOp : MemRef_Op<"buffer_cast",
    [SameOperandsAndResultShape, SameOperandsAndResultElementType, NoSideEffect,
     TypesMatchWith<"type of 'tensor' is the tensor equivalent of 'memref'",
                    "memref", "tensor",
                    "getTensorTypeFromMemRefType($_self)">]> {
  let summary = "tensor to memref cast operation";
  let description = [{
    Casts a tensor to a memref.

    ```mlir
    // Result type is tensor<4x?xf32>
    %12 = memref.buffer_cast %10 : memref<4x?xf32, #map0, 42>
    ```

    Note, that mutating the result of the buffer cast operation leads to
    undefined behavior.

    This operation is a specialized variant of the built-in
    unrealized_conversion_cast and is intended for use in the context of
    gradual bufferization.
  }];

  let arguments = (ins AnyTensor:$tensor);
  let results = (outs AnyRankedOrUnrankedMemRef:$memref);
  // This op is fully verified by traits.
  let verifier = ?;

  let assemblyFormat = "$tensor attr-dict `:` type($memref)";

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def MemRef_CastOp : MemRef_Op<"cast", [
      NoSideEffect, SameOperandsAndResultShape,
      DeclareOpInterfaceMethods<CastOpInterface>,
      ViewLikeOpInterface,
      MemRefsNormalizable
    ]> {
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
    OpBuilder<(ins "Value":$source, "Type":$destType), [{
       impl::buildCastOp($_builder, $_state, source, destType);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Fold the given CastOp into consumer op.
    static bool canFoldIntoConsumerOp(CastOp castOp);

    Value getViewSource() { return source(); }
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CloneOp
//===----------------------------------------------------------------------===//

def CloneOp : MemRef_Op<"clone", [
    CopyOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let builders = [
    OpBuilder<(ins "Value":$value), [{
      return build($_builder, $_state, value.getType(), value);
    }]>];

  let description = [{
    Clones the data in the input view into an implicitly defined output view.

    Usage:

    ```mlir
    %arg1 = memref.clone %arg0 : memref<?xf32> to memref<?xf32>
    ```

    Valid implementations of this operation may alias the input and output
    views or create an actual copy. Mutating the source or result
    of the clone operation after the clone operation thus leads to undefined
    behavior.
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "", []>:$input);
  let results = (outs Arg<AnyRankedOrUnrankedMemRef, "", []>:$output);

  let extraClassDeclaration = [{
    Value getSource() { return input(); }
    Value getTarget() { return output(); }
  }];

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";

  let hasFolder = 1;
  let verifier = ?;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

def CopyOp : MemRef_Op<"copy",
    [CopyOpInterface, SameOperandsElementType, SameOperandsShape]> {

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
                           [MemRead]>:$source,
                       Arg<AnyRankedOrUnrankedMemRef, "the memref to copy to",
                           [MemWrite]>:$target);

  let extraClassDeclaration = [{
    Value getSource() { return source();}
    Value getTarget() { return target(); }
  }];

  let assemblyFormat = [{
    $source `,` $target attr-dict `:` type($source) `to` type($target)
  }];

  let verifier = ?;
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

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "", [MemFree]>:$memref);

  let hasFolder = 1;
  let verifier = ?;
  let assemblyFormat = "$memref attr-dict `:` type($memref)";
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

def MemRef_DimOp : MemRef_Op<"dim", [NoSideEffect, MemRefsNormalizable]> {
  let summary = "dimension index operation";
  let description = [{
    The `dim` operation takes a memref and a dimension operand of type `index`.
    It returns the size of the requested dimension of the given memref.
    If the dimension index is out of bounds the behavior is undefined.

    The specified memref type is that of the first operand.

    Example:

    ```mlir
    // Always returns 4, can be constant folded:
    %c0 = constant 0 : index
    %x = memref.dim %A, %c0 : memref<4 x ? x f32>

    // Returns the dynamic dimension of %A.
    %c1 = constant 1 : index
    %y = memref.dim %A, %c1 : memref<4 x ? x f32>

    // Equivalent generic form:
    %x = "memref.dim"(%A, %c0) : (memref<4 x ? x f32>, index) -> index
    %y = "memref.dim"(%A, %c1) : (memref<4 x ? x f32>, index) -> index
    ```
  }];

  let arguments = (ins AnyRankedOrUnrankedMemRef:$source,
                       Index:$index);
  let results = (outs Index:$result);

  let assemblyFormat = [{
    attr-dict $source `,` $index `:` type($source)
  }];

  let builders = [
    OpBuilder<(ins "Value":$source, "int64_t":$index)>,
    OpBuilder<(ins "Value":$source, "Value":$index)>
  ];

  let extraClassDeclaration = [{
    /// Helper function to get the index as a simple integer if it is constant.
    Optional<int64_t> getConstantIndex();
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
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
// LoadOp
//===----------------------------------------------------------------------===//

def LoadOp : MemRef_Op<"load",
     [TypesMatchWith<"result type matches element type of 'memref'",
                     "memref", "result",
                     "$_self.cast<MemRefType>().getElementType()">,
                     MemRefsNormalizable]> {
  let summary = "load operation";
  let description = [{
    The `load` op reads an element from a memref specified by an index list. The
    output of load is a new value with the same type as the elements of the
    memref. The arity of indices is the rank of the memref (i.e., if the memref
    loaded from is of rank 3, then 3 indices are required for the load following
    the memref identifier).

    In an `affine.if` or `affine.for` body, the indices of a load are restricted
    to SSA values bound to surrounding loop induction variables,
    [symbols](Affine.md/#dimensions-and-symbols), results of a
    [`constant` operation](Standard.md/#stdconstant-constantop), or the result of an
    `affine.apply` operation that can in turn take as arguments all of the
    aforementioned SSA values or the recursively result of such an
    `affine.apply` operation.

    Example:

    ```mlir
    %1 = affine.apply affine_map<(d0, d1) -> (3*d0)> (%i, %j)
    %2 = affine.apply affine_map<(d0, d1) -> (d1+1)> (%i, %j)
    %12 = memref.load %A[%1, %2] : memref<8x?xi32, #layout, memspace0>

    // Example of an indirect load (treated as non-affine)
    %3 = affine.apply affine_map<(d0) -> (2*d0 + 1)>(%12)
    %13 = memref.load %A[%3, %2] : memref<4x?xi32, #layout, memspace0>
    ```

    **Context:** The `load` and `store` operations are specifically crafted to
    fully resolve a reference to an element of a memref, and (in affine
    `affine.if` and `affine.for` operations) the compiler can follow use-def
    chains (e.g. through [`affine.apply`](Affine.md/#affineapply-affineapplyop)
    operations) to precisely analyze references at compile-time using polyhedral
    techniques. This is possible because of the
    [restrictions on dimensions and symbols](Affine.md/#restrictions-on-dimensions-and-symbols)
    in these contexts.
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                           [MemRead]>:$memref,
                       Variadic<Index>:$indices);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "Value":$memref, CArg<"ValueRange", "{}">:$indices), [{
      auto memrefType = memref.getType().cast<MemRefType>();
      $_state.addOperands(memref);
      $_state.addOperands(indices);
      $_state.types.push_back(memrefType.getElementType());
    }]>];

  let extraClassDeclaration = [{
    Value getMemRef() { return getOperand(0); }
    void setMemRef(Value value) { setOperand(0, value); }
    MemRefType getMemRefType() {
      return getMemRef().getType().cast<MemRefType>();
    }

    operand_range getIndices() { return {operand_begin() + 1, operand_end()}; }
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let assemblyFormat = "$memref `[` $indices `]` attr-dict `:` type($memref)";
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
// ReinterpretCastOp
//===----------------------------------------------------------------------===//

def MemRef_ReinterpretCastOp:
    BaseOpWithOffsetSizesAndStrides<MemRef_Dialect, "reinterpret_cast", [
      NoSideEffect, AttrSizedOperandSegments, ViewLikeOpInterface,
      OffsetSizeAndStrideOpInterface, MemRefsNormalizable
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
    : memref<?x?xf32> to memref<?x10xf32, offset: 0, strides: [1, ?]>

    memref.reinterpret_cast %unranked to
      offset: [%offset],
      sizes: [%size0, %size1],
      strides: [%stride0, %stride1]
    : memref<*xf32> to memref<?x?xf32, offset: ?, strides: [?, ?]>
    ```
  }];

  let arguments = (ins
    Arg<AnyRankedOrUnrankedMemRef, "", []>:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    I64ArrayAttr:$static_offsets,
    I64ArrayAttr:$static_sizes,
    I64ArrayAttr:$static_strides
  );
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $source `to` `offset` `` `:`
    custom<OperandsOrIntegersOffsetsOrStridesList>($offsets, $static_offsets)
    `` `,` `sizes` `` `:`
    custom<OperandsOrIntegersSizesList>($sizes, $static_sizes) `` `,` `strides`
    `` `:`
    custom<OperandsOrIntegersOffsetsOrStridesList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)
  }];

  let parser = ?;
  let printer = ?;

  let builders = [
    // Build a ReinterpretCastOp with mixed static and dynamic entries.
    OpBuilder<(ins "MemRefType":$resultType, "Value":$source,
      "OpFoldResult":$offset, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
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
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
    Value getViewSource() { return source(); }

    /// Return the rank of the source ShapedType.
    unsigned getResultRank() {
      return getResult().getType().cast<ShapedType>().getRank();
    }

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned resultRank = getResult().getType().cast<ShapedType>().getRank();
      return {1, resultRank, resultRank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
  }];
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

  let builders = [OpBuilder<
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
// ExpandShapeOp / CollapseShapeOp
//===----------------------------------------------------------------------===//

def IndexListArrayAttr :
  TypedArrayAttrBase<I64ArrayAttr, "Array of 64-bit integer array attributes">;

class MemRef_ReassociativeReshapeOp<string mnemonic, list<OpTrait> traits = []> :
    MemRef_Op<mnemonic, !listconcat(traits,
      [NoSideEffect, ViewLikeOpInterface])>,
    Arguments<(ins AnyStridedMemRef:$src, IndexListArrayAttr:$reassociation)>,
    Results<(outs AnyStridedMemRef:$result)>{
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
          convertReassociationMapsToIndices($_builder, reassociation);
      build($_builder, $_state, src, reassociationMaps, attrs);
    }]>,

    // Builders for a reshape whose result type is passed explicitly. This may
    // be either a contracting or expanding reshape.
    OpBuilder<(ins "Type":$resultType, "Value":$src,
      "ArrayRef<ReassociationIndices>":$reassociation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      build($_builder, $_state, resultType, src, attrs);
      $_state.addAttribute("reassociation",
                          getReassociationIndicesAttribute($_builder, reassociation));
    }]>,
    OpBuilder<(ins "Type":$resultType, "Value":$src,
      "ArrayRef<ReassociationExprs>":$reassociation,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      auto reassociationMaps =
          convertReassociationMapsToIndices($_builder, reassociation);
      build($_builder, $_state, resultType, src, reassociationMaps, attrs);
    }]>
  ];

  code commonExtraClassDeclaration = [{
    SmallVector<AffineMap, 4> getReassociationMaps();
    SmallVector<ReassociationExprs, 4> getReassociationExprs();
    SmallVector<ReassociationIndices, 4> getReassociationIndices() {
      SmallVector<ReassociationIndices, 4> reassociationIndices;
      for (auto attr : reassociation())
        reassociationIndices.push_back(llvm::to_vector<2>(
            llvm::map_range(attr.cast<ArrayAttr>(), [&](Attribute indexAttr) {
              return indexAttr.cast<IntegerAttr>().getInt();
            })));
      return reassociationIndices;
    };
    MemRefType getSrcType() { return src().getType().cast<MemRefType>(); }
    MemRefType getResultType() { return result().getType().cast<MemRefType>(); }
    Value getViewSource() { return src(); }
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parseReshapeLikeOp(parser, result); }];
}

def MemRef_ExpandShapeOp : MemRef_ReassociativeReshapeOp<"expand_shape"> {
  let summary = "operation to produce a memref with a higher rank.";
  let description = [{
    The `memref.expand_shape` op produces a new view with a higher rank whose
    sizes are a reassociation of the original `view`. Depending on whether or
    not the reassociated MemRefType is contiguous, the resulting memref may
    require explicit alloc and copies.

    A reassociation is defined as a continuous grouping of dimensions and is
    represented with an array of I64ArrayAttr attribute.

    For now, it is assumed that either:
      1. a reassociation produces and consumes contiguous MemRefType or,
      2. the reshape op will be folded into its consumers (by changing the shape
         of the computations).
    All other cases are undefined behavior and a reshape op may not lower to
    LLVM if it cannot be proven statically that it does not require alloc+copy.

    The operand memref type when dimensions can be zero-ranked if the result
    memref type is statically shaped with all dimensions being unit extent. In
    such case the reassociation map is empty.

    The verification rule is that the reassociation maps are applied to the
    result memref with the larger rank to obtain the operand memref with the
    smaller rank.

    Example:

    ```mlir
    // Dimension expansion i -> (i', j') and (k) -> (k')
    %1 = memref.expand_shape %0 [[0, 1], [2]] :
      memref<?x?xf32, stride_spec> into memref<?x?x?xf32, stride_spec_2>
    ```
  }];
  let extraClassDeclaration = commonExtraClassDeclaration;
}

def MemRef_CollapseShapeOp : MemRef_ReassociativeReshapeOp<"collapse_shape"> {
  let summary = "operation to produce a memref with a smaller rank.";
  let description = [{
    The `memref.collapse_shape` op produces a new view with a smaller rank
    whose sizes are a reassociation of the original `view`. Depending on
    whether or not the reassociated MemRefType is contiguous, the resulting
    memref may require explicit alloc and copies.

    A reassociation is defined as a continuous grouping of dimensions and is
    represented with an array of I64ArrayAttr attribute.

    For now, it is assumed that either:
      1. a reassociation produces and consumes contiguous MemRefType or,
      2. the reshape op will be folded into its consumers (by changing the shape
         of the computations).
    All other cases are undefined behavior and a reshape op may not lower to
    LLVM if it cannot be proven statically that it does not require alloc+copy.

    The result memref type of a reshape can be zero-ranked if the operand
    memref type is statically shaped with all dimensions being unit extent. In
    such case the reassociation map is empty.

    The verification rule is that the reassociation maps are applied to the
    operand memref with the larger rank to obtain the result memref with the
    smaller rank.

    Examples:

    ```mlir
    // Dimension collapse (i, j) -> i' and k -> k'
    %1 = memref.collapse_shape %0 [[0, 1], [2]] :
      memref<?x?x?xf32, stride_spec> into memref<?x?xf32, stride_spec_2>
    ```
  }];
  let extraClassDeclaration = commonExtraClassDeclaration;
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
    [symbols](Affine.md/#restrictions-on-dimensions-and-symbols), results of a
    [`constant` operation](Standard.md/#stdconstant-constantop), or the result of an
    [`affine.apply`](Affine.md/#affineapply-affineapplyop) operation that can in
    turn take as arguments all of the aforementioned SSA values or the
    recursively result of such an `affine.apply` operation.

    Example:

    ```mlir
    memref.store %100, %A[%1, 1023] : memref<4x?xf32, #layout, memspace0>
    ```

    **Context:** The `load` and `store` operations are specifically crafted to
    fully resolve a reference to an element of a memref, and (in polyhedral
    `affine.if` and `affine.for` operations) the compiler can follow use-def
    chains (e.g. through [`affine.apply`](Affine.md/#affineapply-affineapplyop)
    operations) to precisely analyze references at compile-time using polyhedral
    techniques. This is possible because of the
    [restrictions on dimensions and symbols](Affine.md/#restrictions-on-dimensions-and-symbols)
    in these contexts.
  }];

  let arguments = (ins AnyType:$value,
                       Arg<AnyMemRef, "the reference to store to",
                           [MemWrite]>:$memref,
                       Variadic<Index>:$indices);

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
// SubViewOp
//===----------------------------------------------------------------------===//

def SubViewOp : BaseOpWithOffsetSizesAndStrides<
    MemRef_Dialect, "subview", [DeclareOpInterfaceMethods<ViewLikeOpInterface>,
                NoSideEffect, AttrSizedOperandSegments,
		OffsetSizeAndStrideOpInterface] >  {
  let summary = "memref subview operation";
  let description = [{
    The "subview" operation converts a memref type to another memref type
    which represents a reduced-size view of the original memref as specified by
    the operation's offsets, sizes and strides arguments.

    The SubView operation supports the following arguments:

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
    sentinel value ShapedType::kDynamicSize and
    ShapedType::kDynamicStrideOrOffset encodes that the corresponding entry has
    a dynamic value.

    A subview operation may additionally reduce the rank of the resulting view
    by removing dimensions that are statically known to be of size 1.

    Example 1:

    ```mlir
    %0 = memref.alloc() : memref<64x4xf32, (d0, d1) -> (d0 * 4 + d1)>

    // Create a sub-view of "base" memref '%0' with offset arguments '%c0',
    // dynamic sizes for each dimension, and stride arguments '%c1'.
    %1 = memref.subview %0[%c0, %c0][%size0, %size1][%c1, %c1]
      : memref<64x4xf32, (d0, d1) -> (d0 * 4 + d1) > to
        memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + d1 + s0)>
    ```

    Example 2:

    ```mlir
    %0 = memref.alloc() : memref<8x16x4xf32, (d0, d1, d1) -> (d0 * 64 + d1 * 4 + d2)>

    // Create a sub-view of "base" memref '%0' with dynamic offsets, sizes,
    // and strides.
    // Note that dynamic offsets are represented by the linearized dynamic
    // offset symbol 's0' in the subview memref layout map, and that the
    // dynamic strides operands, after being applied to the base memref
    // strides in each dimension, are represented in the view memref layout
    // map as symbols 's1', 's2' and 's3'.
    %1 = memref.subview %0[%i, %j, %k][%size0, %size1, %size2][%x, %y, %z]
      : memref<8x16x4xf32, (d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)> to
        memref<?x?x?xf32,
          (d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + d1 * s2 + d2 * s3 + s0)>
    ```

    Example 3:

    ```mlir
    %0 = memref.alloc() : memref<8x16x4xf32, (d0, d1, d1) -> (d0 * 64 + d1 * 4 + d2)>

    // Subview with constant offsets, sizes and strides.
    %1 = memref.subview %0[0, 2, 0][4, 4, 4][64, 4, 1]
      : memref<8x16x4xf32, (d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)> to
        memref<4x4x4xf32, (d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2 + 8)>
    ```

    Example 4:

    ```mlir
    %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>

    // Subview with constant size, but dynamic offsets and
    // strides. The resulting memref has a static shape, but if the
    // base memref has an affine map to describe the layout, the result
    // memref also uses an affine map to describe the layout. The
    // strides of the result memref is computed as follows:
    //
    // Let #map1 represents the layout of the base memref, and #map2
    // represents the layout of the result memref. A #mapsubview can be
    // constructed to map an index from the result memref to the base
    // memref (note that the description below uses more convenient
    // naming for symbols, while in affine maps, symbols are
    // represented as unsigned numbers that identify that symbol in the
    // given affine map.
    //
    // #mapsubview = (d0, d1)[o0, o1, t0, t1] -> (d0 * t0 + o0, d1 * t1 + o1)
    //
    // where, o0, o1, ... are offsets, and t0, t1, ... are strides. Then,
    //
    // #map2 = #map1.compose(#mapsubview)
    //
    // If the layout map is represented as
    //
    // #map1 = (d0, d1)[s0, s1, s2] -> (d0 * s1 + d1 * s2 + s0)
    //
    // then,
    //
    // #map2 = (d0, d1)[s0, s1, s2, o0, o1, t0, t1] ->
    //              (d0 * s1 * t0 + d1 * s2 * t1 + o0 * s1 + o1 * s2 + s0)
    //
    // Representing this canonically
    //
    // #map2 = (d0, d1)[r0, r1, r2] -> (d0 * r1 + d1 * r2 + r0)
    //
    // where, r0 = o0 * s1 + o1 * s2 + s0, r1 = s1 * t0, r2 = s2 * t1.
    %1 = memref.subview %0[%i, %j][4, 4][%x, %y] :
      : memref<?x?xf32, (d0, d1)[s0, s1, s2] -> (d0 * s1 + d1 * s2 + s0)> to
        memref<4x4xf32, (d0, d1)[r0, r1, r2] -> (d0 * r1 + d1 * r2 + r0)>

    // Note that the subview op does not guarantee that the result
    // memref is "inbounds" w.r.t to base memref. It is upto the client
    // to ensure that the subview is accessed in a manner that is
    // in-bounds.
    ```

    Example 5:

    ```mlir
    // Rank-reducing subview.
    %1 = memref.subview %0[0, 0, 0][1, 16, 4][1, 1, 1] :
      memref<8x16x4xf32> to memref<16x4xf32>
    %3 = memref.subview %2[3, 4, 2][1, 6, 3][1, 1, 1] :
      memref<8x16x4xf32> to memref<6x3xf32, offset: 210, strides: [4, 1]>
    ```
    }
  }];

  let arguments = (ins
    AnyMemRef:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    I64ArrayAttr:$static_offsets,
    I64ArrayAttr:$static_sizes,
    I64ArrayAttr:$static_strides
  );
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $source ``
    custom<OperandsOrIntegersOffsetsOrStridesList>($offsets, $static_offsets)
    custom<OperandsOrIntegersSizesList>($sizes, $static_sizes)
    custom<OperandsOrIntegersOffsetsOrStridesList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)
  }];

  let builders = [
    // Build a SubViewOp with mixed static and dynamic entries and custom
    // result type. If the type passed is nullptr, it is inferred.
    OpBuilder<(ins "Value":$source, "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$sizes, "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a SubViewOp with mixed static and dynamic entries and inferred
    // result type.
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
      return source().getType().cast<MemRefType>();
    }

    /// The result of a subview is always a memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

    /// A subview result type can be fully inferred from the source type and the
    /// static representation of offsets, sizes and strides. Special sentinels
    /// encode the dynamic case.
    static Type inferResultType(MemRefType sourceMemRefType,
                                ArrayRef<int64_t> staticOffsets,
                                ArrayRef<int64_t> staticSizes,
                                ArrayRef<int64_t> staticStrides);
    static Type inferResultType(MemRefType sourceMemRefType,
                                ArrayRef<OpFoldResult> staticOffsets,
                                ArrayRef<OpFoldResult> staticSizes,
                                ArrayRef<OpFoldResult> staticStrides);
    static Type inferRankReducedResultType(unsigned resultRank,
                                           MemRefType sourceMemRefType,
                                           ArrayRef<int64_t> staticOffsets,
                                           ArrayRef<int64_t> staticSizes,
                                           ArrayRef<int64_t> staticStrides);
    static Type inferRankReducedResultType(unsigned resultRank,
                                           MemRefType sourceMemRefType,
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
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TensorLoadOp
//===----------------------------------------------------------------------===//

def TensorLoadOp : MemRef_Op<"tensor_load",
    [SameOperandsAndResultShape, SameOperandsAndResultElementType,
     TypesMatchWith<"result type matches tensor equivalent of 'memref'",
                    "memref", "result",
                    "getTensorTypeFromMemRefType($_self)">]> {
  let summary = "tensor load operation";
  let description = [{
    Create a tensor from a memref, making an independent copy of the element
    data. The result value is a tensor whose shape and element type match the
    memref operand.

    The opposite of this op is buffer_cast. Together, these two ops are
    useful for source/target materializations when doing type conversions
    involving tensors and memrefs.

    Example:

    ```mlir
    // Produces a value of tensor<4x?xf32> type.
    %12 = memref.tensor_load %10 : memref<4x?xf32, #layout, memspace0>
    ```

    If tensor load is used in the bufferization steps, mutating the source
    buffer after loading leads to undefined behavior.
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef,
                       "the reference to load from", [MemRead]>:$memref);
  let results = (outs AnyTensor:$result);
  // TensorLoadOp is fully verified by traits.
  let verifier = ?;

  let builders = [
    OpBuilder<(ins "Value":$memref), [{
      $_state.addOperands(memref);
      $_state.addTypes(getTensorTypeFromMemRefType(memref.getType()));
    }]>];

  let extraClassDeclaration = [{
    /// The result of a tensor_load is always a tensor.
    TensorType getType() {
      Type resultType = getResult().getType();
      if (resultType.isa<TensorType>())
        return resultType.cast<TensorType>();
      return {};
    }
  }];

  let assemblyFormat = "$memref attr-dict `:` type($memref)";

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TensorStoreOp
//===----------------------------------------------------------------------===//

def TensorStoreOp : MemRef_Op<"tensor_store",
    [SameOperandsShape, SameOperandsElementType,
     TypesMatchWith<"type of 'value' matches tensor equivalent of 'memref'",
                    "memref", "tensor",
                    "getTensorTypeFromMemRefType($_self)">]> {
  let summary = "tensor store operation";
  let description = [{
    Stores the contents of a tensor into a memref. The first operand is a value
    of tensor type, the second operand is a value of memref type. The shapes and
    element types of these must match, and are specified by the memref type.

    Example:

    ```mlir
    %9 = dim %8, 1 : tensor<4x?xf32>
    %10 = alloc(%9) : memref<4x?xf32, #layout, memspace0>
    memref.tensor_store %8, %10 : memref<4x?xf32, #layout, memspace0>
    ```
  }];

  let arguments = (ins AnyTensor:$tensor, Arg<AnyRankedOrUnrankedMemRef,
                       "the reference to store to", [MemWrite]>:$memref);
  // TensorStoreOp is fully verified by traits.
  let verifier = ?;

  let assemblyFormat = "$tensor `,` $memref attr-dict `:` type($memref)";
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
    OpBuilder<(ins "Value":$in, "AffineMapAttr":$permutation,
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
