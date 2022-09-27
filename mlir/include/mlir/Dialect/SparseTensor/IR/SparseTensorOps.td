//===- SparseTensorOps.td - Sparse tensor dialect ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPARSETENSOR_OPS
#define SPARSETENSOR_OPS

include "mlir/Dialect/SparseTensor/IR/SparseTensorAttrDefs.td"
include "mlir/Dialect/SparseTensor/IR/SparseTensorBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class SparseTensor_Op<string mnemonic, list<Trait> traits = []>
  : Op<SparseTensor_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Sparse Tensor Operations.
//===----------------------------------------------------------------------===//

def SparseTensor_NewOp : SparseTensor_Op<"new", [NoSideEffect]>,
    Arguments<(ins AnyType:$source)>,
    Results<(outs AnySparseTensor:$result)> {
  string summary = "Materializes a new sparse tensor from given source";
  string description = [{
    Materializes a sparse tensor with contents taken from an opaque pointer
    provided by `source`. For targets that have access to a file system,
    for example, this pointer may be a filename (or file) of a sparse
    tensor in a particular external storage format. The form of the operation
    is kept deliberately very general to allow for alternative implementations
    in the future, such as pointers to buffers or runnable initialization
    code. The operation is provided as an anchor that materializes a properly
    typed sparse tensor with inital contents into a computation.

    Example:

    ```mlir
    sparse_tensor.new %source : !Source to tensor<1024x1024xf64, #CSR>
    ```
  }];
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
}

def SparseTensor_ConvertOp : SparseTensor_Op<"convert",
  [NoSideEffect, SameOperandsAndResultElementType]>,
    Arguments<(ins AnyTensor:$source)>,
    Results<(outs AnyTensor:$dest)> {
  string summary = "Converts between different tensor types";
  string description = [{
    Converts one sparse or dense tensor type to another tensor type. The rank
    of the source and destination types must match exactly, and the dimension
    sizes must either match exactly or relax from a static to a dynamic size.
    The sparse encoding of the two types can obviously be completely different.
    The name `convert` was preferred over `cast`, since the operation may incur
    a non-trivial cost.

    When converting between two different sparse tensor types, only explicitly
    stored values are moved from one underlying sparse storage format to
    the other. When converting from an unannotated dense tensor type to a
    sparse tensor type, an explicit test for nonzero values is used. When
    converting to an unannotated dense tensor type, implicit zeroes in the
    sparse storage format are made explicit. Note that the conversions can have
    non-trivial costs associated with them, since they may involve elaborate
    data structure transformations. Also, conversions from sparse tensor types
    into dense tensor types may be infeasible in terms of storage requirements.

    Examples:

    ```mlir
    %0 = sparse_tensor.convert %a : tensor<32x32xf32> to tensor<32x32xf32, #CSR>
    %1 = sparse_tensor.convert %a : tensor<32x32xf32> to tensor<?x?xf32, #CSR>
    %2 = sparse_tensor.convert %b : tensor<8x8xi32, #CSC> to tensor<8x8xi32, #CSR>
    %3 = sparse_tensor.convert %c : tensor<4x8xf64, #CSR> to tensor<4x?xf64, #CSC>

    // The following conversion is not allowed (since it would require a
    // runtime assertion that the source's dimension size is actually 100).
    %4 = sparse_tensor.convert %d : tensor<?xf64> to tensor<100xf64, #SV>
    ```

  }];
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def SparseTensor_ToPointersOp : SparseTensor_Op<"pointers", [NoSideEffect]>,
    Arguments<(ins AnySparseTensor:$tensor, IndexAttr:$dimension)>,
    Results<(outs AnyStridedMemRefOfRank<1>:$result)> {
  let summary = "Extracts pointers array at given dimension from a tensor";
  let description = [{
    Returns the pointers array of the sparse storage format at the
    given dimension for the given sparse tensor. This is similar to the
    `bufferization.to_memref` operation in the sense that it provides a bridge
    between a tensor world view and a bufferized world view. Unlike the
    `bufferization.to_memref` operation, however, this sparse operation actually
    lowers into code that extracts the pointers array from the sparse storage
    scheme (either by calling a support library or through direct code).

    Example:

    ```mlir
    %1 = sparse_tensor.pointers %0 { dimension = 1 : index }
       : tensor<64x64xf64, #CSR> to memref<?xindex>
    ```
  }];
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($result)";
  let hasVerifier = 1;
}

def SparseTensor_ToIndicesOp : SparseTensor_Op<"indices", [NoSideEffect]>,
    Arguments<(ins AnySparseTensor:$tensor, IndexAttr:$dimension)>,
    Results<(outs AnyStridedMemRefOfRank<1>:$result)> {
  let summary = "Extracts indices array at given dimension from a tensor";
  let description = [{
    Returns the indices array of the sparse storage format at the
    given dimension for the given sparse tensor. This is similar to the
    `bufferization.to_memref` operation in the sense that it provides a bridge
    between a tensor world view and a bufferized world view. Unlike the
    `bufferization.to_memref` operation, however, this sparse operation actually
    lowers into code that extracts the indices array from the sparse storage
    scheme (either by calling a support library or through direct code).

    Example:

    ```mlir
    %1 = sparse_tensor.indices %0 { dimension = 1 : index }
       : tensor<64x64xf64, #CSR> to memref<?xindex>
    ```
  }];
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($result)";
  let hasVerifier = 1;
}

def SparseTensor_ToValuesOp : SparseTensor_Op<"values", [NoSideEffect]>,
    Arguments<(ins AnySparseTensor:$tensor)>,
    Results<(outs AnyStridedMemRefOfRank<1>:$result)> {
  let summary = "Extracts numerical values array from a tensor";
  let description = [{
    Returns the values array of the sparse storage format for the given
    sparse tensor, independent of the actual dimension. This is similar to
    the `bufferization.to_memref` operation in the sense that it provides a bridge
    between a tensor world view and a bufferized world view. Unlike the
    `bufferization.to_memref` operation, however, this sparse operation actually
    lowers into code that extracts the values array from the sparse storage
    scheme (either by calling a support library or through direct code).

    Example:

    ```mlir
    %1 = sparse_tensor.values %0 : tensor<64x64xf64, #CSR> to memref<?xf64>
    ```
  }];
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($result)";
  let hasVerifier = 1;
}

def SparseTensor_ConcatenateOp : SparseTensor_Op<"concatenate", [NoSideEffect]>,
    Arguments<(ins Variadic<AnyRankedTensor>:$inputs, IndexAttr:$dimension)>,
    Results<(outs AnyRankedTensor:$result)> {

  let summary = "Concatenates a list of tensors into a single tensor.";
  let description = [{
     Concatenates a list input tensors and the output tensor with the same rank.
     The concatenation happens on the specified `dimension` (0<= dimension < rank).
     The resulting `dimension` size is the sum of all the input dimension sizes,
     while all the other dimensions should have the same size in the input and
     output tensors.

     Only statically-sized input tensors are accepted, while the output tensor
     can be dynamically-sized.

     Example:

     ```mlir
     %0 = sparse_tensor.concatenate %1, %2 { dimension = 0 : index }
       : tensor<64x64xf64, #CSR>, tensor<64x64xf64, #CSR> to tensor<128x64xf64, #CSR>
     ```
   }];

  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `to` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Sparse Tensor Management Operations. These operations are "impure" in the
// sense that they do not properly operate on SSA values. Instead, the behavior
// is solely defined by side-effects. These operations provide a bridge between
// "sparsification" on one hand and a support library or actual code generation
// on the other hand. The semantics of these operations may be refined over time
// as our sparse abstractions evolve.
//===----------------------------------------------------------------------===//

def SparseTensor_InsertOp : SparseTensor_Op<"insert", []>,
    Arguments<(ins AnySparseTensor:$tensor,
               StridedMemRefRankOf<[Index], [1]>:$indices,
               AnyType:$value)> {
  string summary = "Inserts a value into given sparse tensor";
  string description = [{
    Inserts the given value at given indices into the underlying sparse
    storage format of the given tensor with the given indices. This
    operation can only be applied when a tensor materializes unintialized
    with a `bufferization.alloc_tensor` operation and the final tensor
    is constructed with a `load` operation that has the `hasInserts`
    attribute set.

    Properties in the sparse tensor type fully describe what kind
    of insertion order is allowed. When all dimensions have "unique"
    and "ordered" properties, for example, insertions should occur in
    strict lexicographical index order. Other properties define
    different insertion regimens. Inserting in a way contrary to
    these properties results in undefined behavior.

    Note that this operation is "impure" in the sense that its behavior
    is solely defined by side-effects and not SSA values. The semantics
    may be refined over time as our sparse abstractions evolve.

    Example:

    ```mlir
    sparse_tensor.insert %tensor, %indices, %val
      : tensor<1024x1024xf64, #CSR>, memref<?xindex>, memref<f64>
    ```
  }];
  let assemblyFormat = "$tensor `,` $indices `,` $value attr-dict `:`"
                       " type($tensor) `,` type($indices) `,` type($value)";
}

def SparseTensor_PushBackOp : SparseTensor_Op<"push_back", []>,
    Arguments<(ins StridedMemRefRankOf<[Index], [1]>:$bufferSizes,
               StridedMemRefRankOf<[AnyType], [1]>:$inBuffer,
               AnyType:$value, IndexAttr:$idx)>,
    Results<(outs StridedMemRefRankOf<[AnyType], [1]>:$outBuffer)>  {
  string summary = "Pushes a value to the back of a given buffer";
  string description = [{
    Push `value` to the end of the given sparse tensor storage buffer
    `inBuffer` and update the size of the buffer in `bufferSizes[idx]`. The
    capacity of the buffer is recorded in the memref type of `inBuffer `. If the
    current buffer is full, then `inBuffer.realloc` is called before pushing the
    data to the buffer. This is similar to std::vector push_back.

    The operation returns an SSA value for the memref. Referencing the memref
    through the old SSA value after this operation is undefined behavior.

    Example:

    ```mlir
    %r = sparse_tensor.push_back %bufferSizes, %buffer, %val {idx = 0 : index}
      : memref<?xindex>, memref<?xf64>, f64 -> memref<?xf64>
    ```
  }];
  let assemblyFormat = "$bufferSizes `,` $inBuffer `,` $value"
                       " attr-dict `:` type($bufferSizes) `,`"
                       " type($inBuffer) `,` type($value) `to`"
                       " type($outBuffer)";
}

def SparseTensor_ExpandOp : SparseTensor_Op<"expand", []>,
    Arguments<(ins AnySparseTensor:$tensor)>,
    Results<(outs AnyStridedMemRefOfRank<1>:$values,
                  StridedMemRefRankOf<[I1],[1]>:$filled,
                  StridedMemRefRankOf<[Index],[1]>:$added,
                  Index:$count)> {
  string summary = "Expands an access pattern for insertion";
  string description = [{
    Performs an access pattern expansion for the innermost dimensions of the
    given tensor. This operation is useful to implement kernels in which a
    sparse tensor appears as output. This technique is known under several
    different names and using several alternative implementations,
    for example, phase counter [Gustavson72], expanded or switch array
    [Pissanetzky84], in phase scan [Duff90], access pattern expansion [Bik96],
    and workspaces [Kjolstad19].

    The values and filled array have sizes that suffice for a *dense* innermost
    dimension (e.g. a full row for matrices). The added array and count are used
    to store new indices when a false value is encountered in the filled array.
    All arrays should be allocated before the loop (possibly even shared between
    loops in a future optimization) so that their *dense* initialization can be
    amortized over many iterations. Setting and resetting the dense arrays in
    the loop nest itself is kept *sparse* by only iterating over set elements
    through an indirection using the added array, so that the operations are
    kept proportional to the number of nonzeros.

    Note that this operation is "impure" in the sense that its behavior
    is solely defined by side-effects and not SSA values. The semantics
    may be refined over time as our sparse abstractions evolve.

    Example:

    ```mlir
    %values, %filled, %added, %count = sparse_tensor.expand %0
      : tensor<4x4xf64, #CSR> to memref<?xf64>, memref<?xi1>, memref<?xindex>, index
    ```
  }];
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($values)"
                       " `,` type($filled) `,` type($added) `,` type($count)";
}

def SparseTensor_CompressOp : SparseTensor_Op<"compress", []>,
    Arguments<(ins AnySparseTensor:$tensor,
                   StridedMemRefRankOf<[Index],[1]>:$indices,
                   AnyStridedMemRefOfRank<1>:$values,
                   StridedMemRefRankOf<[I1],[1]>:$filled,
                   StridedMemRefRankOf<[Index],[1]>:$added,
                   Index:$count)> {
  string summary = "Compressed an access pattern for insertion";
  string description = [{
    Finishes a single access pattern expansion by moving inserted elements
    into the sparse storage scheme. The values and filled array are reset
    in a *sparse* fashion by only iterating over set elements through an
    indirection using the added array, so that the operations are kept
    proportional to the number of nonzeros. See the 'expand' operation
    for more details.

    Note that this operation is "impure" in the sense that its behavior
    is solely defined by side-effects and not SSA values. The semantics
    may be refined over time as our sparse abstractions evolve.

    Example:

    ```mlir
    sparse_tensor.compress %0, %1, %values, %filled, %added, %2
        : tensor<4x4xf64, #CSR>, memref<?xindex>, memref<?xf64>,
          memref<?xi1>, memref<?xindex>, index
    ```
  }];
  let assemblyFormat = "$tensor `,` $indices `,` $values `,` $filled `,`"
                       " $added `,` $count attr-dict `:` type($tensor) `,`"
                       " type($indices) `,` type($values) `,` type($filled) `,`"
                       " type($added) `,` type($count)";
}

def SparseTensor_LoadOp : SparseTensor_Op<"load", [SameOperandsAndResultType]>,
    Arguments<(ins AnySparseTensor:$tensor, UnitAttr:$hasInserts)>,
    Results<(outs AnyTensor:$result)> {
  let summary =
    "Rematerializes tensor from underlying sparse storage format";
  let description = [{
    Rematerializes a tensor from the underlying sparse storage format of the
    given tensor. This is similar to the `bufferization.to_tensor` operation
    in the sense that it provides a bridge between a bufferized world view
    and a tensor world view. Unlike the `bufferization.to_tensor` operation,
    however, this sparse operation is used only temporarily to maintain a
    correctly typed intermediate representation during progressive
    bufferization.

    The `hasInserts` attribute denote whether insertions to the underlying
    sparse storage format may have occurred, in which case the underlying
    sparse storage format needs to be finalized. Otherwise, the operation
    simply folds away.

    Note that this operation is "impure" in the sense that its behavior
    is solely defined by side-effects and not SSA values. The semantics
    may be refined over time as our sparse abstractions evolve.

    Example:

    ```mlir
    %1 = sparse_tensor.load %0 : tensor<8xf64, #SV>
    ```
  }];
  let assemblyFormat = "$tensor (`hasInserts` $hasInserts^)? attr-dict `:` type($tensor)";
}

def SparseTensor_OutOp : SparseTensor_Op<"out", []>,
    Arguments<(ins AnySparseTensor:$tensor, AnyType:$dest)> {
  string summary = "Outputs a sparse tensor to the given destination";
  string description = [{
    Outputs the contents of a sparse tensor to the destination defined by an
    opaque pointer provided by `dest`. For targets that have access to a file
    system, for example, this pointer may specify a filename (or file) for output.
    The form of the operation is kept deliberately very general to allow for
    alternative implementations in the future, such as sending the contents to
    a buffer defined by a pointer.

    Note that this operation is "impure" in the sense that its behavior
    is solely defined by side-effects and not SSA values. The semantics
    may be refined over time as our sparse abstractions evolve.

    Example:

    ```mlir
    sparse_tensor.out %t, %dest : tensor<1024x1024xf64, #CSR>, !Dest
    ```
  }];
  let assemblyFormat = "$tensor `,` $dest attr-dict `:` type($tensor) `,` type($dest)";
}

def SparseTensor_SortOp : SparseTensor_Op<"sort", [AttrSizedOperandSegments]>,
    // TODO: May want to extend tablegen with
    // class NonemptyVariadic<Type type> : Variadic<type> { let minSize = 1; }
    // and then use NonemptyVariadic<...>:$xs here.
    Arguments<(ins Index:$n,
               Variadic<StridedMemRefRankOf<[AnyInteger, Index], [1]>>:$xs,
               Variadic<StridedMemRefRankOf<[AnyType], [1]>>:$ys)>  {
  string summary = "Sorts the arrays in xs and ys lexicographically on the "
                   "integral values found in the xs list";
  string description = [{
    Lexicographically sort the first `n` values in `xs` along with the values in
    `ys`. Conceptually, the values being sorted are tuples produced by
    `zip(zip(xs), zip(ys))`. In particular, values in `ys` needed to be sorted
    along with values in `xs`, but values in `ys` don't affect the
    lexicographical order. The order in which arrays appear in `xs` affects the
    sorting result. The operator updates `xs` and `ys` in place with the result
    of the sorting.

    For example, assume x1=[4, 3], x2=[1, 2], y1=[10, 5], then the output of
    "sort 2, x1, x2 jointly y1" are x1=[3, 4], x2=[2, 1], y1=[5, 10] while the
    output of "sort 2, x2, x1, jointly y1" are x2=[1, 2], x1=[4, 3], y1=[10, 5].

    Buffers in `xs` needs to have the same integral element type while buffers
    in `ys` can have different numeric element types. All buffers in `xs` and
    `ys` should have a dimension not less than `n`. The behavior of the operator
    is undefined if this condition is not met. The operator requires at least
    one buffer in `xs` while `ys` can be empty.

    Note that this operation is "impure" in the sense that its behavior is
    solely defined by side-effects and not SSA values. The semantics may be
    refined over time as our sparse abstractions evolve.

    Example:

    ```mlir
    sparse_tensor.sort %n, %x1, %x2 jointly y1, %y2
      : memref<?xindex>, memref<?xindex> jointly memref<?xindex>, memref<?xf32>
    ```
  }];
  let assemblyFormat = "$n `,` $xs (`jointly` $ys^)? attr-dict"
                       "`:` type($xs) (`jointly` type($ys)^)?";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Sparse Tensor Syntax Operations.
//===----------------------------------------------------------------------===//

def SparseTensor_BinaryOp : SparseTensor_Op<"binary", [NoSideEffect]>,
    Arguments<(ins AnyType:$x, AnyType:$y, UnitAttr:$left_identity, UnitAttr:$right_identity)>,
    Results<(outs AnyType:$output)> {
  let summary = "Binary set operation utilized within linalg.generic";
  let description = [{
      Defines a computation within a `linalg.generic` operation that takes two
      operands and executes one of the regions depending on whether both operands
      or either operand is nonzero (i.e. stored explicitly in the sparse storage
      format).

      Three regions are defined for the operation and must appear in this order:
      - overlap (elements present in both sparse tensors)
      - left (elements only present in the left sparse tensor)
      - right (element only present in the right sparse tensor)

      Each region contains a single block describing the computation and result.
      Every non-empty block must end with a sparse_tensor.yield and the return
      type must match the type of `output`. The primary region's block has two
      arguments, while the left and right region's block has only one argument.

      A region may also be declared empty (i.e. `left={}`), indicating that the
      region does not contribute to the output. For example, setting both
      `left={}` and `right={}` is equivalent to the intersection of the two
      inputs as only the overlap region will contribute values to the output.

      As a convenience, there is also a special token `identity` which can be
      used in place of the left or right region. This token indicates that
      the return value is the input value (i.e. func(%x) => return %x).
      As a practical example, setting `left=identity` and `right=identity`
      would be equivalent to a union operation where non-overlapping values
      in the inputs are copied to the output unchanged.

      Example of isEqual applied to intersecting elements only:

      ```mlir
      %C = bufferization.alloc_tensor...
      %0 = linalg.generic #trait
        ins(%A: tensor<?xf64, #SparseVector>,
            %B: tensor<?xf64, #SparseVector>)
        outs(%C: tensor<?xi8, #SparseVector>) {
        ^bb0(%a: f64, %b: f64, %c: i8) :
          %result = sparse_tensor.binary %a, %b : f64, f64 to i8
            overlap={
              ^bb0(%arg0: f64, %arg1: f64):
                %cmp = arith.cmpf "oeq", %arg0, %arg1 : f64
                %ret_i8 = arith.extui %cmp : i1 to i8
                sparse_tensor.yield %ret_i8 : i8
            }
            left={}
            right={}
          linalg.yield %result : i8
      } -> tensor<?xi8, #SparseVector>
      ```

      Example of A+B in upper triangle, A-B in lower triangle:

      ```mlir
      %C = bufferization.alloc_tensor...
      %1 = linalg.generic #trait
        ins(%A: tensor<?x?xf64, #CSR>, %B: tensor<?x?xf64, #CSR>
        outs(%C: tensor<?x?xf64, #CSR> {
        ^bb0(%a: f64, %b: f64, %c: f64) :
          %row = linalg.index 0 : index
          %col = linalg.index 1 : index
          %result = sparse_tensor.binary %a, %b : f64, f64 to f64
            overlap={
              ^bb0(%x: f64, %y: f64):
                %cmp = arith.cmpi "uge", %col, %row : index
                %upperTriangleResult = arith.addf %x, %y : f64
                %lowerTriangleResult = arith.subf %x, %y : f64
                %ret = arith.select %cmp, %upperTriangleResult, %lowerTriangleResult : f64
                sparse_tensor.yield %ret : f64
            }
            left=identity
            right={
              ^bb0(%y: f64):
                %cmp = arith.cmpi "uge", %col, %row : index
                %lowerTriangleResult = arith.negf %y : f64
                %ret = arith.select %cmp, %y, %lowerTriangleResult : f64
                sparse_tensor.yield %ret : f64
            }
          linalg.yield %result : f64
      } -> tensor<?x?xf64, #CSR>
      ```

      Example of set difference. Returns a copy of A where its sparse structure
      is *not* overlapped by B. The element type of B can be different than A
      because we never use its values, only its sparse structure:

      ```mlir
      %C = bufferization.alloc_tensor...
      %2 = linalg.generic #trait
        ins(%A: tensor<?x?xf64, #CSR>, %B: tensor<?x?xi32, #CSR>
        outs(%C: tensor<?x?xf64, #CSR> {
        ^bb0(%a: f64, %b: i32, %c: f64) :
          %result = sparse_tensor.binary %a, %b : f64, i32 to f64
            overlap={}
            left=identity
            right={}
          linalg.yield %result : f64
      } -> tensor<?x?xf64, #CSR>
      ```
  }];

  let regions = (region AnyRegion:$overlapRegion, AnyRegion:$leftRegion, AnyRegion:$rightRegion);
  let assemblyFormat = [{
        $x `,` $y `:` attr-dict type($x) `,` type($y) `to` type($output) `\n`
        `overlap` `=` $overlapRegion `\n`
        `left` `=` (`identity` $left_identity^):($leftRegion)? `\n`
        `right` `=` (`identity` $right_identity^):($rightRegion)?
  }];
  let hasVerifier = 1;
}

def SparseTensor_UnaryOp : SparseTensor_Op<"unary", [NoSideEffect]>,
    Arguments<(ins AnyType:$x)>,
    Results<(outs AnyType:$output)> {
  let summary = "Unary set operation utilized within linalg.generic";
  let description = [{
      Defines a computation with a `linalg.generic` operation that takes a single
      operand and executes one of two regions depending on whether the operand is
      nonzero (i.e. stored explicitly in the sparse storage format).

      Two regions are defined for the operation must appear in this order:
      - present (elements present in the sparse tensor)
      - absent (elements not present in the sparse tensor)

      Each region contains a single block describing the computation and result.
      A non-empty block must end with a sparse_tensor.yield and the return type
      must match the type of `output`. The primary region's block has one
      argument, while the missing region's block has zero arguments.

      A region may also be declared empty (i.e. `absent={}`), indicating that the
      region does not contribute to the output.

      Example of A+1, restricted to existing elements:

      ```mlir
      %C = bufferization.alloc_tensor...
      %0 = linalg.generic #trait
         ins(%A: tensor<?xf64, #SparseVector>)
        outs(%C: tensor<?xf64, #SparseVector>) {
        ^bb0(%a: f64, %c: f64) :
          %result = sparse_tensor.unary %a : f64 to f64
            present={
              ^bb0(%arg0: f64):
                %cf1 = arith.constant 1.0 : f64
                %ret = arith.addf %arg0, %cf1 : f64
                sparse_tensor.yield %ret : f64
            }
            absent={}
          linalg.yield %result : f64
      } -> tensor<?xf64, #SparseVector>
      ```

      Example returning +1 for existing values and -1 for missing values:

      ```mlir
      %result = sparse_tensor.unary %a : f64 to i32
        present={
          ^bb0(%x: f64):
            %ret = arith.constant 1 : i32
            sparse_tensor.yield %ret : i32
        }
        absent={
          %ret = arith.constant -1 : i32
          sparse_tensor.yield %ret : i32
        }
      ```

      Example showing a structural inversion (existing values become missing in
      the output, while missing values are filled with 1):

      ```mlir
      %result = sparse_tensor.unary %a : f64 to i64
        present={}
        absent={
          %ret = arith.constant 1 : i64
          sparse_tensor.yield %ret : i64
        }
      ```
  }];

  let regions = (region AnyRegion:$presentRegion, AnyRegion:$absentRegion);
  let assemblyFormat = [{
        $x attr-dict `:` type($x) `to` type($output) `\n`
        `present` `=` $presentRegion `\n`
        `absent` `=` $absentRegion
  }];
  let hasVerifier = 1;
}

def SparseTensor_ReduceOp : SparseTensor_Op<"reduce", [NoSideEffect, SameOperandsAndResultType]>,
    Arguments<(ins AnyType:$x, AnyType:$y, AnyType:$identity)>,
    Results<(outs AnyType:$output)> {
  let summary = "Custom reduction operation utilized within linalg.generic";
  let description = [{
      Defines a computation with a `linalg.generic` operation that takes two
      operands and an identity value and reduces all values down to a single
      result based on the computation in the region.

      The region must contain exactly one block taking two arguments. The block
      must end with a sparse_tensor.yield and the output must match the input
      argument types.

      Note that this operation is only required for custom reductions beyond the
      standard operations (add, mul, and, or, etc). The `linalg.generic`
      `iterator_types` defines which indices are being reduced. When the associated
      operands are used in an operation, a reduction will occur. The use of this
      explicit `reduce` operation is not required in most cases.

      Example of Matrix->Vector reduction using max(product(x_i), 100):

      ```mlir
      %cf1 = arith.constant 1.0 : f64
      %cf100 = arith.constant 100.0 : f64
      %C = bufferization.alloc_tensor...
      %0 = linalg.generic #trait
         ins(%A: tensor<?x?xf64, #SparseMatrix>)
        outs(%C: tensor<?xf64, #SparseVector>) {
        ^bb0(%a: f64, %c: f64) :
          %result = sparse_tensor.reduce %c, %a, %cf1 : f64 {
              ^bb0(%arg0: f64, %arg1: f64):
                %0 = arith.mulf %arg0, %arg1 : f64
                %cmp = arith.cmpf "ogt", %0, %cf100 : f64
                %ret = arith.select %cmp, %cf100, %0 : f64
                sparse_tensor.yield %ret : f64
            }
          linalg.yield %result : f64
      } -> tensor<?xf64, #SparseVector>
      ```
  }];

  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = [{
         $x `,` $y `,` $identity attr-dict `:` type($output) $region
  }];
  let hasVerifier = 1;
}

def SparseTensor_SelectOp : SparseTensor_Op<"select", [NoSideEffect, SameOperandsAndResultType]>,
    Arguments<(ins AnyType:$x)>,
    Results<(outs AnyType:$output)> {
  let summary = "Select operation utilized within linalg.generic";
  let description = [{
      Defines an evaluation within a `linalg.generic` operation that takes a single
      operand and decides whether or not to keep that operand in the output.

      A single region must contain exactly one block taking one argument. The block
      must end with a sparse_tensor.yield and the output type must be boolean.

      Value threshold is an obvious usage of the select operation. However, by using
      `linalg.index`, other useful selection can be achieved, such as selecting the
      upper triangle of a matrix.

      Example of selecting A >= 4.0:

      ```mlir
      %C = bufferization.alloc_tensor...
      %0 = linalg.generic #trait
         ins(%A: tensor<?xf64, #SparseVector>)
        outs(%C: tensor<?xf64, #SparseVector>) {
        ^bb0(%a: f64, %c: f64) :
          %result = sparse_tensor.select %a : f64 {
              ^bb0(%arg0: f64):
                %cf4 = arith.constant 4.0 : f64
                %keep = arith.cmpf "uge", %arg0, %cf4 : f64
                sparse_tensor.yield %keep : i1
            }
          linalg.yield %result : f64
      } -> tensor<?xf64, #SparseVector>
      ```

      Example of selecting lower triangle of a matrix:

      ```mlir
      %C = bufferization.alloc_tensor...
      %0 = linalg.generic #trait
         ins(%A: tensor<?x?xf64, #CSR>)
        outs(%C: tensor<?x?xf64, #CSR>) {
        ^bb0(%a: f64, %c: f64) :
          %row = linalg.index 0 : index
          %col = linalg.index 1 : index
          %result = sparse_tensor.select %a : f64 {
              ^bb0(%arg0: f64):
                %keep = arith.cmpf "olt", %col, %row : f64
                sparse_tensor.yield %keep : i1
            }
          linalg.yield %result : f64
      } -> tensor<?x?xf64, #CSR>
      ```
  }];

  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
         $x attr-dict `:` type($x) $region
  }];
  let hasVerifier = 1;
}

def SparseTensor_YieldOp : SparseTensor_Op<"yield", [NoSideEffect, Terminator]>,
    Arguments<(ins Optional<AnyType>:$result)> {
  let summary = "Yield from sparse_tensor set-like operations";
  let description = [{
      Yields a value from within a `binary`, `unary`, `reduce`,
      `select` or `foreach` block.

      Example:

      ```mlir
      %0 = sparse_tensor.unary %a : i64 to i64 {
        ^bb0(%arg0: i64):
          %cst = arith.constant 1 : i64
          %ret = arith.addi %arg0, %cst : i64
          sparse_tensor.yield %ret : i64
      }
      ```
  }];

  let builders = [
    OpBuilder<(ins),
    [{
      build($_builder, $_state, Value());
    }]>
  ];

  let assemblyFormat = [{
        $result attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

def SparseTensor_ForeachOp : SparseTensor_Op<"foreach",
    [SingleBlockImplicitTerminator<"YieldOp">]>,
    Arguments<(ins AnySparseTensor:$tensor)>{
  let summary = "Iterates over non-zero elements in a sparse tensor";
  let description = [{
     Iterates over every non-zero element in the given sparse tensor and executes
     the block.

     For a input sparse tensor with rank n, the block must take n + 1 arguments. The
     first n arguments must be Index type, together indicating the current coordinates
     of the element being visited. The last argument must have the same type as the
     sparse tensor's element type, representing the actual value loaded from the input
     tensor at the given coordinates.

     Example:

     ```mlir
     sparse_tensor.foreach in %0 : tensor<?x?xf64, #DCSR> do {
      ^bb0(%arg1: index, %arg2: index, %arg3: f64):
        do something...
     }
     ```
  }];

  let regions = (region AnyRegion:$region);
  let assemblyFormat = "`in` $tensor attr-dict `:` type($tensor)  `do` $region";
  let hasVerifier = 1;
}

#endif // SPARSETENSOR_OPS
