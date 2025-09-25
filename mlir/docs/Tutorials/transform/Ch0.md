# Chapter 0: A Primer on “Structured” Linalg Operations

Before starting the tutorial on the Transform dialect, let us take a brief look at the concept of Structured operations and its implementation in the Linalg dialect. Note that the Transform dialect does not require Structured operations and vice versa. The two co-evolved at the beginning of the Transform dialect, which makes the subset of transformations for Structured operations the most mature and most suitable for the tutorial. If you are already familiar with this concept, skip to Chapter 1.

Structured code generation intends to preserve the structure of the computation for as long as necessary to enable transformations, up to and including the design of IR abstractions that support specific transformations.

## Uniform Elementwise Extension

Consider a simple scalar arithmetic addition operation in MLIR, which maps directly to a machine instruction on most architectures that support floating point operations:


```mlir
%2 = arith.addf %0, %1 : f32
```

This operation can be easily extended to uniformly apply to elements of a 1D vector, which is also often available as an instruction of vector machines:

```mlir
%2 = arith.addf %0, %1 : vector<8xf32>
```

Only a few modern instruction sets offer instructions for two- or more-dimensional vectors. In MLIR, however, it is possible to transparently extend the uniform elementwise application to vectors of arbitrary rank.

```mlir
%2 = arith.addf %0, %1 : vector<8x4xf32>
%5 = arith.addf %3, %4 : vector<2x2x2x2x2x2x2xf32>
```

As you can notice, MLIR’s arithmetic operations on vectors preserve the structure of uniform elementwise application. This structure can be leveraged by the compiler, for example, to produce smaller-rank operations available on the target or to fuse multiplication and addition when such a fused instruction is available (which becomes complicated when there are a hundred of multiplications followed by a hundred of additions).

## Reduction

Sometimes it is necessary to add elements of a vector to obtain a scalar. Some platforms provide specific instructions for this operation, some others provide ones that can be combined to achieve the desired effect, such as addition of adjacent elements and element shuffle.

The Vector dialect in MLIR defines an operation to explicitly denote a within-vector reduction:

```mlir
%1 = vector.reduction <add>, %0 : vector<8xf32> into f32
```

When no support is available, such an operation can be transformed into a loop:

```mlir
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c8 = arith.constant 8 : index
%init = arith.constant 0.0 : f32
%result = scf.for %i = %c0 to %c8 step %c1 iter_args(%partial = %init) -> (f32) {
  %element = vector.extract %0[%i] : f32 into vector<8xf32>
  %updated = arith.addf %partial, %element : f32
  scf.yield %updated : f32
}
```

Even when special instructions are available, it may still be desirable to use the loop form (with unrolling), depending on instruction latency and register pressure. Preserving the structure of the operation as a single reduction gives the compiler an understanding that a within-vector reduction is performed and, therefore, a choice in implementation.

## Contraction

Contraction is a generalization of reduction that multiplies elements from two vectors before adding them up. A simple “add” reduction can be thought of as a contraction where one of the vectors contains `1.0`, the neutral element of multiplication. Contractions offer even more flexibility to the compiler, and are represented by a dedicated operation in MLIR:

```mlir
// Neutral initializer for the addition.
%init  = arith.constant 0.0 : f32
// Neutral element of multiplication.
%ones = arith.constant dense<1.0> : vector<8xf32>
// Actual contraction.
%result = vector.contract {
  indexing_maps = [affine_map<(i) -> (i)>,
                   affine_map<(i) -> (i)>,
                   affine_map<(i) -> ()>],
  iterator_types = ["reduction"]
} %0, %ones, %init : vector<8xf32>, vector<8xf32> into f32
```

Note the `affine_map` expressions indicating how vector elements are indexed. Their meaning is perhaps most evident when writing the loop form in pseudo-code equivalent to this contraction:

```mlir
for i in 0 to 8:
  init += p0[i] * ones[i]
```

where both `%0` and `%ones` use the loop induction variable `i`, as noted on the right-hand side of the corresponding affine map, `(i) -> (i)`, and `%init` does not, as reflected on the right-hand side of its affine map, `(i) -> ()`.

Similarly to uniform elementwise extension, MLIR vector contractions are not limited to 1D cases. In the 2D+ case, one can additionally specify which of the vector dimensions are being reduced and which ones are being preserved. This can be achieved by using the `iterator_types` attribute that specifies, for each dimension, whether it is being reduced (`"reduction"`) or preserved (`"parallel"`). Consider the following 3D contraction that encodes a matrix-matrix multiplication:

```mlir
%result = vector.contract {
  indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                   affine_map<(i, j, k) -> (k, j)>,
                   affine_map<(i, j, k) -> (i, j)>],
  iterator_types = ["parallel", "parallel", "reduction"]
} %lhs, %rhs, %init: vector<8x10xf32>, vector<10x16xf32> into vector<8x16xf32>
```

Looking at the indexing maps, it is easy to recognize the loop form:

```mlir
for i in 0 to 8:
  for j in 0 to 16:
    for k in 0 to 10:
      init[i, j] += lhs[i, k] * rhs[k, j]
```

Preserving this higher-level structure of a contraction makes it significantly easier for the compiler to recognize operations such as matrix multiplications and dot products and gives it freedom to produce lower-level operations that leverage most advanced instructions or even pre-generated microkernels.

## Generic Operation on Memory

Until now, we have been considering operations on vectors stored in virtual registers. A similar contraction abstraction can be defined in memory:

```mlir
linalg.generic {
  indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                   affine_map<(i, j, k) -> (k, j)>,
                   affine_map<(i, j, k) -> (i, j)>],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%lhs, %rhs : memref<8x10xf32>, memref<10x16xf32>)
  outs(%init : memref<8x16xf32>) {
^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
  %0 = arith.mulf %lhs_one, %rhs_one : f32
  %1 = arith.addf %init_one, %0 : f32
  linalg.yield %1 : f32
}
```

This looks more complicated, so let us unpack. The `indexing_maps` and `iterator_types` are _exactly_ the same as we have seen above for vector contractions. The operands are now split into two lists:


*   `in` operands containing the buffers that are being only read by the operation;
*   `out` operands that are being read and updated by the operation.

This separation wasn’t necessary on vectors because, in MLIR, vectors are read-only (SSA or functional form) and operations mutating a vector are in fact producing a new one instead.

Furthermore, the operation now contains a region that explicitly specifies the multiplication and the addition operations that were implicit in the contraction. Block arguments in the region correspond to individual elements read from the buffer: the first two correspond to the `in` operands and the last one corresponds to the `out` operand. The value yielded from the region is “written” to the `out` operand and is available as the last block argument for future executions of the region. Note that the order in which the region is executed for various tuples of elements read from the buffers is not specified, and the write to the `out` buffer is written as a whole at the end of the operation.

## “Loop” Fusion

Since the region of the `linalg.generic` operation can contain arbitrarily many operations, we can use it to express “fusion” of the implicit loops by simply having more operations chained in the region. For example, the common machine learning rectified linear unit layer (ReLU), which can be defined as `relu(x) = max(0, x)`, can be expressed using the “compare-and-select” idiom in one `linalg.generic` operation, without the temporary buffer for the comparison result and without repeating the outer operation:

```mlir
linalg.generic {
  indexing_maps [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
  iterator_types = ["parallel"]
} ins(%in : memref<?xf32>) outs(%out : memref<?xf32>) {
^bb0(%in_one : f32, %out_one : f32):
  %c0 = arith.constant 0.0 : f32
  %0 = arith.cmpf ogt %in_one, %c0 : f32
  %1 = arith.select %0, %in_one, %c0 : f32
  linalg.yield %1 : f32
}
```

Such operations can be converted to loops or lowered into vector forms after splitting into multiple operations, each of which maps to a Vector dialect primitive. This modeling, again, gives the compiler more choice in selecting the code generation strategy.

## Generic Operation on Tensors

Let us take one last step up on the abstraction ladder. MLIR provides a tensor abstraction that makes it easy for the compiler to reason about multidimensional yet regular data without having to solve complex problems such as alias analysis and dependency satisfaction, which would be necessary on multidimensional buffers. The tensor abstraction is very similar to the vector abstraction (major differences include the availability of unranked tensors, tensor layouts, and vectors being usable as elemental types of tensors but not of other vectors). Tensors are read-only, and operations updating a tensor produce a new tensor.

The `linalg.generic` operation from above can lifted to operate on tensors instead of buffers:

```mlir
%result = linalg.generic {
  indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                   affine_map<(i, j, k) -> (k, j)>,
                   affine_map<(i, j, k) -> (i, j)>],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%lhs, %rhs : tensor<8x10xf32>,tensor<10x16xf32>)
  outs(%init :tensor<8x16xf32>) {
^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
  %0 = arith.mulf %lhs_one, %rhs_one : f32
  %1 = arith.addf %init_one, %0 : f32
  linalg.yield %1 : f32
} -> tensor<8x16xf32>
```

As you can notice, most components of this operation remain identical to its buffer version. It has been specifically designed this way. The main difference, beside the operand types, is that the operation now produces a new result instead of updating the `out` buffer. The `out` operand is used only as the initialization value.

If the `linalg.generic` operation had existed on vectors, it would have had the exact same structure.

## Tiling and Loop Materialization

At this level of abstraction, it becomes easy for the compiler to perform more advanced transformations usually required for high-performance code generation, such as [tiling](https://en.wikipedia.org/wiki/Loop_nest_optimization). Tiling, in general, can be seen as partitioning the iteration space into smaller parts, or tiles, so that the data required by each part fits into a level of cache for example. The order in which tiles are executed must preserve the original data dependencies.

In the case of `linalg.generic` operations, the iteration space is implicit and is defined by the shape of the operands. Therefore, a tile can be expressed by performing the _same_ operation on a subset (slice) of the original data. Since the order in which the body of `linalg.generic` is applied to different tuples of the input elements is unspecified, tiles can be executed in any order, without the need for dependence analysis. In order to control the execution of different tiles, the implementation of tiling produces loops. Thus tiling `linalg.generic` operations can also be seen as materializing the loops that have been implicit until now.

For example, tiling the matrix multiplication presented above with tile sizes `(2, 8)`, we obtain a loop nest around a `linalg.generic` expressing the same operation on a `2x8` tensor.

```mlir
// A special "multi-for" loop that supports tensor-insertion semantics
// as opposed to implicit updates. The resulting 8x16 tensor will be produced
// by this loop.
// The trip count of iterators is computed dividing the original tensor size,
// 8x16, by the tile size, 2x8, to obtain 4x2.
// When tensor sizes are dynamic, the trip count computation is emitted as IR
// and is being computed at runtime.
%0 = scf.forall (%i, %j) in (4, 2)
     shared_outs(%shared = %init) -> (tensor<8x16xf32>) {

  // Scale the loop induction variables by the tile sizes.
  %3 = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
  %4 = affine.apply affine_map<(d0) -> (d0 * 8)>(%j)

  // Take slices of inputs and outputs. Only the "i" and "j" dimensions are sliced.
  %lhs_slice = tensor.extract_slice %lhs[%3, 0] [2, 10] [1, 1]
             : tensor<8x10xf32> to tensor<2x10xf32>
  %rhs_slice = tensor.extract_slice %rhs[0, %4] [10, 8] [1, 1]
             : tensor<10x16xf32> to tensor<10x8xf32>
  %result_slice = tensor.extract_slice %shared[%3, %4] [2, 8] [1, 1]
                : tensor<8x16xf32> to tensor<2x8xf32>

  // This is exactly the same operation as before, but now operating on smaller
  // slices of data.
  %partial =  linalg.generic {
  indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                   affine_map<(i, j, k) -> (k, j)>,
                   affine_map<(i, j, k) -> (i, j)>],
  iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs_slice, %rhs_slice : tensor<2x10xf32>, tensor<10x8xf32>)
    outs(%result_slice : tensor<2x8xf32>) -> tensor<2x8xf32> {
  ^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
    %0 = arith.mulf %lhs_one, %rhs_one : f32
    %1 = arith.addf %init_one, %0 : f32
    linalg.yield %1 : f32
  } : tensor<2x8xf32>

  // Terminator for the loop with tensor-insertion semantics. Inserts a slice
  // into a larger tensor, potentially in parallel.
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %partial into %shared[%3, %4] [2, 8] [1, 1]
        : tensor<2x8xf32> into tensor<8x16xf32>
  }
}
```

## Producer/Consumer Fusion and Rematerialization

After materializing loops with tiling, another key code generation transformation becomes simple – fusion. Unlike loop fusion, the Structured operations approach allows for producer/consumer fusion even when the (implicit) iteration spaces of the operations do not match. Given an high-level structured operation on tensors, such as `linalg.generic`, one can follow use-def chains to identify:

1. the subset (slice) of the operand that is used by the tile, and
2. the tensor-level structured operation producing the whole tensor that is being sliced.

By inverting the `indexing_map` and applying it to the set of elements accessed through the slice, we can compute the part of the iteration space of the operation defining the full tensor necessary to compute the tile. Thus fusion boils down to replacing the `tensor.extract_slice` operation with the tile of the `linalg.generic` producing the original operand.

Let us assume that the matrix multiplication operation is followed by another operation that multiplies each element of the resulting matrix with itself. This trailing elementwise operation has a 2D iteration space, unlike the 3D one in matrix multiplication. Nevertheless, it is possible to tile the trailing operation and then fuse the producer of its operand, the matmul, into the loop generated by tiling. The untiled dimension will be used in its entirety.


```mlir
// Same loop as before.
%0 = scf.forall (%i, %j) in (4, 2)
     shared_outs(%shared = %init)
     -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // Scale the loop induction variables by the tile sizes.
  %1 = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
  %2 = affine.apply affine_map<(d0) -> (d0 * 8)>(%j)

  // Take slices of inputs and outputs. Only the "i" and "j" dimensions are sliced.
  %lhs_slice = tensor.extract_slice %lhs[%1, 0] [2, 10] [1, 1]
             : tensor<8x10xf32> to tensor<2x10xf32>
  %rhs_slice = tensor.extract_slice %rhs[0, %2] [10, 8] [1, 1]
             : tensor<10x16xf32> to tensor<10x8xf32>
  %result_slice = tensor.extract_slice %result[%1, %2] [2, 8] [1, 1]
                : tensor<8x16xf32> to tensor<2x8xf32>

  // This is exactly the same matmul slice as before. It replaces the slice
  // extraction for the generic operation below.
  %partial = linalg.generic {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                     affine_map<(i, j, k) -> (k, j)>,
                     affine_map<(i, j, k) -> (i, j)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs_slice, %rhs_slice : tensor<2x10xf32>, tensor<10x8xf32>)
   outs(%result_slice : tensor<2x8xf32>) {
  ^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
    %5 = arith.mulf %lhs_one, %rhs_one : f32
    %6 = arith.addf %init_one, %5 : f32
    linalg.yield %6 : f32
  } -> tensor<2x8xf32>

  // Take the slice of the final result. Note that we don't need to take
  // the slice of the operand because the matmul operation above computes
  // it in-place.
  %shared_slice = tensor.extract_slice %shared[%1, %2] [2, 8] [1, 1]
                : tensor<8x16xf32> to tensor<2x8xf32>

  // The elementwise operation that we tiled.
  %elemwise = linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>,
                     affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%partial : tensor<2x8xf32>)
   outs(%shared_slice : tensor<2x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %5 = arith.mulf %in, %in : f32
    linalg.yield %5 : f32
  } -> tensor<2x8xf32>

  // Terminator for the loop with tensor-insertion semantics. Inserts a slice
  // into a larger tensor, potentially in parallel.
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %elemwise into %shared[%1, %2] [2, 8] [1, 1]
        : tensor<2x8xf32> into tensor<8x16xf32>
  }
}
```

This process may result in some elements in the operand tensors being (re)computed on every iteration of the loop. This is also known as _rematerialization_ and expresses the tradeoff between performing redundant computations or storing their result in (slow) memory.

## Shorthand “Named” Forms of Linalg Ops

Linalg provides a set of predefined operations for common cases such as matrix multiplication, dot product, convolution, etc. These operations are equivalent to the `generic` ones but spare the need to spell out the access patterns and the bodies. For example, matrix multiplication is simply:

```mlir
%matmul = linalg.matmul ins(%lhs, %rhs: tensor<8x10xf32>, tensor<10x16xf32>)
                        outs(%init: tensor<8x10xf32xf32>) -> tensor<8x16xf32>
```
