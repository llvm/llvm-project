---
title: "MLIR Query"
date: 1970-01-01T00:00:00Z
draft: false
---

<p/>{{< toc >}}


`mlir-query` is an interactive tool designed to simplify IR exploration. It provides a REPL interface and supports an interactive query language for MLIR, enabling developers to query the MLIR IR dynamically.  
The tool uses matchers as its core mechanism for performing queries over the MLIR IR, relying on simple matchers from `Matchers.h` and slicing-related matchers from `SliceMatchers.h`.  

Through its IR exploration capabilities and the interactive query language, `mlir-query` serves both as a prototyping environment for pattern matchers and as a good debugging tool.

## How to use it

The tool’s primary purpose is to develop matchers. A typical workflow follows this structure: build a matcher iteratively, test and visualize it, and extract it to C++.


## Features
### Autocompletion

To simplify usage, `mlir-query` provides autocompletion in the REPL interface, enabling users to ease query input by pressing the Tab key. The GIF below demonstrates an autocompletion use case. When autocompletion is first triggered, a list of available commands is displayed (e.g., `match`, `help`). Triggering autocompletion for the `match` command then shows a list of available matchers.
<div style="overflow-x:auto; margin:1em 0;">
  <img
    src="https://i.imgur.com/3QiJgrU.gif"
    alt="Autocompletion command list"
    style="
      display: block;
      margin: 0 auto;
      width: 1200px !important;
      max-width: none !important;
      height: auto !important;
    "
  />
</div>


The next GIF illustrates an autocompletion use case for constructing queries. 

<div style="overflow-x:auto; margin:1em 0;">
  <img
    src="https://i.imgur.com/bpMS9mf.gif"
    alt="Autocompletion matcher"
    style="
      display: block;
      margin: 0 auto;
      width: 1200px !important;
      max-width: none !important;
      height: auto !important;
    "
  />
</div>

### Function extraction

Matcher results can easily be extracted into a custom function by using `extract("functionName")` feature. For instance, the following GIF demonstrates how the reuslt of a slicing query are extracted into a function called `backward_slice`.

## Matcher overview

This section details the current matchers and their capabilities. It does not include examples of every matcher but rather aims to showcase and explain the types of matchers, along with useful examples that should be sufficient for comprehension. For a detailed explanation of each matcher's functionality and its parameters, please refer to the matchers list section.

### Simple matchers

The tool supports a variety of simple matchers, including `isConstantOp`, which finds all constant operations, `hasOpName`, which finds all operations with a given name and `hasOpAttrName`, which finds all operations with a certain attribute. The next GIF demonstrates a simple query being performed on `mixedOperations` function, specifically matching all the `arith.addf` operations.

```mlir
func.func @mixedOperations(%a: f32, %b: f32, %c: f32) -> f32 {
  %sum0 = arith.addf %a, %b : f32
  %sub0 = arith.subf %sum0, %c : f32
  %mul0 = arith.mulf %a, %sub0 : f32
  %sum1 = arith.addf %b, %c : f32
  %mul1 = arith.mulf %sum1, %mul0 : f32
  %sub2 = arith.subf %mul1, %a : f32
  %sum2 = arith.addf %mul1, %b : f32
  %mul2 = arith.mulf %sub2, %sum2 : f32
  return %mul2 : f32
}
```

<div style="overflow-x:auto; margin:1em 0;">
  <img
    src="https://i.imgur.com/dbpn3Xo.gif"
    alt="Autocompletion matcher"
    style="
      display: block;
      margin: 0 auto;
      width: 1200px !important;
      max-width: none !important;
      height: auto !important;
    "
  />
</div>

### Slice matchers

`mlir-query` includes slicing matchers that compute forward and backward slices. These are abstractions over the methods from the `SliceAnalysis` library, enabling their use in a query context. In contrast to simple matchers, slicing matchers introduce the concept of `inner matchers`, which allow users to specify the `root operation` and the exit condition via other `matchers`. 

Two useful backward-slicing matchers are `getDefinitionsByPredicate` and `getDefinitions`. The former matches all definitions by specifying both the starting point of the slice computation and the exit condition using an inner matcher. The latter is similar, except it limits the exit condition to a depth level specified as a numeric literal argument. Both matchers accept three boolean arguments: `omitBlockArguments`, `omitUsesFromAbove`, and `inclusive`. The first two specify traversal configuration, while the last controls inclusion of the root operation in the slice. 

Forward-slicing matchers are similar, but their exit condition is currently limited to specification via a nested matcher. 

To illustrate, the next two GIFs demonstrate queries using the matchers described above.

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @slice_use_from_above(%arg0: tensor<5x5xf32>, %arg1: tensor<5x5xf32>) {
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<5x5xf32>) outs(%arg1 : tensor<5x5xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.addf %in, %in : f32
    linalg.yield %2 : f32
  } -> tensor<5x5xf32>
  %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<5x5xf32> into tensor<25xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<5x5xf32>) outs(%arg1 : tensor<5x5xf32>) {
  ^bb0(%in: f32, %out: f32):
    %c2 = arith.constant 2 : index
    %extracted = tensor.extract %collapsed[%c2] : tensor<25xf32>
    %2 = arith.addf %extracted, %extracted : f32
    linalg.yield %2 : f32
  } -> tensor<5x5xf32>
  return
}
```

Matches all defining operations, using the `hasOpName("arith.addf")` inner matcher for the root operation and the `hasOpName("linalg.generic")` inner matcher for the exit. condition.
<div style="overflow-x:auto; margin:1em 0;">
  <img
    src="https://i.imgur.com/e7ObI7P.gif"
    alt="getDefinitionsByPredicate matcher"
    style="
      display: block;
      margin: 0 auto;
      width: 1200px !important;
      max-width: none !important;
      height: auto !important;
    "
  />
</div>

Matches all defining operations by using `hasOpName("arith.addf")` inner matcher for the root operation and limiting traversal to a depth of two levels.
<div style="overflow-x:auto; margin:1em 0;">
  <img
    src="https://i.imgur.com/V4uegw2.gif"
    alt="getDefinitions matcher"
    style="
      display: block;
      margin: 0 auto;
      width: 1200px !important;
      max-width: none !important;
      height: auto !important;
    "
  />
</div>

### Variadic matchers

At this moment, the tool supports two variadic matchers: `anyOf` and `allOf`, which can be conceptualized as matcher combinators, as one can group multiple matchers together to facilitate the construction of complex matchers.

Operator `anyOf` matches if any of the matchers in a given set match succeed. Using it brings several benefits, for example, one could construct a matcher that computes the union of two or more slices, initiating slice computation from one or multiple points of interest, or limiting slice computation by a set of inner matchers.

Operator `allOf`  matches only if all matchers in a set of matchers succeed. For example, it enables finding all operations with a certain attribute, or starting/limiting slice computation when an operation with a certain attribute is encountered.

The next three GIFs illustrate the behavior described above.

```mlir
func.func @slice_depth1_loop_nest_with_offsets() {
  %0 = memref.alloc() : memref<100xf32>
  %cst = arith.constant 7.000000e+00 : f32
  affine.for %i0 = 0 to 16 {
    %a0 = affine.apply affine_map<(d0) -> (d0 + 2)>(%i0)
    affine.store %cst, %0[%a0] : memref<100xf32>
  }
  affine.for %i1 = 4 to 8 {
    %a1 = affine.apply affine_map<(d0) -> (d0 - 1)>(%i1)
    %1 = affine.load %0[%a1] : memref<100xf32>
  }
  return
}
```
Computes the union of two backward slices
<div style="overflow-x:auto; margin:1em 0;">
  <img
    src="https://i.imgur.com/qQhfyX4.gif"
    alt="backward-slice-union-anyof matcher"
    style="
      display: block;
      margin: 0 auto;
      width: 1200px !important;
      max-width: none !important;
      height: auto !important;
    "
  />
</div>

## Matcher Reference


| Matcher                               | Documentation                                                                                       |
|---------------------------------------|-----------------------------------------------------------------------------------------------------|
| `isConstantOp()`                      | [IsConstantOp](https://mlir.llvm.org/doxygen/structmlir_1_1detail_1_1IsConstantOp.html)             |
| `hasOpName(StringRef name)`           | [NameOpMatcher](https://mlir.llvm.org/doxygen/structmlir_1_1detail_1_1NameOpMatcher.html)           |
| `hasOpAttrName(StringRef attrName)`   | [AttrOpMatcher](https://mlir.llvm.org/doxygen/structmlir_1_1detail_1_1AttrOpMatcher.html)           |
| `getDefinitions(...)`                 | [MLIR-Query docs](#getdefinitions)                                                                  |
| `getDefinitionsByPredicate(...)`      | [MLIR-Query docs](#getdefinitionsbypredicate)                                                       |
| `anyOf(Matcher…)`                     | [MLIR-Query docs](#anyof)                                                                           |
| `allOf(Matcher…)`                     | [MLIR-Query docs](#allof)                                                                           |

