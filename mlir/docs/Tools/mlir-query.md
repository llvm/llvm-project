# MLIR Query

[TOC]


`mlir-query` is an interactive tool designed to simplify IR exploration. The tool provides a REPL interface and supports an interactive query language for MLIR, enabling developers to dynamically query the MLIR IR.
The tool uses matchers as its core mechanism for peforming queries over the MLIR IR. It relies on simple matchers from `Matchers.h` and slicing-related ones from `SliceMatchers.h`.

## Features
#### Autocompletion

To simplify usage, `mlir-query` provides autocompletion in the REPL interface, enabling users to ease query input by pressing the Tab key. The GIF below demonstrates an autocompletion use case. When autocompletion is first triggered, a list of available commands is displayed (e.g., `match`, `help`). Triggering autocompletion for the `match` command then shows a list of available matchers.

![Autocompletion demo for mlir-query](https://i.imgur.com/3QiJgrU.gif)

The next GIF illustrates an autocompletion use case for constructing queries. 

![Autocompletion demo for mlir-query](https://i.imgur.com/bpMS9mf.gif)

## Types of matchers
#### Simple matchers

The tool supports a variety of simple matchers, including `isConstantOp`, which finds all constant operations, and `hasOpName(String)`, which finds all operations with a given name. The next GIF demonstrates a simple query being performed on `mixedOperations` function, specifically matching all the `arith.addf` operations.

```llvm
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

![Simple query demo for mlir-query](https://i.imgur.com/dbpn3Xo.gif)

#### Slice matchers

`mlir-query` includes slicing matchers that compute the forward and backward slices. These are abstractions over the methods from `SliceAnalysis` library, facilitating exposure of those methods to users in query context. In contrast to simple matchers, slice matchers also introduce the concept of `nested matchers` which allow one to specify the `root operation` and the exit condition via other `matchers`. 

Two useful slicing matchers are `getDefinitionsByPredicate` and `getDefinitions`. The former matches all definitions by specifying both the starting point of the slice computation (root operation) and the exit condition using a nested matcher. The latter is very similar, except the exit condition is limited by a depth level specified as a numeric literal argument to the matcher. Both matchers accept three boolean arguments: `omitBlockArguments`, `omitUsesFromAbove`, and `inclusive`. The first two specify traversal configuration, while the last controls root operation inclusion in the slice. For forward slicing, the exit condition is currently limited to specification via a nested matcher.

The next two GIFs demonstrate queries using the matchers described above.

```llvm
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

![Simple query demo for mlir-query](https://i.imgur.com/e7ObI7P.gif)

#### Variadic matchers

At this moment, the tool supports two variadic matchers: `anyOf` and `allOf`. These could be conceptualized as matcher combinators, as one can group multiple matchers together to facilitate the construction of complex and powerful queries.
