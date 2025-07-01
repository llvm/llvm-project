# MLIR Query

[TOC]


`mlir-query` is an interactive tool designed to simplify IR exploration and visualization. The tool provides a REPL interface and supports an interactive query language for MLIR, enabling developers to dynamically query the MLIR IR.
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

`mlir-query` includes multiple slicing matchers. These are abstractions over the methods from `SliceAnalysis`, facilitating exposure of those methods to users in query context. In contrast to simple matchers, slice matchers also introduce the concept of `nested matchers` which allows one to specify the `root operation` and the exit condition via other `matchers`.

#### Variadic matchers

At this moment, the tool supports two variadic matchers: `anyOf` and `allOf`. These could be conceptualized as matcher combinators, as one can group multiple matchers together to facilitate the construction of complex and powerful queries.
