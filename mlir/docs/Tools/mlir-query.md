`mlir-query` is an interactive tool designed to simplify IR exploration. It provides a REPL interface and supports an interactive query language for MLIR, enabling developers to query the MLIR IR dynamically.  
The tool uses matchers as its core mechanism for performing queries over the MLIR IR, relying on simple matchers from `Matchers.h` and slicing-related matchers from `SliceMatchers.h`.  

Through its IR exploration capabilities and the interactive query language, `mlir-query` serves both as a prototyping environment for pattern matchers and as a good debugging tool.

## Usage

### Query modes
In order to prototype pattern matchers, explore, test, or debug the MLIR IR, the tool provides two main usage modes:

* **Run queries directly from the CLI:**
    ```shell
    ./mlir-query input.mlir -c "<your_query_1>" -c "<your_query_2>" ... "<your_query_N>"
    ```
    The commands are executed and the program exits immediately.

* **Launch an interactive session:**
    ```shell
    ./mlir-query input.mlir
    ```
    Opens a REPL-like interface where you can type queries interactively.

### Use with `mlir-opt`

The tool can easily be used with the MLIR pass pipeline infrastructure by running a pass pipeline and passing the result as input to `mlir-query`.

```shell
./mlir-opt input.mlir -canonicalize -o test.mlir | ./mlir-query test.mlir -c "<your_query_1>" -c "<your_query_2>" ... "<your_query_N>"
```
*Command example*

## Register a new matcher

To register a new matcher with `mlir-query`, you need to define a new structure that implements one of the following signatures: `bool match(Operation* op)` or `bool match(Operation* op, SetVector<Operation*> &matchedOps)`. Next, link `MLIRQueryLib` and register the matcher.

```cpp
#include "mlir/Tools/mlir-query/MlirQueryMain.h"
using namespace mlir;

int main(int argc, char **argv) {

  DialectRegistry dialectRegistry;
  registerAllDialects(dialectRegistry);

  query::matcher::Registry matcherRegistry;

  // Replace <matcher_name> with your desired matcher identifier string.
  matcherRegistry.registerMatcher("<matcher_name>", matcherInstance);

  MLIRContext context(dialectRegistry);
  return failed(mlirQueryMain(argc, argv, context, matcherRegistry));
}
```

## Features
### Autocompletion

To simplify usage, `mlir-query` provides autocompletion in the REPL interface, enabling users to ease query input by pressing the Tab key.
<div style="overflow-x:auto; margin:1em 0;">
  <img
    src="https://i.imgur.com/3QiJgrU.gif"
    alt="Autocompletion command list"
    style="
      display: block;
      margin: 0 auto;
      width: 1250px !important;
      max-width: none !important;
      height: auto !important;
    "
  />
</div>


*When autocompletion is first triggered, a list of available commands is displayed* (e.g., `match`, `help`). *Triggering autocompletion for the* `match` *command then shows a list of available matchers.*

<div style="overflow-x:auto; margin:1em 0;">
  <img
    src="https://i.imgur.com/bpMS9mf.gif"
    alt="Autocompletion matcher"
    style="
      display: block;
      margin: 0 auto;
      width: 1250px !important;
      max-width: none !important;
      height: auto !important;
    "
  />
</div>

*Autocompletion use case for constructing queries.*

### Function extraction

Results from a matcher can be isolated into a custom function using the `extract("functionName")` feature, facilitating further exploration or testing.

#### Example

```mlir
func.func @slicing_memref_store_trivial() {
  %0 = memref.alloc() : memref<10xf32>
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %i1 = 0 to 10 {
    %1 = affine.apply affine_map<()[s0] -> (s0)>()[%c0]
    memref.store %cst, %0[%1] : memref<10xf32>
    %2 = memref.load %0[%c0] : memref<10xf32>
    %3 = affine.apply affine_map<()[] -> (0)>()[]
    memref.store %cst, %0[%3] : memref<10xf32>
    memref.store %2, %0[%c0] : memref<10xf32>
  }
  return
}
```

*Initial function.*

```shell
./mlir-opt /home/user/llvm-project/mlir/test/mlir-query/slice-function-extraction.mlir "m getDefinitionsByPredicate(hasOpName(\"memref.store\"),hasOpName(\"memref.alloc\"),true,false,false).extract(\"backward_slice\")"
```

*Command used to extract the results of* `getDefinitionsByPredicate` *query.*

```mlir
func.func @backward_slice(%arg0: memref<10xf32>) -> (f32, index, index, f32, index, index, f32) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = affine.apply affine_map<()[s0] -> (s0)>()[%c0]
  memref.store %cst, %arg0[%0] : memref<10xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %1 = affine.apply affine_map<() -> (0)>()
  memref.store %cst_0, %arg0[%1] : memref<10xf32>
  %c0_1 = arith.constant 0 : index
  %2 = memref.load %arg0[%c0_1] : memref<10xf32>
  memref.store %2, %arg0[%c0_1] : memref<10xf32>
  return %cst, %c0, %0, %cst_0, %1, %c0_1, %2 : f32, index, index, f32, index, index, f32
}
```

*The function containing only the relevant slice.*

## Matcher overview

This section details the current matchers and their capabilities. It does not include examples of every matcher but rather aims to showcase and explain the types of matchers, along with useful examples that should be sufficient for comprehension. For a detailed explanation of each matcher's functionality and its parameters, please refer to the matchers reference section.

### Simple matchers

The tool supports a variety of simple matchers, including `isConstantOp`, which finds all constant operations, `hasOpName`, which finds all operations with a given name and `hasOpAttrName`, which finds all operations with a certain attribute.

#### Simple matcher example
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
      width: 1250px !important;
      max-width: none !important;
      height: auto !important;
    "
  />
</div>

*Matches all* `arith.addf` *operations*

### Slice matchers

`mlir-query` includes slicing matchers that compute forward and backward slices. These are abstractions over the methods from the `SliceAnalysis` library, enabling their use in a query context. In contrast to simple matchers, slicing matchers introduce the concept of `inner matchers`, which allow users to specify the `root operation` and the exit condition via other `matchers`. 

Two useful backward-slicing matchers are `getDefinitionsByPredicate` and `getDefinitions`. The former matches all definitions by specifying both the starting point of the slice computation and the exit condition using an inner matcher. The latter is similar, except it limits the exit condition to a depth level specified as a numeric literal argument. Both matchers accept three boolean arguments: `omitBlockArguments`, `omitUsesFromAbove`, and `inclusive`. The first two specify traversal configuration, while the last controls inclusion of the root operation in the slice. 

Forward-slicing matchers are similar, but their exit condition is currently limited to specification via a nested matcher. 

#### Slice matchers examples

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

<div style="overflow-x:auto; margin:1em 0;">
  <img
    src="https://i.imgur.com/e7ObI7P.gif"
    alt="getDefinitionsByPredicate matcher"
    style="
      display: block;
      margin: 0 auto;
      width: 1250px !important;
      max-width: none !important;
      height: auto !important;
    "
  />
</div>

*Matches all defining operations, using the* `hasOpName("arith.addf")` *inner matcher for the root operation and the* `hasOpName("linalg.generic")` *inner matcher for the exit condition.*

<div style="overflow-x:auto; margin:1em 0;">
  <img
    src="https://i.imgur.com/V4uegw2.gif"
    alt="getDefinitions matcher"
    style="
      display: block;
      margin: 0 auto;
      width: 1250px !important;
      max-width: none !important;
      height: auto !important;
    "
  />
</div>

*Matches all defining operations by using* `hasOpName("arith.addf")` *inner matcher for the root operation and limiting traversal to a depth of two levels.*

### Variadic matchers

At this moment, the tool supports two variadic matchers: `anyOf` and `allOf`, which can be conceptualized as matcher combinators, as one can group multiple matchers together to facilitate the construction of complex matchers.

Operator `anyOf` matches if any of the matchers in a given set match succeed (e.g `anyOf(m1, m2 ...)`). Using it brings several benefits, for example, one could construct a matcher that computes the union of two or more slices, initiating slice computation from one or multiple points of interest, or limiting slice computation by a set of inner matchers.

Operator `allOf`  matches only if all matchers in a set of matchers succeed (e.g `allOf(m1, m2 ...)`). For example, it enables finding all operations with a certain attribute, or initiating/limiting slice computation when an operation with a certain attribute is encountered.

#### Variadic matchers examples

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
<div style="overflow-x:auto; margin:1em 0;">
  <img
    src="https://i.imgur.com/qQhfyX4.gif"
    alt="backward-slice-union-anyof matcher"
    style="
      display: block;
      margin: 0 auto;
      width: 1250px !important;
      max-width: none !important;
      height: auto !important;
    "
  />
</div>

*Computes the union of two backward slices.*
<div style="overflow-x:auto; margin:1em 0;">
  <img
    src="https://i.imgur.com/b1EMdIv.gif"
    alt="backward-slice-union-anyof matcher"
    style="
      display: block;
      margin: 0 auto;
      width: 1250px !important;
      max-width: none !important;
      height: auto !important;
    "
  />
</div>

*Computes the forward slice by specifying the root operation using* `anyOf(hasOpName("memref.alloc"),isConstant())` *inner matcher and exit condition via* `anyOf(hasOpName("affine.load"),hasOpName("memref.dealloc"))` *inner matcher.*

```mlir
func.func @no_hoisting_collapse_shape(%in_0: memref<1x20x1xi32>, %1: memref<9x1xi32>, %vec: vector<4xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c20 = arith.constant 20 : index
  %alloca = memref.alloca() {alignment = 64 : i64} : memref<1x4x1xi32>
  scf.for %arg0 = %c0 to %c20 step %c4 {
    %subview = memref.subview %in_0[0, %arg0, 0] [1, 4, 1] [1, 1, 1] : memref<1x20x1xi32> to memref<1x4x1xi32, strided<[20, 1, 1], offset: ?>>
    %collapse_shape = memref.collapse_shape %alloca [[0, 1, 2]] : memref<1x4x1xi32> into memref<4xi32>
    vector.transfer_write %vec, %collapse_shape[%c0] {in_bounds = [true]} : vector<4xi32>, memref<4xi32>
    %read = vector.transfer_read %alloca[%c0, %c0, %c0], %c0_i32 {in_bounds = [true, true, true]} : memref<1x4x1xi32>, vector<1x4x1xi32>
    vector.transfer_write %read, %subview[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x1xi32>, memref<1x4x1xi32, strided<[20, 1, 1], offset: ?>>
  }
  return
}
```

<div style="overflow-x:auto; margin:1em 0;">
  <img
    src="https://i.imgur.com/MJnhvfD.gif"
    alt="backward-slice-union-anyof matcher"
    style="
      display: block;
      margin: 0 auto;
      width: 1250px !important;
      max-width: none !important;
      height: auto !important;
    "
  />
</div>

*Computes the forward slice by specifying the root operation using* `allOf(hasOpName("memref.alloca"),hasOpAttrName("alignment"))` *inner matcher and exit condition via* `hasOpName("vector.transfer_read")` *inner matcher.*

## Matcher Reference


| Matcher                                                                                                                       | Type                                                                                                                                 |
| ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `allOf`                                                                                                                       | `allOfVariadicOperator`                                                                                                              |
| `anyOf`                                                                                                                       | `anyOfVariadicOperator`                                                                                                              |
| [`getAllDefinitions`](https://mlir.llvm.org/doxygen/namespacemlir_1_1query_1_1matcher.html#a9a0dba8d855564b67517c778c915389f)         | [`BackwardSliceMatcher`](https://mlir.llvm.org/doxygen/classmlir_1_1query_1_1matcher_1_1BackwardSliceMatcher.html)                   |
| [`getDefinitions`](https://mlir.llvm.org/doxygen/namespacemlir_1_1query_1_1matcher.html#a9a0dba8d855564b67517c778c915389f)            | [`BackwardSliceMatcher`](https://mlir.llvm.org/doxygen/classmlir_1_1query_1_1matcher_1_1BackwardSliceMatcher.html)                   |
| [`getDefinitionsByPredicate`](https://mlir.llvm.org/doxygen/namespacemlir_1_1query_1_1matcher.html#a57916f218941284d7a5c8c912cd7d9f8) | [`PredicateBackwardSliceMatcher`](https://mlir.llvm.org/doxygen/classmlir_1_1query_1_1matcher_1_1PredicateBackwardSliceMatcher.html) |
| [`getUsersByPredicate`](https://mlir.llvm.org/doxygen/namespacemlir_1_1query_1_1matcher.html#a4cfbf14535ac0078e22cf89cafee1fd8)       | [`PredicateForwardSliceMatcher`](https://mlir.llvm.org/doxygen/classmlir_1_1query_1_1matcher_1_1PredicateForwardSliceMatcher.html)   |
| [`hasOpAttrName`](https://mlir.llvm.org/doxygen/structmlir_1_1detail_1_1AttrOpMatcher.html)                                   | [`AttrOpMatcher`](https://mlir.llvm.org/doxygen/structmlir_1_1detail_1_1AttrOpMatcher.html)                                          |
| [`hasOpName`](https://mlir.llvm.org/doxygen/namespacemlir.html#a69b52f968271c9a4da1bc766ee083a9c)                            | [`NameOpMatcher`](https://mlir.llvm.org/doxygen/structmlir_1_1detail_1_1NameOpMatcher.html)                                          |
| [`isConstantOp`](https://mlir.llvm.org/doxygen/namespacemlir.html#ad402a86ee4c9000c6fa1fceaddab560b)                         | [`constant_op_matcher`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Matchers.h#L182)                         |
| [`isNegInfFloat`](https://mlir.llvm.org/doxygen/namespacemlir.html#a9e89b015211525b010832d2d2c37650b)                        | [`constant_float_predicate_matcher`](https://mlir.llvm.org/doxygen/structmlir_1_1detail_1_1constant__float__predicate__matcher.html) |
| [`isNegZeroFloat`](https://mlir.llvm.org/doxygen/namespacemlir.html#aa9eba8d1292854c0da6c062988ecac9b)                       | [`constant_float_predicate_matcher`](https://mlir.llvm.org/doxygen/structmlir_1_1detail_1_1constant__float__predicate__matcher.html) |
| [`isNonZero`](https://mlir.llvm.org/doxygen/namespacemlir.html#a94bb42600b9be680591776fdc14a53cd)                            | [`constant_int_predicate_matcher`](https://mlir.llvm.org/doxygen/structmlir_1_1detail_1_1constant__int__predicate__matcher.html)     |
| [`isOne`](https://mlir.llvm.org/doxygen/namespacemlir.html#a907f415a4c803b15ef57db37cc732f39)                                | [`constant_int_predicate_matcher`](https://mlir.llvm.org/doxygen/structmlir_1_1detail_1_1constant__int__predicate__matcher.html)     |
| [`isOneFloat`](https://mlir.llvm.org/doxygen/namespacemlir.html#af0495d84f34cf3238a7741fa6974a485)                           | [`constant_float_predicate_matcher`](https://mlir.llvm.org/doxygen/structmlir_1_1detail_1_1constant__float__predicate__matcher.html) |
| [`isPosInfFloat`](https://mlir.llvm.org/doxygen/namespacemlir.html#adc93dfeaa35bda23b16591c462c335f6)                        | [`constant_float_predicate_matcher`](https://mlir.llvm.org/doxygen/structmlir_1_1detail_1_1constant__float__predicate__matcher.html) |
| [`isPosZeroFloat`](https://mlir.llvm.org/doxygen/namespacemlir.html#a774a1ae971f4ef00eb57389293dfe617)                       | [`constant_float_predicate_matcher`](https://mlir.llvm.org/doxygen/structmlir_1_1detail_1_1constant__float__predicate__matcher.html) |
| [`isZero`](https://mlir.llvm.org/doxygen/namespacemlir.html#a7f5d8af15bd8994b1a7abeaaacfe1b06)                               | [`constant_int_predicate_matcher`](https://mlir.llvm.org/doxygen/structmlir_1_1detail_1_1constant__int__predicate__matcher.html)     |
| [`isZeroFloat`](https://mlir.llvm.org/doxygen/namespacemlir.html#a8ea33aa665368d4f2108eb2d41c85111)                          | [`constant_float_predicate_matcher`](https://mlir.llvm.org/doxygen/structmlir_1_1detail_1_1constant__float__predicate__matcher.html) |