# Using `mlir-opt`

`mlir-opt` is the command-line entry point for running passes and lowerings on MLIR code.
This tutorial will explain how to use `mlir-opt` to run passes, and explain
some details about MLIR's built-in dialects along the way.

Prerequisites:

- [Building MLIR from source](/getting_started/)

[TOC]

## Overview

We start with a brief summary of context that helps to frame
the uses of `mlir-opt` detailed in this article.
For a deeper dive on motivation and design,
see [the MLIR paper](https://arxiv.org/abs/2002.11054).

Two of the central concepts in MLIR are *dialects* and *lowerings*.
In traditional compilers, there is typically one "dialect,"
called an *intermediate representation*, or IR,
that is the textual or data-structural description of a program
within the scope of the compiler's execution.
For example, in GCC the IR is called GIMPLE,
and in LLVM it's called LLVM-IR.
Compilers typically convert an input program to their IR,
run optimization passes,
and then convert the optimized IR to machine code.

MLIR's philosophy is to split the job into smaller steps.
First, MLIR allows one to define many IRs called *dialects*,
some considered "high level" and some "low level,"
but each with a set of types, operations, metadata,
and semantics that defines what the operations do.
Different dialects may coexist in the same program.
Then, one writes a set of *lowering passes*
that incrementally converts different parts of the program
from higher level dialects to lower and lower dialects
until you get to machine code
(or, in many cases, LLVM, which finishes the job).
Along the way,
*optimizing passes* are run to make the code more efficient.
The main point here is that the high level dialects exist
*so that* they make it easy to write these important optimizing passes.

A central motivation for building MLIR
was to build the `affine` dialect,
which is designed to enable [polyhedral optimizations](https://polyhedral.info/)
for loop transformations.
Compiler engineers had previously implemented polyhedral optimizations
in LLVM and GCC (without an `affine` dialect),
and it was difficult because they had to reconstruct well-structured loop nests
from a much more complicated set of low-level operations.
Having a higher level `affine` dialect preserves the loop nest structure
at an abstraction layer that makes it easier to write optimizations,
and then discards it during lowering passes.

The `mlir-opt` tool can run both
optimization passes and lowerings,
though the final code generation
is performed by a different tool called `mlir-translate`.
In particular, `mlir-opt` consumes MLIR as input and produce MLIR as output,
while `mlir-translate` consumes MLIR as input
and produces non-MLIR program representations as output.

## Two example programs

Here are two MLIR programs.
The first defines a function that counts the leading zeroes of a 32-bit integer (`i32`)
using the [`math` dialect's](/docs/Dialects/MathOps/) `ctlz` operation.

```mlir
func.func @main(%arg0: i32) -> i32 {
  %0 = math.ctlz %arg0 : i32
  func.return %0 : i32
}
```

This shows the basic structure of an MLIR operation
([see here](https://mlir.llvm.org/docs/LangRef/#operations) for a more complete spec).
Variable names are prefixed with `%`,
functions by `@`,
and each variable/value in a program has a type,
often expressed after a colon.
In this case all the types are `i32`,
except for the function type which is `(i32) -> i32`
(not specified explicitly above, but you'll see it in the `func.call` later).

Each statement is anchored around an expression like `math.ctlz`
which specifies the dialect [`math`](https://mlir.llvm.org/docs/Dialects/MathOps/) via a namespace,
and the operation [`ctlz`](https://mlir.llvm.org/docs/Dialects/MathOps/#mathctlz-mathcountleadingzerosop) after the `.`.
The rest of the syntax of the operation
is determined by a parser defined by the dialect,
and so many operations will have different syntaxes.
In the case of `math.ctlz`,
the sole argument is an integer whose leading zeros are to be counted,
and the trailing ` : i32` denotes the output type storing the count.

It's important to note that [`func`](https://mlir.llvm.org/docs/Dialects/Func/) is itself a dialect,
and [`func.func`](https://mlir.llvm.org/docs/Dialects/Func/#funcfunc-funcfuncop) is an operation,
where the braces and the function's body is part of the syntax.
In MLIR a list of operations within braces is called a [*region*](https://mlir.llvm.org/docs/LangRef/#regions),
and an operation can have zero regions like `math.ctlz`,
one region like `func.func`,
or multiple regions like [`scf.if`](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfif-scfifop),
which has a region for each of its two control flow branches.

The second program is a sequence of loops
that exhibits poor cache locality.

```mlir
func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
  %0 = memref.alloc() : memref<10xf32>
  %1 = memref.alloc() : memref<10xf32>
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %arg2 = 0 to 10 {
    affine.store %cst, %0[%arg2] : memref<10xf32>
    affine.store %cst, %1[%arg2] : memref<10xf32>
  }
  affine.for %arg2 = 0 to 10 {
    %2 = affine.load %0[%arg2] : memref<10xf32>
    %3 = arith.addf %2, %2 : f32
    affine.store %3, %arg0[%arg2] : memref<10xf32>
  }
  affine.for %arg2 = 0 to 10 {
    %2 = affine.load %1[%arg2] : memref<10xf32>
    %3 = arith.mulf %2, %2 : f32
    affine.store %3, %arg1[%arg2] : memref<10xf32>
  }
  return
}
```

This program introduces some additional dialects.
The [`affine` dialect](https://mlir.llvm.org/docs/Dialects/Affine/) mentioned in the introduction
represents well-structured loop nests,
and the [`affine.for` operation](https://mlir.llvm.org/docs/Dialects/Affine/#affinefor-affineaffineforop)
whose region corresponds to the loop's body.
`affine.for` also showcases some custom-defined syntax
to represent the loop bounds and loop induction variable.
The [`memref` dialect](https://mlir.llvm.org/docs/Dialects/MemRef/)
defines types and operations related to memory management
with pointer semantics.
Note also that while `memref` has store and load operations,
`affine` has its own that limit what types of memory accesses are allowed,
so as to ensure the well-structuredness of the loop nest.

## Running `mlir-opt`

After building the MLIR project,
the `mlir-opt` binary (located in `build/bin`)
is the entry point for running passes and lowerings,
as well as emitting debug and diagnostic data.

Running `mlir-opt` with no flags will consume MLIR input
from standard in, parse and run verifiers on it,
and write the MLIR back to standard out.
This is a good way to test if an input MLIR is well-formed.

`mlir-opt --help` shows a complete list of flags
(there are nearly 1000).
Each pass gets its own flag.

## Lowering `ctlz`

Next we will show two of MLIR's lowering passes.
The first, `convert-math-to-llvm`, converts the `ctlz` op
to the `llvm` dialect's [`intr.ctlz` op](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrctlz-llvmcountleadingzerosop)
which is an LLVM intrinsic.
Note that `llvm` here is MLIR's `llvm` dialect,
which would still need to be processed through `mlir-translate`
to generate LLVM-IR.

Recall our ctlz program:

```mlir
# mlir/test/Examples/mlir-opt/ctlz.mlir
func.func @main(%arg0: i32) -> i32 {
  %0 = math.ctlz %arg0 : i32
  func.return %0 : i32
}
```

After building MLIR, and from the `llvm-project` base directory, run

```bash
build/bin/mlir-opt --convert-math-to-llvm mlir/test/Examples/mlir-opt/ctlz.mlir
```

which produces

```mlir
module {
  func.func @main(%arg0: i32) -> i32 {
    %0 = "llvm.intr.ctlz"(%arg0) <{is_zero_poison = false}> : (i32) -> i32
    return %0 : i32
  }
}
```

As you can see, the `math` dialect was the only thing that changed
due to the lowering.
The `func` dialect was left alone,
even though it also needs to be converted to the `llvm` dialect
to generate LLVM-IR.

What if the target machine does not have a `ctlz` intrinsic?
In this case, one can run the `--convert-math-to-funcs`
pass, which replaces the op with an implementation using
other MLIR dialects.

```bash
build/bin/mlir-opt --convert-math-to-funcs=convert-ctlz mlir/test/Examples/mlir-opt/ctlz.mlir
```

You will see something similar to:

```mlir
module {
  func.func @main(%arg0: i32) -> i32 {
    %0 = call @__mlir_math_ctlz_i32(%arg0) : (i32) -> i32
    return %0 : i32
  }
  func.func private @__mlir_math_ctlz_i32(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi eq, %arg0, %c0_i32 : i32
    %1 = scf.if %0 -> (i32) {
      scf.yield %c32_i32 : i32
    } else {
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c32 = arith.constant 32 : index
      %c0_i32_0 = arith.constant 0 : i32
      %2:2 = scf.for %arg1 = %c1 to %c32 step %c1 iter_args(%arg2 = %arg0, %arg3 = %c0_i32_0) -> (i32, i32) {
        %3 = arith.cmpi slt, %arg2, %c0_i32 : i32
        %4:2 = scf.if %3 -> (i32, i32) {
          scf.yield %arg2, %arg3 : i32, i32
        } else {
          %5 = arith.addi %arg3, %c1_i32 : i32
          %6 = arith.shli %arg2, %c1_i32 : i32
          scf.yield %6, %5 : i32, i32
        }
        scf.yield %4#0, %4#1 : i32, i32
      }
      scf.yield %2#1 : i32
    }
    return %1 : i32
  }
}
```

The algorithm above is not relevant to this tutorial,
but either way it is quite simple:
count the leading zeros by shifting the input left one bit at a time
until it becomes negative (as a signed integer),
because that occurs exactly when its leading bit is a 1.
Then add a special case to handle zero,
which would loop infinitely otherwise.

Here you can see two more MLIR dialects.
[`arith`](https://mlir.llvm.org/docs/Dialects/ArithOps/)
is for low-level arithmetic
and boolean conditions on integers and floats.
You can define constants,
compare integers with `arith.cmpi`,
and do things like add and bit shift (`arith.shli` is a left shift).
[`scf`](https://mlir.llvm.org/docs/Dialects/SCFDialect/),
short for "structured control flow,"
defines for loops, while loops,
and control flow branching using regions.
`scf.yield` defines the "output" value
from each region of an if/else operation or loop body
which is necessary here because an `if` operation has a result value.
The "structured" in `scf` is in contrast to
[`cf`](https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/),
for "(unstructured) control flow,"
which does low-level, region-free control flow
that jumps between SSA blocks on a control flow graph.

Two other minor aspects of the syntax are on display.
First is the syntax `%4:2`,
which defines a variable `%4` as a tuple of two values.
The corresponding `%4#1` accesses the second entry in the tuple.
Second, there's a type called `index` that is different from `i32`.
Though they both represent integers,
[`index`](https://mlir.llvm.org/docs/Rationale/Rationale/#integer-signedness-semantics) is a platform-dependent integer type
suitable for indexing arrays,
representing sizes and dimensions of things,
and, in the above program,
being loop counters and iteration bounds.

## Optimizing loop nests

Recall our second program, the poorly-tuned loops.

```mlir
// mlir/test/Examples/mlir-opt/loop_fusion.mlir
func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
  %0 = memref.alloc() : memref<10xf32>
  %1 = memref.alloc() : memref<10xf32>
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %arg2 = 0 to 10 {
    affine.store %cst, %0[%arg2] : memref<10xf32>
    affine.store %cst, %1[%arg2] : memref<10xf32>
  }
  affine.for %arg2 = 0 to 10 {
    %2 = affine.load %0[%arg2] : memref<10xf32>
    %3 = arith.addf %2, %2 : f32
    affine.store %3, %arg0[%arg2] : memref<10xf32>
  }
  affine.for %arg2 = 0 to 10 {
    %2 = affine.load %1[%arg2] : memref<10xf32>
    %3 = arith.mulf %2, %2 : f32
    affine.store %3, %arg1[%arg2] : memref<10xf32>
  }
  return
}
```

Running this with the [`affine-loop-fusion`](https://mlir.llvm.org/docs/Passes/#-affine-loop-fusion) pass
produces a fused loop.

```bash
build/bin/mlir-opt --affine-loop-fusion mlir/test/Examples/mlir-opt/loop_fusion.mlir
```

```mlir
module {
  func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
    %alloc = memref.alloc() : memref<1xf32>
    %alloc_0 = memref.alloc() : memref<1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg2 = 0 to 10 {
      affine.store %cst, %alloc[0] : memref<1xf32>
      affine.store %cst, %alloc_0[0] : memref<1xf32>
      %0 = affine.load %alloc_0[0] : memref<1xf32>
      %1 = arith.mulf %0, %0 : f32
      affine.store %1, %arg1[%arg2] : memref<10xf32>
      %2 = affine.load %alloc[0] : memref<1xf32>
      %3 = arith.addf %2, %2 : f32
      affine.store %3, %arg0[%arg2] : memref<10xf32>
    }
    return
  }
}
```

This pass has options that allow the user to configure its behavior.
For example, the `fusion-compute-tolerance` option
is described as the "fractional increase in additional computation tolerated while fusing."
If this value is set to zero on the command line,
the pass will not fuse the loops.

```bash
build/bin/mlir-opt --affine-loop-fusion='fusion-compute-tolerance=0' mlir/test/Examples/mlir-opt/loop_fusion.mlir
```

```mlir
module {
  func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
    %alloc = memref.alloc() : memref<10xf32>
    %alloc_0 = memref.alloc() : memref<10xf32>
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg2 = 0 to 10 {
      affine.store %cst, %alloc[%arg2] : memref<10xf32>
      affine.store %cst, %alloc_0[%arg2] : memref<10xf32>
    }
    affine.for %arg2 = 0 to 10 {
      %0 = affine.load %alloc[%arg2] : memref<10xf32>
      %1 = arith.addf %0, %0 : f32
      affine.store %1, %arg0[%arg2] : memref<10xf32>
    }
    affine.for %arg2 = 0 to 10 {
      %0 = affine.load %alloc_0[%arg2] : memref<10xf32>
      %1 = arith.mulf %0, %0 : f32
      affine.store %1, %arg1[%arg2] : memref<10xf32>
    }
    return
  }
}
```

Options passed to a pass
should come in the form of a quoted string
(to join all options into a single shell argument)
with space-separated `key=value` pairs.

## Building a pass pipeline on the command line

One can combine passes on the command line in two ways.

First, by simply placing the pass flags one after the other,
they will be run in order.

```bash
build/bin/mlir-opt --convert-math-to-llvm --convert-func-to-llvm mlir/test/Examples/mlir-opt/ctlz.mlir
```

Passes can also be configured to run
in a way that is limited to a particular sub-IR
nested under scope-isolated ops like functions.
For example, one could run `--convert-math-to-llvm`
on each `func` separately, by running

```bash
build/bin/mlir-opt mlir/test/Examples/mlir-opt/ctlz.mlir --pass-pipeline='
    builtin.module(
        convert-math-to-funcs{convert-ctlz=1},
        func.func(cse,canonicalize),
        convert-scf-to-cf,
        convert-to-llvm
    )'
```

The outer nesting tells `mlir-opt` to run the pass pipeline
on each `module` op,
and then within that to run `convert-math-to-funcs`,
then (on each `func.func` op), the [`cse`](https://mlir.llvm.org/docs/Passes/#-cse)
and [`canonicalize`](https://mlir.llvm.org/docs/Passes/#-canonicalize) passes,
and then convert the rest to the `llvm` dialect.

For a spec of the pass-pipeline textual description language,
see [the docs](https://mlir.llvm.org/docs/PassManagement/#textual-pass-pipeline-specification).

## Further readering

- [List of passes](https://mlir.llvm.org/docs/Passes/)
- [List of dialects](https://mlir.llvm.org/docs/Dialects/)
