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

The main motivation for building MLIR
was to build the `affine` dialect,
which is designed to enable [polyhedral optimizations](https://polyhedral.info/)
for loop transformations.
Compiler engineers had previously implemented polyhedral optimizations in LLVM and GCC (without an `affine` dialect),
and it was difficult because they had to take a reconstruct a well-structured loop nest
from a much more complicated set of low-level operations.
MLIR instead keeps the structure in the higher level operations for optimizations,
and then discards it during lowering passes.

The `mlir-opt` tool can run both
optimization passes and lowerings,
though the final code generation
is performed by a different tool called `mlir-translate`.
In particular, `mlir-opt` consumes MLIR as input and produce MLIR as output,
while `mlir-translate` consumes MLIR as input
and produces non-MLIR program representations as output.

## Two example programs

Here are two MLIR programs
that define a function that counts the leading zeroes of a 32-bit integer (`i32`).
The first uses the [`math` dialect's](/docs/Dialects/MathOps/) `ctlz` operation and just returns the result.

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
and [`func.func`](https://mlir.llvm.org/docs/Dialects/Func/#funcfunc-funcfuncop) is considered an "operation,"
where the braces and the function's body is part of the syntax.
In MLIR a list of operations within braces is called a [*region*](https://mlir.llvm.org/docs/LangRef/#regions),
and an operation can have zero regions like `math.ctlz`,
one region like `func.func`,
or multiple regions like [`scf.if`](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfif-scfifop),
which has a region for each of its two control flow branches.

The second program is a sequence of two loops
that exhibit poor cache locality.

```mlir

```

## Lowering `ctlz`

The second version of this program has a software implementation of the `ctlz` function and calls it.

```mlir
func.func @main(%arg0: i32) -> i32 {
  %0 = func.call @my_ctlz(%arg0) : (i32) -> i32
  func.return %0 : i32
}
func.func @my_ctlz(%arg0: i32) -> i32 {
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
  func.return %1 : i32
}
```

The algorithm above is not relevant to this post, but either way it is quite simple: count the leading zeros by shifting the input left one bit at a time until it becomes negative (as a signed integer), because that occurs exactly when its leading bit is a 1. Then add a special case to handle zero, which would loop infinitely otherwise.

Here you can see two more MLIR dialects. [`arith`](https://mlir.llvm.org/docs/Dialects/ArithOps/) is for low-level arithmetic and boolean conditions on integers and floats. You can define constants, compare integers with `arith.cmpi`, and do things like add and bit shift (`arith.shli` is a left shift). [`scf`](https://mlir.llvm.org/docs/Dialects/SCFDialect/), short for "structured control flow," defines for loops, while loops, and control flow branching. `scf.yield` defines the "output" value from each region of an if/else operation or loop body which is necessary here because, as you can see, an `if` operation has a result value.

Two other minor aspects of the syntax are on display. First is the syntax `%4:2`, which defines a variable `%4` which is a tuple of two values. The corresponding `%4#1` accesses the second entry in the tuple. Second, you'll notice there's a type called `index` that is different from `i32`. Though they both represent integers, `index` is intended to be a platform-dependent integer type which is suitable for indexing arrays, representing sizes and dimensions of things, and, in our case, being loop counters and iteration bounds. More details on [`index` in the MLIR docs](https://mlir.llvm.org/docs/Rationale/Rationale/#integer-signedness-semantics).

## Lowerings and the math-to-funcs pass

We have two versions of the same program because one is a lowered version of the other. In most cases, the machine you're going to run a program on has a "count leading zeros" function, so the lowering would simply map `math.ctlz` to the corresponding machine instruction. But if there is no `ctlz` instruction, a lowering can provide an implementation in terms of lower level dialects and ops. Specifically, this one lowers ctlz to {`func`, `arith`, `scf`}.

The second version of this code was actually generated by the `mlir-opt` command line tool, which is the main entry-point to running MLIR passes on specific MLIR code. For starters, one can take the `mlir-opt` tool and run it with no arguments on any MLIR code, and it will parse it, verify it is well formed, and print it back out with some slight normalizations. In this case, it will wrap the code in a `module`, which is a namespace isolation mechanism.

```bash
$ echo 'func.func @main(%arg0: i32) -> i32 {
  %0 = math.ctlz %arg0 : i32
  func.return %0 : i32
}' > ctlz.mlir
$ bazel run @llvm-project//mlir:mlir-opt -- $(pwd)/ctlz.mlir
<... snip ...>
module {
  func.func @main(%arg0: i32) -> i32 {
    %0 = math.ctlz %arg0 : i32
    return %0 : i32
  }
}
```

**Aside:** The `-- $(pwd)/ctlz.mlir` is a quirk of bazel. When one program runs another program, the `--` is the standard mechanism to separate CLI flags from the runner program (`bazel`) and the run program (`mlir-opt`). Everything after `--` goes to `mlir-opt`. Also, the need for `$(pwd)` is because when bazel runs `mlir-opt`, it runs it with a working directory that is in some temporary, isolated location on the filesystem. So we need to give it an absolute path to the MLIR file to input. Or we could pipe from standard in. Or we could run the `mlir-opt` binary directly from `bazel-bin/external/llvm-project/mlir/mlir-opt`.

Next we can run our first lowering, which is already built-in to `mlir-opt`, and which generates the long program above.

```bash
$ bazel run @llvm-project//mlir:mlir-opt -- --convert-math-to-funcs=convert-ctlz $(pwd)/ctlz.mlir
<... snip ...>
module {
  func.func @main(%arg0: i32) {
    %0 = call @__mlir_math_ctlz_i32(%arg0) : (i32) -> i32
    return
  }
  func.func private @__mlir_math_ctlz_i32(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
<... snip ...>
```

Each pass gets its own command line flag, some are grouped into pipelines, and the `--pass-pipeline` command line flag can be used to provide a (serialized version of) an ordered list of passes to run on the input MLIR.1

We won't cover the internal workings of the `math-to-funcs` pass in this or a future article, but next time we will actually write our own, simpler pass that does something nontrivial. Until then, I'll explain a bit about how testing works in MLIR, using these two ctlz programs as example test cases.

For those who are interested, the MLIR documentation contains a [complete list of passes](https://mlir.llvm.org/docs/Passes/) owned by the upstream MLIR project, which can be used by invoking the corresponding command line flag or nesting it inside of a larger `--pass-pipeline`.

## Optimizing `affine.for`
