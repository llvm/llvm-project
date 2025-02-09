# Using `mlir-opt`

`mlir-opt` is a command-line entry point for running passes and lowerings on MLIR code.
This tutorial will explain how to use `mlir-opt`, show some examples of its usage,
and mention some useful tips for working with it.

Prerequisites:

- [Building MLIR from source](/getting_started/)
- [MLIR Language Reference](/docs/LangRef/)

[TOC]

## `mlir-opt` basics

The `mlir-opt` tool loads a textual IR or bytecode into an in-memory structure,
and optionally executes a sequence of passes
before serializing back the IR (textual form by default).
It is intended as a testing and debugging utility.

After building the MLIR project,
the `mlir-opt` binary (located in `build/bin`)
is the entry point for running passes and lowerings,
as well as emitting debug and diagnostic data.

Running `mlir-opt` with no flags will consume textual or bytecode IR
from the standard input, parse and run verifiers on it,
and write the textual format back to the standard output.
This is a good way to test if an input MLIR is well-formed.

`mlir-opt --help` shows a complete list of flags
(there are nearly 1000).
Each pass has its own flag,
though it is recommended to use `--pass-pipeline`
to run passes rather than bare flags.

## Running a pass

Next we run [`convert-to-llvm`](/docs/Passes/#-convert-to-llvm),
which converts all supported dialects to the `llvm` dialect,
on the following IR:

```mlir
// mlir/test/Examples/mlir-opt/ctlz.mlir
module {
  func.func @main(%arg0: i32) -> i32 {
    %0 = math.ctlz %arg0 : i32
    func.return %0 : i32
  }
}
```

After building MLIR, and from the `llvm-project` base directory, run

```bash
build/bin/mlir-opt --pass-pipeline="builtin.module(convert-math-to-llvm)" mlir/test/Examples/mlir-opt/ctlz.mlir
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

Note that `llvm` here is MLIR's `llvm` dialect,
which would still need to be processed through `mlir-translate`
to generate LLVM-IR.

## Running a pass with options

Next we will show how to run a pass that takes configuration options.
Consider the following IR containing loops with poor cache locality.

```mlir
// mlir/test/Examples/mlir-opt/loop_fusion.mlir
module {
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
}
```

Running this with the [`affine-loop-fusion`](/docs/Passes/#-affine-loop-fusion) pass
produces a fused loop.

```bash
build/bin/mlir-opt --pass-pipeline="builtin.module(affine-loop-fusion)" mlir/test/Examples/mlir-opt/loop_fusion.mlir
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
build/bin/mlir-opt --pass-pipeline="builtin.module(affine-loop-fusion{fusion-compute-tolerance=0})" \
mlir/test/Examples/mlir-opt/loop_fusion.mlir
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
are specified via the syntax `{option1=value1 option2=value2 ...}`,
i.e., use space-separated `key=value` pairs for each option.

## Building a pass pipeline on the command line

The `--pass-pipeline` flag supports combining multiple passes into a pipeline.
So far we have used the trivial pipeline with a single pass
that is "anchored" on the top-level `builtin.module` op.
[Pass anchoring](/docs/PassManagement/#oppassmanager)
is a way for passes to specify
that they only run on particular ops.
While many passes are anchored on `builtin.module`,
if you try to run a pass that is anchored on some other op
inside `--pass-pipeline="builtin.module(pass-name)"`,
it will not run.

Multiple passes can be chained together
by providing the pass names in a comma-separated list
in the `--pass-pipeline` string,
e.g.,
`--pass-pipeline="builtin.module(pass1,pass2)"`.
The passes will be run sequentially.

To use passes that have nontrivial anchoring,
the appropriate level of nesting must be specified
in the pass pipeline.
For example, consider the following IR which has the same redundant code,
but in two different levels of nesting.

```mlir
module {
  module {
    func.func @func1(%arg0: i32) -> i32 {
      %0 = arith.addi %arg0, %arg0 : i32
      %1 = arith.addi %arg0, %arg0 : i32
      %2 = arith.addi %0, %1 : i32
      func.return %2 : i32
    }
  }

  gpu.module @gpu_module {
    gpu.func @func2(%arg0: i32) -> i32 {
      %0 = arith.addi %arg0, %arg0 : i32
      %1 = arith.addi %arg0, %arg0 : i32
      %2 = arith.addi %0, %1 : i32
      gpu.return %2 : i32
    }
  }
}
```

The following pipeline runs `cse` (common subexpression elimination)
but only on the `func.func` inside the two `builtin.module` ops.

```bash
build/bin/mlir-opt mlir/test/Examples/mlir-opt/ctlz.mlir --pass-pipeline='
    builtin.module(
        builtin.module(
            func.func(cse,canonicalize),
            convert-to-llvm
        )
    )'
```

The output leaves the `gpu.module` alone

```mlir
module {
  module {
    llvm.func @func1(%arg0: i32) -> i32 {
      %0 = llvm.add %arg0, %arg0 : i32
      %1 = llvm.add %0, %0 : i32
      llvm.return %1 : i32
    }
  }
  gpu.module @gpu_module {
    gpu.func @func2(%arg0: i32) -> i32 {
      %0 = arith.addi %arg0, %arg0 : i32
      %1 = arith.addi %arg0, %arg0 : i32
      %2 = arith.addi %0, %1 : i32
      gpu.return %2 : i32
    }
  }
}
```

Specifying a pass pipeline with nested anchoring
is also beneficial for performance reasons:
passes with anchoring can run on IR subsets in parallel,
which provides better threaded runtime and cache locality
within threads.
For example,
even if a pass is not restricted to anchor on `func.func`,
running `builtin.module(func.func(cse, canonicalize))`
is more efficient than `builtin.module(cse, canonicalize)`.

For a spec of the pass-pipeline textual description language,
see [the docs](/docs/PassManagement/#textual-pass-pipeline-specification).
For more general information on pass management, see [Pass Infrastructure](/docs/PassManagement/#).

## Useful CLI flags

- `--debug` prints all debug information produced by `LLVM_DEBUG` calls.
- `--debug-only="my-tag"` prints only the debug information produced by `LLVM_DEBUG`
  in files that have the macro `#define DEBUG_TYPE "my-tag"`.
  This often allows you to print only debug information associated with a specific pass.
    - `"greedy-rewriter"` only prints debug information
      for patterns applied with the greedy rewriter engine.
    - `"dialect-conversion"` only prints debug information
      for the dialect conversion framework.
 - `--emit-bytecode` emits MLIR in the bytecode format.
 - `--mlir-pass-statistics` print statistics about the passes run.
    These are generated via [pass statistics](/docs/PassManagement/#pass-statistics).
 - `--mlir-print-ir-after-all` prints the IR after each pass.
    - See also `--mlir-print-ir-after-change`, `--mlir-print-ir-after-failure`,
      and analogous versions of these flags with `before` instead of `after`.
    - When using `print-ir` flags, adding `--mlir-print-ir-tree-dir` writes the
      IRs to files in a directory tree, making them easier to inspect versus a
      large dump to the terminal.
 - `--mlir-timing` displays execution times of each pass.

## Further readering

- [List of passes](/docs/Passes/)
- [List of dialects](/docs/Dialects/)
