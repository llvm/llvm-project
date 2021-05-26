# Toy Tutorial

This tutorial runs through the implementation of a basic toy language on top of
MLIR. The goal of this tutorial is to introduce the concepts of MLIR; in
particular, how [dialects](../../LangRef.md/#dialects) can help easily support
language specific constructs and transformations while still offering an easy
path to lower to LLVM or other codegen infrastructure. This tutorial is based on
the model of the
[LLVM Kaleidoscope Tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html).

Another good source of introduction is the online [recording](https://www.youtube.com/watch?v=Y4SvqTtOIDk)
from the 2020 LLVM Dev Conference ([slides](https://llvm.org/devmtg/2020-09/slides/MLIR_Tutorial.pdf)).

This tutorial assumes you have cloned and built MLIR; if you have not yet done
so, see
[Getting started with MLIR](../../../getting_started/).

This tutorial is divided in the following chapters:

-   [Chapter #1](Ch-1.md): Introduction to the Toy language and the definition
    of its AST.
-   [Chapter #2](Ch-2.md): Traversing the AST to emit a dialect in MLIR,
    introducing base MLIR concepts. Here we show how to start attaching
    semantics to our custom operations in MLIR.
-   [Chapter #3](Ch-3.md): High-level language-specific optimization using
    pattern rewriting system.
-   [Chapter #4](Ch-4.md): Writing generic dialect-independent transformations
    with Interfaces. Here we will show how to plug dialect specific information
    into generic transformations like shape inference and inlining.
-   [Chapter #5](Ch-5.md): Partially lowering to lower-level dialects. We'll
    convert some of our high level language specific semantics towards a generic
    affine oriented dialect for optimization.
-   [Chapter #6](Ch-6.md): Lowering to LLVM and code generation. Here we'll
    target LLVM IR for code generation, and detail more of the lowering
    framework.
-   [Chapter #7](Ch-7.md): Extending Toy: Adding support for a composite type.
    We'll demonstrate how to add a custom type to MLIR, and how it fits in the
    existing pipeline.

The [first chapter](Ch-1.md) will introduce the Toy language and AST.
