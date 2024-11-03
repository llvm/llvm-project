# Transform Dialect Tutorial

MLIR supports declarative specification for controlling compiler transformations
via the transform dialect. It allows one to request compiler transformations
using compiler IR itself, which can be embedded into the original IR that is
being transformed (similarly to pragmas) or supplied separately (similarly to
scheduling languages). This tutorial presents the concepts of the MLIR transform
dialect and related infrastructure. It will be accompanied by a practical
demonstration of three use scenarios:

- Composing transform dialect operations available in (upstream) MLIR to perform
  a sequence of optimizing transformations that results in efficient code for an
  MLIR linear algebra operation.
- Defining new transform dialect operations and adapting existing transformation
  code to work with the transform dialect infrastructure.
- Setting up and using the transform dialect infrastructure in a downstream
  out-of-tree project with custom dialects, transformations and passes.

After following the tutorial, one will be able to apply the transform dialect in
their work and extend it when necessary. Basic familiarity with MLIR is a
prerequisite. See [Toy tutorial](../Toy) for introduction to MLIR.

The tutorial is divided into the following chapters.

-  [Chapter #0](Ch0.md): A Primer on “Structured” Linalg Operations
-  [Chapter #1](Ch1.md): Combining Existing Transformations
-  [Chapter #2](Ch2.md): Adding a Simple New Transformation Operation
-  [Chapter #3](Ch3.md): More than Simple Transform Operations
-  [Chapter H](ChH.md): Reproducing Halide Schedule

The code corresponding to this tutorial is located under
`mlir/Examples/transform` and the corresponding tests in
`mlir/test/Examples/transform`.
