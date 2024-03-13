# Standalone Transform Dialect Interpreter

This is an example of using the Transform dialect interpreter functionality standalone, that is, outside of the regular pass pipeline. The example is a
binary capable of processing MLIR source files similar to `mlir-opt` and other
optimizer drivers, with the entire transformation process driven by a Transform
dialect script. This script can be embedded into the source file or provided in
a separate MLIR source file.

Either the input module or the transform module must contain a top-level symbol
named `__transform_main`, which is used as the entry point to the transformation
script.

```sh
mlir-transform-opt payload_with_embedded_transform.mlir
mlir-transform-opt payload.mlir -transform=transform.mlir
```

The name of the entry point can be overridden using command-line options.

```sh
mlir-transform-opt payload-mlir -transform-entry-point=another_entry_point
```

Transform scripts can reference symbols defined in other source files, called
libraries, which can be supplied to the binary through command-line options.
Libraries will be embedded into the main transformation module by the tool and
the interpreter will process everything as a single module. A debug option is
available to see the contents of the transform module before it goes into the interpreter.

```sh
mlir-transform-opt payload.mlir -transform=transform.mlir \
  -transform-library=external_definitions_1.mlir \
  -transform-library=external_definitions_2.mlir \
  -dump-library-module
```

Check out the [Transform dialect
tutorial](https://mlir.llvm.org/docs/Tutorials/transform/) as well as
[documentation](https://mlir.llvm.org/docs/Dialects/Transform/) to learn more
about the dialect. 
