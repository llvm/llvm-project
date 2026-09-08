# CMake Infrastructure

[TOC]

MLIR extends LLVM's CMake helpers in
[`AddMLIR.cmake`](../cmake/modules/AddMLIR.cmake). This document describes how
MLIR libraries declare their dependencies, in particular on TableGen-generated
headers. Getting these right matters: a missing dependency does not fail a
warm build, it only shows up as a race in a clean, parallel build.

## Libraries

`add_mlir_library` and its wrappers (`add_mlir_dialect_library`,
`add_mlir_conversion_library`, `add_mlir_extension_library`,
`add_mlir_translation_library`) build on `llvm_add_library` and accept the
same keywords, plus `HEADER_LIBS`:

~~~cmake
add_mlir_dialect_library(MLIRFooDialect
  FooDialect.cpp
  FooOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Foo

  DEPENDS
  MLIRFooOpsIncGen
  MLIRFooInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBarDialect
  )
~~~

## TableGen targets

Set `LLVM_TARGET_DEFINITIONS`, call `mlir_tablegen` once per output, then
create the target:

| Helper | Use for |
| --- | --- |
| `add_mlir_dialect` | The standard op, type and dialect declarations of a dialect |
| `add_mlir_dialect_tablegen_target` | Other dialect-specific headers |
| `add_mlir_interface`, `add_mlir_generic_tablegen_target` | Dialect-independent headers such as interfaces and pass registries |
| `add_public_tablegen_target` | Files private to one library, such as rewrite patterns |

Dialect-independent targets are collected in `mlir-generic-headers`, which
every MLIR library depends on. Dialect-specific targets are collected in
`mlir-headers`; do not depend on that aggregate, it serializes every dialect
behind every generator.

## Generated header dependencies

Three rules cover every generated header in MLIR:

1. **A library lists its own TableGen targets in `DEPENDS`.** These are the
   targets producing the `.inc` files that its sources or public headers
   include, normally the ones declared in the matching `include/` directory.

2. **Linking a library orders compilation after that library's generated
   headers.** Every `LINK_LIBS` entry, `PUBLIC` or `PRIVATE`, becomes a
   target-level ordering dependency of the object library, and Ninja follows
   those dependencies transitively without waiting for archives to be linked.
   Never list another library's `*IncGen` target: link the library that owns it.

3. **`HEADER_LIBS` names libraries whose generated headers are included
   without linking them.** Compilation is ordered after the TableGen targets of
   that library and of the libraries it links, but not after the libraries
   themselves, so this cannot create a dependency cycle when the provider links
   the consumer. Add a comment naming the include that requires the edge:

   ~~~cmake
   # ArithDialect.cpp includes the generated Bufferization interfaces.
   HEADER_LIBS
   MLIRBufferizationDialect
   ~~~

   A misspelled entry is diagnosed at configure time.

Generated LLVM headers are not covered by these rules; libraries that include
them still list `intrinsics_gen` or the relevant LLVM target in `DEPENDS`.

## Checking dependencies

A missing dependency is only visible from a clean build. After configuring a
fresh Ninja build directory, build a few leaf libraries in parallel rather
than an aggregate target, for example:

~~~sh
ninja -C <build> MLIRArmNeonDialect MLIRLinalgDialect
~~~

An aggregate such as `check-mlir` often generates headers incidentally and
hides the race. After a successful build has populated the compiler depfiles,
`ninja -C <build> -t missingdeps` lists every generated file that an object
includes without a path to its generator in the build graph.
