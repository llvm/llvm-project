# CMake Infrastructure

[TOC]

MLIR extends LLVM's CMake infrastructure with helpers for TableGen, libraries,
dialects, interfaces, tools, installation, and aggregate libraries. This guide
describes the MLIR-specific conventions implemented by
[`AddMLIR.cmake`](../cmake/modules/AddMLIR.cmake). The LLVM CMake documentation
still applies to the underlying LLVM helpers.

## Loading the MLIR CMake modules

The monorepo build loads the required modules. An out-of-tree project using an
installed MLIR package normally starts with:

~~~cmake
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
~~~

The project under `mlir/examples/standalone` is the canonical out-of-tree
template. It demonstrates package discovery, generated files, libraries,
tools, tests, and installation without depending on the monorepo build.

## Source and generated-file layout

Public declarations normally live below `mlir/include/mlir`, with
implementations in the corresponding directory below `mlir/lib`. For example,
a dialect declared in `include/mlir/Dialect/Foo/IR` is normally implemented by
a library in `lib/Dialect/Foo/IR`.

Generated target names such as `MLIRFooOpsIncGen` are build-tree details. The
library that compiles or publishes those generated files lists the target in
its own `DEPENDS`. Other libraries normally depend on the logical library
through `LINK_LIBS`, not on its private generation target.

Generated source files are different from published generated headers. A
source-generation target, such as one created for sharded operations, remains
an explicit `DEPENDS` entry of the library compiling those sources.

## TableGen

Set `LLVM_TARGET_DEFINITIONS`, call `mlir_tablegen` once for each output, and
finish with the helper matching the output:

| Helper | Intended output |
| --- | --- |
| `add_mlir_dialect` | Standard operation, type, and dialect fragments |
| `add_mlir_dialect_tablegen_target` | Other dialect-specific headers |
| `add_mlir_generic_tablegen_target` | Dialect-independent headers |
| `add_public_tablegen_target` | A library-specific header or source |

For example:

~~~cmake
set(LLVM_TARGET_DEFINITIONS FooPatterns.td)
mlir_tablegen(FooPatterns.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRFooPatternsIncGen)
~~~

The library that includes `FooPatterns.h.inc` then lists
`MLIRFooPatternsIncGen` in `DEPENDS`.

`mlir-generic-headers` collects dialect-independent public generation targets,
and every MLIR library depends on it. `mlir-headers` also includes
dialect-specific targets. The latter is a conservative compatibility aggregate
and should not replace precise library dependencies in new code.

### Dialects

The common dialect declaration is:

~~~cmake
add_mlir_dialect(FooOps foo)
~~~

This generates operation, type, and dialect declaration and definition
fragments and creates `MLIRFooOpsIncGen`. The implementation library lists that
target explicitly:

~~~cmake
add_mlir_dialect_library(MLIRFooDialect
  FooDialect.cpp
  FooOps.cpp

  DEPENDS
  MLIRFooOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
~~~

### Operation, type, and attribute interfaces

`add_mlir_interface(FooOpInterface)` emits operation-interface declaration and
definition fragments. `add_mlir_type_interface(FooTypeInterface)` does the same
for a type interface. Attribute interfaces and specialized interface forms use
the corresponding `mlir_tablegen` generators followed by a dialect or generic
TableGen target.

The library implementing an interface lists its generation target in
`DEPENDS`. A consumer links that interface library when its public or private
C++ interface uses the generated declarations.

### Passes

Pass declarations use `-gen-pass-decls`; C API fragments may be emitted from
the same `.td` file:

~~~cmake
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Foo)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Foo)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Foo)
add_mlir_dialect_tablegen_target(MLIRFooPassIncGen)
~~~

The library that defines or publishes these passes keeps
`MLIRFooPassIncGen` in `DEPENDS`. A header-only consumer uses the logical pass
library through `HEADER_LIBS` only when linking it would be incorrect.

### PDLL and generated documentation

Use `add_mlir_pdll_library` to compile a PDLL source and make its generated
output available to another target. Use `add_mlir_doc` for generated dialect,
operation, type, attribute, interface, or pass documentation. Documentation
targets are collected under the `mlir-doc` aggregate and are not compilation
prerequisites unless a source target explicitly consumes their output.

## Libraries

`add_mlir_library` is the base helper for MLIR libraries:

~~~cmake
add_mlir_library(MLIRFooTransforms
  FooTransforms.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Foo

  DEPENDS
  MLIRFooTransformsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRFooDialect
  MLIRPass
  )
~~~

`LINK_COMPONENTS` names LLVM components. `LINK_LIBS` names CMake or MLIR
library targets. Keeping them separate allows LLVM and MLIR to substitute their
monolithic shared libraries correctly.

Frequently used options include:

| Option | Purpose |
| --- | --- |
| `SHARED` or `OBJECT` | Select a non-default library form |
| `INSTALL_WITH_TOOLCHAIN` | Install with the toolchain distribution |
| `EXCLUDE_FROM_LIBMLIR` | Exclude the library from monolithic MLIR |
| `DISABLE_INSTALL` | Omit standard installation rules |
| `ENABLE_AGGREGATION` | Make objects available to an MLIR aggregate |
| `STANDALONE` | Do not add the implicit `LLVMSupport` dependency |
| `ADDITIONAL_HEADERS` | Associate individual headers with the target |
| `ADDITIONAL_HEADER_DIRS` | Add public headers to IDE source groups |
| `DEPENDS` | Add the library's non-library build prerequisites |
| `HEADER_LIBS` | Order after generated headers without linking |

Prefer the wrapper describing a library's role:

| Helper | Additional behavior |
| --- | --- |
| `add_mlir_dialect_library` | Records a target in `MLIR_DIALECT_LIBS` |
| `add_mlir_conversion_library` | Records it in `MLIR_CONVERSION_LIBS` |
| `add_mlir_extension_library` | Records it in `MLIR_EXTENSION_LIBS` |
| `add_mlir_translation_library` | Records it in `MLIR_TRANSLATION_LIBS` |
| `add_mlir_example_library` | Applies the conventions to examples |
| `add_mlir_public_c_api_library` | Creates an aggregatable C API library |

The global categories support tools and aggregates that intentionally collect
an entire class of libraries. Ordinary libraries should list only their actual
dependencies.

### Link visibility

Choose `LINK_LIBS` visibility from the C++ interface:

| Declaration | Meaning |
| --- | --- |
| `PUBLIC A` | The target and its consumers use `A` |
| `PRIVATE A` | Only the target implementation uses `A` |
| `INTERFACE A` | Only consumers use `A` |

Unqualified entries retain CMake's legacy signature behavior. New code should
use explicit visibility when the distinction matters.

Use `mlir_target_link_libraries` when adding links after an MLIR target was
created, particularly for a library excluded from `libMLIR`. It applies
`MLIR_LINK_MLIR_DYLIB` substitution and also records the generated-header
ordering needed by an object-backed library.

### Generated-header dependencies

Three rules cover generated headers:

1. A library lists its own TableGen and generated-source targets in `DEPENDS`.
2. `LINK_LIBS` orders compilation after the generated-header targets reachable
   through the linked library's interface. PUBLIC, PRIVATE, and unqualified
   direct links are considered; INTERFACE-only links are not used by the
   current target's compilation.
3. `HEADER_LIBS` names a library whose generated header is included without a
   link relationship.

`HEADER_LIBS` is a rare layering escape hatch. It is appropriate when linking
the provider would be semantically wrong or would create a circular link
relationship. Keep it a flat list of literal target names and document the
source-level include that requires every entry:

~~~cmake
# FooAnalysis.cpp includes mlir/Dialect/Bar/IR/BarOps.h.
HEADER_LIBS
MLIRBarDialect
~~~

Aliases and forward declarations are supported. Imported library targets are
accepted because their installed generated headers already exist. Misspelled
targets, generator expressions, executables, and utility targets are diagnosed
at configure time. Nested and cyclic `HEADER_LIBS` relationships are safe: the
resolver adds only generated-header leaf targets, never provider libraries.

Libraries outside `add_mlir_library` do not accept `HEADER_LIBS`. A non-MLIR
header-only consumer should retain the relevant generator in `DEPENDS`.

### How ordering is modeled

`add_public_tablegen_target` marks generated-header utility targets. Each
`llvm_add_library` invocation records its exact `DEPENDS` list separately from
the cumulative `LLVM_COMMON_DEPENDS` directory state. After the full target
graph exists, a deferred traversal:

* resolves aliases and forward references;
* follows direct implementation links and transitive link interfaces;
* follows nested `HEADER_LIBS` relationships;
* conservatively extracts targets from link generator expressions; and
* creates a build-local interface target for each header provider.

These interface targets depend only on marked generators and link to other
header interfaces. CMake handles transitive ordering and cycles in that graph.
Consumers reference the header interfaces through their implementation links,
without publishing them in exported link interfaces. Provider objects and
archives are never added by this mechanism, so mutually linked static libraries
do not form strong build cycles. Disabled generator-expression arms may
generate extra headers. Imported libraries, paths, and flags require no
build-tree ordering.

This target-level modeling works with all supported CMake generators. Ninja's
weaker ordering still completes a dependency's custom commands before compiling
the consumer, while generators with stronger target ordering see only the leaf
generator targets and do not wait for provider archives.

## C API libraries and aggregation

`add_mlir_public_c_api_library` creates an object-enabled MLIR library with the
visibility definitions needed by the C API. Use `add_mlir_aggregate` to build a
shared or static library from such components. `EMBED_LIBS` contribute their
objects; `PUBLIC_LIBS` remain normal exported link dependencies.

Aggregation metadata is exported only for libraries created with
`ENABLE_AGGREGATION`. `MLIR_INSTALL_AGGREGATE_OBJECTS` controls whether the
object libraries needed by an out-of-tree aggregate are installed. Imported
components must have been installed with compatible aggregation metadata.

## Tools, installation, and exports

Use `add_mlir_tool` for MLIR command-line tools. It delegates to LLVM's tool
infrastructure and participates in the normal runtime and installation layout.

`add_mlir_library` installs and exports its target by default.
`add_mlir_library_install` exposes the same rules for a non-standard library
construction path. `DISABLE_INSTALL` suppresses those rules, while
`INSTALL_WITH_TOOLCHAIN` includes the library when only the toolchain component
is installed.

Installed packages contain generated headers already, so imported targets do
not contribute build-tree generator prerequisites. Exported logical library
interfaces, rather than private `*IncGen` target names, are the contract for
standalone consumers.

## Validating dependency changes

Generated-header correctness must be tested from empty generated-header state.
An incremental build can leave files behind, and a broad aggregate can generate
a missing header incidentally before its consumer compiles.

Configure a fresh Ninja build and first build representative leaf libraries at
normal parallelism:

~~~shell
cmake -S llvm -B <build> -G Ninja <configuration options>
cmake --build <build> --target \
  MLIRArmNeonDialect MLIRLinalgDialect --parallel 32
~~~

Then build the broader graph to populate compiler dependency files. Only after
compilation succeeds, run:

~~~shell
ninja -C <build> -t missingdeps
~~~

`missingdeps` compares generated-file producers with include relationships in
compiler depfiles. Running it before compilation cannot discover those
includes. The clean leaf build and populated-depfile audit cover different
failure modes.

When changing the CMake infrastructure, also configure the focused CMake tests
with Ninja and Unix Makefiles, test the oldest supported CMake release, and
configure `mlir/examples/standalone` against an installed or build-tree MLIR
package. Inspect generated object-order rules to confirm they contain generator
targets and not provider archives.

## Common mistakes

* Depending on another library's `MLIRFooOpsIncGen` instead of linking
  `MLIRFooDialect`.
* Using `HEADER_LIBS` where an ordinary link accurately models the C++ layer.
* Putting generator expressions or visibility keywords in `HEADER_LIBS`.
* Moving dialect-specific generators into `mlir-generic-headers` to hide a
  missing logical dependency.
* Repairing a leaf race with the global `mlir-headers` aggregate without first
  identifying the owning library.
* Running `missingdeps` before compiler depfiles have been populated.
* Validating only an incremental or broad aggregate build.
