# MLIR Python Bindings

Current status: Under development and not enabled by default


## Building

### Pre-requisites

* [`pybind11`](https://github.com/pybind/pybind11) must be installed and able to
  be located by CMake.
* A relatively recent Python3 installation

### CMake variables

* **`MLIR_BINDINGS_PYTHON_ENABLED`**`:BOOL`

  Enables building the Python bindings. Defaults to `OFF`.

* **`MLIR_PYTHON_BINDINGS_VERSION_LOCKED`**`:BOOL`

  Links the native extension against the Python runtime library, which is
  optional on some platforms. While setting this to `OFF` can yield some greater
  deployment flexibility, linking in this way allows the linker to report
  compile time errors for unresolved symbols on all platforms, which makes for a
  smoother development workflow. Defaults to `ON`.

* **`PYTHON_EXECUTABLE`**:`STRING`

  Specifies the `python` executable used for the LLVM build, including for
  determining header/link flags for the Python bindings. On systems with
  multiple Python implementations, setting this explicitly to the preferred
  `python3` executable is strongly recommended.


## Design

### Use cases

There are likely two primary use cases for the MLIR python bindings:

1. Support users who expect that an installed version of LLVM/MLIR will yield
   the ability to `import mlir` and use the API in a pure way out of the box.

2. Downstream integrations will likely want to include parts of the API in their
   private namespace or specially built libraries, probably mixing it with other
   python native bits.


### Composable modules

In order to support use case #2, the Python bindings are organized into
composable modules that downstream integrators can include and re-export into
their own namespace if desired. This forces several design points:

* Separate the construction/populating of a `py::module` from `PYBIND11_MODULE`
  global constructor.

* Introduce headers for C++-only wrapper classes as other related C++ modules
  will need to interop with it.

* Separate any initialization routines that depend on optional components into
  its own module/dependency (currently, things like `registerAllDialects` fall
  into this category).

There are a lot of co-related issues of shared library linkage, distribution
concerns, etc that affect such things. Organizing the code into composable
modules (versus a monolithic `cpp` file) allows the flexibility to address many
of these as needed over time. Also, compilation time for all of the template
meta-programming in pybind scales with the number of things you define in a
translation unit. Breaking into multiple translation units can significantly aid
compile times for APIs with a large surface area.

### Submodules

Generally, the C++ codebase namespaces most things into the `mlir` namespace.
However, in order to modularize and make the Python bindings easier to
understand, sub-packages are defined that map roughly to the directory structure
of functional units in MLIR.

Examples:

* `mlir.ir`
* `mlir.passes` (`pass` is a reserved word :( )
* `mlir.dialect`
* `mlir.execution_engine` (aside from namespacing, it is important that
  "bulky"/optional parts like this are isolated)

In addition, initialization functions that imply optional dependencies should
be in underscored (notionally private) modules such as `_init` and linked
separately. This allows downstream integrators to completely customize what is
included "in the box" and covers things like dialect registration,
pass registration, etc.

### Loader

LLVM/MLIR is a non-trivial python-native project that is likely to co-exist with
other non-trivial native extensions. As such, the native extension (i.e. the
`.so`/`.pyd`/`.dylib`) is exported as a notionally private top-level symbol
(`_mlir`), while a small set of Python code is provided in `mlir/__init__.py`
and siblings which loads and re-exports it. This split provides a place to stage
code that needs to prepare the environment *before* the shared library is loaded
into the Python runtime, and also provides a place that one-time initialization
code can be invoked apart from module constructors.

To start with the `mlir/__init__.py` loader shim can be very simple and scale to
future need:

```python
from _mlir import *
```

### Limited use of globals

For normal operations, parent-child constructor relationships are realized with
constructor methods on a parent class as opposed to requiring
invocation/creation from a global symbol.

For example, consider two code fragments:

```python

op = build_my_op()

region = mlir.Region(op)

```

vs

```python

op = build_my_op()

region = op.new_region()

```

For tightly coupled data structures like `Operation`, the latter is generally
preferred because:

* It is syntactically less possible to create something that is going to access
  illegal memory (less error handling in the bindings, less testing, etc).

* It reduces the global-API surface area for creating related entities. This
  makes it more likely that if constructing IR based on an Operation instance of
  unknown providence, receiving code can just call methods on it to do what they
  want versus needing to reach back into the global namespace and find the right
  `Region` class.

* It leaks fewer things that are in place for C++ convenience (i.e. default
  constructors to invalid instances).

### Use the C-API

The Python APIs should seek to layer on top of the C-API to the degree possible.
Especially for the core, dialect-independent parts, such a binding enables
packaging decisions that would be difficult or impossible if spanning a C++ ABI
boundary. In addition, factoring in this way side-steps some very difficult
issues that arise when combining RTTI-based modules (which pybind derived things
are) with non-RTTI polymorphic C++ code (the default compilation mode of LLVM).


## Style

In general, for the core parts of MLIR, the Python bindings should be largely
isomorphic with the underlying C++ structures. However, concessions are made
either for practicality or to give the resulting library an appropriately
"Pythonic" flavor.

### Properties vs get*() methods

Generally favor converting trivial methods like `getContext()`, `getName()`,
`isEntryBlock()`, etc to read-only Python properties (i.e. `context`). It is
primarily a matter of calling `def_property_readonly` vs `def` in binding code,
and makes things feel much nicer to the Python side.

For example, prefer:

```c++
m.def_property_readonly("context", ...)
```

Over:

```c++
m.def("getContext", ...)
```

### __repr__ methods

Things that have nice printed representations are really great :)  If there is a
reasonable printed form, it can be a significant productivity boost to wire that
to the `__repr__` method (and verify it with a [doctest](#sample-doctest)).

### CamelCase vs snake_case

Name functions/methods/properties in `snake_case` and classes in `CamelCase`. As
a mechanical concession to Python style, this can go a long way to making the
API feel like it fits in with its peers in the Python landscape.

If in doubt, choose names that will flow properly with other
[PEP 8 style names](https://pep8.org/#descriptive-naming-styles).

### Prefer pseudo-containers

Many core IR constructs provide methods directly on the instance to query count
and begin/end iterators. Prefer hoisting these to dedicated pseudo containers.

For example, a direct mapping of blocks within regions could be done this way:

```python
region = ...

for block in region:

  pass
```

However, this way is preferred:

```python
region = ...

for block in region.blocks:

  pass

print(len(region.blocks))
print(region.blocks[0])
print(region.blocks[-1])
```

Instead of leaking STL-derived identifiers (`front`, `back`, etc), translate
them to appropriate `__dunder__` methods and iterator wrappers in the bindings.

Note that this can be taken too far, so use good judgment. For example, block
arguments may appear container-like but have defined methods for lookup and
mutation that would be hard to model properly without making semantics
complicated. If running into these, just mirror the C/C++ API.

### Provide one stop helpers for common things

One stop helpers that aggregate over multiple low level entities can be
incredibly helpful and are encouraged within reason. For example, making
`Context` have a `parse_asm` or equivalent that avoids needing to explicitly
construct a SourceMgr can be quite nice. One stop helpers do not have to be
mutually exclusive with a more complete mapping of the backing constructs.

## Testing

Tests should be added in the `test/Bindings/Python` directory and should
typically be `.py` files that have a lit run line.

While lit can run any python module, prefer to lay tests out according to these
rules:

* For tests of the API surface area, prefer
  [`doctest`](https://docs.python.org/3/library/doctest.html).
* For generative tests (those that produce IR), define a Python module that
  constructs/prints the IR and pipe it through `FileCheck`.
* Parsing should be kept self-contained within the module under test by use of
  raw constants and an appropriate `parse_asm` call.
* Any file I/O code should be staged through a tempfile vs relying on file
  artifacts/paths outside of the test module.

### Sample Doctest

```python
# RUN: %PYTHON %s

"""
  >>> m = load_test_module()
Test basics:
  >>> m.operation.name
  "module"
  >>> m.operation.is_registered
  True
  >>> ... etc ...

Verify that repr prints:
  >>> m.operation
  <operation 'module'>
"""

import mlir

TEST_MLIR_ASM = r"""
func @test_operation_correct_regions() {
  // ...
}
"""

# TODO: Move to a test utility class once any of this actually exists.
def load_test_module():
  ctx = mlir.ir.Context()
  ctx.allow_unregistered_dialects = True
  module = ctx.parse_asm(TEST_MLIR_ASM)
  return module


if __name__ == "__main__":
  import doctest
  doctest.testmod()
```

### Sample FileCheck test

```python
# RUN: %PYTHON %s | mlir-opt -split-input-file | FileCheck

# TODO: Move to a test utility class once any of this actually exists.
def print_module(f):
  m = f()
  print("// -----")
  print("// TEST_FUNCTION:", f.__name__)
  print(m.to_asm())
  return f

# CHECK-LABEL: TEST_FUNCTION: create_my_op
@print_module
def create_my_op():
  m = mlir.ir.Module()
  builder = m.new_op_builder()
  # CHECK: mydialect.my_operation ...
  builder.my_op()
  return m
```
