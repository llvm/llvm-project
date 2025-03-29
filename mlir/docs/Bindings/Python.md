# MLIR Python Bindings

**Current status**: Under development and not enabled by default

[TOC]

## Building

### Pre-requisites

*   A relatively recent Python3 installation
*   Installation of python dependencies as specified in
    `mlir/python/requirements.txt`

### CMake variables

*   **`MLIR_ENABLE_BINDINGS_PYTHON`**`:BOOL`

    Enables building the Python bindings. Defaults to `OFF`.

*   **`Python3_EXECUTABLE`**:`STRING`

    Specifies the `python` executable used for the LLVM build, including for
    determining header/link flags for the Python bindings. On systems with
    multiple Python implementations, setting this explicitly to the preferred
    `python3` executable is strongly recommended.

### Recommended development practices

It is recommended to use a python virtual environment. Many ways exist for this,
but the following is the simplest:

```shell
# Make sure your 'python' is what you expect. Note that on multi-python
# systems, this may have a version suffix, and on many Linuxes and MacOS where
# python2 and python3 co-exist, you may also want to use `python3`.
which python
python -m venv ~/.venv/mlirdev
source ~/.venv/mlirdev/bin/activate

# Note that many LTS distros will bundle a version of pip itself that is too
# old to download all of the latest binaries for certain platforms.
# The pip version can be obtained with `python -m pip --version`, and for
# Linux specifically, this should be cross checked with minimum versions
# here: https://github.com/pypa/manylinux
# It is recommended to upgrade pip:
python -m pip install --upgrade pip


# Now the `python` command will resolve to your virtual environment and
# packages will be installed there.
python -m pip install -r mlir/python/requirements.txt

# Now run `cmake`, `ninja`, et al.
```

For interactive use, it is sufficient to add the
`tools/mlir/python_packages/mlir_core/` directory in your `build/` directory to
the `PYTHONPATH`. Typically:

```shell
export PYTHONPATH=$(cd build && pwd)/tools/mlir/python_packages/mlir_core
```

Note that if you have installed (i.e. via `ninja install`, et al), then python
packages for all enabled projects will be in your install tree under
`python_packages/` (i.e. `python_packages/mlir_core`). Official distributions
are built with a more specialized setup.

## Design

### Use cases

There are likely two primary use cases for the MLIR python bindings:

1.  Support users who expect that an installed version of LLVM/MLIR will yield
    the ability to `import mlir` and use the API in a pure way out of the box.

1.  Downstream integrations will likely want to include parts of the API in
    their private namespace or specially built libraries, probably mixing it
    with other python native bits.

### Composable modules

In order to support use case \#2, the Python bindings are organized into
composable modules that downstream integrators can include and re-export into
their own namespace if desired. This forces several design points:

*   Separate the construction/populating of a `py::module` from
    `PYBIND11_MODULE` global constructor.

*   Introduce headers for C++-only wrapper classes as other related C++ modules
    will need to interop with it.

*   Separate any initialization routines that depend on optional components into
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

*   `mlir.ir`
*   `mlir.passes` (`pass` is a reserved word :( )
*   `mlir.dialect`
*   `mlir.execution_engine` (aside from namespacing, it is important that
    "bulky"/optional parts like this are isolated)

In addition, initialization functions that imply optional dependencies should be
in underscored (notionally private) modules such as `_init` and linked
separately. This allows downstream integrators to completely customize what is
included "in the box" and covers things like dialect registration, pass
registration, etc.

### Loader

LLVM/MLIR is a non-trivial python-native project that is likely to co-exist with
other non-trivial native extensions. As such, the native extension (i.e. the
`.so`/`.pyd`/`.dylib`) is exported as a notionally private top-level symbol
(`_mlir`), while a small set of Python code is provided in
`mlir/_cext_loader.py` and siblings which loads and re-exports it. This split
provides a place to stage code that needs to prepare the environment *before*
the shared library is loaded into the Python runtime, and also provides a place
that one-time initialization code can be invoked apart from module constructors.

It is recommended to avoid using `__init__.py` files to the extent possible,
until reaching a leaf package that represents a discrete component. The rule to
keep in mind is that the presence of an `__init__.py` file prevents the ability
to split anything at that level or below in the namespace into different
directories, deployment packages, wheels, etc.

See the documentation for more information and advice:
https://packaging.python.org/guides/packaging-namespace-packages/

### Use the C-API

The Python APIs should seek to layer on top of the C-API to the degree possible.
Especially for the core, dialect-independent parts, such a binding enables
packaging decisions that would be difficult or impossible if spanning a C++ ABI
boundary. In addition, factoring in this way side-steps some very difficult
issues that arise when combining RTTI-based modules (which pybind derived things
are) with non-RTTI polymorphic C++ code (the default compilation mode of LLVM).

### Ownership in the Core IR

There are several top-level types in the core IR that are strongly owned by
their python-side reference:

*   `PyContext` (`mlir.ir.Context`)
*   `PyModule` (`mlir.ir.Module`)
*   `PyOperation` (`mlir.ir.Operation`) - but with caveats

All other objects are dependent. All objects maintain a back-reference
(keep-alive) to their closest containing top-level object. Further, dependent
objects fall into two categories: a) uniqued (which live for the life-time of
the context) and b) mutable. Mutable objects need additional machinery for
keeping track of when the C++ instance that backs their Python object is no
longer valid (typically due to some specific mutation of the IR, deletion, or
bulk operation).

### Optionality and argument ordering in the Core IR

The following types support being bound to the current thread as a context
manager:

*   `PyLocation` (`loc: mlir.ir.Location = None`)
*   `PyInsertionPoint` (`ip: mlir.ir.InsertionPoint = None`)
*   `PyMlirContext` (`context: mlir.ir.Context = None`)

In order to support composability of function arguments, when these types appear
as arguments, they should always be the last and appear in the above order and
with the given names (which is generally the order in which they are expected to
need to be expressed explicitly in special cases) as necessary. Each should
carry a default value of `py::none()` and use either a manual or automatic
conversion for resolving either with the explicit value or a value from the
thread context manager (i.e. `DefaultingPyMlirContext` or
`DefaultingPyLocation`).

The rationale for this is that in Python, trailing keyword arguments to the
*right* are the most composable, enabling a variety of strategies such as kwarg
passthrough, default values, etc. Keeping function signatures composable
increases the chances that interesting DSLs and higher level APIs can be
constructed without a lot of exotic boilerplate.

Used consistently, this enables a style of IR construction that rarely needs to
use explicit contexts, locations, or insertion points but is free to do so when
extra control is needed.

#### Operation hierarchy

As mentioned above, `PyOperation` is special because it can exist in either a
top-level or dependent state. The life-cycle is unidirectional: operations can
be created detached (top-level) and once added to another operation, they are
then dependent for the remainder of their lifetime. The situation is more
complicated when considering construction scenarios where an operation is added
to a transitive parent that is still detached, necessitating further accounting
at such transition points (i.e. all such added children are initially added to
the IR with a parent of their outer-most detached operation, but then once it is
added to an attached operation, they need to be re-parented to the containing
module).

Due to the validity and parenting accounting needs, `PyOperation` is the owner
for regions and blocks and needs to be a top-level type that we can count on not
aliasing. This let's us do things like selectively invalidating instances when
mutations occur without worrying that there is some alias to the same operation
in the hierarchy. Operations are also the only entity that are allowed to be in
a detached state, and they are interned at the context level so that there is
never more than one Python `mlir.ir.Operation` object for a unique
`MlirOperation`, regardless of how it is obtained.

The C/C++ API allows for Region/Block to also be detached, but it simplifies the
ownership model a lot to eliminate that possibility in this API, allowing the
Region/Block to be completely dependent on its owning operation for accounting.
The aliasing of Python `Region`/`Block` instances to underlying
`MlirRegion`/`MlirBlock` is considered benign and these objects are not interned
in the context (unlike operations).

If we ever want to re-introduce detached regions/blocks, we could do so with new
"DetachedRegion" class or similar and also avoid the complexity of accounting.
With the way it is now, we can avoid having a global live list for regions and
blocks. We may end up needing an op-local one at some point TBD, depending on
how hard it is to guarantee how mutations interact with their Python peer
objects. We can cross that bridge easily when we get there.

Module, when used purely from the Python API, can't alias anyway, so we can use
it as a top-level ref type without a live-list for interning. If the API ever
changes such that this cannot be guaranteed (i.e. by letting you marshal a
native-defined Module in), then there would need to be a live table for it too.

## User-level API

### Context Management

The bindings rely on Python
[context managers](https://docs.python.org/3/reference/datamodel.html#context-managers)
(`with` statements) to simplify creation and handling of IR objects by omitting
repeated arguments such as MLIR contexts, operation insertion points and
locations. A context manager sets up the default object to be used by all
binding calls within the following context and in the same thread. This default
can be overridden by specific calls through the dedicated keyword arguments.

#### MLIR Context

An MLIR context is a top-level entity that owns attributes and types and is
referenced from virtually all IR constructs. Contexts also provide thread safety
at the C++ level. In Python bindings, the MLIR context is also a Python context
manager, one can write:

```python
from mlir.ir import Context, Module

with Context() as ctx:
  # IR construction using `ctx` as context.

  # For example, parsing an MLIR module from string requires the context.
  Module.parse("builtin.module {}")
```

IR objects referencing a context usually provide access to it through the
`.context` property. Most IR-constructing functions expect the context to be
provided in some form. In case of attributes and types, the context may be
extracted from the contained attribute or type. In case of operations, the
context is systematically extracted from Locations (see below). When the context
cannot be extracted from any argument, the bindings API expects the (keyword)
argument `context`. If it is not provided or set to `None` (default), it will be
looked up from an implicit stack of contexts maintained by the bindings in the
current thread and updated by context managers. If there is no surrounding
context, an error will be raised.

Note that it is possible to manually specify the MLIR context both inside and
outside of the `with` statement:

```python
from mlir.ir import Context, Module

standalone_ctx = Context()
with Context() as managed_ctx:
  # Parse a module in managed_ctx.
  Module.parse("...")

  # Parse a module in standalone_ctx (override the context manager).
  Module.parse("...", context=standalone_ctx)

# Parse a module without using context managers.
Module.parse("...", context=standalone_ctx)
```

The context object remains live as long as there are IR objects referencing it.

#### Insertion Points and Locations

When constructing an MLIR operation, two pieces of information are required:

-   an *insertion point* that indicates where the operation is to be created in
    the IR region/block/operation structure (usually before or after another
    operation, or at the end of some block); it may be missing, at which point
    the operation is created in the *detached* state;
-   a *location* that contains user-understandable information about the source
    of the operation (for example, file/line/column information), which must
    always be provided as it carries a reference to the MLIR context.

Both can be provided using context managers or explicitly as keyword arguments
in the operation constructor. They can be also provided as keyword arguments
`ip` and `loc` both within and outside of the context manager.

```python
from mlir.ir import Context, InsertionPoint, Location, Module, Operation

with Context() as ctx:
  module = Module.create()

  # Prepare for inserting operations into the body of the module and indicate
  # that these operations originate in the "f.mlir" file at the given line and
  # column.
  with InsertionPoint(module.body), Location.file("f.mlir", line=42, col=1):
    # This operation will be inserted at the end of the module body and will
    # have the location set up by the context manager.
    Operation(<...>)

    # This operation will be inserted at the end of the module (and after the
    # previously constructed operation) and will have the location provided as
    # the keyword argument.
    Operation(<...>, loc=Location.file("g.mlir", line=1, col=10))

    # This operation will be inserted at the *beginning* of the block rather
    # than at its end.
    Operation(<...>, ip=InsertionPoint.at_block_begin(module.body))
```

Note that `Location` needs an MLIR context to be constructed. It can take the
context set up in the current thread by some surrounding context manager, or
accept it as an explicit argument:

```python
from mlir.ir import Context, Location

# Create a context and a location in this context in the same `with` statement.
with Context() as ctx, Location.file("f.mlir", line=42, col=1, context=ctx):
  pass
```

Locations are owned by the context and live as long as they are (transitively)
referenced from somewhere in Python code.

Unlike locations, the insertion point may be left unspecified (or, equivalently,
set to `None` or `False`) during operation construction. In this case, the
operation is created in the *detached* state, that is, it is not added into the
region of another operation and is owned by the caller. This is usually the case
for top-level operations that contain the IR, such as modules. Regions, blocks
and values contained in an operation point back to it and maintain it live.

### Inspecting IR Objects

Inspecting the IR is one of the primary tasks the Python bindings are designed
for. One can traverse the IR operation/region/block structure and inspect their
aspects such as operation attributes and value types.

#### Operations, Regions and Blocks

Operations are represented as either:

-   the generic `Operation` class, useful in particular for generic processing
    of unregistered operations; or
-   a specific subclass of `OpView` that provides more semantically-loaded
    accessors to operation properties.

Given an `OpView` subclass, one can obtain an `Operation` using its `.operation`
property. Given an `Operation`, one can obtain the corresponding `OpView` using
its `.opview` property *as long as* the corresponding class has been set up.
This typically means that the Python module of its dialect has been loaded. By
default, the `OpView` version is produced when navigating the IR tree.

One can check if an operation has a specific type by means of Python's
`isinstance` function:

```python
operation = <...>
opview = <...>
if isinstance(operation.opview, mydialect.MyOp):
  pass
if isinstance(opview, mydialect.MyOp):
  pass
```

The components of an operation can be inspected using its properties.

-   `attributes` is a collection of operation attributes . It can be subscripted
    as both dictionary and sequence, e.g., both `operation.attributes["value"]`
    and `operation.attributes[0]` will work. There is no guarantee on the order
    in which the attributes are traversed when iterating over the `attributes`
    property as sequence.
-   `operands` is a sequence collection of operation operands.
-   `results` is a sequence collection of operation results.
-   `regions` is a sequence collection of regions attached to the operation.

The objects produced by `operands` and `results` have a `.types` property that
contains a sequence collection of types of the corresponding values.

```python
from mlir.ir import Operation

operation1 = <...>
operation2 = <...>
if operation1.results.types == operation2.operand.types:
  pass
```

`OpView` subclasses for specific operations may provide leaner accessors to
properties of an operation. For example, named attributes, operand and results
are usually accessible as properties of the `OpView` subclass with the same
name, such as `operation.const_value` instead of
`operation.attributes["const_value"]`. If this name is a reserved Python
keyword, it is suffixed with an underscore.

The operation itself is iterable, which provides access to the attached regions
in order:

```python
from mlir.ir import Operation

operation = <...>
for region in operation:
  do_something_with_region(region)
```

A region is conceptually a sequence of blocks. Objects of the `Region` class are
thus iterable, which provides access to the blocks. One can also use the
`.blocks` property.

```python
# Regions are directly iterable and give access to blocks.
for block1, block2 in zip(operation.regions[0], operation.regions[0].blocks)
  assert block1 == block2
```

A block contains a sequence of operations, and has several additional
properties. Objects of the `Block` class are iterable and provide access to the
operations contained in the block. So does the `.operations` property. Blocks
also have a list of arguments available as a sequence collection using the
`.arguments` property.

Block and region belong to the parent operation in Python bindings and keep it
alive. This operation can be accessed using the `.owner` property.

#### Attributes and Types

Attributes and types are (mostly) immutable context-owned objects. They are
represented as either:

-   an opaque `Attribute` or `Type` object supporting printing and comparison;
    or
-   a concrete subclass thereof with access to properties of the attribute or
    type.

Given an `Attribute` or `Type` object, one can obtain a concrete subclass using
the constructor of the subclass. This may raise a `ValueError` if the attribute
or type is not of the expected subclass:

```python
from mlir.ir import Attribute, Type
from mlir.<dialect> import ConcreteAttr, ConcreteType

attribute = <...>
type = <...>
try:
  concrete_attr = ConcreteAttr(attribute)
  concrete_type = ConcreteType(type)
except ValueError as e:
  # Handle incorrect subclass.
```

In addition, concrete attribute and type classes provide a static `isinstance`
method to check whether an object of the opaque `Attribute` or `Type` type can
be downcasted:

```python
from mlir.ir import Attribute, Type
from mlir.<dialect> import ConcreteAttr, ConcreteType

attribute = <...>
type = <...>

# No need to handle errors here.
if ConcreteAttr.isinstance(attribute):
  concrete_attr = ConcreteAttr(attribute)
if ConcreteType.isinstance(type):
  concrete_type = ConcreteType(type)
```

By default, and unlike operations, attributes and types are returned from IR
traversals using the opaque `Attribute` or `Type` that needs to be downcasted.

Concrete attribute and type classes usually expose their properties as Python
readonly properties. For example, the elemental type of a tensor type can be
accessed using the `.element_type` property.

#### Values

MLIR has two kinds of values based on their defining object: block arguments and
operation results. Values are handled similarly to attributes and types. They
are represented as either:

-   a generic `Value` object; or
-   a concrete `BlockArgument` or `OpResult` object.

The former provides all the generic functionality such as comparison, type
access and printing. The latter provide access to the defining block or
operation and the position of the value within it. By default, the generic
`Value` objects are returned from IR traversals. Downcasting is implemented
through concrete subclass constructors, similarly to attribtues and types:

```python
from mlir.ir import BlockArgument, OpResult, Value

value = ...

# Set `concrete` to the specific value subclass.
try:
  concrete = BlockArgument(value)
except ValueError:
  # This must not raise another ValueError as values are either block arguments
  # or op results.
  concrete = OpResult(value)
```

#### Interfaces

MLIR interfaces are a mechanism to interact with the IR without needing to know
specific types of operations but only some of their aspects. Operation
interfaces are available as Python classes with the same name as their C++
counterparts. Objects of these classes can be constructed from either:

-   an object of the `Operation` class or of any `OpView` subclass; in this
    case, all interface methods are available;
-   a subclass of `OpView` and a context; in this case, only the *static*
    interface methods are available as there is no associated operation.

In both cases, construction of the interface raises a `ValueError` if the
operation class does not implement the interface in the given context (or, for
operations, in the context that the operation is defined in). Similarly to
attributes and types, the MLIR context may be set up by a surrounding context
manager.

```python
from mlir.ir import Context, InferTypeOpInterface

with Context():
  op = <...>

  # Attempt to cast the operation into an interface.
  try:
    iface = InferTypeOpInterface(op)
  except ValueError:
    print("Operation does not implement InferTypeOpInterface.")
    raise

  # All methods are available on interface objects constructed from an Operation
  # or an OpView.
  iface.someInstanceMethod()

  # An interface object can also be constructed given an OpView subclass. It
  # also needs a context in which the interface will be looked up. The context
  # can be provided explicitly or set up by the surrounding context manager.
  try:
    iface = InferTypeOpInterface(some_dialect.SomeOp)
  except ValueError:
    print("SomeOp does not implement InferTypeOpInterface.")
    raise

  # Calling an instance method on an interface object constructed from a class
  # will raise TypeError.
  try:
    iface.someInstanceMethod()
  except TypeError:
    pass

  # One can still call static interface methods though.
  iface.inferOpReturnTypes(<...>)
```

If an interface object was constructed from an `Operation` or an `OpView`, they
are available as `.operation` and `.opview` properties of the interface object,
respectively.

Only a subset of operation interfaces are currently provided in Python bindings.
Attribute and type interfaces are not yet available in Python bindings.

### Creating IR Objects

Python bindings also support IR creation and manipulation.

#### Operations, Regions and Blocks

Operations can be created given a `Location` and an optional `InsertionPoint`.
It is often easier to user context managers to specify locations and insertion
points for several operations created in a row as described above.

Concrete operations can be created by using constructors of the corresponding
`OpView` subclasses. The generic, default form of the constructor accepts:

-   an optional sequence of types for operation results (`results`);
-   an optional sequence of values for operation operands, or another operation
    producing those values (`operands`);
-   an optional dictionary of operation attributes (`attributes`);
-   an optional sequence of successor blocks (`successors`);
-   the number of regions to attach to the operation (`regions`, default `0`);
-   the `loc` keyword argument containing the `Location` of this operation; if
    `None`, the location created by the closest context manager is used or an
    exception will be raised if there is no context manager;
-   the `ip` keyword argument indicating where the operation will be inserted in
    the IR; if `None`, the insertion point created by the closest context
    manager is used; if there is no surrounding context manager, the operation
    is created in the detached state.

Most operations will customize the constructor to accept a reduced list of
arguments that are relevant for the operation. For example, zero-result
operations may omit the `results` argument, so can the operations where the
result types can be derived from operand types unambiguously. As a concrete
example, built-in function operations can be constructed by providing a function
name as string and its argument and result types as a tuple of sequences:

```python
from mlir.ir import Context, Module
from mlir.dialects import builtin

with Context():
  module = Module.create()
  with InsertionPoint(module.body), Location.unknown():
    func = func.FuncOp("main", ([], []))
```

Also see below for constructors generated from ODS.

Operations can also be constructed using the generic class and based on the
canonical string name of the operation using `Operation.create`. It accepts the
operation name as string, which must exactly match the canonical name of the
operation in C++ or ODS, followed by the same argument list as the default
constructor for `OpView`. *This form is discouraged* from use and is intended
for generic operation processing.

```python
from mlir.ir import Context, Module
from mlir.dialects import builtin

with Context():
  module = Module.create()
  with InsertionPoint(module.body), Location.unknown():
    # Operations can be created in a generic way.
    func = Operation.create(
        "func.func", results=[], operands=[],
        attributes={"function_type":TypeAttr.get(FunctionType.get([], []))},
        successors=None, regions=1)
    # The result will be downcasted to the concrete `OpView` subclass if
    # available.
    assert isinstance(func, func.FuncOp)
```

Regions are created for an operation when constructing it on the C++ side. They
are not constructible in Python and are not expected to exist outside of
operations (unlike in C++ that supports detached regions).

Blocks can be created within a given region and inserted before or after another
block of the same region using `create_before()`, `create_after()` methods of
the `Block` class, or the `create_at_start()` static method of the same class.
They are not expected to exist outside of regions (unlike in C++ that supports
detached blocks).

```python
from mlir.ir import Block, Context, Operation

with Context():
  op = Operation.create("generic.op", regions=1)

  # Create the first block in the region.
  entry_block = Block.create_at_start(op.regions[0])

  # Create further blocks.
  other_block = entry_block.create_after()
```

Blocks can be used to create `InsertionPoint`s, which can point to the beginning
or the end of the block, or just before its terminator. It is common for
`OpView` subclasses to provide a `.body` property that can be used to construct
an `InsertionPoint`. For example, builtin `Module` and `FuncOp` provide a
`.body` and `.add_entry_blocK()`, respectively.

#### Attributes and Types

Attributes and types can be created given a `Context` or another attribute or
type object that already references the context. To indicate that they are owned
by the context, they are obtained by calling the static `get` method on the
concrete attribute or type class. These method take as arguments the data
necessary to construct the attribute or type and a the keyword `context`
argument when the context cannot be derived from other arguments.

```python
from mlir.ir import Context, F32Type, FloatAttr

# Attribute and types require access to an MLIR context, either directly or
# through another context-owned object.
ctx = Context()
f32 = F32Type.get(context=ctx)
pi = FloatAttr.get(f32, 3.14)

# They may use the context defined by the surrounding context manager.
with Context():
  f32 = F32Type.get()
  pi = FloatAttr.get(f32, 3.14)
```

Some attributes provide additional construction methods for clarity.

```python
from mlir.ir import Context, IntegerAttr, IntegerType

with Context():
  i8 = IntegerType.get_signless(8)
  IntegerAttr.get(i8, 42)
```

Builtin attribute can often be constructed from Python types with similar
structure. For example, `ArrayAttr` can be constructed from a sequence
collection of attributes, and a `DictAttr` can be constructed from a dictionary:

```python
from mlir.ir import ArrayAttr, Context, DictAttr, UnitAttr

with Context():
  array = ArrayAttr.get([UnitAttr.get(), UnitAttr.get()])
  dictionary = DictAttr.get({"array": array, "unit": UnitAttr.get()})
```

Custom builders for Attributes to be used during Operation creation can be
registered by way of the `register_attribute_builder`. In particular the
following is how a custom builder is registered for `I32Attr`:

```python
@register_attribute_builder("I32Attr")
def _i32Attr(x: int, context: Context):
  return IntegerAttr.get(
        IntegerType.get_signless(32, context=context), x)
```

This allows to invoke op creation of an op with a `I32Attr` with

```python
foo.Op(30)
```

The registration is based on the ODS name but registry is via pure python
method. Only single custom builder is allowed to be registered per ODS attribute
type (e.g., I32Attr can have only one, which can correspond to multiple of the
underlying IntegerAttr type).

instead of

```python
foo.Op(IntegerAttr.get(IndexType.get_signless(32, context=context), 30))
```

## Style

In general, for the core parts of MLIR, the Python bindings should be largely
isomorphic with the underlying C++ structures. However, concessions are made
either for practicality or to give the resulting library an appropriately
"Pythonic" flavor.

### Properties vs get\*() methods

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

### **repr** methods

Things that have nice printed representations are really great :) If there is a
reasonable printed form, it can be a significant productivity boost to wire that
to the `__repr__` method (and verify it with a [doctest](#sample-doctest)).

### CamelCase vs snake\_case

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

We use `lit` and `FileCheck` based tests:

*   For generative tests (those that produce IR), define a Python module that
    constructs/prints the IR and pipe it through `FileCheck`.
*   Parsing should be kept self-contained within the module under test by use of
    raw constants and an appropriate `parse_asm` call.
*   Any file I/O code should be staged through a tempfile vs relying on file
    artifacts/paths outside of the test module.
*   For convenience, we also test non-generative API interactions with the same
    mechanisms, printing and `CHECK`ing as needed.

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

## Integration with ODS

The MLIR Python bindings integrate with the tablegen-based ODS system for
providing user-friendly wrappers around MLIR dialects and operations. There are
multiple parts to this integration, outlined below. Most details have been
elided: refer to the build rules and python sources under `mlir.dialects` for
the canonical way to use this facility.

Users are responsible for providing a `{DIALECT_NAMESPACE}.py` (or an equivalent
directory with `__init__.py` file) as the entrypoint.

### Generating `_{DIALECT_NAMESPACE}_ops_gen.py` wrapper modules

Each dialect with a mapping to python requires that an appropriate
`_{DIALECT_NAMESPACE}_ops_gen.py` wrapper module is created. This is done by
invoking `mlir-tblgen` on a python-bindings specific tablegen wrapper that
includes the boilerplate and actual dialect specific `td` file. An example, for
the `Func` (which is assigned the namespace `func` as a special case):

```tablegen
#ifndef PYTHON_BINDINGS_FUNC_OPS
#define PYTHON_BINDINGS_FUNC_OPS

include "mlir/Dialect/Func/IR/FuncOps.td"

#endif // PYTHON_BINDINGS_FUNC_OPS
```

In the main repository, building the wrapper is done via the CMake function
`declare_mlir_dialect_python_bindings`, which invokes:

```
mlir-tblgen -gen-python-op-bindings -bind-dialect={DIALECT_NAMESPACE} \
    {PYTHON_BINDING_TD_FILE}
```

The generates op classes must be included in the `{DIALECT_NAMESPACE}.py` file
in a similar way that generated headers are included for C++ generated code:

```python
from ._my_dialect_ops_gen import *
```

### Extending the search path for wrapper modules

When the python bindings need to locate a wrapper module, they consult the
`dialect_search_path` and use it to find an appropriately named module. For the
main repository, this search path is hard-coded to include the `mlir.dialects`
module, which is where wrappers are emitted by the above build rule. Out of tree
dialects can add their modules to the search path by calling:

```python
from mlir.dialects._ods_common import _cext
_cext.globals.append_dialect_search_prefix("myproject.mlir.dialects")
```

### Wrapper module code organization

The wrapper module tablegen emitter outputs:

*   A `_Dialect` class (extending `mlir.ir.Dialect`) with a `DIALECT_NAMESPACE`
    attribute.
*   An `{OpName}` class for each operation (extending `mlir.ir.OpView`).
*   Decorators for each of the above to register with the system.

Note: In order to avoid naming conflicts, all internal names used by the wrapper
module are prefixed by `_ods_`.

Each concrete `OpView` subclass further defines several public-intended
attributes:

*   `OPERATION_NAME` attribute with the `str` fully qualified operation name
    (i.e. `math.absf`).
*   An `__init__` method for the *default builder* if one is defined or inferred
    for the operation.
*   `@property` getter for each operand or result (using an auto-generated name
    for unnamed of each).
*   `@property` getter, setter and deleter for each declared attribute.

It further emits additional private-intended attributes meant for subclassing
and customization (default cases omit these attributes in favor of the defaults
on `OpView`):

*   `_ODS_REGIONS`: A specification on the number and types of regions.
    Currently a tuple of (min_region_count, has_no_variadic_regions). Note that
    the API does some light validation on this but the primary purpose is to
    capture sufficient information to perform other default building and region
    accessor generation.
*   `_ODS_OPERAND_SEGMENTS` and `_ODS_RESULT_SEGMENTS`: Black-box value which
    indicates the structure of either the operand or results with respect to
    variadics. Used by `OpView._ods_build_default` to decode operand and result
    lists that contain lists.

#### Default Builder

Presently, only a single, default builder is mapped to the `__init__` method.
The intent is that this `__init__` method represents the *most specific* of the
builders typically generated for C++; however currently it is just the generic
form below.

*   One argument for each declared result:
    *   For single-valued results: Each will accept an `mlir.ir.Type`.
    *   For variadic results: Each will accept a `List[mlir.ir.Type]`.
*   One argument for each declared operand or attribute:
    *   For single-valued operands: Each will accept an `mlir.ir.Value`.
    *   For variadic operands: Each will accept a `List[mlir.ir.Value]`.
    *   For attributes, it will accept an `mlir.ir.Attribute`.
*   Trailing usage-specific, optional keyword arguments:
    *   `loc`: An explicit `mlir.ir.Location` to use. Defaults to the location
        bound to the thread (i.e. `with Location.unknown():`) or an error if
        none is bound nor specified.
    *   `ip`: An explicit `mlir.ir.InsertionPoint` to use. Default to the
        insertion point bound to the thread (i.e. `with InsertionPoint(...):`).

In addition, each `OpView` inherits a `build_generic` method which allows
construction via a (nested in the case of variadic) sequence of `results` and
`operands`. This can be used to get some default construction semantics for
operations that are otherwise unsupported in Python, at the expense of having a
very generic signature.

#### Extending Generated Op Classes

As mentioned above, the build system generates Python sources like
`_{DIALECT_NAMESPACE}_ops_gen.py` for each dialect with Python bindings. It is
often desirable to use these generated classes as a starting point for
further customization, so an extension mechanism is provided to make this easy.
This mechanism uses conventional inheritance combined with `OpView` registration.
For example, the default builder for `arith.constant`

```python
class ConstantOp(_ods_ir.OpView):
  OPERATION_NAME = "arith.constant"

  _ODS_REGIONS = (0, True)

  def __init__(self, value, *, loc=None, ip=None):
    ...
```

expects `value` to be a `TypedAttr` (e.g., `IntegerAttr` or `FloatAttr`).
Thus, a natural extension is a builder that accepts a MLIR type and a Python value and instantiates the appropriate `TypedAttr`:

```python
from typing import Union

from mlir.ir import Type, IntegerAttr, FloatAttr
from mlir.dialects._arith_ops_gen import _Dialect, ConstantOp
from mlir.dialects._ods_common import _cext

@_cext.register_operation(_Dialect, replace=True)
class ConstantOpExt(ConstantOp):
    def __init__(
        self, result: Type, value: Union[int, float], *, loc=None, ip=None
    ):
        if isinstance(value, int):
            super().__init__(IntegerAttr.get(result, value), loc=loc, ip=ip)
        elif isinstance(value, float):
            super().__init__(FloatAttr.get(result, value), loc=loc, ip=ip)
        else:
            raise NotImplementedError(f"Building `arith.constant` not supported for {result=} {value=}")
```

which enables building an instance of `arith.constant` like so:

```python
from mlir.ir import F32Type

a = ConstantOpExt(F32Type.get(), 42.42)
b = ConstantOpExt(IntegerType.get_signless(32), 42)
```

Note, three key aspects of the extension mechanism in this example:

1. `ConstantOpExt` directly inherits from the generated `ConstantOp`;
2. in this, simplest, case all that's required is a call to the super class' initializer, i.e., `super().__init__(...)`;
3. in order to register `ConstantOpExt` as the preferred `OpView` that is returned by `mlir.ir.Operation.opview` (see [Operations, Regions and Blocks](#operations-regions-and-blocks))
   we decorate the class with `@_cext.register_operation(_Dialect, replace=True)`, **where the `replace=True` must be used**.

In some more complex cases it might be necessary to explicitly build the `OpView` through `OpView.build_generic` (see [Default Builder](#default-builder)), just as is performed by the generated builders.
I.e., we must call `OpView.build_generic` **and pass the result to `OpView.__init__`**, where the small issue becomes that the latter is already overridden by the generated builder.
Thus, we must call a method of a super class' super class (the "grandparent"); for example:

```python
from mlir.dialects._scf_ops_gen import _Dialect, ForOp
from mlir.dialects._ods_common import _cext

@_cext.register_operation(_Dialect, replace=True)
class ForOpExt(ForOp):
    def __init__(self, lower_bound, upper_bound, step, iter_args, *, loc=None, ip=None):
        ...
        super(ForOp, self).__init__(self.build_generic(...))
```

where `OpView.__init__` is called via `super(ForOp, self).__init__`.
Note, there are alternatives ways to implement this (e.g., explicitly writing `OpView.__init__`); see any discussion on Python inheritance.

## Providing Python bindings for a dialect

Python bindings are designed to support MLIR’s open dialect ecosystem. A dialect
can be exposed to Python as a submodule of `mlir.dialects` and interoperate with
the rest of the bindings. For dialects containing only operations, it is
sufficient to provide Python APIs for those operations. Note that the majority
of boilerplate APIs can be generated from ODS. For dialects containing
attributes and types, it is necessary to thread those through the C API since
there is no generic mechanism to create attributes and types. Passes need to be
registered with the context in order to be usable in a text-specified pass
manager, which may be done at Python module load time. Other functionality can
be provided, similar to attributes and types, by exposing the relevant C API and
building Python API on top.


### Operations

Dialect operations are provided in Python by wrapping the generic
`mlir.ir.Operation` class with operation-specific builder functions and
properties. Therefore, there is no need to implement a separate C API for them.
For operations defined in ODS, `mlir-tblgen -gen-python-op-bindings
-bind-dialect=<dialect-namespace>` generates the Python API from the declarative
description.
It is sufficient to create a new `.td` file that includes the original ODS
definition and use it as source for the `mlir-tblgen` call.
Such `.td` files reside in
[`python/mlir/dialects/`](https://github.com/llvm/llvm-project/tree/main/mlir/python/mlir/dialects).
The results of `mlir-tblgen` are expected to produce a file named
`_<dialect-namespace>_ops_gen.py` by convention. The generated operation classes
can be extended as described above. MLIR provides [CMake
functions](https://github.com/llvm/llvm-project/blob/main/mlir/cmake/modules/AddMLIRPython.cmake)
to automate the production of such files. Finally, a
`python/mlir/dialects/<dialect-namespace>.py` or a
`python/mlir/dialects/<dialect-namespace>/__init__.py` file must be created and
filled with `import`s from the generated files to enable `import
mlir.dialects.<dialect-namespace>` in Python.


### Attributes and Types

Dialect attributes and types are provided in Python as subclasses of the
`mlir.ir.Attribute` and `mlir.ir.Type` classes, respectively. Python APIs for
attributes and types must connect to the relevant C APIs for building and
inspection, which must be provided first. Bindings for `Attribute` and `Type`
subclasses can be defined using
[`include/mlir/Bindings/Python/PybindAdaptors.h`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Bindings/Python/PybindAdaptors.h)
or
[`include/mlir/Bindings/Python/NanobindAdaptors.h`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Bindings/Python/NanobindAdaptors.h)
utilities that mimic pybind11/nanobind API for defining functions and
properties. These bindings are to be included in a separate module. The
utilities also provide automatic casting between C API handles `MlirAttribute`
and `MlirType` and their Python counterparts so that the C API handles can be
used directly in binding implementations. The methods and properties provided by
the bindings should follow the principles discussed above.

The attribute and type bindings for a dialect can be located in
`lib/Bindings/Python/Dialect<Name>.cpp` and should be compiled into a separate
“Python extension” library placed in `python/mlir/_mlir_libs` that will be
loaded by Python at runtime. MLIR provides [CMake
functions](https://github.com/llvm/llvm-project/blob/main/mlir/cmake/modules/AddMLIRPython.cmake)
to automate the production of such libraries. This library should be `import`ed
from the main dialect file, i.e. `python/mlir/dialects/<dialect-namespace>.py`
or `python/mlir/dialects/<dialect-namespace>/__init__.py`, to ensure the types
are available when the dialect is loaded from Python.


### Passes

Dialect-specific passes can be made available to the pass manager in Python by
registering them with the context and relying on the API for pass pipeline
parsing from string descriptions. This can be achieved by creating a new
pybind11 module, defined in `lib/Bindings/Python/<Dialect>Passes.cpp`, that
calls the registration C API, which must be provided first. For passes defined
declaratively using Tablegen, `mlir-tblgen -gen-pass-capi-header` and
`-mlir-tblgen -gen-pass-capi-impl` automate the generation of C API. The
pybind11 module must be compiled into a separate “Python extension” library,
which can be `import`ed  from the main dialect file, i.e.
`python/mlir/dialects/<dialect-namespace>.py` or
`python/mlir/dialects/<dialect-namespace>/__init__.py`, or from a separate
`passes` submodule to be put in
`python/mlir/dialects/<dialect-namespace>/passes.py` if it is undesirable to
make the passes available along with the dialect.


### Other functionality

Dialect functionality other than IR objects or passes, such as helper functions,
can be exposed to Python similarly to attributes and types. C API is expected to
exist for this functionality, which can then be wrapped using pybind11 and
[`include/mlir/Bindings/Python/PybindAdaptors.h`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Bindings/Python/PybindAdaptors.h),
or nanobind and
[`include/mlir/Bindings/Python/NanobindAdaptors.h`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Bindings/Python/NanobindAdaptors.h)
utilities to connect to the rest of Python API. The bindings can be located in a
separate module or in the same module as attributes and types, and
loaded along with the dialect.

## Free-threading (No-GIL) support

Free-threading or no-GIL support refers to CPython interpreter (>=3.13) with Global Interpreter Lock made optional. For details on the topic, please check [PEP-703](https://peps.python.org/pep-0703/) and this [Python free-threading guide](https://py-free-threading.github.io/).

MLIR Python bindings are free-threading compatible with exceptions (discussed below) in the following sense: it is safe to work in multiple threads with **independent** contexts. Below we show an example code of safe usage:

```python
# python3.13t example.py
import concurrent.futures

import mlir.dialects.arith as arith
from mlir.ir import Context, Location, Module, IntegerType, InsertionPoint


def func(py_value):
    with Context() as ctx:
        module = Module.create(loc=Location.file("foo.txt", 0, 0))

        dtype = IntegerType.get_signless(64)
        with InsertionPoint(module.body), Location.name("a"):
            arith.constant(dtype, py_value)

    return module


num_workers = 8
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = []
    for i in range(num_workers):
        futures.append(executor.submit(func, i))
    assert len(list(f.result() for f in futures)) == num_workers
```

The exceptions to the free-threading compatibility:
- IR printing is unsafe, e.g. when using `PassManager` with `PassManager.enable_ir_printing()` which calls thread-unsafe `llvm::raw_ostream`.
- Usage of `Location.emit_error` is unsafe (due to thread-unsafe `llvm::raw_ostream`).
- Usage of `Module.dump` is unsafe (due to thread-unsafe `llvm::raw_ostream`).
- Usage of `mlir.dialects.transform.interpreter` is unsafe.
- Usage of `mlir.dialects.gpu` and `gpu-module-to-binary` is unsafe.