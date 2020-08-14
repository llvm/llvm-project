# MLIR C API

**Current status: Under development, API unstable, built by default.**

## Design

Many languages can interoperate with C but have a harder time with C++ due to
name mangling and memory model differences. Although the C API for MLIR can be
used directly from C, it is primarily intended to be wrapped in higher-level
language- or library-specific constructs. Therefore the API tends towards
simplicity and feature minimalism.

**Note:** while the C API is expected to be more stable than C++ API, it
currently offers no stability guarantees.

### Scope

The API is provided for core IR components (attributes, blocks, operations,
regions, types, values), Passes and some fundamental type and attribute kinds.
The core IR API is intentionally low-level, e.g. exposes a plain list of
operation's operands and attributes without attempting to assign "semantic"
names to them. Users of specific dialects are expected to wrap the core API in a
dialect-specific way, for example, by implementing an ODS backend.

### Object Model

Core IR components are exposed as opaque _handles_ to an IR object existing in
C++. They are not intended to be inspected by the API users (and, in many cases,
cannot be meaningfully inspected). Instead the users are expected to pass
handles to the appropriate manipulation functions.

The handle _may or may not_ own the underlying object.

### Naming Convention and Ownership Model

All objects are prefixed with `Mlir`. They are typedefs and should be used
without `struct`.

All functions are prefixed with `mlir`.

Functions primarily operating on an instance of `MlirX` are prefixed with
`mlirX`. They take the instance being acted upon as their first argument (except
for creation functions). For example, `mlirOperationGetNumOperands` inspects an
`MlirOperation`, which it takes as its first operand.

The *ownership* model is encoded in the naming convention as follows.

-   By default, the ownership is not transerred.
-   Functions that tranfer the ownership of the result to the caller can be in
    one of two forms:
    *   functions that create a new object have the name `mlirXCreate<...>`, for
        example, `mlirOperationCreate`;
    *   functions that detach an object from a parent object have the name
        `mlirYTake<...>`, for example `mlirOperationStateTakeRegion`.
-   Functions that take ownership of some of their arguments have the form
    `mlirY<...>OwnedX<...>` where `X` can refer to the type or any other
    sufficiently unique description of the argument, the ownership of which will
    be taken by the callee, for example `mlirRegionAppendOwnedBlock`.
-   Functions that create an object by default do not transfer its ownership to
    the caller, i.e. one of other objects passed in as an argument retains the
    ownership, they have the form `mlirX<...>Get`. For example,
    `mlirTypeParseGet`.
-   Functions that destroy an object owned by the caller are of the form
    `mlirXDestroy`.

If the code owns an object, it is responsible for destroying the object when it
is no longer necessary. If an object that owns other objects is destroyed, any
handles to those objects become invalid. Note that types and attributes are
owned by the `MlirContext` in which they were created.

### Nullity

A handle may refer to a _null_ object. It is the responsibility of the caller to
check if an object is null by using `mlirXIsNull(MlirX)`. API functions do _not_
expect null objects as arguments unless explicitly stated otherwise. API
functions _may_ return null objects.

### Conversion To String and Printing

IR objects can be converted to a string representation, for example for
printing, using `mlirXPrint(MlirX, MlirPrintCallback, void *)` functions. These
functions accept take arguments a callback with signature `void (*)(const char
*, intptr_t, void *)` and a pointer to user-defined data. They call the callback
and supply it with chunks of the string representation, provided as a pointer to
the first character and a length, and forward the user-defined data unmodified.
It is up to the caller to allocate memory if the string representation must be
stored and perform the copy. There is no guarantee that the pointer supplied to
the callback points to a null-terminated string, the size argument should be
used to find the end of the string. The callback may be called multiple times
with consecutive chunks of the string representation (the printing itself is
bufferred).

*Rationale*: this approach allows the caller to have full control of the
allocation and avoid unnecessary allocation and copying inside the printer.

For convenience, `mlirXDump(MlirX)` functions are provided to print the given
object to the standard error stream.

### Common Patterns

The API adopts the following patterns for recurrent functionality in MLIR.

#### Indexed Components

An object has an _indexed component_ if it has fields accessible using a
zero-based contiguous integer index, typically arrays. For example, an
`MlirBlock` has its arguments as a indexed component. An object may have several
such components. For example, an `MlirOperation` has attributes, operands,
regions, results and successors.

For indexed components, the following pair of functions is provided.

-   `intptr_t mlirXGetNum<Y>s(MlirX)` returns the upper bound on the index.
-   `MlirY mlirXGet<Y>(MlirX, intptr_t pos)` returns 'pos'-th subobject.

The sizes are accepted and returned as signed pointer-sized integers, i.e.
`intptr_t`. This typedef is avalable in C99.

Note that the name of subobject in the function does not necessarily match the
type of the subobject. For example, `mlirOperationGetOperand` returns a
`MlirValue`.

#### Iterable Components

An object has an _iterable component_ if it has iterators accessing its fields
in some order other than integer indexing, typically linked lists. For example,
an `MlirBlock` has an iterable list of operations it contains. An object may
have several iterable components.

For iterable components, the following triple of functions is provided.

-   `MlirY mlirXGetFirst<Y>(MlirX)` returns the first subobject in the list.
-   `MlirY mlirYGetNextIn<X>(MlirY)` returns the next subobject in the list that
    contains the given object, or a null object if the given object is the last
    in this list.
-   `int mlirYIsNull(MlirY)` returns 1 if the given object is null.

Note that the name of subobject in the function may or may not match its type.

This approach enables one to iterate as follows.

```c++
MlirY iter;
for (iter = mlirXGetFirst<Y>(x); !mlirYIsNull(iter);
     iter = mlirYGetNextIn<X>(iter)) {
  /* User 'iter'. */
}
```
