# Data Layout Modeling

Data layout information allows the compiler to answer questions related to how a
value of a particular type is stored in memory. For example, the size of a value
or its address alignment requirements. It enables, among others, the generation
of various linear memory addressing schemes for containers of abstract types and
deeper reasoning about vectors.

The data layout subsystem is designed to scale to MLIR's open type and operation
system. At the top level, it consists of:

*   attribute interfaces that can be implemented by concrete data layout
    specifications;
*   type interfaces that should be implemented by types subject to data layout;
*   operation interfaces that must be implemented by operations that can serve
    as data layout scopes (e.g., modules);
*   and dialect interfaces for data layout properties unrelated to specific
    types.

Built-in types are handled specially to decrease the overall query cost.
Similarly, built-in `ModuleOp` supports data layouts without going through the
interface.

## Usage

### Scoping

Following MLIR's nested structure, data layout properties are _scoped_ to
regions belonging to either operations that implement the
`DataLayoutOpInterface` or `ModuleOp` operations. Such scoping operations
partially control the data layout properties and may have attributes that affect
them, typically organized in a data layout specification.

Types may have a different data layout in different scopes, including scopes
that are nested in other scopes such as modules contained in other modules. At
the same time, within the given scope excluding any nested scope, a given type
has fixed data layout properties. Types are also expected to have a default,
"natural" data layout in case they are used outside of any operation that
provides data layout scope for them. This ensures that data layout queries
always have a valid result.

### Compatibility and Transformations

The information necessary to compute layout properties can be combined from
nested scopes. For example, an outer scope can define layout properties for a
subset of types while inner scopes define them for a disjoint subset, or scopes
can progressively relax alignment requirements on a type. This mechanism is
supported by the notion of data layout _compatibility_: the layout defined in a
nested scope is expected to be compatible with that of the outer scope. MLIR
does not prescribe what compatibility means for particular ops and types but
provides hooks for them to provide target- and type-specific checks. For
example, one may want to only allow relaxation of alignment constraints (i.e.,
smaller alignment) in nested modules or, alternatively, one may require nested
modules to fully redefine all constraints of the outer scope.

Data layout compatibility is also relevant during IR transformation. Any
transformation that affects the data layout scoping operation is expected to
maintain data layout compatibility. It is under responsibility of the
transformation to ensure it is indeed the case.

### Queries

Data layout property queries can be performed on the special object --
`DataLayout` -- which can be created for the given scoping operation. These
objects allow one to interface with the data layout infrastructure and query
properties of given types in the scope of the object. The signature of
`DataLayout` class is as follows.

```c++
class DataLayout {
public:
  explicit DataLayout(DataLayoutOpInterface scope);

  unsigned getTypeSize(Type type) const;
  unsigned getTypeSizeInBits(Type type) const;
  unsigned getTypeABIAlignment(Type type) const;
  unsigned getTypePreferredAlignment(Type type) const;
};
```

The user can construct the `DataLayout` object for the scope of interest. Since
the data layout properties are fixed in the scope, they will be computed only
once upon first request and cached for further use. Therefore,
`DataLayout(op.getParentOfType<DataLayoutOpInterface>()).getTypeSize(type)` is
considered an anti-pattern since it discards the cache after use. Because of
caching, a `DataLayout` object returns valid results as long as the data layout
properties of enclosing scopes remain the same, that is, as long as none of the
ancestor operations are modified in a way that affects data layout. After such a
modification, the user is expected to create a fresh `DataLayout` object. To aid
with this, `DataLayout` asserts that the scope remains identical if MLIR is
compiled with assertions enabled.

## Custom Implementations

Extensibility of the data layout modeling is provided through a set of MLIR
[Interfaces](Interfaces.md).

### Data Layout Specifications

Data layout specification is an [attribute](LangRef.md/#attributes) that is
conceptually a collection of key-value pairs called data layout specification
_entries_. Data layout specification attributes implement the
`DataLayoutSpecInterface`, described below. Each entry is itself an attribute
that implements the `DataLayoutEntryInterface`. Entries have a key, either a
`Type` or an `Identifier`, and a value. Keys are used to associate entries with
specific types or dialects: when handling a data layout properties request, a
type or a dialect can only see the specification entries relevant to them and
must go through the supplied `DataLayout` object for any recursive query. This
supports and enforces better composability because types cannot (and should not)
understand layout details of other types. Entry values are arbitrary attributes,
specific to the type.

For example, a data layout specification may be an actual list of pairs with
simple custom syntax resembling the following:

```
#my_dialect.layout_spec<
  #my_dialect.layout_entry<!my_dialect.type, size=42>,
  #my_dialect.layout_entry<"my_dialect.endianness", "little">,
  #my_dialect.layout_entry<!my_dialect.vector, prefer_large_alignment>>
```

The exact details of the specification and entry attributes, as well as their
syntax, are up to implementations.

We use the notion of _type class_ throughout the data layout subsystem. It
corresponds to the C++ class of the given type, e.g., `IntegerType` for built-in
integers. MLIR does not have a mechanism to represent type classes in the IR.
Instead, data layout entries contain specific _instances_ of a type class, for
example, `IntegerType{signedness=signless, bitwidth=8}` (or `i8` in the IR) or
`IntegerType{signedness=unsigned, bitwidth=32}` (or `ui32` in the IR). When
handling a data layout property query, a type class will be supplied with _all_
entries with keys belonging to this type class. For example, `IntegerType` will
see the entries for `i8`, `si16` and `ui32`, but will _not_ see those for `f32`
or `memref<?xi32>` (neither will `MemRefType` see the entry for `i32`). This
allows for type-specific "interpolation" behavior where a type class can compute
data layout properties of _any_ specific type instance given properties of other
instances. Using integers as an example again, their alignment could be computed
by taking that of the closest from above integer type with power-of-two
bitwidth.

[include "Interfaces/DataLayoutAttrInterface.md"]

### Data Layout Scoping Operations

Operations that define a scope for data layout queries, and that can be used to
create a `DataLayout` object, are expected to implement the
`DataLayoutOpInterface`. Such ops must provide at least a way of obtaining the
data layout specification. The specification need not be necessarily attached to
the operation as an attribute and may be constructed on-the-fly; it is only
fetched once per `DataLayout` object and cached. Such ops may also provide
custom handlers for data layout queries that provide results without forwarding
the queries down to specific types or post-processing the results returned by
types in target- or scope-specific ways. These custom handlers make it possible
for scoping operations to (re)define data layout properties for types without
having to modify the types themselves, e.g., when types are defined in another
dialect.

[include "Interfaces/DataLayoutOpInterface.md"]

### Types with Data Layout

Type classes that intend to handle data layout queries themselves are expected
to implement the `DataLayoutTypeInterface`. This interface provides overridable
hooks for each data layout query. Each of these hooks is supplied with the type
instance, a `DataLayout` object suitable for recursive queries, and a list of
data layout queries relevant for the type class. It is expected to provide a
valid result even if the list of entries is empty. These hooks do not have
access to the operation in the scope of which the query is handled and should
use the supplied entries instead.

[include "Interfaces/DataLayoutTypeInterface.md"]

### Dialects with Data Layout Identifiers

For data layout entries that are not related to a particular type class, the key
of the entry is an Identifier that belongs to some dialect. In this case, the
dialect is expected to implement the `DataLayoutDialectInterface`. This dialect
provides hooks for verifying the validity of the entry value attributes and for
and the compatibility of nested entries.

### Bits and Bytes

Two versions of hooks are provided for sizes: in bits and in bytes. The version
in bytes has a default implementation that derives the size in bytes by rounding
up the result of division of the size in bits by 8. Types exclusively targeting
architectures with different assumptions can override this. Operations can
redefine this for all types, providing scoped versions for cases of byte sizes
other than eight without having to modify types, including built-in types.

### Query Dispatch

The overall flow of a data layout property query is as follows.

1.  The user constructs a `DataLayout` at the given scope. The constructor
    fetches the data layout specification and combines it with those of
    enclosing scopes (layouts are expected to be compatible).
2.  The user calls `DataLayout::query(Type ty)`.
3.  If `DataLayout` has a cached response, this response is returned
    immediately.
4.  Otherwise, the query is handed down by `DataLayout` to the closest layout
    scoping operation. If it implements `DataLayoutOpInterface`, then the query
    is forwarded to`DataLayoutOpInterface::query(ty, *this, relevantEntries)`
    where the relevant entries are computed as described above. If it does not
    implement `DataLayoutOpInterface`, it must be a `ModuleOp`, and the query is
    forwarded to `DataLayoutTypeInterface::query(dataLayout, relevantEntries)`
    after casting `ty` to the type interface.
5.  Unless the `query` hook is reimplemented by the op interface, the query is
    handled further down to `DataLayoutTypeInterface::query(dataLayout,
    relevantEntries)` after casting `ty` to the type interface. If the type does
    not implement the interface, an unrecoverable fatal error is produced.
6.  The type is expected to always provide the response, which is returned up
    the call stack and cached by the `DataLayout.`

## Default Implementation

The default implementation of the data layout interfaces directly handles
queries for a subset of built-in types.

### Built-in Modules

Built-in `ModuleOp` allows at most one attribute that implements
`DataLayoutSpecInterface`. It does not implement the entire interface for
efficiency and layering reasons. Instead, `DataLayout` can be constructed for
`ModuleOp` and handles modules transparently alongside other operations that
implement the interface.

### Built-in Types

The following describes the default properties of built-in types.

The size of built-in integers and floats in bytes is computed as
`ceildiv(bitwidth, 8)`. The ABI alignment of integer types with bitwidth below
64 and of the float types is the closest from above power-of-two number of
bytes. The ABI alignment of integer types with bitwidth 64 and above is 4 bytes
(32 bits).

The size of built-in vectors is computed by first rounding their number of
elements in the _innermost_ dimension to the closest power-of-two from above,
then getting the total number of elements, and finally multiplying it with the
element size. For example, `vector<3xi32>` and `vector<4xi32>` have the same
size. So do `vector<2x3xf32>` and `vector<2x4xf32>`, but `vector<3x4xf32>` and
`vector<4x4xf32>` have different sizes. The ABI and preferred alignment of
vector types is computed by taking the innermost dimension of the vector,
rounding it up to the closest power-of-two, taking a product of that with
element size in bytes, and rounding the result up again to the closest
power-of-two.

Note: these values are selected for consistency with the
[default data layout in LLVM](https://llvm.org/docs/LangRef.html#data-layout),
which MLIR assumed until the introduction of proper data layout modeling, and
with the
[modeling of n-D vectors](https://mlir.llvm.org/docs/Dialects/Vector/#deeperdive).
They **may change** in the future.

#### `index` type

Index type is an integer type used for target-specific size information in,
e.g., `memref` operations. Its data layout is parameterized by a single integer
data layout entry that specifies its bitwidth. For example,

```
module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<index, 32>
>} {}
```

specifies that `index` has 32 bits. All other layout properties of `index` match
those of the integer type with the same bitwidth defined above.

In absence of the corresponding entry, `index` is assumed to be a 64-bit
integer.

#### `complex` type

By default complex type is treated like a 2 element structure of its given
element type. This is to say that each of its elements are aligned to their
preferred alignment, the entire complex type is also aligned to this preference,
and the complex type size includes the possible padding between elements to enforce
alignment.

### Byte Size

The default data layout assumes 8-bit bytes.

### DLTI Dialect

The [DLTI](Dialects/DLTI.md) dialect provides the attributes implementing
`DataLayoutSpecInterface` and `DataLayoutEntryInterface`, as well as a dialect
attribute that can be used to attach the specification to a given operation. The
verifier of this attribute triggers those of the specification and checks the
compatiblity of nested specifications.
