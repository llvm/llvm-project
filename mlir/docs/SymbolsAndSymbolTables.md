# Symbols and Symbol Tables

[TOC]

With [Regions](LangRef.md/#regions), the multi-level aspect of MLIR is
structural in the IR. A lot of infrastructure within the compiler is built
around this nesting structure; including the processing of operations within the
[pass manager](PassManagement.md/#pass-manager). One advantage of the MLIR
design is that it is able to process operations in parallel, utilizing multiple
threads. This is possible due to a property of the IR known as
[`IsolatedFromAbove`](Traits.md/#isolatedfromabove).

Without this property, any operation could affect or mutate the use-list of
operations defined above. Making this thread-safe requires expensive locking in
some of the core IR data structures, which becomes quite inefficient. To enable
multi-threaded compilation without this locking, MLIR uses local pools for
constant values as well as `Symbol` accesses for global values and variables.
This document details the design of `Symbol`s, what they are and how they fit
into the system.

The `Symbol` infrastructure essentially provides a non-SSA mechanism in which to
refer to an operation symbolically with a name. This allows for referring to
operations defined above regions that were defined as `IsolatedFromAbove` in a
safe way. It also allows for symbolically referencing operations define below
other regions as well.

## Symbol

A `Symbol` is a named operation that resides immediately within a region that
defines a [`SymbolTable`](#symbol-table). The name of a symbol *must* be unique
within the parent `SymbolTable`. This name is semantically similarly to an SSA
result value, and may be referred to by other operations to provide a symbolic
link, or use, to the symbol. An example of a `Symbol` operation is
[`func.func`](Dialects/Builtin.md/#func-mlirfuncop). `func.func` defines a
symbol name, which is [referred to](#referencing-a-symbol) by operations like
[`func.call`](Dialects/Func.md/#funccall-callop).

### Defining or declaring a Symbol

A `Symbol` operation should use the `SymbolOpInterface` interface to provide the
necessary verification and accessors; it also supports operations, such as
`builtin.module`, that conditionally define a symbol. `Symbol`s must have the
following properties:

*   A `StringAttr` attribute named
    'SymbolTable::getSymbolAttrName()'(`sym_name`).
    -   This attribute defines the symbolic 'name' of the operation.
*   An optional `StringAttr` attribute named
    'SymbolTable::getVisibilityAttrName()'(`sym_visibility`)
    -   This attribute defines the [visibility](#symbol-visibility) of the
        symbol, or more specifically in-which scopes it may be accessed.
*   No SSA results
    -   Intermixing the different ways to `use` an operation quickly becomes
        unwieldy and difficult to analyze.
*   Whether this operation is a declaration or definition (`isDeclaration`)
    -   Declarations do not define a new symbol but reference a symbol defined
        outside the visible IR.

## Symbol Table

Described above are `Symbol`s, which reside within a region of an operation
defining a `SymbolTable`. A `SymbolTable` operation provides the container for
the [`Symbol`](#symbol) operations. It verifies that all `Symbol` operations
have a unique name, and provides facilities for looking up symbols by name.
Operations defining a `SymbolTable` must use the `OpTrait::SymbolTable` trait.

### Referencing a Symbol

`Symbol`s are referenced symbolically by name via the
[`SymbolRefAttr`](Dialects/Builtin.md/#symbolrefattr) attribute. A symbol
reference attribute contains a named reference to an operation that is nested
within a symbol table. It may optionally contain a set of nested references that
further resolve to a symbol nested within a different symbol table. When
resolving a nested reference, each non-leaf reference must refer to a symbol
operation that is also a [symbol table](#symbol-table).

Below is an example of how an operation can reference a symbol operation:

```mlir
// This `func.func` operation defines a symbol named `symbol`.
func.func @symbol()

// Our `foo.user` operation contains a SymbolRefAttr with the name of the
// `symbol` func.
"foo.user"() {uses = [@symbol]} : () -> ()

// Symbol references resolve to the nearest parent operation that defines a
// symbol table, so we can have references with arbitrary nesting levels.
func.func @other_symbol() {
  affine.for %i0 = 0 to 10 {
    // Our `foo.user` operation resolves to the same `symbol` func as defined
    // above.
    "foo.user"() {uses = [@symbol]} : () -> ()
  }
  return
}

// Here we define a nested symbol table. References within this operation will
// not resolve to any symbols defined above.
module {
  // Error. We resolve references with respect to the closest parent operation
  // that defines a symbol table, so this reference can't be resolved.
  "foo.user"() {uses = [@symbol]} : () -> ()
}

// Here we define another nested symbol table, except this time it also defines
// a symbol.
module @module_symbol {
  // This `func.func` operation defines a symbol named `nested_symbol`.
  func.func @nested_symbol()
}

// Our `foo.user` operation may refer to the nested symbol, by resolving through
// the parent.
"foo.user"() {uses = [@module_symbol::@nested_symbol]} : () -> ()
```

Using an attribute, as opposed to an SSA value, has several benefits:

*   References may appear in more places than the operand list; including
    [nested attribute dictionaries](Dialects/Builtin.md/dictionaryattr),
    [array attributes](Dialects/Builtin.md/#arrayattr), etc.

*   Handling of SSA dominance remains unchanged.

    -   If we were to use SSA values, we would need to create some mechanism in
        which to opt-out of certain properties of it such as dominance.
        Attributes allow for referencing the operations irregardless of the
        order in which they were defined.
    -   Attributes simplify referencing operations within nested symbol tables,
        which are traditionally not visible outside of the parent region.

The impact of this choice to use attributes as opposed to SSA values is that we
now have two mechanisms with reference operations. This means that some dialects
must either support both `SymbolRefs` and SSA value references, or provide
operations that materialize SSA values from a symbol reference. Each has
different trade offs depending on the situation. A function call may directly
use a `SymbolRef` as the callee, whereas a reference to a global variable might
use a materialization operation so that the variable can be used in other
operations like `arith.addi`.
[`llvm.mlir.addressof`](Dialects/LLVM.md/#llvmmliraddressof-mlirllvmaddressofop)
is one example of such an operation.

See the `LangRef` definition of the
[`SymbolRefAttr`](Dialects/Builtin.md/#symbolrefattr) for more information about
the structure of this attribute.

Operations that reference a `Symbol` and want to perform verification and
general mutation of the symbol should implement the `SymbolUserOpInterface` to
ensure that symbol accesses are legal and efficient.

### Manipulating a Symbol

As described above, `SymbolRefs` act as an auxiliary way of defining uses of
operations to the traditional SSA use-list. As such, it is imperative to provide
similar functionality to manipulate and inspect the list of uses and the users.
The following are a few of the utilities provided by the `SymbolTable`:

*   `SymbolTable::getSymbolUses`

    -   Access an iterator range over all of the uses on and nested within a
        particular operation.

*   `SymbolTable::symbolKnownUseEmpty`

    -   Check if a particular symbol is known to be unused within a specific
        section of the IR.

*   `SymbolTable::replaceAllSymbolUses`

    -   Replace all of the uses of one symbol with a new one within a specific
        section of the IR.

*   `SymbolTable::lookupNearestSymbolFrom`

    -   Lookup the definition of a symbol in the nearest symbol table from some
        anchor operation.

## Symbol Visibility

Along with a name, a `Symbol` also has a `visibility` attached to it. The
`visibility` of a symbol defines its structural reachability within the IR. A
symbol has one of the following visibilities:

*   Public (Default)

    -   The symbol may be referenced from outside of the visible IR. We cannot
        assume that all of the uses of this symbol are observable. If the
        operation declares a symbol (as opposed to defining it), public
        visibility is not allowed because symbol declarations are not intended
        to be used from outside the visible IR.

*   Private

    -   The symbol may only be referenced from within the current symbol table.

*   Nested

    -   The symbol may be referenced by operations outside of the current symbol
        table, but not outside of the visible IR, as long as each symbol table
        parent also defines a non-private symbol.

For Functions, the visibility is printed after the operation name without a
quote. A few examples of what this looks like in the IR are shown below:

```mlir
module @public_module {
  // This function can be accessed by 'live.user', but cannot be referenced
  // externally; all uses are known to reside within parent regions.
  func.func nested @nested_function()

  // This function cannot be accessed outside of 'public_module'.
  func.func private @private_function()
}

// This function can only be accessed from within the top-level module.
func.func private @private_function()

// This function may be referenced externally.
func.func @public_function()

"live.user"() {uses = [
  @public_module::@nested_function,
  @private_function,
  @public_function
]} : () -> ()
```
