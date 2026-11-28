# Strongly typed `DIExpression` API

```{contents}
:local:
```

## Introduction

The current [`DIExpression`](#diexpression) representation has a number of nice
properties:

* The core C++ API is just `SmallVector<uint64_t>` and a `get` function.
  * Code can inspect and edit the representation with no hard dependencies.
  * The cost to `#include` the support code for this is light, making it fast
    to compile.
  * The representation is pretty compact, and plays nice with the cache and
    branch predictor.

It also has some downsides:

* The core C++ API is just `SmallVector<uint64_t>` and a `get` function.
  * There is no boundary at which to separate concerns, and no easy path to
    improving the representation without affecting users.
  * There are places in the codebase which muck about with the internals
    directly.
* In response to the relaxed API there are myriad attempts to systematize
  certain aspects, but these do not cohere well, since they have grown up
  organically and often are fit to a specific use-case.
  * These are spread across:
    * `static` member functions,
    * non-`static` member function,
    * free functions across the codebase.
  * Among the `static` and free function cases there are also two flavors:
    * functions which implicitly `get`, and so accept/return `unique`d `const
      DIExpression *`,
    * functions which accept a `SmallVectorImpl<uint64_t>&`.

The goal of the design in this document is to maintain as many of the positive
aspects as possible, while codifying the best parts of the code that has grown
up around it, and packaging it all in a strongly typed shell.

## Design

The core principle is to leave the current representation alone, with the
`unique`d `DIExpression *` in `LLVMContext` not changing.

On top of that, three new types are added that nearly all code will be moved
over to using: `DIOp::Op`, `DIExprRef`, and `DIExprBuf`.

By way of analogy:

* `DIExpression *` is to `const char *`, as
* `DIOp::Op` is to `char`, as
* `DIExprRef` is to `std::string_view`/`llvm::StringRef`, as
* `DIExprBuf` is to `std::string`/`llvm::SmallString`.

:::{note}
The analogy isn't airtight. The `DIExpression *` doesn't actually point to a
contiguous view of `DIOp::Op`s. In the new hierarchy it acts more as an opaque
handle to a `unique`d expression.

There is also no equivalent to `std::char_traits`.
:::

### `DIOp::Op`

The `namespace DIOp` contains `Op`, a compact (2-`qword` on `x86_64`) value
type, modeled around `std::variant` but not actually deriving from it.
`DIOp::Op` is passed by value.

The alternative types of `Op` have `PascalCase`-ified names corresponding to
the `DW_OP_*` names currently in use, and all live in `namespace DIOp`, next to
`Op`. For example:

:::{list-table}
:header-rows: 1
* - Old
  - New
* - `{DW_OP_lit3}`
  - `DIOp::Lit3()`
* - `{DW_OP_breg4, 21}`
  - `DIOp::BReg4(21)`
* - `{DW_OP_LLVM_fragment, 24, 8}`
  - `DIOp::LLVMFragment(24, 8)`
* - `{DW_OP_addr, 42, DW_OP_plus_uconst, 8, DW_OP_deref}`
  - `{DIOp::Addr(42), DIOp::PlusUConst(8), DIOp::Deref()}`
:::

The new type is not `POD`, but is standard layout and trivially destructible.
This enables a couple optimizations:

* The tag can be part of the "Common Initial Sequence", reducing the size of
  the `Op`. This is part of the rationale for a new type rather than a
  `std::variant`.
* Operations over containers of `Op` can e.g. use `memcpy` and avoid having to
  call destructors in some cases.

### `DIExprRef`

Compact (2-`qword` on `x86_64`) non-owning view type. `DIExprRef` is passed by
value.

Generally constructed from an existing `DIExpression *`, and so has the
same lifetime as the associated `LLVMContext`.

Always verified at construction (either by a cached property or by iterating
over the underlying `uint64_t[]`) to cover a syntactically valid expression.
For example, the `DIExpressions::getRef` might return
`std::optional<DIExprRef>`.

The expected general shape of this type is:

```cpp
class DIExprRef {
  friend class DIExpression;
  friend class DIExprBuf;

  iterator_range<DIOp::FromUIntIterator> Ops;

  // Private constructor assumes Ops is syntactically valid, which
  // is ensured by friends before construction.
  explicit DIExprRef(iterator_range<DIOp::FromUIntIterator> Ops) : Ops(Ops) {};
public:
  DIExprRef(const DIExprRef&) = default;
  DIExprRef(DIExprRef&&) = default;
  DIExprRef &operator=(const DIExprRef&) = default;
  DIExprRef &operator=(DIExprRef&&) = default;
  ~DIExprRef() = default;

  bool isValid() const;
  bool isSingleLocationExpression() const;
  // more query methods ...
};
```

### `DIExprBuf`

Larger, owning buffer type. `DIExprBuf` is passed by value (when moved) or by
non-`const` reference (when shared).

Double buffered internally, to avoid extra allocations for chained mutations.
Can also be reused by e.g. a pass which needs to update many expressions, to
further reduce allocations.

:::{note}
The double buffering essentially just codifies an existing pattern whereby most
mutation method for `DIExpression` looks like:

```cpp
// check preconditions...

// start making a new expression:
SmallVector<uint64_t> NewOps;

// push to NewOps...

// unique the new expression:
return DIExpression::get(NewOPs);
```

As `DIExprBuf` is intended to be reused for e.g. all expression updates within
a pass, this change means the difference between `O(N)` and `O(1)` allocations,
while not cluttering the interface with mutable buffer parameters.
:::

Since mutations on this type can be chained without `unique`ing the
intermediate results, there is some potential improvement to RSS, although
the overall contribution of `DIExpression` to the memory used during
compilation isn't so high that this is particularly significant.

The expected general shape of this type is:

```cpp
class DIExprBuf {
  LLVMContext *Ctx = nullptr;
  SmallVector<uint64_t, 0> Elements;
  // Nearly all operations require double-buffering, so we bake it in.
  // This allows us to re-use a small number of allocations for the
  // processing of many expressions, even where each expression may require
  // multiple operations.
  //
  // Each method has as an implicit post-condition that the backing buffer
  // NewElements is empty, and so on entry it can be used without being cleared.
  SmallVector<uint64_t, 0> NewElements;

public:
  DIExprBuf() = default;
  explicit DIExprBuf(LLVMContext *Ctx);
  explicit DIExprBuf(const DIExpression *From);
  explicit DIExprBuf(DIExprRef From, LLVMContext *Ctx = nullptr);

  // Any operation on *this invalidates the returned DIExprRef
  DIExprRef asRef() const;

  DIExprBuf &clear();
  DIExprBuf &convertToUndefExpression();
  DIExprBuf &prepend(uint8_t Flags, int64_t Offset = 0);
  DIExprBuf &foldConstantMath();
  DIExprBuf &replaceArg(uint64_t OldArgIndex, uint64_t NewArgIndex);
  // more mutation methods ...

  DIExpression *toExpr() const;
};
```

## Future Work

### IR representation

Initially, the IR representation can be kept identical.

Eventually, we can consider supporting a new syntax in the IR to track the
logical encapsulation of each operation and more directly track the
implementation:

:::{list-table}
:header-rows: 1
* - Old
  - New
* - `DIExpression(DW_OP_addr, 42, DW_OP_plus_uconst, 8, DW_OP_deref)`
  - `DIExpression(DIOp::Addr(42), DIOp::PlusUConst(8), DIOp::Deref())`
:::

This can be bi-direction and forward compatible from existing IR. As the
bitcode does not change in any event (it is still always `vector<uint64_t>`)
this does not represent a compatibility break.

### `LLVMContext` representation

If you squint, `DIOp::Op` is just a vector of two `uint64_t`. Some initial
profiling shows that replacing the `vector<uint64_t>` representation with
`vector<DIOp::Op>` may amount to a wash in terms of and instructions retired
and RSS, with some significant improvements in some cases and some significant
regressions in others:

:::{table} instructions:u
|                       | Old                  | New          |
| :-------------------- | -------------------: | -----------: |
| stage1-O3             | `   60554M (-0.02%)` | `   60563M`  |
| stage1-ReleaseThinLTO | `   76249M (+0.00%)` | `   76246M`  |
| stage1-ReleaseLTO-g   | `   89006M (-0.04%)` | `   89038M`  |
| stage1-O0-g           | `   18520M (+0.01%)` | `   18519M`  |
| stage1-aarch64-O3     | `   67536M (+0.00%)` | `   67536M`  |
| stage1-aarch64-O0-g   | `   22600M (-0.01%)` | `   22603M`  |
| stage2-O3             | `   52437M (-0.04%)` | `   52460M`  |
| stage2-O0-g           | `   16219M (-0.01%)` | `   16221M`  |
| stage2-clang          | `34618385M (+0.07%)` | `34592728M`  |
:::

:::{table} max-rss
|                       | Old                | New       |
| :-------------------- | -----------------: | --------: |
| stage1-O3             | `2780MiB (-0.82%)` | `2803MiB` |
| stage1-ReleaseThinLTO | `2712MiB (-0.57%)` | `2727MiB` |
| stage1-ReleaseLTO-g   | `3280MiB (-0.77%)` | `3305MiB` |
| stage1-O0-g           | `2605MiB (-0.90%)` | `2629MiB` |
| stage1-aarch64-O3     | `2974MiB (-0.68%)` | `2995MiB` |
| stage1-aarch64-O0-g   | `2596MiB (-0.78%)` | `2617MiB` |
| stage2-O3             | `2590MiB (+0.17%)` | `2586MiB` |
| stage2-O0-g           | `2493MiB (+0.14%)` | `2490MiB` |
| stage2-clang          | `2321MiB (+0.91%)` | `2300MiB` |
:::

However, the situation does get marginally worse once we have eliminated
`DW_OP_LLVM_fragment`/`DIOp::LLVMFragment`, since the existing representation
is larger in this case and it is relatively common.

This switch is not inevitable, and the hope is to make dealing with the
internal representation uncommon for most developers anyway, but it could be
considered in the future.

