# Builtin Dialect

The builtin dialect contains a core set of Attributes, Operations, and Types
that have wide applicability across a very large number of domains and
abstractions. Many of the components of this dialect are also instrumental in
the implementation of the core IR. As such, this dialect is implicitly loaded in
every `MLIRContext`, and available directly to all users of MLIR.

Given the far-reaching nature of this dialect and the fact that MLIR is
extensible by design, any potential additions are heavily scrutinized.

[TOC]

## Attributes

[include "Dialects/BuiltinAttributes.md"]

## Location Attributes

A subset of the builtin attribute values correspond to
[source locations](../Diagnostics.md/#source-locations), that may be attached to
Operations.

[include "Dialects/BuiltinLocationAttributes.md"]

## DistinctAttribute

A DistinctAttribute associates an attribute with a unique identifier.
As a result, multiple DistinctAttribute instances may point to the same
attribute. Every call to the `create` function allocates a new
DistinctAttribute instance. The address of the attribute instance serves as a
temporary unique identifier. Similar to the names of SSA values, the final
unique identifiers are generated during pretty printing. This delayed
numbering ensures the printed identifiers are deterministic even if
multiple DistinctAttribute instances are created in-parallel.

Syntax:

```
distinct-id ::= integer-literal
distinct-attribute ::= `distinct` `[` distinct-id `]<` attribute `>`
```

Examples:

```mlir
#distinct = distinct[0]<42.0 : f32>
#distinct1 = distinct[1]<42.0 : f32>
#distinct2 = distinct[2]<array<i32: 10, 42>>
```

This mechanism is meant to generate attributes with a unique
identifier, which can be used to mark groups of operations that share a
common property. For example, groups of aliasing memory operations may be
marked using one DistinctAttribute instance per alias group.

## Operations

[include "Dialects/BuiltinOps.md"]

## Types

[include "Dialects/BuiltinTypes.md"]

## Type Interfaces

[include "Dialects/BuiltinTypeInterfaces.md"]
