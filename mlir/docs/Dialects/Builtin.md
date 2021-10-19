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

## Operations

[include "Dialects/BuiltinOps.md"]

## Types

[include "Dialects/BuiltinTypes.md"]

## Type Interfaces

[include "Dialects/BuiltinTypeInterfaces.md"]
