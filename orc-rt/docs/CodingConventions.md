# ORC-RT Coding Conventions

## Overview

ORC-RT is part of the LLVM project and generally follows the
[LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html), including
LLVM's naming conventions for C++ code: `PascalCase` for types, values, and
variables, and `camelBack` (lower-camelCase) for functions.

This document records conventions that are specific to ORC-RT, or that need
restating because they are not obvious from the LLVM standards alone. Where this
document is silent, follow the LLVM Coding Standards.

## C API naming

The C API headers (under `include/orc-rt-c/`) cannot use C++ namespaces, so
namespacing is expressed through symbol-name prefixes, while the *casing* of
each name still follows the LLVM naming conventions used elsewhere in the
project.

The rules are:

- **`orc_rt_` is the top-level C namespace.** Every public C symbol begins with
  it.
- **Nested namespaces are additional lower-case, underscore-separated
  segments.** For example, `orc_rt_log_` is the nested namespace for the logging
  API.
- **Types, values, and variables are `PascalCase`** (as in LLVM), appended to
  their namespace prefix.
- **Functions are `camelBack`** (as in LLVM), appended to their namespace
  prefix.
- **Enumerators are `PascalCase` values, scoped by prefixing them with their
  enclosing type's name** (C has no scoped enums), joined with an underscore.
- **Macros are `UPPER_CASE`**, as in LLVM and C generally.

A PascalCase type name may itself act as a scope for its "member" functions,
exactly like a nested namespace does; the two are indistinguishable in the
flattened C name, and both are spelled with a trailing underscore.

### Examples

| Symbol                        | Kind     |
|-------------------------------|----------|
| `orc_rt_ErrorRef`             | type     |
| `orc_rt_Error_getTypeId`      | function |
| `orc_rt_StringError_create`   | function |
| `orc_rt_log_Category`         | type     |
| `orc_rt_log_Category_General` | value    |
| `orc_rt_log_formatCheck`      | function |
| `ORC_RT_LOG`                  | macro    |

Reading these: `orc_rt_Error_getTypeId` is the `getTypeId` function scoped to
the `Error` type; `orc_rt_log_formatCheck` is the `formatCheck` function in the
`log` namespace; and `orc_rt_log_Category_General` is the `General` value of the
`Category` type in the `log` namespace.

## C++ code

C++ code follows the LLVM Coding Standards directly: symbols live in real
namespaces (e.g. `orc_rt`), types/values/variables are `PascalCase`, functions
are `camelBack`, and macros are `UPPER_CASE`.

## Parameter ordering for asynchronous APIs

Many ORC-RT entry points are asynchronous: they take a completion callback that
is invoked, possibly on another thread, when the operation finishes. Because C
has no closures, such a callback is paired with an opaque context (a `void *`
supplied by the caller, or a framework-issued token such as a call id) that is
threaded back to it. To keep these APIs learnable as a family, order their
parameters consistently:

1. The primary handle or receiver (e.g. `orc_rt_SessionRef`) comes first.
2. The call inputs follow (e.g. a target/tag, an argument buffer).
3. The completion callback comes next, immediately followed by its context.
4. The context/token is always **last** -- in both the call and the callback
   signature.

For example, `orc_rt_WrapperFunction` is
`(S, ArgBytes, Return, CallId)` and its return callback
`orc_rt_WrapperFunctionReturn` is `(S, ResultBytes, CallId)`: the `CallId`
token that the framework threads from the call to its completion is last in
both. C++ helpers that mirror these signatures (e.g. `WrapperFunction::handle`)
follow the same order, appending any implementation-only parameters (such as a
serializer or handler) after the token.

This ordering is a convention, not a hard rule; where an existing API or an
external interface being wrapped dictates otherwise, match that instead.

