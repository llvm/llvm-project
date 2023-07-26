# MLIR Release Notes

This document tries to provide some context about MLIR important changes in the
context of LLVM releases. It is updated on a best effort basis.

At the moment the MLIR community does not qualify the LLVM release branch
specifically, it is a snapshot of the MLIR development at the time of the release.

[TOC]

## LLVM 17

See also the [deprecations and refactoring](https://mlir.llvm.org/deprecation/) doc.

### Bytecode

MLIR now support a [bytecode serialization](https://mlir.llvm.org/docs/BytecodeFormat/)
with versionning compatibility allowing 2 ways compatibility scheme, and lazy-loading
capabilities.

### Properties: beyond attributes

This is a new mechanism to implement storage for operations without having to
use attributes. You can opt-in to use Properties for ODS inherent attributes
using `let usePropertiesForAttributes = 1;` in your dialect definition (the flag
will be default in the next release). See
[slides](https://mlir.llvm.org/OpenMeetings/2023-02-09-Properties.pdf) and
[recording](https://youtu.be/7ofnlCFzlqg) of the open meeting presentation for
details.

### Action: Tracing and Debugging MLIR-based Compilers

[Action](https://mlir.llvm.org/docs/ActionTracing/) is a new mechanism to
encapsulate any transformation of any granularity in a way that can be
intercepted by the framework for debugging or tracing purposes, including
skipping a transformation programmatically (think about “compiler fuel” or
“debug counters” in LLVM). As such, “executing a pass” is an Action, so is “try
to apply one canonicalization pattern”, or “tile this loop”.

[slides](https://mlir.llvm.org/OpenMeetings/2023-02-23-Actions.pdf) and
[recording](https://youtu.be/ayQSyekVa3c) of the open meeting presentation for
details.

### Transform Dialect

See this [EuroLLVM talk](https://www.youtube.com/watch?v=P4gUj3QtH_Y&t=1s) and
[the online tutorial](https://mlir.llvm.org/docs/Tutorials/transform/).

### Others

- There is now support for
  "[distinct attributes](https://mlir.llvm.org/docs/Dialects/Builtin/#distinctattribute)".
- "Resources" (a way to store data outside the MLIR context) and "configuration"
  can now be serialized alongside the IR.
