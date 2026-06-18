==================
Callgraph Metadata
==================

Introduction
============

The ``!callgraph`` metadata is introduced to support the generation of a call graph
section in the object file. It associates a function definition with its generalized
type identifier.

Syntax
======

A ``!callgraph`` metadata node is attached to a function definition as follows:

.. code-block:: llvm

   define void @foo() !callgraph !0
   !0 = !{!"_ZTSFvvE.generalized"}

The metadata node is a 1-element tuple containing only the generalized type identifier
as an ``MDString``, without any offset.

Relation to Control Flow Integrity (CFI)
========================================

While ``!callgraph`` metadata is structurally similar to LLVM's ``!type`` metadata
(which is used by `Control Flow Integrity (CFI)`_ and Whole Program Devirtualization),
they serve different purposes:

* **!type (CFI)**: Contains an offset (e.g., ``!{i64 0, !"_ZTSFvvE.generalized"}``) to support virtual table offset calculations and devirtualization. This requires ThinLTO symbol promotion and LTO splitting.
* **!callgraph (Call Graph Section)**: Does not contain an offset. This is private to the Call Graph Section pipeline and bypasses ThinLTO promotion, avoiding unnecessary symbol export and LTO splitting bloat.

The generalized type identifier format used by both is identical. For more details on the
generalized type identifier format and CFI's metadata, see :doc:`TypeMetadata` and :doc:`CalleeTypeMetadata`.

.. _Control Flow Integrity (CFI): https://clang.llvm.org/docs/ControlFlowIntegrity.html
