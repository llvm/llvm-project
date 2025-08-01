====================
Callee Type Metadata
====================

Introduction
============
This ``!callee_type`` metadata is introduced to support the generation of a call graph
section in the object file.  The ``!callee_type`` metadata is used
to identify the types of the intended callees of indirect call instructions. The ``!callee_type`` metadata is a
list of one or more generalized ``!type`` metadata objects (See :doc:`TypeMetadata`) with each ``!type``
metadata pointing to a callee's :ref:`type identifier <calleetype-type-identifier>`.
LLVM's `Control Flow Integrity (CFI)`_ also uses the ``!type`` metadata in its implementation.

.. _Control Flow Integrity (CFI): https://clang.llvm.org/docs/ControlFlowIntegrity.html

.. _calleetype-type-identifier:

Type identifier
================

The type for an indirect call target is the callee's function signature.
Mapping from a type to an identifier is an ABI detail.
In the current implementation, an identifier of type T is
computed as follows:

  -  Obtain the generalized mangled name for “typeinfo name for T”.
  -  Compute MD5 hash of the name as a string.
  -  Reinterpret the first 8 bytes of the hash as a little-endian 64-bit integer.

To avoid mismatched pointer types, generalizations are applied.
Pointers in return and argument types are treated as equivalent as long as the qualifiers for the 
type they point to match. For example, ``char*``, ``char**``, and ``int*`` are considered equivalent
types. However, ``char*`` and ``const char*`` are considered distinct types.
