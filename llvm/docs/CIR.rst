===============================
CIR - Clang IR Design and Implementation
===============================

.. contents::
   :local:

Introduction
============

This document aims to provide an overview of the design and
implementation of a Clang IR, a high level IR allowing more
analysis and future optimizations.

CIR is used as a short for ClangIR over commit messages and
other related code.

Usage in Clang
==============

Current strategy is to replace analysis based warnings with
analysis on top of CIR, using ``-fcir-warnings`` turns on such
analysis (current none).

The ``-fcir-output`` and ``-fcir-output=<file>`` flags can be used
to output the generated CIR (currently needs to be combined with
``-fcir-warnings`` to work).

Additionally, clang can run it's full compilation pipeline with
the CIR phase inserted between clang and llvm. Passing
``-fclangir`` to ``clang -cc1`` will opt in to clang generating
CIR which is lowered to LLVMIR and continued through the
backend. (WIP -- the backend is not yet functional).

A new flag ``-emit-cir`` can be used in combination with
``-fclangir`` to emit pristine CIR right out of the CIRGen phase.

Adding flags to select between different levels of lowerings
between MLIR dialects (e.g.to STD/Affine/SCF) are a WIP.


Implementation Notes
====================

- ``PopFunctionScopeInfo`` is the currentt entry point for CFG usage
in ``AnalysisBasedWarning.cpp``. The same entry point is used by the
CIR builder to emit functions.

TODO's
======
- LValues
  - Add proper alignment information
- Other module related emission besides functions (and all currently
end of translation defered stuff).
- Some data structures used for LLVM codegen can be made more
generic and be reused from CIRBuilder. Duplicating content right
now to prevent potential frequent merge conflicts.
  - Split out into header files all potential common code.
