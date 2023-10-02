================================
Floating Point Semantics in LLVM
================================

.. contents::
   :local:

Introduction
============
This document describes the design principles, goal, assumptions, and semantics
for floating-point representation and optimization in the LLVM optimizer.

This document is currently a work-in-progress and should be considered as an
RFC until the direction set forth here is agreed upon and approved.

IEC 60559 IEEE 754 Conformance
==================================
It is a goal of the LLVM project to enable the development of compilers that
conform to the IEC 60559 and IEEE 754 standards. The latest version of the
standard as of this writing is IEC 60559:2020, and that will be used as the
point of reference here.

Conformance to these standards requires collaboration between the compiler
front end, the LLVM optimizer, and the target-specific backend. This
document focuses on the IR constructs and semantics that are required in
order to enable various levels of conformance to the IEC 60559 standard.

Levels of conformance
---------------------
The degree of conformance to the IEC 60559 standard that is required is left
to the front end implementation and is expected to vary depending on various
controls set by the users (command-line options, pragmas, etc.). Here we
describe three basic levels of conformance: basic functionality,
numerical reproducibility, and strict semantic conformance.

Basic functionality
~~~~~~~~~~~~~~~~~~~

By "basic functionality" we mean that the required operations specified by
the IEC 60559 standard are supported and the prescribed numeric formats are
used. The `LLVM Language Reference Manual <LangRef.html>`_ describes the
floating-point instructions and intrinsics that are supported in LLVM. Unless
otherwise noted, these are assumed to have the basic behavior descrinbed in
the IEC 60559 standard for the equivalent operation. This does not include
the full exception semantics or dynamic rounding mode behavior described by
the standard, which are considered here to be beyond the scope of "basic
functionality."

While the latest version of the IEC 60559 standard requires all operations to
return correctly results for the applicable rounding direction, this is not
assumed to be part of the basic functionality conformance described here, and
it is not a general assumption of the LLVM IR definition.

Only the basic arithmetic operations (fadd, fsub, fmul, fdiv, and frem) can
be assumed to provide correctly rounded results, and even in those cases the
default rounding mode is assumed. Math library function calls and the LLVM
intrinsics which are defined in terms of those calls are governed by the
definition of those functions in the C language standard. Some of these,
such as sqrt and fma are required to return correctly rounded results.
Where the C language standard does not specifically state an accuracy
requirement the accuracy of the operation is regarded as unspecified in
the LLVM IR definition.

When numeric consistency of floating-point results is not required, clients
of the LLVM optimizer may use `fast-math flags <LangRef.html#_fastmath>`_
and other constructs to describe specific relaxation of the usual semantics.

Numerical Reproducibility
~~~~~~~~~~~~~~~~~~~~~~~~~

Section 10.4 of the IEC 60559 standard describes recommendations for defining
the literal meaning of the source code of a program and defines rules for
specific value-changing transformations which are still considered to preserve
the literal meaning of the source code. This level of reproducibility is
recommended by the standard but is not required. Here we describe a level of
conformance to the standard which seeks to preserve the numeric reproducibility
of a program without requiring full support for exception semantics or status
flags. This is the base expectation of LLVM IR semantics when no modifiers
(such as fast-math flags or constrained intrinsics) are used.

Basic numeric reproducibility is the default assumption for LLVM IR, with two
caveats. First, LLVM IR assumes that the default rounding mode
(round-to-nearest) will be used. All constant folding is performed using the
default rounding mode unless constrained floating-point intrinsics are used
to indicate that the rounding mode might have been changed. Second, there is
no guarantee or expectation of numeric reproducibility across different
targets and architectures. Numeric consistency across architectures requires
math library functions to be implemented in such a way that equivalent function
calls provide numerically consistent results on all architectures. Since LLVM
makes no assumptions about the math library which will be used, numerical
reproducibility is limited to reproducibility when the exact same math library
implementation is used.

All LLVM transformations should preserve the numeric results of the operations
involved unless some construct in the IR gives explicit permissions for a
value-changing transformation or the transformation is explicitly identified
in the IEC 60559 standard as a value-changing transformation that preserves the
literal meaning of the source code. More details will be provided below.


Strict semantic conformance
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Section 11 of the IEC 60559 standard describes recommendations for reproducible
floating-point results. The standard's description of reproducible results
includes both numerical results and status flags. The default floating-point
environment assumed by LLVM IR does not support strict semantic conformance.
Therefore, extensions must be used by any front end or other LLVM client that
intends to provide strict semantic conformance.

By default, the LLVM optimizer makes three assumptions that prevent strict
semantic conformance:

1. Floating-point operations do not have side-effects
2. All NaN values may be treated as if they were quiet NaNs
3. Results are rounded using the round-to-nearest method

The first two assumptions prevent strict exception semantics from being
represented using the basic LLVM IR floating-point operations. The third
prevents strict numeric reproducibility if a rounding-mode other than
round-to-nearest is used.

To achieve strict semantic conformance to the source code, a front end must use
`constrained-floating point intrinsics <LangRef.html#_constrainedfp>`_ and
follow the related rules when generating LLVM IR.


Consistency of Numeric Results
==============================

This section provides details on the expected handling of various issues
related to consistency of numeric results. This is intended both as a
normative reference for resolving questions about how such issues should
be handled and as a guide for understanding when and why value-changing
transformations are allowed.

Fused operations
----------------

fma (library call)
__builtin_fma
llvm.fma
llvm.fmuladd
contract fast-math flag

Math library function calls
---------------------------

builtin and nobuiltin attributes
Constant folding
Consistent results for same inputs
Conversion to intrinsics

LLVM math Intrinsics
--------------------

NaN payloads
------------

Denormal values
---------------

Hardware ftz/daz
"denromal-fp-math" attribute

Use of x87 instructions
-----------------------

Excess precision
Intermediate rounding
x87 precision control


Complex arithmetic
==================

Data representation
ABI issues
Range and domain


Floating-Point Environment
==========================

(This is copied from LangRef -- maybe it's not needed here)

The default LLVM floating-point environment assumes that traps are disabled and
status flags are not observable. Therefore, floating-point math operations do
not have side effects and may be speculated freely. Results assume the
round-to-nearest rounding mode.

Floating-point math operations are allowed to treat all NaNs as if they were
quiet NaNs. For example, "pow(1.0, SNaN)" may be simplified to 1.0. This also
means that SNaN may be passed through a math operation without quieting. For
example, "fmul SNaN, 1.0" may be simplified to SNaN rather than QNaN. However,
SNaN values are never created by math operations. They may only occur when
provided as a program input value.

Code that requires different behavior than this should use the
:ref:`Constrained Floating-Point Intrinsics <constrainedfp>`.


Recommendations to front end implementers
=========================================

Expectations of back-ends
=========================

