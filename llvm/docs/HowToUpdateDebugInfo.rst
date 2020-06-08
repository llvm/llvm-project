=======================================================
How to Update Debug Info: A Guide for LLVM Pass Authors
=======================================================

.. contents::
   :local:

Introduction
============

Certain kinds of code transformations can inadvertently result in a loss of
debug info, or worse, make debug info misrepresent the state of a program.

This document specifies how to correctly update debug info in various kinds of
code transformations, and offers suggestions for how to create targeted debug
info tests for arbitrary transformations.

For more on the philosophy behind LLVM debugging information, see
:doc:`SourceLevelDebugging`.

IR-level transformations
========================

Deleting an Instruction
-----------------------

When an ``Instruction`` is deleted, its debug uses change to ``undef``. This is
a loss of debug info: the value of a one or more source variables becomes
unavailable, starting with the ``llvm.dbg.value(undef, ...)``. When there is no
way to reconstitute the value of the lost instruction, this is the best
possible outcome. However, it's often possible to do better:

* If the dying instruction can be RAUW'd, do so. The
  ``Value::replaceAllUsesWith`` API transparently updates debug uses of the
  dying instruction to point to the replacement value.

* If the dying instruction cannot be RAUW'd, call ``llvm::salvageDebugInfo`` on
  it. This makes a best-effort attempt to rewrite debug uses of the dying
  instruction by describing its effect as a ``DIExpression``.

* If one of the **operands** of a dying instruction would become trivially
  dead, use ``llvm::replaceAllDbgUsesWith`` to rewrite the debug uses of that
  operand. Consider the following example function:

.. code-block:: llvm

  define i16 @foo(i16 %a) {
    %b = sext i16 %a to i32
    %c = and i32 %b, 15
    call void @llvm.dbg.value(metadata i32 %c, ...)
    %d = trunc i32 %c to i16
    ret i16 %d
  }

Now, here's what happens after the unnecessary truncation instruction ``%d`` is
replaced with a simplified instruction:

.. code-block:: llvm

  define i16 @foo(i16 %a) {
    call void @llvm.dbg.value(metadata i32 undef, ...)
    %simplified = and i16 %a, 15
    ret i16 %simplified
  }

Note that after deleting ``%d``, all uses of its operand ``%c`` become
trivially dead. The debug use which used to point to ``%c`` is now ``undef``,
and debug info is needlessly lost.

To solve this problem, do:

.. code-block:: cpp

  llvm::replaceAllDbgUsesWith(%c, theSimplifiedAndInstruction, ...)

This results in better debug info because the debug use of ``%c`` is preserved:

.. code-block:: llvm

  define i16 @foo(i16 %a) {
    %simplified = and i16 %a, 15
    call void @llvm.dbg.value(metadata i16 %simplified, ...)
    ret i16 %simplified
  }

You may have noticed that ``%simplified`` is narrower than ``%c``: this is not
a problem, because ``llvm::replaceAllDbgUsesWith`` takes care of inserting the
necessary conversion operations into the DIExpressions of updated debug uses.

Deleting a MIR-level MachineInstr
---------------------------------

TODO

How to automatically convert tests into debug info tests
========================================================

.. _IRDebugify:

Mutation testing for IR-level transformations
---------------------------------------------

An IR test case for a transformation can, in many cases, be automatically
mutated to test debug info handling within that transformation. This is a
simple way to test for proper debug info handling.

The ``debugify`` utility
^^^^^^^^^^^^^^^^^^^^^^^^

The ``debugify`` testing utility is just a pair of passes: ``debugify`` and
``check-debugify``.

The first applies synthetic debug information to every instruction of the
module, and the second checks that this DI is still available after an
optimization has occurred, reporting any errors/warnings while doing so.

The instructions are assigned sequentially increasing line locations, and are
immediately used by debug value intrinsics everywhere possible.

For example, here is a module before:

.. code-block:: llvm

   define void @f(i32* %x) {
   entry:
     %x.addr = alloca i32*, align 8
     store i32* %x, i32** %x.addr, align 8
     %0 = load i32*, i32** %x.addr, align 8
     store i32 10, i32* %0, align 4
     ret void
   }

and after running ``opt -debugify``:

.. code-block:: llvm

   define void @f(i32* %x) !dbg !6 {
   entry:
     %x.addr = alloca i32*, align 8, !dbg !12
     call void @llvm.dbg.value(metadata i32** %x.addr, metadata !9, metadata !DIExpression()), !dbg !12
     store i32* %x, i32** %x.addr, align 8, !dbg !13
     %0 = load i32*, i32** %x.addr, align 8, !dbg !14
     call void @llvm.dbg.value(metadata i32* %0, metadata !11, metadata !DIExpression()), !dbg !14
     store i32 10, i32* %0, align 4, !dbg !15
     ret void, !dbg !16
   }

   !llvm.dbg.cu = !{!0}
   !llvm.debugify = !{!3, !4}
   !llvm.module.flags = !{!5}

   !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
   !1 = !DIFile(filename: "debugify-sample.ll", directory: "/")
   !2 = !{}
   !3 = !{i32 5}
   !4 = !{i32 2}
   !5 = !{i32 2, !"Debug Info Version", i32 3}
   !6 = distinct !DISubprogram(name: "f", linkageName: "f", scope: null, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !8)
   !7 = !DISubroutineType(types: !2)
   !8 = !{!9, !11}
   !9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
   !10 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
   !11 = !DILocalVariable(name: "2", scope: !6, file: !1, line: 3, type: !10)
   !12 = !DILocation(line: 1, column: 1, scope: !6)
   !13 = !DILocation(line: 2, column: 1, scope: !6)
   !14 = !DILocation(line: 3, column: 1, scope: !6)
   !15 = !DILocation(line: 4, column: 1, scope: !6)
   !16 = !DILocation(line: 5, column: 1, scope: !6)

Using ``debugify``
^^^^^^^^^^^^^^^^^^

A simple way to use ``debugify`` is as follows:

.. code-block:: bash

  $ opt -debugify -pass-to-test -check-debugify sample.ll

This will inject synthetic DI to ``sample.ll`` run the ``pass-to-test`` and
then check for missing DI. The ``-check-debugify`` step can of course be
omitted in favor of more customizable FileCheck directives.

Some other ways to run debugify are available:

.. code-block:: bash

   # Same as the above example.
   $ opt -enable-debugify -pass-to-test sample.ll

   # Suppresses verbose debugify output.
   $ opt -enable-debugify -debugify-quiet -pass-to-test sample.ll

   # Prepend -debugify before and append -check-debugify -strip after
   # each pass on the pipeline (similar to -verify-each).
   $ opt -debugify-each -O2 sample.ll

In order for ``check-debugify`` to work, the DI must be coming from
``debugify``. Thus, modules with existing DI will be skipped.

``debugify`` can be used to test a backend, e.g:

.. code-block:: bash

   $ opt -debugify < sample.ll | llc -o -

There is also a MIR-level debugify pass that can be run before each backend
pass, see:
:ref:`Mutation testing for MIR-level transformations<MIRDebugify>`.

``debugify`` in regression tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The output of the ``debugify`` pass must be stable enough to use in regression
tests. Changes to this pass are not allowed to break existing tests.

.. note::

   Regression tests must be robust. Avoid hardcoding line/variable numbers in
   check lines. In cases where this can't be avoided (say, if a test wouldn't
   be precise enough), moving the test to its own file is preferred.

.. _MIRDebugify:

Mutation testing for MIR-level transformations
----------------------------------------------

A variant of the ``debugify`` utility described in
:ref:`Mutation testing for IR-level transformations<IRDebugify>` can be used
for MIR-level transformations as well: much like the IR-level pass,
``mir-debugify`` inserts sequentially increasing line locations to each
``MachineInstr`` in a ``Module`` (although there is no equivalent MIR-level
``check-debugify`` pass).

For example, here is a snippet before:

.. code-block:: llvm

  name:            test
  body:             |
    bb.1 (%ir-block.0):
      %0:_(s32) = IMPLICIT_DEF
      %1:_(s32) = IMPLICIT_DEF
      %2:_(s32) = G_CONSTANT i32 2
      %3:_(s32) = G_ADD %0, %2
      %4:_(s32) = G_SUB %3, %1

and after running ``llc -run-pass=mir-debugify``:

.. code-block:: llvm

  name:            test
  body:             |
    bb.0 (%ir-block.0):
      %0:_(s32) = IMPLICIT_DEF debug-location !12
      DBG_VALUE %0(s32), $noreg, !9, !DIExpression(), debug-location !12
      %1:_(s32) = IMPLICIT_DEF debug-location !13
      DBG_VALUE %1(s32), $noreg, !11, !DIExpression(), debug-location !13
      %2:_(s32) = G_CONSTANT i32 2, debug-location !14
      DBG_VALUE %2(s32), $noreg, !9, !DIExpression(), debug-location !14
      %3:_(s32) = G_ADD %0, %2, debug-location !DILocation(line: 4, column: 1, scope: !6)
      DBG_VALUE %3(s32), $noreg, !9, !DIExpression(), debug-location !DILocation(line: 4, column: 1, scope: !6)
      %4:_(s32) = G_SUB %3, %1, debug-location !DILocation(line: 5, column: 1, scope: !6)
      DBG_VALUE %4(s32), $noreg, !9, !DIExpression(), debug-location !DILocation(line: 5, column: 1, scope: !6)

By default, ``mir-debugify`` inserts ``DBG_VALUE`` instructions **everywhere**
it is legal to do so.  In particular, every (non-PHI) machine instruction that
defines a register must be followed by a ``DBG_VALUE`` use of that def.  If
an instruction does not define a register, but can be followed by a debug inst,
MIRDebugify inserts a ``DBG_VALUE`` that references a constant.  Insertion of
``DBG_VALUE``'s can be disabled by setting ``-debugify-level=locations``.

To run MIRDebugify once, simply insert ``mir-debugify`` into your ``llc``
invocation, like:

.. code-block:: bash

  # Before some other pass.
  $ llc -run-pass=mir-debugify,other-pass ...

  # After some other pass.
  $ llc -run-pass=other-pass,mir-debugify ...

To run MIRDebugify before each pass in a pipeline, use
``-debugify-and-strip-all-safe``. This can be combined with ``-start-before``
and ``-start-after``. For example:

.. code-block:: bash

  $ llc -debugify-and-strip-all-safe -run-pass=... <other llc args>
  $ llc -debugify-and-strip-all-safe -O1 <other llc args>

To strip out all debug info from a test, use ``mir-strip-debug``, like:

.. code-block:: bash

  $ llc -run-pass=mir-debugify,other-pass,mir-strip-debug

It can be useful to combine ``mir-debugify`` and ``mir-strip-debug`` to
identify backend transformations which break in the presence of debug info.
For example, to run the AArch64 backend tests with all normal passes
"sandwiched" in between MIRDebugify and MIRStripDebugify mutation passes, run:

.. code-block:: bash

  $ llvm-lit test/CodeGen/AArch64 -Dllc="llc -debugify-and-strip-all-safe"

Using LostDebugLocObserver
--------------------------

TODO
