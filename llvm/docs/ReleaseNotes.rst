========================
LLVM 9.0.0 Release Notes
========================

.. contents::
    :local:

.. warning::
   These are in-progress notes for the upcoming LLVM 9 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.


Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release 9.0.0.  Here we describe the status of LLVM, including major improvements
from the previous release, improvements in various subprojects of LLVM, and
some of the current users of the code.  All LLVM releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

For more information about LLVM, including information about the latest
release, please check out the `main LLVM web site <https://llvm.org/>`_.  If you
have questions or comments, the `LLVM Developer's Mailing List
<https://lists.llvm.org/mailman/listinfo/llvm-dev>`_ is a good place to send
them.

Note that if you are reading this file from a Subversion checkout or the main
LLVM web page, this document applies to the *next* release, not the current
one.  To see the release notes for a specific release, please see the `releases
page <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================
.. NOTE
   For small 1-3 sentence descriptions, just add an entry at the end of
   this list. If your description won't fit comfortably in one bullet
   point (e.g. maybe you would like to give an example of the
   functionality, or simply have a lot to talk about), see the `NOTE` below
   for adding a new subsection.

* The optimizer will now convert calls to ``memcmp`` into a calls to ``bcmp`` in
  some circumstances. Users who are building freestanding code (not depending on
  the platform's libc) without specifying ``-ffreestanding`` may need to either
  pass ``-fno-builtin-bcmp``, or provide a ``bcmp`` function.

* Two new extension points, namely ``EP_FullLinkTimeOptimizationEarly`` and
  ``EP_FullLinkTimeOptimizationLast`` are available for plugins to specialize
  the legacy pass manager full LTO pipeline.

* **llvm-objcopy/llvm-strip** got support for COFF object files/executables,
  supporting the most common copying/stripping options.

* The CMake parameter ``CLANG_ANALYZER_ENABLE_Z3_SOLVER`` has been replaced by
  ``LLVM_ENABLE_Z3_SOLVER``.


.. NOTE
   If you would like to document a larger change, then you can add a
   subsection about it right here. You can copy the following boilerplate
   and un-indent it (the indentation causes it to be inside this comment).

   Special New Feature
   -------------------

   Makes programs 10x faster by doing Special New Thing.

Noteworthy optimizations
------------------------

* LLVM will now remove stores to constant memory (since this is a
  contradiction) under the assumption the code in question must be dead.  This
  has proven to be problematic for some C/C++ code bases which expect to be
  able to cast away 'const'.  This is (and has always been) undefined
  behavior, but up until now had not been actively utilized for optimization
  purposes in this exact way.  For more information, please see:
  `bug 42763 <https://bugs.llvm.org/show_bug.cgi?id=42763>_` and
  `post commit discussion <http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20190422/646945.html>_`.  


Changes to the LLVM IR
----------------------

* Added ``immarg`` parameter attribute. This indicates an intrinsic
  parameter is required to be a simple constant. This annotation must
  be accurate to avoid possible miscompiles.

* The 2-field form of global variables ``@llvm.global_ctors`` and
  ``@llvm.global_dtors`` has been deleted. The third field of their element
  type is now mandatory. Specify `i8* null` to migrate from the obsoleted
  2-field form.

* The ``byval`` attribute can now take a type parameter:
  ``byval(<ty>)``. If present it must be identical to the argument's
  pointee type. In the next release we intend to make this parameter
  mandatory in preparation for opaque pointer types.

* ``atomicrmw xchg`` now allows floating point types

* ``atomicrmw`` now supports ``fadd`` and ``fsub``

Changes to building LLVM
------------------------

* Building LLVM with Visual Studio now requires version 2017 or later.


Changes to the ARM Backend
--------------------------

 During this release ...


Changes to the MIPS Target
--------------------------

* Support for ``.cplocal`` assembler directive.

* Support for ``sge``, ``sgeu``, ``sgt``, ``sgtu`` pseudo instructions.

* Support for ``o`` inline asm constraint.

* Improved support of GlobalISel instruction selection framework.
  This feature is still in experimental state for MIPS targets though.

* Various code-gen improvements, related to improved and fixed instruction
  selection and encoding and floating-point registers allocation.

* Complete P5600 scheduling model.


Changes to the PowerPC Target
-----------------------------

 During this release ...

Changes to the SystemZ Target
-----------------------------

* Support for the arch13 architecture has been added.  When using the
  ``-march=arch13`` option, the compiler will generate code making use of
  new instructions introduced with the vector enhancement facility 2
  and the miscellaneous instruction extension facility 2.
  The ``-mtune=arch13`` option enables arch13 specific instruction
  scheduling and tuning without making use of new instructions.

* Builtins for the new vector instructions have been added and can be
  enabled using the ``-mzvector`` option.  Support for these builtins
  is indicated by the compiler predefining the ``__VEC__`` macro to
  the value ``10303``.

* The compiler now supports and automatically generates alignment hints
  on vector load and store instructions.

* Various code-gen improvements, in particular related to improved
  instruction selection and register allocation.

Changes to the X86 Target
-------------------------

* Fixed a bug in generating DWARF unwind information for 32 bit MinGW

Changes to the AMDGPU Target
-----------------------------

* Function call support is now enabled by default

* Improved support for 96-bit loads and stores

* DPP combiner pass is now enabled by default

* Support for gfx10

Changes to the AVR Target
-----------------------------

 During this release ...

Changes to the WebAssembly Target
---------------------------------

 During this release ...


Changes to the OCaml bindings
-----------------------------



Changes to the C API
--------------------


Changes to the DAG infrastructure
---------------------------------

Changes to LLDB
===============

* Backtraces are now color highlighting in the terminal.

* DWARF4 (debug_types) and DWARF5 (debug_info) type units are now supported.

* This release will be the last where ``lldb-mi`` is shipped as part of LLDB.
  The tool will still be available in a `downstream repository on GitHub
  <https://github.com/lldb-tools/lldb-mi>`_.

External Open Source Projects Using LLVM 9
==========================================

* A project...


Additional Information
======================

A wide variety of additional information is available on the `LLVM web page
<https://llvm.org/>`_, in particular in the `documentation
<https://llvm.org/docs/>`_ section.  The web page also contains versions of the
API documentation which is up-to-date with the Subversion version of the source
code.  You can access versions of these documents specific to this release by
going into the ``llvm/docs/`` directory in the LLVM tree.

If you have any questions or comments about LLVM, please feel free to contact
us via the `mailing lists <https://llvm.org/docs/#mailing-lists>`_.
