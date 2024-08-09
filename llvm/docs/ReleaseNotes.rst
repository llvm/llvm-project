============================
LLVM |release| Release Notes
============================

.. contents::
    :local:

.. only:: PreRelease

  .. warning::
     These are in-progress notes for the upcoming LLVM |version| release.
     Release notes for previous releases can be found on
     `the Download Page <https://releases.llvm.org/download.html>`_.


Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release |release|.  Here we describe the status of LLVM, including major improvements
from the previous release, improvements in various subprojects of LLVM, and
some of the current users of the code.  All LLVM releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

For more information about LLVM, including information about the latest
release, please check out the `main LLVM web site <https://llvm.org/>`_.  If you
have questions or comments, the `Discourse forums
<https://discourse.llvm.org>`_ is a good place to ask
them.

Note that if you are reading this file from a Git checkout or the main
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

* ...

Update on required toolchains to build LLVM
-------------------------------------------

Changes to the LLVM IR
----------------------

* The ``x86_mmx`` IR type has been removed. It will be translated to
  the standard vector type ``<1 x i64>`` in bitcode upgrade.

Changes to LLVM infrastructure
------------------------------

Changes to building LLVM
------------------------

Changes to TableGen
-------------------

Changes to Interprocedural Optimizations
----------------------------------------

Changes to the AArch64 Backend
------------------------------

* `.balign N, 0`, `.p2align N, 0`, `.align N, 0` in code sections will now fill
  the required alignment space with a sequence of `0x0` bytes (the requested
  fill value) rather than NOPs.

Changes to the AMDGPU Backend
-----------------------------

Changes to the ARM Backend
--------------------------

* `.balign N, 0`, `.p2align N, 0`, `.align N, 0` in code sections will now fill
  the required alignment space with a sequence of `0x0` bytes (the requested
  fill value) rather than NOPs.

Changes to the AVR Backend
--------------------------

Changes to the DirectX Backend
------------------------------

Changes to the Hexagon Backend
------------------------------

Changes to the LoongArch Backend
--------------------------------

Changes to the MIPS Backend
---------------------------

Changes to the PowerPC Backend
------------------------------

Changes to the RISC-V Backend
-----------------------------

* `.balign N, 0`, `.p2align N, 0`, `.align N, 0` in code sections will now fill
  the required alignment space with a sequence of `0x0` bytes (the requested
  fill value) rather than NOPs.
* Added Syntacore SCR4 CPUs: ``-mcpu=syntacore-scr4-rv32/64``
* ``-mcpu=sifive-p470`` was added.
* Fixed length vector support using RVV instructions now requires VLEN>=64. This
  means Zve32x and Zve32f will also require Zvl64b. The prior support was
  largely untested.

Changes to the WebAssembly Backend
----------------------------------

Changes to the Windows Target
-----------------------------

Changes to the X86 Backend
--------------------------

* `.balign N, 0x90`, `.p2align N, 0x90`, and `.align N, 0x90` in code sections
  now fill the required alignment space with repeating `0x90` bytes, rather than
  using optimised NOP filling. Optimised NOP filling fills the space with NOP
  instructions of various widths, not just those that use the `0x90` byte
  encoding. To use optimised NOP filling in a code section, leave off the
  "fillval" argument, i.e. `.balign N`, `.p2align N` or `.align N` respectively.

* Due to the removal of the ``x86_mmx`` IR type, functions with
  ``x86_mmx`` arguments or return values will use a different,
  incompatible, calling convention ABI. Such functions are not
  generally seen in the wild (Clang never generates them!), so this is
  not expected to result in real-world compatibility problems.

* Support ISA of ``AVX10.2-256`` and ``AVX10.2-512``.

Changes to the OCaml bindings
-----------------------------

Changes to the Python bindings
------------------------------

Changes to the C API
--------------------

* The following symbols are deleted due to the removal of the ``x86_mmx`` IR type:

  * ``LLVMX86_MMXTypeKind``
  * ``LLVMX86MMXTypeInContext``
  * ``LLVMX86MMXType``

Changes to the CodeGen infrastructure
-------------------------------------

Changes to the Metadata Info
---------------------------------

Changes to the Debug Info
---------------------------------

Changes to the LLVM tools
---------------------------------

Changes to LLDB
---------------------------------

Changes to BOLT
---------------------------------

Changes to Sanitizers
---------------------

Other Changes
-------------

External Open Source Projects Using LLVM 19
===========================================

* A project...

Additional Information
======================

A wide variety of additional information is available on the `LLVM web page
<https://llvm.org/>`_, in particular in the `documentation
<https://llvm.org/docs/>`_ section.  The web page also contains versions of the
API documentation which is up-to-date with the Git version of the source
code.  You can access versions of these documents specific to this release by
going into the ``llvm/docs/`` directory in the LLVM tree.

If you have any questions or comments about LLVM, please feel free to contact
us via the `Discourse forums <https://discourse.llvm.org>`_.
