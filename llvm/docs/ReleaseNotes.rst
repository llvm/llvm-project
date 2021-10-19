=========================
LLVM 13.0.0 Release Notes
=========================

.. contents::
    :local:


Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release 13.0.0.  Here we describe the status of LLVM, including major improvements
from the previous release, improvements in various subprojects of LLVM, and
some of the current users of the code.  All LLVM releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

For more information about LLVM, including information about the latest
release, please check out the `main LLVM web site <https://llvm.org/>`_.  If you
have questions or comments, the `LLVM Developer's Mailing List
<https://lists.llvm.org/mailman/listinfo/llvm-dev>`_ is a good place to send
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


.. NOTE
   If you would like to document a larger change, then you can add a
   subsection about it right here. You can copy the following boilerplate
   and un-indent it (the indentation causes it to be inside this comment).

   Special New Feature
   -------------------

   Makes programs 10x faster by doing Special New Thing.

* Windows Control-flow Enforcement Technology: the ``-ehcontguard`` option now
  emits valid unwind entrypoints which are validated when the context is being
  set during exception handling.

* Flang is now included in the binary packages released by LLVM.

* The debuginfo-test project has been renamed cross-project-tests and is now
  intended for testing components from multiple projects, not just debug
  information. The new "cross-project-tests" name replaces "debuginfo-test" in
  LLVM_ENABLE_PROJECTS, and a new check-cross-project-tests target has been
  added for running all tests in the project. The pre-existing check-debuginfo-
  test target remains for running just the debug information tests.
  (`D95339 <https://reviews.llvm.org/D95339>`_ and
  `D96513 <https://reviews.llvm.org/D96513>`_)

Changes to the LLVM IR
----------------------

* The ``inalloca`` attribute now has a mandatory type field, similar
  to ``byval`` and ``sret``.

* The opaque pointer type ``ptr`` has been introduced. It is still in the
  process of being worked on and should not be used yet.

* Using the legacy pass manager for the optimization pipeline is deprecated and
  will be removed after LLVM 14. In the meantime, only minimal effort will be
  made to maintain the legacy pass manager for the optimization pipeline.

Changes to building LLVM
------------------------

* The build system now supports building multiple distributions, so that you can
  e.g. have one distribution containing just tools and another for libraries (to
  enable development). See :ref:`Multi-distribution configurations` for details.

Changes to TableGen
-------------------

Changes to the AArch64 Backend
------------------------------

* Introduced assembly support for Armv9-A's Realm Management Extension (RME)
  and Scalable Matrix Extension (SME).

* Produce proper cross-section relative relocations on COFF

* Fixed the calling convention on Windows for variadic functions involving
  floats in the fixed arguments

Changes to the ARM Backend
--------------------------

* Produce proper cross-section relative relocations on COFF

Changes to the MIPS Target
--------------------------

During this release ...

Changes to the Hexagon Target
-----------------------------

* The Hexagon target now supports V68/HVX ISA.

Changes to the PowerPC Target
-----------------------------

During this release ...

Changes to the X86 Target
-------------------------

During this release ...

Changes to the AMDGPU Target
-----------------------------

During this release ...

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

* The C API functions ``LLVMGetAlignment`` and ``LLVMSetAlignment`` now allow
  changing alignment on atomicrmw and cmpxchg instructions

* A new entry ``LLVMDIArgListMetadataKind`` was added to the
  ``LLVMMetadataKind`` enum, representing DIArgLists
  (`D88175 <https://reviews.llvm.org/D88175>`_)

* Type attributes have been added to LLVM-C, introducing
  LLVMCreateTypeAttribute, LLVMGetTypeAttributeValue and LLVMIsTypeAttribute.
  (`D977763' <https://reviews.llvm.org/D97763>`_)

* The ``LTO_API_VERSION`` was bumped to 28, introducing a new function
  ``lto_set_debug_options`` for parsing LTO debug options
  (`D92611 <https://reviews.llvm.org/D92611>`_)

* ``LLVMJITTargetSymbolFlags`` was renamed to ``LLVMJITSymbolTargetFlags``
  (`rG8d718a0bff73af066675a6258c01307937c33cf9
  <https://reviews.llvm.org/rG8d718a0bff73af066675a6258c01307937c33cf9>`_)

* The C API received support for creating custom ORCv2 MaterializationUnits and
  APIs to retrieve an LLJIT instance's linking layers. An ABI breaking change
  for ``LLVMOrcAbsoluteSymbols`` was introduced to make it consistent with
  ``LLVMOrcCreateCustomMaterializationUnit``.
  (`rGc8fc5e3ba942057d6c4cdcd1faeae69a28e7b671
  <https://reviews.llvm.org/rGc8fc5e3ba942057d6c4cdcd1faeae69a28e7b671>`_)

* The C API received support for adding ORCv2 object buffers directly to an object
  layer. (`rG7b73cd684a8d5fb44d34064200f10e2723085c33
  <https://reviews.llvm.org/rG7b73cd684a8d5fb44d34064200f10e2723085c33>`_)

* A breaking change to ``LLVMGetInlineAsm`` was introduced, adding a ninth
  argument ``LLVMBool CanThrow`` (`D95745 <https://reviews.llvm.org/D95745>`_)

* The C API received support for calling into the new pass manager.
  (`D102136 <https://reviews.llvm.org/D102136>`_)

* The C API function ``LLVMIntrinsicCopyOverloadedName`` has been deprecated.
  Please migrate to ``LLVMIntrinsicCopyOverloadedName2`` which takes an extra
  module argument and which also handles unnamed types.
  (`D99173 <https://reviews.llvm.org/D99173>`_)

* The C API received support for dumping objects from ORCv2
  (`rGcec8e69f01c3374cb38c6683058381b96fab8f89
  <https://reviews.llvm.org/rGcec8e69f01c3374cb38c6683058381b96fab8f89>`_)

* A breaking change to ``LLVMOrcObjectTransformLayerTransformFunction`` was
  introduced, changing the order of the function pointer's arguments.
  (`rG8962c68ad007a525f9daa987c99eda57e0d0069a
  <https://reviews.llvm.org/rG8962c68ad007a525f9daa987c99eda57e0d0069a>`_)

* The C API received support for accessing utilities from the LLJIT
  ``IRTransformLayer`` and ``ThreadSafeModule`` classes. (`D103855
  <https://reviews.llvm.org/D103855>`_)

* The C API received support for creating lazy-export MaterializationUnits
  (`D104672 <https://reviews.llvm.org/D104672>`_)

* The C API function ``LLVMPassBuilderOptionsSetCoroutines`` was removed because
  couroutine passes have been enabled by default. (`D105877
  <https://reviews.llvm.org/D105877>`_)

* ``comdat noduplicates`` was renamed to ``comdat nodeduplicate`` and as a
  result, ``LLVMNoDuplicatesComdatSelectionKind`` was renamed to
  ``LLVMNoDeduplicateComdatSelectionKind``. (`D106319
  <https://reviews.llvm.org/D106319>`_)

Changes to the Go bindings
--------------------------


Changes to the FastISel infrastructure
--------------------------------------

* FastISel no longer tracks killed registers, and instead leaves this to the
  register allocator. This means that ``hasTrivialKill()`` is removed, as well
  as the ``OpNIsKill`` parameters to the ``fastEmit_*()`` family of functions.

Changes to the DAG infrastructure
---------------------------------


Changes to the Debug Info
---------------------------------

During this release ...

Changes to the LLVM tools
---------------------------------

* The options ``--build-id-link-{dir,input,output}`` have been deleted.
  (`D96310 <https://reviews.llvm.org/D96310>`_)

* Support for in-order processors has been added to ``llvm-mca``.
  (`D94928 <https://reviews.llvm.org/D94928>`_)

* llvm-objdump supports ``-M {att,intel}`` now.
  ``--x86-asm-syntax`` is a deprecated internal option which will be removed in LLVM 14.0.0.
  (`D101695 <https://reviews.llvm.org/D101695>`_)

* The llvm-readobj short aliases ``-s`` (previously ``--sections``) and ``-t``
  (previously ``--syms``) have been changed to ``--syms`` and
  ``--section-details`` respectively, to match llvm-readelf.
  (`D105055 <https://reviews.llvm.org/D105055>`_)

* The llvm-nm short aliases ``-M`` (``--print-armap``), ``-U``
  (``--defined-only``), and ``-W`` (``--no-weak``) are now deprecated.
  Use the long form versions instead.
  The alias ``--just-symbol-name`` is now deprecated in favor of
  ``--format=just-symbols`` and ``-j``.
  (`D105330 <https://reviews.llvm.org/D105330>`_)

* In lli the default JIT engine switched from MCJIT (``-jit-kind=mcjit``) to ORC (``-jit-kind=orc``).
  (`D98931 <https://reviews.llvm.org/D98931>`_)

* llvm-rc got support for invoking Clang to preprocess its input.
  (`D100755 <https://reviews.llvm.org/D100755>`_)

* llvm-rc got a GNU windres compatible frontend, llvm-windres.
  (`D100756 <https://reviews.llvm.org/D100756>`_)

* llvm-ml has improved compatibility with MS ml.exe, managing to assemble
  more asm files.

Changes to LLDB
---------------------------------

* LLDB executable is now included in pre-built LLVM binaries.

* LLDB now includes full featured support for AArch64 SVE register access.

* LLDB now supports AArch64 Pointer Authentication, allowing stack unwind with signed return address.

* LLDB now supports debugging programs on AArch64 Linux that use memory tagging (MTE).
* Added ``memory tag read`` and ``memory tag write`` commands.
* The ``memory region`` command will note when a region has memory tagging enabled.
* Synchronous and asynchronous tag faults are recognised.
* Synchronous tag faults have memory tag annotations in addition to the usual fault address.

Changes to Sanitizers
---------------------

External Open Source Projects Using LLVM 13
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
us via the `mailing lists <https://llvm.org/docs/#mailing-lists>`_.
