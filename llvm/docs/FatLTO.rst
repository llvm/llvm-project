===================
FatLTO
===================
.. contents::
   :local:
   :depth: 2

.. toctree::
   :maxdepth: 1

Introduction
============

FatLTO objects are a special type of `fat object file
<https://en.wikipedia.org/wiki/Fat_binary>`_ that contain LTO compatible IR in
addition to generated object code, instead of containing object code for
multiple target architectures. This allows users to defer the choice of whether
to use LTO or not to link-time, and has been a feature available in other
compilers, like `GCC
<https://gcc.gnu.org/onlinedocs/gccint/LTO-Overview.html>`_, for some time.

Under FatLTO the compiler can emit standard object files which contain both the
machine code in the ``.text`` section and LLVM bitcode in the ``.llvm.lto``
section.

Overview
========

Within LLVM, FatLTO is supported by choosing the ``FatLTODefaultPipeline``.
This pipeline will:

#) Clone the IR module.
#) Run the pre-link (Thin)LTO pipeline using the cloned module.
#) Embed the pre-link bitcode in a special ``.llvm.lto`` section.
#) Optimize the unmodified copy of the module using the normal compilation pipeline.
#) Emit the object file, including the new ``.llvm.lto`` section.

.. NOTE

   At the time of writing, we conservatively run independent pipelines to
   generate the bitcode section and the object code, which happen to be
   identical to those used outside of FatLTO. This results in  compiled
   artifacts that are identical to those produced by the default and (Thin)LTO
   pipelines. However, this is not a guarantee, and we reserve the right to
   change this at any time. Explicitly, users should not rely on the produced
   bitcode or object code to mach their non-LTO counterparts precisely. They
   will exhibit similar performance characteristics, but may not be bit-for-bit
   the same.

Internally, the ``.llvm.lto`` section is created by running the
``EmbedBitcodePass`` at the start of the ``PerModuleDefaultPipeline``. This
pass is responsible for cloning and optimizing the module with the appropriate
LTO pipeline and emitting the ``.llvm.lto`` section. Afterwards, the
``PerModuleDefaultPipeline`` runs normally and the compiler can emit the fat
object file.

Limitations
===========

Linkers
-------

Currently, using LTO with LLVM fat lto objects is supported by LLD and by the
GNU linkers via :doc:`GoldPlugin`. This may change in the future, but
extending support to other linkers isn't planned for now.

.. NOTE
   For standard linking the fat object files should be usable by any
   linker capable of using ELF objects, since the ``.llvm.lto`` section is
   marked ``SHF_EXLUDE``.

Supported File Formats
----------------------

The current implementation only supports ELF files. At time of writing, it is
unclear if it will be useful to support other object file formats like ``COFF``
or ``Mach-O``.
