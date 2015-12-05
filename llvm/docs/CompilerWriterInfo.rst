========================================================
Architecture & Platform Information for Compiler Writers
========================================================

.. contents::
   :local:

.. note::

  This document is a work-in-progress.  Additions and clarifications are
  welcome.

Hardware
========

ARM
---

* `ARM documentation <http://www.arm.com/documentation/>`_ (`Processor Cores <http://www.arm.com/documentation/ARMProcessor_Cores/>`_ Cores)

* `ABI <http://www.arm.com/products/DevTools/ABI.html>`_

* `ABI Addenda and Errata <http://infocenter.arm.com/help/topic/com.arm.doc.ihi0045d/IHI0045D_ABI_addenda.pdf>`_

* `ARM C Language Extensions <http://infocenter.arm.com/help/topic/com.arm.doc.ihi0053c/IHI0053C_acle_2_0.pdf>`_

AArch64
-------

* `ARMv8 Architecture Reference Manual <http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.ddi0487a.h/index.html>`_

* `ARMv8 Instruction Set Overview <http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.genc010197a/index.html>`_

* `ARM C Language Extensions <http://infocenter.arm.com/help/topic/com.arm.doc.ihi0053c/IHI0053C_acle_2_0.pdf>`_

Itanium (ia64)
--------------

* `Itanium documentation <http://developer.intel.com/design/itanium2/documentation.htm>`_

MIPS
----

* `MIPS Processor Architecture <http://imgtec.com/mips/mips-architectures.asp>`_

* `MIPS 64-bit ELF Object File Specification <http://techpubs.sgi.com/library/manuals/4000/007-4658-001/pdf/007-4658-001.pdf>`_

PowerPC
-------

IBM - Official manuals and docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Power Instruction Set Architecture, Versions 2.03 through 2.06 (authentication required, free sign-up) <https://www.power.org/technology-introduction/standards-specifications>`_

* `PowerPC Compiler Writer's Guide <http://www.ibm.com/chips/techlib/techlib.nsf/techdocs/852569B20050FF7785256996007558C6>`_

* `Intro to PowerPC Architecture <http://www.ibm.com/developerworks/linux/library/l-powarch/>`_

* `PowerPC Processor Manuals (embedded) <http://www.ibm.com/chips/techlib/techlib.nsf/products/PowerPC>`_

* `Various IBM specifications and white papers <https://www.power.org/documentation/?document_company=105&document_category=all&publish_year=all&grid_order=DESC&grid_sort=title>`_

* `IBM AIX/5L for POWER Assembly Reference <http://publibn.boulder.ibm.com/doc_link/en_US/a_doc_lib/aixassem/alangref/alangreftfrm.htm>`_

Other documents, collections, notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `PowerPC ABI documents <http://penguinppc.org/dev/#library>`_
* `PowerPC64 alignment of long doubles (from GCC) <http://gcc.gnu.org/ml/gcc-patches/2003-09/msg00997.html>`_
* `Long branch stubs for powerpc64-linux (from binutils) <http://sources.redhat.com/ml/binutils/2002-04/msg00573.html>`_

AMDGPU
------

* `AMD R6xx shader ISA <http://developer.amd.com/wordpress/media/2012/10/R600_Instruction_Set_Architecture.pdf>`_
* `AMD R7xx shader ISA <http://developer.amd.com/wordpress/media/2012/10/R700-Family_Instruction_Set_Architecture.pdf>`_
* `AMD Evergreen shader ISA <http://developer.amd.com/wordpress/media/2012/10/AMD_Evergreen-Family_Instruction_Set_Architecture.pdf>`_
* `AMD Cayman/Trinity shader ISA <http://developer.amd.com/wordpress/media/2012/10/AMD_HD_6900_Series_Instruction_Set_Architecture.pdf>`_
* `AMD Southern Islands Series ISA <http://developer.amd.com/wordpress/media/2012/12/AMD_Southern_Islands_Instruction_Set_Architecture.pdf>`_
* `AMD Sea Islands Series ISA <http://developer.amd.com/wordpress/media/2013/07/AMD_Sea_Islands_Instruction_Set_Architecture.pdf>`_
* `AMD GPU Programming Guide <http://developer.amd.com/download/AMD_Accelerated_Parallel_Processing_OpenCL_Programming_Guide.pdf>`_
* `AMD Compute Resources <http://developer.amd.com/tools/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/documentation/>`_

SPARC
-----

* `SPARC standards <http://sparc.org/standards>`_
* `SPARC V9 ABI <http://sparc.org/standards/64.psabi.1.35.ps.Z>`_
* `SPARC V8 ABI <http://sparc.org/standards/psABI3rd.pdf>`_

SystemZ
-------

* `z/Architecture Principles of Operation (registration required, free sign-up) <http://www-01.ibm.com/support/docview.wss?uid=isg2b9de5f05a9d57819852571c500428f9a>`_

X86
---

AMD - Official manuals and docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `AMD processor manuals <http://www.amd.com/us-en/Processors/TechnicalResources/0,,30_182_739,00.html>`_
* `X86-64 ABI <http://www.x86-64.org/documentation>`_

Intel - Official manuals and docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Intel 64 and IA-32 manuals <http://www.intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html>`_
* `Intel Itanium documentation <http://www.intel.com/design/itanium/documentation.htm?iid=ipp_srvr_proc_itanium2+techdocs>`_

Other x86-specific information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Calling conventions for different C++ compilers and operating systems  <http://www.agner.org/optimize/calling_conventions.pdf>`_

XCore
-----

* `The XMOS XS1 Architecture (ISA) <https://www.xmos.com/en/download/public/The-XMOS-XS1-Architecture%28X7879A%29.pdf>`_
* `Tools Development Guide (includes ABI) <https://www.xmos.com/download/public/Tools-Development-Guide%28X9114A%29.pdf>`_

Other relevant lists
--------------------

* `GCC reading list <http://gcc.gnu.org/readings.html>`_

ABI
===

* `System V Application Binary Interface <http://www.sco.com/developers/gabi/latest/contents.html>`_
* `Itanium C++ ABI <http://mentorembedded.github.io/cxx-abi/>`_

Linux
-----

* `PowerPC 64-bit ELF ABI Supplement <http://www.linuxbase.org/spec/ELF/ppc64/>`_
* `Procedure Call Standard for the AArch64 Architecture <http://infocenter.arm.com/help/topic/com.arm.doc.ihi0055a/IHI0055A_aapcs64.pdf>`_
* `ELF for the ARM Architecture <http://infocenter.arm.com/help/topic/com.arm.doc.ihi0044e/IHI0044E_aaelf.pdf>`_
* `ELF for the ARM 64-bit Architecture (AArch64) <http://infocenter.arm.com/help/topic/com.arm.doc.ihi0056a/IHI0056A_aaelf64.pdf>`_
* `System z ELF ABI Supplement <http://legacy.redhat.com/pub/redhat/linux/7.1/es/os/s390x/doc/lzsabi0.pdf>`_

OS X
----

* `Mach-O Runtime Architecture <http://developer.apple.com/documentation/Darwin/RuntimeArchitecture-date.html>`_
* `Notes on Mach-O ABI <http://www.unsanity.org/archives/000044.php>`_

Windows
-------

* `Microsoft PE/COFF Specification <http://www.microsoft.com/whdc/system/platform/firmware/pecoff.mspx>`_

NVPTX
=====

* `CUDA Documentation <http://docs.nvidia.com/cuda/index.html>`_ includes the PTX
  ISA and Driver API documentation

Miscellaneous Resources
=======================

* `Executable File Format library <http://www.nondot.org/sabre/os/articles/ExecutableFileFormats/>`_

* `GCC prefetch project <http://gcc.gnu.org/projects/prefetch.html>`_ page has a
  good survey of the prefetching capabilities of a variety of modern
  processors.
