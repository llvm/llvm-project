===========================
OpenMP 16.0.0 Release Notes
===========================


.. warning::
   These are in-progress notes for the upcoming LLVM 16.0.0 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.


Introduction
============

This document contains the release notes for the OpenMP runtime, release 16.0.0.
Here we describe the status of OpenMP, including major improvements
from the previous release. All OpenMP releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

* Support for building the OpenMP runtime for Windows on AArch64 and ARM
  with MinGW based toolchains.

* Made the OpenMP runtime tests run successfully on Windows.

* Improved performance and internalization when compiling in LTO mode using 
  ``-foffload-lto``.

* Created the ``nvptx-arch`` and ``amdgpu-arch`` tools to query the user's 
  installed GPUs.

* Removed ``CLANG_OPENMP_NVPTX_DEFAULT_ARCH`` in favor of using the new 
  ``nvptx-arch`` tool.

* Added support for ``--offload-arch=native`` which queries the user's locally 
  available GPU architectures. Now ``-fopenmp --offload-arch=native`` is 
  sufficient to target all of the user's GPUs.

* Added ``-fopenmp-target-jit`` to enable JIT support.

* OpenMP now supports ``-Xarch_host`` to control sending compiler arguments only 
  to the host compilation.

* Improved ``clang-format`` when used on OpenMP offloading applications.
