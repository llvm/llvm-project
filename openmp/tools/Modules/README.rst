=========================
LLVM OpenMP CMake Modules
=========================

This directory contains CMake modules for OpenMP. These can be included into a
project to include different OpenMP features.

.. contents::
   :local:

Find OpenMP Target Support
==========================

This module will attempt to find OpenMP target offloading support for a given
device. The module will attempt to compile a test program using known compiler
flags for each requested architecture. If successful, the flags required for
offloading will be loaded into the ``OpenMPTarget::OpenMPTarget_<device>``
target or the ``OpenMPTarget_NVPTX_FLAGS`` variable. Currently supported target
devices are ``NVPTX`` and ``AMDGCN``. This module is still under development so
some features may be missing.

To use this module, simply add the path to CMake's current module path and call
``find_package``. The module will be installed with your OpenMP installation by
default. Including OpenMP offloading support in an application should now only
require a few additions.

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.13.4)
  project(offloadTest VERSION 1.0 LANGUAGES CXX)
  
  list(APPEND CMAKE_MODULE_PATH "${PATH_TO_OPENMP_INSTALL}/lib/cmake/openmp")
  
  find_package(OpenMPTarget REQUIRED NVPTX)
  
  add_executable(offload)
  target_link_libraries(offload PRIVATE OpenMPTarget::OpenMPTarget_NVPTX)
  target_sources(offload PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/Main.cpp)

Using this module requires at least CMake version 3.13.4. Supported languages
are C and C++ with Fortran support planned in the future. If your application
requires building for a specific device architecture you can set the
``OpenMPTarget_<device>_ARCH=<flag>`` variable. Compiler support is best for
Clang but this module should work for other compiler vendors such as IBM or GNU.
