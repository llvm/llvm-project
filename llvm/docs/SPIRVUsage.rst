=============================
User Guide for SPIR-V Target
=============================

.. contents::
   :local:

.. toctree::
   :hidden:

Introduction
============

The SPIR-V target provides code generation for the SPIR-V binary format described
in  `the official SPIR-V specification <https://www.khronos.org/registry/SPIR-V/>`_.

.. _spirv-target-triples:

Target Triples
==============

For cross-compilation into SPIR-V use option

``-target <Architecture><Subarchitecture>-<Vendor>-<OS>-<Environment>``

to specify the target triple:

  .. table:: SPIR-V Architectures

     ============ ==============================================================
     Architecture Description
     ============ ==============================================================
     ``spirv32``   SPIR-V with 32-bit pointer width.
     ``spirv64``   SPIR-V with 64-bit pointer width.
     ============ ==============================================================

  .. table:: SPIR-V Subarchitectures

     ============ ==============================================================
     Architecture Description
     ============ ==============================================================
     *<empty>*     SPIR-V version deduced by tools based on the compiled input.
     ``v1.0``      SPIR-V version 1.0.
     ``v1.1``      SPIR-V version 1.1.
     ``v1.2``      SPIR-V version 1.2.
     ``v1.3``      SPIR-V version 1.3.
     ``v1.4``      SPIR-V version 1.4.
     ``v1.5``      SPIR-V version 1.5.
     ============ ==============================================================

  .. table:: SPIR-V Vendors

     ===================== ==============================================================
     Vendor                Description
     ===================== ==============================================================
     *<empty>*/``unknown``  Generic SPIR-V target without any vendor-specific settings.
     ===================== ==============================================================

  .. table:: Operating Systems

     ===================== ============================================================
     OS                    Description
     ===================== ============================================================
     *<empty>*/``unknown``  Defaults to the OpenCL runtime.
     ===================== ============================================================

  .. table:: SPIR-V Environments

     ===================== ==============================================================
     Environment           Description
     ===================== ==============================================================
     *<empty>*/``unknown``  Defaults to the OpenCL environment.
     ===================== ==============================================================
