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

     =============== ==============================================================
     Subarchitecture Description
     =============== ==============================================================
     *<empty>*        SPIR-V version deduced by tools based on the compiled input.
     ``v1.0``         SPIR-V version 1.0.
     ``v1.1``         SPIR-V version 1.1.
     ``v1.2``         SPIR-V version 1.2.
     ``v1.3``         SPIR-V version 1.3.
     ``v1.4``         SPIR-V version 1.4.
     ``v1.5``         SPIR-V version 1.5.
     =============== ==============================================================

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

Example:

``-target spirv64v1.0`` can be used to compile for SPIR-V version 1.0 with 64-bit pointer width.

.. _spirv-types:

Representing special types in SPIR-V
====================================

SPIR-V specifies several kinds of opaque types. These types are represented
using target extension types. These types are represented as follows:

  .. table:: SPIR-V Opaque Types

     ================== ====================== =========================================================================================
     SPIR-V Type        LLVM type name         LLVM type arguments
     ================== ====================== =========================================================================================
     OpTypeImage        ``spirv.Image``        sampled type, dimensionality, depth, arrayed, MS, sampled, image format, access qualifier
     OpTypeSampler      ``spirv.Sampler``      (none)
     OpTypeSampledImage ``spirv.SampledImage`` sampled type, dimensionality, depth, arrayed, MS, sampled, image format, access qualifier
     OpTypeEvent        ``spirv.Event``        (none)
     OpTypeDeviceEvent  ``spirv.DeviceEvent``  (none)
     OpTypeReserveId    ``spirv.ReserveId``    (none)
     OpTypeQueue        ``spirv.Queue``        (none)
     OpTypePipe         ``spirv.Pipe``         access qualifier
     OpTypePipeStorage  ``spirv.PipeStorage``  (none)
     ================== ====================== =========================================================================================

All integer arguments take the same value as they do in their `corresponding
SPIR-V instruction <https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_type_declaration_instructions>`_.
For example, the OpenCL type ``image2d_depth_ro_t`` would be represented in
SPIR-V IR as ``target("spirv.Image", void, 1, 1, 0, 0, 0, 0, 0)``, with its
dimensionality parameter as ``1`` meaning 2D. Sampled image types include the
parameters of its underlying image type, so that a sampled image for the
previous type has the representation
``target("spirv.SampledImage, void, 1, 1, 0, 0, 0, 0, 0)``.
