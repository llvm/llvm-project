..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid8_dst_mimg_gather4:

vdst
===========================

Image data to load by an *image_gather4* instruction.

*Size:* 4 data elements by default. Each data element occupies either 32 bits or 16 bits depending on :ref:`d16<amdgpu_synid_d16>`.

:ref:`d16<amdgpu_synid_d16>` and :ref:`tfe<amdgpu_synid_tfe>` affect operand size as follows:

* :ref:`d16<amdgpu_synid_d16>` has different meaning for GFX8.0 and GFX8.1:

  * For GFX8.0 this modifier does not affect size of data elements in registers. Data in registers are stored in low 16 bits, high 16 bits are unused. There is no packing.
  * Starting from GFX8.1 this modifier specifies that data elements in registers are packed; each value occupies 16 bits.

* :ref:`tfe<amdgpu_synid_tfe>` adds one dword if specified.

*Operands:* :ref:`v<amdgpu_synid_v>`
