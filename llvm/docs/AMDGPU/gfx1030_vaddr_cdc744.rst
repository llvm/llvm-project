..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid_gfx1030_vaddr_cdc744:

vaddr
=====

Image address which includes from one to four dimensional coordinates and other data used to locate a position in the image.

This operand may be specified using either :ref:`standard VGPR syntax<amdgpu_synid_v>` or special :ref:`NSA VGPR syntax<amdgpu_synid_nsa>`.

*Size:* 1-13 dwords. Actual size depends on syntax, opcode, :ref:`dim<amdgpu_synid_dim>` and :ref:`a16<amdgpu_synid_a16>`.

* If specified using :ref:`NSA VGPR syntax<amdgpu_synid_nsa>`, the size is 1-13 dwords.
* If specified using :ref:`standard VGPR syntax<amdgpu_synid_v>`, the size is 1-8 dwords. Opcodes which require more than 8 dwords for address size must specify 16 dwords due to a limited range of supported register sequences.

*Operands:* :ref:`v<amdgpu_synid_v>`
