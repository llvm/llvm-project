..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid10_addr_mimg:

vaddr
===========================

Image address which includes from one to four dimensional coordinates and other data used to locate a position in the image.

This operand may be specified using either :ref:`standard VGPR syntax<amdgpu_synid_v>` or special :ref:`NSA VGPR syntax<amdgpu_synid_nsa>`.

*Size:* 1-13 dwords. Actual size depends on syntax, opcode, :ref:`dim<amdgpu_synid_dim>` and :ref:`a16<amdgpu_synid_a16>`.

* If specified using :ref:`NSA VGPR syntax<amdgpu_synid_nsa>`, the size is 1-13 dwords.
* If specified using :ref:`standard VGPR syntax<amdgpu_synid_vcc_lo>`, the size is 1, 2, 3, 4, 8 or 16 dwords. Note that assembler currently supports a limited range of register sequences.


*Operands:* :ref:`v<amdgpu_synid_v>`
