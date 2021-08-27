..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid_gfx10_vdst_13:

vdst
====

Instruction output: data read from a memory buffer.

If :ref:`lds<amdgpu_synid_lds>` is specified, this operand is ignored by H/W and data are stored directly into LDS.

*Size:* 1 dword by default. :ref:`tfe<amdgpu_synid_tfe>` adds 1 dword if specified.

    Note that :ref:`tfe<amdgpu_synid_tfe>` and :ref:`lds<amdgpu_synid_lds>` cannot be used together.

*Operands:* :ref:`v<amdgpu_synid_v>`
