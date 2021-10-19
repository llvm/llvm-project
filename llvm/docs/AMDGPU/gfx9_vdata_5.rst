..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid_gfx9_vdata_5:

vdata
=====

Input data for an atomic instruction.

Optionally may serve as an output data:

* If :ref:`glc<amdgpu_synid_glc>` is specified, gets the memory value before the operation.

*Size:* depends on :ref:`dmask<amdgpu_synid_dmask>` and :ref:`tfe<amdgpu_synid_tfe>`:

* :ref:`dmask<amdgpu_synid_dmask>` may specify 2 data elements for 32-bit-per-pixel surfaces or 4 data elements for 64-bit-per-pixel surfaces. Each data element occupies 1 dword.
* :ref:`tfe<amdgpu_synid_tfe>` adds 1 dword if specified.

  Note: the surface data format is indicated in the image resource constant but not in the instruction.

*Operands:* :ref:`v<amdgpu_synid_v>`
