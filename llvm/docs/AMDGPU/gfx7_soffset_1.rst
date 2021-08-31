..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid_gfx7_soffset_1:

soffset
=======

An unsigned offset added to the base address to get memory address.

* If offset is specified as a register, it supplies an unsigned byte offset but 2 lsb's are ignored.
* If offset is specified as an :ref:`uimm32<amdgpu_synid_uimm32>`, it supplies a 32-bit unsigned byte offset but 2 lsb's are ignored.
* If offset is specified as an :ref:`uimm8<amdgpu_synid_uimm8>`, it supplies an 8-bit unsigned dword offset.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`trap<amdgpu_synid_trap>`, :ref:`uimm8<amdgpu_synid_uimm8>`, :ref:`uimm32<amdgpu_synid_uimm32>`
