..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid10_src_exp:

vsrc
===========================

Data to copy to export buffers. This is an optional operand. Must be specified as :ref:`off<amdgpu_synid_off>` if not used.

:ref:`compr<amdgpu_synid_compr>` modifier indicates use of compressed (16-bit) data. This limits number of source operands from 4 to 2:

* src0 and src1 must specify the first register (or :ref:`off<amdgpu_synid_off>`).
* src2 and src3 must specify the second register (or :ref:`off<amdgpu_synid_off>`).

An example:

.. parsed-literal::

  exp mrtz v3, v3, off, off compr

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`off<amdgpu_synid_off>`
