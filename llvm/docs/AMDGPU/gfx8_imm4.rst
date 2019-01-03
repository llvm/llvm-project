..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid8_imm4:

imm4
===========================

A positive :ref:`integer_number<amdgpu_synid_integer_number>`. The value is truncated to 4 bits.

This operand is a mask which controls indexing mode for operands of subsequent instructions. Value 1 enables indexing and value 0 disables it.

    ============ ========================================
    Bit          Meaning
    ============ ========================================
    0            Enables or disables *src0* indexing.
    1            Enables or disables *src1* indexing.
    2            Enables or disables *src2* indexing.
    3            Enables or disables *dst* indexing.
    ============ ========================================

