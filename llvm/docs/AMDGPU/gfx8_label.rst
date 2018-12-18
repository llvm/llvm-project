..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid8_label:

label
===========================

A branch target which is a 16-bit signed integer treated as a PC-relative dword offset.

This operand may be specified as:

* An :ref:`integer_number<amdgpu_synid_integer_number>`. The number is truncated to 16 bits.
* An :ref:`absolute_expression<amdgpu_synid_absolute_expression>` which must start with an :ref:`integer_number<amdgpu_synid_integer_number>`. The value of the expression is truncated to 16 bits.
* A :ref:`symbol<amdgpu_synid_symbol>` (for example, a label). The value is handled as a 16-bit PC-relative dword offset to be resolved by a linker.

Examples:

.. parsed-literal::

  offset = 30
  s_branch loop_end
  s_branch 2 + offset
  s_branch 32
  loop_end:

