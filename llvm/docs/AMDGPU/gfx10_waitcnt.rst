..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid10_waitcnt:

waitcnt
===========================

Counts of outstanding instructions to wait for.

The bits of this operand have the following meaning:

    ============ ======================================================
    Bits         Description
    ============ ======================================================
    3:0          VM_CNT: vector memory operations count, lower bits.
    6:4          EXP_CNT: export count.
    11:8         LGKM_CNT: LDS, GDS, Constant and Message count.
    15:14        VM_CNT: vector memory operations count, upper bits.
    ============ ======================================================

This operand may be specified as a positive 16-bit :ref:`integer_number<amdgpu_synid_integer_number>`
or as a combination of the following symbolic helpers:

    ====================== ======================================================================
    Syntax                 Description
    ====================== ======================================================================
    vmcnt(<*N*>)           VM_CNT value. *N* must not exceed the largest VM_CNT value.
    expcnt(<*N*>)          EXP_CNT value. *N* must not exceed the largest EXP_CNT value.
    lgkmcnt(<*N*>)         LGKM_CNT value. *N* must not exceed the largest LGKM_CNT value.
    vmcnt_sat(<*N*>)       VM_CNT value computed as min(*N*, the largest VM_CNT value).
    expcnt_sat(<*N*>)      EXP_CNT value computed as min(*N*, the largest EXP_CNT value).
    lgkmcnt_sat(<*N*>)     LGKM_CNT value computed as min(*N*, the largest LGKM_CNT value).
    ====================== ======================================================================

These helpers may be specified in any order. Ampersands and commas may be used as optional separators.

*N* is either an
:ref:`integer number<amdgpu_synid_integer_number>` or an
:ref:`absolute expression<amdgpu_synid_absolute_expression>`.

Examples:

.. parsed-literal::

    s_waitcnt 0
    s_waitcnt vmcnt(1)
    s_waitcnt expcnt(2) lgkmcnt(3)
    s_waitcnt vmcnt(1) expcnt(2) lgkmcnt(3)
    s_waitcnt vmcnt(1), expcnt(2), lgkmcnt(3)
    s_waitcnt vmcnt(1) & lgkmcnt_sat(100) & expcnt(2)

