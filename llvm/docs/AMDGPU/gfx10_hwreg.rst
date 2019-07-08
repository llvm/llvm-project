..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid10_hwreg:

hwreg
===========================

Bits of a hardware register being accessed.

The bits of this operand have the following meaning:

    ============ ===================================
    Bits         Description
    ============ ===================================
    5:0          Register *id*.
    10:6         First bit *offset* (0..31).
    15:11        *Size* in bits (1..32).
    ============ ===================================

This operand may be specified as a positive 16-bit :ref:`integer_number<amdgpu_synid_integer_number>` or using the syntax described below.

    ==================================== ============================================================================
    Syntax                               Description
    ==================================== ============================================================================
    hwreg({0..63})                       All bits of a register indicated by its *id*.
    hwreg(<*name*>)                      All bits of a register indicated by its *name*.
    hwreg({0..63}, {0..31}, {1..32})     Register bits indicated by register *id*, first bit *offset* and *size*.
    hwreg(<*name*>, {0..31}, {1..32})    Register bits indicated by register *name*, first bit *offset* and *size*.
    ==================================== ============================================================================

Register *id*, *offset* and *size* must be specified as positive :ref:`integer numbers<amdgpu_synid_integer_number>`.

Defined register *names* include:

    =================== ==========================================
    Name                Description
    =================== ==========================================
    HW_REG_MODE         Shader writeable mode bits.
    HW_REG_STATUS       Shader read-only status.
    HW_REG_TRAPSTS      Trap status.
    HW_REG_HW_ID        Id of wave, simd, compute unit, etc.
    HW_REG_GPR_ALLOC    Per-wave SGPR and VGPR allocation.
    HW_REG_LDS_ALLOC    Per-wave LDS allocation.
    HW_REG_IB_STS       Counters of outstanding instructions.
    HW_REG_SH_MEM_BASES Memory aperture.
    HW_REG_TBA_LO       tba_lo register.
    HW_REG_TBA_HI       tba_hi register.
    HW_REG_TMA_LO       tma_lo register.
    HW_REG_TMA_HI       tma_hi register.
    HW_REG_FLAT_SCR_LO  flat_scratch_lo register.
    HW_REG_FLAT_SCR_HI  flat_scratch_hi register.
    HW_REG_XNACK_MASK   xnack_mask register.
    HW_REG_POPS_PACKER  pops_packer register.
    =================== ==========================================

Examples:

.. parsed-literal::

    s_getreg_b32 s2, 0x6
    s_getreg_b32 s2, hwreg(15)
    s_getreg_b32 s2, hwreg(51, 1, 31)
    s_getreg_b32 s2, hwreg(HW_REG_LDS_ALLOC, 0, 1)

