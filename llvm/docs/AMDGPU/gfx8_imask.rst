..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid8_imask:

imask
===========================

This operand is a mask which controls indexing mode for operands of subsequent instructions.
Bits 0, 1 and 2 control indexing of *src0*, *src1* and *src2*, while bit 3 controls indexing of *dst*.
Value 1 enables indexing and value 0 disables it.

    ===== ========================================
    Bit   Meaning
    ===== ========================================
    0     Enables or disables *src0* indexing.
    1     Enables or disables *src1* indexing.
    2     Enables or disables *src2* indexing.
    3     Enables or disables *dst* indexing.
    ===== ========================================

This operand may be specified as one of the following:

* An :ref:`integer_number<amdgpu_synid_integer_number>` or an :ref:`absolute_expression<amdgpu_synid_absolute_expression>`. The value must be in the range 0..15.
* A *gpr_idx* value described below.

    ==================================== ===========================================
    Gpr_idx Value Syntax                 Description
    ==================================== ===========================================
    gpr_idx(*<operands>*)                Enable indexing for specified *operands*
                                         and disable it for the rest.
                                         *Operands* is a comma-separated list of
                                         values which may include:

                                         * "SRC0" - enable *src0* indexing.

                                         * "SRC1" - enable *src1* indexing.

                                         * "SRC2" - enable *src2* indexing.

                                         * "DST"  - enable *dst* indexing.

                                         Each of these values may be specified only
                                         once.

                                         *Operands* list may be empty; this syntax
                                         disables indexing for all operands.
    ==================================== ===========================================

Examples:

.. parsed-literal::

    s_set_gpr_idx_mode 0
    s_set_gpr_idx_mode gpr_idx()                        // the same as above

    s_set_gpr_idx_mode 15
    s_set_gpr_idx_mode gpr_idx(DST,SRC0,SRC1,SRC2)      // the same as above
    s_set_gpr_idx_mode gpr_idx(SRC0,SRC1,SRC2,DST)      // the same as above

    s_set_gpr_idx_mode gpr_idx(DST,SRC1)
