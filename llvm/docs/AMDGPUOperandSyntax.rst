=================================================
Syntax of AMDGPU Assembler Operands and Modifiers
=================================================

.. contents::
   :local:

Conventions
===========

The following conventions are used in syntax description:

    =================== =============================================================
    Notation            Description
    =================== =============================================================
    {0..N}              Any integer value in the range from 0 to N (inclusive).
                        Unless stated otherwise, this value may be specified as
                        either a literal or an llvm expression.
    <x>                 Syntax and meaning of *<x>* is explained elsewhere.
    =================== =============================================================

.. _amdgpu_syn_operands:

Operands
========

TBD

.. _amdgpu_syn_modifiers:

Modifiers
=========

DS Modifiers
------------

.. _amdgpu_synid_ds_offset8:

ds_offset8
~~~~~~~~~~

Specifies an immediate unsigned 8-bit offset, in bytes. The default value is 0.

Used with DS instructions which have 2 addresses.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    offset:{0..0xFF}                         Specifies a 8-bit offset.
    ======================================== ================================================

.. _amdgpu_synid_ds_offset16:

ds_offset16
~~~~~~~~~~~

Specifies an immediate unsigned 16-bit offset, in bytes. The default value is 0.

Used with DS instructions which have 1 address.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    offset:{0..0xFFFF}                       Specifies a 16-bit offset.
    ======================================== ================================================

.. _amdgpu_synid_sw_offset16:

sw_offset16
~~~~~~~~~~~

This is a special modifier which may be used with *ds_swizzle_b32* instruction only.
Specifies a sizzle pattern in numeric or symbolic form. The default value is 0.

See AMD documentation for more information.

    ======================================================= ===================================================
    Syntax                                                  Description
    ======================================================= ===================================================
    offset:{0..0xFFFF}                                      Specifies a 16-bit swizzle pattern
                                                            in a numeric form.
    offset:swizzle(QUAD_PERM,{0..3},{0..3},{0..3},{0..3})   Specifies a quad permute mode pattern; each
                                                            number is a lane id.
    offset:swizzle(BITMASK_PERM, "<mask>")                  Specifies a bitmask permute mode pattern
                                                            which converts a 5-bit lane id to another
                                                            lane id with which the lane interacts.

                                                            <mask> is a 5 character sequence which
                                                            specifies how to transform the bits of the
                                                            lane id. The following characters are allowed:

                                                              * "0" - set bit to 0.

                                                              * "1" - set bit to 1.

                                                              * "p" - preserve bit.

                                                              * "i" - inverse bit.

    offset:swizzle(BROADCAST,{2..32},{0..N})                Specifies a broadcast mode.
                                                            Broadcasts the value of any particular lane to
                                                            all lanes in its group.

                                                            The first numeric parameter is a group
                                                            size and must be equal to 2, 4, 8, 16 or 32.

                                                            The second numeric parameter is an index of the
                                                            lane being broadcasted. The index must not exceed
                                                            group size.
    offset:swizzle(SWAP,{1..16})                            Specifies a swap mode.
                                                            Swaps the neighboring groups of
                                                            1, 2, 4, 8 or 16 lanes.
    offset:swizzle(REVERSE,{2..32})                         Specifies a reverse mode. Reverses
                                                            the lanes for groups of 2, 4, 8, 16 or 32 lanes.
    ======================================================= ===================================================

.. _amdgpu_synid_gds:

gds
~~~

Specifies whether to use GDS or LDS memory (LDS is the default).

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    gds                                      Use GDS memory.
    ======================================== ================================================


EXP Modifiers
-------------

.. _amdgpu_synid_done:

done
~~~~

Specifies if this is the last export from the shader to the target. By default, current
instruction does not finish an export sequence.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    done                                     Indicates the last export operation.
    ======================================== ================================================

.. _amdgpu_synid_compr:

compr
~~~~~

Indicates if the data are compressed (not compressed by default).

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    compr                                    Data are compressed.
    ======================================== ================================================

.. _amdgpu_synid_vm:

vm
~~

Specifies valid mask flag state (off by default).

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    vm                                       Set valid mask flag.
    ======================================== ================================================

FLAT Modifiers
--------------

.. _amdgpu_synid_flat_offset12:

flat_offset12
~~~~~~~~~~~~~

Specifies an immediate unsigned 12-bit offset, in bytes. The default value is 0.

Cannot be used with *global/scratch* opcodes. GFX9 only.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    offset:{0..4095}                         Specifies a 12-bit unsigned offset.
    ======================================== ================================================

.. _amdgpu_synid_flat_offset13:

flat_offset13
~~~~~~~~~~~~~

Specifies an immediate signed 13-bit offset, in bytes. The default value is 0.

Can be used with *global/scratch* opcodes only. GFX9 only.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    offset:{-4096..+4095}                    Specifies a 13-bit signed offset.
    ======================================== ================================================

glc
~~~

See a description :ref:`here<amdgpu_synid_glc>`.

slc
~~~

See a description :ref:`here<amdgpu_synid_slc>`.

tfe
~~~

See a description :ref:`here<amdgpu_synid_tfe>`.

nv
~~

See a description :ref:`here<amdgpu_synid_nv>`.

MIMG Modifiers
--------------

.. _amdgpu_synid_dmask:

dmask
~~~~~

Specifies which channels (image components) are used by the operation. By default, no channels
are used.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    dmask:{0..15}                            Each bit corresponds to one of 4 image
                                             components (RGBA). If the specified bit value
                                             is 0, the component is not used, value 1 means
                                             that the component is used.
    ======================================== ================================================

This modifier has some limitations depending on instruction kind:

    ======================================== ================================================
    Instruction Kind                         Valid dmask Values
    ======================================== ================================================
    32-bit atomic cmpswap                    0x3
    other 32-bit atomic instructions         0x1
    64-bit atomic cmpswap                    0xF
    other 64-bit atomic instructions         0x3
    GATHER4                                  0x1, 0x2, 0x4, 0x8
    Other instructions                       any value
    ======================================== ================================================

.. _amdgpu_synid_unorm:

unorm
~~~~~

Specifies whether address is normalized or not (normalized by default).

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    unorm                                    Force address to be un-normalized.
    ======================================== ================================================

glc
~~~

See a description :ref:`here<amdgpu_synid_glc>`.

slc
~~~

See a description :ref:`here<amdgpu_synid_slc>`.

.. _amdgpu_synid_r128:

r128
~~~~

Specifies texture resource size. The default size is 256 bits.

GFX7 and GFX8 only.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    r128                                     Specifies 128 bits texture resource size.
    ======================================== ================================================

tfe
~~~

See a description :ref:`here<amdgpu_synid_tfe>`.

.. _amdgpu_synid_lwe:

lwe
~~~

Specifies LOD warning status (LOD warning is disabled by default).

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    lwe                                      Enables LOD warning.
    ======================================== ================================================

.. _amdgpu_synid_da:

da
~~

Specifies if an array index must be sent to TA. By default, array index is not sent.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    da                                       Send an array-index to TA.
    ======================================== ================================================

.. _amdgpu_synid_d16:

d16
~~~

Specifies data size: 16 or 32 bits (32 bits by default). Not supported by GFX7.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    d16                                      Enables 16-bits data mode.

                                             On loads, convert data in memory to 16-bit
                                             format before storing it in VGPRs.

                                             For stores, convert 16-bit data in VGPRs to
                                             32 bits before going to memory.

                                             Note that 16-bit data are stored in VGPRs
                                             unpacked in GFX8.0. In GFX8.1 and GFX9 16-bit
                                             data are packed.
    ======================================== ================================================

.. _amdgpu_synid_a16:

a16
~~~

Specifies size of image address components: 16 or 32 bits (32 bits by default). GFX9 only.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    a16                                      Enables 16-bits image address components.
    ======================================== ================================================

Miscellaneous Modifiers
-----------------------

.. _amdgpu_synid_glc:

glc
~~~

This modifier has different meaning for loads, stores, and atomic operations.
The default value is off (0).

See AMD documentation for details.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    glc                                      Set glc bit to 1.
    ======================================== ================================================

.. _amdgpu_synid_slc:

slc
~~~

Specifies cache policy. The default value is off (0).

See AMD documentation for details.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    slc                                      Set slc bit to 1.
    ======================================== ================================================

.. _amdgpu_synid_tfe:

tfe
~~~

Controls access to partially resident textures. The default value is off (0).

See AMD documentation for details.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    tfe                                      Set tfe bit to 1.
    ======================================== ================================================

.. _amdgpu_synid_nv:

nv
~~

Specifies if instruction is operating on non-volatile memory. By default, memory is volatile.

GFX9 only.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    nv                                       Indicates that instruction operates on
                                             non-volatile memory.
    ======================================== ================================================

MUBUF/MTBUF Modifiers
---------------------

.. _amdgpu_synid_idxen:

idxen
~~~~~

Specifies whether address components include an index. By default, no components are used.

Can be used together with :ref:`offen<amdgpu_synid_offen>`.

Cannot be used with :ref:`addr64<amdgpu_synid_addr64>`.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    idxen                                    Address components include an index.
    ======================================== ================================================

.. _amdgpu_synid_offen:

offen
~~~~~

Specifies whether address components include an offset. By default, no components are used.

Can be used together with :ref:`idxen<amdgpu_synid_idxen>`.

Cannot be used with :ref:`addr64<amdgpu_synid_addr64>`.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    offen                                    Address components include an offset.
    ======================================== ================================================

.. _amdgpu_synid_addr64:

addr64
~~~~~~

Specifies whether a 64-bit address is used. By default, no address is used.

GFX7 only. Cannot be used with :ref:`offen<amdgpu_synid_offen>` and
:ref:`idxen<amdgpu_synid_idxen>` modifiers.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    addr64                                   A 64-bit address is used.
    ======================================== ================================================

.. _amdgpu_synid_buf_offset12:

buf_offset12
~~~~~~~~~~~~

Specifies an immediate unsigned 12-bit offset, in bytes. The default value is 0.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    offset:{0..0xFFF}                        Specifies a 12-bit unsigned offset.
    ======================================== ================================================

glc
~~~

See a description :ref:`here<amdgpu_synid_glc>`.

slc
~~~

See a description :ref:`here<amdgpu_synid_slc>`.

.. _amdgpu_synid_lds:

lds
~~~

Specifies where to store the result: VGPRs or LDS (VGPRs by default).

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    lds                                      Store result in LDS.
    ======================================== ================================================

tfe
~~~

See a description :ref:`here<amdgpu_synid_tfe>`.

.. _amdgpu_synid_dfmt:

dfmt
~~~~

TBD

.. _amdgpu_synid_nfmt:

nfmt
~~~~

TBD

SMRD/SMEM Modifiers
-------------------

glc
~~~

See a description :ref:`here<amdgpu_synid_glc>`.

nv
~~

See a description :ref:`here<amdgpu_synid_nv>`.

VINTRP Modifiers
----------------

.. _amdgpu_synid_high:

high
~~~~

Specifies which half of the LDS word to use. Low half of LDS word is used by default.
GFX9 only.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    high                                     Use high half of LDS word.
    ======================================== ================================================

VOP1/VOP2 DPP Modifiers
-----------------------

GFX8 and GFX9 only.

.. _amdgpu_synid_dpp_ctrl:

dpp_ctrl
~~~~~~~~

Specifies how data are shared between threads. This is a mandatory modifier.
There is no default value.

Note. The lanes of a wavefront are organized in four banks and four rows.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    quad_perm:[{0..3},{0..3},{0..3},{0..3}]  Full permute of 4 threads.
    row_mirror                               Mirror threads within row.
    row_half_mirror                          Mirror threads within 1/2 row (8 threads).
    row_bcast:15                             Broadcast 15th thread of each row to next row.
    row_bcast:31                             Broadcast thread 31 to rows 2 and 3.
    wave_shl:1                               Wavefront left shift by 1 thread.
    wave_rol:1                               Wavefront left rotate by 1 thread.
    wave_shr:1                               Wavefront right shift by 1 thread.
    wave_ror:1                               Wavefront right rotate by 1 thread.
    row_shl:{1..15}                          Row shift left by 1-15 threads.
    row_shr:{1..15}                          Row shift right by 1-15 threads.
    row_ror:{1..15}                          Row rotate right by 1-15 threads.
    ======================================== ================================================

.. _amdgpu_synid_row_mask:

row_mask
~~~~~~~~

Controls which rows are enabled for data sharing. By default, all rows are enabled.

Note. The lanes of a wavefront are organized in four banks and four rows.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    row_mask:{0..15}                         Each of 4 bits in the mask controls one
                                             row (0 - disabled, 1 - enabled).
    ======================================== ================================================

.. _amdgpu_synid_bank_mask:

bank_mask
~~~~~~~~~

Controls which banks are enabled for data sharing. By default, all banks are enabled.

Note. The lanes of a wavefront are organized in four banks and four rows.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    bank_mask:{0..15}                        Each of 4 bits in the mask controls one
                                             bank (0 - disabled, 1 - enabled).
    ======================================== ================================================

.. _amdgpu_synid_bound_ctrl:

bound_ctrl
~~~~~~~~~~

Controls data sharing when accessing an invalid lane. By default, data sharing with
invalid lanes is disabled.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    bound_ctrl:0                             Enables data sharing with invalid lanes.
                                             Accessing data from an invalid lane will
                                             return zero.
    ======================================== ================================================

VOP1/VOP2/VOPC SDWA Modifiers
-----------------------------

GFX8 and GFX9 only.

clamp
~~~~~

See a description :ref:`here<amdgpu_synid_clamp>`.

omod
~~~~

See a description :ref:`here<amdgpu_synid_omod>`.

GFX9 only.

.. _amdgpu_synid_dst_sel:

dst_sel
~~~~~~~

Selects which bits in the destination are affected. By default, all bits are affected.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    dst_sel:DWORD                            Use bits 31:0.
    dst_sel:BYTE_0                           Use bits 7:0.
    dst_sel:BYTE_1                           Use bits 15:8.
    dst_sel:BYTE_2                           Use bits 23:16.
    dst_sel:BYTE_3                           Use bits 31:24.
    dst_sel:WORD_0                           Use bits 15:0.
    dst_sel:WORD_1                           Use bits 31:16.
    ======================================== ================================================


.. _amdgpu_synid_dst_unused:

dst_unused
~~~~~~~~~~

Controls what to do with the bits in the destination which are not selected
by :ref:`dst_sel<amdgpu_synid_dst_sel>`.
By default, unused bits are preserved.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    dst_unused:UNUSED_PAD                    Pad with zeros.
    dst_unused:UNUSED_SEXT                   Sign-extend upper bits, zero lower bits.
    dst_unused:UNUSED_PRESERVE               Preserve bits.
    ======================================== ================================================

.. _amdgpu_synid_src0_sel:

src0_sel
~~~~~~~~

Controls which bits in the src0 are used. By default, all bits are used.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    src0_sel:DWORD                           Use bits 31:0.
    src0_sel:BYTE_0                          Use bits 7:0.
    src0_sel:BYTE_1                          Use bits 15:8.
    src0_sel:BYTE_2                          Use bits 23:16.
    src0_sel:BYTE_3                          Use bits 31:24.
    src0_sel:WORD_0                          Use bits 15:0.
    src0_sel:WORD_1                          Use bits 31:16.
    ======================================== ================================================

.. _amdgpu_synid_src1_sel:

src1_sel
~~~~~~~~

Controls which bits in the src1 are used. By default, all bits are used.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    src1_sel:DWORD                           Use bits 31:0.
    src1_sel:BYTE_0                          Use bits 7:0.
    src1_sel:BYTE_1                          Use bits 15:8.
    src1_sel:BYTE_2                          Use bits 23:16.
    src1_sel:BYTE_3                          Use bits 31:24.
    src1_sel:WORD_0                          Use bits 15:0.
    src1_sel:WORD_1                          Use bits 31:16.
    ======================================== ================================================

VOP1/VOP2/VOPC SDWA Operand Modifiers
-------------------------------------

Operand modifiers are not used separately. They are applied to source operands.

GFX8 and GFX9 only.

abs
~~~

See a description :ref:`here<amdgpu_synid_abs>`.

neg
~~~

See a description :ref:`here<amdgpu_synid_neg>`.

.. _amdgpu_synid_sext:

sext
~~~~

Sign-extends value of a (sub-dword) operand to fill all 32 bits.
Has no effect for 32-bit operands.

Valid for integer operands only.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    sext(<operand>)                          Sign-extend operand value.
    ======================================== ================================================

VOP3 Modifiers
--------------

.. _amdgpu_synid_vop3_op_sel:

vop3_op_sel
~~~~~~~~~~~

Selects the low [15:0] or high [31:16] operand bits for source and destination operands.
By default, low bits are used for all operands.

The number of values specified with the op_sel modifier must match the number of instruction
operands (both source and destination). First value controls src0, second value controls src1
and so on, except that the last value controls destination.
The value 0 selects the low bits, while 1 selects the high bits.

Note. op_sel modifier affects 16-bit operands only. For 32-bit operands the value specified
by op_sel must be 0.

GFX9 only.

    ======================================== ============================================================
    Syntax                                   Description
    ======================================== ============================================================
    op_sel:[{0..1},{0..1}]                   Select operand bits for instructions with 1 source operand.
    op_sel:[{0..1},{0..1},{0..1}]            Select operand bits for instructions with 2 source operands.
    op_sel:[{0..1},{0..1},{0..1},{0..1}]     Select operand bits for instructions with 3 source operands.
    ======================================== ============================================================

.. _amdgpu_synid_clamp:

clamp
~~~~~

Clamp meaning depends on instruction.

For *v_cmp* instructions, clamp modifier indicates that the compare signals
if a floating point exception occurs. By default, signaling is disabled.
Not supported by GFX7.

For integer operations, clamp modifier indicates that the result must be clamped
to the largest and smallest representable value. By default, there is no clamping.
Integer clamping is not supported by GFX7.

For floating point operations, clamp modifier indicates that the result must be clamped
to the range [0.0, 1.0]. By default, there is no clamping.

Note. Clamp modifier is applied after :ref:`output modifiers<amdgpu_synid_omod>` (if any).

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    clamp                                    Enables clamping (or signaling).
    ======================================== ================================================

.. _amdgpu_synid_omod:

omod
~~~~

Specifies if an output modifier must be applied to the result.
By default, no output modifiers are applied.

Note. Output modifiers are applied before :ref:`clamping<amdgpu_synid_clamp>` (if any).

Output modifiers are valid for f32 and f64 floating point results only.
They must not be used with f16.

Note. *v_cvt_f16_f32* is an exception. This instruction produces f16 result
but accepts output modifiers.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    mul:2                                    Multiply the result by 2.
    mul:4                                    Multiply the result by 4.
    div:2                                    Multiply the result by 0.5.
    ======================================== ================================================

VOP3 Operand Modifiers
----------------------

Operand modifiers are not used separately. They are applied to source operands.

.. _amdgpu_synid_abs:

abs
~~~

Computes absolute value of its operand. Applied before :ref:`neg<amdgpu_synid_neg>` (if any).
Valid for floating point operands only.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    abs(<operand>)                           Get absolute value of operand.
    \|<operand>|                             The same as above.
    ======================================== ================================================

.. _amdgpu_synid_neg:

neg
~~~

Computes negative value of its operand. Applied after :ref:`abs<amdgpu_synid_abs>` (if any).
Valid for floating point operands only.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    neg(<operand>)                           Get negative value of operand.
    -<operand>                               The same as above.
    ======================================== ================================================

VOP3P Modifiers
---------------

This section describes modifiers of regular VOP3P instructions.
*v_mad_mix* modifiers are described :ref:`in a separate section<amdgpu_synid_mad_mix>`.

GFX9 only.

.. _amdgpu_synid_op_sel:

op_sel
~~~~~~

Selects the low [15:0] or high [31:16] operand bits as input to the operation
which results in the lower-half of the destination.
By default, low bits are used for all operands.

The number of values specified with the op_sel modifier must match the number of source
operands. First value controls src0, second value controls src1 and so on.
The value 0 selects the low bits, while 1 selects the high bits.

    ======================================== =============================================================
    Syntax                                   Description
    ======================================== =============================================================
    op_sel:[{0..1}]                          Select operand bits for instructions with 1 source operand.
    op_sel:[{0..1},{0..1}]                   Select operand bits for instructions with 2 source operands.
    op_sel:[{0..1},{0..1},{0..1}]            Select operand bits for instructions with 3 source operands.
    ======================================== =============================================================

.. _amdgpu_synid_op_sel_hi:

op_sel_hi
~~~~~~~~~

Selects the low [15:0] or high [31:16] operand bits as input to the operation
which results in the upper-half of the destination.
By default, high bits are used for all operands.

The number of values specified with the op_sel_hi modifier must match the number of source
operands. First value controls src0, second value controls src1 and so on.
The value 0 selects the low bits, while 1 selects the high bits.

    ======================================== =============================================================
    Syntax                                   Description
    ======================================== =============================================================
    op_sel_hi:[{0..1}]                       Select operand bits for instructions with 1 source operand.
    op_sel_hi:[{0..1},{0..1}]                Select operand bits for instructions with 2 source operands.
    op_sel_hi:[{0..1},{0..1},{0..1}]         Select operand bits for instructions with 3 source operands.
    ======================================== =============================================================

.. _amdgpu_synid_neg_lo:

neg_lo
~~~~~~

Specifies whether to change sign of operand values selected by
:ref:`op_sel<amdgpu_synid_op_sel>`. These values are then used
as input to the operation which results in the upper-half of the destination.

The number of values specified with this modifier must match the number of source
operands. First value controls src0, second value controls src1 and so on.

The value 0 indicates that the corresponding operand value is used unmodified,
the value 1 indicates that negative value of the operand must be used.

By default, operand values are used unmodified.

This modifier is valid for floating point operands only.

    ======================================== ==================================================================
    Syntax                                   Description
    ======================================== ==================================================================
    neg_lo:[{0..1}]                          Select affected operands for instructions with 1 source operand.
    neg_lo:[{0..1},{0..1}]                   Select affected operands for instructions with 2 source operands.
    neg_lo:[{0..1},{0..1},{0..1}]            Select affected operands for instructions with 3 source operands.
    ======================================== ==================================================================

.. _amdgpu_synid_neg_hi:

neg_hi
~~~~~~

Specifies whether to change sign of operand values selected by
:ref:`op_sel_hi<amdgpu_synid_op_sel_hi>`. These values are then used
as input to the operation which results in the upper-half of the destination.

The number of values specified with this modifier must match the number of source
operands. First value controls src0, second value controls src1 and so on.

The value 0 indicates that the corresponding operand value is used unmodified,
the value 1 indicates that negative value of the operand must be used.

By default, operand values are used unmodified.

This modifier is valid for floating point operands only.

    ======================================== ==================================================================
    Syntax                                   Description
    ======================================== ==================================================================
    neg_hi:[{0..1}]                          Select affected operands for instructions with 1 source operand.
    neg_hi:[{0..1},{0..1}]                   Select affected operands for instructions with 2 source operands.
    neg_hi:[{0..1},{0..1},{0..1}]            Select affected operands for instructions with 3 source operands.
    ======================================== ==================================================================

clamp
~~~~~

See a description :ref:`here<amdgpu_synid_clamp>`.

.. _amdgpu_synid_mad_mix:

VOP3P V_MAD_MIX Modifiers
-------------------------

These instructions use VOP3P format but have different modifiers.

GFX9 only.

.. _amdgpu_synid_mad_op_sel:

mad_op_sel
~~~~~~~~~~

Selects the size of source operands: either 32 bits or 16 bits.
By default, 32 bits are used for all source operands.

The value 0 indicates 32 bits, the value 1 indicates 16 bits.
The location of 16 bits in the operand may be specified by
:ref:`mad_op_sel_hi<amdgpu_synid_mad_op_sel_hi>`.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    op_sel:[{0..1},{0..1},{0..1}]            Select size of each source operand.
    ======================================== ================================================

.. _amdgpu_synid_mad_op_sel_hi:

mad_op_sel_hi
~~~~~~~~~~~~~

This operand has meaning only for 16-bit source operands as indicated by
:ref:`mad_op_sel<amdgpu_synid_mad_op_sel>`.
It specifies to select either the low [15:0] or high [31:16] operand bits
as input to the operation.

The value 0 indicates the low bits, the value 1 indicates the high 16 bits.
By default, low bits are used for all operands.

    ======================================== ================================================
    Syntax                                   Description
    ======================================== ================================================
    op_sel_hi:[{0..1},{0..1},{0..1}]         Select location of each 16-bit source operand.
    ======================================== ================================================

abs
~~~

See a description :ref:`here<amdgpu_synid_abs>`.

neg
~~~

See a description :ref:`here<amdgpu_synid_neg>`.

clamp
~~~~~

See a description :ref:`here<amdgpu_synid_clamp>`.
