..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid_gfx12_addr:

addr
----

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_attr:

attr
----

Interpolation attribute and channel:

    ============== ===================================
    Syntax         Description
    ============== ===================================
    attr{0..32}.x  Attribute 0..32 with *x* channel.
    attr{0..32}.y  Attribute 0..32 with *y* channel.
    attr{0..32}.z  Attribute 0..32 with *z* channel.
    attr{0..32}.w  Attribute 0..32 with *w* channel.
    ============== ===================================

Examples:

.. parsed-literal::

    ds_param_load v5, attr0.z

.. _amdgpu_synid_gfx12_data0_6802ce:

data0
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_data0_fd235e:

data0
-----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_data0_56f215:

data0
-----

Instruction input.

*Size:* 3 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_data0_e016a1:

data0
-----

Instruction input.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_data1_6802ce:

data1
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_data1_fd235e:

data1
-----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_data1_e016a1:

data1
-----

Instruction input.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_data1_731030:

data1
-----

Instruction input.

*Size:* 8 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_ioffset:

ioffset
-------

*Size:* 1 dword.

*Operands:* 

.. _amdgpu_synid_gfx12_literal_81e671:

literal
-------

*Size:* 1 dword.

*Operands:* 

.. _amdgpu_synid_gfx12_literal_ad4155:

literal
-------

*Size:* 1 dword.

*Operands:* :ref:`simm16<amdgpu_synid_simm16>`

.. _amdgpu_synid_gfx12_literal_6f0844:

literal
-------

A :ref:`floating-point_number<amdgpu_synid_floating-point_number>`, an :ref:`integer_number<amdgpu_synid_integer_number>`, or an :ref:`absolute_expression<amdgpu_synid_absolute_expression>`.
The value is converted to *f32* as described :ref:`here<amdgpu_synid_conv>`.

.. _amdgpu_synid_gfx12_literal_a3e80c:

literal
-------

An :ref:`integer_number<amdgpu_synid_integer_number>` or an :ref:`absolute_expression<amdgpu_synid_absolute_expression>`. The value is truncated to 32 bits.

.. _amdgpu_synid_gfx12_rsrc_5fe6d8:

rsrc
----

Instruction input.

*Size:* 4 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_rsrc_c9f929:

rsrc
----

Instruction input.

*Size:* 8 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_saddr_8ad588:

saddr
-----

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_saddr_6c410e:

saddr
-----

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_samp:

samp
----

*Size:* 4 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_sbase_453b95:

sbase
-----

A 128-bit buffer resource constant for scalar memory operations which provides a base address, a size and a stride.

*Size:* 4 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_sbase_30e75b:

sbase
-----

A 64-bit base address for scalar memory operations.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_sdata_5c7b50:

sdata
-----

Instruction input.

*Size:* 1 dword.

*Operands:* 

.. _amdgpu_synid_gfx12_sdata_54e16e:

sdata
-----

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_sdata_d725ab:

sdata
-----

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`simm8<amdgpu_synid_simm8>`

.. _amdgpu_synid_gfx12_sdata_6c003b:

sdata
-----

Instruction output.

*Size:* 16 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_sdata_386c33:

sdata
-----

Instruction output.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_sdata_dd9dd8:

sdata
-----

Instruction output.

*Size:* 3 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_sdata_4585b8:

sdata
-----

Instruction output.

*Size:* 4 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_sdata_0974a4:

sdata
-----

Instruction output.

*Size:* 8 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_sdst_acfb90:

sdst
----

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`

.. _amdgpu_synid_gfx12_sdst_3cd7ad:

sdst
----

Instruction output.

*Size:* 1 dword if wavefront size is 32, otherwise 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_sdst_54e16e:

sdst
----

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_sdst_8078f5:

sdst
----

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`

.. _amdgpu_synid_gfx12_sdst_386c33:

sdst
----

Instruction output.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_sdst_ea3f10:

sdst
----

Instruction output.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`exec<amdgpu_synid_exec>`

.. _amdgpu_synid_gfx12_sdst_006c40:

sdst
----

Instruction output.

*Size:* 2 dwords.

*Operands:* :ref:`vcc<amdgpu_synid_vcc>`

.. _amdgpu_synid_gfx12_simm16_81e671:

simm16
------

*Size:* 1 dword.

*Operands:* 

.. _amdgpu_synid_gfx12_simm16_ad4155:

simm16
------

*Size:* 1 dword.

*Operands:* :ref:`simm16<amdgpu_synid_simm16>`

.. _amdgpu_synid_gfx12_simm16_8ccd1e:

simm16
------

A 16-bit message code. The bits of this operand have the following meaning:

    ============ =============================== ===============
    Bits         Description                     Value Range
    ============ =============================== ===============
    3:0          Message *type*.                 0..15
    6:4          Optional *operation*.           0..7
    7:7          Must be 0.                      0
    9:8          Optional *stream*.              0..3
    15:10        Unused.                         \-
    ============ =============================== ===============

This operand may be specified as one of the following:

* An :ref:`integer_number<amdgpu_synid_integer_number>` or an :ref:`absolute_expression<amdgpu_synid_absolute_expression>`. The value must be in the range 0..0xFFFF.
* A *sendmsg* value described below.

    ==================================== ====================================================
    Sendmsg Value Syntax                 Description
    ==================================== ====================================================
    sendmsg(<*type*>)                    A message identified by its *type*.
    sendmsg(<*type*>,<*op*>)             A message identified by its *type* and *operation*.
    sendmsg(<*type*>,<*op*>,<*stream*>)  A message identified by its *type* and *operation*
                                         with a stream *id*.
    ==================================== ====================================================

*Type* may be specified using message *name* or message *id*.

*Op* may be specified using operation *name* or operation *id*.

Stream *id* is an integer in the range 0..3.

Numeric values may be specified as positive :ref:`integer numbers<amdgpu_synid_integer_number>`
or :ref:`absolute expressions<amdgpu_synid_absolute_expression>`.

Each message type supports specific operations:

    ====================== ========== ============================== ============ ==========
    Message name           Message Id Supported Operations           Operation Id Stream Id
    ====================== ========== ============================== ============ ==========
    MSG_INTERRUPT          1          \-                             \-           \-
    MSG_HS_TESSFACTOR      2          \-                             \-           \-
    MSG_DEALLOC_VGPRS      3          \-                             \-           \-
    MSG_STALL_WAVE_GEN     5          \-                             \-           \-
    MSG_HALT_WAVES         6          \-                             \-           \-
    MSG_GS_ALLOC_REQ       9          \-                             \-           \-
    MSG_SYSMSG             15         SYSMSG_OP_ECC_ERR_INTERRUPT    1            \-
    \                                 SYSMSG_OP_REG_RD               2            \-
    \                                 SYSMSG_OP_TTRACE_PC            4            \-
    ====================== ========== ============================== ============ ==========

*Sendmsg* arguments are validated depending on how *type* value is specified:

* If message *type* is specified by name, arguments values must satisfy limitations detailed in the table above.
* If message *type* is specified as a number, each argument must not exceed corresponding value range (see the first table).

Examples:

.. parsed-literal::

    // numeric message code
    msg = 0x10
    s_sendmsg 0x12
    s_sendmsg msg + 2

    // sendmsg with strict arguments validation
    s_sendmsg sendmsg(MSG_INTERRUPT)
    s_sendmsg sendmsg(MSG_SYSMSG, SYSMSG_OP_TTRACE_PC)

    // sendmsg with validation of value range only
    msg = 2
    op = 3
    s_sendmsg sendmsg(msg, op)

.. _amdgpu_synid_gfx12_simm16_dff4f4:

simm16
------

A branch target which is a 16-bit signed integer treated as a PC-relative dword offset.

This operand may be specified as one of the following:

* An :ref:`integer_number<amdgpu_synid_integer_number>` or an :ref:`absolute_expression<amdgpu_synid_absolute_expression>`. The value must be in the range -32768..32767.
* A :ref:`symbol<amdgpu_synid_symbol>` (for example, a label) representing a relocatable address in the same compilation unit where it is referred from. The value is handled as a 16-bit PC-relative dword offset to be resolved by a linker.

Examples:

.. parsed-literal::

  offset = 30
  label_1:
  label_2 = . + 4

  s_branch 32
  s_branch offset + 2
  s_branch label_1
  s_branch label_2
  s_branch label_3
  s_branch label_4

  label_3 = label_2 + 4
  label_4:

.. _amdgpu_synid_gfx12_simm16_faa0f8:

simm16
------

A delay between dependent SALU/VALU instructions.
This operand may specify a delay for 2 instructions:
the one after the current *s_delay_alu* instruction
and for the second instruction indicated by *SKIP*.

The bits of this operand have the following meaning:

    ===== ========================================================== ============
    Bits  Description                                                Value Range
    ===== ========================================================== ============
    3:0   ID0: indicates a delay for the first instruction.          0..11
    6:4   SKIP: indicates the position of the second instruction.    0..5
    10:7  ID1: indicates a delay for the second instruction.         0..11
    ===== ========================================================== ============

This operand may be specified as one of the following:

* An :ref:`integer_number<amdgpu_synid_integer_number>` or an :ref:`absolute_expression<amdgpu_synid_absolute_expression>`. The value must be in the range 0..0xFFFF.
* A combination of *instid0*, *instskip*, *instid1* values described below.

    ======================== =========================== ===============
    Syntax                   Description                 Default Value
    ======================== =========================== ===============
    instid0(<*ID name*>)     A symbolic *ID0* value.     instid0(NO_DEP)
    instskip(<*SKIP name*>)  A symbolic *SKIP* value.    instskip(SAME)
    instid1(<*ID name*>)     A symbolic *ID1* value.     instid1(NO_DEP)
    ======================== =========================== ===============

These values may be specified in any order.
When more than one value is specified, the values must be separated from each other by a '|'.

Valid *ID names* are defined below.

    =================== ===================================================================
    Name                Description
    =================== ===================================================================
    NO_DEP              No dependency on any prior instruction. This is the default value.
    VALU_DEP_1          Dependency on a previous VALU instruction, 1 opcode back.
    VALU_DEP_2          Dependency on a previous VALU instruction, 2 opcodes back.
    VALU_DEP_3          Dependency on a previous VALU instruction, 3 opcodes back.
    VALU_DEP_4          Dependency on a previous VALU instruction, 4 opcodes back.
    TRANS32_DEP_1       Dependency on a previous TRANS32 instruction, 1 opcode back.
    TRANS32_DEP_2       Dependency on a previous TRANS32 instruction, 2 opcodes back.
    TRANS32_DEP_3       Dependency on a previous TRANS32 instruction, 3 opcodes back.
    FMA_ACCUM_CYCLE_1   Single cycle penalty for FMA accumulation.
    SALU_CYCLE_1        1 cycle penalty for a prior SALU instruction.
    SALU_CYCLE_2        2 cycle penalty for a prior SALU instruction.
    SALU_CYCLE_3        3 cycle penalty for a prior SALU instruction.
    =================== ===================================================================

Legal *SKIP names* are described in the following table.

    ======== ============================================================================
    Name     Description
    ======== ============================================================================
    SAME     Apply second dependency to the same instruction. This is the default value.
    NEXT     Apply second dependency to the next instruction.
    SKIP_1   Skip 1 instruction then apply dependency.
    SKIP_2   Skip 2 instructions then apply dependency.
    SKIP_3   Skip 3 instructions then apply dependency.
    SKIP_4   Skip 4 instructions then apply dependency.
    ======== ============================================================================

Examples:

.. parsed-literal::

    s_delay_alu instid0(VALU_DEP_1)
    s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)

.. _amdgpu_synid_gfx12_simm16_165f69:

simm16
------

Bits of a hardware register being accessed.

The bits of this operand have the following meaning:

    ======= ===================== ============
    Bits    Description           Value Range
    ======= ===================== ============
    5:0     Register *id*.        0..63
    10:6    First bit *offset*.   0..31
    15:11   *Size* in bits.       1..32
    ======= ===================== ============

This operand may be specified as one of the following:

* An :ref:`integer_number<amdgpu_synid_integer_number>` or an :ref:`absolute_expression<amdgpu_synid_absolute_expression>`. The value must be in the range 0..0xFFFF.
* An *hwreg* value described below.

    ==================================== ============================================================================
    Hwreg Value Syntax                   Description
    ==================================== ============================================================================
    hwreg({0..63})                       All bits of a register indicated by its *id*.
    hwreg(<*name*>)                      All bits of a register indicated by its *name*.
    hwreg({0..63}, {0..31}, {1..32})     Register bits indicated by register *id*, first bit *offset* and *size*.
    hwreg(<*name*>, {0..31}, {1..32})    Register bits indicated by register *name*, first bit *offset* and *size*.
    ==================================== ============================================================================

Numeric values may be specified as positive :ref:`integer numbers<amdgpu_synid_integer_number>`
or :ref:`absolute expressions<amdgpu_synid_absolute_expression>`.

Defined register *names* include:

    =================== ==========================================
    Name                Description
    =================== ==========================================
    HW_REG_MODE         Shader writeable mode bits.
    HW_REG_STATUS       Shader read-only status.
    HW_REG_TRAPSTS      Trap status.
    HW_REG_HW_ID1       Id of wave, simd, compute unit, etc.
    HW_REG_HW_ID2       Id of queue, pipeline, etc.
    HW_REG_GPR_ALLOC    Per-wave SGPR and VGPR allocation.
    HW_REG_LDS_ALLOC    Per-wave LDS allocation.
    HW_REG_IB_STS       Counters of outstanding instructions.
    HW_REG_SH_MEM_BASES Memory aperture.
    HW_REG_FLAT_SCR_LO  flat_scratch_lo register.
    HW_REG_FLAT_SCR_HI  flat_scratch_hi register.
    =================== ==========================================

Examples:

.. parsed-literal::

    reg = 1
    offset = 2
    size = 4
    hwreg_enc = reg | (offset << 6) | ((size - 1) << 11)

    s_getreg_b32 s2, 0x1881
    s_getreg_b32 s2, hwreg_enc                     // the same as above
    s_getreg_b32 s2, hwreg(1, 2, 4)                // the same as above
    s_getreg_b32 s2, hwreg(reg, offset, size)      // the same as above

    s_getreg_b32 s2, hwreg(15)
    s_getreg_b32 s2, hwreg(51, 1, 31)
    s_getreg_b32 s2, hwreg(HW_REG_LDS_ALLOC, 0, 1)

.. _amdgpu_synid_gfx12_simm16_f7832e:

simm16
------

The clause operand encodes the length and break_span of the clause:

* LENGTH = SIMM16[5:0]
  This field is set to the logical number of instructions in the clause, minus 1 (e.g. if a clause has 4 instructions,
  program this field to 3). The minimum number of instructions required for a clause is 2 and the maximum
  number of instructions is 63, therefore this field must be programmed in the range [1, 62] inclusive.

* BREAK_SPAN = SIMM16[11:8]
  This field is set to the number of instructions to issue before each clause break. If set to zero then there are
  no clause breaks. If set to nonzero value then the maximum number of instructions between clause breaks
  is 15.

.. _amdgpu_synid_gfx12_simm16_343e98:

simm16
------

The version operand spacifies the microcode version.

* SIMM16[7:0] specifies the microcode version.
* SIMM16[15:8] must be set to zero.

.. _amdgpu_synid_gfx12_soffset_422f18:

soffset
-------

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`m0<amdgpu_synid_m0>`

.. _amdgpu_synid_gfx12_soffset_ab3dc3:

soffset
-------

An offset added to the base address to get memory address.

* If offset is specified as a register, it supplies an unsigned byte offset.
* If offset is specified as a 21-bit immediate, it supplies a signed byte offset.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`m0<amdgpu_synid_m0>`

.. _amdgpu_synid_gfx12_soffset_1764b4:

soffset
-------

An unsigned 20-bit offset added to the base address to get memory address.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`m0<amdgpu_synid_m0>`

.. _amdgpu_synid_gfx12_src0_6802ce:

src0
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_src0_7987e4:

src0
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx12_src0_fd235e:

src0
----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_src0_420095:

src0
----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx12_src0_e016a1:

src0
----

Instruction input.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_src0_1d4114:

src0
----

attr0.x through attr63.w, parameter attribute and channel to be interpolated

.. _amdgpu_synid_gfx12_src1_e18ccf:

src1
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`m0<amdgpu_synid_m0>`

.. _amdgpu_synid_gfx12_src1_0f3a08:

src1
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx12_src1_6802ce:

src1
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_src1_7987e4:

src1
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx12_src1_fd235e:

src1
----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_src1_420095:

src1
----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx12_src1_e016a1:

src1
----

Instruction input.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_src1_731030:

src1
----

Instruction input.

*Size:* 8 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_src2_0f3a08:

src2
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx12_src2_6802ce:

src2
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_src2_7987e4:

src2
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx12_src2_81ba27:

src2
----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_src2_420095:

src2
----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx12_src2_e016a1:

src2
----

Instruction input.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_src2_7b936a:

src2
----

Instruction input.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx12_src2_96fbd3:

src2
----

Instruction input.

*Size:* 8 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx12_srcx0:

srcx0
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx12_srcy0:

srcy0
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx12_ssrc0_19fcd2:

ssrc0
-----

.. _amdgpu_synid_sendmsg_rtn:

sendmsg_rtn
===========

An 8-bit value in the instruction to encode the message type.

This operand may be specified as one of the following:

    * An :ref:`integer_number<amdgpu_synid_integer_number>` or an :ref:`absolute_expression<amdgpu_synid_absolute_expression>`. The value must be in the range 0..0xFFFF.
    * A *sendmsg* value described below.

    ==================================== ====================================================
    Sendmsg Value Syntax                 Description
    ==================================== ====================================================
    sendmsg(MSG_RTN_GET_DOORBELL)        Get doorbell ID.
    sendmsg(MSG_RTN_GET_DDID)            Get Draw/Dispatch ID.
    sendmsg(MSG_RTN_GET_TMA)             Get TMA value.
    sendmsg(MSG_RTN_GET_TBA)             Get TBA value.
    sendmsg(MSG_RTN_GET_REALTIME)        Get REALTIME value.
    sendmsg(MSG_RTN_SAVE_WAVE)           Report that this wave is ready to be context-saved.
    ==================================== ====================================================

Examples:

.. parsed-literal::

    s_sendmsg_rtn_b32 s0, 132
    s_sendmsg_rtn_b32 s0, sendmsg(MSG_GET_REALTIME)

.. _amdgpu_synid_gfx12_ssrc0_1a9ca5:

ssrc0
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`m0<amdgpu_synid_m0>`

.. _amdgpu_synid_gfx12_ssrc0_6fbc49:

ssrc0
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_ssrc0_0f3a08:

ssrc0
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx12_ssrc0_81ba27:

ssrc0
-----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_ssrc0_dfa11e:

ssrc0
-----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx12_ssrc1_0f3a08:

ssrc1
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx12_ssrc1_dfa11e:

ssrc1
-----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx12_tgt:

tgt
---

Instruction output.

*Size:* 4 dwords.

*Operands:* 

.. _amdgpu_synid_gfx12_vaddr_c8b8d4:

vaddr
-----

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vaddr_a972b9:

vaddr
-----

*Size:* 11 dwords.

*Operands:* 

.. _amdgpu_synid_gfx12_vaddr_c12f43:

vaddr
-----

*Size:* 12 dwords.

*Operands:* 

.. _amdgpu_synid_gfx12_vaddr_f2b449:

vaddr
-----

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vaddr_d82160:

vaddr
-----

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vcc:

vcc
---

Vector condition code. This operand depends on wavefront size:

* Should be :ref:`vcc_lo<amdgpu_synid_vcc_lo>` if wavefront size is 32.
* Should be :ref:`vcc<amdgpu_synid_vcc>` if wavefront size is 64.

.. _amdgpu_synid_gfx12_vdata_89680f:

vdata
-----

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vdata_aac3e8:

vdata
-----

Instruction output.

*Size:* 10 dwords.

*Operands:* 

.. _amdgpu_synid_gfx12_vdata_bdb32f:

vdata
-----

Instruction output.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vdata_48e42f:

vdata
-----

Instruction output.

*Size:* 3 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vdata_69a144:

vdata
-----

Instruction output.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vdst_54e16e:

vdst
----

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`null<amdgpu_synid_null>`

.. _amdgpu_synid_gfx12_vdst_89680f:

vdst
----

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vdst_7de8e7:

vdst
----

Instruction output.

*Size:* 2 dwords.

*Operands:* :ref:`exec<amdgpu_synid_exec>`

.. _amdgpu_synid_gfx12_vdst_bdb32f:

vdst
----

Instruction output.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vdst_006c40:

vdst
----

Instruction output.

*Size:* 2 dwords.

*Operands:* :ref:`vcc<amdgpu_synid_vcc>`

.. _amdgpu_synid_gfx12_vdst_48e42f:

vdst
----

Instruction output.

*Size:* 3 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vdst_2eda77:

vdst
----

Instruction output.

*Size:* 32 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vdst_227281:

vdst
----

Instruction output.

*Size:* 4 dwords if wavefront size is 64, otherwise 8 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vdst_69a144:

vdst
----

Instruction output.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vdst_47d3bc:

vdst
----

Instruction output.

*Size:* 8 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vdstx:

vdstx
-----

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vdsty:

vdsty
-----

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vsrc_6802ce:

vsrc
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vsrc_fd235e:

vsrc
----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vsrc_56f215:

vsrc
----

Instruction input.

*Size:* 3 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vsrc_89fd7b:

vsrc
----

Instruction input.

*Size:* 32 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vsrc_e016a1:

vsrc
----

Instruction input.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vsrc0:

vsrc0
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vsrc1_6802ce:

vsrc1
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vsrc1_fd235e:

vsrc1
-----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vsrc2:

vsrc2
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vsrc3:

vsrc3
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vsrcx1:

vsrcx1
------

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx12_vsrcy1:

vsrcy1
------

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

