..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

.. _amdgpu_synid_gfx950_addr_c8b8d4:

addr
----

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_addr_f2b449:

addr
----

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_attr:

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

    v_interp_p1_f32 v1, v0, attr0.x
    v_interp_p1_f32 v1, v0, attr32.w

.. _amdgpu_synid_gfx950_data_be4895:

data
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_data_9ad749:

data
----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_data_cfb402:

data
----

Instruction input.

*Size:* 3 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_data_848ff7:

data
----

Instruction input.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_data0_be4895:

data0
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_data0_9ad749:

data0
-----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_data0_cfb402:

data0
-----

Instruction input.

*Size:* 3 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_data0_848ff7:

data0
-----

Instruction input.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_data1_be4895:

data1
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_data1_9ad749:

data1
-----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_literal_81e671:

literal
-------

*Size:* 1 dword.

*Operands:* 

.. _amdgpu_synid_gfx950_literal_39b593:

literal
-------

*Size:* 1 dword.

*Operands:* :ref:`imm16<amdgpu_synid_imm16>`

.. _amdgpu_synid_gfx950_saddr_13d69a:

saddr
-----

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_saddr_ce8216:

saddr
-----

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_sbase_010ce0:

sbase
-----

A 128-bit buffer resource constant for scalar memory operations which provides a base address, a size and a stride.

*Size:* 4 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_sbase_044055:

sbase
-----

A 64-bit base address for scalar memory operations.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_sbase_0cd545:

sbase
-----

This operand is ignored by H/W and :ref:`flat_scratch<amdgpu_synid_flat_scratch>` is supplied instead.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_scale_src0:

scale_src0
----------

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx950_scale_src1:

scale_src1
----------

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx950_sdata_aefe00:

sdata
-----

Input data for an atomic instruction.

Optionally may serve as an output data:

* If :ref:`glc<amdgpu_synid_glc>` is specified, gets the memory value before the operation.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_sdata_eb6f2a:

sdata
-----

Input data for an atomic instruction.

Optionally may serve as an output data:

* If :ref:`glc<amdgpu_synid_glc>` is specified, gets the memory value before the operation.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_sdata_c6aec1:

sdata
-----

Input data for an atomic instruction.

Optionally may serve as an output data:

* If :ref:`glc<amdgpu_synid_glc>` is specified, gets the memory value before the operation.

*Size:* 4 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_sdata_94342d:

sdata
-----

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_sdata_d725ab:

sdata
-----

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`simm8<amdgpu_synid_simm8>`

.. _amdgpu_synid_gfx950_sdata_3bc700:

sdata
-----

Instruction output.

*Size:* 16 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_sdata_718cc4:

sdata
-----

Instruction output.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_sdata_0804b1:

sdata
-----

Instruction output.

*Size:* 4 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_sdata_362c37:

sdata
-----

Instruction output.

*Size:* 8 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_sdst_02b357:

sdst
----

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`

.. _amdgpu_synid_gfx950_sdst_3bec61:

sdst
----

Instruction output.

*Size:* 1 dword if wavefront size is 32, otherwise 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_sdst_1db612:

sdst
----

Instruction output.

*Size:* 1 dword if wavefront size is 32, otherwise 2 dwords.

*Operands:* :ref:`vcc<amdgpu_synid_vcc>`

.. _amdgpu_synid_gfx950_sdst_94342d:

sdst
----

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_sdst_06b266:

sdst
----

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`

.. _amdgpu_synid_gfx950_sdst_718cc4:

sdst
----

Instruction output.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_sdst_a319e6:

sdst
----

Instruction output.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`exec<amdgpu_synid_exec>`

.. _amdgpu_synid_gfx950_simm16_7ed651:

simm16
------

*Size:* 1 dword.

*Operands:* :ref:`hwreg<amdgpu_synid_hwreg>`

.. _amdgpu_synid_gfx950_simm16_39b593:

simm16
------

*Size:* 1 dword.

*Operands:* :ref:`imm16<amdgpu_synid_imm16>`

.. _amdgpu_synid_gfx950_simm16_3d2a4f:

simm16
------

*Size:* 1 dword.

*Operands:* :ref:`label<amdgpu_synid_label>`

.. _amdgpu_synid_gfx950_simm16_ee8b30:

simm16
------

*Size:* 1 dword.

*Operands:* :ref:`sendmsg<amdgpu_synid_sendmsg>`

.. _amdgpu_synid_gfx950_simm16_218bea:

simm16
------

*Size:* 1 dword.

*Operands:* :ref:`waitcnt<amdgpu_synid_waitcnt>`

.. _amdgpu_synid_gfx950_simm16_cc1716:

simm16
------

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`hwreg<amdgpu_synid_hwreg>`

.. _amdgpu_synid_gfx950_soffset_1189ef:

soffset
-------

An offset added to the base address to get memory address.

* If offset is specified as a register, it supplies an unsigned byte offset.
* If offset is specified as a 21-bit immediate, it supplies a signed byte offset.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`m0<amdgpu_synid_m0>`

.. _amdgpu_synid_gfx950_soffset_8aa27a:

soffset
-------

An unsigned 20-bit offset added to the base address to get memory address.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`m0<amdgpu_synid_m0>`

.. _amdgpu_synid_gfx950_soffset_d856a0:

soffset
-------

An unsigned byte offset.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx950_src0_1027ca:

src0
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_src0_6802ce:

src0
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_src0_be4895:

src0
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_src0_516946:

src0
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`lds_direct<amdgpu_synid_lds_direct>`

.. _amdgpu_synid_gfx950_src0_14b47a:

src0
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx950_src0_0f0007:

src0
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx950_src0_168f33:

src0
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`lds_direct<amdgpu_synid_lds_direct>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx950_src0_06ee74:

src0
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`lds_direct<amdgpu_synid_lds_direct>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx950_src0_9ad749:

src0
----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_src0_e30a18:

src0
----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx950_src0_62f8c2:

src0
----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx950_src0_848ff7:

src0
----

Instruction input.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_src0_ca334d:

src0
----

Instruction input.

*Size:* 8 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_src0_1d4114:

src0
----

attr0.x through attr63.w, parameter attribute and channel to be interpolated

.. _amdgpu_synid_gfx950_src1_43aa79:

src1
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`m0<amdgpu_synid_m0>`

.. _amdgpu_synid_gfx950_src1_6802ce:

src1
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_src1_be4895:

src1
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_src1_14b47a:

src1
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx950_src1_d52854:

src1
----

Instruction input.

*Size:* 16 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_src1_9ad749:

src1
----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_src1_e30a18:

src1
----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx950_src1_848ff7:

src1
----

Instruction input.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_src1_ca334d:

src1
----

Instruction input.

*Size:* 8 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_src2_6802ce:

src2
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_src2_581e7b:

src2
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx950_src2_14b47a:

src2
----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx950_src2_14f1c8:

src2
----

Instruction input.

*Size:* 16 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx950_src2_1ff383:

src2
----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx950_src2_e30a18:

src2
----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx950_src2_a90bd6:

src2
----

Instruction input.

*Size:* 32 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx950_src2_e016a1:

src2
----

Instruction input.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_src2_ca14ce:

src2
----

Instruction input.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx950_src2_f36021:

src2
----

Instruction input.

*Size:* 8 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx950_srsrc_e73d16:

srsrc
-----

Buffer resource constant which defines the address and characteristics of the buffer in memory.

*Size:* 4 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_srsrc_79ffcd:

srsrc
-----

Image resource constant which defines the location of the image buffer in memory, its dimensions, tiling, and data format.

*Size:* 8 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_ssamp:

ssamp
-----

Sampler constant used to specify filtering options applied to the image data after it is read.

*Size:* 4 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_ssrc0_595c25:

ssrc0
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_ssrc0_eecc17:

ssrc0
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx950_ssrc0_e9f591:

ssrc0
-----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_ssrc0_1ce478:

ssrc0
-----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx950_ssrc0_83ef5a:

ssrc0
-----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx950_ssrc1_5c7b50:

ssrc1
-----

Instruction input.

*Size:* 1 dword.

*Operands:* 

.. _amdgpu_synid_gfx950_ssrc1_eecc17:

ssrc1
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`m0<amdgpu_synid_m0>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx950_ssrc1_1ce478:

ssrc1
-----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`

.. _amdgpu_synid_gfx950_ssrc1_83ef5a:

ssrc1
-----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`vcc<amdgpu_synid_vcc>`, :ref:`ttmp<amdgpu_synid_ttmp>`, :ref:`exec<amdgpu_synid_exec>`, :ref:`vccz<amdgpu_synid_vccz>`, :ref:`execz<amdgpu_synid_execz>`, :ref:`scc<amdgpu_synid_scc>`, :ref:`fconst<amdgpu_synid_fconst>`, :ref:`literal<amdgpu_synid_literal>`

.. _amdgpu_synid_gfx950_tgt:

tgt
---

An export target:

    ================== ===================================
    Syntax             Description
    ================== ===================================
    pos{0..3}          Copy vertex position 0..3.
    param{0..31}       Copy vertex parameter 0..31.
    mrt{0..7}          Copy pixel color to the MRTs 0..7.
    mrtz               Copy pixel depth (Z) data.
    null               Copy nothing.
    ================== ===================================

.. _amdgpu_synid_gfx950_vaddr_5d0b42:

vaddr
-----

Image address which includes from one to four dimensional coordinates and other data used to locate a position in the image.

*Size:* 1, 2, 3, 4, 8 or 16 dwords. Actual size depends on opcode, specific image being handled and :ref:`a16<amdgpu_synid_a16>`.

    Note 1. Image format and dimensions are encoded in the image resource constant but not in the instruction.

    Note 2. Actually image address size may vary from 1 to 13 dwords, but assembler currently supports a limited range of register sequences.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_vaddr_7a736f:

vaddr
-----

This is an optional operand which may specify offset and/or index.

*Size:* 0, 1 or 2 dwords. Size is controlled by modifiers :ref:`offen<amdgpu_synid_offen>` and :ref:`idxen<amdgpu_synid_idxen>`:

* If only :ref:`idxen<amdgpu_synid_idxen>` is specified, this operand supplies an index. Size is 1 dword.
* If only :ref:`offen<amdgpu_synid_offen>` is specified, this operand supplies an offset. Size is 1 dword.
* If both modifiers are specified, index is in the first register and offset is in the second. Size is 2 dwords.
* If none of these modifiers are specified, this operand must be set to :ref:`off<amdgpu_synid_off>`.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_vcc:

vcc
---

Vector condition code.

*Size:* 1 dword.

*Operands:* :ref:`vcc<amdgpu_synid_vcc>`

.. _amdgpu_synid_gfx950_vdata_a5f23e:

vdata
-----

Image data to store by an *image_store* instruction.

*Size:* depends on :ref:`dmask<amdgpu_synid_dmask>` and :ref:`d16<amdgpu_synid_d16>`:

* :ref:`dmask<amdgpu_synid_dmask>` may specify from 1 to 4 data elements. Each data element occupies either 32 bits or 16 bits depending on :ref:`d16<amdgpu_synid_d16>`.
* :ref:`d16<amdgpu_synid_d16>` specifies that data in registers are packed; each value occupies 16 bits.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdata_2a60db:

vdata
-----

Input data for an atomic instruction.

Optionally may serve as an output data:

* If :ref:`glc<amdgpu_synid_glc>` is specified, gets the memory value before the operation.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdata_2d0375:

vdata
-----

Input data for an atomic instruction.

Optionally may serve as an output data:

* If :ref:`glc<amdgpu_synid_glc>` is specified, gets the memory value before the operation.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdata_8e9b87:

vdata
-----

Input data for an atomic instruction.

Optionally may serve as an output data:

* If :ref:`glc<amdgpu_synid_glc>` is specified, gets the memory value before the operation.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdata_2a143d:

vdata
-----

Input data for an atomic instruction.

Optionally may serve as an output data:

* If :ref:`glc<amdgpu_synid_glc>` is specified, gets the memory value before the operation.

*Size:* depends on :ref:`dmask<amdgpu_synid_dmask>` and :ref:`tfe<amdgpu_synid_tfe>`:

* :ref:`dmask<amdgpu_synid_dmask>` may specify 1 data element for 32-bit-per-pixel surfaces or 2 data elements for 64-bit-per-pixel surfaces. Each data element occupies 1 dword.
* :ref:`tfe<amdgpu_synid_tfe>` adds 1 dword if specified.

  Note: the surface data format is indicated in the image resource constant but not in the instruction.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdata_576598:

vdata
-----

Input data for an atomic instruction.

Optionally may serve as an output data:

* If :ref:`glc<amdgpu_synid_glc>` is specified, gets the memory value before the operation.

*Size:* depends on :ref:`dmask<amdgpu_synid_dmask>` and :ref:`tfe<amdgpu_synid_tfe>`:

* :ref:`dmask<amdgpu_synid_dmask>` may specify 2 data elements for 32-bit-per-pixel surfaces or 4 data elements for 64-bit-per-pixel surfaces. Each data element occupies 1 dword.
* :ref:`tfe<amdgpu_synid_tfe>` adds 1 dword if specified.

  Note: the surface data format is indicated in the image resource constant but not in the instruction.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdata_fa7dbd:

vdata
-----

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdata_0f48d1:

vdata
-----

Instruction output.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdata_260aca:

vdata
-----

Instruction output.

*Size:* 3 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdata_180bef:

vdata
-----

Instruction output.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdata_a507a0:

vdata
-----

Instruction output.

*Size:* depends on :ref:`dmask<amdgpu_synid_dmask>` and :ref:`d16<amdgpu_synid_d16>`:

* :ref:`dmask<amdgpu_synid_dmask>` may specify from 1 to 4 data elements. Each data element occupies either 32 bits or 16 bits depending on :ref:`d16<amdgpu_synid_d16>`.
* :ref:`d16<amdgpu_synid_d16>` specifies that data in registers are packed; each value occupies 16 bits.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdst_c8ee02:

vdst
----

Data returned by a 32-bit atomic flat instruction.

This is an optional operand. It must be used if and only if :ref:`glc<amdgpu_synid_glc>` is specified.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdst_ef6c94:

vdst
----

Data returned by a 64-bit atomic flat instruction.

This is an optional operand. It must be used if and only if :ref:`glc<amdgpu_synid_glc>` is specified.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdst_78dd0a:

vdst
----

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdst_59204c:

vdst
----

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`s<amdgpu_synid_s>`, :ref:`flat_scratch<amdgpu_synid_flat_scratch>`, :ref:`xnack_mask<amdgpu_synid_xnack_mask>`, :ref:`ttmp<amdgpu_synid_ttmp>`

.. _amdgpu_synid_gfx950_vdst_89680f:

vdst
----

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_vdst_fa7dbd:

vdst
----

Instruction output.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdst_5f7812:

vdst
----

Instruction output.

*Size:* 16 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_vdst_d6f4bd:

vdst
----

Instruction output.

*Size:* 16 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdst_bdb32f:

vdst
----

Instruction output.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_vdst_0f48d1:

vdst
----

Instruction output.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdst_260aca:

vdst
----

Instruction output.

*Size:* 3 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdst_2eda77:

vdst
----

Instruction output.

*Size:* 32 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_vdst_8c77d4:

vdst
----

Instruction output.

*Size:* 32 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdst_69a144:

vdst
----

Instruction output.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_vdst_180bef:

vdst
----

Instruction output.

*Size:* 4 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vdst_363335:

vdst
----

Instruction output.

*Size:* 6 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_vdst_c8d317:

vdst
----

Instruction output.

*Size:* 8 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`, :ref:`a<amdgpu_synid_a>`

.. _amdgpu_synid_gfx950_vsrc:

vsrc
----

Interpolation parameter to read:

    ============ ===================================
    Syntax       Description
    ============ ===================================
    p0           Parameter *P0*.
    p10          Parameter *P10*.
    p20          Parameter *P20*.
    ============ ===================================

.. _amdgpu_synid_gfx950_vsrc0:

vsrc0
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_vsrc1_6802ce:

vsrc1
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_vsrc1_fd235e:

vsrc1
-----

Instruction input.

*Size:* 2 dwords.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_vsrc2:

vsrc2
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

.. _amdgpu_synid_gfx950_vsrc3:

vsrc3
-----

Instruction input.

*Size:* 1 dword.

*Operands:* :ref:`v<amdgpu_synid_v>`

