..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

====================================================================================
Syntax of gfx1011 and gfx1012 Instructions
====================================================================================

.. contents::
  :local:

Introduction
============

This document describes the syntax of *instructions specific to gfx1011 and gfx1012*.

For a description of other gfx1011 and gfx1012 instructions see :doc:`Syntax of Core GFX10 Instructions<AMDGPUAsmGFX10>`.

Notation
========

Notation used in this document is explained :ref:`here<amdgpu_syn_instruction_notation>`.

Overview
========

An overview of generic syntax and other features of AMDGPU instructions may be found :ref:`in this document<amdgpu_syn_instructions>`.

Instructions
============


DPP16
-----------------------

.. parsed-literal::

    **INSTRUCTION**           **DST**      **SRC0**         **SRC1**          **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    v_dot2c_f32_f16_dpp   :ref:`vdst<amdgpu_synid1011_vdst32_0>`,    :ref:`vsrc0<amdgpu_synid1011_vsrc32_0>`::ref:`f16x2<amdgpu_synid1011_type_dev>`, :ref:`vsrc1<amdgpu_synid1011_vsrc32_0>`::ref:`f16x2<amdgpu_synid1011_type_dev>`   :ref:`dpp16_ctrl<amdgpu_synid_dpp16_ctrl>` :ref:`row_mask<amdgpu_synid_row_mask>` :ref:`bank_mask<amdgpu_synid_bank_mask>` :ref:`bound_ctrl<amdgpu_synid_bound_ctrl>` :ref:`fi<amdgpu_synid_fi16>`
    v_dot4c_i32_i8_dpp    :ref:`vdst<amdgpu_synid1011_vdst32_0>`,    :ref:`vsrc0<amdgpu_synid1011_vsrc32_0>`::ref:`i8x4<amdgpu_synid1011_type_dev>`,  :ref:`vsrc1<amdgpu_synid1011_vsrc32_0>`::ref:`i8x4<amdgpu_synid1011_type_dev>`    :ref:`dpp16_ctrl<amdgpu_synid_dpp16_ctrl>` :ref:`row_mask<amdgpu_synid_row_mask>` :ref:`bank_mask<amdgpu_synid_bank_mask>` :ref:`bound_ctrl<amdgpu_synid_bound_ctrl>` :ref:`fi<amdgpu_synid_fi16>`

DPP8
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC0**         **SRC1**             **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    v_dot2c_f32_f16_dpp            :ref:`vdst<amdgpu_synid1011_vdst32_0>`,     :ref:`vsrc0<amdgpu_synid1011_vsrc32_0>`::ref:`f16x2<amdgpu_synid1011_type_dev>`, :ref:`vsrc1<amdgpu_synid1011_vsrc32_0>`::ref:`f16x2<amdgpu_synid1011_type_dev>`      :ref:`dpp8_sel<amdgpu_synid_dpp8_sel>` :ref:`fi<amdgpu_synid_fi8>`
    v_dot4c_i32_i8_dpp             :ref:`vdst<amdgpu_synid1011_vdst32_0>`,     :ref:`vsrc0<amdgpu_synid1011_vsrc32_0>`::ref:`i8x4<amdgpu_synid1011_type_dev>`,  :ref:`vsrc1<amdgpu_synid1011_vsrc32_0>`::ref:`i8x4<amdgpu_synid1011_type_dev>`       :ref:`dpp8_sel<amdgpu_synid_dpp8_sel>` :ref:`fi<amdgpu_synid_fi8>`

VOP2
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC0**        **SRC1**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    v_dot2c_f32_f16                :ref:`vdst<amdgpu_synid1011_vdst32_0>`,     :ref:`src0<amdgpu_synid1011_src32_0>`::ref:`f16x2<amdgpu_synid1011_type_dev>`, :ref:`vsrc1<amdgpu_synid1011_vsrc32_0>`::ref:`f16x2<amdgpu_synid1011_type_dev>`
    v_dot4c_i32_i8                 :ref:`vdst<amdgpu_synid1011_vdst32_0>`,     :ref:`src0<amdgpu_synid1011_src32_0>`::ref:`i8x4<amdgpu_synid1011_type_dev>`,  :ref:`vsrc1<amdgpu_synid1011_vsrc32_0>`::ref:`i8x4<amdgpu_synid1011_type_dev>`

VOP3P
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC0**        **SRC1**        **SRC2**           **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    v_dot2_f32_f16                 :ref:`vdst<amdgpu_synid1011_vdst32_0>`,     :ref:`src0<amdgpu_synid1011_src32_0>`::ref:`f16x2<amdgpu_synid1011_type_dev>`, :ref:`src1<amdgpu_synid1011_src32_1>`::ref:`f16x2<amdgpu_synid1011_type_dev>`, :ref:`src2<amdgpu_synid1011_src32_1>`::ref:`f32<amdgpu_synid1011_type_dev>`       :ref:`neg_lo<amdgpu_synid_neg_lo>` :ref:`neg_hi<amdgpu_synid_neg_hi>` :ref:`clamp<amdgpu_synid_clamp>`
    v_dot2_i32_i16                 :ref:`vdst<amdgpu_synid1011_vdst32_0>`,     :ref:`src0<amdgpu_synid1011_src32_2>`::ref:`i16x2<amdgpu_synid1011_type_dev>`, :ref:`src1<amdgpu_synid1011_src32_3>`::ref:`i16x2<amdgpu_synid1011_type_dev>`, :ref:`src2<amdgpu_synid1011_src32_1>`::ref:`i32<amdgpu_synid1011_type_dev>`       :ref:`clamp<amdgpu_synid_clamp>`
    v_dot2_u32_u16                 :ref:`vdst<amdgpu_synid1011_vdst32_0>`,     :ref:`src0<amdgpu_synid1011_src32_2>`::ref:`u16x2<amdgpu_synid1011_type_dev>`, :ref:`src1<amdgpu_synid1011_src32_3>`::ref:`u16x2<amdgpu_synid1011_type_dev>`, :ref:`src2<amdgpu_synid1011_src32_1>`::ref:`u32<amdgpu_synid1011_type_dev>`       :ref:`clamp<amdgpu_synid_clamp>`
    v_dot4_i32_i8                  :ref:`vdst<amdgpu_synid1011_vdst32_0>`,     :ref:`src0<amdgpu_synid1011_src32_0>`::ref:`i8x4<amdgpu_synid1011_type_dev>`,  :ref:`src1<amdgpu_synid1011_src32_1>`::ref:`i8x4<amdgpu_synid1011_type_dev>`,  :ref:`src2<amdgpu_synid1011_src32_1>`::ref:`i32<amdgpu_synid1011_type_dev>`       :ref:`clamp<amdgpu_synid_clamp>`
    v_dot4_u32_u8                  :ref:`vdst<amdgpu_synid1011_vdst32_0>`,     :ref:`src0<amdgpu_synid1011_src32_0>`::ref:`u8x4<amdgpu_synid1011_type_dev>`,  :ref:`src1<amdgpu_synid1011_src32_1>`::ref:`u8x4<amdgpu_synid1011_type_dev>`,  :ref:`src2<amdgpu_synid1011_src32_1>`::ref:`u32<amdgpu_synid1011_type_dev>`       :ref:`clamp<amdgpu_synid_clamp>`
    v_dot8_i32_i4                  :ref:`vdst<amdgpu_synid1011_vdst32_0>`,     :ref:`src0<amdgpu_synid1011_src32_0>`::ref:`i4x8<amdgpu_synid1011_type_dev>`,  :ref:`src1<amdgpu_synid1011_src32_1>`::ref:`i4x8<amdgpu_synid1011_type_dev>`,  :ref:`src2<amdgpu_synid1011_src32_1>`::ref:`i32<amdgpu_synid1011_type_dev>`       :ref:`clamp<amdgpu_synid_clamp>`
    v_dot8_u32_u4                  :ref:`vdst<amdgpu_synid1011_vdst32_0>`,     :ref:`src0<amdgpu_synid1011_src32_0>`::ref:`u4x8<amdgpu_synid1011_type_dev>`,  :ref:`src1<amdgpu_synid1011_src32_1>`::ref:`u4x8<amdgpu_synid1011_type_dev>`,  :ref:`src2<amdgpu_synid1011_src32_1>`::ref:`u32<amdgpu_synid1011_type_dev>`       :ref:`clamp<amdgpu_synid_clamp>`

.. |---| unicode:: U+02014 .. em dash


.. toctree::
    :hidden:

    AMDGPUAsmGFX10
    gfx1011_src32_0
    gfx1011_src32_1
    gfx1011_src32_2
    gfx1011_src32_3
    gfx1011_vdst32_0
    gfx1011_vsrc32_0
    gfx1011_type_dev
