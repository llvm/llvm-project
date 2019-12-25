..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

====================================================================================
Syntax of gfx908 Instructions
====================================================================================

.. contents::
  :local:

Introduction
============

This document describes the syntax of *instructions specific to gfx908*.

For a description of other gfx908 instructions see :doc:`Syntax of Core GFX9 Instructions<AMDGPUAsmGFX9>`.

Notation
========

Notation used in this document is explained :ref:`here<amdgpu_syn_instruction_notation>`.

Overvew
=======

An overview of generic syntax and other features of AMDGPU instructions may be found :ref:`in this document<amdgpu_syn_instructions>`.

Instructions
============


FLAT
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**             **SRC0**      **SRC1**         **SRC2**           **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    global_atomic_add_f32          :ref:`vdst<amdgpu_synid908_dst_flat_atomic32>`::ref:`opt<amdgpu_synid908_opt>`,       :ref:`vaddr<amdgpu_synid908_vaddr_flat_global>`,    :ref:`vdata<amdgpu_synid908_vdata32_0>`,       :ref:`saddr<amdgpu_synid908_saddr_flat_global>`          :ref:`offset13s<amdgpu_synid_flat_offset13s>`
    global_atomic_pk_add_f16       :ref:`vdst<amdgpu_synid908_dst_flat_atomic32>`::ref:`opt<amdgpu_synid908_opt>`::ref:`f16x2<amdgpu_synid908_type_dev>`, :ref:`vaddr<amdgpu_synid908_vaddr_flat_global>`,    :ref:`vdata<amdgpu_synid908_vdata32_0>`::ref:`f16x2<amdgpu_synid908_type_dev>`, :ref:`saddr<amdgpu_synid908_saddr_flat_global>`          :ref:`offset13s<amdgpu_synid_flat_offset13s>`

MUBUF
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **SRC0**             **SRC1**      **SRC2**      **SRC3**           **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    buffer_atomic_add_f32          :ref:`vdata<amdgpu_synid908_data_buf_atomic32>`::ref:`dst<amdgpu_synid908_ret>`,       :ref:`vaddr<amdgpu_synid908_addr_buf>`,    :ref:`srsrc<amdgpu_synid908_rsrc_buf>`,    :ref:`soffset<amdgpu_synid908_offset_buf>`        :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_pk_add_f16       :ref:`vdata<amdgpu_synid908_data_buf_atomic32>`::ref:`dst<amdgpu_synid908_ret>`::ref:`f16x2<amdgpu_synid908_type_dev>`, :ref:`vaddr<amdgpu_synid908_addr_buf>`,    :ref:`srsrc<amdgpu_synid908_rsrc_buf>`,    :ref:`soffset<amdgpu_synid908_offset_buf>`        :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`slc<amdgpu_synid_slc>`

VOP2
-----------------------

.. parsed-literal::

    **INSTRUCTION**           **DST**         **SRC0**         **SRC1**          **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    v_dot2c_f32_f16       :ref:`vdst<amdgpu_synid908_vdst32_0>`,       :ref:`src0<amdgpu_synid908_src32_0>`::ref:`f16x2<amdgpu_synid908_type_dev>`,  :ref:`vsrc1<amdgpu_synid908_vsrc32_0>`::ref:`f16x2<amdgpu_synid908_type_dev>`
    v_dot2c_f32_f16_dpp   :ref:`vdst<amdgpu_synid908_vdst32_0>`,       :ref:`vsrc0<amdgpu_synid908_vsrc32_0>`::ref:`f16x2<amdgpu_synid908_type_dev>`, :ref:`vsrc1<amdgpu_synid908_vsrc32_0>`::ref:`f16x2<amdgpu_synid908_type_dev>`   :ref:`dpp_ctrl<amdgpu_synid_dpp_ctrl>` :ref:`row_mask<amdgpu_synid_row_mask>` :ref:`bank_mask<amdgpu_synid_bank_mask>` :ref:`bound_ctrl<amdgpu_synid_bound_ctrl>`
    v_dot2c_i32_i16       :ref:`vdst<amdgpu_synid908_vdst32_0>`,       :ref:`src0<amdgpu_synid908_src32_0>`::ref:`i16x2<amdgpu_synid908_type_dev>`,  :ref:`vsrc1<amdgpu_synid908_vsrc32_0>`::ref:`i16x2<amdgpu_synid908_type_dev>`
    v_dot2c_i32_i16_dpp   :ref:`vdst<amdgpu_synid908_vdst32_0>`,       :ref:`vsrc0<amdgpu_synid908_vsrc32_0>`::ref:`i16x2<amdgpu_synid908_type_dev>`, :ref:`vsrc1<amdgpu_synid908_vsrc32_0>`::ref:`i16x2<amdgpu_synid908_type_dev>`   :ref:`dpp_ctrl<amdgpu_synid_dpp_ctrl>` :ref:`row_mask<amdgpu_synid_row_mask>` :ref:`bank_mask<amdgpu_synid_bank_mask>` :ref:`bound_ctrl<amdgpu_synid_bound_ctrl>`
    v_dot4c_i32_i8        :ref:`vdst<amdgpu_synid908_vdst32_0>`,       :ref:`src0<amdgpu_synid908_src32_0>`::ref:`i8x4<amdgpu_synid908_type_dev>`,   :ref:`vsrc1<amdgpu_synid908_vsrc32_0>`::ref:`i8x4<amdgpu_synid908_type_dev>`
    v_dot4c_i32_i8_dpp    :ref:`vdst<amdgpu_synid908_vdst32_0>`,       :ref:`vsrc0<amdgpu_synid908_vsrc32_0>`::ref:`i8x4<amdgpu_synid908_type_dev>`,  :ref:`vsrc1<amdgpu_synid908_vsrc32_0>`::ref:`i8x4<amdgpu_synid908_type_dev>`    :ref:`dpp_ctrl<amdgpu_synid_dpp_ctrl>` :ref:`row_mask<amdgpu_synid_row_mask>` :ref:`bank_mask<amdgpu_synid_bank_mask>` :ref:`bound_ctrl<amdgpu_synid_bound_ctrl>`
    v_dot8c_i32_i4        :ref:`vdst<amdgpu_synid908_vdst32_0>`,       :ref:`src0<amdgpu_synid908_src32_0>`::ref:`i4x8<amdgpu_synid908_type_dev>`,   :ref:`vsrc1<amdgpu_synid908_vsrc32_0>`::ref:`i4x8<amdgpu_synid908_type_dev>`
    v_dot8c_i32_i4_dpp    :ref:`vdst<amdgpu_synid908_vdst32_0>`,       :ref:`vsrc0<amdgpu_synid908_vsrc32_0>`::ref:`i4x8<amdgpu_synid908_type_dev>`,  :ref:`vsrc1<amdgpu_synid908_vsrc32_0>`::ref:`i4x8<amdgpu_synid908_type_dev>`    :ref:`dpp_ctrl<amdgpu_synid_dpp_ctrl>` :ref:`row_mask<amdgpu_synid_row_mask>` :ref:`bank_mask<amdgpu_synid_bank_mask>` :ref:`bound_ctrl<amdgpu_synid_bound_ctrl>`
    v_fmac_f32            :ref:`vdst<amdgpu_synid908_vdst32_0>`,       :ref:`src0<amdgpu_synid908_src32_0>`,        :ref:`vsrc1<amdgpu_synid908_vsrc32_0>`
    v_fmac_f32_dpp        :ref:`vdst<amdgpu_synid908_vdst32_0>`,       :ref:`vsrc0<amdgpu_synid908_vsrc32_0>`::ref:`m<amdgpu_synid908_mod_dpp_sdwa_abs_neg>`,     :ref:`vsrc1<amdgpu_synid908_vsrc32_0>`::ref:`m<amdgpu_synid908_mod_dpp_sdwa_abs_neg>`       :ref:`dpp_ctrl<amdgpu_synid_dpp_ctrl>` :ref:`row_mask<amdgpu_synid_row_mask>` :ref:`bank_mask<amdgpu_synid_bank_mask>` :ref:`bound_ctrl<amdgpu_synid_bound_ctrl>`
    v_pk_fmac_f16         :ref:`vdst<amdgpu_synid908_vdst32_0>`::ref:`f16x2<amdgpu_synid908_type_dev>`, :ref:`src0<amdgpu_synid908_src32_0>`::ref:`f16x2<amdgpu_synid908_type_dev>`,  :ref:`vsrc1<amdgpu_synid908_vsrc32_0>`::ref:`f16x2<amdgpu_synid908_type_dev>`
    v_xnor_b32            :ref:`vdst<amdgpu_synid908_vdst32_0>`,       :ref:`src0<amdgpu_synid908_src32_0>`,        :ref:`vsrc1<amdgpu_synid908_vsrc32_0>`
    v_xnor_b32_dpp        :ref:`vdst<amdgpu_synid908_vdst32_0>`,       :ref:`vsrc0<amdgpu_synid908_vsrc32_0>`,       :ref:`vsrc1<amdgpu_synid908_vsrc32_0>`         :ref:`dpp_ctrl<amdgpu_synid_dpp_ctrl>` :ref:`row_mask<amdgpu_synid_row_mask>` :ref:`bank_mask<amdgpu_synid_bank_mask>` :ref:`bound_ctrl<amdgpu_synid_bound_ctrl>`
    v_xnor_b32_sdwa       :ref:`vdst<amdgpu_synid908_vdst32_0>`,       :ref:`src0<amdgpu_synid908_src32_0>`::ref:`m<amdgpu_synid908_mod_sdwa_sext>`,      :ref:`vsrc1<amdgpu_synid908_vsrc32_0>`::ref:`m<amdgpu_synid908_mod_sdwa_sext>`       :ref:`dst_sel<amdgpu_synid_dst_sel>` :ref:`dst_unused<amdgpu_synid_dst_unused>` :ref:`src0_sel<amdgpu_synid_src0_sel>` :ref:`src1_sel<amdgpu_synid_src1_sel>`

VOP3
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC0**      **SRC1**           **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    v_fmac_f32_e64                 :ref:`vdst<amdgpu_synid908_vdst32_0>`,     :ref:`src0<amdgpu_synid908_src32_1>`::ref:`m<amdgpu_synid908_mod_vop3_abs_neg>`,   :ref:`src1<amdgpu_synid908_src32_2>`::ref:`m<amdgpu_synid908_mod_vop3_abs_neg>`         :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_xnor_b32_e64                 :ref:`vdst<amdgpu_synid908_vdst32_0>`,     :ref:`src0<amdgpu_synid908_src32_1>`,     :ref:`src1<amdgpu_synid908_src32_2>`

VOP3P
-----------------------

.. parsed-literal::

    **INSTRUCTION**            **DST**          **SRC0**          **SRC1**          **SRC2**           **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    v_accvgpr_read_b32     :ref:`vdst<amdgpu_synid908_vdst32_0>`,        :ref:`asrc<amdgpu_synid908_asrc32_0>`
    v_accvgpr_write_b32    :ref:`adst<amdgpu_synid908_adst32_0>`,        :ref:`src<amdgpu_synid908_src32_3>`
    v_dot2_f32_f16         :ref:`vdst<amdgpu_synid908_vdst32_0>`,        :ref:`src0<amdgpu_synid908_src32_1>`::ref:`f16x2<amdgpu_synid908_type_dev>`,   :ref:`src1<amdgpu_synid908_src32_2>`::ref:`f16x2<amdgpu_synid908_type_dev>`,   :ref:`src2<amdgpu_synid908_src32_2>`::ref:`f32<amdgpu_synid908_type_dev>`       :ref:`neg_lo<amdgpu_synid_neg_lo>` :ref:`neg_hi<amdgpu_synid_neg_hi>` :ref:`clamp<amdgpu_synid_clamp>`
    v_dot2_i32_i16         :ref:`vdst<amdgpu_synid908_vdst32_0>`,        :ref:`src0<amdgpu_synid908_src32_1>`::ref:`i16x2<amdgpu_synid908_type_dev>`,   :ref:`src1<amdgpu_synid908_src32_2>`::ref:`i16x2<amdgpu_synid908_type_dev>`,   :ref:`src2<amdgpu_synid908_src32_2>`::ref:`i32<amdgpu_synid908_type_dev>`       :ref:`clamp<amdgpu_synid_clamp>`
    v_dot2_u32_u16         :ref:`vdst<amdgpu_synid908_vdst32_0>`,        :ref:`src0<amdgpu_synid908_src32_1>`::ref:`u16x2<amdgpu_synid908_type_dev>`,   :ref:`src1<amdgpu_synid908_src32_2>`::ref:`u16x2<amdgpu_synid908_type_dev>`,   :ref:`src2<amdgpu_synid908_src32_2>`::ref:`u32<amdgpu_synid908_type_dev>`       :ref:`clamp<amdgpu_synid_clamp>`
    v_dot4_i32_i8          :ref:`vdst<amdgpu_synid908_vdst32_0>`,        :ref:`src0<amdgpu_synid908_src32_1>`::ref:`i8x2<amdgpu_synid908_type_dev>`,    :ref:`src1<amdgpu_synid908_src32_2>`::ref:`i8x2<amdgpu_synid908_type_dev>`,    :ref:`src2<amdgpu_synid908_src32_2>`::ref:`i32<amdgpu_synid908_type_dev>`       :ref:`clamp<amdgpu_synid_clamp>`
    v_dot4_u32_u8          :ref:`vdst<amdgpu_synid908_vdst32_0>`,        :ref:`src0<amdgpu_synid908_src32_1>`::ref:`u8x2<amdgpu_synid908_type_dev>`,    :ref:`src1<amdgpu_synid908_src32_2>`::ref:`u8x2<amdgpu_synid908_type_dev>`,    :ref:`src2<amdgpu_synid908_src32_2>`::ref:`u32<amdgpu_synid908_type_dev>`       :ref:`clamp<amdgpu_synid_clamp>`
    v_dot8_i32_i4          :ref:`vdst<amdgpu_synid908_vdst32_0>`,        :ref:`src0<amdgpu_synid908_src32_1>`::ref:`i4x2<amdgpu_synid908_type_dev>`,    :ref:`src1<amdgpu_synid908_src32_2>`::ref:`i4x2<amdgpu_synid908_type_dev>`,    :ref:`src2<amdgpu_synid908_src32_2>`::ref:`i32<amdgpu_synid908_type_dev>`       :ref:`clamp<amdgpu_synid_clamp>`
    v_dot8_u32_u4          :ref:`vdst<amdgpu_synid908_vdst32_0>`,        :ref:`src0<amdgpu_synid908_src32_1>`::ref:`u4x2<amdgpu_synid908_type_dev>`,    :ref:`src1<amdgpu_synid908_src32_2>`::ref:`u4x2<amdgpu_synid908_type_dev>`,    :ref:`src2<amdgpu_synid908_src32_2>`::ref:`u32<amdgpu_synid908_type_dev>`       :ref:`clamp<amdgpu_synid_clamp>`
    v_fma_mix_f32          :ref:`vdst<amdgpu_synid908_vdst32_0>`,        :ref:`src0<amdgpu_synid908_src32_1>`::ref:`m<amdgpu_synid908_mod_vop3_abs_neg>`::ref:`fx<amdgpu_synid908_mad_type_dev>`,    :ref:`src1<amdgpu_synid908_src32_2>`::ref:`m<amdgpu_synid908_mod_vop3_abs_neg>`::ref:`fx<amdgpu_synid908_mad_type_dev>`,    :ref:`src2<amdgpu_synid908_src32_2>`::ref:`m<amdgpu_synid908_mod_vop3_abs_neg>`::ref:`fx<amdgpu_synid908_mad_type_dev>`      :ref:`m_op_sel<amdgpu_synid_mad_mix_op_sel>` :ref:`m_op_sel_hi<amdgpu_synid_mad_mix_op_sel_hi>` :ref:`clamp<amdgpu_synid_clamp>`
    v_fma_mixhi_f16        :ref:`vdst<amdgpu_synid908_vdst32_0>`,        :ref:`src0<amdgpu_synid908_src32_1>`::ref:`m<amdgpu_synid908_mod_vop3_abs_neg>`::ref:`fx<amdgpu_synid908_mad_type_dev>`,    :ref:`src1<amdgpu_synid908_src32_2>`::ref:`m<amdgpu_synid908_mod_vop3_abs_neg>`::ref:`fx<amdgpu_synid908_mad_type_dev>`,    :ref:`src2<amdgpu_synid908_src32_2>`::ref:`m<amdgpu_synid908_mod_vop3_abs_neg>`::ref:`fx<amdgpu_synid908_mad_type_dev>`      :ref:`m_op_sel<amdgpu_synid_mad_mix_op_sel>` :ref:`m_op_sel_hi<amdgpu_synid_mad_mix_op_sel_hi>` :ref:`clamp<amdgpu_synid_clamp>`
    v_fma_mixlo_f16        :ref:`vdst<amdgpu_synid908_vdst32_0>`,        :ref:`src0<amdgpu_synid908_src32_1>`::ref:`m<amdgpu_synid908_mod_vop3_abs_neg>`::ref:`fx<amdgpu_synid908_mad_type_dev>`,    :ref:`src1<amdgpu_synid908_src32_2>`::ref:`m<amdgpu_synid908_mod_vop3_abs_neg>`::ref:`fx<amdgpu_synid908_mad_type_dev>`,    :ref:`src2<amdgpu_synid908_src32_2>`::ref:`m<amdgpu_synid908_mod_vop3_abs_neg>`::ref:`fx<amdgpu_synid908_mad_type_dev>`      :ref:`m_op_sel<amdgpu_synid_mad_mix_op_sel>` :ref:`m_op_sel_hi<amdgpu_synid_mad_mix_op_sel_hi>` :ref:`clamp<amdgpu_synid_clamp>`
    v_mfma_f32_16x16x16f16 :ref:`adst<amdgpu_synid908_adst128_0>`::ref:`f32x4<amdgpu_synid908_type_dev>`,  :ref:`vasrc0<amdgpu_synid908_vasrc64_0>`::ref:`f16x4<amdgpu_synid908_type_dev>`, :ref:`vasrc1<amdgpu_synid908_vasrc64_0>`::ref:`f16x4<amdgpu_synid908_type_dev>`, :ref:`asrc2<amdgpu_synid908_asrc128_0>`::ref:`f32x4<amdgpu_synid908_type_dev>`    :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_f32_16x16x1f32  :ref:`adst<amdgpu_synid908_adst512_0>`::ref:`f32x16<amdgpu_synid908_type_dev>`, :ref:`vasrc0<amdgpu_synid908_vasrc32_0>`::ref:`f32<amdgpu_synid908_type_dev>`,   :ref:`vasrc1<amdgpu_synid908_vasrc32_0>`::ref:`f32<amdgpu_synid908_type_dev>`,   :ref:`asrc2<amdgpu_synid908_asrc512_0>`::ref:`f32x16<amdgpu_synid908_type_dev>`   :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_f32_16x16x2bf16 :ref:`adst<amdgpu_synid908_adst512_0>`::ref:`f32x16<amdgpu_synid908_type_dev>`, :ref:`vasrc0<amdgpu_synid908_vasrc32_0>`::ref:`i16x2<amdgpu_synid908_type_dev>`, :ref:`vasrc1<amdgpu_synid908_vasrc32_0>`::ref:`i16x2<amdgpu_synid908_type_dev>`, :ref:`asrc2<amdgpu_synid908_asrc512_0>`::ref:`f32x16<amdgpu_synid908_type_dev>`   :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_f32_16x16x4f16  :ref:`adst<amdgpu_synid908_adst512_0>`::ref:`f32x16<amdgpu_synid908_type_dev>`, :ref:`vasrc0<amdgpu_synid908_vasrc64_0>`::ref:`f16x4<amdgpu_synid908_type_dev>`, :ref:`vasrc1<amdgpu_synid908_vasrc64_0>`::ref:`f16x4<amdgpu_synid908_type_dev>`, :ref:`asrc2<amdgpu_synid908_asrc512_0>`::ref:`f32x16<amdgpu_synid908_type_dev>`   :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_f32_16x16x4f32  :ref:`adst<amdgpu_synid908_adst128_0>`::ref:`f32x4<amdgpu_synid908_type_dev>`,  :ref:`vasrc0<amdgpu_synid908_vasrc32_0>`::ref:`f32<amdgpu_synid908_type_dev>`,   :ref:`vasrc1<amdgpu_synid908_vasrc32_0>`::ref:`f32<amdgpu_synid908_type_dev>`,   :ref:`asrc2<amdgpu_synid908_asrc128_0>`::ref:`f32x4<amdgpu_synid908_type_dev>`    :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_f32_16x16x8bf16 :ref:`adst<amdgpu_synid908_adst128_0>`::ref:`f32x4<amdgpu_synid908_type_dev>`,  :ref:`vasrc0<amdgpu_synid908_vasrc32_0>`::ref:`i16x2<amdgpu_synid908_type_dev>`, :ref:`vasrc1<amdgpu_synid908_vasrc32_0>`::ref:`i16x2<amdgpu_synid908_type_dev>`, :ref:`asrc2<amdgpu_synid908_asrc128_0>`::ref:`f32x4<amdgpu_synid908_type_dev>`    :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_f32_32x32x1f32  :ref:`adst<amdgpu_synid908_adst1024_0>`::ref:`f32x32<amdgpu_synid908_type_dev>`, :ref:`vasrc0<amdgpu_synid908_vasrc32_0>`::ref:`f32<amdgpu_synid908_type_dev>`,   :ref:`vasrc1<amdgpu_synid908_vasrc32_0>`::ref:`f32<amdgpu_synid908_type_dev>`,   :ref:`asrc2<amdgpu_synid908_asrc1024_0>`::ref:`f32x32<amdgpu_synid908_type_dev>`   :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_f32_32x32x2bf16 :ref:`adst<amdgpu_synid908_adst1024_0>`::ref:`f32x32<amdgpu_synid908_type_dev>`, :ref:`vasrc0<amdgpu_synid908_vasrc32_0>`::ref:`i16x2<amdgpu_synid908_type_dev>`, :ref:`vasrc1<amdgpu_synid908_vasrc32_0>`::ref:`i16x2<amdgpu_synid908_type_dev>`, :ref:`asrc2<amdgpu_synid908_asrc1024_0>`::ref:`f32x32<amdgpu_synid908_type_dev>`   :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_f32_32x32x2f32  :ref:`adst<amdgpu_synid908_adst512_0>`::ref:`f32x16<amdgpu_synid908_type_dev>`, :ref:`vasrc0<amdgpu_synid908_vasrc32_0>`::ref:`f32<amdgpu_synid908_type_dev>`,   :ref:`vasrc1<amdgpu_synid908_vasrc32_0>`::ref:`f32<amdgpu_synid908_type_dev>`,   :ref:`asrc2<amdgpu_synid908_asrc512_0>`::ref:`f32x16<amdgpu_synid908_type_dev>`   :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_f32_32x32x4bf16 :ref:`adst<amdgpu_synid908_adst512_0>`::ref:`f32x16<amdgpu_synid908_type_dev>`, :ref:`vasrc0<amdgpu_synid908_vasrc32_0>`::ref:`i16x2<amdgpu_synid908_type_dev>`, :ref:`vasrc1<amdgpu_synid908_vasrc32_0>`::ref:`i16x2<amdgpu_synid908_type_dev>`, :ref:`asrc2<amdgpu_synid908_asrc512_0>`::ref:`f32x16<amdgpu_synid908_type_dev>`   :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_f32_32x32x4f16  :ref:`adst<amdgpu_synid908_adst1024_0>`::ref:`f32x32<amdgpu_synid908_type_dev>`, :ref:`vasrc0<amdgpu_synid908_vasrc64_0>`::ref:`f16x4<amdgpu_synid908_type_dev>`, :ref:`vasrc1<amdgpu_synid908_vasrc64_0>`::ref:`f16x4<amdgpu_synid908_type_dev>`, :ref:`asrc2<amdgpu_synid908_asrc1024_0>`::ref:`f32x32<amdgpu_synid908_type_dev>`   :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_f32_32x32x8f16  :ref:`adst<amdgpu_synid908_adst512_0>`::ref:`f32x16<amdgpu_synid908_type_dev>`, :ref:`vasrc0<amdgpu_synid908_vasrc64_0>`::ref:`f16x4<amdgpu_synid908_type_dev>`, :ref:`vasrc1<amdgpu_synid908_vasrc64_0>`::ref:`f16x4<amdgpu_synid908_type_dev>`, :ref:`asrc2<amdgpu_synid908_asrc512_0>`::ref:`f32x16<amdgpu_synid908_type_dev>`   :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_f32_4x4x1f32    :ref:`adst<amdgpu_synid908_adst128_0>`::ref:`f32x4<amdgpu_synid908_type_dev>`,  :ref:`vasrc0<amdgpu_synid908_vasrc32_0>`::ref:`f32<amdgpu_synid908_type_dev>`,   :ref:`vasrc1<amdgpu_synid908_vasrc32_0>`::ref:`f32<amdgpu_synid908_type_dev>`,   :ref:`asrc2<amdgpu_synid908_asrc128_0>`::ref:`f32x4<amdgpu_synid908_type_dev>`    :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_f32_4x4x2bf16   :ref:`adst<amdgpu_synid908_adst128_0>`::ref:`f32x4<amdgpu_synid908_type_dev>`,  :ref:`vasrc0<amdgpu_synid908_vasrc32_0>`::ref:`i16x2<amdgpu_synid908_type_dev>`, :ref:`vasrc1<amdgpu_synid908_vasrc32_0>`::ref:`i16x2<amdgpu_synid908_type_dev>`, :ref:`asrc2<amdgpu_synid908_asrc128_0>`::ref:`f32x4<amdgpu_synid908_type_dev>`    :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_f32_4x4x4f16    :ref:`adst<amdgpu_synid908_adst128_0>`::ref:`f32x4<amdgpu_synid908_type_dev>`,  :ref:`vasrc0<amdgpu_synid908_vasrc64_0>`::ref:`f16x4<amdgpu_synid908_type_dev>`, :ref:`vasrc1<amdgpu_synid908_vasrc64_0>`::ref:`f16x4<amdgpu_synid908_type_dev>`, :ref:`asrc2<amdgpu_synid908_asrc128_0>`::ref:`f32x4<amdgpu_synid908_type_dev>`    :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_i32_16x16x16i8  :ref:`adst<amdgpu_synid908_adst128_0>`::ref:`i32x4<amdgpu_synid908_type_dev>`,  :ref:`vasrc0<amdgpu_synid908_vasrc32_0>`::ref:`i32<amdgpu_synid908_type_dev>`,   :ref:`vasrc1<amdgpu_synid908_vasrc32_0>`::ref:`i32<amdgpu_synid908_type_dev>`,   :ref:`asrc2<amdgpu_synid908_asrc128_0>`::ref:`i32x4<amdgpu_synid908_type_dev>`    :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_i32_16x16x4i8   :ref:`adst<amdgpu_synid908_adst512_0>`::ref:`i32x16<amdgpu_synid908_type_dev>`, :ref:`vasrc0<amdgpu_synid908_vasrc32_0>`::ref:`i32<amdgpu_synid908_type_dev>`,   :ref:`vasrc1<amdgpu_synid908_vasrc32_0>`::ref:`i32<amdgpu_synid908_type_dev>`,   :ref:`asrc2<amdgpu_synid908_asrc512_0>`::ref:`i32x16<amdgpu_synid908_type_dev>`   :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_i32_32x32x4i8   :ref:`adst<amdgpu_synid908_adst1024_0>`::ref:`i32x32<amdgpu_synid908_type_dev>`, :ref:`vasrc0<amdgpu_synid908_vasrc32_0>`::ref:`i32<amdgpu_synid908_type_dev>`,   :ref:`vasrc1<amdgpu_synid908_vasrc32_0>`::ref:`i32<amdgpu_synid908_type_dev>`,   :ref:`asrc2<amdgpu_synid908_asrc1024_0>`::ref:`i32x32<amdgpu_synid908_type_dev>`   :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_i32_32x32x8i8   :ref:`adst<amdgpu_synid908_adst512_0>`::ref:`i32x16<amdgpu_synid908_type_dev>`, :ref:`vasrc0<amdgpu_synid908_vasrc32_0>`::ref:`i32<amdgpu_synid908_type_dev>`,   :ref:`vasrc1<amdgpu_synid908_vasrc32_0>`::ref:`i32<amdgpu_synid908_type_dev>`,   :ref:`asrc2<amdgpu_synid908_asrc512_0>`::ref:`i32x16<amdgpu_synid908_type_dev>`   :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`
    v_mfma_i32_4x4x4i8     :ref:`adst<amdgpu_synid908_adst128_0>`::ref:`i32x4<amdgpu_synid908_type_dev>`,  :ref:`vasrc0<amdgpu_synid908_vasrc32_0>`::ref:`i32<amdgpu_synid908_type_dev>`,   :ref:`vasrc1<amdgpu_synid908_vasrc32_0>`::ref:`i32<amdgpu_synid908_type_dev>`,   :ref:`asrc2<amdgpu_synid908_asrc128_0>`::ref:`i32x4<amdgpu_synid908_type_dev>`    :ref:`cbsz<amdgpu_synid_cbsz>` :ref:`abid<amdgpu_synid_abid>` :ref:`blgp<amdgpu_synid_blgp>`

.. |---| unicode:: U+02014 .. em dash


.. toctree::
    :hidden:

    AMDGPUAsmGFX9
    gfx908_addr_buf
    gfx908_adst1024_0
    gfx908_adst128_0
    gfx908_adst32_0
    gfx908_adst512_0
    gfx908_asrc1024_0
    gfx908_asrc128_0
    gfx908_asrc32_0
    gfx908_asrc512_0
    gfx908_data_buf_atomic32
    gfx908_dst_flat_atomic32
    gfx908_offset_buf
    gfx908_rsrc_buf
    gfx908_saddr_flat_global
    gfx908_src32_0
    gfx908_src32_1
    gfx908_src32_2
    gfx908_src32_3
    gfx908_vaddr_flat_global
    gfx908_vasrc32_0
    gfx908_vasrc64_0
    gfx908_vdata32_0
    gfx908_vdst32_0
    gfx908_vsrc32_0
    gfx908_mad_type_dev
    gfx908_mod_dpp_sdwa_abs_neg
    gfx908_mod_sdwa_sext
    gfx908_mod_vop3_abs_neg
    gfx908_opt
    gfx908_ret
    gfx908_type_dev
