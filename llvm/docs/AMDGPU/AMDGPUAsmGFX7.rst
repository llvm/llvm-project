..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

====================================================================================
Syntax of GFX7 Instructions
====================================================================================

.. contents::
  :local:

Introduction
============

This document describes the syntax of GFX7 instructions.

Notation
========

Notation used in this document is explained :ref:`here<amdgpu_syn_instruction_notation>`.

Overview
========

An overview of generic syntax and other features of AMDGPU instructions may be found :ref:`in this document<amdgpu_syn_instructions>`.

Instructions
============


DS
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**         **SRC0**      **SRC1**      **SRC2**           **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    ds_add_rtn_u32                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_add_rtn_u64                 :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_add_src2_u32                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_add_src2_u64                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_add_u32                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_add_u64                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_and_b32                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_and_b64                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_and_rtn_b32                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_and_rtn_b64                 :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_and_src2_b32                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_and_src2_b64                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_append                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`                                           :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_b32                               :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1>`         :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_b64                               :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0_1>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1_1>`         :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_f32                               :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1>`         :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_f64                               :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0_1>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1_1>`         :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_rtn_b32               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1>`         :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_rtn_b64               :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0_1>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1_1>`         :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_rtn_f32               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1>`         :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_rtn_f64               :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0_1>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1_1>`         :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_condxchg32_rtn_b64          :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_consume                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`                                           :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_dec_rtn_u32                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_dec_rtn_u64                 :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_dec_src2_u32                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_dec_src2_u64                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_dec_u32                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_dec_u64                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_gws_barrier                             :ref:`vdata<amdgpu_synid_gfx7_vdata>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_gws_init                                :ref:`vdata<amdgpu_synid_gfx7_vdata>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_gws_sema_br                             :ref:`vdata<amdgpu_synid_gfx7_vdata>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_gws_sema_p                                                                 :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_gws_sema_release_all                                                       :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_gws_sema_v                                                                 :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_inc_rtn_u32                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_inc_rtn_u64                 :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_inc_src2_u32                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_inc_src2_u64                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_inc_u32                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_inc_u64                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_f32                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_f64                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_i32                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_i64                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_rtn_f32                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_rtn_f64                 :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_rtn_i32                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_rtn_i64                 :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_rtn_u32                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_rtn_u64                 :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_src2_f32                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_src2_f64                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_src2_i32                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_src2_i64                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_src2_u32                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_src2_u64                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_u32                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_u64                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_f32                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_f64                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_i32                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_i64                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_rtn_f32                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_rtn_f64                 :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_rtn_i32                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_rtn_i64                 :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_rtn_u32                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_rtn_u64                 :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_src2_f32                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_src2_f64                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_src2_i32                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_src2_i64                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_src2_u32                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_src2_u64                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_u32                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_u64                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_mskor_b32                               :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1>`         :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_mskor_b64                               :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0_1>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1_1>`         :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_mskor_rtn_b32               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1>`         :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_mskor_rtn_b64               :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0_1>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1_1>`         :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_nop
    ds_or_b32                                  :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_or_b64                                  :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_or_rtn_b32                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_or_rtn_b64                  :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_or_src2_b32                             :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_or_src2_b64                             :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_ordered_count               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read2_b32                   :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`::ref:`b32x2<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset0<amdgpu_synid_ds_offset80>` :ref:`offset1<amdgpu_synid_ds_offset81>` :ref:`gds<amdgpu_synid_gds>`
    ds_read2_b64                   :ref:`vdst<amdgpu_synid_gfx7_vdst_2>`::ref:`b64x2<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset0<amdgpu_synid_ds_offset80>` :ref:`offset1<amdgpu_synid_ds_offset81>` :ref:`gds<amdgpu_synid_gds>`
    ds_read2st64_b32               :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`::ref:`b32x2<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset0<amdgpu_synid_ds_offset80>` :ref:`offset1<amdgpu_synid_ds_offset81>` :ref:`gds<amdgpu_synid_gds>`
    ds_read2st64_b64               :ref:`vdst<amdgpu_synid_gfx7_vdst_2>`::ref:`b64x2<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset0<amdgpu_synid_ds_offset80>` :ref:`offset1<amdgpu_synid_ds_offset81>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_b128                   :ref:`vdst<amdgpu_synid_gfx7_vdst_2>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_b32                    :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_b64                    :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_b96                    :ref:`vdst<amdgpu_synid_gfx7_vdst_3>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_i16                    :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_i8                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_u16                    :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_u8                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_rsub_rtn_u32                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_rsub_rtn_u64                :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_rsub_src2_u32                           :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_rsub_src2_u64                           :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_rsub_u32                                :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_rsub_u64                                :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_sub_rtn_u32                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_sub_rtn_u64                 :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_sub_src2_u32                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_sub_src2_u64                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_sub_u32                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_sub_u64                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_swizzle_b32                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`pattern<amdgpu_synid_sw_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrap_rtn_b32                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1>`         :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write2_b32                              :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1>`         :ref:`offset0<amdgpu_synid_ds_offset80>` :ref:`offset1<amdgpu_synid_ds_offset81>` :ref:`gds<amdgpu_synid_gds>`
    ds_write2_b64                              :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0_1>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1_1>`         :ref:`offset0<amdgpu_synid_ds_offset80>` :ref:`offset1<amdgpu_synid_ds_offset81>` :ref:`gds<amdgpu_synid_gds>`
    ds_write2st64_b32                          :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1>`         :ref:`offset0<amdgpu_synid_ds_offset80>` :ref:`offset1<amdgpu_synid_ds_offset81>` :ref:`gds<amdgpu_synid_gds>`
    ds_write2st64_b64                          :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0_1>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1_1>`         :ref:`offset0<amdgpu_synid_ds_offset80>` :ref:`offset1<amdgpu_synid_ds_offset81>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_b128                              :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_2>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_b16                               :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_b32                               :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_b64                               :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_b8                                :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_b96                               :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_3>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_src2_b32                          :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_src2_b64                          :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrxchg2_rtn_b32             :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`::ref:`b32x2<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1>`         :ref:`offset0<amdgpu_synid_ds_offset80>` :ref:`offset1<amdgpu_synid_ds_offset81>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrxchg2_rtn_b64             :ref:`vdst<amdgpu_synid_gfx7_vdst_2>`::ref:`b64x2<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0_1>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1_1>`         :ref:`offset0<amdgpu_synid_ds_offset80>` :ref:`offset1<amdgpu_synid_ds_offset81>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrxchg2st64_rtn_b32         :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`::ref:`b32x2<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1>`         :ref:`offset0<amdgpu_synid_ds_offset80>` :ref:`offset1<amdgpu_synid_ds_offset81>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrxchg2st64_rtn_b64         :ref:`vdst<amdgpu_synid_gfx7_vdst_2>`::ref:`b64x2<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata0<amdgpu_synid_gfx7_vdata0_1>`,   :ref:`vdata1<amdgpu_synid_gfx7_vdata1_1>`         :ref:`offset0<amdgpu_synid_ds_offset80>` :ref:`offset1<amdgpu_synid_ds_offset81>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrxchg_rtn_b32              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrxchg_rtn_b64              :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_xor_b32                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_xor_b64                                 :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_xor_rtn_b32                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_xor_rtn_b64                 :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`                    :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_xor_src2_b32                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_xor_src2_b64                            :ref:`vaddr<amdgpu_synid_gfx7_vaddr>`                              :ref:`offset<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`

EXP
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC0**      **SRC1**      **SRC2**      **SRC3**           **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    exp                            :ref:`tgt<amdgpu_synid_gfx7_tgt>`,      :ref:`vsrc0<amdgpu_synid_gfx7_vsrc>`,    :ref:`vsrc1<amdgpu_synid_gfx7_vsrc>`,    :ref:`vsrc2<amdgpu_synid_gfx7_vsrc>`,    :ref:`vsrc3<amdgpu_synid_gfx7_vsrc>`          :ref:`done<amdgpu_synid_done>` :ref:`compr<amdgpu_synid_compr>` :ref:`vm<amdgpu_synid_vm>`

FLAT
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**           **SRC0**      **SRC1**             **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    flat_atomic_add                :ref:`vdst<amdgpu_synid_gfx7_vdst_4>`::ref:`opt<amdgpu_synid_gfx7_opt>`,     :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_add_x2             :ref:`vdst<amdgpu_synid_gfx7_vdst_5>`::ref:`opt<amdgpu_synid_gfx7_opt>`,     :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_and                :ref:`vdst<amdgpu_synid_gfx7_vdst_4>`::ref:`opt<amdgpu_synid_gfx7_opt>`,     :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_and_x2             :ref:`vdst<amdgpu_synid_gfx7_vdst_5>`::ref:`opt<amdgpu_synid_gfx7_opt>`,     :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_cmpswap            :ref:`vdst<amdgpu_synid_gfx7_vdst_4>`::ref:`opt<amdgpu_synid_gfx7_opt>`,     :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`::ref:`b32x2<amdgpu_synid_gfx7_type_deviation>`      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_cmpswap_x2         :ref:`vdst<amdgpu_synid_gfx7_vdst_5>`::ref:`opt<amdgpu_synid_gfx7_opt>`,     :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_2>`::ref:`b64x2<amdgpu_synid_gfx7_type_deviation>`      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_dec                :ref:`vdst<amdgpu_synid_gfx7_vdst_4>`::ref:`opt<amdgpu_synid_gfx7_opt>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_dec_x2             :ref:`vdst<amdgpu_synid_gfx7_vdst_5>`::ref:`opt<amdgpu_synid_gfx7_opt>`::ref:`u64<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`::ref:`u64<amdgpu_synid_gfx7_type_deviation>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_fcmpswap           :ref:`vdst<amdgpu_synid_gfx7_vdst_4>`::ref:`opt<amdgpu_synid_gfx7_opt>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`::ref:`f32x2<amdgpu_synid_gfx7_type_deviation>`      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_fcmpswap_x2        :ref:`vdst<amdgpu_synid_gfx7_vdst_5>`::ref:`opt<amdgpu_synid_gfx7_opt>`::ref:`f64<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_2>`::ref:`f64x2<amdgpu_synid_gfx7_type_deviation>`      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_fmax               :ref:`vdst<amdgpu_synid_gfx7_vdst_4>`::ref:`opt<amdgpu_synid_gfx7_opt>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_fmax_x2            :ref:`vdst<amdgpu_synid_gfx7_vdst_5>`::ref:`opt<amdgpu_synid_gfx7_opt>`::ref:`f64<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`::ref:`f64<amdgpu_synid_gfx7_type_deviation>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_fmin               :ref:`vdst<amdgpu_synid_gfx7_vdst_4>`::ref:`opt<amdgpu_synid_gfx7_opt>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_fmin_x2            :ref:`vdst<amdgpu_synid_gfx7_vdst_5>`::ref:`opt<amdgpu_synid_gfx7_opt>`::ref:`f64<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`::ref:`f64<amdgpu_synid_gfx7_type_deviation>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_inc                :ref:`vdst<amdgpu_synid_gfx7_vdst_4>`::ref:`opt<amdgpu_synid_gfx7_opt>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_inc_x2             :ref:`vdst<amdgpu_synid_gfx7_vdst_5>`::ref:`opt<amdgpu_synid_gfx7_opt>`::ref:`u64<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`::ref:`u64<amdgpu_synid_gfx7_type_deviation>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_or                 :ref:`vdst<amdgpu_synid_gfx7_vdst_4>`::ref:`opt<amdgpu_synid_gfx7_opt>`,     :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_or_x2              :ref:`vdst<amdgpu_synid_gfx7_vdst_5>`::ref:`opt<amdgpu_synid_gfx7_opt>`,     :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_smax               :ref:`vdst<amdgpu_synid_gfx7_vdst_4>`::ref:`opt<amdgpu_synid_gfx7_opt>`::ref:`i32<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`::ref:`i32<amdgpu_synid_gfx7_type_deviation>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_smax_x2            :ref:`vdst<amdgpu_synid_gfx7_vdst_5>`::ref:`opt<amdgpu_synid_gfx7_opt>`::ref:`i64<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`::ref:`i64<amdgpu_synid_gfx7_type_deviation>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_smin               :ref:`vdst<amdgpu_synid_gfx7_vdst_4>`::ref:`opt<amdgpu_synid_gfx7_opt>`::ref:`i32<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`::ref:`i32<amdgpu_synid_gfx7_type_deviation>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_smin_x2            :ref:`vdst<amdgpu_synid_gfx7_vdst_5>`::ref:`opt<amdgpu_synid_gfx7_opt>`::ref:`i64<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`::ref:`i64<amdgpu_synid_gfx7_type_deviation>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_sub                :ref:`vdst<amdgpu_synid_gfx7_vdst_4>`::ref:`opt<amdgpu_synid_gfx7_opt>`,     :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_sub_x2             :ref:`vdst<amdgpu_synid_gfx7_vdst_5>`::ref:`opt<amdgpu_synid_gfx7_opt>`,     :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_swap               :ref:`vdst<amdgpu_synid_gfx7_vdst_4>`::ref:`opt<amdgpu_synid_gfx7_opt>`,     :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_swap_x2            :ref:`vdst<amdgpu_synid_gfx7_vdst_5>`::ref:`opt<amdgpu_synid_gfx7_opt>`,     :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_umax               :ref:`vdst<amdgpu_synid_gfx7_vdst_4>`::ref:`opt<amdgpu_synid_gfx7_opt>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_umax_x2            :ref:`vdst<amdgpu_synid_gfx7_vdst_5>`::ref:`opt<amdgpu_synid_gfx7_opt>`::ref:`u64<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`::ref:`u64<amdgpu_synid_gfx7_type_deviation>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_umin               :ref:`vdst<amdgpu_synid_gfx7_vdst_4>`::ref:`opt<amdgpu_synid_gfx7_opt>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_umin_x2            :ref:`vdst<amdgpu_synid_gfx7_vdst_5>`::ref:`opt<amdgpu_synid_gfx7_opt>`::ref:`u64<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`::ref:`u64<amdgpu_synid_gfx7_type_deviation>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_xor                :ref:`vdst<amdgpu_synid_gfx7_vdst_4>`::ref:`opt<amdgpu_synid_gfx7_opt>`,     :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_xor_x2             :ref:`vdst<amdgpu_synid_gfx7_vdst_5>`::ref:`opt<amdgpu_synid_gfx7_opt>`,     :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_dword                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,         :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_dwordx2              :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,         :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_dwordx3              :ref:`vdst<amdgpu_synid_gfx7_vdst_3>`,         :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_dwordx4              :ref:`vdst<amdgpu_synid_gfx7_vdst_2>`,         :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_sbyte                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,         :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_sshort               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,         :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_ubyte                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,         :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_ushort               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,         :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_store_byte                              :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_store_dword                             :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_store_dwordx2                           :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_store_dwordx3                           :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_3>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_store_dwordx4                           :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata_2>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_store_short                             :ref:`vaddr<amdgpu_synid_gfx7_vaddr_1>`,    :ref:`vdata<amdgpu_synid_gfx7_vdata>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`

MIMG
-----------------------

.. parsed-literal::

    **INSTRUCTION**                **DST**      **SRC0**       **SRC1**     **SRC2**          **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    image_atomic_add                    :ref:`vdata<amdgpu_synid_gfx7_vdata_4>`::ref:`dst<amdgpu_synid_gfx7_dst>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_and                    :ref:`vdata<amdgpu_synid_gfx7_vdata_4>`::ref:`dst<amdgpu_synid_gfx7_dst>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_cmpswap                :ref:`vdata<amdgpu_synid_gfx7_vdata_5>`::ref:`dst<amdgpu_synid_gfx7_dst>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_dec                    :ref:`vdata<amdgpu_synid_gfx7_vdata_4>`::ref:`dst<amdgpu_synid_gfx7_dst>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_fcmpswap               :ref:`vdata<amdgpu_synid_gfx7_vdata_5>`::ref:`dst<amdgpu_synid_gfx7_dst>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_fmax                   :ref:`vdata<amdgpu_synid_gfx7_vdata_4>`::ref:`dst<amdgpu_synid_gfx7_dst>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_fmin                   :ref:`vdata<amdgpu_synid_gfx7_vdata_4>`::ref:`dst<amdgpu_synid_gfx7_dst>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_inc                    :ref:`vdata<amdgpu_synid_gfx7_vdata_4>`::ref:`dst<amdgpu_synid_gfx7_dst>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_or                     :ref:`vdata<amdgpu_synid_gfx7_vdata_4>`::ref:`dst<amdgpu_synid_gfx7_dst>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_smax                   :ref:`vdata<amdgpu_synid_gfx7_vdata_4>`::ref:`dst<amdgpu_synid_gfx7_dst>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_smin                   :ref:`vdata<amdgpu_synid_gfx7_vdata_4>`::ref:`dst<amdgpu_synid_gfx7_dst>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_sub                    :ref:`vdata<amdgpu_synid_gfx7_vdata_4>`::ref:`dst<amdgpu_synid_gfx7_dst>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_swap                   :ref:`vdata<amdgpu_synid_gfx7_vdata_4>`::ref:`dst<amdgpu_synid_gfx7_dst>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_umax                   :ref:`vdata<amdgpu_synid_gfx7_vdata_4>`::ref:`dst<amdgpu_synid_gfx7_dst>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_umin                   :ref:`vdata<amdgpu_synid_gfx7_vdata_4>`::ref:`dst<amdgpu_synid_gfx7_dst>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_xor                    :ref:`vdata<amdgpu_synid_gfx7_vdata_4>`::ref:`dst<amdgpu_synid_gfx7_dst>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4              :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_b            :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_b_cl         :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_b_cl_o       :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_b_o          :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c            :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_b          :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_b_cl       :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_b_cl_o     :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_b_o        :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_cl         :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_cl_o       :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_l          :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_l_o        :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_lz         :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_lz_o       :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_o          :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_cl           :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_cl_o         :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_l            :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_l_o          :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_lz           :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_lz_o         :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_o            :ref:`vdst<amdgpu_synid_gfx7_vdst_6>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_get_lod              :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_get_resinfo          :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`                  :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_load                 :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`                  :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_load_mip             :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`                  :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_load_mip_pck         :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`                  :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_load_mip_pck_sgn     :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`                  :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_load_pck             :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`                  :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_load_pck_sgn         :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`                  :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample               :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_b             :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_b_cl          :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_b_cl_o        :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_b_o           :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c             :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_b           :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_b_cl        :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_b_cl_o      :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_b_o         :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_cd          :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_cd_cl       :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_cd_cl_o     :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_cd_o        :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_cl          :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_cl_o        :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_d           :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_d_cl        :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_d_cl_o      :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_d_o         :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_l           :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_l_o         :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_lz          :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_lz_o        :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_o           :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_cd            :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_cd_cl         :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_cd_cl_o       :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_cd_o          :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_cl            :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_cl_o          :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_d             :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_d_cl          :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_d_cl_o        :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_d_o           :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_l             :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_l_o           :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_lz            :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_lz_o          :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_o             :ref:`vdst<amdgpu_synid_gfx7_vdst_7>`,    :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,     :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`,   :ref:`ssamp<amdgpu_synid_gfx7_ssamp>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_store                         :ref:`vdata<amdgpu_synid_gfx7_vdata_6>`,     :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_store_mip                     :ref:`vdata<amdgpu_synid_gfx7_vdata_6>`,     :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_store_mip_pck                 :ref:`vdata<amdgpu_synid_gfx7_vdata_6>`,     :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_store_pck                     :ref:`vdata<amdgpu_synid_gfx7_vdata_6>`,     :ref:`vaddr<amdgpu_synid_gfx7_vaddr_2>`,   :ref:`srsrc<amdgpu_synid_gfx7_srsrc>`         :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`

MTBUF
-----------------------

.. parsed-literal::

    **INSTRUCTION**                **DST**   **SRC0**   **SRC1**   **SRC2**    **SRC3**     **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    tbuffer_load_format_x      :ref:`vdst<amdgpu_synid_gfx7_vdst_8>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`, :ref:`soffset<amdgpu_synid_gfx7_soffset>`          :ref:`fmt<amdgpu_synid_fmt>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    tbuffer_load_format_xy     :ref:`vdst<amdgpu_synid_gfx7_vdst_9>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`, :ref:`soffset<amdgpu_synid_gfx7_soffset>`          :ref:`fmt<amdgpu_synid_fmt>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    tbuffer_load_format_xyz    :ref:`vdst<amdgpu_synid_gfx7_vdst_10>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`, :ref:`soffset<amdgpu_synid_gfx7_soffset>`          :ref:`fmt<amdgpu_synid_fmt>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    tbuffer_load_format_xyzw   :ref:`vdst<amdgpu_synid_gfx7_vdst_11>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`, :ref:`soffset<amdgpu_synid_gfx7_soffset>`          :ref:`fmt<amdgpu_synid_fmt>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    tbuffer_store_format_x           :ref:`vdata<amdgpu_synid_gfx7_vdata>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`fmt<amdgpu_synid_fmt>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    tbuffer_store_format_xy          :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`fmt<amdgpu_synid_fmt>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    tbuffer_store_format_xyz         :ref:`vdata<amdgpu_synid_gfx7_vdata_3>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`fmt<amdgpu_synid_fmt>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    tbuffer_store_format_xyzw        :ref:`vdata<amdgpu_synid_gfx7_vdata_2>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`fmt<amdgpu_synid_fmt>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`

MUBUF
-----------------------

.. parsed-literal::

    **INSTRUCTION**                **DST**   **SRC0**             **SRC1**   **SRC2**    **SRC3**     **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    buffer_atomic_add                :ref:`vdata<amdgpu_synid_gfx7_vdata_7>`::ref:`dst<amdgpu_synid_gfx7_dst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_add_x2             :ref:`vdata<amdgpu_synid_gfx7_vdata_8>`::ref:`dst<amdgpu_synid_gfx7_dst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_and                :ref:`vdata<amdgpu_synid_gfx7_vdata_7>`::ref:`dst<amdgpu_synid_gfx7_dst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_and_x2             :ref:`vdata<amdgpu_synid_gfx7_vdata_8>`::ref:`dst<amdgpu_synid_gfx7_dst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_cmpswap            :ref:`vdata<amdgpu_synid_gfx7_vdata_8>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`b32x2<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_cmpswap_x2         :ref:`vdata<amdgpu_synid_gfx7_vdata_9>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`b64x2<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_dec                :ref:`vdata<amdgpu_synid_gfx7_vdata_7>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`,   :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_dec_x2             :ref:`vdata<amdgpu_synid_gfx7_vdata_8>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`u64<amdgpu_synid_gfx7_type_deviation>`,   :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_fcmpswap           :ref:`vdata<amdgpu_synid_gfx7_vdata_8>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`f32x2<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_fcmpswap_x2        :ref:`vdata<amdgpu_synid_gfx7_vdata_9>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`f64x2<amdgpu_synid_gfx7_type_deviation>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_fmax               :ref:`vdata<amdgpu_synid_gfx7_vdata_7>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`,   :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_fmax_x2            :ref:`vdata<amdgpu_synid_gfx7_vdata_8>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`f64<amdgpu_synid_gfx7_type_deviation>`,   :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_fmin               :ref:`vdata<amdgpu_synid_gfx7_vdata_7>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`,   :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_fmin_x2            :ref:`vdata<amdgpu_synid_gfx7_vdata_8>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`f64<amdgpu_synid_gfx7_type_deviation>`,   :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_inc                :ref:`vdata<amdgpu_synid_gfx7_vdata_7>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`,   :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_inc_x2             :ref:`vdata<amdgpu_synid_gfx7_vdata_8>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`u64<amdgpu_synid_gfx7_type_deviation>`,   :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_or                 :ref:`vdata<amdgpu_synid_gfx7_vdata_7>`::ref:`dst<amdgpu_synid_gfx7_dst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_or_x2              :ref:`vdata<amdgpu_synid_gfx7_vdata_8>`::ref:`dst<amdgpu_synid_gfx7_dst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_smax               :ref:`vdata<amdgpu_synid_gfx7_vdata_7>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`i32<amdgpu_synid_gfx7_type_deviation>`,   :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_smax_x2            :ref:`vdata<amdgpu_synid_gfx7_vdata_8>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`i64<amdgpu_synid_gfx7_type_deviation>`,   :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_smin               :ref:`vdata<amdgpu_synid_gfx7_vdata_7>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`i32<amdgpu_synid_gfx7_type_deviation>`,   :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_smin_x2            :ref:`vdata<amdgpu_synid_gfx7_vdata_8>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`i64<amdgpu_synid_gfx7_type_deviation>`,   :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_sub                :ref:`vdata<amdgpu_synid_gfx7_vdata_7>`::ref:`dst<amdgpu_synid_gfx7_dst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_sub_x2             :ref:`vdata<amdgpu_synid_gfx7_vdata_8>`::ref:`dst<amdgpu_synid_gfx7_dst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_swap               :ref:`vdata<amdgpu_synid_gfx7_vdata_7>`::ref:`dst<amdgpu_synid_gfx7_dst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_swap_x2            :ref:`vdata<amdgpu_synid_gfx7_vdata_8>`::ref:`dst<amdgpu_synid_gfx7_dst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_umax               :ref:`vdata<amdgpu_synid_gfx7_vdata_7>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`,   :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_umax_x2            :ref:`vdata<amdgpu_synid_gfx7_vdata_8>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`u64<amdgpu_synid_gfx7_type_deviation>`,   :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_umin               :ref:`vdata<amdgpu_synid_gfx7_vdata_7>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`,   :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_umin_x2            :ref:`vdata<amdgpu_synid_gfx7_vdata_8>`::ref:`dst<amdgpu_synid_gfx7_dst>`::ref:`u64<amdgpu_synid_gfx7_type_deviation>`,   :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_xor                :ref:`vdata<amdgpu_synid_gfx7_vdata_7>`::ref:`dst<amdgpu_synid_gfx7_dst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_xor_x2             :ref:`vdata<amdgpu_synid_gfx7_vdata_8>`::ref:`dst<amdgpu_synid_gfx7_dst>`,       :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_dword          :ref:`vdst<amdgpu_synid_gfx7_vdst_12>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`,           :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`, :ref:`soffset<amdgpu_synid_gfx7_soffset>`          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lds<amdgpu_synid_lds>`
    buffer_load_dwordx2        :ref:`vdst<amdgpu_synid_gfx7_vdst_9>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`,           :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`, :ref:`soffset<amdgpu_synid_gfx7_soffset>`          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_dwordx3        :ref:`vdst<amdgpu_synid_gfx7_vdst_10>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`,           :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`, :ref:`soffset<amdgpu_synid_gfx7_soffset>`          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_dwordx4        :ref:`vdst<amdgpu_synid_gfx7_vdst_11>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`,           :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`, :ref:`soffset<amdgpu_synid_gfx7_soffset>`          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_format_x       :ref:`vdst<amdgpu_synid_gfx7_vdst_12>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`,           :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`, :ref:`soffset<amdgpu_synid_gfx7_soffset>`          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lds<amdgpu_synid_lds>`
    buffer_load_format_xy      :ref:`vdst<amdgpu_synid_gfx7_vdst_9>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`,           :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`, :ref:`soffset<amdgpu_synid_gfx7_soffset>`          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_format_xyz     :ref:`vdst<amdgpu_synid_gfx7_vdst_10>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`,           :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`, :ref:`soffset<amdgpu_synid_gfx7_soffset>`          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_format_xyzw    :ref:`vdst<amdgpu_synid_gfx7_vdst_11>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`,           :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`, :ref:`soffset<amdgpu_synid_gfx7_soffset>`          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_sbyte          :ref:`vdst<amdgpu_synid_gfx7_vdst_12>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`,           :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`, :ref:`soffset<amdgpu_synid_gfx7_soffset>`          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lds<amdgpu_synid_lds>`
    buffer_load_sshort         :ref:`vdst<amdgpu_synid_gfx7_vdst_12>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`,           :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`, :ref:`soffset<amdgpu_synid_gfx7_soffset>`          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lds<amdgpu_synid_lds>`
    buffer_load_ubyte          :ref:`vdst<amdgpu_synid_gfx7_vdst_12>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`,           :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`, :ref:`soffset<amdgpu_synid_gfx7_soffset>`          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lds<amdgpu_synid_lds>`
    buffer_load_ushort         :ref:`vdst<amdgpu_synid_gfx7_vdst_12>`, :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`,           :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`, :ref:`soffset<amdgpu_synid_gfx7_soffset>`          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lds<amdgpu_synid_lds>`
    buffer_store_byte                :ref:`vdata<amdgpu_synid_gfx7_vdata>`,           :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_dword               :ref:`vdata<amdgpu_synid_gfx7_vdata>`,           :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_dwordx2             :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`,           :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_dwordx3             :ref:`vdata<amdgpu_synid_gfx7_vdata_3>`,           :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_dwordx4             :ref:`vdata<amdgpu_synid_gfx7_vdata_2>`,           :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_format_x            :ref:`vdata<amdgpu_synid_gfx7_vdata>`,           :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_format_xy           :ref:`vdata<amdgpu_synid_gfx7_vdata_1>`,           :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_format_xyz          :ref:`vdata<amdgpu_synid_gfx7_vdata_3>`,           :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_format_xyzw         :ref:`vdata<amdgpu_synid_gfx7_vdata_2>`,           :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_short               :ref:`vdata<amdgpu_synid_gfx7_vdata>`,           :ref:`vaddr<amdgpu_synid_gfx7_vaddr_3>`, :ref:`srsrc<amdgpu_synid_gfx7_srsrc_1>`,  :ref:`soffset<amdgpu_synid_gfx7_soffset>`  :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_wbinvl1
    buffer_wbinvl1_vol

SMRD
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC0**      **SRC1**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    s_buffer_load_dword            :ref:`sdst<amdgpu_synid_gfx7_sdst>`,     :ref:`sbase<amdgpu_synid_gfx7_sbase>`,    :ref:`soffset<amdgpu_synid_gfx7_soffset_1>`
    s_buffer_load_dwordx16         :ref:`sdst<amdgpu_synid_gfx7_sdst_1>`,     :ref:`sbase<amdgpu_synid_gfx7_sbase>`,    :ref:`soffset<amdgpu_synid_gfx7_soffset_1>`
    s_buffer_load_dwordx2          :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`sbase<amdgpu_synid_gfx7_sbase>`,    :ref:`soffset<amdgpu_synid_gfx7_soffset_1>`
    s_buffer_load_dwordx4          :ref:`sdst<amdgpu_synid_gfx7_sdst_3>`,     :ref:`sbase<amdgpu_synid_gfx7_sbase>`,    :ref:`soffset<amdgpu_synid_gfx7_soffset_1>`
    s_buffer_load_dwordx8          :ref:`sdst<amdgpu_synid_gfx7_sdst_4>`,     :ref:`sbase<amdgpu_synid_gfx7_sbase>`,    :ref:`soffset<amdgpu_synid_gfx7_soffset_1>`
    s_dcache_inv
    s_dcache_inv_vol
    s_load_dword                   :ref:`sdst<amdgpu_synid_gfx7_sdst>`,     :ref:`sbase<amdgpu_synid_gfx7_sbase_1>`,    :ref:`soffset<amdgpu_synid_gfx7_soffset_1>`
    s_load_dwordx16                :ref:`sdst<amdgpu_synid_gfx7_sdst_1>`,     :ref:`sbase<amdgpu_synid_gfx7_sbase_1>`,    :ref:`soffset<amdgpu_synid_gfx7_soffset_1>`
    s_load_dwordx2                 :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`sbase<amdgpu_synid_gfx7_sbase_1>`,    :ref:`soffset<amdgpu_synid_gfx7_soffset_1>`
    s_load_dwordx4                 :ref:`sdst<amdgpu_synid_gfx7_sdst_3>`,     :ref:`sbase<amdgpu_synid_gfx7_sbase_1>`,    :ref:`soffset<amdgpu_synid_gfx7_soffset_1>`
    s_load_dwordx8                 :ref:`sdst<amdgpu_synid_gfx7_sdst_4>`,     :ref:`sbase<amdgpu_synid_gfx7_sbase_1>`,    :ref:`soffset<amdgpu_synid_gfx7_soffset_1>`
    s_memtime                      :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`::ref:`b64<amdgpu_synid_gfx7_type_deviation>`

SOP1
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    s_abs_i32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc>`
    s_and_saveexec_b64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_andn2_saveexec_b64           :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_bcnt0_i32_b32                :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc>`
    s_bcnt0_i32_b64                :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_bcnt1_i32_b32                :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc>`
    s_bcnt1_i32_b64                :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_bitset0_b32                  :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc>`
    s_bitset0_b64                  :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`
    s_bitset1_b32                  :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc>`
    s_bitset1_b64                  :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`
    s_brev_b32                     :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc>`
    s_brev_b64                     :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_cbranch_join                           :ref:`ssrc<amdgpu_synid_gfx7_ssrc_2>`
    s_cmov_b32                     :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc>`
    s_cmov_b64                     :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_ff0_i32_b32                  :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc>`
    s_ff0_i32_b64                  :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_ff1_i32_b32                  :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc>`
    s_ff1_i32_b64                  :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_flbit_i32                    :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc>`
    s_flbit_i32_b32                :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc>`
    s_flbit_i32_b64                :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_flbit_i32_i64                :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_getpc_b64                    :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`
    s_mov_b32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc>`
    s_mov_b64                      :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_movreld_b32                  :ref:`sdst<amdgpu_synid_gfx7_sdst>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc>`
    s_movreld_b64                  :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_movrels_b32                  :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_3>`
    s_movrels_b64                  :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_4>`
    s_nand_saveexec_b64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_nor_saveexec_b64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_not_b32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc>`
    s_not_b64                      :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_or_saveexec_b64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_orn2_saveexec_b64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_quadmask_b32                 :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc>`
    s_quadmask_b64                 :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_rfe_b64                                :ref:`ssrc<amdgpu_synid_gfx7_ssrc_4>`
    s_setpc_b64                              :ref:`ssrc<amdgpu_synid_gfx7_ssrc_4>`
    s_sext_i32_i16                 :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_5>`
    s_sext_i32_i8                  :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_5>`
    s_swappc_b64                   :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_4>`
    s_wqm_b32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc>`
    s_wqm_b64                      :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_xnor_saveexec_b64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`
    s_xor_saveexec_b64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`ssrc<amdgpu_synid_gfx7_ssrc_1>`

SOP2
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC0**       **SRC1**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    s_absdiff_i32                  :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_add_i32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_add_u32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_addc_u32                     :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_and_b32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_and_b64                      :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc_1>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc_1>`
    s_andn2_b32                    :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_andn2_b64                    :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc_1>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc_1>`
    s_ashr_i32                     :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    s_ashr_i64                     :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc_1>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    s_bfe_i32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    s_bfe_i64                      :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc_1>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    s_bfe_u32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_bfe_u64                      :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc_1>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    s_bfm_b32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_bfm_b64                      :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`, :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`
    s_cbranch_g_fork                         :ref:`ssrc0<amdgpu_synid_gfx7_ssrc_6>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc_6>`
    s_cselect_b32                  :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_cselect_b64                  :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc_1>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc_1>`
    s_lshl_b32                     :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    s_lshl_b64                     :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc_1>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    s_lshr_b32                     :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    s_lshr_b64                     :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc_1>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    s_max_i32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_max_u32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_min_i32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_min_u32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_mul_i32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_nand_b32                     :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_nand_b64                     :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc_1>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc_1>`
    s_nor_b32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_nor_b64                      :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc_1>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc_1>`
    s_or_b32                       :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_or_b64                       :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc_1>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc_1>`
    s_orn2_b32                     :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_orn2_b64                     :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc_1>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc_1>`
    s_sub_i32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_sub_u32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_subb_u32                     :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_xnor_b32                     :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_xnor_b64                     :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc_1>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc_1>`
    s_xor_b32                      :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_xor_b64                      :ref:`sdst<amdgpu_synid_gfx7_sdst_6>`,     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc_1>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc_1>`

SOPC
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **SRC0**      **SRC1**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    s_bitcmp0_b32                  :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,    :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_bitcmp0_b64                  :ref:`ssrc0<amdgpu_synid_gfx7_ssrc_1>`,    :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    s_bitcmp1_b32                  :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,    :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_bitcmp1_b64                  :ref:`ssrc0<amdgpu_synid_gfx7_ssrc_1>`,    :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    s_cmp_eq_i32                   :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,    :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_cmp_eq_u32                   :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,    :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_cmp_ge_i32                   :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,    :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_cmp_ge_u32                   :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,    :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_cmp_gt_i32                   :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,    :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_cmp_gt_u32                   :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,    :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_cmp_le_i32                   :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,    :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_cmp_le_u32                   :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,    :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_cmp_lg_i32                   :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,    :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_cmp_lg_u32                   :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,    :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_cmp_lt_i32                   :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,    :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_cmp_lt_u32                   :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,    :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`
    s_setvskip                     :ref:`ssrc0<amdgpu_synid_gfx7_ssrc>`,    :ref:`ssrc1<amdgpu_synid_gfx7_ssrc>`

SOPK
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC0**      **SRC1**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    s_addk_i32                     :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`imm16<amdgpu_synid_gfx7_imm16>`
    s_cbranch_i_fork                         :ref:`ssrc<amdgpu_synid_gfx7_ssrc_7>`,     :ref:`label<amdgpu_synid_gfx7_label>`
    s_cmovk_i32                    :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`imm16<amdgpu_synid_gfx7_imm16>`
    s_cmpk_eq_i32                            :ref:`ssrc<amdgpu_synid_gfx7_ssrc_8>`,     :ref:`imm16<amdgpu_synid_gfx7_imm16>`
    s_cmpk_eq_u32                            :ref:`ssrc<amdgpu_synid_gfx7_ssrc_8>`,     :ref:`imm16<amdgpu_synid_gfx7_imm16_1>`
    s_cmpk_ge_i32                            :ref:`ssrc<amdgpu_synid_gfx7_ssrc_8>`,     :ref:`imm16<amdgpu_synid_gfx7_imm16>`
    s_cmpk_ge_u32                            :ref:`ssrc<amdgpu_synid_gfx7_ssrc_8>`,     :ref:`imm16<amdgpu_synid_gfx7_imm16_1>`
    s_cmpk_gt_i32                            :ref:`ssrc<amdgpu_synid_gfx7_ssrc_8>`,     :ref:`imm16<amdgpu_synid_gfx7_imm16>`
    s_cmpk_gt_u32                            :ref:`ssrc<amdgpu_synid_gfx7_ssrc_8>`,     :ref:`imm16<amdgpu_synid_gfx7_imm16_1>`
    s_cmpk_le_i32                            :ref:`ssrc<amdgpu_synid_gfx7_ssrc_8>`,     :ref:`imm16<amdgpu_synid_gfx7_imm16>`
    s_cmpk_le_u32                            :ref:`ssrc<amdgpu_synid_gfx7_ssrc_8>`,     :ref:`imm16<amdgpu_synid_gfx7_imm16_1>`
    s_cmpk_lg_i32                            :ref:`ssrc<amdgpu_synid_gfx7_ssrc_8>`,     :ref:`imm16<amdgpu_synid_gfx7_imm16>`
    s_cmpk_lg_u32                            :ref:`ssrc<amdgpu_synid_gfx7_ssrc_8>`,     :ref:`imm16<amdgpu_synid_gfx7_imm16_1>`
    s_cmpk_lt_i32                            :ref:`ssrc<amdgpu_synid_gfx7_ssrc_8>`,     :ref:`imm16<amdgpu_synid_gfx7_imm16>`
    s_cmpk_lt_u32                            :ref:`ssrc<amdgpu_synid_gfx7_ssrc_8>`,     :ref:`imm16<amdgpu_synid_gfx7_imm16_1>`
    s_getreg_b32                   :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`hwreg<amdgpu_synid_gfx7_hwreg>`
    s_movk_i32                     :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`imm16<amdgpu_synid_gfx7_imm16>`
    s_mulk_i32                     :ref:`sdst<amdgpu_synid_gfx7_sdst_5>`,     :ref:`imm16<amdgpu_synid_gfx7_imm16>`
    s_setreg_b32                   :ref:`hwreg<amdgpu_synid_gfx7_hwreg>`,    :ref:`ssrc<amdgpu_synid_gfx7_ssrc_8>`
    s_setreg_imm32_b32             :ref:`hwreg<amdgpu_synid_gfx7_hwreg>`,    :ref:`simm32<amdgpu_synid_gfx7_simm32>`

SOPP
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **SRC**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    s_barrier
    s_branch                       :ref:`label<amdgpu_synid_gfx7_label>`
    s_cbranch_cdbgsys              :ref:`label<amdgpu_synid_gfx7_label>`
    s_cbranch_cdbgsys_and_user     :ref:`label<amdgpu_synid_gfx7_label>`
    s_cbranch_cdbgsys_or_user      :ref:`label<amdgpu_synid_gfx7_label>`
    s_cbranch_cdbguser             :ref:`label<amdgpu_synid_gfx7_label>`
    s_cbranch_execnz               :ref:`label<amdgpu_synid_gfx7_label>`
    s_cbranch_execz                :ref:`label<amdgpu_synid_gfx7_label>`
    s_cbranch_scc0                 :ref:`label<amdgpu_synid_gfx7_label>`
    s_cbranch_scc1                 :ref:`label<amdgpu_synid_gfx7_label>`
    s_cbranch_vccnz                :ref:`label<amdgpu_synid_gfx7_label>`
    s_cbranch_vccz                 :ref:`label<amdgpu_synid_gfx7_label>`
    s_decperflevel                 :ref:`imm16<amdgpu_synid_gfx7_imm16_2>`
    s_endpgm
    s_icache_inv
    s_incperflevel                 :ref:`imm16<amdgpu_synid_gfx7_imm16_2>`
    s_nop                          :ref:`imm16<amdgpu_synid_gfx7_imm16_2>`
    s_sendmsg                      :ref:`msg<amdgpu_synid_gfx7_msg>`
    s_sendmsghalt                  :ref:`msg<amdgpu_synid_gfx7_msg>`
    s_sethalt                      :ref:`imm16<amdgpu_synid_gfx7_imm16_2>`
    s_setkill                      :ref:`imm16<amdgpu_synid_gfx7_imm16_2>`
    s_setprio                      :ref:`imm16<amdgpu_synid_gfx7_imm16_2>`
    s_sleep                        :ref:`imm16<amdgpu_synid_gfx7_imm16_2>`
    s_trap                         :ref:`imm16<amdgpu_synid_gfx7_imm16_2>`
    s_ttracedata
    s_waitcnt                      :ref:`waitcnt<amdgpu_synid_gfx7_waitcnt>`

VINTRP
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC0**       **SRC1**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    v_interp_mov_f32               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`param<amdgpu_synid_gfx7_param>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`, :ref:`attr<amdgpu_synid_gfx7_attr>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`
    v_interp_p1_f32                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`vsrc<amdgpu_synid_gfx7_vsrc_1>`,      :ref:`attr<amdgpu_synid_gfx7_attr>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`
    v_interp_p2_f32                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`vsrc<amdgpu_synid_gfx7_vsrc_1>`,      :ref:`attr<amdgpu_synid_gfx7_attr>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`

VOP1
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    v_bfrev_b32                    :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_ceil_f32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_ceil_f64                     :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,     :ref:`src<amdgpu_synid_gfx7_src_1>`
    v_clrexcp
    v_cos_f32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_cvt_f16_f32                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_cvt_f32_f16                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src_2>`
    v_cvt_f32_f64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src_1>`
    v_cvt_f32_i32                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_cvt_f32_u32                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_cvt_f32_ubyte0               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_cvt_f32_ubyte1               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_cvt_f32_ubyte2               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_cvt_f32_ubyte3               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_cvt_f64_f32                  :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_cvt_f64_i32                  :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_cvt_f64_u32                  :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_cvt_flr_i32_f32              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_cvt_i32_f32                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_cvt_i32_f64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src_1>`
    v_cvt_off_f32_i4               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src_2>`
    v_cvt_rpi_i32_f32              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_cvt_u32_f32                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_cvt_u32_f64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src_1>`
    v_exp_f32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_exp_legacy_f32               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_ffbh_i32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_ffbh_u32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_ffbl_b32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_floor_f32                    :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_floor_f64                    :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,     :ref:`src<amdgpu_synid_gfx7_src_1>`
    v_fract_f32                    :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_fract_f64                    :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,     :ref:`src<amdgpu_synid_gfx7_src_1>`
    v_frexp_exp_i32_f32            :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_frexp_exp_i32_f64            :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src_1>`
    v_frexp_mant_f32               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_frexp_mant_f64               :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,     :ref:`src<amdgpu_synid_gfx7_src_1>`
    v_log_clamp_f32                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_log_f32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_log_legacy_f32               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_mov_b32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_movreld_b32                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_movrels_b32                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`vsrc<amdgpu_synid_gfx7_vsrc_1>`
    v_movrelsd_b32                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`vsrc<amdgpu_synid_gfx7_vsrc_1>`
    v_nop
    v_not_b32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_rcp_clamp_f32                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_rcp_clamp_f64                :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,     :ref:`src<amdgpu_synid_gfx7_src_1>`
    v_rcp_f32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_rcp_f64                      :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,     :ref:`src<amdgpu_synid_gfx7_src_1>`
    v_rcp_iflag_f32                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_rcp_legacy_f32               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_readfirstlane_b32            :ref:`sdst<amdgpu_synid_gfx7_sdst_7>`,     :ref:`src<amdgpu_synid_gfx7_src_3>`
    v_rndne_f32                    :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_rndne_f64                    :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,     :ref:`src<amdgpu_synid_gfx7_src_1>`
    v_rsq_clamp_f32                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_rsq_clamp_f64                :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,     :ref:`src<amdgpu_synid_gfx7_src_1>`
    v_rsq_f32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_rsq_f64                      :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,     :ref:`src<amdgpu_synid_gfx7_src_1>`
    v_rsq_legacy_f32               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_sin_f32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_sqrt_f32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_sqrt_f64                     :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,     :ref:`src<amdgpu_synid_gfx7_src_1>`
    v_trunc_f32                    :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`src<amdgpu_synid_gfx7_src>`
    v_trunc_f64                    :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,     :ref:`src<amdgpu_synid_gfx7_src_1>`

VOP2
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST0**      **DST1**      **SRC0**      **SRC1**      **SRC2**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    v_add_f32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_add_i32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_addc_u32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`,    :ref:`vcc<amdgpu_synid_gfx7_vcc>`
    v_and_b32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_ashr_i32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_ashrrev_i32                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src_4>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`, :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_bcnt_u32_b32                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_bfm_b32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cndmask_b32                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`,    :ref:`vcc<amdgpu_synid_gfx7_vcc>`
    v_cvt_pk_i16_i32               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`::ref:`i32<amdgpu_synid_gfx7_type_deviation>`, :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`::ref:`i32<amdgpu_synid_gfx7_type_deviation>`
    v_cvt_pk_u16_u32               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`, :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_cvt_pkaccum_u8_f32           :ref:`vdst<amdgpu_synid_gfx7_vdst>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`,           :ref:`src0<amdgpu_synid_gfx7_src>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`, :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_cvt_pknorm_i16_f32           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`, :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`
    v_cvt_pknorm_u16_f32           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`, :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`
    v_cvt_pkrtz_f16_f32            :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`, :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`
    v_ldexp_f32                    :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`::ref:`i32<amdgpu_synid_gfx7_type_deviation>`
    v_lshl_b32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_lshlrev_b32                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src_4>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`, :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_lshr_b32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_lshrrev_b32                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src_4>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`, :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_mac_f32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_mac_legacy_f32               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_madak_f32                    :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src_5>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`,    :ref:`simm32<amdgpu_synid_gfx7_simm32_1>`
    v_madmk_f32                    :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src_5>`,     :ref:`simm32<amdgpu_synid_gfx7_simm32_1>`,   :ref:`vsrc2<amdgpu_synid_gfx7_vsrc_1>`
    v_max_f32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_max_i32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_max_legacy_f32               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_max_u32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_mbcnt_hi_u32_b32             :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_mbcnt_lo_u32_b32             :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_min_f32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_min_i32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_min_legacy_f32               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_min_u32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_mul_f32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_mul_hi_i32_i24               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_mul_hi_u32_u24               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_mul_i32_i24                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_mul_legacy_f32               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_mul_u32_u24                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_or_b32                       :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_readlane_b32                 :ref:`sdst<amdgpu_synid_gfx7_sdst_7>`,               :ref:`src0<amdgpu_synid_gfx7_src_3>`,     :ref:`ssrc1<amdgpu_synid_gfx7_ssrc_9>`
    v_sub_f32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_sub_i32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_subb_u32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`,    :ref:`vcc<amdgpu_synid_gfx7_vcc>`
    v_subbrev_u32                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_4>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`,    :ref:`vcc<amdgpu_synid_gfx7_vcc>`
    v_subrev_f32                   :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src_4>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_subrev_i32                   :ref:`vdst<amdgpu_synid_gfx7_vdst>`,     :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_4>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_writelane_b32                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`ssrc0<amdgpu_synid_gfx7_ssrc_10>`,    :ref:`ssrc1<amdgpu_synid_gfx7_ssrc_9>`
    v_xor_b32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,               :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`

VOP3
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST0**        **DST1**      **SRC0**        **SRC1**        **SRC2**         **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    v_add_f32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_add_f64                      :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_add_i32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_addc_u32_e64                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`ssrc2<amdgpu_synid_gfx7_ssrc_4>`
    v_alignbit_b32                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`src2<amdgpu_synid_gfx7_src_6>`
    v_alignbyte_b32                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`src2<amdgpu_synid_gfx7_src_6>`
    v_and_b32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_ashr_i32_e64                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_ashr_i64                     :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_ashrrev_i32_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`,   :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_bcnt_u32_b32_e64             :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_bfe_i32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`,   :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_bfe_u32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`src2<amdgpu_synid_gfx7_src_6>`
    v_bfi_b32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`src2<amdgpu_synid_gfx7_src_6>`
    v_bfm_b32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_bfrev_b32_e64                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`
    v_ceil_f32_e64                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_ceil_f64_e64                 :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_clrexcp_e64
    v_cmp_class_f32_e64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`
    v_cmp_class_f64_e64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`
    v_cmp_eq_f32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_eq_f64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_eq_i32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmp_eq_i64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmp_eq_u32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmp_eq_u64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmp_f_f32_e64                :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_f_f64_e64                :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_f_i32_e64                :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmp_f_i64_e64                :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmp_f_u32_e64                :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmp_f_u64_e64                :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmp_ge_f32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_ge_f64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_ge_i32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmp_ge_i64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmp_ge_u32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmp_ge_u64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmp_gt_f32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_gt_f64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_gt_i32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmp_gt_i64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmp_gt_u32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmp_gt_u64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmp_le_f32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_le_f64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_le_i32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmp_le_i64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmp_le_u32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmp_le_u64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmp_lg_f32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_lg_f64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_lt_f32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_lt_f64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_lt_i32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmp_lt_i64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmp_lt_u32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmp_lt_u64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmp_ne_i32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmp_ne_i64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmp_ne_u32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmp_ne_u64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmp_neq_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_neq_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_nge_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_nge_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_ngt_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_ngt_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_nle_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_nle_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_nlg_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_nlg_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_nlt_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_nlt_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_o_f32_e64                :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_o_f64_e64                :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_t_i32_e64                :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmp_t_i64_e64                :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmp_t_u32_e64                :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmp_t_u64_e64                :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmp_tru_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_tru_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_u_f32_e64                :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmp_u_f64_e64                :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_eq_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_eq_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_f_f32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_f_f64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_ge_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_ge_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_gt_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_gt_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_le_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_le_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_lg_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_lg_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_lt_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_lt_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_neq_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_neq_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_nge_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_nge_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_ngt_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_ngt_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_nle_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_nle_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_nlg_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_nlg_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_nlt_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_nlt_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_o_f32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_o_f64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_tru_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_tru_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_u_f32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmps_u_f64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_eq_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_eq_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_f_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_f_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_ge_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_ge_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_gt_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_gt_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_le_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_le_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_lg_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_lg_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_lt_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_lt_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_neq_f32_e64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_neq_f64_e64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_nge_f32_e64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_nge_f64_e64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_ngt_f32_e64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_ngt_f64_e64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_nle_f32_e64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_nle_f64_e64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_nlg_f32_e64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_nlg_f64_e64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_nlt_f32_e64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_nlt_f64_e64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_o_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_o_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_tru_f32_e64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_tru_f64_e64            :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_u_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpsx_u_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_class_f32_e64           :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`
    v_cmpx_class_f64_e64           :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`
    v_cmpx_eq_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_eq_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_eq_i32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmpx_eq_i64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmpx_eq_u32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmpx_eq_u64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmpx_f_f32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_f_f64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_f_i32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmpx_f_i64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmpx_f_u32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmpx_f_u64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmpx_ge_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_ge_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_ge_i32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmpx_ge_i64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmpx_ge_u32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmpx_ge_u64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmpx_gt_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_gt_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_gt_i32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmpx_gt_i64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmpx_gt_u32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmpx_gt_u64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmpx_le_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_le_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_le_i32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmpx_le_i64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmpx_le_u32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmpx_le_u64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmpx_lg_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_lg_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_lt_f32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_lt_f64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_lt_i32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmpx_lt_i64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmpx_lt_u32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmpx_lt_u64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmpx_ne_i32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmpx_ne_i64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmpx_ne_u32_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmpx_ne_u64_e64              :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmpx_neq_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_neq_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_nge_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_nge_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_ngt_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_ngt_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_nle_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_nle_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_nlg_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_nlg_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_nlt_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_nlt_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_o_f32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_o_f64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_t_i32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmpx_t_i64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmpx_t_u32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_cmpx_t_u64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`
    v_cmpx_tru_f32_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_tru_f64_e64             :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_u_f32_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cmpx_u_f64_e64               :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cndmask_b32_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`ssrc2<amdgpu_synid_gfx7_ssrc_4>`
    v_cos_f32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cubeid_f32                   :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`       :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cubema_f32                   :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`       :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cubesc_f32                   :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`       :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cubetc_f32                   :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`       :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f16_f32_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cvt_f32_f16_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_8>`                                  :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f32_f64_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f32_i32_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`                                  :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f32_u32_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`                                  :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f32_ubyte0_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`
    v_cvt_f32_ubyte1_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`
    v_cvt_f32_ubyte2_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`
    v_cvt_f32_ubyte3_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`
    v_cvt_f64_f32_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f64_i32_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`                                  :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f64_u32_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`                                  :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_flr_i32_f32_e64          :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cvt_i32_f32_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cvt_i32_f64_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cvt_off_f32_i4_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_9>`                                  :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_pk_i16_i32_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`i32<amdgpu_synid_gfx7_type_deviation>`,   :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`i32<amdgpu_synid_gfx7_type_deviation>`
    v_cvt_pk_u16_u32_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`,   :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_cvt_pk_u8_f32                :ref:`vdst<amdgpu_synid_gfx7_vdst>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`,             :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`,   :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`,   :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_cvt_pkaccum_u8_f32_e64       :ref:`vdst<amdgpu_synid_gfx7_vdst>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`,             :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`, :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_cvt_pknorm_i16_f32_e64       :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`, :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`
    v_cvt_pknorm_u16_f32_e64       :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`, :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`
    v_cvt_pkrtz_f16_f32_e64        :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`, :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`::ref:`f32<amdgpu_synid_gfx7_type_deviation>`
    v_cvt_rpi_i32_f32_e64          :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cvt_u32_f32_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_cvt_u32_f64_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_div_fixup_f32                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`       :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_div_fixup_f64                :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src2<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`       :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_div_fmas_f32                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`       :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_div_fmas_f64                 :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src2<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`       :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_div_scale_f32                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`src2<amdgpu_synid_gfx7_src_6>`
    v_div_scale_f64                :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_7>`,       :ref:`src2<amdgpu_synid_gfx7_src_7>`
    v_exp_f32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_exp_legacy_f32_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_ffbh_i32_e64                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`
    v_ffbh_u32_e64                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`
    v_ffbl_b32_e64                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`
    v_floor_f32_e64                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_floor_f64_e64                :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_fma_f32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`       :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_fma_f64                      :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src2<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`       :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_fract_f32_e64                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_fract_f64_e64                :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_frexp_exp_i32_f32_e64        :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`
    v_frexp_exp_i32_f64_e64        :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`
    v_frexp_mant_f32_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`
    v_frexp_mant_f64_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_ldexp_f32_e64                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`i32<amdgpu_synid_gfx7_type_deviation>`                 :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_ldexp_f64                    :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`i32<amdgpu_synid_gfx7_type_deviation>`                 :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_lerp_u8                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`,             :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`,   :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`,   :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`
    v_log_clamp_f32_e64            :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_log_f32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_log_legacy_f32_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_lshl_b32_e64                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_lshl_b64                     :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_lshlrev_b32_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`,   :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_lshr_b32_e64                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_lshr_b64                     :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_lshrrev_b32_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`,   :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_mac_f32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mac_legacy_f32_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mad_f32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`       :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mad_i32_i24                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`i32<amdgpu_synid_gfx7_type_deviation>`
    v_mad_i64_i32                  :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`src2<amdgpu_synid_gfx7_src_7>`::ref:`i64<amdgpu_synid_gfx7_type_deviation>`
    v_mad_legacy_f32               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`       :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mad_u32_u24                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_mad_u64_u32                  :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,       :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`src2<amdgpu_synid_gfx7_src_7>`::ref:`u64<amdgpu_synid_gfx7_type_deviation>`
    v_max3_f32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`       :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_max3_i32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`src2<amdgpu_synid_gfx7_src_6>`
    v_max3_u32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`src2<amdgpu_synid_gfx7_src_6>`
    v_max_f32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_max_f64                      :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_max_i32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_max_legacy_f32_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_max_u32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_mbcnt_hi_u32_b32_e64         :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_mbcnt_lo_u32_b32_e64         :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_med3_f32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`       :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_med3_i32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`src2<amdgpu_synid_gfx7_src_6>`
    v_med3_u32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`src2<amdgpu_synid_gfx7_src_6>`
    v_min3_f32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`       :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_min3_i32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`src2<amdgpu_synid_gfx7_src_6>`
    v_min3_u32                     :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`src2<amdgpu_synid_gfx7_src_6>`
    v_min_f32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_min_f64                      :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_min_i32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_min_legacy_f32_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_min_u32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_mov_b32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`
    v_movreld_b32_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`
    v_movrels_b32_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`vsrc<amdgpu_synid_gfx7_vsrc_1>`
    v_movrelsd_b32_e64             :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`vsrc<amdgpu_synid_gfx7_vsrc_1>`
    v_mqsad_pk_u16_u8              :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`::ref:`u16x4<amdgpu_synid_gfx7_type_deviation>`,           :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`u8x8<amdgpu_synid_gfx7_type_deviation>`,  :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`u8x4<amdgpu_synid_gfx7_type_deviation>`,  :ref:`src2<amdgpu_synid_gfx7_src_7>`::ref:`u16x4<amdgpu_synid_gfx7_type_deviation>`
    v_mqsad_u32_u8                 :ref:`vdst<amdgpu_synid_gfx7_vdst_2>`::ref:`u32x4<amdgpu_synid_gfx7_type_deviation>`,           :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`u8x8<amdgpu_synid_gfx7_type_deviation>`,  :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`u8x4<amdgpu_synid_gfx7_type_deviation>`,  :ref:`vsrc2<amdgpu_synid_gfx7_vsrc_2>`::ref:`u32x4<amdgpu_synid_gfx7_type_deviation>`
    v_msad_u8                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`,             :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`u8x4<amdgpu_synid_gfx7_type_deviation>`,  :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`u8x4<amdgpu_synid_gfx7_type_deviation>`,  :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_mul_f32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mul_f64                      :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mul_hi_i32                   :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_mul_hi_i32_i24_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_mul_hi_u32                   :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_mul_hi_u32_u24_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_mul_i32_i24_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_mul_legacy_f32_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mul_lo_i32                   :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_mul_lo_u32                   :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_mul_u32_u24_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_mullit_f32                   :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`       :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_nop_e64
    v_not_b32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`
    v_or_b32_e64                   :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_qsad_pk_u16_u8               :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`::ref:`u16x4<amdgpu_synid_gfx7_type_deviation>`,           :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`u8x8<amdgpu_synid_gfx7_type_deviation>`,  :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`u8x4<amdgpu_synid_gfx7_type_deviation>`,  :ref:`src2<amdgpu_synid_gfx7_src_7>`::ref:`u16x4<amdgpu_synid_gfx7_type_deviation>`
    v_rcp_clamp_f32_e64            :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rcp_clamp_f64_e64            :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rcp_f32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rcp_f64_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rcp_iflag_f32_e64            :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rcp_legacy_f32_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rndne_f32_e64                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rndne_f64_e64                :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rsq_clamp_f32_e64            :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rsq_clamp_f64_e64            :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rsq_f32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rsq_f64_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rsq_legacy_f32_e64           :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_sad_hi_u8                    :ref:`vdst<amdgpu_synid_gfx7_vdst>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`,             :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`u8x4<amdgpu_synid_gfx7_type_deviation>`,  :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`u8x4<amdgpu_synid_gfx7_type_deviation>`,  :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_sad_u16                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`,             :ref:`src0<amdgpu_synid_gfx7_src_9>`::ref:`u16x2<amdgpu_synid_gfx7_type_deviation>`, :ref:`src1<amdgpu_synid_gfx7_src_10>`::ref:`u16x2<amdgpu_synid_gfx7_type_deviation>`, :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_sad_u32                      :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`src2<amdgpu_synid_gfx7_src_6>`
    v_sad_u8                       :ref:`vdst<amdgpu_synid_gfx7_vdst>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`,             :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`u8x4<amdgpu_synid_gfx7_type_deviation>`,  :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`u8x4<amdgpu_synid_gfx7_type_deviation>`,  :ref:`src2<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`
    v_sin_f32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_sqrt_f32_e64                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_sqrt_f64_e64                 :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_sub_f32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_sub_i32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_subb_u32_e64                 :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`ssrc2<amdgpu_synid_gfx7_ssrc_4>`
    v_subbrev_u32_e64              :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`src0<amdgpu_synid_gfx7_src_6>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`,       :ref:`ssrc2<amdgpu_synid_gfx7_ssrc_4>`
    v_subrev_f32_e64               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`m<amdgpu_synid_gfx7_m>`                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_subrev_i32_e64               :ref:`vdst<amdgpu_synid_gfx7_vdst>`,       :ref:`sdst<amdgpu_synid_gfx7_sdst_2>`,     :ref:`src0<amdgpu_synid_gfx7_src_6>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`
    v_trig_preop_f64               :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src0<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`,     :ref:`src1<amdgpu_synid_gfx7_src_6>`::ref:`u32<amdgpu_synid_gfx7_type_deviation>`                 :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_trunc_f32_e64                :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src<amdgpu_synid_gfx7_src_5>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_trunc_f64_e64                :ref:`vdst<amdgpu_synid_gfx7_vdst_1>`,                 :ref:`src<amdgpu_synid_gfx7_src_7>`::ref:`m<amdgpu_synid_gfx7_m>`                                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_xor_b32_e64                  :ref:`vdst<amdgpu_synid_gfx7_vdst>`,                 :ref:`src0<amdgpu_synid_gfx7_src_5>`,       :ref:`src1<amdgpu_synid_gfx7_src_6>`

VOPC
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC0**      **SRC1**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    v_cmp_class_f32                :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`
    v_cmp_class_f64                :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`
    v_cmp_eq_f32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_eq_f64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_eq_i32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_eq_i64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_eq_u32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_eq_u64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_f_f32                    :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_f_f64                    :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_f_i32                    :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_f_i64                    :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_f_u32                    :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_f_u64                    :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_ge_f32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_ge_f64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_ge_i32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_ge_i64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_ge_u32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_ge_u64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_gt_f32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_gt_f64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_gt_i32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_gt_i64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_gt_u32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_gt_u64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_le_f32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_le_f64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_le_i32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_le_i64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_le_u32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_le_u64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_lg_f32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_lg_f64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_lt_f32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_lt_f64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_lt_i32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_lt_i64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_lt_u32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_lt_u64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_ne_i32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_ne_i64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_ne_u32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_ne_u64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_neq_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_neq_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_nge_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_nge_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_ngt_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_ngt_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_nle_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_nle_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_nlg_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_nlg_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_nlt_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_nlt_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_o_f32                    :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_o_f64                    :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_t_i32                    :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_t_i64                    :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_t_u32                    :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_t_u64                    :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_tru_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_tru_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmp_u_f32                    :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmp_u_f64                    :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmps_eq_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmps_eq_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmps_f_f32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmps_f_f64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmps_ge_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmps_ge_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmps_gt_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmps_gt_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmps_le_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmps_le_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmps_lg_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmps_lg_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmps_lt_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmps_lt_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmps_neq_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmps_neq_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmps_nge_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmps_nge_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmps_ngt_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmps_ngt_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmps_nle_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmps_nle_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmps_nlg_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmps_nlg_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmps_nlt_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmps_nlt_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmps_o_f32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmps_o_f64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmps_tru_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmps_tru_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmps_u_f32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmps_u_f64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpsx_eq_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpsx_eq_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpsx_f_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpsx_f_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpsx_ge_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpsx_ge_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpsx_gt_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpsx_gt_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpsx_le_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpsx_le_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpsx_lg_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpsx_lg_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpsx_lt_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpsx_lt_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpsx_neq_f32                :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpsx_neq_f64                :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpsx_nge_f32                :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpsx_nge_f64                :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpsx_ngt_f32                :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpsx_ngt_f64                :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpsx_nle_f32                :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpsx_nle_f64                :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpsx_nlg_f32                :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpsx_nlg_f64                :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpsx_nlt_f32                :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpsx_nlt_f64                :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpsx_o_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpsx_o_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpsx_tru_f32                :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpsx_tru_f64                :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpsx_u_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpsx_u_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_class_f32               :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`
    v_cmpx_class_f64               :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`::ref:`b32<amdgpu_synid_gfx7_type_deviation>`
    v_cmpx_eq_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_eq_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_eq_i32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_eq_i64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_eq_u32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_eq_u64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_f_f32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_f_f64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_f_i32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_f_i64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_f_u32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_f_u64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_ge_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_ge_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_ge_i32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_ge_i64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_ge_u32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_ge_u64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_gt_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_gt_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_gt_i32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_gt_i64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_gt_u32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_gt_u64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_le_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_le_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_le_i32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_le_i64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_le_u32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_le_u64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_lg_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_lg_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_lt_f32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_lt_f64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_lt_i32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_lt_i64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_lt_u32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_lt_u64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_ne_i32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_ne_i64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_ne_u32                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_ne_u64                  :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_neq_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_neq_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_nge_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_nge_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_ngt_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_ngt_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_nle_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_nle_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_nlg_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_nlg_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_nlt_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_nlt_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_o_f32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_o_f64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_t_i32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_t_i64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_t_u32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_t_u64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_tru_f32                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_tru_f64                 :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`
    v_cmpx_u_f32                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_1>`
    v_cmpx_u_f64                   :ref:`vcc<amdgpu_synid_gfx7_vcc>`,      :ref:`src0<amdgpu_synid_gfx7_src_1>`,     :ref:`vsrc1<amdgpu_synid_gfx7_vsrc_3>`

.. |---| unicode:: U+02014 .. em dash

.. toctree::
    :hidden:

    gfx7_attr
    gfx7_dst
    gfx7_hwreg
    gfx7_imm16
    gfx7_imm16_1
    gfx7_imm16_2
    gfx7_label
    gfx7_m
    gfx7_msg
    gfx7_opt
    gfx7_param
    gfx7_sbase
    gfx7_sbase_1
    gfx7_sdst
    gfx7_sdst_1
    gfx7_sdst_2
    gfx7_sdst_3
    gfx7_sdst_4
    gfx7_sdst_5
    gfx7_sdst_6
    gfx7_sdst_7
    gfx7_simm32
    gfx7_simm32_1
    gfx7_soffset
    gfx7_soffset_1
    gfx7_src
    gfx7_src_1
    gfx7_src_10
    gfx7_src_2
    gfx7_src_3
    gfx7_src_4
    gfx7_src_5
    gfx7_src_6
    gfx7_src_7
    gfx7_src_8
    gfx7_src_9
    gfx7_srsrc
    gfx7_srsrc_1
    gfx7_ssamp
    gfx7_ssrc
    gfx7_ssrc_1
    gfx7_ssrc_10
    gfx7_ssrc_2
    gfx7_ssrc_3
    gfx7_ssrc_4
    gfx7_ssrc_5
    gfx7_ssrc_6
    gfx7_ssrc_7
    gfx7_ssrc_8
    gfx7_ssrc_9
    gfx7_tgt
    gfx7_type_deviation
    gfx7_vaddr
    gfx7_vaddr_1
    gfx7_vaddr_2
    gfx7_vaddr_3
    gfx7_vcc
    gfx7_vdata
    gfx7_vdata0
    gfx7_vdata0_1
    gfx7_vdata1
    gfx7_vdata1_1
    gfx7_vdata_1
    gfx7_vdata_2
    gfx7_vdata_3
    gfx7_vdata_4
    gfx7_vdata_5
    gfx7_vdata_6
    gfx7_vdata_7
    gfx7_vdata_8
    gfx7_vdata_9
    gfx7_vdst
    gfx7_vdst_1
    gfx7_vdst_10
    gfx7_vdst_11
    gfx7_vdst_12
    gfx7_vdst_2
    gfx7_vdst_3
    gfx7_vdst_4
    gfx7_vdst_5
    gfx7_vdst_6
    gfx7_vdst_7
    gfx7_vdst_8
    gfx7_vdst_9
    gfx7_vsrc
    gfx7_vsrc_1
    gfx7_vsrc_2
    gfx7_vsrc_3
    gfx7_waitcnt
