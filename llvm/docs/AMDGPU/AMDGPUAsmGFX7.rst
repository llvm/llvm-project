..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

============================
Syntax of GFX7 Instructions
============================

.. contents::
  :local:

Notation
========

Notation used in this document is explained :ref:`here<amdgpu_syn_instruction_notation>`.

Introduction
============

An overview of generic syntax and other features of AMDGPU instructions may be found :ref:`in this document<amdgpu_syn_instructions>`.

Instructions
============


DS
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**         **SRC0**      **SRC1**      **SRC2**           **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    ds_add_rtn_u32                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_add_rtn_u64                 :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_add_src2_u32                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_add_src2_u64                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_add_u32                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_add_u64                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_and_b32                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_and_b64                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_and_rtn_b32                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_and_rtn_b64                 :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_and_src2_b32                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_and_src2_b64                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_append                      :ref:`vdst<amdgpu_synid7_vdst32_0>`                                           :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_b32                               :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata32_0>`,   :ref:`vdata1<amdgpu_synid7_vdata32_0>`         :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_b64                               :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata64_0>`,   :ref:`vdata1<amdgpu_synid7_vdata64_0>`         :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_f32                               :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata32_0>`,   :ref:`vdata1<amdgpu_synid7_vdata32_0>`         :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_f64                               :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata64_0>`,   :ref:`vdata1<amdgpu_synid7_vdata64_0>`         :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_rtn_b32               :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata32_0>`,   :ref:`vdata1<amdgpu_synid7_vdata32_0>`         :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_rtn_b64               :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata64_0>`,   :ref:`vdata1<amdgpu_synid7_vdata64_0>`         :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_rtn_f32               :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata32_0>`,   :ref:`vdata1<amdgpu_synid7_vdata32_0>`         :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_rtn_f64               :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata64_0>`,   :ref:`vdata1<amdgpu_synid7_vdata64_0>`         :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_condxchg32_rtn_b64          :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_consume                     :ref:`vdst<amdgpu_synid7_vdst32_0>`                                           :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_dec_rtn_u32                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_dec_rtn_u64                 :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_dec_src2_u32                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_dec_src2_u64                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_dec_u32                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_dec_u64                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_gws_barrier                             :ref:`vdata<amdgpu_synid7_vdata32_0>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_gws_init                                :ref:`vdata<amdgpu_synid7_vdata32_0>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_gws_sema_br                             :ref:`vdata<amdgpu_synid7_vdata32_0>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_gws_sema_p                                                                 :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_gws_sema_release_all                                                       :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_gws_sema_v                                                                 :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_inc_rtn_u32                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_inc_rtn_u64                 :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_inc_src2_u32                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_inc_src2_u64                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_inc_u32                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_inc_u64                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_f32                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_f64                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_i32                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_i64                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_rtn_f32                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_rtn_f64                 :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_rtn_i32                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_rtn_i64                 :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_rtn_u32                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_rtn_u64                 :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_src2_f32                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_src2_f64                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_src2_i32                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_src2_i64                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_src2_u32                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_src2_u64                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_u32                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_u64                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_f32                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_f64                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_i32                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_i64                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_rtn_f32                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_rtn_f64                 :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_rtn_i32                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_rtn_i64                 :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_rtn_u32                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_rtn_u64                 :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_src2_f32                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_src2_f64                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_src2_i32                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_src2_i64                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_src2_u32                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_src2_u64                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_u32                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_u64                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_mskor_b32                               :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata32_0>`,   :ref:`vdata1<amdgpu_synid7_vdata32_0>`         :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_mskor_b64                               :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata64_0>`,   :ref:`vdata1<amdgpu_synid7_vdata64_0>`         :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_mskor_rtn_b32               :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata32_0>`,   :ref:`vdata1<amdgpu_synid7_vdata32_0>`         :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_mskor_rtn_b64               :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata64_0>`,   :ref:`vdata1<amdgpu_synid7_vdata64_0>`         :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_nop
    ds_or_b32                                  :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_or_b64                                  :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_or_rtn_b32                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_or_rtn_b64                  :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_or_src2_b32                             :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_or_src2_b64                             :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_ordered_count               :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read2_b32                   :ref:`vdst<amdgpu_synid7_vdst64_0>`::ref:`b32x2<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_read2_b64                   :ref:`vdst<amdgpu_synid7_vdst128_0>`::ref:`b64x2<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_read2st64_b32               :ref:`vdst<amdgpu_synid7_vdst64_0>`::ref:`b32x2<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_read2st64_b64               :ref:`vdst<amdgpu_synid7_vdst128_0>`::ref:`b64x2<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_b128                   :ref:`vdst<amdgpu_synid7_vdst128_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_b32                    :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_b64                    :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_b96                    :ref:`vdst<amdgpu_synid7_vdst96_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_i16                    :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_i8                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_u16                    :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_u8                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_rsub_rtn_u32                :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_rsub_rtn_u64                :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_rsub_src2_u32                           :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_rsub_src2_u64                           :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_rsub_u32                                :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_rsub_u64                                :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_sub_rtn_u32                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_sub_rtn_u64                 :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_sub_src2_u32                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_sub_src2_u64                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_sub_u32                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_sub_u64                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_swizzle_b32                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`pattern<amdgpu_synid_sw_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrap_rtn_b32                :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata32_0>`,   :ref:`vdata1<amdgpu_synid7_vdata32_0>`         :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write2_b32                              :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata32_0>`,   :ref:`vdata1<amdgpu_synid7_vdata32_0>`         :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_write2_b64                              :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata64_0>`,   :ref:`vdata1<amdgpu_synid7_vdata64_0>`         :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_write2st64_b32                          :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata32_0>`,   :ref:`vdata1<amdgpu_synid7_vdata32_0>`         :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_write2st64_b64                          :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata64_0>`,   :ref:`vdata1<amdgpu_synid7_vdata64_0>`         :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_b128                              :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata128_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_b16                               :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_b32                               :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_b64                               :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_b8                                :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_b96                               :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata96_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_src2_b32                          :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_src2_b64                          :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrxchg2_rtn_b32             :ref:`vdst<amdgpu_synid7_vdst64_0>`::ref:`b32x2<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata32_0>`,   :ref:`vdata1<amdgpu_synid7_vdata32_0>`         :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrxchg2_rtn_b64             :ref:`vdst<amdgpu_synid7_vdst128_0>`::ref:`b64x2<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata64_0>`,   :ref:`vdata1<amdgpu_synid7_vdata64_0>`         :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrxchg2st64_rtn_b32         :ref:`vdst<amdgpu_synid7_vdst64_0>`::ref:`b32x2<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata32_0>`,   :ref:`vdata1<amdgpu_synid7_vdata32_0>`         :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrxchg2st64_rtn_b64         :ref:`vdst<amdgpu_synid7_vdst128_0>`::ref:`b64x2<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata0<amdgpu_synid7_vdata64_0>`,   :ref:`vdata1<amdgpu_synid7_vdata64_0>`         :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrxchg_rtn_b32              :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrxchg_rtn_b64              :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_xor_b32                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_xor_b64                                 :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_xor_rtn_b32                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_xor_rtn_b64                 :ref:`vdst<amdgpu_synid7_vdst64_0>`,       :ref:`vaddr<amdgpu_synid7_addr_ds>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`                    :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_xor_src2_b32                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_xor_src2_b64                            :ref:`vaddr<amdgpu_synid7_addr_ds>`                              :ref:`offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`

EXP
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC0**      **SRC1**      **SRC2**      **SRC3**           **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    exp                            :ref:`tgt<amdgpu_synid7_tgt>`,      :ref:`vsrc0<amdgpu_synid7_src_exp>`,    :ref:`vsrc1<amdgpu_synid7_src_exp>`,    :ref:`vsrc2<amdgpu_synid7_src_exp>`,    :ref:`vsrc3<amdgpu_synid7_src_exp>`          :ref:`done<amdgpu_synid_done>` :ref:`compr<amdgpu_synid_compr>` :ref:`vm<amdgpu_synid_vm>`

FLAT
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**           **SRC0**      **SRC1**             **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    flat_atomic_add                :ref:`vdst<amdgpu_synid7_dst_flat_atomic32>`::ref:`opt<amdgpu_synid7_opt>`,     :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_add_x2             :ref:`vdst<amdgpu_synid7_dst_flat_atomic64>`::ref:`opt<amdgpu_synid7_opt>`,     :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_and                :ref:`vdst<amdgpu_synid7_dst_flat_atomic32>`::ref:`opt<amdgpu_synid7_opt>`,     :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_and_x2             :ref:`vdst<amdgpu_synid7_dst_flat_atomic64>`::ref:`opt<amdgpu_synid7_opt>`,     :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_cmpswap            :ref:`vdst<amdgpu_synid7_dst_flat_atomic32>`::ref:`opt<amdgpu_synid7_opt>`,     :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`::ref:`b32x2<amdgpu_synid7_type_dev>`      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_cmpswap_x2         :ref:`vdst<amdgpu_synid7_dst_flat_atomic64>`::ref:`opt<amdgpu_synid7_opt>`,     :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata128_0>`::ref:`b64x2<amdgpu_synid7_type_dev>`      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_dec                :ref:`vdst<amdgpu_synid7_dst_flat_atomic32>`::ref:`opt<amdgpu_synid7_opt>`::ref:`u32<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`::ref:`u32<amdgpu_synid7_type_dev>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_dec_x2             :ref:`vdst<amdgpu_synid7_dst_flat_atomic64>`::ref:`opt<amdgpu_synid7_opt>`::ref:`u64<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`::ref:`u64<amdgpu_synid7_type_dev>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_fcmpswap           :ref:`vdst<amdgpu_synid7_dst_flat_atomic32>`::ref:`opt<amdgpu_synid7_opt>`::ref:`f32<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`::ref:`f32x2<amdgpu_synid7_type_dev>`      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_fcmpswap_x2        :ref:`vdst<amdgpu_synid7_dst_flat_atomic64>`::ref:`opt<amdgpu_synid7_opt>`::ref:`f64<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata128_0>`::ref:`f64x2<amdgpu_synid7_type_dev>`      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_fmax               :ref:`vdst<amdgpu_synid7_dst_flat_atomic32>`::ref:`opt<amdgpu_synid7_opt>`::ref:`f32<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`::ref:`f32<amdgpu_synid7_type_dev>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_fmax_x2            :ref:`vdst<amdgpu_synid7_dst_flat_atomic64>`::ref:`opt<amdgpu_synid7_opt>`::ref:`f64<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`::ref:`f64<amdgpu_synid7_type_dev>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_fmin               :ref:`vdst<amdgpu_synid7_dst_flat_atomic32>`::ref:`opt<amdgpu_synid7_opt>`::ref:`f32<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`::ref:`f32<amdgpu_synid7_type_dev>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_fmin_x2            :ref:`vdst<amdgpu_synid7_dst_flat_atomic64>`::ref:`opt<amdgpu_synid7_opt>`::ref:`f64<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`::ref:`f64<amdgpu_synid7_type_dev>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_inc                :ref:`vdst<amdgpu_synid7_dst_flat_atomic32>`::ref:`opt<amdgpu_synid7_opt>`::ref:`u32<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`::ref:`u32<amdgpu_synid7_type_dev>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_inc_x2             :ref:`vdst<amdgpu_synid7_dst_flat_atomic64>`::ref:`opt<amdgpu_synid7_opt>`::ref:`u64<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`::ref:`u64<amdgpu_synid7_type_dev>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_or                 :ref:`vdst<amdgpu_synid7_dst_flat_atomic32>`::ref:`opt<amdgpu_synid7_opt>`,     :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_or_x2              :ref:`vdst<amdgpu_synid7_dst_flat_atomic64>`::ref:`opt<amdgpu_synid7_opt>`,     :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_smax               :ref:`vdst<amdgpu_synid7_dst_flat_atomic32>`::ref:`opt<amdgpu_synid7_opt>`::ref:`s32<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`::ref:`s32<amdgpu_synid7_type_dev>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_smax_x2            :ref:`vdst<amdgpu_synid7_dst_flat_atomic64>`::ref:`opt<amdgpu_synid7_opt>`::ref:`s64<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`::ref:`s64<amdgpu_synid7_type_dev>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_smin               :ref:`vdst<amdgpu_synid7_dst_flat_atomic32>`::ref:`opt<amdgpu_synid7_opt>`::ref:`s32<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`::ref:`s32<amdgpu_synid7_type_dev>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_smin_x2            :ref:`vdst<amdgpu_synid7_dst_flat_atomic64>`::ref:`opt<amdgpu_synid7_opt>`::ref:`s64<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`::ref:`s64<amdgpu_synid7_type_dev>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_sub                :ref:`vdst<amdgpu_synid7_dst_flat_atomic32>`::ref:`opt<amdgpu_synid7_opt>`,     :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_sub_x2             :ref:`vdst<amdgpu_synid7_dst_flat_atomic64>`::ref:`opt<amdgpu_synid7_opt>`,     :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_swap               :ref:`vdst<amdgpu_synid7_dst_flat_atomic32>`::ref:`opt<amdgpu_synid7_opt>`,     :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_swap_x2            :ref:`vdst<amdgpu_synid7_dst_flat_atomic64>`::ref:`opt<amdgpu_synid7_opt>`,     :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_umax               :ref:`vdst<amdgpu_synid7_dst_flat_atomic32>`::ref:`opt<amdgpu_synid7_opt>`::ref:`u32<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`::ref:`u32<amdgpu_synid7_type_dev>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_umax_x2            :ref:`vdst<amdgpu_synid7_dst_flat_atomic64>`::ref:`opt<amdgpu_synid7_opt>`::ref:`u64<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`::ref:`u64<amdgpu_synid7_type_dev>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_umin               :ref:`vdst<amdgpu_synid7_dst_flat_atomic32>`::ref:`opt<amdgpu_synid7_opt>`::ref:`u32<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`::ref:`u32<amdgpu_synid7_type_dev>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_umin_x2            :ref:`vdst<amdgpu_synid7_dst_flat_atomic64>`::ref:`opt<amdgpu_synid7_opt>`::ref:`u64<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`::ref:`u64<amdgpu_synid7_type_dev>`        :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_xor                :ref:`vdst<amdgpu_synid7_dst_flat_atomic32>`::ref:`opt<amdgpu_synid7_opt>`,     :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_xor_x2             :ref:`vdst<amdgpu_synid7_dst_flat_atomic64>`::ref:`opt<amdgpu_synid7_opt>`,     :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_dword                :ref:`vdst<amdgpu_synid7_vdst32_0>`,         :ref:`vaddr<amdgpu_synid7_addr_flat>`                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_dwordx2              :ref:`vdst<amdgpu_synid7_vdst64_0>`,         :ref:`vaddr<amdgpu_synid7_addr_flat>`                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_dwordx3              :ref:`vdst<amdgpu_synid7_vdst96_0>`,         :ref:`vaddr<amdgpu_synid7_addr_flat>`                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_dwordx4              :ref:`vdst<amdgpu_synid7_vdst128_0>`,         :ref:`vaddr<amdgpu_synid7_addr_flat>`                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_sbyte                :ref:`vdst<amdgpu_synid7_vdst32_0>`,         :ref:`vaddr<amdgpu_synid7_addr_flat>`                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_sshort               :ref:`vdst<amdgpu_synid7_vdst32_0>`,         :ref:`vaddr<amdgpu_synid7_addr_flat>`                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_ubyte                :ref:`vdst<amdgpu_synid7_vdst32_0>`,         :ref:`vaddr<amdgpu_synid7_addr_flat>`                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_ushort               :ref:`vdst<amdgpu_synid7_vdst32_0>`,         :ref:`vaddr<amdgpu_synid7_addr_flat>`                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_store_byte                              :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_store_dword                             :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_store_dwordx2                           :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata64_0>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_store_dwordx3                           :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata96_0>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_store_dwordx4                           :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata128_0>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_store_short                             :ref:`vaddr<amdgpu_synid7_addr_flat>`,    :ref:`vdata<amdgpu_synid7_vdata32_0>`            :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`

MIMG
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC0**       **SRC1**      **SRC2**           **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    image_atomic_add                         :ref:`vdata<amdgpu_synid7_data_mimg_atomic_reg>`::ref:`dst<amdgpu_synid7_ret>`, :ref:`vaddr<amdgpu_synid7_addr_mimg>`,    :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_and                         :ref:`vdata<amdgpu_synid7_data_mimg_atomic_reg>`::ref:`dst<amdgpu_synid7_ret>`, :ref:`vaddr<amdgpu_synid7_addr_mimg>`,    :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_cmpswap                     :ref:`vdata<amdgpu_synid7_data_mimg_atomic_cmp>`::ref:`dst<amdgpu_synid7_ret>`, :ref:`vaddr<amdgpu_synid7_addr_mimg>`,    :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_dec                         :ref:`vdata<amdgpu_synid7_data_mimg_atomic_reg>`::ref:`dst<amdgpu_synid7_ret>`, :ref:`vaddr<amdgpu_synid7_addr_mimg>`,    :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_inc                         :ref:`vdata<amdgpu_synid7_data_mimg_atomic_reg>`::ref:`dst<amdgpu_synid7_ret>`, :ref:`vaddr<amdgpu_synid7_addr_mimg>`,    :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_or                          :ref:`vdata<amdgpu_synid7_data_mimg_atomic_reg>`::ref:`dst<amdgpu_synid7_ret>`, :ref:`vaddr<amdgpu_synid7_addr_mimg>`,    :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_smax                        :ref:`vdata<amdgpu_synid7_data_mimg_atomic_reg>`::ref:`dst<amdgpu_synid7_ret>`, :ref:`vaddr<amdgpu_synid7_addr_mimg>`,    :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_smin                        :ref:`vdata<amdgpu_synid7_data_mimg_atomic_reg>`::ref:`dst<amdgpu_synid7_ret>`, :ref:`vaddr<amdgpu_synid7_addr_mimg>`,    :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_sub                         :ref:`vdata<amdgpu_synid7_data_mimg_atomic_reg>`::ref:`dst<amdgpu_synid7_ret>`, :ref:`vaddr<amdgpu_synid7_addr_mimg>`,    :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_swap                        :ref:`vdata<amdgpu_synid7_data_mimg_atomic_reg>`::ref:`dst<amdgpu_synid7_ret>`, :ref:`vaddr<amdgpu_synid7_addr_mimg>`,    :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_umax                        :ref:`vdata<amdgpu_synid7_data_mimg_atomic_reg>`::ref:`dst<amdgpu_synid7_ret>`, :ref:`vaddr<amdgpu_synid7_addr_mimg>`,    :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_umin                        :ref:`vdata<amdgpu_synid7_data_mimg_atomic_reg>`::ref:`dst<amdgpu_synid7_ret>`, :ref:`vaddr<amdgpu_synid7_addr_mimg>`,    :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_xor                         :ref:`vdata<amdgpu_synid7_data_mimg_atomic_reg>`::ref:`dst<amdgpu_synid7_ret>`, :ref:`vaddr<amdgpu_synid7_addr_mimg>`,    :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4                  :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_b                :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_b_cl             :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_b_cl_o           :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_b_o              :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c                :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_b              :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_b_cl           :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_b_cl_o         :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_b_o            :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_cl             :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_cl_o           :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_l              :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_l_o            :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_lz             :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_lz_o           :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_o              :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_cl               :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_cl_o             :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_l                :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_l_o              :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_lz               :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_lz_o             :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_o                :ref:`vdst<amdgpu_synid7_dst_mimg_gather4>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_get_lod                  :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_get_resinfo              :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`                    :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_load                     :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`                    :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_load_mip                 :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`                    :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_load_mip_pck             :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`                    :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_load_mip_pck_sgn         :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`                    :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_load_pck                 :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`                    :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_load_pck_sgn             :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`                    :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample                   :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_b                 :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_b_cl              :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_b_cl_o            :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_b_o               :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c                 :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_b               :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_b_cl            :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_b_cl_o          :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_b_o             :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_cd              :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_cd_cl           :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_cd_cl_o         :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_cd_o            :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_cl              :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_cl_o            :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_d               :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_d_cl            :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_d_cl_o          :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_d_o             :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_l               :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_l_o             :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_lz              :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_lz_o            :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_o               :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_cd                :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_cd_cl             :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_cd_cl_o           :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_cd_o              :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_cl                :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_cl_o              :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_d                 :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_d_cl              :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_d_cl_o            :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_d_o               :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_l                 :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_l_o               :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_lz                :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_lz_o              :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_o                 :ref:`vdst<amdgpu_synid7_dst_mimg_regular>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,     :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`,    :ref:`ssamp<amdgpu_synid7_samp_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_store                              :ref:`vdata<amdgpu_synid7_data_mimg_store>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,    :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_store_mip                          :ref:`vdata<amdgpu_synid7_data_mimg_store>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,    :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_store_mip_pck                      :ref:`vdata<amdgpu_synid7_data_mimg_store>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,    :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_store_pck                          :ref:`vdata<amdgpu_synid7_data_mimg_store>`,     :ref:`vaddr<amdgpu_synid7_addr_mimg>`,    :ref:`srsrc<amdgpu_synid7_rsrc_mimg>`          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`

MUBUF
-----------------------

.. parsed-literal::

    **INSTRUCTION**              **DST**   **SRC0**             **SRC1**   **SRC2**    **SRC3**    **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    buffer_atomic_add              :ref:`vdata<amdgpu_synid7_data_buf_atomic32>`::ref:`dst<amdgpu_synid7_ret>`,       :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_add_x2           :ref:`vdata<amdgpu_synid7_data_buf_atomic64>`::ref:`dst<amdgpu_synid7_ret>`,       :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_and              :ref:`vdata<amdgpu_synid7_data_buf_atomic32>`::ref:`dst<amdgpu_synid7_ret>`,       :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_and_x2           :ref:`vdata<amdgpu_synid7_data_buf_atomic64>`::ref:`dst<amdgpu_synid7_ret>`,       :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_cmpswap          :ref:`vdata<amdgpu_synid7_data_buf_atomic64>`::ref:`dst<amdgpu_synid7_ret>`::ref:`b32x2<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_cmpswap_x2       :ref:`vdata<amdgpu_synid7_data_buf_atomic128>`::ref:`dst<amdgpu_synid7_ret>`::ref:`b64x2<amdgpu_synid7_type_dev>`, :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_dec              :ref:`vdata<amdgpu_synid7_data_buf_atomic32>`::ref:`dst<amdgpu_synid7_ret>`::ref:`u32<amdgpu_synid7_type_dev>`,   :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_dec_x2           :ref:`vdata<amdgpu_synid7_data_buf_atomic64>`::ref:`dst<amdgpu_synid7_ret>`::ref:`u64<amdgpu_synid7_type_dev>`,   :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_inc              :ref:`vdata<amdgpu_synid7_data_buf_atomic32>`::ref:`dst<amdgpu_synid7_ret>`::ref:`u32<amdgpu_synid7_type_dev>`,   :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_inc_x2           :ref:`vdata<amdgpu_synid7_data_buf_atomic64>`::ref:`dst<amdgpu_synid7_ret>`::ref:`u64<amdgpu_synid7_type_dev>`,   :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_or               :ref:`vdata<amdgpu_synid7_data_buf_atomic32>`::ref:`dst<amdgpu_synid7_ret>`,       :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_or_x2            :ref:`vdata<amdgpu_synid7_data_buf_atomic64>`::ref:`dst<amdgpu_synid7_ret>`,       :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_smax             :ref:`vdata<amdgpu_synid7_data_buf_atomic32>`::ref:`dst<amdgpu_synid7_ret>`::ref:`s32<amdgpu_synid7_type_dev>`,   :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_smax_x2          :ref:`vdata<amdgpu_synid7_data_buf_atomic64>`::ref:`dst<amdgpu_synid7_ret>`::ref:`s64<amdgpu_synid7_type_dev>`,   :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_smin             :ref:`vdata<amdgpu_synid7_data_buf_atomic32>`::ref:`dst<amdgpu_synid7_ret>`::ref:`s32<amdgpu_synid7_type_dev>`,   :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_smin_x2          :ref:`vdata<amdgpu_synid7_data_buf_atomic64>`::ref:`dst<amdgpu_synid7_ret>`::ref:`s64<amdgpu_synid7_type_dev>`,   :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_sub              :ref:`vdata<amdgpu_synid7_data_buf_atomic32>`::ref:`dst<amdgpu_synid7_ret>`,       :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_sub_x2           :ref:`vdata<amdgpu_synid7_data_buf_atomic64>`::ref:`dst<amdgpu_synid7_ret>`,       :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_swap             :ref:`vdata<amdgpu_synid7_data_buf_atomic32>`::ref:`dst<amdgpu_synid7_ret>`,       :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_swap_x2          :ref:`vdata<amdgpu_synid7_data_buf_atomic64>`::ref:`dst<amdgpu_synid7_ret>`,       :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_umax             :ref:`vdata<amdgpu_synid7_data_buf_atomic32>`::ref:`dst<amdgpu_synid7_ret>`::ref:`u32<amdgpu_synid7_type_dev>`,   :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_umax_x2          :ref:`vdata<amdgpu_synid7_data_buf_atomic64>`::ref:`dst<amdgpu_synid7_ret>`::ref:`u64<amdgpu_synid7_type_dev>`,   :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_umin             :ref:`vdata<amdgpu_synid7_data_buf_atomic32>`::ref:`dst<amdgpu_synid7_ret>`::ref:`u32<amdgpu_synid7_type_dev>`,   :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_umin_x2          :ref:`vdata<amdgpu_synid7_data_buf_atomic64>`::ref:`dst<amdgpu_synid7_ret>`::ref:`u64<amdgpu_synid7_type_dev>`,   :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_xor              :ref:`vdata<amdgpu_synid7_data_buf_atomic32>`::ref:`dst<amdgpu_synid7_ret>`,       :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_xor_x2           :ref:`vdata<amdgpu_synid7_data_buf_atomic64>`::ref:`dst<amdgpu_synid7_ret>`,       :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_dword        :ref:`vdst<amdgpu_synid7_dst_buf_lds>`, :ref:`vaddr<amdgpu_synid7_addr_buf>`,           :ref:`srsrc<amdgpu_synid7_rsrc_buf>`, :ref:`soffset<amdgpu_synid7_offset_buf>`         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lds<amdgpu_synid_lds>`
    buffer_load_dwordx2      :ref:`vdst<amdgpu_synid7_dst_buf_64>`, :ref:`vaddr<amdgpu_synid7_addr_buf>`,           :ref:`srsrc<amdgpu_synid7_rsrc_buf>`, :ref:`soffset<amdgpu_synid7_offset_buf>`         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_dwordx3      :ref:`vdst<amdgpu_synid7_dst_buf_96>`, :ref:`vaddr<amdgpu_synid7_addr_buf>`,           :ref:`srsrc<amdgpu_synid7_rsrc_buf>`, :ref:`soffset<amdgpu_synid7_offset_buf>`         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_dwordx4      :ref:`vdst<amdgpu_synid7_dst_buf_128>`, :ref:`vaddr<amdgpu_synid7_addr_buf>`,           :ref:`srsrc<amdgpu_synid7_rsrc_buf>`, :ref:`soffset<amdgpu_synid7_offset_buf>`         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_format_x     :ref:`vdst<amdgpu_synid7_dst_buf_lds>`, :ref:`vaddr<amdgpu_synid7_addr_buf>`,           :ref:`srsrc<amdgpu_synid7_rsrc_buf>`, :ref:`soffset<amdgpu_synid7_offset_buf>`         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lds<amdgpu_synid_lds>`
    buffer_load_format_xy    :ref:`vdst<amdgpu_synid7_dst_buf_64>`, :ref:`vaddr<amdgpu_synid7_addr_buf>`,           :ref:`srsrc<amdgpu_synid7_rsrc_buf>`, :ref:`soffset<amdgpu_synid7_offset_buf>`         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_format_xyz   :ref:`vdst<amdgpu_synid7_dst_buf_96>`, :ref:`vaddr<amdgpu_synid7_addr_buf>`,           :ref:`srsrc<amdgpu_synid7_rsrc_buf>`, :ref:`soffset<amdgpu_synid7_offset_buf>`         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_format_xyzw  :ref:`vdst<amdgpu_synid7_dst_buf_128>`, :ref:`vaddr<amdgpu_synid7_addr_buf>`,           :ref:`srsrc<amdgpu_synid7_rsrc_buf>`, :ref:`soffset<amdgpu_synid7_offset_buf>`         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_sbyte        :ref:`vdst<amdgpu_synid7_dst_buf_lds>`, :ref:`vaddr<amdgpu_synid7_addr_buf>`,           :ref:`srsrc<amdgpu_synid7_rsrc_buf>`, :ref:`soffset<amdgpu_synid7_offset_buf>`         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lds<amdgpu_synid_lds>`
    buffer_load_sshort       :ref:`vdst<amdgpu_synid7_dst_buf_lds>`, :ref:`vaddr<amdgpu_synid7_addr_buf>`,           :ref:`srsrc<amdgpu_synid7_rsrc_buf>`, :ref:`soffset<amdgpu_synid7_offset_buf>`         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lds<amdgpu_synid_lds>`
    buffer_load_ubyte        :ref:`vdst<amdgpu_synid7_dst_buf_lds>`, :ref:`vaddr<amdgpu_synid7_addr_buf>`,           :ref:`srsrc<amdgpu_synid7_rsrc_buf>`, :ref:`soffset<amdgpu_synid7_offset_buf>`         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lds<amdgpu_synid_lds>`
    buffer_load_ushort       :ref:`vdst<amdgpu_synid7_dst_buf_lds>`, :ref:`vaddr<amdgpu_synid7_addr_buf>`,           :ref:`srsrc<amdgpu_synid7_rsrc_buf>`, :ref:`soffset<amdgpu_synid7_offset_buf>`         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lds<amdgpu_synid_lds>`
    buffer_store_byte              :ref:`vdata<amdgpu_synid7_vdata32_0>`,           :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_dword             :ref:`vdata<amdgpu_synid7_vdata32_0>`,           :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_dwordx2           :ref:`vdata<amdgpu_synid7_vdata64_0>`,           :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_dwordx3           :ref:`vdata<amdgpu_synid7_vdata96_0>`,           :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_dwordx4           :ref:`vdata<amdgpu_synid7_vdata128_0>`,           :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_format_x          :ref:`vdata<amdgpu_synid7_vdata32_0>`,           :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_format_xy         :ref:`vdata<amdgpu_synid7_vdata64_0>`,           :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_format_xyz        :ref:`vdata<amdgpu_synid7_vdata96_0>`,           :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_format_xyzw       :ref:`vdata<amdgpu_synid7_vdata128_0>`,           :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_short             :ref:`vdata<amdgpu_synid7_vdata32_0>`,           :ref:`vaddr<amdgpu_synid7_addr_buf>`, :ref:`srsrc<amdgpu_synid7_rsrc_buf>`,  :ref:`soffset<amdgpu_synid7_offset_buf>` :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_wbinvl1
    buffer_wbinvl1_vol

SMRD
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC0**      **SRC1**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    s_buffer_load_dword            :ref:`sdst<amdgpu_synid7_sdst32_0>`,     :ref:`sbase<amdgpu_synid7_base_smem_buf>`,    :ref:`soffset<amdgpu_synid7_offset_smem>`
    s_buffer_load_dwordx16         :ref:`sdst<amdgpu_synid7_sdst512_0>`,     :ref:`sbase<amdgpu_synid7_base_smem_buf>`,    :ref:`soffset<amdgpu_synid7_offset_smem>`
    s_buffer_load_dwordx2          :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`sbase<amdgpu_synid7_base_smem_buf>`,    :ref:`soffset<amdgpu_synid7_offset_smem>`
    s_buffer_load_dwordx4          :ref:`sdst<amdgpu_synid7_sdst128_0>`,     :ref:`sbase<amdgpu_synid7_base_smem_buf>`,    :ref:`soffset<amdgpu_synid7_offset_smem>`
    s_buffer_load_dwordx8          :ref:`sdst<amdgpu_synid7_sdst256_0>`,     :ref:`sbase<amdgpu_synid7_base_smem_buf>`,    :ref:`soffset<amdgpu_synid7_offset_smem>`
    s_dcache_inv
    s_dcache_inv_vol
    s_load_dword                   :ref:`sdst<amdgpu_synid7_sdst32_0>`,     :ref:`sbase<amdgpu_synid7_base_smem_addr>`,    :ref:`soffset<amdgpu_synid7_offset_smem>`
    s_load_dwordx16                :ref:`sdst<amdgpu_synid7_sdst512_0>`,     :ref:`sbase<amdgpu_synid7_base_smem_addr>`,    :ref:`soffset<amdgpu_synid7_offset_smem>`
    s_load_dwordx2                 :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`sbase<amdgpu_synid7_base_smem_addr>`,    :ref:`soffset<amdgpu_synid7_offset_smem>`
    s_load_dwordx4                 :ref:`sdst<amdgpu_synid7_sdst128_0>`,     :ref:`sbase<amdgpu_synid7_base_smem_addr>`,    :ref:`soffset<amdgpu_synid7_offset_smem>`
    s_load_dwordx8                 :ref:`sdst<amdgpu_synid7_sdst256_0>`,     :ref:`sbase<amdgpu_synid7_base_smem_addr>`,    :ref:`soffset<amdgpu_synid7_offset_smem>`
    s_memtime                      :ref:`sdst<amdgpu_synid7_sdst64_0>`

SOP1
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    s_abs_i32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`
    s_and_saveexec_b64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_andn2_saveexec_b64           :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_bcnt0_i32_b32                :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`
    s_bcnt0_i32_b64                :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_bcnt1_i32_b32                :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`
    s_bcnt1_i32_b64                :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_bitset0_b32                  :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`
    s_bitset0_b64                  :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`::ref:`b32<amdgpu_synid7_type_dev>`
    s_bitset1_b32                  :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`
    s_bitset1_b64                  :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`::ref:`b32<amdgpu_synid7_type_dev>`
    s_brev_b32                     :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`
    s_brev_b64                     :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_cbranch_join                           :ref:`ssrc<amdgpu_synid7_ssrc32_1>`
    s_cmov_b32                     :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`
    s_cmov_b64                     :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_ff0_i32_b32                  :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`
    s_ff0_i32_b64                  :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_ff1_i32_b32                  :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`
    s_ff1_i32_b64                  :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_flbit_i32                    :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`
    s_flbit_i32_b32                :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`
    s_flbit_i32_b64                :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_flbit_i32_i64                :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_getpc_b64                    :ref:`sdst<amdgpu_synid7_sdst64_1>`
    s_mov_b32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`
    s_mov_b64                      :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_mov_fed_b32                  :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`
    s_movreld_b32                  :ref:`sdst<amdgpu_synid7_sdst32_0>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`
    s_movreld_b64                  :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_movrels_b32                  :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_2>`
    s_movrels_b64                  :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_1>`
    s_nand_saveexec_b64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_nor_saveexec_b64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_not_b32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`
    s_not_b64                      :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_or_saveexec_b64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_orn2_saveexec_b64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_quadmask_b32                 :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`
    s_quadmask_b64                 :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_rfe_b64                                :ref:`ssrc<amdgpu_synid7_ssrc64_1>`
    s_setpc_b64                              :ref:`ssrc<amdgpu_synid7_ssrc64_1>`
    s_sext_i32_i16                 :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_3>`
    s_sext_i32_i8                  :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_3>`
    s_swappc_b64                   :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_1>`
    s_wqm_b32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc32_0>`
    s_wqm_b64                      :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_xnor_saveexec_b64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`
    s_xor_saveexec_b64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`ssrc<amdgpu_synid7_ssrc64_0>`

SOP2
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC0**       **SRC1**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    s_absdiff_i32                  :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_add_i32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_add_u32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_addc_u32                     :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_and_b32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_and_b64                      :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc64_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc64_0>`
    s_andn2_b32                    :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_andn2_b64                    :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc64_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc64_0>`
    s_ashr_i32                     :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`::ref:`u32<amdgpu_synid7_type_dev>`
    s_ashr_i64                     :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc64_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`::ref:`u32<amdgpu_synid7_type_dev>`
    s_bfe_i32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`::ref:`u32<amdgpu_synid7_type_dev>`
    s_bfe_i64                      :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc64_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`::ref:`u32<amdgpu_synid7_type_dev>`
    s_bfe_u32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_bfe_u64                      :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc64_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`::ref:`u32<amdgpu_synid7_type_dev>`
    s_bfm_b32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_bfm_b64                      :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`::ref:`b32<amdgpu_synid7_type_dev>`, :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`::ref:`b32<amdgpu_synid7_type_dev>`
    s_cbranch_g_fork                         :ref:`ssrc0<amdgpu_synid7_ssrc64_2>`,     :ref:`ssrc1<amdgpu_synid7_ssrc64_2>`
    s_cselect_b32                  :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_cselect_b64                  :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc64_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc64_0>`
    s_lshl_b32                     :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`::ref:`u32<amdgpu_synid7_type_dev>`
    s_lshl_b64                     :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc64_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`::ref:`u32<amdgpu_synid7_type_dev>`
    s_lshr_b32                     :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`::ref:`u32<amdgpu_synid7_type_dev>`
    s_lshr_b64                     :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc64_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`::ref:`u32<amdgpu_synid7_type_dev>`
    s_max_i32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_max_u32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_min_i32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_min_u32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_mul_i32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_nand_b32                     :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_nand_b64                     :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc64_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc64_0>`
    s_nor_b32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_nor_b64                      :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc64_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc64_0>`
    s_or_b32                       :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_or_b64                       :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc64_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc64_0>`
    s_orn2_b32                     :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_orn2_b64                     :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc64_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc64_0>`
    s_sub_i32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_sub_u32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_subb_u32                     :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_xnor_b32                     :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_xnor_b64                     :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc64_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc64_0>`
    s_xor_b32                      :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_xor_b64                      :ref:`sdst<amdgpu_synid7_sdst64_1>`,     :ref:`ssrc0<amdgpu_synid7_ssrc64_0>`,     :ref:`ssrc1<amdgpu_synid7_ssrc64_0>`

SOPC
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **SRC0**      **SRC1**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    s_bitcmp0_b32                  :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_bitcmp0_b64                  :ref:`ssrc0<amdgpu_synid7_ssrc64_0>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`::ref:`u32<amdgpu_synid7_type_dev>`
    s_bitcmp1_b32                  :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_bitcmp1_b64                  :ref:`ssrc0<amdgpu_synid7_ssrc64_0>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`::ref:`u32<amdgpu_synid7_type_dev>`
    s_cmp_eq_i32                   :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_cmp_eq_u32                   :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_cmp_ge_i32                   :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_cmp_ge_u32                   :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_cmp_gt_i32                   :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_cmp_gt_u32                   :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_cmp_le_i32                   :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_cmp_le_u32                   :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_cmp_lg_i32                   :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_cmp_lg_u32                   :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_cmp_lt_i32                   :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_cmp_lt_u32                   :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`
    s_setvskip                     :ref:`ssrc0<amdgpu_synid7_ssrc32_0>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_0>`

SOPK
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC0**      **SRC1**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    s_addk_i32                     :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`imm16<amdgpu_synid7_simm16>`
    s_cbranch_i_fork                         :ref:`ssrc<amdgpu_synid7_ssrc64_3>`,     :ref:`label<amdgpu_synid7_label>`
    s_cmovk_i32                    :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`imm16<amdgpu_synid7_simm16>`
    s_cmpk_eq_i32                            :ref:`ssrc<amdgpu_synid7_ssrc32_4>`,     :ref:`imm16<amdgpu_synid7_simm16>`
    s_cmpk_eq_u32                            :ref:`ssrc<amdgpu_synid7_ssrc32_4>`,     :ref:`imm16<amdgpu_synid7_uimm16>`
    s_cmpk_ge_i32                            :ref:`ssrc<amdgpu_synid7_ssrc32_4>`,     :ref:`imm16<amdgpu_synid7_simm16>`
    s_cmpk_ge_u32                            :ref:`ssrc<amdgpu_synid7_ssrc32_4>`,     :ref:`imm16<amdgpu_synid7_uimm16>`
    s_cmpk_gt_i32                            :ref:`ssrc<amdgpu_synid7_ssrc32_4>`,     :ref:`imm16<amdgpu_synid7_simm16>`
    s_cmpk_gt_u32                            :ref:`ssrc<amdgpu_synid7_ssrc32_4>`,     :ref:`imm16<amdgpu_synid7_uimm16>`
    s_cmpk_le_i32                            :ref:`ssrc<amdgpu_synid7_ssrc32_4>`,     :ref:`imm16<amdgpu_synid7_simm16>`
    s_cmpk_le_u32                            :ref:`ssrc<amdgpu_synid7_ssrc32_4>`,     :ref:`imm16<amdgpu_synid7_uimm16>`
    s_cmpk_lg_i32                            :ref:`ssrc<amdgpu_synid7_ssrc32_4>`,     :ref:`imm16<amdgpu_synid7_simm16>`
    s_cmpk_lg_u32                            :ref:`ssrc<amdgpu_synid7_ssrc32_4>`,     :ref:`imm16<amdgpu_synid7_uimm16>`
    s_cmpk_lt_i32                            :ref:`ssrc<amdgpu_synid7_ssrc32_4>`,     :ref:`imm16<amdgpu_synid7_simm16>`
    s_cmpk_lt_u32                            :ref:`ssrc<amdgpu_synid7_ssrc32_4>`,     :ref:`imm16<amdgpu_synid7_uimm16>`
    s_getreg_b32                   :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`hwreg<amdgpu_synid7_hwreg>`
    s_movk_i32                     :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`imm16<amdgpu_synid7_simm16>`
    s_mulk_i32                     :ref:`sdst<amdgpu_synid7_sdst32_1>`,     :ref:`imm16<amdgpu_synid7_simm16>`
    s_setreg_b32                   :ref:`hwreg<amdgpu_synid7_hwreg>`,    :ref:`ssrc<amdgpu_synid7_ssrc32_4>`
    s_setreg_imm32_b32             :ref:`hwreg<amdgpu_synid7_hwreg>`,    :ref:`imm32<amdgpu_synid7_bimm32>`

SOPP
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **SRC**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    s_barrier
    s_branch                       :ref:`label<amdgpu_synid7_label>`
    s_cbranch_cdbgsys              :ref:`label<amdgpu_synid7_label>`
    s_cbranch_cdbgsys_and_user     :ref:`label<amdgpu_synid7_label>`
    s_cbranch_cdbgsys_or_user      :ref:`label<amdgpu_synid7_label>`
    s_cbranch_cdbguser             :ref:`label<amdgpu_synid7_label>`
    s_cbranch_execnz               :ref:`label<amdgpu_synid7_label>`
    s_cbranch_execz                :ref:`label<amdgpu_synid7_label>`
    s_cbranch_scc0                 :ref:`label<amdgpu_synid7_label>`
    s_cbranch_scc1                 :ref:`label<amdgpu_synid7_label>`
    s_cbranch_vccnz                :ref:`label<amdgpu_synid7_label>`
    s_cbranch_vccz                 :ref:`label<amdgpu_synid7_label>`
    s_decperflevel                 :ref:`imm16<amdgpu_synid7_bimm16>`
    s_endpgm
    s_icache_inv
    s_incperflevel                 :ref:`imm16<amdgpu_synid7_bimm16>`
    s_nop                          :ref:`imm16<amdgpu_synid7_bimm16>`
    s_sendmsg                      :ref:`msg<amdgpu_synid7_msg>`
    s_sendmsghalt                  :ref:`msg<amdgpu_synid7_msg>`
    s_sethalt                      :ref:`imm16<amdgpu_synid7_bimm16>`
    s_setkill                      :ref:`imm16<amdgpu_synid7_bimm16>`
    s_setprio                      :ref:`imm16<amdgpu_synid7_bimm16>`
    s_sleep                        :ref:`imm16<amdgpu_synid7_bimm16>`
    s_trap                         :ref:`imm16<amdgpu_synid7_bimm16>`
    s_ttracedata
    s_waitcnt                      :ref:`waitcnt<amdgpu_synid7_waitcnt>`

VINTRP
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC0**       **SRC1**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    v_interp_mov_f32               :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`param<amdgpu_synid7_param>`::ref:`b32<amdgpu_synid7_type_dev>`, :ref:`attr<amdgpu_synid7_attr>`::ref:`b32<amdgpu_synid7_type_dev>`
    v_interp_p1_f32                :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`vsrc<amdgpu_synid7_vsrc32_0>`,      :ref:`attr<amdgpu_synid7_attr>`::ref:`b32<amdgpu_synid7_type_dev>`
    v_interp_p2_f32                :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`vsrc<amdgpu_synid7_vsrc32_0>`,      :ref:`attr<amdgpu_synid7_attr>`::ref:`b32<amdgpu_synid7_type_dev>`

VOP1
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    v_bfrev_b32                    :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_ceil_f32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_ceil_f64                     :ref:`vdst<amdgpu_synid7_vdst64_0>`,     :ref:`src<amdgpu_synid7_src64_0>`
    v_clrexcp
    v_cos_f32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_cvt_f16_f32                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_cvt_f32_f16                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_1>`
    v_cvt_f32_f64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src64_0>`
    v_cvt_f32_i32                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_cvt_f32_u32                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_cvt_f32_ubyte0               :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_cvt_f32_ubyte1               :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_cvt_f32_ubyte2               :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_cvt_f32_ubyte3               :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_cvt_f64_f32                  :ref:`vdst<amdgpu_synid7_vdst64_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_cvt_f64_i32                  :ref:`vdst<amdgpu_synid7_vdst64_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_cvt_f64_u32                  :ref:`vdst<amdgpu_synid7_vdst64_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_cvt_flr_i32_f32              :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_cvt_i32_f32                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_cvt_i32_f64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src64_0>`
    v_cvt_off_f32_i4               :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_cvt_rpi_i32_f32              :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_cvt_u32_f32                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_cvt_u32_f64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src64_0>`
    v_exp_f32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_exp_legacy_f32               :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_ffbh_i32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_ffbh_u32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_ffbl_b32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_floor_f32                    :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_floor_f64                    :ref:`vdst<amdgpu_synid7_vdst64_0>`,     :ref:`src<amdgpu_synid7_src64_0>`
    v_fract_f32                    :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_fract_f64                    :ref:`vdst<amdgpu_synid7_vdst64_0>`,     :ref:`src<amdgpu_synid7_src64_0>`
    v_frexp_exp_i32_f32            :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_frexp_exp_i32_f64            :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src64_0>`
    v_frexp_mant_f32               :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_frexp_mant_f64               :ref:`vdst<amdgpu_synid7_vdst64_0>`,     :ref:`src<amdgpu_synid7_src64_0>`
    v_log_clamp_f32                :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_log_f32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_log_legacy_f32               :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_mov_b32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_mov_fed_b32                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_movreld_b32                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_movrels_b32                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`vsrc<amdgpu_synid7_vsrc32_0>`
    v_movrelsd_b32                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`vsrc<amdgpu_synid7_vsrc32_0>`
    v_nop
    v_not_b32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_rcp_clamp_f32                :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_rcp_clamp_f64                :ref:`vdst<amdgpu_synid7_vdst64_0>`,     :ref:`src<amdgpu_synid7_src64_0>`
    v_rcp_f32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_rcp_f64                      :ref:`vdst<amdgpu_synid7_vdst64_0>`,     :ref:`src<amdgpu_synid7_src64_0>`
    v_rcp_iflag_f32                :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_rcp_legacy_f32               :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_readfirstlane_b32            :ref:`sdst<amdgpu_synid7_sdst32_2>`,     :ref:`vsrc<amdgpu_synid7_vsrc32_1>`
    v_rndne_f32                    :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_rndne_f64                    :ref:`vdst<amdgpu_synid7_vdst64_0>`,     :ref:`src<amdgpu_synid7_src64_0>`
    v_rsq_clamp_f32                :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_rsq_clamp_f64                :ref:`vdst<amdgpu_synid7_vdst64_0>`,     :ref:`src<amdgpu_synid7_src64_0>`
    v_rsq_f32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_rsq_f64                      :ref:`vdst<amdgpu_synid7_vdst64_0>`,     :ref:`src<amdgpu_synid7_src64_0>`
    v_rsq_legacy_f32               :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_sin_f32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_sqrt_f32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_sqrt_f64                     :ref:`vdst<amdgpu_synid7_vdst64_0>`,     :ref:`src<amdgpu_synid7_src64_0>`
    v_trunc_f32                    :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`src<amdgpu_synid7_src32_0>`
    v_trunc_f64                    :ref:`vdst<amdgpu_synid7_vdst64_0>`,     :ref:`src<amdgpu_synid7_src64_0>`

VOP2
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST0**      **DST1**      **SRC0**      **SRC1**      **SRC2**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    v_add_f32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_add_i32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_addc_u32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`,    :ref:`vcc<amdgpu_synid7_vcc_64>`
    v_and_b32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_ashr_i32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`::ref:`u32<amdgpu_synid7_type_dev>`
    v_ashrrev_i32                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_2>`::ref:`u32<amdgpu_synid7_type_dev>`, :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_bcnt_u32_b32                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_bfm_b32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cndmask_b32                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`,    :ref:`vcc<amdgpu_synid7_vcc_64>`
    v_cvt_pk_i16_i32               :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cvt_pk_u16_u32               :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cvt_pkaccum_u8_f32           :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`::ref:`u32<amdgpu_synid7_type_dev>`
    v_cvt_pknorm_i16_f32           :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cvt_pknorm_u16_f32           :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cvt_pkrtz_f16_f32            :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_ldexp_f32                    :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`::ref:`i32<amdgpu_synid7_type_dev>`
    v_lshl_b32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`::ref:`u32<amdgpu_synid7_type_dev>`
    v_lshlrev_b32                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_2>`::ref:`u32<amdgpu_synid7_type_dev>`, :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_lshr_b32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`::ref:`u32<amdgpu_synid7_type_dev>`
    v_lshrrev_b32                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_2>`::ref:`u32<amdgpu_synid7_type_dev>`, :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_mac_f32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_mac_legacy_f32               :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_madak_f32                    :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`,    :ref:`imm32<amdgpu_synid7_fimm32>`
    v_madmk_f32                    :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`imm32<amdgpu_synid7_fimm32>`,    :ref:`vsrc2<amdgpu_synid7_vsrc32_0>`
    v_max_f32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_max_i32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_max_legacy_f32               :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_max_u32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_mbcnt_hi_u32_b32             :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_mbcnt_lo_u32_b32             :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_min_f32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_min_i32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_min_legacy_f32               :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_min_u32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_mul_f32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_mul_hi_i32_i24               :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_mul_hi_u32_u24               :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_mul_i32_i24                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_mul_legacy_f32               :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_mul_u32_u24                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_or_b32                       :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_readlane_b32                 :ref:`sdst<amdgpu_synid7_sdst32_2>`,               :ref:`vsrc0<amdgpu_synid7_vsrc32_1>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_5>`
    v_sub_f32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_sub_i32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_subb_u32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`,    :ref:`vcc<amdgpu_synid7_vcc_64>`
    v_subbrev_u32                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_2>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`,    :ref:`vcc<amdgpu_synid7_vcc_64>`
    v_subrev_f32                   :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_2>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_subrev_i32                   :ref:`vdst<amdgpu_synid7_vdst32_0>`,     :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_2>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_writelane_b32                :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`ssrc0<amdgpu_synid7_ssrc32_6>`,    :ref:`ssrc1<amdgpu_synid7_ssrc32_5>`
    v_xor_b32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,               :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`

VOP3
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST0**       **DST1**      **SRC0**        **SRC1**        **SRC2**            **MODIFIERS**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    v_add_f32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_add_f64                      :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_add_i32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,      :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_addc_u32_e64                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,      :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`ssrc2<amdgpu_synid7_ssrc64_1>`
    v_alignbit_b32                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`src2<amdgpu_synid7_src32_4>`
    v_alignbyte_b32                :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`src2<amdgpu_synid7_src32_4>`
    v_and_b32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_ashr_i32_e64                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`
    v_ashr_i64                     :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`
    v_ashrrev_i32_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`,   :ref:`src1<amdgpu_synid7_src32_4>`
    v_bcnt_u32_b32_e64             :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_bfe_i32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`,   :ref:`src2<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`
    v_bfe_u32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`src2<amdgpu_synid7_src32_4>`
    v_bfi_b32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`src2<amdgpu_synid7_src32_4>`
    v_bfm_b32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_bfrev_b32_e64                :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`
    v_ceil_f32_e64                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_ceil_f64_e64                 :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_clrexcp_e64
    v_cmp_class_f32_e64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`b32<amdgpu_synid7_type_dev>`
    v_cmp_class_f64_e64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`b32<amdgpu_synid7_type_dev>`
    v_cmp_eq_f32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_eq_f64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_eq_i32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmp_eq_i64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmp_eq_u32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmp_eq_u64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmp_f_f32_e64                :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_f_f64_e64                :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_f_i32_e64                :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmp_f_i64_e64                :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmp_f_u32_e64                :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmp_f_u64_e64                :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmp_ge_f32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_ge_f64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_ge_i32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmp_ge_i64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmp_ge_u32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmp_ge_u64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmp_gt_f32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_gt_f64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_gt_i32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmp_gt_i64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmp_gt_u32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmp_gt_u64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmp_le_f32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_le_f64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_le_i32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmp_le_i64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmp_le_u32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmp_le_u64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmp_lg_f32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_lg_f64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_lt_f32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_lt_f64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_lt_i32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmp_lt_i64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmp_lt_u32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmp_lt_u64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmp_ne_i32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmp_ne_i64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmp_ne_u32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmp_ne_u64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmp_neq_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_neq_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_nge_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_nge_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_ngt_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_ngt_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_nle_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_nle_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_nlg_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_nlg_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_nlt_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_nlt_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_o_f32_e64                :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_o_f64_e64                :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_t_i32_e64                :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmp_t_i64_e64                :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmp_t_u32_e64                :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmp_t_u64_e64                :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmp_tru_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_tru_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_u_f32_e64                :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmp_u_f64_e64                :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_eq_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_eq_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_f_f32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_f_f64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_ge_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_ge_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_gt_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_gt_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_le_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_le_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_lg_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_lg_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_lt_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_lt_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_neq_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_neq_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_nge_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_nge_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_ngt_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_ngt_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_nle_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_nle_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_nlg_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_nlg_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_nlt_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_nlt_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_o_f32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_o_f64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_tru_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_tru_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_u_f32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmps_u_f64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_eq_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_eq_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_f_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_f_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_ge_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_ge_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_gt_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_gt_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_le_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_le_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_lg_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_lg_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_lt_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_lt_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_neq_f32_e64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_neq_f64_e64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_nge_f32_e64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_nge_f64_e64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_ngt_f32_e64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_ngt_f64_e64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_nle_f32_e64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_nle_f64_e64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_nlg_f32_e64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_nlg_f64_e64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_nlt_f32_e64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_nlt_f64_e64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_o_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_o_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_tru_f32_e64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_tru_f64_e64            :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_u_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpsx_u_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_class_f32_e64           :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`b32<amdgpu_synid7_type_dev>`
    v_cmpx_class_f64_e64           :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`b32<amdgpu_synid7_type_dev>`
    v_cmpx_eq_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_eq_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_eq_i32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmpx_eq_i64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmpx_eq_u32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmpx_eq_u64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmpx_f_f32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_f_f64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_f_i32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmpx_f_i64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmpx_f_u32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmpx_f_u64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmpx_ge_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_ge_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_ge_i32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmpx_ge_i64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmpx_ge_u32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmpx_ge_u64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmpx_gt_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_gt_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_gt_i32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmpx_gt_i64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmpx_gt_u32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmpx_gt_u64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmpx_le_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_le_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_le_i32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmpx_le_i64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmpx_le_u32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmpx_le_u64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmpx_lg_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_lg_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_lt_f32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_lt_f64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_lt_i32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmpx_lt_i64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmpx_lt_u32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmpx_lt_u64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmpx_ne_i32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmpx_ne_i64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmpx_ne_u32_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmpx_ne_u64_e64              :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmpx_neq_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_neq_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_nge_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_nge_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_ngt_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_ngt_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_nle_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_nle_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_nlg_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_nlg_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_nlt_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_nlt_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_o_f32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_o_f64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_t_i32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmpx_t_i64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmpx_t_u32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cmpx_t_u64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`
    v_cmpx_tru_f32_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_tru_f64_e64             :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_u_f32_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cmpx_u_f64_e64               :ref:`sdst<amdgpu_synid7_sdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cndmask_b32_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`ssrc2<amdgpu_synid7_ssrc64_1>`
    v_cos_f32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cubeid_f32                   :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src2<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cubema_f32                   :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src2<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cubesc_f32                   :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src2<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cubetc_f32                   :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src2<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f16_f32_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`
    v_cvt_f32_f16_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_5>`                                     :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f32_f64_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f32_i32_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`                                     :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f32_u32_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`                                     :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f32_ubyte0_e64           :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`
    v_cvt_f32_ubyte1_e64           :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`
    v_cvt_f32_ubyte2_e64           :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`
    v_cvt_f32_ubyte3_e64           :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`
    v_cvt_f64_f32_e64              :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f64_i32_e64              :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src<amdgpu_synid7_src32_3>`                                     :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f64_u32_e64              :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src<amdgpu_synid7_src32_3>`                                     :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_flr_i32_f32_e64          :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`
    v_cvt_i32_f32_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`
    v_cvt_i32_f64_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_cvt_off_f32_i4_e64           :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`                                     :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_pk_i16_i32_e64           :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cvt_pk_u16_u32_e64           :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_cvt_pk_u8_f32                :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`,   :ref:`src2<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`
    v_cvt_pkaccum_u8_f32_e64       :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`
    v_cvt_pknorm_i16_f32_e64       :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cvt_pknorm_u16_f32_e64       :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cvt_pkrtz_f16_f32_e64        :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`
    v_cvt_rpi_i32_f32_e64          :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`
    v_cvt_u32_f32_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`
    v_cvt_u32_f64_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_div_fixup_f32                :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src2<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_div_fixup_f64                :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src2<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_div_fmas_f32                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src2<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_div_fmas_f64                 :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src2<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_div_scale_f32                :ref:`vdst<amdgpu_synid7_vdst32_0>`,      :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`src2<amdgpu_synid7_src32_4>`
    v_div_scale_f64                :ref:`vdst<amdgpu_synid7_vdst64_0>`,      :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src64_1>`,       :ref:`src2<amdgpu_synid7_src64_1>`
    v_exp_f32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_exp_legacy_f32_e64           :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_ffbh_i32_e64                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`
    v_ffbh_u32_e64                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`
    v_ffbl_b32_e64                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`
    v_floor_f32_e64                :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_floor_f64_e64                :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_fma_f32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src2<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_fma_f64                      :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src2<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_fract_f32_e64                :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_fract_f64_e64                :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_frexp_exp_i32_f32_e64        :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`
    v_frexp_exp_i32_f64_e64        :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`
    v_frexp_mant_f32_e64           :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`
    v_frexp_mant_f64_e64           :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_ldexp_f32_e64                :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`i32<amdgpu_synid7_type_dev>`                    :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_ldexp_f64                    :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`i32<amdgpu_synid7_type_dev>`                    :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_lerp_u8                      :ref:`vdst<amdgpu_synid7_vdst32_0>`::ref:`u32<amdgpu_synid7_type_dev>`,            :ref:`src0<amdgpu_synid7_src32_1>`::ref:`b32<amdgpu_synid7_type_dev>`,   :ref:`src1<amdgpu_synid7_src32_6>`::ref:`b32<amdgpu_synid7_type_dev>`,   :ref:`src2<amdgpu_synid7_src32_6>`::ref:`b32<amdgpu_synid7_type_dev>`
    v_log_clamp_f32_e64            :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_log_f32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_log_legacy_f32_e64           :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_lshl_b32_e64                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`
    v_lshl_b64                     :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`
    v_lshlrev_b32_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`,   :ref:`src1<amdgpu_synid7_src32_4>`
    v_lshr_b32_e64                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`
    v_lshr_b64                     :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`,       :ref:`src1<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`
    v_lshrrev_b32_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`,   :ref:`src1<amdgpu_synid7_src32_4>`
    v_mac_f32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mac_legacy_f32_e64           :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mad_f32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src2<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mad_i32_i24                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`src2<amdgpu_synid7_src32_4>`::ref:`i32<amdgpu_synid7_type_dev>`
    v_mad_i64_i32                  :ref:`vdst<amdgpu_synid7_vdst64_0>`,      :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`src2<amdgpu_synid7_src64_1>`::ref:`i64<amdgpu_synid7_type_dev>`
    v_mad_legacy_f32               :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src2<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mad_u32_u24                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`src2<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`
    v_mad_u64_u32                  :ref:`vdst<amdgpu_synid7_vdst64_0>`,      :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`src2<amdgpu_synid7_src64_1>`::ref:`u64<amdgpu_synid7_type_dev>`
    v_max3_f32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src2<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_max3_i32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`src2<amdgpu_synid7_src32_4>`
    v_max3_u32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`src2<amdgpu_synid7_src32_4>`
    v_max_f32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_max_f64                      :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_max_i32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_max_legacy_f32_e64           :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_max_u32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_mbcnt_hi_u32_b32_e64         :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_mbcnt_lo_u32_b32_e64         :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_med3_f32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src2<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_med3_i32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`src2<amdgpu_synid7_src32_4>`
    v_med3_u32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`src2<amdgpu_synid7_src32_4>`
    v_min3_f32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src2<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_min3_i32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`src2<amdgpu_synid7_src32_4>`
    v_min3_u32                     :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`src2<amdgpu_synid7_src32_4>`
    v_min_f32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_min_f64                      :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_min_i32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_min_legacy_f32_e64           :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_min_u32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_mov_b32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`
    v_mov_fed_b32_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`
    v_movreld_b32_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`
    v_movrels_b32_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`vsrc<amdgpu_synid7_vsrc32_0>`
    v_movrelsd_b32_e64             :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`vsrc<amdgpu_synid7_vsrc32_0>`
    v_mqsad_pk_u16_u8              :ref:`vdst<amdgpu_synid7_vdst64_0>`::ref:`b64<amdgpu_synid7_type_dev>`,            :ref:`src0<amdgpu_synid7_src64_2>`::ref:`b64<amdgpu_synid7_type_dev>`,   :ref:`src1<amdgpu_synid7_src32_6>`::ref:`b32<amdgpu_synid7_type_dev>`,   :ref:`src2<amdgpu_synid7_src64_2>`::ref:`b64<amdgpu_synid7_type_dev>`
    v_mqsad_u32_u8                 :ref:`vdst<amdgpu_synid7_vdst128_0>`::ref:`b128<amdgpu_synid7_type_dev>`,           :ref:`src0<amdgpu_synid7_src64_2>`::ref:`b64<amdgpu_synid7_type_dev>`,   :ref:`src1<amdgpu_synid7_src32_6>`::ref:`b32<amdgpu_synid7_type_dev>`,   :ref:`vsrc2<amdgpu_synid7_vsrc128_0>`::ref:`b128<amdgpu_synid7_type_dev>`
    v_msad_u8                      :ref:`vdst<amdgpu_synid7_vdst32_0>`::ref:`u32<amdgpu_synid7_type_dev>`,            :ref:`src0<amdgpu_synid7_src32_1>`::ref:`b32<amdgpu_synid7_type_dev>`,   :ref:`src1<amdgpu_synid7_src32_6>`::ref:`b32<amdgpu_synid7_type_dev>`,   :ref:`src2<amdgpu_synid7_src32_6>`::ref:`b32<amdgpu_synid7_type_dev>`
    v_mul_f32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mul_f64                      :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mul_hi_i32                   :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_mul_hi_i32_i24_e64           :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_mul_hi_u32                   :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_mul_hi_u32_u24_e64           :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_mul_i32_i24_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_mul_legacy_f32_e64           :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mul_lo_i32                   :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_mul_lo_u32                   :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_mul_u32_u24_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_mullit_f32                   :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src2<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_nop_e64
    v_not_b32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`
    v_or_b32_e64                   :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_qsad_pk_u16_u8               :ref:`vdst<amdgpu_synid7_vdst64_0>`::ref:`b64<amdgpu_synid7_type_dev>`,            :ref:`src0<amdgpu_synid7_src64_2>`::ref:`b64<amdgpu_synid7_type_dev>`,   :ref:`src1<amdgpu_synid7_src32_6>`::ref:`b32<amdgpu_synid7_type_dev>`,   :ref:`src2<amdgpu_synid7_src64_2>`::ref:`b64<amdgpu_synid7_type_dev>`
    v_rcp_clamp_f32_e64            :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rcp_clamp_f64_e64            :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rcp_f32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rcp_f64_e64                  :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rcp_iflag_f32_e64            :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rcp_legacy_f32_e64           :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rndne_f32_e64                :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rndne_f64_e64                :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rsq_clamp_f32_e64            :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rsq_clamp_f64_e64            :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rsq_f32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rsq_f64_e64                  :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rsq_legacy_f32_e64           :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_sad_hi_u8                    :ref:`vdst<amdgpu_synid7_vdst32_0>`::ref:`u32<amdgpu_synid7_type_dev>`,            :ref:`src0<amdgpu_synid7_src32_1>`::ref:`u8x4<amdgpu_synid7_type_dev>`,  :ref:`src1<amdgpu_synid7_src32_6>`::ref:`u8x4<amdgpu_synid7_type_dev>`,  :ref:`src2<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`
    v_sad_u16                      :ref:`vdst<amdgpu_synid7_vdst32_0>`::ref:`u32<amdgpu_synid7_type_dev>`,            :ref:`src0<amdgpu_synid7_src32_1>`::ref:`u16x2<amdgpu_synid7_type_dev>`, :ref:`src1<amdgpu_synid7_src32_6>`::ref:`u16x2<amdgpu_synid7_type_dev>`, :ref:`src2<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`
    v_sad_u32                      :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`src2<amdgpu_synid7_src32_4>`
    v_sad_u8                       :ref:`vdst<amdgpu_synid7_vdst32_0>`::ref:`u32<amdgpu_synid7_type_dev>`,            :ref:`src0<amdgpu_synid7_src32_1>`::ref:`u8x4<amdgpu_synid7_type_dev>`,  :ref:`src1<amdgpu_synid7_src32_6>`::ref:`u8x4<amdgpu_synid7_type_dev>`,  :ref:`src2<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`
    v_sin_f32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_sqrt_f32_e64                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_sqrt_f64_e64                 :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_sub_f32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_sub_i32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,      :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_subb_u32_e64                 :ref:`vdst<amdgpu_synid7_vdst32_0>`,      :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`ssrc2<amdgpu_synid7_ssrc64_1>`
    v_subbrev_u32_e64              :ref:`vdst<amdgpu_synid7_vdst32_0>`,      :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`src0<amdgpu_synid7_src32_4>`,       :ref:`src1<amdgpu_synid7_src32_4>`,       :ref:`ssrc2<amdgpu_synid7_ssrc64_1>`
    v_subrev_f32_e64               :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`m<amdgpu_synid7_mod>`                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_subrev_i32_e64               :ref:`vdst<amdgpu_synid7_vdst32_0>`,      :ref:`sdst<amdgpu_synid7_sdst64_0>`,     :ref:`src0<amdgpu_synid7_src32_4>`,       :ref:`src1<amdgpu_synid7_src32_4>`
    v_trig_preop_f64               :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src0<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`,     :ref:`src1<amdgpu_synid7_src32_4>`::ref:`u32<amdgpu_synid7_type_dev>`                    :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_trunc_f32_e64                :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src<amdgpu_synid7_src32_3>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_trunc_f64_e64                :ref:`vdst<amdgpu_synid7_vdst64_0>`,                :ref:`src<amdgpu_synid7_src64_1>`::ref:`m<amdgpu_synid7_mod>`                                   :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_xor_b32_e64                  :ref:`vdst<amdgpu_synid7_vdst32_0>`,                :ref:`src0<amdgpu_synid7_src32_3>`,       :ref:`src1<amdgpu_synid7_src32_4>`

VOPC
-----------------------

.. parsed-literal::

    **INSTRUCTION**                    **DST**       **SRC0**      **SRC1**
    \ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|\ |---|
    v_cmp_class_f32                :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`::ref:`b32<amdgpu_synid7_type_dev>`
    v_cmp_class_f64                :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`::ref:`b32<amdgpu_synid7_type_dev>`
    v_cmp_eq_f32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_eq_f64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_eq_i32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_eq_i64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_eq_u32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_eq_u64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_f_f32                    :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_f_f64                    :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_f_i32                    :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_f_i64                    :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_f_u32                    :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_f_u64                    :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_ge_f32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_ge_f64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_ge_i32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_ge_i64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_ge_u32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_ge_u64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_gt_f32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_gt_f64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_gt_i32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_gt_i64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_gt_u32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_gt_u64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_le_f32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_le_f64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_le_i32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_le_i64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_le_u32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_le_u64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_lg_f32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_lg_f64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_lt_f32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_lt_f64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_lt_i32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_lt_i64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_lt_u32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_lt_u64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_ne_i32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_ne_i64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_ne_u32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_ne_u64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_neq_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_neq_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_nge_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_nge_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_ngt_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_ngt_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_nle_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_nle_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_nlg_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_nlg_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_nlt_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_nlt_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_o_f32                    :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_o_f64                    :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_t_i32                    :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_t_i64                    :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_t_u32                    :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_t_u64                    :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_tru_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_tru_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmp_u_f32                    :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmp_u_f64                    :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmps_eq_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmps_eq_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmps_f_f32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmps_f_f64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmps_ge_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmps_ge_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmps_gt_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmps_gt_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmps_le_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmps_le_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmps_lg_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmps_lg_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmps_lt_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmps_lt_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmps_neq_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmps_neq_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmps_nge_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmps_nge_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmps_ngt_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmps_ngt_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmps_nle_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmps_nle_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmps_nlg_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmps_nlg_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmps_nlt_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmps_nlt_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmps_o_f32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmps_o_f64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmps_tru_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmps_tru_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmps_u_f32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmps_u_f64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpsx_eq_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpsx_eq_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpsx_f_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpsx_f_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpsx_ge_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpsx_ge_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpsx_gt_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpsx_gt_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpsx_le_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpsx_le_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpsx_lg_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpsx_lg_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpsx_lt_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpsx_lt_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpsx_neq_f32                :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpsx_neq_f64                :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpsx_nge_f32                :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpsx_nge_f64                :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpsx_ngt_f32                :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpsx_ngt_f64                :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpsx_nle_f32                :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpsx_nle_f64                :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpsx_nlg_f32                :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpsx_nlg_f64                :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpsx_nlt_f32                :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpsx_nlt_f64                :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpsx_o_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpsx_o_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpsx_tru_f32                :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpsx_tru_f64                :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpsx_u_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpsx_u_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_class_f32               :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`::ref:`b32<amdgpu_synid7_type_dev>`
    v_cmpx_class_f64               :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`::ref:`b32<amdgpu_synid7_type_dev>`
    v_cmpx_eq_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_eq_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_eq_i32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_eq_i64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_eq_u32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_eq_u64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_f_f32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_f_f64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_f_i32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_f_i64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_f_u32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_f_u64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_ge_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_ge_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_ge_i32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_ge_i64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_ge_u32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_ge_u64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_gt_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_gt_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_gt_i32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_gt_i64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_gt_u32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_gt_u64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_le_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_le_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_le_i32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_le_i64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_le_u32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_le_u64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_lg_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_lg_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_lt_f32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_lt_f64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_lt_i32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_lt_i64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_lt_u32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_lt_u64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_ne_i32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_ne_i64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_ne_u32                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_ne_u64                  :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_neq_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_neq_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_nge_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_nge_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_ngt_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_ngt_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_nle_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_nle_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_nlg_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_nlg_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_nlt_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_nlt_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_o_f32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_o_f64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_t_i32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_t_i64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_t_u32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_t_u64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_tru_f32                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_tru_f64                 :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`
    v_cmpx_u_f32                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src32_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc32_0>`
    v_cmpx_u_f64                   :ref:`vcc<amdgpu_synid7_vcc_64>`,      :ref:`src0<amdgpu_synid7_src64_0>`,     :ref:`vsrc1<amdgpu_synid7_vsrc64_0>`

.. |---| unicode:: U+02014 .. em dash


.. toctree::
    :hidden:

    gfx7_attr
    gfx7_bimm16
    gfx7_bimm32
    gfx7_fimm32
    gfx7_hwreg
    gfx7_label
    gfx7_msg
    gfx7_param
    gfx7_simm16
    gfx7_tgt
    gfx7_uimm16
    gfx7_waitcnt
    gfx7_addr_buf
    gfx7_addr_ds
    gfx7_addr_flat
    gfx7_addr_mimg
    gfx7_base_smem_addr
    gfx7_base_smem_buf
    gfx7_data_buf_atomic128
    gfx7_data_buf_atomic32
    gfx7_data_buf_atomic64
    gfx7_data_mimg_atomic_cmp
    gfx7_data_mimg_atomic_reg
    gfx7_data_mimg_store
    gfx7_dst_buf_128
    gfx7_dst_buf_64
    gfx7_dst_buf_96
    gfx7_dst_buf_lds
    gfx7_dst_flat_atomic32
    gfx7_dst_flat_atomic64
    gfx7_dst_mimg_gather4
    gfx7_dst_mimg_regular
    gfx7_offset_buf
    gfx7_offset_smem
    gfx7_rsrc_buf
    gfx7_rsrc_mimg
    gfx7_samp_mimg
    gfx7_sdst128_0
    gfx7_sdst256_0
    gfx7_sdst32_0
    gfx7_sdst32_1
    gfx7_sdst32_2
    gfx7_sdst512_0
    gfx7_sdst64_0
    gfx7_sdst64_1
    gfx7_src32_0
    gfx7_src32_1
    gfx7_src32_2
    gfx7_src32_3
    gfx7_src32_4
    gfx7_src32_5
    gfx7_src32_6
    gfx7_src64_0
    gfx7_src64_1
    gfx7_src64_2
    gfx7_src_exp
    gfx7_ssrc32_0
    gfx7_ssrc32_1
    gfx7_ssrc32_2
    gfx7_ssrc32_3
    gfx7_ssrc32_4
    gfx7_ssrc32_5
    gfx7_ssrc32_6
    gfx7_ssrc64_0
    gfx7_ssrc64_1
    gfx7_ssrc64_2
    gfx7_ssrc64_3
    gfx7_vcc_64
    gfx7_vdata128_0
    gfx7_vdata32_0
    gfx7_vdata64_0
    gfx7_vdata96_0
    gfx7_vdst128_0
    gfx7_vdst32_0
    gfx7_vdst64_0
    gfx7_vdst96_0
    gfx7_vsrc128_0
    gfx7_vsrc32_0
    gfx7_vsrc32_1
    gfx7_vsrc64_0
    gfx7_mod
    gfx7_opt
    gfx7_ret
    gfx7_type_dev
