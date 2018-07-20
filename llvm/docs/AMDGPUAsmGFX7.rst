..
    **************************************************
    *                                                *
    *   Automatically generated file, do not edit!   *
    *                                                *
    **************************************************

===========================
Syntax of GFX7 Instructions
===========================

.. contents::
  :local:


DS
===========================

.. parsed-literal::

    ds_add_rtn_u32                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_add_rtn_u64                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_add_src2_u32                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_add_src2_u64                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_add_u32                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_add_u64                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_and_b32                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_and_b64                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_and_rtn_b32                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_and_rtn_b64                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_and_src2_b32                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_and_src2_b64                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_append                      dst                            :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_b32                   src0, src1, src2               :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_b64                   src0, src1, src2               :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_f32                   src0, src1, src2               :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_f64                   src0, src1, src2               :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_rtn_b32               dst, src0, src1, src2          :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_rtn_b64               dst, src0, src1, src2          :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_rtn_f32               dst, src0, src1, src2          :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_cmpst_rtn_f64               dst, src0, src1, src2          :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_condxchg32_rtn_b64          dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_consume                     dst                            :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_dec_rtn_u32                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_dec_rtn_u64                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_dec_src2_u32                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_dec_src2_u64                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_dec_u32                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_dec_u64                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_gws_barrier                 src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_gws_init                    src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_gws_sema_br                 src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_gws_sema_p                  src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_gws_sema_release_all        src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_gws_sema_v                  src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_inc_rtn_u32                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_inc_rtn_u64                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_inc_src2_u32                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_inc_src2_u64                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_inc_u32                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_inc_u64                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_f32                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_f64                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_i32                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_i64                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_rtn_f32                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_rtn_f64                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_rtn_i32                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_rtn_i64                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_rtn_u32                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_rtn_u64                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_src2_f32                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_src2_f64                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_src2_i32                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_src2_i64                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_src2_u32                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_src2_u64                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_u32                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_max_u64                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_f32                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_f64                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_i32                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_i64                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_rtn_f32                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_rtn_f64                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_rtn_i32                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_rtn_i64                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_rtn_u32                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_rtn_u64                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_src2_f32                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_src2_f64                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_src2_i32                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_src2_i64                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_src2_u32                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_src2_u64                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_u32                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_min_u64                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_mskor_b32                   src0, src1, src2               :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_mskor_b64                   src0, src1, src2               :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_mskor_rtn_b32               dst, src0, src1, src2          :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_mskor_rtn_b64               dst, src0, src1, src2          :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_nop                         src0
    ds_or_b32                      src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_or_b64                      src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_or_rtn_b32                  dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_or_rtn_b64                  dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_or_src2_b32                 src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_or_src2_b64                 src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_ordered_count               dst, src0                      :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read2_b32                   dst, src0                      :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_read2_b64                   dst, src0                      :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_read2st64_b32               dst, src0                      :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_read2st64_b64               dst, src0                      :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_b128                   dst, src0                      :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_b32                    dst, src0                      :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_b64                    dst, src0                      :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_b96                    dst, src0                      :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_i16                    dst, src0                      :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_i8                     dst, src0                      :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_u16                    dst, src0                      :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_read_u8                     dst, src0                      :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_rsub_rtn_u32                dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_rsub_rtn_u64                dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_rsub_src2_u32               src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_rsub_src2_u64               src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_rsub_u32                    src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_rsub_u64                    src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_sub_rtn_u32                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_sub_rtn_u64                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_sub_src2_u32                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_sub_src2_u64                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_sub_u32                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_sub_u64                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_swizzle_b32                 dst, src0                      :ref:`sw_offset16<amdgpu_synid_sw_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrap_rtn_b32                dst, src0, src1, src2          :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write2_b32                  src0, src1, src2               :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_write2_b64                  src0, src1, src2               :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_write2st64_b32              src0, src1, src2               :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_write2st64_b64              src0, src1, src2               :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_b128                  src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_b16                   src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_b32                   src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_b64                   src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_b8                    src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_b96                   src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_src2_b32              src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_write_src2_b64              src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrxchg2_rtn_b32             dst, src0, src1, src2          :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrxchg2_rtn_b64             dst, src0, src1, src2          :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrxchg2st64_rtn_b32         dst, src0, src1, src2          :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrxchg2st64_rtn_b64         dst, src0, src1, src2          :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`ds_offset8<amdgpu_synid_ds_offset8>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrxchg_rtn_b32              dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_wrxchg_rtn_b64              dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_xor_b32                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_xor_b64                     src0, src1                     :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_xor_rtn_b32                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_xor_rtn_b64                 dst, src0, src1                :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_xor_src2_b32                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`
    ds_xor_src2_b64                src0                           :ref:`ds_offset16<amdgpu_synid_ds_offset16>` :ref:`gds<amdgpu_synid_gds>`

EXP
===========================

.. parsed-literal::

    exp                            dst, src0, src1, src2, src3    :ref:`done<amdgpu_synid_done>` :ref:`compr<amdgpu_synid_compr>` :ref:`vm<amdgpu_synid_vm>`

FLAT
===========================

.. parsed-literal::

    flat_atomic_add                dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_add_x2             dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_and                dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_and_x2             dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_cmpswap            dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_cmpswap_x2         dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_dec                dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_dec_x2             dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_fcmpswap           dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_fcmpswap_x2        dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_fmax               dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_fmax_x2            dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_fmin               dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_fmin_x2            dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_inc                dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_inc_x2             dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_or                 dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_or_x2              dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_smax               dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_smax_x2            dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_smin               dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_smin_x2            dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_sub                dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_sub_x2             dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_swap               dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_swap_x2            dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_umax               dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_umax_x2            dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_umin               dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_umin_x2            dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_xor                dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_atomic_xor_x2             dst, src0, src1                :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_dword                dst, src0                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_dwordx2              dst, src0                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_dwordx3              dst, src0                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_dwordx4              dst, src0                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_sbyte                dst, src0                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_sshort               dst, src0                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_ubyte                dst, src0                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_load_ushort               dst, src0                      :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_store_byte                src0, src1                     :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_store_dword               src0, src1                     :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_store_dwordx2             src0, src1                     :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_store_dwordx3             src0, src1                     :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_store_dwordx4             src0, src1                     :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    flat_store_short               src0, src1                     :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`

MIMG
===========================

.. parsed-literal::

    image_atomic_add               src0, src1, src2               :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_and               src0, src1, src2               :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_cmpswap           src0, src1, src2               :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_dec               src0, src1, src2               :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_inc               src0, src1, src2               :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_or                src0, src1, src2               :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_smax              src0, src1, src2               :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_smin              src0, src1, src2               :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_sub               src0, src1, src2               :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_swap              src0, src1, src2               :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_umax              src0, src1, src2               :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_umin              src0, src1, src2               :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_atomic_xor               src0, src1, src2               :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4                  dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_b                dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_b_cl             dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_b_cl_o           dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_b_o              dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c                dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_b              dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_b_cl           dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_b_cl_o         dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_b_o            dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_cl             dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_cl_o           dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_l              dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_l_o            dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_lz             dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_lz_o           dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_c_o              dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_cl               dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_cl_o             dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_l                dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_l_o              dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_lz               dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_lz_o             dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_gather4_o                dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_get_lod                  dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_get_resinfo              dst, src0, src1                :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_load                     dst, src0, src1                :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_load_mip                 dst, src0, src1                :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample                   dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_b                 dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_b_cl              dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c                 dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_b               dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_b_cl            dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_cd              dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_cl              dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_d               dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_l               dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_c_lz              dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_cl                dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_l                 dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_sample_lz                dst, src0, src1, src2          :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`tfe<amdgpu_synid_tfe>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_store                    src0, src1, src2               :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`
    image_store_mip                src0, src1, src2               :ref:`dmask<amdgpu_synid_dmask>` :ref:`unorm<amdgpu_synid_unorm>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lwe<amdgpu_synid_lwe>` :ref:`da<amdgpu_synid_da>`

MUBUF
===========================

.. parsed-literal::

    buffer_atomic_add              src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_add_x2           src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_and              src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_and_x2           src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_cmpswap          src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_cmpswap_x2       src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_dec              src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_dec_x2           src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_inc              src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_inc_x2           src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_or               src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_or_x2            src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_smax             src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_smax_x2          src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_smin             src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_smin_x2          src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_sub              src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_sub_x2           src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_swap             src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_swap_x2          src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_umax             src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_umax_x2          src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_umin             src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_umin_x2          src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_xor              src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_atomic_xor_x2           src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_dword              dst, src0, src1, src2          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lds<amdgpu_synid_lds>`
    buffer_load_dwordx2            dst, src0, src1, src2          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_dwordx3            dst, src0, src1, src2          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_dwordx4            dst, src0, src1, src2          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_format_x           dst, src0, src1, src2          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lds<amdgpu_synid_lds>`
    buffer_load_format_xy          dst, src0, src1, src2          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_format_xyz         dst, src0, src1, src2          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_format_xyzw        dst, src0, src1, src2          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_load_sbyte              dst, src0, src1, src2          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lds<amdgpu_synid_lds>`
    buffer_load_sshort             dst, src0, src1, src2          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lds<amdgpu_synid_lds>`
    buffer_load_ubyte              dst, src0, src1, src2          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lds<amdgpu_synid_lds>`
    buffer_load_ushort             dst, src0, src1, src2          :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>` :ref:`lds<amdgpu_synid_lds>`
    buffer_store_byte              src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_dword             src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_dwordx2           src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_dwordx3           src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_dwordx4           src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_format_x          src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_format_xy         src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_format_xyz        src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_format_xyzw       src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_store_short             src0, src1, src2, src3         :ref:`idxen<amdgpu_synid_idxen>` :ref:`offen<amdgpu_synid_offen>` :ref:`addr64<amdgpu_synid_addr64>` :ref:`buf_offset12<amdgpu_synid_buf_offset12>` :ref:`glc<amdgpu_synid_glc>` :ref:`slc<amdgpu_synid_slc>`
    buffer_wbinvl1
    buffer_wbinvl1_vol

SMRD
===========================

.. parsed-literal::

    s_buffer_load_dword            dst, src0, src1
    s_buffer_load_dwordx16         dst, src0, src1
    s_buffer_load_dwordx2          dst, src0, src1
    s_buffer_load_dwordx4          dst, src0, src1
    s_buffer_load_dwordx8          dst, src0, src1
    s_dcache_inv
    s_dcache_inv_vol
    s_load_dword                   dst, src0, src1
    s_load_dwordx16                dst, src0, src1
    s_load_dwordx2                 dst, src0, src1
    s_load_dwordx4                 dst, src0, src1
    s_load_dwordx8                 dst, src0, src1
    s_memtime                      dst

SOP1
===========================

.. parsed-literal::

    s_abs_i32                      dst, src0
    s_and_saveexec_b64             dst, src0
    s_andn2_saveexec_b64           dst, src0
    s_bcnt0_i32_b32                dst, src0
    s_bcnt0_i32_b64                dst, src0
    s_bcnt1_i32_b32                dst, src0
    s_bcnt1_i32_b64                dst, src0
    s_bitset0_b32                  dst, src0
    s_bitset0_b64                  dst, src0
    s_bitset1_b32                  dst, src0
    s_bitset1_b64                  dst, src0
    s_brev_b32                     dst, src0
    s_brev_b64                     dst, src0
    s_cbranch_join                 src0
    s_cmov_b32                     dst, src0
    s_cmov_b64                     dst, src0
    s_ff0_i32_b32                  dst, src0
    s_ff0_i32_b64                  dst, src0
    s_ff1_i32_b32                  dst, src0
    s_ff1_i32_b64                  dst, src0
    s_flbit_i32                    dst, src0
    s_flbit_i32_b32                dst, src0
    s_flbit_i32_b64                dst, src0
    s_flbit_i32_i64                dst, src0
    s_getpc_b64                    dst
    s_mov_b32                      dst, src0
    s_mov_b64                      dst, src0
    s_mov_fed_b32                  dst, src0
    s_movreld_b32                  dst, src0
    s_movreld_b64                  dst, src0
    s_movrels_b32                  dst, src0
    s_movrels_b64                  dst, src0
    s_nand_saveexec_b64            dst, src0
    s_nor_saveexec_b64             dst, src0
    s_not_b32                      dst, src0
    s_not_b64                      dst, src0
    s_or_saveexec_b64              dst, src0
    s_orn2_saveexec_b64            dst, src0
    s_quadmask_b32                 dst, src0
    s_quadmask_b64                 dst, src0
    s_rfe_b64                      src0
    s_setpc_b64                    src0
    s_sext_i32_i16                 dst, src0
    s_sext_i32_i8                  dst, src0
    s_swappc_b64                   dst, src0
    s_wqm_b32                      dst, src0
    s_wqm_b64                      dst, src0
    s_xnor_saveexec_b64            dst, src0
    s_xor_saveexec_b64             dst, src0

SOP2
===========================

.. parsed-literal::

    s_absdiff_i32                  dst, src0, src1
    s_add_i32                      dst, src0, src1
    s_add_u32                      dst, src0, src1
    s_addc_u32                     dst, src0, src1
    s_and_b32                      dst, src0, src1
    s_and_b64                      dst, src0, src1
    s_andn2_b32                    dst, src0, src1
    s_andn2_b64                    dst, src0, src1
    s_ashr_i32                     dst, src0, src1
    s_ashr_i64                     dst, src0, src1
    s_bfe_i32                      dst, src0, src1
    s_bfe_i64                      dst, src0, src1
    s_bfe_u32                      dst, src0, src1
    s_bfe_u64                      dst, src0, src1
    s_bfm_b32                      dst, src0, src1
    s_bfm_b64                      dst, src0, src1
    s_cbranch_g_fork               src0, src1
    s_cselect_b32                  dst, src0, src1
    s_cselect_b64                  dst, src0, src1
    s_lshl_b32                     dst, src0, src1
    s_lshl_b64                     dst, src0, src1
    s_lshr_b32                     dst, src0, src1
    s_lshr_b64                     dst, src0, src1
    s_max_i32                      dst, src0, src1
    s_max_u32                      dst, src0, src1
    s_min_i32                      dst, src0, src1
    s_min_u32                      dst, src0, src1
    s_mul_i32                      dst, src0, src1
    s_nand_b32                     dst, src0, src1
    s_nand_b64                     dst, src0, src1
    s_nor_b32                      dst, src0, src1
    s_nor_b64                      dst, src0, src1
    s_or_b32                       dst, src0, src1
    s_or_b64                       dst, src0, src1
    s_orn2_b32                     dst, src0, src1
    s_orn2_b64                     dst, src0, src1
    s_sub_i32                      dst, src0, src1
    s_sub_u32                      dst, src0, src1
    s_subb_u32                     dst, src0, src1
    s_xnor_b32                     dst, src0, src1
    s_xnor_b64                     dst, src0, src1
    s_xor_b32                      dst, src0, src1
    s_xor_b64                      dst, src0, src1

SOPC
===========================

.. parsed-literal::

    s_bitcmp0_b32                  src0, src1
    s_bitcmp0_b64                  src0, src1
    s_bitcmp1_b32                  src0, src1
    s_bitcmp1_b64                  src0, src1
    s_cmp_eq_i32                   src0, src1
    s_cmp_eq_u32                   src0, src1
    s_cmp_ge_i32                   src0, src1
    s_cmp_ge_u32                   src0, src1
    s_cmp_gt_i32                   src0, src1
    s_cmp_gt_u32                   src0, src1
    s_cmp_le_i32                   src0, src1
    s_cmp_le_u32                   src0, src1
    s_cmp_lg_i32                   src0, src1
    s_cmp_lg_u32                   src0, src1
    s_cmp_lt_i32                   src0, src1
    s_cmp_lt_u32                   src0, src1
    s_setvskip                     src0, src1

SOPK
===========================

.. parsed-literal::

    s_addk_i32                     dst, src0
    s_cbranch_i_fork               src0, src1
    s_cmovk_i32                    dst, src0
    s_cmpk_eq_i32                  src0, src1
    s_cmpk_eq_u32                  src0, src1
    s_cmpk_ge_i32                  src0, src1
    s_cmpk_ge_u32                  src0, src1
    s_cmpk_gt_i32                  src0, src1
    s_cmpk_gt_u32                  src0, src1
    s_cmpk_le_i32                  src0, src1
    s_cmpk_le_u32                  src0, src1
    s_cmpk_lg_i32                  src0, src1
    s_cmpk_lg_u32                  src0, src1
    s_cmpk_lt_i32                  src0, src1
    s_cmpk_lt_u32                  src0, src1
    s_getreg_b32                   dst, src0
    s_movk_i32                     dst, src0
    s_mulk_i32                     dst, src0
    s_setreg_b32                   dst, src0
    s_setreg_imm32_b32             dst, src0

SOPP
===========================

.. parsed-literal::

    s_barrier
    s_branch                       src0
    s_cbranch_cdbgsys              src0
    s_cbranch_cdbgsys_and_user     src0
    s_cbranch_cdbgsys_or_user      src0
    s_cbranch_cdbguser             src0
    s_cbranch_execnz               src0
    s_cbranch_execz                src0
    s_cbranch_scc0                 src0
    s_cbranch_scc1                 src0
    s_cbranch_vccnz                src0
    s_cbranch_vccz                 src0
    s_decperflevel                 src0
    s_endpgm
    s_icache_inv
    s_incperflevel                 src0
    s_nop                          src0
    s_sendmsg                      src0
    s_sendmsghalt                  src0
    s_sethalt                      src0
    s_setkill                      src0
    s_setprio                      src0
    s_sleep                        src0
    s_trap                         src0
    s_ttracedata
    s_waitcnt                      src0

VINTRP
===========================

.. parsed-literal::

    v_interp_mov_f32               dst, src0, src1
    v_interp_p1_f32                dst, src0, src1
    v_interp_p2_f32                dst, src0, src1

VOP1
===========================

.. parsed-literal::

    v_bfrev_b32                    dst, src0
    v_ceil_f32                     dst, src0
    v_ceil_f64                     dst, src0
    v_clrexcp
    v_cos_f32                      dst, src0
    v_cvt_f16_f32                  dst, src0
    v_cvt_f32_f16                  dst, src0
    v_cvt_f32_f64                  dst, src0
    v_cvt_f32_i32                  dst, src0
    v_cvt_f32_u32                  dst, src0
    v_cvt_f32_ubyte0               dst, src0
    v_cvt_f32_ubyte1               dst, src0
    v_cvt_f32_ubyte2               dst, src0
    v_cvt_f32_ubyte3               dst, src0
    v_cvt_f64_f32                  dst, src0
    v_cvt_f64_i32                  dst, src0
    v_cvt_f64_u32                  dst, src0
    v_cvt_flr_i32_f32              dst, src0
    v_cvt_i32_f32                  dst, src0
    v_cvt_i32_f64                  dst, src0
    v_cvt_off_f32_i4               dst, src0
    v_cvt_rpi_i32_f32              dst, src0
    v_cvt_u32_f32                  dst, src0
    v_cvt_u32_f64                  dst, src0
    v_exp_f32                      dst, src0
    v_exp_legacy_f32               dst, src0
    v_ffbh_i32                     dst, src0
    v_ffbh_u32                     dst, src0
    v_ffbl_b32                     dst, src0
    v_floor_f32                    dst, src0
    v_floor_f64                    dst, src0
    v_fract_f32                    dst, src0
    v_fract_f64                    dst, src0
    v_frexp_exp_i32_f32            dst, src0
    v_frexp_exp_i32_f64            dst, src0
    v_frexp_mant_f32               dst, src0
    v_frexp_mant_f64               dst, src0
    v_log_clamp_f32                dst, src0
    v_log_f32                      dst, src0
    v_log_legacy_f32               dst, src0
    v_mov_b32                      dst, src0
    v_mov_fed_b32                  dst, src0
    v_movreld_b32                  dst, src0
    v_movrels_b32                  dst, src0
    v_movrelsd_b32                 dst, src0
    v_nop
    v_not_b32                      dst, src0
    v_rcp_clamp_f32                dst, src0
    v_rcp_clamp_f64                dst, src0
    v_rcp_f32                      dst, src0
    v_rcp_f64                      dst, src0
    v_rcp_iflag_f32                dst, src0
    v_rcp_legacy_f32               dst, src0
    v_readfirstlane_b32            dst, src0
    v_rndne_f32                    dst, src0
    v_rndne_f64                    dst, src0
    v_rsq_clamp_f32                dst, src0
    v_rsq_clamp_f64                dst, src0
    v_rsq_f32                      dst, src0
    v_rsq_f64                      dst, src0
    v_rsq_legacy_f32               dst, src0
    v_sin_f32                      dst, src0
    v_sqrt_f32                     dst, src0
    v_sqrt_f64                     dst, src0
    v_trunc_f32                    dst, src0
    v_trunc_f64                    dst, src0

VOP2
===========================

.. parsed-literal::

    v_add_f32                      dst, src0, src1
    v_add_i32                      dst0, dst1, src0, src1
    v_addc_u32                     dst0, dst1, src0, src1, src2
    v_and_b32                      dst, src0, src1
    v_ashr_i32                     dst, src0, src1
    v_ashrrev_i32                  dst, src0, src1
    v_bcnt_u32_b32                 dst, src0, src1
    v_bfm_b32                      dst, src0, src1
    v_cndmask_b32                  dst, src0, src1, src2
    v_cvt_pk_i16_i32               dst, src0, src1
    v_cvt_pk_u16_u32               dst, src0, src1
    v_cvt_pkaccum_u8_f32           dst, src0, src1
    v_cvt_pknorm_i16_f32           dst, src0, src1
    v_cvt_pknorm_u16_f32           dst, src0, src1
    v_cvt_pkrtz_f16_f32            dst, src0, src1
    v_ldexp_f32                    dst, src0, src1
    v_lshl_b32                     dst, src0, src1
    v_lshlrev_b32                  dst, src0, src1
    v_lshr_b32                     dst, src0, src1
    v_lshrrev_b32                  dst, src0, src1
    v_mac_f32                      dst, src0, src1
    v_mac_legacy_f32               dst, src0, src1
    v_madak_f32                    dst, src0, src1, src2
    v_madmk_f32                    dst, src0, src1, src2
    v_max_f32                      dst, src0, src1
    v_max_i32                      dst, src0, src1
    v_max_legacy_f32               dst, src0, src1
    v_max_u32                      dst, src0, src1
    v_mbcnt_hi_u32_b32             dst, src0, src1
    v_mbcnt_lo_u32_b32             dst, src0, src1
    v_min_f32                      dst, src0, src1
    v_min_i32                      dst, src0, src1
    v_min_legacy_f32               dst, src0, src1
    v_min_u32                      dst, src0, src1
    v_mul_f32                      dst, src0, src1
    v_mul_hi_i32_i24               dst, src0, src1
    v_mul_hi_u32_u24               dst, src0, src1
    v_mul_i32_i24                  dst, src0, src1
    v_mul_legacy_f32               dst, src0, src1
    v_mul_u32_u24                  dst, src0, src1
    v_or_b32                       dst, src0, src1
    v_readlane_b32                 dst, src0, src1
    v_sub_f32                      dst, src0, src1
    v_sub_i32                      dst0, dst1, src0, src1
    v_subb_u32                     dst0, dst1, src0, src1, src2
    v_subbrev_u32                  dst0, dst1, src0, src1, src2
    v_subrev_f32                   dst, src0, src1
    v_subrev_i32                   dst0, dst1, src0, src1
    v_writelane_b32                dst, src0, src1
    v_xor_b32                      dst, src0, src1

VOP3
===========================

.. parsed-literal::

    v_add_f32_e64                  dst, src0, src1                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_add_f64                      dst, src0, src1                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_add_i32_e64                  dst0, dst1, src0, src1         :ref:`omod<amdgpu_synid_omod>`
    v_addc_u32_e64                 dst0, dst1, src0, src1, src2   :ref:`omod<amdgpu_synid_omod>`
    v_alignbit_b32                 dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_alignbyte_b32                dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_and_b32_e64                  dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_ashr_i32_e64                 dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_ashr_i64                     dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_ashrrev_i32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_bcnt_u32_b32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_bfe_i32                      dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_bfe_u32                      dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_bfi_b32                      dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_bfm_b32_e64                  dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_bfrev_b32_e64                dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_ceil_f32_e64                 dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_ceil_f64_e64                 dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_clrexcp_e64                                                 :ref:`omod<amdgpu_synid_omod>`
    v_cmp_class_f32_e64            dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_class_f64_e64            dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_eq_f32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_eq_f64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_eq_i32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_eq_i64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_eq_u32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_eq_u64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_f_f32_e64                dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_f_f64_e64                dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_f_i32_e64                dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_f_i64_e64                dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_f_u32_e64                dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_f_u64_e64                dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_ge_f32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_ge_f64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_ge_i32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_ge_i64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_ge_u32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_ge_u64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_gt_f32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_gt_f64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_gt_i32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_gt_i64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_gt_u32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_gt_u64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_le_f32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_le_f64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_le_i32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_le_i64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_le_u32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_le_u64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_lg_f32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_lg_f64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_lt_f32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_lt_f64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_lt_i32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_lt_i64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_lt_u32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_lt_u64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_ne_i32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_ne_i64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_ne_u32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_ne_u64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_neq_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_neq_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_nge_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_nge_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_ngt_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_ngt_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_nle_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_nle_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_nlg_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_nlg_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_nlt_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_nlt_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_o_f32_e64                dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_o_f64_e64                dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_t_i32_e64                dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_t_i64_e64                dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_t_u32_e64                dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_t_u64_e64                dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_tru_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_tru_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_u_f32_e64                dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmp_u_f64_e64                dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_eq_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_eq_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_f_f32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_f_f64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_ge_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_ge_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_gt_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_gt_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_le_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_le_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_lg_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_lg_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_lt_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_lt_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_neq_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_neq_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_nge_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_nge_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_ngt_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_ngt_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_nle_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_nle_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_nlg_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_nlg_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_nlt_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_nlt_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_o_f32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_o_f64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_tru_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_tru_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_u_f32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmps_u_f64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_eq_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_eq_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_f_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_f_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_ge_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_ge_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_gt_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_gt_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_le_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_le_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_lg_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_lg_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_lt_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_lt_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_neq_f32_e64            dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_neq_f64_e64            dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_nge_f32_e64            dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_nge_f64_e64            dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_ngt_f32_e64            dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_ngt_f64_e64            dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_nle_f32_e64            dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_nle_f64_e64            dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_nlg_f32_e64            dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_nlg_f64_e64            dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_nlt_f32_e64            dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_nlt_f64_e64            dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_o_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_o_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_tru_f32_e64            dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_tru_f64_e64            dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_u_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpsx_u_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_class_f32_e64           dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_class_f64_e64           dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_eq_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_eq_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_eq_i32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_eq_i64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_eq_u32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_eq_u64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_f_f32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_f_f64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_f_i32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_f_i64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_f_u32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_f_u64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_ge_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_ge_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_ge_i32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_ge_i64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_ge_u32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_ge_u64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_gt_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_gt_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_gt_i32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_gt_i64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_gt_u32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_gt_u64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_le_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_le_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_le_i32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_le_i64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_le_u32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_le_u64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_lg_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_lg_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_lt_f32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_lt_f64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_lt_i32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_lt_i64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_lt_u32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_lt_u64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_ne_i32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_ne_i64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_ne_u32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_ne_u64_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_neq_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_neq_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_nge_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_nge_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_ngt_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_ngt_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_nle_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_nle_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_nlg_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_nlg_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_nlt_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_nlt_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_o_f32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_o_f64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_t_i32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_t_i64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_t_u32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_t_u64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_tru_f32_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_tru_f64_e64             dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_u_f32_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cmpx_u_f64_e64               dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cndmask_b32_e64              dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_cos_f32_e64                  dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cubeid_f32                   dst, src0, src1, src2          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cubema_f32                   dst, src0, src1, src2          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cubesc_f32                   dst, src0, src1, src2          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cubetc_f32                   dst, src0, src1, src2          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f16_f32_e64              dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f32_f16_e64              dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f32_f64_e64              dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f32_i32_e64              dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f32_u32_e64              dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f32_ubyte0_e64           dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f32_ubyte1_e64           dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f32_ubyte2_e64           dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f32_ubyte3_e64           dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f64_f32_e64              dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f64_i32_e64              dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_f64_u32_e64              dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_flr_i32_f32_e64          dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_cvt_i32_f32_e64              dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_cvt_i32_f64_e64              dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_cvt_off_f32_i4_e64           dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_cvt_pk_i16_i32_e64           dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cvt_pk_u16_u32_e64           dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cvt_pk_u8_f32                dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_cvt_pkaccum_u8_f32_e64       dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cvt_pknorm_i16_f32_e64       dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cvt_pknorm_u16_f32_e64       dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cvt_pkrtz_f16_f32_e64        dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_cvt_rpi_i32_f32_e64          dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_cvt_u32_f32_e64              dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_cvt_u32_f64_e64              dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_div_fixup_f32                dst, src0, src1, src2          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_div_fixup_f64                dst, src0, src1, src2          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_div_fmas_f32                 dst, src0, src1, src2          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_div_fmas_f64                 dst, src0, src1, src2          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_div_scale_f32                dst0, dst1, src0, src1, src2   :ref:`omod<amdgpu_synid_omod>`
    v_div_scale_f64                dst0, dst1, src0, src1, src2   :ref:`omod<amdgpu_synid_omod>`
    v_exp_f32_e64                  dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_exp_legacy_f32_e64           dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_ffbh_i32_e64                 dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_ffbh_u32_e64                 dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_ffbl_b32_e64                 dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_floor_f32_e64                dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_floor_f64_e64                dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_fma_f32                      dst, src0, src1, src2          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_fma_f64                      dst, src0, src1, src2          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_fract_f32_e64                dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_fract_f64_e64                dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_frexp_exp_i32_f32_e64        dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_frexp_exp_i32_f64_e64        dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_frexp_mant_f32_e64           dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_frexp_mant_f64_e64           dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_ldexp_f32_e64                dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_ldexp_f64                    dst, src0, src1                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_lerp_u8                      dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_log_clamp_f32_e64            dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_log_f32_e64                  dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_log_legacy_f32_e64           dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_lshl_b32_e64                 dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_lshl_b64                     dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_lshlrev_b32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_lshr_b32_e64                 dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_lshr_b64                     dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_lshrrev_b32_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_mac_f32_e64                  dst, src0, src1                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mac_legacy_f32_e64           dst, src0, src1                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mad_f32                      dst, src0, src1, src2          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mad_i32_i24                  dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_mad_i64_i32                  dst0, dst1, src0, src1, src2   :ref:`omod<amdgpu_synid_omod>`
    v_mad_legacy_f32               dst, src0, src1, src2          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mad_u32_u24                  dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_mad_u64_u32                  dst0, dst1, src0, src1, src2   :ref:`omod<amdgpu_synid_omod>`
    v_max3_f32                     dst, src0, src1, src2          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_max3_i32                     dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_max3_u32                     dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_max_f32_e64                  dst, src0, src1                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_max_f64                      dst, src0, src1                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_max_i32_e64                  dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_max_legacy_f32_e64           dst, src0, src1                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_max_u32_e64                  dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_mbcnt_hi_u32_b32_e64         dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_mbcnt_lo_u32_b32_e64         dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_med3_f32                     dst, src0, src1, src2          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_med3_i32                     dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_med3_u32                     dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_min3_f32                     dst, src0, src1, src2          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_min3_i32                     dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_min3_u32                     dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_min_f32_e64                  dst, src0, src1                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_min_f64                      dst, src0, src1                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_min_i32_e64                  dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_min_legacy_f32_e64           dst, src0, src1                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_min_u32_e64                  dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_mov_b32_e64                  dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_mov_fed_b32_e64              dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_movreld_b32_e64              dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_movrels_b32_e64              dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_movrelsd_b32_e64             dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_mqsad_pk_u16_u8              dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_mqsad_u32_u8                 dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_msad_u8                      dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_mul_f32_e64                  dst, src0, src1                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mul_f64                      dst, src0, src1                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mul_hi_i32                   dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_mul_hi_i32_i24_e64           dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_mul_hi_u32                   dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_mul_hi_u32_u24_e64           dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_mul_i32_i24_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_mul_legacy_f32_e64           dst, src0, src1                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_mul_lo_i32                   dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_mul_lo_u32                   dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_mul_u32_u24_e64              dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_mullit_f32                   dst, src0, src1, src2          :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_nop_e64                                                     :ref:`omod<amdgpu_synid_omod>`
    v_not_b32_e64                  dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_or_b32_e64                   dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`
    v_qsad_pk_u16_u8               dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_rcp_clamp_f32_e64            dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_rcp_clamp_f64_e64            dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rcp_f32_e64                  dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_rcp_f64_e64                  dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rcp_iflag_f32_e64            dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rcp_legacy_f32_e64           dst, src0                      :ref:`omod<amdgpu_synid_omod>`
    v_rndne_f32_e64                dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rndne_f64_e64                dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rsq_clamp_f32_e64            dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rsq_clamp_f64_e64            dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rsq_f32_e64                  dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rsq_f64_e64                  dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_rsq_legacy_f32_e64           dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_sad_hi_u8                    dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_sad_u16                      dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_sad_u32                      dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_sad_u8                       dst, src0, src1, src2          :ref:`omod<amdgpu_synid_omod>`
    v_sin_f32_e64                  dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_sqrt_f32_e64                 dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_sqrt_f64_e64                 dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_sub_f32_e64                  dst, src0, src1                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_sub_i32_e64                  dst0, dst1, src0, src1         :ref:`omod<amdgpu_synid_omod>`
    v_subb_u32_e64                 dst0, dst1, src0, src1, src2   :ref:`omod<amdgpu_synid_omod>`
    v_subbrev_u32_e64              dst0, dst1, src0, src1, src2   :ref:`omod<amdgpu_synid_omod>`
    v_subrev_f32_e64               dst, src0, src1                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_subrev_i32_e64               dst0, dst1, src0, src1         :ref:`omod<amdgpu_synid_omod>`
    v_trig_preop_f64               dst, src0, src1                :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_trunc_f32_e64                dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_trunc_f64_e64                dst, src0                      :ref:`clamp<amdgpu_synid_clamp>` :ref:`omod<amdgpu_synid_omod>`
    v_xor_b32_e64                  dst, src0, src1                :ref:`omod<amdgpu_synid_omod>`

VOPC
===========================

.. parsed-literal::

    v_cmp_class_f32                dst, src0, src1
    v_cmp_class_f64                dst, src0, src1
    v_cmp_eq_f32                   dst, src0, src1
    v_cmp_eq_f64                   dst, src0, src1
    v_cmp_eq_i32                   dst, src0, src1
    v_cmp_eq_i64                   dst, src0, src1
    v_cmp_eq_u32                   dst, src0, src1
    v_cmp_eq_u64                   dst, src0, src1
    v_cmp_f_f32                    dst, src0, src1
    v_cmp_f_f64                    dst, src0, src1
    v_cmp_f_i32                    dst, src0, src1
    v_cmp_f_i64                    dst, src0, src1
    v_cmp_f_u32                    dst, src0, src1
    v_cmp_f_u64                    dst, src0, src1
    v_cmp_ge_f32                   dst, src0, src1
    v_cmp_ge_f64                   dst, src0, src1
    v_cmp_ge_i32                   dst, src0, src1
    v_cmp_ge_i64                   dst, src0, src1
    v_cmp_ge_u32                   dst, src0, src1
    v_cmp_ge_u64                   dst, src0, src1
    v_cmp_gt_f32                   dst, src0, src1
    v_cmp_gt_f64                   dst, src0, src1
    v_cmp_gt_i32                   dst, src0, src1
    v_cmp_gt_i64                   dst, src0, src1
    v_cmp_gt_u32                   dst, src0, src1
    v_cmp_gt_u64                   dst, src0, src1
    v_cmp_le_f32                   dst, src0, src1
    v_cmp_le_f64                   dst, src0, src1
    v_cmp_le_i32                   dst, src0, src1
    v_cmp_le_i64                   dst, src0, src1
    v_cmp_le_u32                   dst, src0, src1
    v_cmp_le_u64                   dst, src0, src1
    v_cmp_lg_f32                   dst, src0, src1
    v_cmp_lg_f64                   dst, src0, src1
    v_cmp_lt_f32                   dst, src0, src1
    v_cmp_lt_f64                   dst, src0, src1
    v_cmp_lt_i32                   dst, src0, src1
    v_cmp_lt_i64                   dst, src0, src1
    v_cmp_lt_u32                   dst, src0, src1
    v_cmp_lt_u64                   dst, src0, src1
    v_cmp_ne_i32                   dst, src0, src1
    v_cmp_ne_i64                   dst, src0, src1
    v_cmp_ne_u32                   dst, src0, src1
    v_cmp_ne_u64                   dst, src0, src1
    v_cmp_neq_f32                  dst, src0, src1
    v_cmp_neq_f64                  dst, src0, src1
    v_cmp_nge_f32                  dst, src0, src1
    v_cmp_nge_f64                  dst, src0, src1
    v_cmp_ngt_f32                  dst, src0, src1
    v_cmp_ngt_f64                  dst, src0, src1
    v_cmp_nle_f32                  dst, src0, src1
    v_cmp_nle_f64                  dst, src0, src1
    v_cmp_nlg_f32                  dst, src0, src1
    v_cmp_nlg_f64                  dst, src0, src1
    v_cmp_nlt_f32                  dst, src0, src1
    v_cmp_nlt_f64                  dst, src0, src1
    v_cmp_o_f32                    dst, src0, src1
    v_cmp_o_f64                    dst, src0, src1
    v_cmp_t_i32                    dst, src0, src1
    v_cmp_t_i64                    dst, src0, src1
    v_cmp_t_u32                    dst, src0, src1
    v_cmp_t_u64                    dst, src0, src1
    v_cmp_tru_f32                  dst, src0, src1
    v_cmp_tru_f64                  dst, src0, src1
    v_cmp_u_f32                    dst, src0, src1
    v_cmp_u_f64                    dst, src0, src1
    v_cmps_eq_f32                  dst, src0, src1
    v_cmps_eq_f64                  dst, src0, src1
    v_cmps_f_f32                   dst, src0, src1
    v_cmps_f_f64                   dst, src0, src1
    v_cmps_ge_f32                  dst, src0, src1
    v_cmps_ge_f64                  dst, src0, src1
    v_cmps_gt_f32                  dst, src0, src1
    v_cmps_gt_f64                  dst, src0, src1
    v_cmps_le_f32                  dst, src0, src1
    v_cmps_le_f64                  dst, src0, src1
    v_cmps_lg_f32                  dst, src0, src1
    v_cmps_lg_f64                  dst, src0, src1
    v_cmps_lt_f32                  dst, src0, src1
    v_cmps_lt_f64                  dst, src0, src1
    v_cmps_neq_f32                 dst, src0, src1
    v_cmps_neq_f64                 dst, src0, src1
    v_cmps_nge_f32                 dst, src0, src1
    v_cmps_nge_f64                 dst, src0, src1
    v_cmps_ngt_f32                 dst, src0, src1
    v_cmps_ngt_f64                 dst, src0, src1
    v_cmps_nle_f32                 dst, src0, src1
    v_cmps_nle_f64                 dst, src0, src1
    v_cmps_nlg_f32                 dst, src0, src1
    v_cmps_nlg_f64                 dst, src0, src1
    v_cmps_nlt_f32                 dst, src0, src1
    v_cmps_nlt_f64                 dst, src0, src1
    v_cmps_o_f32                   dst, src0, src1
    v_cmps_o_f64                   dst, src0, src1
    v_cmps_tru_f32                 dst, src0, src1
    v_cmps_tru_f64                 dst, src0, src1
    v_cmps_u_f32                   dst, src0, src1
    v_cmps_u_f64                   dst, src0, src1
    v_cmpsx_eq_f32                 dst, src0, src1
    v_cmpsx_eq_f64                 dst, src0, src1
    v_cmpsx_f_f32                  dst, src0, src1
    v_cmpsx_f_f64                  dst, src0, src1
    v_cmpsx_ge_f32                 dst, src0, src1
    v_cmpsx_ge_f64                 dst, src0, src1
    v_cmpsx_gt_f32                 dst, src0, src1
    v_cmpsx_gt_f64                 dst, src0, src1
    v_cmpsx_le_f32                 dst, src0, src1
    v_cmpsx_le_f64                 dst, src0, src1
    v_cmpsx_lg_f32                 dst, src0, src1
    v_cmpsx_lg_f64                 dst, src0, src1
    v_cmpsx_lt_f32                 dst, src0, src1
    v_cmpsx_lt_f64                 dst, src0, src1
    v_cmpsx_neq_f32                dst, src0, src1
    v_cmpsx_neq_f64                dst, src0, src1
    v_cmpsx_nge_f32                dst, src0, src1
    v_cmpsx_nge_f64                dst, src0, src1
    v_cmpsx_ngt_f32                dst, src0, src1
    v_cmpsx_ngt_f64                dst, src0, src1
    v_cmpsx_nle_f32                dst, src0, src1
    v_cmpsx_nle_f64                dst, src0, src1
    v_cmpsx_nlg_f32                dst, src0, src1
    v_cmpsx_nlg_f64                dst, src0, src1
    v_cmpsx_nlt_f32                dst, src0, src1
    v_cmpsx_nlt_f64                dst, src0, src1
    v_cmpsx_o_f32                  dst, src0, src1
    v_cmpsx_o_f64                  dst, src0, src1
    v_cmpsx_tru_f32                dst, src0, src1
    v_cmpsx_tru_f64                dst, src0, src1
    v_cmpsx_u_f32                  dst, src0, src1
    v_cmpsx_u_f64                  dst, src0, src1
    v_cmpx_class_f32               dst, src0, src1
    v_cmpx_class_f64               dst, src0, src1
    v_cmpx_eq_f32                  dst, src0, src1
    v_cmpx_eq_f64                  dst, src0, src1
    v_cmpx_eq_i32                  dst, src0, src1
    v_cmpx_eq_i64                  dst, src0, src1
    v_cmpx_eq_u32                  dst, src0, src1
    v_cmpx_eq_u64                  dst, src0, src1
    v_cmpx_f_f32                   dst, src0, src1
    v_cmpx_f_f64                   dst, src0, src1
    v_cmpx_f_i32                   dst, src0, src1
    v_cmpx_f_i64                   dst, src0, src1
    v_cmpx_f_u32                   dst, src0, src1
    v_cmpx_f_u64                   dst, src0, src1
    v_cmpx_ge_f32                  dst, src0, src1
    v_cmpx_ge_f64                  dst, src0, src1
    v_cmpx_ge_i32                  dst, src0, src1
    v_cmpx_ge_i64                  dst, src0, src1
    v_cmpx_ge_u32                  dst, src0, src1
    v_cmpx_ge_u64                  dst, src0, src1
    v_cmpx_gt_f32                  dst, src0, src1
    v_cmpx_gt_f64                  dst, src0, src1
    v_cmpx_gt_i32                  dst, src0, src1
    v_cmpx_gt_i64                  dst, src0, src1
    v_cmpx_gt_u32                  dst, src0, src1
    v_cmpx_gt_u64                  dst, src0, src1
    v_cmpx_le_f32                  dst, src0, src1
    v_cmpx_le_f64                  dst, src0, src1
    v_cmpx_le_i32                  dst, src0, src1
    v_cmpx_le_i64                  dst, src0, src1
    v_cmpx_le_u32                  dst, src0, src1
    v_cmpx_le_u64                  dst, src0, src1
    v_cmpx_lg_f32                  dst, src0, src1
    v_cmpx_lg_f64                  dst, src0, src1
    v_cmpx_lt_f32                  dst, src0, src1
    v_cmpx_lt_f64                  dst, src0, src1
    v_cmpx_lt_i32                  dst, src0, src1
    v_cmpx_lt_i64                  dst, src0, src1
    v_cmpx_lt_u32                  dst, src0, src1
    v_cmpx_lt_u64                  dst, src0, src1
    v_cmpx_ne_i32                  dst, src0, src1
    v_cmpx_ne_i64                  dst, src0, src1
    v_cmpx_ne_u32                  dst, src0, src1
    v_cmpx_ne_u64                  dst, src0, src1
    v_cmpx_neq_f32                 dst, src0, src1
    v_cmpx_neq_f64                 dst, src0, src1
    v_cmpx_nge_f32                 dst, src0, src1
    v_cmpx_nge_f64                 dst, src0, src1
    v_cmpx_ngt_f32                 dst, src0, src1
    v_cmpx_ngt_f64                 dst, src0, src1
    v_cmpx_nle_f32                 dst, src0, src1
    v_cmpx_nle_f64                 dst, src0, src1
    v_cmpx_nlg_f32                 dst, src0, src1
    v_cmpx_nlg_f64                 dst, src0, src1
    v_cmpx_nlt_f32                 dst, src0, src1
    v_cmpx_nlt_f64                 dst, src0, src1
    v_cmpx_o_f32                   dst, src0, src1
    v_cmpx_o_f64                   dst, src0, src1
    v_cmpx_t_i32                   dst, src0, src1
    v_cmpx_t_i64                   dst, src0, src1
    v_cmpx_t_u32                   dst, src0, src1
    v_cmpx_t_u64                   dst, src0, src1
    v_cmpx_tru_f32                 dst, src0, src1
    v_cmpx_tru_f64                 dst, src0, src1
    v_cmpx_u_f32                   dst, src0, src1
    v_cmpx_u_f64                   dst, src0, src1
