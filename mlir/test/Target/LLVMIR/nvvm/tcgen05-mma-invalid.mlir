// RUN: mlir-translate --mlir-to-llvmir -verify-diagnostics -split-input-file %s

// CHECK-LABEL: @nvvm_tcgen05_mma_disable_output_lane_cta_1
llvm.func @nvvm_tcgen05_mma_disable_output_lane_cta_1(%d_tmem : !llvm.ptr<6>, %a_tmem: !llvm.ptr<6>, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %disableOutputLanev4: vector<4 x i32>, %disableOutputLanev8: vector<8 x i32>) {
  // expected-error @below {{Disable Output Lane of length 8 is incompatible with CtaGroupAttr}}
  nvvm.tcgen05.mma %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d mask = %disableOutputLanev8
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, vector<8 x i32>)
  llvm.return
}

// -----

// CHECK-LABEL: @nvvm_tcgen05_mma_disable_output_lane_cta_2
llvm.func @nvvm_tcgen05_mma_disable_output_lane_cta_1(%d_tmem : !llvm.ptr<6>, %a_tmem: !llvm.ptr<6>, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %disableOutputLanev4: vector<4 x i32>, %disableOutputLanev8: vector<8 x i32>) {
  // expected-error @below {{Disable Output Lane of length 8 is incompatible with CtaGroupAttr}}
  nvvm.tcgen05.mma %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d mask = %disableOutputLanev8
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, vector<8 x i32>)
  llvm.return
}

// -----

// CHECK-LABEL: @nvvm_tcgen05_mma_shared_ashift
llvm.func @nvvm_tcgen05_mma_shared_ashift(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1) {
  // expected-error @below {{A-shift can be applied only when matrix A is in tensor memory}}
  nvvm.tcgen05.mma %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, aShift} : (!llvm.ptr<6>, i64, i64, i32, i1)
  llvm.return
}

// -----

// CHECK-LABEL: @nvvm_tcgen05_mma_ashift
llvm.func @nvvm_tcgen05_mma_ashift(%d_tmem : !llvm.ptr<6>, %a_tmem: !llvm.ptr<6>, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1) {
  // expected-error @below {{Cannot use collector buffer operation fill or use with ashift}}
  nvvm.tcgen05.mma %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>, aShift} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1)
  llvm.return
}

// -----

// CHECK-LABEL: @nvvm_tcgen05_mma_mxf4nvf4_block_scale_default
llvm.func @nvvm_tcgen05_mma_mxf4nvf4_block_scale_default(%d_tmem : !llvm.ptr<6>, %a_tmem: !llvm.ptr<6>, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %scalea: !llvm.ptr<6>, %scaleb: !llvm.ptr<6>) {
  // expected-error @below {{mxf4nvf4 requires block scale attribute}}
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %scalea, %scaleb
  {kind = #nvvm.tcgen05_mma_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>, aShift} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)
  llvm.return
}

// -----

// CHECK-LABEL: @nvvm_tcgen05_mma_mxf4_block_scale_default
llvm.func @nvvm_tcgen05_mma_mxf4_block_scale_default(%d_tmem : !llvm.ptr<6>, %a_tmem: !llvm.ptr<6>, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %scalea: !llvm.ptr<6>, %scaleb: !llvm.ptr<6>) {
  // expected-error @below {{mxf4 kind does not support block16 attribute}}
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %scalea, %scaleb
  {kind = #nvvm.tcgen05_mma_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>, ashift, blockScale = #nvvm.tcgen05_mma_block_scale<block16>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)
  llvm.return
}

// -----

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_disable_output_lane_cta_1
llvm.func @nvvm_tcgen05_mma_sp_disable_output_lane_cta_1(%d_tmem : !llvm.ptr<6>, %a_tmem: !llvm.ptr<6>, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %disableOutputLanev4: vector<4 x i32>, %disableOutputLanev8: vector<8 x i32>, %spmetadata: !llvm.ptr<6>) {
  // expected-error @below {{Disable Output Lane of length 8 is incompatible with CtaGroupAttr}}
  nvvm.tcgen05.mma.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLanev8
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)
  llvm.return
}

// -----

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_disable_output_lane_cta_2
llvm.func @nvvm_tcgen05_mma_sp_disable_output_lane_cta_1(%d_tmem : !llvm.ptr<6>, %a_tmem: !llvm.ptr<6>, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %disableOutputLanev4: vector<4 x i32>, %disableOutputLanev8: vector<8 x i32>, %spmetadata: !llvm.ptr<6>) {
  // expected-error @below {{Disable Output Lane of length 8 is incompatible with CtaGroupAttr}}
  nvvm.tcgen05.mma.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLanev8
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)
  llvm.return
}

// -----

// CHECK-LABEL: @nvvm_tcgen05_sp_mma_shared_ashift
llvm.func @nvvm_tcgen05_sp_mma_shared_ashift(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %spmetadata: !llvm.ptr<6>) {
  // expected-error @below {{A-shift can be applied only when matrix A is in tensor memory}}
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, aShift} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)
  llvm.return
}

// -----

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_ashift
llvm.func @nvvm_tcgen05_mma_sp_ashift(%d_tmem : !llvm.ptr<6>, %a_tmem: !llvm.ptr<6>, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %spmetadata: !llvm.ptr<6>) {
  // expected-error @below {{Cannot use collector buffer operation fill or use with ashift}}
  nvvm.tcgen05.mma.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>, aShift} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)
  llvm.return
}

// -----

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_mxf4nvf4_block_scale_default
llvm.func @nvvm_tcgen05_mma_sp_mxf4nvf4_block_scale_default(%d_tmem : !llvm.ptr<6>, %a_tmem: !llvm.ptr<6>, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %scalea: !llvm.ptr<6>, %scaleb: !llvm.ptr<6>, %spmetadata: !llvm.ptr<6>) {
  // expected-error @below {{mxf4nvf4 requires block scale attribute}}
  nvvm.tcgen05.mma.sp.block_scale %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata, %scalea, %scaleb
  {kind = #nvvm.tcgen05_mma_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>, aShift} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>, !llvm.ptr<6>)
  llvm.return
}

// -----

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_mxf4_block_scale_default
llvm.func @nvvm_tcgen05_mma_sp_mxf4_block_scale_default(%d_tmem : !llvm.ptr<6>, %a_tmem: !llvm.ptr<6>, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %scalea: !llvm.ptr<6>, %scaleb: !llvm.ptr<6>, %spmetadata: !llvm.ptr<6>) {
  // expected-error @below {{mxf4 kind does not support block16 attribute}}
  nvvm.tcgen05.mma.sp.block_scale %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata, %scalea, %scaleb
  {kind = #nvvm.tcgen05_mma_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>, ashift, blockScale = #nvvm.tcgen05_mma_block_scale<block16>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>, !llvm.ptr<6>)
  llvm.return
}

// -----

// Invalid Tcgen05MMAKind for tcgen05.mma: mxf8f6f4 is only for block_scale ops.
// CHECK-LABEL: @nvvm_tcgen05_mma_invalid_kind_mxf8f6f4
llvm.func @nvvm_tcgen05_mma_invalid_kind_mxf8f6f4(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1) {
  // expected-error @below {{attribute 'kind' failed to satisfy constraint: tcgen05 MMA Supported Types whose value is one of {f16, tf32, f8f6f4, i8}}}
  nvvm.tcgen05.mma %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d {kind = #nvvm.tcgen05_mma_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1)
  llvm.return
}

// -----

// Invalid Tcgen05MMAKind for tcgen05.mma.sp: mxf4 is only for block_scale ops.
// CHECK-LABEL: @nvvm_tcgen05_mma_sp_invalid_kind_mxf4
llvm.func @nvvm_tcgen05_mma_sp_invalid_kind_mxf4(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %spmetadata: !llvm.ptr<6>) {
  // expected-error @below {{attribute 'kind' failed to satisfy constraint: tcgen05 MMA Supported Types whose value is one of {f16, tf32, f8f6f4, i8}}}
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata {kind = #nvvm.tcgen05_mma_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)
  llvm.return
}

// -----

// Invalid Tcgen05MMAKind for tcgen05.mma.ws: mxf4nvf4 is only for block_scale ops.
// CHECK-LABEL: @nvvm_tcgen05_mma_ws_invalid_kind_mxf4nvf4
llvm.func @nvvm_tcgen05_mma_ws_invalid_kind_mxf4nvf4(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1) {
  // expected-error @below {{attribute 'kind' failed to satisfy constraint: tcgen05 MMA Supported Types whose value is one of {f16, tf32, f8f6f4, i8}}}
  nvvm.tcgen05.mma.ws %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d {kind = #nvvm.tcgen05_mma_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1)
  llvm.return
}

// -----

// Invalid Tcgen05MMAKind for tcgen05.mma.ws.sp: mxf8f6f4 is only for block_scale ops.
// CHECK-LABEL: @nvvm_tcgen05_mma_ws_sp_invalid_kind_mxf8f6f4
llvm.func @nvvm_tcgen05_mma_ws_sp_invalid_kind_mxf8f6f4(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %spmetadata: !llvm.ptr<6>) {
  // expected-error @below {{attribute 'kind' failed to satisfy constraint: tcgen05 MMA Supported Types whose value is one of {f16, tf32, f8f6f4, i8}}}
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata {kind = #nvvm.tcgen05_mma_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)
  llvm.return
}

// -----

// Invalid Tcgen05MMAKind for tcgen05.mma.block_scale: f16 is only for mma/mma.sp/ws/ws.sp.
// CHECK-LABEL: @nvvm_tcgen05_mma_block_scale_invalid_kind_f16
llvm.func @nvvm_tcgen05_mma_block_scale_invalid_kind_f16(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %scale_a: !llvm.ptr<6>, %scale_b: !llvm.ptr<6>) {
  // expected-error @below {{attribute 'kind' failed to satisfy constraint: tcgen05 MMA Supported Types whose value is one of {mxf8f6f4, mxf4, mxf4nvf4}}}
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)
  llvm.return
}

// -----

// Invalid Tcgen05MMAKind for tcgen05.mma.sp.block_scale: tf32 is only for mma/mma.sp/ws/ws.sp.
// CHECK-LABEL: @nvvm_tcgen05_mma_sp_block_scale_invalid_kind_tf32
llvm.func @nvvm_tcgen05_mma_sp_block_scale_invalid_kind_tf32(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %spmetadata: !llvm.ptr<6>, %scale_a: !llvm.ptr<6>, %scale_b: !llvm.ptr<6>) {
  // expected-error @below {{attribute 'kind' failed to satisfy constraint: tcgen05 MMA Supported Types whose value is one of {mxf8f6f4, mxf4, mxf4nvf4}}}
  nvvm.tcgen05.mma.sp.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata, %scale_a, %scale_b {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>, !llvm.ptr<6>)
  llvm.return
}
