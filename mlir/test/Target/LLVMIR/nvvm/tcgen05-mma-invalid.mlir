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
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>, aShift} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)
  llvm.return
}

// -----

// CHECK-LABEL: @nvvm_tcgen05_mma_mxf4_block_scale_default
llvm.func @nvvm_tcgen05_mma_mxf4_block_scale_default(%d_tmem : !llvm.ptr<6>, %a_tmem: !llvm.ptr<6>, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %scalea: !llvm.ptr<6>, %scaleb: !llvm.ptr<6>) {
  // expected-error @below {{mxf4 kind does not support block16 attribute}}
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %scalea, %scaleb
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>, ashift, blockScale = #nvvm.tcgen05_mma_block_scale<block16>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)
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
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>, aShift} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>, !llvm.ptr<6>)
  llvm.return
}

// -----

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_mxf4_block_scale_default
llvm.func @nvvm_tcgen05_mma_sp_mxf4_block_scale_default(%d_tmem : !llvm.ptr<6>, %a_tmem: !llvm.ptr<6>, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %scalea: !llvm.ptr<6>, %scaleb: !llvm.ptr<6>, %spmetadata: !llvm.ptr<6>) {
  // expected-error @below {{mxf4 kind does not support block16 attribute}}
  nvvm.tcgen05.mma.sp.block_scale %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata, %scalea, %scaleb
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>, ashift, blockScale = #nvvm.tcgen05_mma_block_scale<block16>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>, !llvm.ptr<6>)
  llvm.return
}
