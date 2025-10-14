// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_tcgen05_mma_mxf8f6f4_block_scale_cta_1
llvm.func @nvvm_tcgen05_mma_mxf8f6f4_block_scale_cta_1(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %scale_a: !llvm.ptr<6>, %scale_b : !llvm.ptr<6>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 0)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 0)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 1)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 1)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 2)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 2)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 3)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 3)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_mxf8f6f4_block_scale_cta_2
llvm.func @nvvm_tcgen05_mma_mxf8f6f4_block_scale_cta_2(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %scale_a: !llvm.ptr<6>, %scale_b : !llvm.ptr<6>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 0)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_2>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 0)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_2>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 1)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 1)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_2>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 2)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 2)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_2>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 3)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 3)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf8f6f4>, ctaGroup = #nvvm.cta_group<cta_2>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_mxf4_block_scale_cta_1
llvm.func @nvvm_tcgen05_mma_mxf4_block_scale_cta_1(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %scale_a: !llvm.ptr<6>, %scale_b : !llvm.ptr<6>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4.block_scale(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 0)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 0)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_1>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4.block_scale(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 1)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 1)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_1>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4.block_scale(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 2)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 2)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_1>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4.block_scale(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 3)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 3)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_1>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_mxf4_block_scale_cta_2
llvm.func @nvvm_tcgen05_mma_mxf4_block_scale_cta_2(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %scale_a: !llvm.ptr<6>, %scale_b : !llvm.ptr<6>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4.block_scale(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 0)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_2>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 0)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_2>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4.block_scale(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 1)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 1)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_2>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4.block_scale(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 2)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 2)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_2>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4.block_scale(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 3)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 3)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4>, ctaGroup = #nvvm.cta_group<cta_2>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_mxf4nvf4_block_scale_cta_1
llvm.func @nvvm_tcgen05_mma_mxf4nvf4_block_scale_cta_1(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %scale_a: !llvm.ptr<6>, %scale_b : !llvm.ptr<6>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block16(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 0)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_1>, blockScale = #nvvm.tcgen05_mma_block_scale<block16>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 0)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_1>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block16(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 1)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>, blockScale = #nvvm.tcgen05_mma_block_scale<block16>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 1)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_1>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block16(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 2)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>, blockScale = #nvvm.tcgen05_mma_block_scale<block16>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 2)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_1>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block16(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 3)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>, blockScale = #nvvm.tcgen05_mma_block_scale<block16>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 3)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_1>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_mxf4nvf4_block_scale_cta_2
llvm.func @nvvm_tcgen05_mma_mxf4nvf4_block_scale_cta_2(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %scale_a: !llvm.ptr<6>, %scale_b : !llvm.ptr<6>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block16(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 0)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_2>, blockScale = #nvvm.tcgen05_mma_block_scale<block16>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 0)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_2>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block16(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 1)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>, blockScale = #nvvm.tcgen05_mma_block_scale<block16>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 1)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_2>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block16(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 2)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>, blockScale = #nvvm.tcgen05_mma_block_scale<block16>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 2)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_2>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block16(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 3)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>, blockScale = #nvvm.tcgen05_mma_block_scale<block16>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf4nvf4.block_scale.block32(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 3)
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b
  {kind = #nvvm.tcgen05_mma_block_scale_kind<mxf4nvf4>, ctaGroup = #nvvm.cta_group<cta_2>, blockScale = #nvvm.tcgen05_mma_block_scale<block32>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)

  llvm.return
}
