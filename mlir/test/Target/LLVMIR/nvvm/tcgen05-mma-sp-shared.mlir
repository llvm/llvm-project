// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_cta_1
llvm.func @nvvm_tcgen05_mma_sp_cta_1(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %spmetadata: !llvm.ptr<6>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 0, i32 1, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 1, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 1, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 3, i32 1, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<i8>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 0, i32 1, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 1, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 1, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 3, i32 1, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<i8>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 0, i32 1, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 1, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 1, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 3, i32 1, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<i8>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 0, i32 1, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 1, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 1, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 3, i32 1, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<i8>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_cta_2
llvm.func @nvvm_tcgen05_mma_sp_cta_2(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %spmetadata: !llvm.ptr<6>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 0, i32 2, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_2>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 2, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_2>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 2, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>, ctaGroup = #nvvm.cta_group<cta_2>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 3, i32 2, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<i8>, ctaGroup = #nvvm.cta_group<cta_2>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 0, i32 2, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 2, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 2, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 3, i32 2, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<i8>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 0, i32 2, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 2, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 2, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 3, i32 2, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<i8>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 0, i32 2, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 2, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 2, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 3, i32 2, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<i8>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  llvm.return
}


// CHECK-LABEL: @nvvm_tcgen05_mma_sp_scale_d_imm_cta_1
llvm.func @nvvm_tcgen05_mma_sp_scale_d_imm_cta_1(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %spmetadata: !llvm.ptr<6>) {

  %scale_d_imm = llvm.mlir.constant(0:i64) : i64

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i32 0, i32 1, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i32 1, i32 1, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i32 0, i32 1, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i32 1, i32 1, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i32 0, i32 1, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i32 1, i32 1, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i32 0, i32 1, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i32 1, i32 1, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_scale_d_imm_cta_2
llvm.func @nvvm_tcgen05_mma_sp_scale_d_imm_cta_2(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %spmetadata: !llvm.ptr<6>) {

  %scale_d_imm = llvm.mlir.constant(0:i64) : i64

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i32 0, i32 2, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_2>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i32 1, i32 2, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_2>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i32 0, i32 2, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i32 1, i32 2, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i32 0, i32 2, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i32 1, i32 2, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i32 0, i32 2, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, i32 1, i32 2, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_disable_output_lane_cta_1
llvm.func @nvvm_tcgen05_mma_sp_disable_output_lane_cta_1(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %disableOutputLane : vector<4 x i32>, %spmetadata: !llvm.ptr<6>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i32 0, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i32 1, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i32 2, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i32 3, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<i8>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i32 0, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i32 1, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i32 2, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i32 3, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<i8>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i32 0, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i32 1, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i32 2, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i32 3, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<i8>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i32 0, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i32 1, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i32 2, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, i32 3, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<i8>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_disable_output_lane_cta_2
llvm.func @nvvm_tcgen05_mma_sp_disable_output_lane_cta_2(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %disableOutputLane: vector<8 x i32>, %spmetadata: !llvm.ptr<6>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i32 0, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_2>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i32 1, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_2>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i32 2, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>, ctaGroup = #nvvm.cta_group<cta_2>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i32 3, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<i8>, ctaGroup = #nvvm.cta_group<cta_2>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i32 0, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i32 1, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i32 2, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i32 3, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<i8>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i32 0, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i32 1, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i32 2, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i32 3, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<i8>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i32 0, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i32 1, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i32 2, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, i32 3, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<i8>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_scale_d_imm_disable_output_lane_cta_1
llvm.func @nvvm_tcgen05_mma_sp_scale_d_imm_disable_output_lane_cta_1(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %disableOutputLane: vector<4 x i32>, %spmetadata: !llvm.ptr<6>) {

  %scale_d_imm = llvm.mlir.constant(0:i64) : i64

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <4 x i32> {{%[0-9]+}}, i32 0, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <4 x i32> {{%[0-9]+}}, i32 1, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_1>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <4 x i32> {{%[0-9]+}}, i32 0, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <4 x i32> {{%[0-9]+}}, i32 1, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <4 x i32> {{%[0-9]+}}, i32 0, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <4 x i32> {{%[0-9]+}}, i32 1, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <4 x i32> {{%[0-9]+}}, i32 0, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <4 x i32> {{%[0-9]+}}, i32 1, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<4 x i32>)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_scale_d_imm_disable_output_lane_cta_2
llvm.func @nvvm_tcgen05_mma_sp_scale_d_imm_disable_output_lane_cta_2(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %disableOutputLane: vector<8 x i32>, %spmetadata: !llvm.ptr<6>) {

  %scale_d_imm = llvm.mlir.constant(0:i64) : i64

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <8 x i32> {{%[0-9]+}}, i32 0, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_2>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <8 x i32> {{%[0-9]+}}, i32 1, i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_2>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <8 x i32> {{%[0-9]+}}, i32 0, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <8 x i32> {{%[0-9]+}}, i32 1, i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <8 x i32> {{%[0-9]+}}, i32 0, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <8 x i32> {{%[0-9]+}}, i32 1, i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <8 x i32> {{%[0-9]+}}, i32 0, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <8 x i32> {{%[0-9]+}}, i32 1, i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  {kind = #nvvm.tcgen05_mma_kind<tf32>, ctaGroup = #nvvm.cta_group<cta_2>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<8 x i32>)

  llvm.return
}
