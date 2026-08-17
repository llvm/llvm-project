// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_collector_b
llvm.func @nvvm_tcgen05_mma_sp_collector_b(%d_tmem : !llvm.ptr<6>, %a_tmem: !llvm.ptr<6>, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %spmetadata: !llvm.ptr<6>) {

  // collector_a=discard(0), collector_b=lastuse(1)
  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=discard */ i32 0, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOpB = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // collector_a=lastuse(1), collector_b=lastuse(1)
  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=lastuse */ i32 1, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>, collectorOpB = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // collector_a=fill(2), collector_b=lastuse(1)
  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=fill */ i32 2, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>, collectorOpB = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // collector_a=use(3), collector_b=lastuse(1)
  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=use */ i32 3, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>, collectorOpB = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // collector_a=discard(0), collector_b=fill(2)
  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=discard */ i32 0, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOpB = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // collector_a=lastuse(1), collector_b=fill(2)
  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=lastuse */ i32 1, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>, collectorOpB = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // collector_a=fill(2), collector_b=fill(2)
  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=fill */ i32 2, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>, collectorOpB = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // collector_a=use(3), collector_b=fill(2)
  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=use */ i32 3, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>, collectorOpB = #nvvm.tcgen05_mma_collectorop<fill>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // collector_a=discard(0), collector_b=use(3)
  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=discard */ i32 0, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOpB = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // collector_a=lastuse(1), collector_b=use(3)
  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=lastuse */ i32 1, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>, collectorOpB = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // collector_a=fill(2), collector_b=use(3)
  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=fill */ i32 2, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<fill>, collectorOpB = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // collector_a=use(3), collector_b=use(3)
  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=use */ i32 3, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>, collectorOp = #nvvm.tcgen05_mma_collectorop<use>, collectorOpB = #nvvm.tcgen05_mma_collectorop<use>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  llvm.return
}
