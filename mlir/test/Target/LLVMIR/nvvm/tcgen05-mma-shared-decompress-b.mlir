// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_tcgen05_mma_shared_decompress_b_cta_1
llvm.func @nvvm_tcgen05_mma_shared_decompress_b_cta_1(
    %d_tmem             : !llvm.ptr<6>,
    %a_desc             : i64,
    %b_desc             : i64,
    %idesc              : i32,
    %enable_input_d     : i1,
    %decompress_metadata: !llvm.ptr<6>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 1, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 1, /* collector_a=discard */ i32 0, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_1> collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 1, /* collector_a=discard */ i32 0, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_1> collector_b = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 1, /* collector_a=discard */ i32 0, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_1> collector_b = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 1, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_1> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 1, /* collector_a=lastuse */ i32 1, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_1> collector_a = lastuse collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 1, /* collector_a=lastuse */ i32 1, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_1> collector_a = lastuse collector_b = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 1, /* collector_a=lastuse */ i32 1, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_1> collector_a = lastuse collector_b = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 1, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_1> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 1, /* collector_a=fill */ i32 2, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_1> collector_a = fill collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 1, /* collector_a=fill */ i32 2, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_1> collector_a = fill collector_b = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 1, /* collector_a=fill */ i32 2, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_1> collector_a = fill collector_b = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 1, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_1> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 1, /* collector_a=use */ i32 3, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_1> collector_a = use collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 1, /* collector_a=use */ i32 3, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_1> collector_a = use collector_b = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 1, /* collector_a=use */ i32 3, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_1> collector_a = use collector_b = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_shared_decompress_b_cta_2
llvm.func @nvvm_tcgen05_mma_shared_decompress_b_cta_2(
    %d_tmem             : !llvm.ptr<6>,
    %a_desc             : i64,
    %b_desc             : i64,
    %idesc              : i32,
    %enable_input_d     : i1,
    %decompress_metadata: !llvm.ptr<6>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 2, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_2> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 2, /* collector_a=discard */ i32 0, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_2> collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 2, /* collector_a=discard */ i32 0, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_2> collector_b = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 2, /* collector_a=discard */ i32 0, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_2> collector_b = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 2, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_2> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 2, /* collector_a=lastuse */ i32 1, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_2> collector_a = lastuse collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 2, /* collector_a=lastuse */ i32 1, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_2> collector_a = lastuse collector_b = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 2, /* collector_a=lastuse */ i32 1, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_2> collector_a = lastuse collector_b = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 2, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_2> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 2, /* collector_a=fill */ i32 2, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_2> collector_a = fill collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 2, /* collector_a=fill */ i32 2, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_2> collector_a = fill collector_b = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 2, /* collector_a=fill */ i32 2, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_2> collector_a = fill collector_b = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 2, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_2> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 2, /* collector_a=use */ i32 3, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_2> collector_a = use collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 2, /* collector_a=use */ i32 3, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_2> collector_a = use collector_b = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* cta_group= */ i32 2, /* collector_a=use */ i32 3, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata
  cta_group = <cta_2> collector_a = use collector_b = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_shared_decompress_b_mask_cta_1
llvm.func @nvvm_tcgen05_mma_shared_decompress_b_mask_cta_1(
    %d_tmem             : !llvm.ptr<6>,
    %a_desc             : i64,
    %b_desc             : i64,
    %idesc              : i32,
    %enable_input_d     : i1,
    %decompress_metadata: !llvm.ptr<6>,
    %wdm                : vector<4xi32>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg1.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg1.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* collector_a=discard */ i32 0, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_1> collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg1.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* collector_a=discard */ i32 0, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_1> collector_b = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg1.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* collector_a=discard */ i32 0, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_1> collector_b = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg1.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_1> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg1.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* collector_a=lastuse */ i32 1, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_1> collector_a = lastuse collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg1.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* collector_a=lastuse */ i32 1, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_1> collector_a = lastuse collector_b = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg1.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* collector_a=lastuse */ i32 1, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_1> collector_a = lastuse collector_b = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg1.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_1> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg1.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* collector_a=fill */ i32 2, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_1> collector_a = fill collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg1.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* collector_a=fill */ i32 2, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_1> collector_a = fill collector_b = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg1.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* collector_a=fill */ i32 2, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_1> collector_a = fill collector_b = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg1.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_1> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg1.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* collector_a=use */ i32 3, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_1> collector_a = use collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg1.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* collector_a=use */ i32 3, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_1> collector_a = use collector_b = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg1.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* collector_a=use */ i32 3, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_1> collector_a = use collector_b = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4xi32>)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_shared_decompress_b_mask_cta_2
llvm.func @nvvm_tcgen05_mma_shared_decompress_b_mask_cta_2(
    %d_tmem             : !llvm.ptr<6>,
    %a_desc             : i64,
    %b_desc             : i64,
    %idesc              : i32,
    %enable_input_d     : i1,
    %decompress_metadata: !llvm.ptr<6>,
    %wdm                : vector<8xi32>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg2.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_2> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg2.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* collector_a=discard */ i32 0, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_2> collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg2.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* collector_a=discard */ i32 0, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_2> collector_b = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg2.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* collector_a=discard */ i32 0, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_2> collector_b = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg2.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_2> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg2.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* collector_a=lastuse */ i32 1, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_2> collector_a = lastuse collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg2.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* collector_a=lastuse */ i32 1, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_2> collector_a = lastuse collector_b = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg2.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* collector_a=lastuse */ i32 1, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_2> collector_a = lastuse collector_b = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg2.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_2> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg2.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* collector_a=fill */ i32 2, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_2> collector_a = fill collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg2.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* collector_a=fill */ i32 2, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_2> collector_a = fill collector_b = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg2.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* collector_a=fill */ i32 2, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_2> collector_a = fill collector_b = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg2.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_2> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg2.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* collector_a=use */ i32 3, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_2> collector_a = use collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg2.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* collector_a=use */ i32 3, /* collector_b=fill */ i32 2)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_2> collector_a = use collector_b = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8xi32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg2.decompress_b(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* collector_a=use */ i32 3, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma.decompress_b %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %decompress_metadata, mask = %wdm
  cta_group = <cta_2> collector_a = use collector_b = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8xi32>)

  llvm.return
}
