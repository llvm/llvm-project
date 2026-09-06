// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_cta_1
llvm.func @nvvm_tcgen05_mma_sp_cta_1(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %spmetadata: !llvm.ptr<6>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = f16, cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* cta_group= */ i32 1, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = tf32, cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* cta_group= */ i32 1, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = f8f6f4, cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=i8 */ i32 3, /* cta_group= */ i32 1, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = i8, cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = f16, cta_group = <cta_1> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* cta_group= */ i32 1, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = tf32, cta_group = <cta_1> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* cta_group= */ i32 1, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = f8f6f4, cta_group = <cta_1> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=i8 */ i32 3, /* cta_group= */ i32 1, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = i8, cta_group = <cta_1> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = f16, cta_group = <cta_1> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* cta_group= */ i32 1, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = tf32, cta_group = <cta_1> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* cta_group= */ i32 1, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = f8f6f4, cta_group = <cta_1> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=i8 */ i32 3, /* cta_group= */ i32 1, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = i8, cta_group = <cta_1> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = f16, cta_group = <cta_1> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* cta_group= */ i32 1, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = tf32, cta_group = <cta_1> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* cta_group= */ i32 1, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = f8f6f4, cta_group = <cta_1> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=i8 */ i32 3, /* cta_group= */ i32 1, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = i8, cta_group = <cta_1> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_cta_2
llvm.func @nvvm_tcgen05_mma_sp_cta_2(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %spmetadata: !llvm.ptr<6>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 2, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = f16, cta_group = <cta_2> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* cta_group= */ i32 2, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = tf32, cta_group = <cta_2> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* cta_group= */ i32 2, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = f8f6f4, cta_group = <cta_2> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=i8 */ i32 3, /* cta_group= */ i32 2, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = i8, cta_group = <cta_2> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 2, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = f16, cta_group = <cta_2> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* cta_group= */ i32 2, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = tf32, cta_group = <cta_2> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* cta_group= */ i32 2, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = f8f6f4, cta_group = <cta_2> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=i8 */ i32 3, /* cta_group= */ i32 2, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = i8, cta_group = <cta_2> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 2, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = f16, cta_group = <cta_2> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* cta_group= */ i32 2, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = tf32, cta_group = <cta_2> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* cta_group= */ i32 2, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = f8f6f4, cta_group = <cta_2> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=i8 */ i32 3, /* cta_group= */ i32 2, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = i8, cta_group = <cta_2> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* cta_group= */ i32 2, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = f16, cta_group = <cta_2> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* cta_group= */ i32 2, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = tf32, cta_group = <cta_2> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* cta_group= */ i32 2, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = f8f6f4, cta_group = <cta_2> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=i8 */ i32 3, /* cta_group= */ i32 2, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
  , kind = i8, cta_group = <cta_2> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  llvm.return
}


// CHECK-LABEL: @nvvm_tcgen05_mma_sp_scale_d_imm_cta_1
llvm.func @nvvm_tcgen05_mma_sp_scale_d_imm_cta_1(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %spmetadata: !llvm.ptr<6>) {

  %scale_d_imm = llvm.mlir.constant(0:i64) : i64

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  , kind = f16, cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, /* kind=tf32 */ i32 1, /* cta_group= */ i32 1, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  , kind = tf32, cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  , kind = f16, cta_group = <cta_1> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, /* kind=tf32 */ i32 1, /* cta_group= */ i32 1, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  , kind = tf32, cta_group = <cta_1> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  , kind = f16, cta_group = <cta_1> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, /* kind=tf32 */ i32 1, /* cta_group= */ i32 1, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  , kind = tf32, cta_group = <cta_1> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, /* kind=f16 */ i32 0, /* cta_group= */ i32 1, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  , kind = f16, cta_group = <cta_1> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, /* kind=tf32 */ i32 1, /* cta_group= */ i32 1, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  , kind = tf32, cta_group = <cta_1> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_scale_d_imm_cta_2
llvm.func @nvvm_tcgen05_mma_sp_scale_d_imm_cta_2(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %spmetadata: !llvm.ptr<6>) {

  %scale_d_imm = llvm.mlir.constant(0:i64) : i64

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, /* kind=f16 */ i32 0, /* cta_group= */ i32 2, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  , kind = f16, cta_group = <cta_2> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, /* kind=tf32 */ i32 1, /* cta_group= */ i32 2, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  , kind = tf32, cta_group = <cta_2> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, /* kind=f16 */ i32 0, /* cta_group= */ i32 2, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  , kind = f16, cta_group = <cta_2> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, /* kind=tf32 */ i32 1, /* cta_group= */ i32 2, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  , kind = tf32, cta_group = <cta_2> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, /* kind=f16 */ i32 0, /* cta_group= */ i32 2, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  , kind = f16, cta_group = <cta_2> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, /* kind=tf32 */ i32 1, /* cta_group= */ i32 2, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  , kind = tf32, cta_group = <cta_2> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, /* kind=f16 */ i32 0, /* cta_group= */ i32 2, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  , kind = f16, cta_group = <cta_2> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, /* kind=tf32 */ i32 1, /* cta_group= */ i32 2, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm
  , kind = tf32, cta_group = <cta_2> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_disable_output_lane_cta_1
llvm.func @nvvm_tcgen05_mma_sp_disable_output_lane_cta_1(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %disableOutputLane : vector<4 x i32>, %spmetadata: !llvm.ptr<6>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = f16, cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = tf32, cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = f8f6f4, cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* kind=i8 */ i32 3, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = i8, cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = f16, cta_group = <cta_1> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = tf32, cta_group = <cta_1> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = f8f6f4, cta_group = <cta_1> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* kind=i8 */ i32 3, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = i8, cta_group = <cta_1> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = f16, cta_group = <cta_1> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = tf32, cta_group = <cta_1> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = f8f6f4, cta_group = <cta_1> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* kind=i8 */ i32 3, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = i8, cta_group = <cta_1> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = f16, cta_group = <cta_1> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = tf32, cta_group = <cta_1> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = f8f6f4, cta_group = <cta_1> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <4 x i32> {{%[0-9]+}}, /* kind=i8 */ i32 3, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = i8, cta_group = <cta_1> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<4 x i32>)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_disable_output_lane_cta_2
llvm.func @nvvm_tcgen05_mma_sp_disable_output_lane_cta_2(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %disableOutputLane: vector<8 x i32>, %spmetadata: !llvm.ptr<6>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = f16, cta_group = <cta_2> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = tf32, cta_group = <cta_2> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = f8f6f4, cta_group = <cta_2> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* kind=i8 */ i32 3, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = i8, cta_group = <cta_2> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = f16, cta_group = <cta_2> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = tf32, cta_group = <cta_2> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = f8f6f4, cta_group = <cta_2> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* kind=i8 */ i32 3, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = i8, cta_group = <cta_2> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = f16, cta_group = <cta_2> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = tf32, cta_group = <cta_2> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = f8f6f4, cta_group = <cta_2> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* kind=i8 */ i32 3, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = i8, cta_group = <cta_2> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = f16, cta_group = <cta_2> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = tf32, cta_group = <cta_2> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = f8f6f4, cta_group = <cta_2> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, <8 x i32> {{%[0-9]+}}, /* kind=i8 */ i32 3, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata mask = %disableOutputLane
  , kind = i8, cta_group = <cta_2> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, vector<8 x i32>)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_scale_d_imm_disable_output_lane_cta_1
llvm.func @nvvm_tcgen05_mma_sp_scale_d_imm_disable_output_lane_cta_1(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %disableOutputLane: vector<4 x i32>, %spmetadata: !llvm.ptr<6>) {

  %scale_d_imm = llvm.mlir.constant(0:i64) : i64

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <4 x i32> {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  , kind = f16, cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <4 x i32> {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  , kind = tf32, cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <4 x i32> {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  , kind = f16, cta_group = <cta_1> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <4 x i32> {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  , kind = tf32, cta_group = <cta_1> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <4 x i32> {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  , kind = f16, cta_group = <cta_1> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <4 x i32> {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  , kind = tf32, cta_group = <cta_1> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <4 x i32> {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  , kind = f16, cta_group = <cta_1> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<4 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg1(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <4 x i32> {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  , kind = tf32, cta_group = <cta_1> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<4 x i32>)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_sp_scale_d_imm_disable_output_lane_cta_2
llvm.func @nvvm_tcgen05_mma_sp_scale_d_imm_disable_output_lane_cta_2(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %adesc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %disableOutputLane: vector<8 x i32>, %spmetadata: !llvm.ptr<6>) {

  %scale_d_imm = llvm.mlir.constant(0:i64) : i64

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <8 x i32> {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  , kind = f16, cta_group = <cta_2> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <8 x i32> {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  , kind = tf32, cta_group = <cta_2> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <8 x i32> {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  , kind = f16, cta_group = <cta_2> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <8 x i32> {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_a=lastuse */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  , kind = tf32, cta_group = <cta_2> collector_a = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <8 x i32> {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  , kind = f16, cta_group = <cta_2> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <8 x i32> {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_a=fill */ i32 2, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  , kind = tf32, cta_group = <cta_2> collector_a = fill : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <8 x i32> {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  , kind = f16, cta_group = <cta_2> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<8 x i32>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.scale_d.disable_output_lane.cg2(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 0, <8 x i32> {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_a=use */ i32 3, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata scale = %scale_d_imm mask = %disableOutputLane
  , kind = tf32, cta_group = <cta_2> collector_a = use : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64, vector<8 x i32>)

  llvm.return
}
