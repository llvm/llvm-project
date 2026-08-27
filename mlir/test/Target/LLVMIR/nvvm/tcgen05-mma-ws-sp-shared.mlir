// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_tcgen05_mma_ws_sp
llvm.func @nvvm_tcgen05_mma_ws_sp(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %spmetadata: !llvm.ptr<6>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_b_buffer=b0 */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
   kind = f16 : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_b_buffer=b0 */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
   kind = tf32 : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* collector_b_buffer=b0 */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
   kind = f8f6f4 : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=i8 */ i32 3, /* collector_b_buffer=b0 */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
   kind = i8 : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_b_buffer=b1 */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
   kind = f16 collector_b_buffer = b1 : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_b_buffer=b1 */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
   kind = tf32 collector_b_buffer = b1 : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* collector_b_buffer=b1 */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
   kind = f8f6f4 collector_b_buffer = b1 : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=i8 */ i32 3, /* collector_b_buffer=b1 */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
   kind = i8 collector_b_buffer = b1 : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_b_buffer=b1 */ i32 1, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
   kind = f16 collector_b_buffer = b1 collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_b_buffer=b1 */ i32 1, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
   kind = tf32 collector_b_buffer = b1 collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* collector_b_buffer=b1 */ i32 1, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
   kind = f8f6f4 collector_b_buffer = b1 collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, /* kind=i8 */ i32 3, /* collector_b_buffer=b1 */ i32 1, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata
   kind = i8 collector_b_buffer = b1 collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_ws_sp_zero_col_mask
llvm.func @nvvm_tcgen05_mma_ws_sp_zero_col_mask(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %spmetadata: !llvm.ptr<6>, %zero_col_mask: i64) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_b_buffer=b0 */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
   kind = f16 : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_b_buffer=b0 */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
   kind = tf32 : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* collector_b_buffer=b0 */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
   kind = f8f6f4 : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, /* kind=i8 */ i32 3, /* collector_b_buffer=b0 */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
   kind = i8 : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_b_buffer=b1 */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
   kind = f16 collector_b_buffer = b1 : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_b_buffer=b1 */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
   kind = tf32 collector_b_buffer = b1 : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* collector_b_buffer=b1 */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
   kind = f8f6f4 collector_b_buffer = b1 : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, /* kind=i8 */ i32 3, /* collector_b_buffer=b1 */ i32 1, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
   kind = i8 collector_b_buffer = b1 : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, /* kind=f16 */ i32 0, /* collector_b_buffer=b1 */ i32 1, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
   kind = f16 collector_b_buffer = b1 collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, /* kind=tf32 */ i32 1, /* collector_b_buffer=b1 */ i32 1, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
   kind = tf32 collector_b_buffer = b1 collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, /* kind=f8f6f4 */ i32 2, /* collector_b_buffer=b1 */ i32 1, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
   kind = f8f6f4 collector_b_buffer = b1 collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.shared.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, /* kind=i8 */ i32 3, /* collector_b_buffer=b1 */ i32 1, /* collector_b=lastuse */ i32 1)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
   kind = i8 collector_b_buffer = b1 collector_b = lastuse : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, i64)

  llvm.return
}
