// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_tcgen05_mma_ws_tensor_ti16
llvm.func @nvvm_tcgen05_mma_ws_tensor_ti16(%d_tmem : !llvm.ptr<6>, %a_tmem: !llvm.ptr<6>, %b_desc: i64, %idesc: i32, %enable_input_d: i1) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, /* kind=ti16 */ i32 4, /* collector_b_buffer=b0 */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.ws %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d
   kind = ti16 : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_ws_tensor_zero_col_mask_ti16
llvm.func @nvvm_tcgen05_mma_ws_tensor_zero_col_mask_ti16(%d_tmem : !llvm.ptr<6>, %a_tmem: !llvm.ptr<6>, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %zero_col_mask: i64) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.tensor.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, i64 {{%[0-9]+}}, /* kind=ti16 */ i32 4, /* collector_b_buffer=b0 */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma.ws %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %zero_col_mask
   kind = ti16 : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, i64)

  llvm.return
}
