// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_tcgen05_mma_ws_sp
llvm.func @nvvm_tcgen05_mma_ws_sp(%d_tmem : !llvm.ptr<6>, %a_tmem: !llvm.ptr<6>, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %spmetadata: !llvm.ptr<6>) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 0, i32 0, i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 0, i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<tf32>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 0, i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 3, i32 0, i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<i8>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 0, i32 1, i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>,
   collectorBBuffer = #nvvm.tcgen05_mma_collectorb<b1>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 1, i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<tf32>,
   collectorBBuffer = #nvvm.tcgen05_mma_collectorb<b1>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 1, i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>,
   collectorBBuffer = #nvvm.tcgen05_mma_collectorb<b1>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 3, i32 1, i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<i8>,
   collectorBBuffer = #nvvm.tcgen05_mma_collectorb<b1>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 0, i32 1, i32 1)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f16>,
   collectorBBuffer = #nvvm.tcgen05_mma_collectorb<b1>,
   collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 1, i32 1, i32 1)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<tf32>,
   collectorBBuffer = #nvvm.tcgen05_mma_collectorb<b1>,
   collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 2, i32 1, i32 1)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>,
   collectorBBuffer = #nvvm.tcgen05_mma_collectorb<b1>,
   collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i32 3, i32 1, i32 1)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata
  {kind = #nvvm.tcgen05_mma_kind<i8>,
   collectorBBuffer = #nvvm.tcgen05_mma_collectorb<b1>,
   collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>)

  llvm.return
}

// CHECK-LABEL: @nvvm_tcgen05_mma_ws_sp_zero_col_mask
llvm.func @nvvm_tcgen05_mma_ws_sp_zero_col_mask(%d_tmem : !llvm.ptr<6>, %a_tmem: !llvm.ptr<6>, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %spmetadata: !llvm.ptr<6>, %zero_col_mask: i64) {

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 0, i32 0, i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
  {kind = #nvvm.tcgen05_mma_kind<f16>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 1, i32 0, i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
  {kind = #nvvm.tcgen05_mma_kind<tf32>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 2, i32 0, i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 3, i32 0, i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
  {kind = #nvvm.tcgen05_mma_kind<i8>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 0, i32 1, i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
  {kind = #nvvm.tcgen05_mma_kind<f16>,
   collectorBBuffer = #nvvm.tcgen05_mma_collectorb<b1>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 1, i32 1, i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
  {kind = #nvvm.tcgen05_mma_kind<tf32>,
   collectorBBuffer = #nvvm.tcgen05_mma_collectorb<b1>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 2, i32 1, i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>,
   collectorBBuffer = #nvvm.tcgen05_mma_collectorb<b1>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 3, i32 1, i32 0)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
  {kind = #nvvm.tcgen05_mma_kind<i8>,
   collectorBBuffer = #nvvm.tcgen05_mma_collectorb<b1>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 0, i32 1, i32 1)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
  {kind = #nvvm.tcgen05_mma_kind<f16>,
   collectorBBuffer = #nvvm.tcgen05_mma_collectorb<b1>,
   collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 1, i32 1, i32 1)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
  {kind = #nvvm.tcgen05_mma_kind<tf32>,
   collectorBBuffer = #nvvm.tcgen05_mma_collectorb<b1>,
   collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 2, i32 1, i32 1)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
  {kind = #nvvm.tcgen05_mma_kind<f8f6f4>,
   collectorBBuffer = #nvvm.tcgen05_mma_collectorb<b1>,
   collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, i64)

  // CHECK: call void @llvm.nvvm.tcgen05.mma.ws.sp.tensor.zero_col_mask(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 3, i32 1, i32 1)
  nvvm.tcgen05.mma.ws.sp %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d, %spmetadata, %zero_col_mask
  {kind = #nvvm.tcgen05_mma_kind<i8>,
   collectorBBuffer = #nvvm.tcgen05_mma_collectorb<b1>,
   collectorOp = #nvvm.tcgen05_mma_collectorop<lastuse>} : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1, !llvm.ptr<6>, i64)

  llvm.return
}
