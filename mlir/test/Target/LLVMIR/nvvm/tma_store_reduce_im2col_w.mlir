// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @tma_store_reduce_3d_im2colw(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %ch : i64) {
  // CHECK-LABEL: define void @tma_store_reduce_3d_im2colw(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i64 %5) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.3d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i64 %5, /* red_op=add */ i32 0, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.3d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i64 %5, /* red_op=min */ i32 1, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.3d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i64 %5, /* red_op=max */ i32 2, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.3d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i64 %5, /* red_op=inc */ i32 3, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.3d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i64 %5, /* red_op=dec */ i32 4, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.3d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i64 %5, /* red_op=and */ i32 5, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.3d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i64 %5, /* red_op=or */ i32 6, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.3d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i64 %5, /* red_op=xor */ i32 7, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.3d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i64 0, /* red_op=add */ i32 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.3d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i64 0, /* red_op=min */ i32 1, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.3d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i64 0, /* red_op=max */ i32 2, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.3d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i64 0, /* red_op=inc */ i32 3, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.3d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i64 0, /* red_op=dec */ i32 4, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.3d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i64 0, /* red_op=and */ i32 5, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.3d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i64 0, /* red_op=or */ i32 6, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.3d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i64 0, /* red_op=xor */ i32 7, i1 false)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = add mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = min mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = max mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = inc mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = dec mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = and mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = or mode = im2col_w  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] l2_cache_hint = %ch , reduction = xor mode = im2col_w : !llvm.ptr, !llvm.ptr<3>

  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = add mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = min mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = max mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = inc mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = dec mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = and mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = or mode = im2col_w  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2] , reduction = xor mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

llvm.func @tma_store_reduce_4d_im2colw(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %ch : i64) {
  // CHECK-LABEL: define void @tma_store_reduce_4d_im2colw(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 %6) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.4d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 %6, /* red_op=add */ i32 0, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.4d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 %6, /* red_op=min */ i32 1, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.4d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 %6, /* red_op=max */ i32 2, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.4d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 %6, /* red_op=inc */ i32 3, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.4d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 %6, /* red_op=dec */ i32 4, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.4d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 %6, /* red_op=and */ i32 5, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.4d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 %6, /* red_op=or */ i32 6, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.4d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 %6, /* red_op=xor */ i32 7, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.4d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=add */ i32 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.4d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=min */ i32 1, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.4d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=max */ i32 2, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.4d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=inc */ i32 3, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.4d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=dec */ i32 4, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.4d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=and */ i32 5, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.4d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=or */ i32 6, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.4d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 0, /* red_op=xor */ i32 7, i1 false)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = add mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = min mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = max mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = inc mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = dec mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = and mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = or mode = im2col_w  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] l2_cache_hint = %ch , reduction = xor mode = im2col_w : !llvm.ptr, !llvm.ptr<3>

  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = add mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = min mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = max mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = inc mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = dec mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = and mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = or mode = im2col_w  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3] , reduction = xor mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

llvm.func @tma_store_reduce_5d_im2colw(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ch : i64) {
  // CHECK-LABEL: define void @tma_store_reduce_5d_im2colw(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %7) {
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.5d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %7, /* red_op=add */ i32 0, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.5d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %7, /* red_op=min */ i32 1, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.5d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %7, /* red_op=max */ i32 2, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.5d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %7, /* red_op=inc */ i32 3, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.5d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %7, /* red_op=dec */ i32 4, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.5d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %7, /* red_op=and */ i32 5, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.5d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %7, /* red_op=or */ i32 6, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.5d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 %7, /* red_op=xor */ i32 7, i1 true)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.5d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, /* red_op=add */ i32 0, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.5d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, /* red_op=min */ i32 1, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.5d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, /* red_op=max */ i32 2, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.5d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, /* red_op=inc */ i32 3, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.5d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, /* red_op=dec */ i32 4, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.5d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, /* red_op=and */ i32 5, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.5d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, /* red_op=or */ i32 6, i1 false)
  // CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.reduce.im2col.w.5d(ptr addrspace(3) %0, ptr %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i64 0, /* red_op=xor */ i32 7, i1 false)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = add mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = min mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = max mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = inc mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = dec mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = and mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = or mode = im2col_w  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] l2_cache_hint = %ch , reduction = xor mode = im2col_w : !llvm.ptr, !llvm.ptr<3>

  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = add mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = min mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = max mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = inc mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = dec mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = and mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = or mode = im2col_w  : !llvm.ptr, !llvm.ptr<3>
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1, %d2, %d3, %d4] , reduction = xor mode = im2col_w : !llvm.ptr, !llvm.ptr<3>
  llvm.return
}
