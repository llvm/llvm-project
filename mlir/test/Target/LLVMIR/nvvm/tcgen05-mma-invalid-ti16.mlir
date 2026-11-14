// RUN: mlir-translate --mlir-to-llvmir -verify-diagnostics -split-input-file %s

// Invalid Tcgen05MMAKind for tcgen05.mma.block_scale: ti16 is valid only for
// mma/mma.sp/ws/ws.sp, not for block_scale ops.
// CHECK-LABEL: @nvvm_tcgen05_mma_block_scale_invalid_kind_ti16
llvm.func @nvvm_tcgen05_mma_block_scale_invalid_kind_ti16(%d_tmem : !llvm.ptr<6>, %a_desc: i64, %b_desc: i64, %idesc: i32, %enable_input_d: i1, %scale_a: !llvm.ptr<6>, %scale_b: !llvm.ptr<6>) {
  // expected-error @below {{attribute 'kind' failed to satisfy constraint: tcgen05 MMA Supported Types whose value is one of {mxf8f6f4, mxf4, mxf4nvf4}}}
  nvvm.tcgen05.mma.block_scale %d_tmem, %a_desc, %b_desc, %idesc, %enable_input_d, %scale_a, %scale_b, kind = ti16, cta_group = <cta_1> : (!llvm.ptr<6>, i64, i64, i32, i1, !llvm.ptr<6>, !llvm.ptr<6>)
  llvm.return
}
