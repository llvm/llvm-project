// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @nvvm_tcgen05_mma_tensor_ti16
llvm.func @nvvm_tcgen05_mma_tensor_ti16(%d_tmem : !llvm.ptr<6>, %a_tmem: !llvm.ptr<6>, %b_desc: i64, %idesc: i32, %enable_input_d: i1) {

  // ti16 enum value is 7, but maps to intrinsic kind 4.
  // CHECK: call void @llvm.nvvm.tcgen05.mma.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, /* kind=ti16 */ i32 4, /* cta_group= */ i32 1, /* collector_a=discard */ i32 0, /* collector_b=discard */ i32 0)
  nvvm.tcgen05.mma %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d
  , kind = ti16, cta_group = <cta_1> : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1)

  // collector_a=fill(2), collector_b=use(3), kind=ti16
  // CHECK: call void @llvm.nvvm.tcgen05.mma.tensor(ptr addrspace(6) {{%[0-9]+}}, ptr addrspace(6) {{%[0-9]+}}, i64 {{%[0-9]+}}, i32 {{%[0-9]+}}, i1 {{%[0-9]+}}, /* kind=ti16 */ i32 4, /* cta_group= */ i32 1, /* collector_a=fill */ i32 2, /* collector_b=use */ i32 3)
  nvvm.tcgen05.mma %d_tmem, %a_tmem, %b_desc, %idesc, %enable_input_d
  , kind = ti16, cta_group = <cta_1> collector_a = fill collector_b = use : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1)

  llvm.return
}
