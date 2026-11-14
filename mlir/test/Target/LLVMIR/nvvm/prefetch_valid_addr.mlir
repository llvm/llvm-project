// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s
// RUN: %if has_llc_binary && host-supports-nvptx %{ mlir-translate --mlir-to-llvmir %s | %llc-verify -march=nvptx64 -mcpu=sm_90 -mattr=+ptx94 %}

// CHECK-LABEL: define void @prefetch_generic(
// CHECK-SAME: ptr %[[ADDR:[0-9]+]])
// CHECK-NEXT: call void @llvm.nvvm.prefetch.L1.32B.valid_addr.p0(ptr %[[ADDR]])
// CHECK-NEXT: ret void
llvm.func @prefetch_generic(%addr: !llvm.ptr) {
  nvvm.prefetch.valid_addr level = L1, %addr : !llvm.ptr
  llvm.return
}

// CHECK-LABEL: define void @prefetch_global(
// CHECK-SAME: ptr addrspace(1) %[[ADDR:[0-9]+]])
// CHECK-NEXT: call void @llvm.nvvm.prefetch.L1.32B.valid_addr.p1(ptr addrspace(1) %[[ADDR]])
// CHECK-NEXT: ret void
llvm.func @prefetch_global(%addr: !llvm.ptr<1>) {
  nvvm.prefetch.valid_addr level = L1, %addr : !llvm.ptr<1>
  llvm.return
}
