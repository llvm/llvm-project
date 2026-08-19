// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @mbarrier_check_layout(%barrier: !llvm.ptr<3>) -> i1 {
  // CHECK-LABEL: define i1 @mbarrier_check_layout(ptr addrspace(3) %0) {
  // CHECK-NEXT: %[[V0:.+]] = call i1 @llvm.nvvm.mbarrier.check_layout.p3(ptr addrspace(3) %0, /* layout=v0 */ i32 0)
  // CHECK-NEXT: %[[V1:.+]] = call i1 @llvm.nvvm.mbarrier.check_layout.p3(ptr addrspace(3) %0, /* layout=v1 */ i32 1)
  // CHECK-NEXT: %[[RES:.+]] = or i1 %[[V0]], %[[V1]]
  // CHECK-NEXT: ret i1 %[[RES]]
  // CHECK-NEXT: }
  %v0 = nvvm.mbarrier.check_layout %barrier, 0 : !llvm.ptr<3> -> i1
  %v1 = nvvm.mbarrier.check_layout %barrier, 1 : !llvm.ptr<3> -> i1
  %res = llvm.or %v0, %v1 : i1
  llvm.return %res : i1
}
