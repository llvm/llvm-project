// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @nvvm_pmevent_mask() {
  // CHECK-LABEL: define void @nvvm_pmevent_mask() {
  // CHECK-NEXT: call void @llvm.nvvm.pm.event.mask(i16 15000)
  // CHECK-NEXT: call void @llvm.nvvm.pm.event.mask(i16 4)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.pmevent mask = 15000
  nvvm.pmevent mask = 4
  llvm.return
}

llvm.func @nvvm_pmevent_id() {
  // CHECK-LABEL: define void @nvvm_pmevent_id() {
  // CHECK-NEXT: call void @llvm.nvvm.pm.event.mask(i16 1024)
  // CHECK-NEXT: call void @llvm.nvvm.pm.event.mask(i16 16)
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  nvvm.pmevent id = 10
  nvvm.pmevent id = 4
  llvm.return
}
