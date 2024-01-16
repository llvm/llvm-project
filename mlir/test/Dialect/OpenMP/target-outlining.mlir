// RUN: mlir-opt %s --omp-target-outline-to-gpu | FileCheck %s

module attributes {omp.is_target_device = false, omp.is_gpu = false} {
  func.func @targetFn() -> () attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} {
    return
  }
  llvm.func @main() {
    omp.target {
      func.call @targetFn() : () -> ()
      omp.terminator
    }
    omp.target {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: module attributes {gpu.container_module, omp.is_gpu = false, omp.is_target_device = false} {
// CHECK-NEXT: gpu.module @[[DEV_MODULE:.*]] <#gpu.offload_embedding<omp>> attributes {omp.is_gpu = true, omp.is_target_device = true} {
// CHECK-NEXT: func.func @{{.*}}() attributes {omp.outline_parent_name = "main"} {
// CHECK-NEXT: omp.target info = #omp.tgt_entry_info<deviceID = [[DEVID_1:.*]], fileID = [[FILEID_1:.*]], line = [[LINE_1:.*]], section = @[[DEV_MODULE]]> {
// CHECK-NEXT: func.call @targetFn() : () -> ()
// CHECK-NEXT: omp.terminator
// CHECK-NEXT: }
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func.func @targetFn() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} {
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func.func @{{.*}}() attributes {omp.outline_parent_name = "main"} {
// CHECK-NEXT: omp.target info = #omp.tgt_entry_info<deviceID = [[DEVID_2:.*]], fileID = [[FILEID_2:.*]], line = [[LINE_2:.*]], section = @[[DEV_MODULE]]> {
// CHECK-NEXT: omp.terminator
// CHECK-NEXT: }
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: func.func @targetFn() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} {
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @main() {
// CHECK-NEXT: omp.target info = #omp.tgt_entry_info<deviceID = [[DEVID_1]], fileID = [[FILEID_1]], line = [[LINE_1]], section = @[[DEV_MODULE]]> {
// CHECK-NEXT: func.call @targetFn() : () -> ()
// CHECK-NEXT: omp.terminator
// CHECK-NEXT: }
// CHECK-NEXT: omp.target info = #omp.tgt_entry_info<deviceID = [[DEVID_2]], fileID = [[FILEID_2]], line = [[LINE_2]], section = @[[DEV_MODULE]]> {
// CHECK-NEXT: omp.terminator
// CHECK-NEXT: }
// CHECK-NEXT: llvm.return
// CHECK-NEXT: }
// CHECK-NEXT: }
