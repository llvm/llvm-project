// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @test_nvvm_elect_sync
llvm.func @test_nvvm_elect_sync() -> i1 {
  // CHECK: %[[RES:.*]] = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1)
  // CHECK-NEXT: %[[PRED:.*]] = extractvalue { i32, i1 } %[[RES]], 1
  // CHECK-NEXT: ret i1 %[[PRED]]
  %0 = nvvm.elect.sync -> i1
  llvm.return %0 : i1
}

// CHECK-LABEL: @test_nvvm_elect_sync_mask
llvm.func @test_nvvm_elect_sync_mask(%mask : i32) -> i1 {
  // CHECK: %[[RES:.*]] = call { i32, i1 } @llvm.nvvm.elect.sync(i32 %0)
  // CHECK-NEXT: %[[PRED:.*]] = extractvalue { i32, i1 } %[[RES]], 1
  // CHECK-NEXT: ret i1 %[[PRED]]
  %0 = nvvm.elect.sync %mask -> i1
  llvm.return %0 : i1
}

