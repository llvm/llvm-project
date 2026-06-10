// RUN: mlir-opt %s -convert-gpu-to-rocdl='chipset=gfx1250' --mlir-print-local-scope | FileCheck %s

gpu.module @test_module {

// CHECK-LABEL: func @named_barrier
func.func @named_barrier() {
  %member_count = arith.constant 4 : i32
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @[[NB:__named_barrier[_0-9]*]] : !llvm.ptr<3>
  // CHECK: rocdl.s.barrier.init %[[ADDR]] member_cnt = 4
  %nb = gpu.initialize_named_barrier %member_count : i32 -> !gpu.named_barrier
  // CHECK: llvm.fence syncscope("workgroup") release
  // CHECK: rocdl.s.barrier.join %[[ADDR]]
  // CHECK: rocdl.s.barrier.signal.var %[[ADDR]] member_cnt = 0
  // CHECK: rocdl.s.barrier.wait id = 1
  // CHECK: llvm.fence syncscope("workgroup") acquire
  gpu.barrier named(%nb : !gpu.named_barrier)
  func.return
}

// CHECK-LABEL: func @two_named_barriers
func.func @two_named_barriers() {
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  // CHECK: %[[ADDR0:.*]] = llvm.mlir.addressof @[[NB0:__named_barrier[_0-9]*]] : !llvm.ptr<3>
  // CHECK: rocdl.s.barrier.init %[[ADDR0]] member_cnt = 4
  %nb0 = gpu.initialize_named_barrier %c4 : i32 -> !gpu.named_barrier
  // CHECK: %[[ADDR1:.*]] = llvm.mlir.addressof @[[NB1:__named_barrier[_0-9]*]] : !llvm.ptr<3>
  // CHECK: rocdl.s.barrier.init %[[ADDR1]] member_cnt = 8
  %nb1 = gpu.initialize_named_barrier %c8 : i32 -> !gpu.named_barrier
  // CHECK: rocdl.s.barrier.join %[[ADDR0]]
  // CHECK: rocdl.s.barrier.signal.var %[[ADDR0]] member_cnt = 0
  // CHECK: rocdl.s.barrier.wait id = 1
  gpu.barrier named(%nb0 : !gpu.named_barrier)
  // CHECK: rocdl.s.barrier.join %[[ADDR1]]
  // CHECK: rocdl.s.barrier.signal.var %[[ADDR1]] member_cnt = 0
  // CHECK: rocdl.s.barrier.wait id = 1
  gpu.barrier named(%nb1 : !gpu.named_barrier)
  func.return
}

// CHECK-LABEL: func @cluster_scope
func.func @cluster_scope() {
  // CHECK: llvm.fence syncscope("cluster") release
  // CHECK-NEXT: rocdl.s.barrier.signal id = -3
  // CHECK-NEXT: rocdl.s.barrier.wait id = -3
  // CHECK-NEXT: llvm.fence syncscope("cluster") acquire
  gpu.barrier scope <cluster>
  func.return
}

// One LDS global per gpu.initialize_named_barrier.
// CHECK-COUNT-3: llvm.mlir.global internal @__named_barrier{{[_0-9]*}}() {addr_space = 3 : i32} : !llvm.target<"amdgcn.named.barrier", 0>

}
