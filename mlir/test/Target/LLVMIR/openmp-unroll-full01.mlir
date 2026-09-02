// Test lowering of omp.unroll_full (single loop)
// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @unroll_full_trivial_loop(%baseptr: !llvm.ptr) -> () {
  %tc = llvm.mlir.constant(100 : i32) : i32
  %literal_cli = omp.new_cli
  omp.canonical_loop(%literal_cli) %iv : i32 in range(%tc) {
    %ptr = llvm.getelementptr inbounds %baseptr[%iv] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %val = llvm.mlir.constant(42.0 : f32) : f32
    llvm.store %val, %ptr : f32, !llvm.ptr
    omp.terminator
  }
  omp.unroll_full(%literal_cli)
  llvm.return
}

// CHECK-LABEL: define void @unroll_full_trivial_loop(
// The loop is marked for full unrolling; LLVM's LoopUnroll pass performs it.
// CHECK: br label %omp_omp.loop.header, !llvm.loop ![[MD:[0-9]+]]
// CHECK: ![[MD]] = distinct !{![[MD]], ![[ENABLE:[0-9]+]], ![[FULL:[0-9]+]]}
// CHECK-DAG: ![[ENABLE]] = !{!"llvm.loop.unroll.enable"}
// CHECK-DAG: ![[FULL]] = !{!"llvm.loop.unroll.full"}
