// Tests that static alloca's in `omp.private ... init` regions are hoisted to
// the parent construct's alloca IP.
// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @foo1()
llvm.func @foo2()
llvm.func @foo3()
llvm.func @foo4()

omp.private {type = private} @multi_block.privatizer : f32 init {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i32) : i32
  %alloca1 = llvm.alloca %0 x !llvm.struct<(i64)> {alignment = 8 : i64} : (i32) -> !llvm.ptr

  %1 = llvm.load %arg0 : !llvm.ptr -> f32

  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c2 = llvm.mlir.constant(2 : i32) : i32
  %cond1 = llvm.icmp "eq" %c1, %c2 : i32
  llvm.cond_br %cond1, ^bb1, ^bb2

^bb1:
  llvm.call @foo1() : () -> ()
  llvm.br ^bb3

^bb2:
  llvm.call @foo2() : () -> ()
  llvm.br ^bb3

^bb3:
  llvm.store %1, %arg1 : f32, !llvm.ptr

  omp.yield(%arg1 : !llvm.ptr)
}

omp.private {type = private} @multi_block.privatizer2 : f32 init {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i32) : i32
  %alloca1 = llvm.alloca %0 x !llvm.struct<(ptr)> {alignment = 8 : i64} : (i32) -> !llvm.ptr

  %1 = llvm.load %arg0 : !llvm.ptr -> f32

  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c2 = llvm.mlir.constant(2 : i32) : i32
  %cond1 = llvm.icmp "eq" %c1, %c2 : i32
  llvm.cond_br %cond1, ^bb1, ^bb2

^bb1:
  llvm.call @foo3() : () -> ()
  llvm.br ^bb3

^bb2:
  llvm.call @foo4() : () -> ()
  llvm.br ^bb3

^bb3:
  llvm.store %1, %arg1 : f32, !llvm.ptr

  omp.yield(%arg1 : !llvm.ptr)
}

llvm.func @parallel_op_private_multi_block(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  omp.parallel private(@multi_block.privatizer %arg0 -> %arg2,
                       @multi_block.privatizer2 %arg1 -> %arg3 : !llvm.ptr, !llvm.ptr) {
    %0 = llvm.load %arg2 : !llvm.ptr -> f32
    %1 = llvm.load %arg3 : !llvm.ptr -> f32
    omp.terminator
  }
  llvm.return
}

// CHECK: define internal void @parallel_op_private_multi_block..omp_par({{.*}}) {{.*}} {
// CHECK: omp.par.entry:
// Varify that both allocas were hoisted to the parallel region's entry block.
// CHECK:        %{{.*}} = alloca { i64 }, align 8
// CHECK-NEXT:   %{{.*}} = alloca { ptr }, align 8
// CHECK-NEXT:   br label %omp.region.after_alloca
// CHECK: omp.region.after_alloca:
// CHECK: }
