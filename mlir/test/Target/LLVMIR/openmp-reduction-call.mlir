// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Test that we don't crash when there is a call operation in the combiner

omp.declare_reduction @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = llvm.mlir.constant(0.0 : f32) : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
// test this call here:
  llvm.call @test_call() : () -> ()
  %1 = llvm.fadd %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}

llvm.func @simple_reduction(%lb : i64, %ub : i64, %step : i64) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  omp.parallel reduction(@add_f32 %0 -> %prv : !llvm.ptr) {
    %1 = llvm.mlir.constant(2.0 : f32) : f32
    %2 = llvm.load %prv : !llvm.ptr -> f32
    %3 = llvm.fadd %1, %2 : f32
    llvm.store %3, %prv : f32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

llvm.func @test_call() -> ()

// Call to the troublesome function will have been inlined twice: once into
// main and once into the outlined function
// CHECK: call void @test_call()
// CHECK: call void @test_call()
