// RUN: mlir-opt %s -convert-math-to-xevm | FileCheck %s

module @test_module {
  // CHECK: llvm.func @_Z22__spirv_ocl_native_expDh(f16) -> f16
  // CHECK: llvm.func @_Z22__spirv_ocl_native_expd(f64) -> f64
  // CHECK-LABEL: func @math_ops
  func.func @math_ops(%arg_f16 : f16, %arg_f64 : f64) -> (f16, f64) {

    // CHECK: llvm.call @_Z22__spirv_ocl_native_expDh(%{{.*}}) : (f16) -> f16
    %result16 = math.exp %arg_f16 fastmath<fast> : f16
    
    // CHECK: llvm.call @_Z22__spirv_ocl_native_expd(%{{.*}}) : (f64) -> f64
    %result64 = math.exp %arg_f64 fastmath<afn> : f64

    // CHECK: math.exp
    %result_no_fast = math.exp %arg_f64 : f64

    // TODO check fastmath<none>

    func.return %result16, %result64 : f16, f64
  }
}