// RUN: mlir-opt %s -gpu-module-to-binary="format=isa" \
// RUN:             -debug-only=serialize-to-isa 2> %t 
// RUN: FileCheck --input-file=%t %s
//
// MathToXeVM pass generates OpenCL intrinsics function calls when converting
// Math ops with `fastmath` attr to native function calls. It is assumed that
// the SPIRV backend would correctly convert these intrinsics calls to OpenCL
// ExtInst instructions in SPIRV (See llvm/lib/Target/SPIRV/SPIRVBuiltins.cpp).
//
// To ensure this assumption holds, this test verifies that the SPIRV backend
// behaves as expected.

module @test_ocl_intrinsics attributes {gpu.container_module} {
  gpu.module @kernel [#xevm.target] {
    llvm.func spir_kernelcc @native_fcns() attributes {gpu.kernel} {
      // CHECK-DAG: %[[F16T:.+]] = OpTypeFloat 16
      // CHECK-DAG: %[[ZERO_F16:.+]] = OpConstantNull %[[F16T]]
      %c0_f16 = llvm.mlir.constant(0. : f16) : f16
      // CHECK-DAG: %[[F64T:.+]] = OpTypeFloat 64
      // CHECK-DAG: %[[ZERO_F64:.+]] = OpConstantNull %[[F64T]]
      %c0_f64 = llvm.mlir.constant(0. : f64) : f64

      // CHECK: %{{.+}} = OpExtInst %[[F16T]] %{{.+}} native_exp %[[ZERO_F16]]
      %exp_f16 = llvm.call @_Z22__spirv_ocl_native_expDh(%c0_f16) : (f16) -> f16
      // CHECK: %{{.+}} = OpExtInst %[[F64T]] %{{.+}} native_exp %[[ZERO_F64]]
      %exp_f64 = llvm.call @_Z22__spirv_ocl_native_expd(%c0_f64) : (f64) -> f64

      llvm.return
    }
    llvm.func @_Z22__spirv_ocl_native_expDh(f16) -> f16
    llvm.func @_Z22__spirv_ocl_native_expd(f64) -> f64
  }
}
