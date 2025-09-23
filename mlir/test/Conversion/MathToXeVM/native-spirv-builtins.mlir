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
      // CHECK-DAG: %[[F32T:.+]] = OpTypeFloat 32
      // CHECK-DAG: %[[ZERO_F32:.+]] = OpConstantNull %[[F32T]]
      %c0_f32 = llvm.mlir.constant(0. : f32) : f32
      // CHECK-DAG: %[[F64T:.+]] = OpTypeFloat 64
      // CHECK-DAG: %[[ZERO_F64:.+]] = OpConstantNull %[[F64T]]
      %c0_f64 = llvm.mlir.constant(0. : f64) : f64

      // CHECK-DAG: %[[V2F64T:.+]] = OpTypeVector %[[F64T]] 2
      // CHECK-DAG: %[[V2_ZERO_F64:.+]] = OpConstantNull %[[V2F64T]]
      %v2_c0_f64 = llvm.mlir.constant(dense<0.> : vector<2xf64>) : vector<2xf64>
      // CHECK-DAG: %[[V3F32T:.+]] = OpTypeVector %[[F32T]] 3
      // CHECK-DAG: %[[V3_ZERO_F32:.+]] = OpConstantNull %[[V3F32T]]
      %v3_c0_f32 = llvm.mlir.constant(dense<0.> : vector<3xf32>) : vector<3xf32>
      // CHECK-DAG: %[[V4F64T:.+]] = OpTypeVector %[[F64T]] 4
      // CHECK-DAG: %[[V4_ZERO_F64:.+]] = OpConstantNull %[[V4F64T]]
      %v4_c0_f64 = llvm.mlir.constant(dense<0.> : vector<4xf64>) : vector<4xf64>
      // CHECK-DAG: %[[V8F64T:.+]] = OpTypeVector %[[F64T]] 8
      // CHECK-DAG: %[[V8_ZERO_F64:.+]] = OpConstantNull %[[V8F64T]]
      %v8_c0_f64 = llvm.mlir.constant(dense<0.> : vector<8xf64>) : vector<8xf64>
      // CHECK-DAG: %[[V16F16T:.+]] = OpTypeVector %[[F16T]] 16
      // CHECK-DAG: %[[V16_ZERO_F16:.+]] = OpConstantNull %[[V16F16T]]
      %v16_c0_f16 = llvm.mlir.constant(dense<0.> : vector<16xf16>) : vector<16xf16>     

      // CHECK: %{{.+}} = OpExtInst %[[F16T]] %{{.+}} native_exp %[[ZERO_F16]]
      %exp_f16 = llvm.call @_Z22__spirv_ocl_native_expDh(%c0_f16) : (f16) -> f16
      // CHECK: %{{.+}} = OpExtInst %[[F32T]] %{{.+}} native_exp %[[ZERO_F32]]
      %exp_f32 = llvm.call @_Z22__spirv_ocl_native_expf(%c0_f32) : (f32) -> f32
      // CHECK: %{{.+}} = OpExtInst %[[F64T]] %{{.+}} native_exp %[[ZERO_F64]]
      %exp_f64 = llvm.call @_Z22__spirv_ocl_native_expd(%c0_f64) : (f64) -> f64

      // CHECK: %{{.+}} = OpExtInst %[[V2F64T]] %{{.+}} native_exp %[[V2_ZERO_F64]]
      %exp_v2_f64 = llvm.call @_Z22__spirv_ocl_native_expDv2_f64(%v2_c0_f64) : (vector<2xf64>) -> vector<2xf64>
      // CHECK: %{{.+}} = OpExtInst %[[V3F32T]] %{{.+}} native_exp %[[V3_ZERO_F32]]
      %exp_v3_f32 = llvm.call @_Z22__spirv_ocl_native_expDv3_f32(%v3_c0_f32) : (vector<3xf32>) -> vector<3xf32>
      // CHECK: %{{.+}} = OpExtInst %[[V4F64T]] %{{.+}} native_exp %[[V4_ZERO_F64]]
      %exp_v4_f64 = llvm.call @_Z22__spirv_ocl_native_expDv4_f64(%v4_c0_f64) : (vector<4xf64>) -> vector<4xf64>
      // CHECK: %{{.+}} = OpExtInst %[[V8F64T]] %{{.+}} native_exp %[[V8_ZERO_F64]]
      %exp_v8_f64 = llvm.call @_Z22__spirv_ocl_native_expDv8_f64(%v8_c0_f64) : (vector<8xf64>) -> vector<8xf64>
      // CHECK: %{{.+}} = OpExtInst %[[V16F16T]] %{{.+}} native_exp %[[V16_ZERO_F16]]
      %exp_v16_f16 = llvm.call @_Z22__spirv_ocl_native_expDv16_f16(%v16_c0_f16) : (vector<16xf16>) -> vector<16xf16>

      llvm.return
    }
    llvm.func @_Z22__spirv_ocl_native_expDh(f16) -> f16
    llvm.func @_Z22__spirv_ocl_native_expf(f32) -> f32
    llvm.func @_Z22__spirv_ocl_native_expd(f64) -> f64
    llvm.func @_Z22__spirv_ocl_native_expDv2_f64(vector<2xf64>) -> vector<2xf64>
    llvm.func @_Z22__spirv_ocl_native_expDv3_f32(vector<3xf32>) -> vector<3xf32>
    llvm.func @_Z22__spirv_ocl_native_expDv4_f64(vector<4xf64>) -> vector<4xf64>
    llvm.func @_Z22__spirv_ocl_native_expDv8_f64(vector<8xf64>) -> vector<8xf64>
    llvm.func @_Z22__spirv_ocl_native_expDv16_f16(vector<16xf16>) -> vector<16xf16>


  }
}
