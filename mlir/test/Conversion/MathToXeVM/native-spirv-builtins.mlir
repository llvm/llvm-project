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

      // CHECK: OpExtInst %[[F16T]] %{{.+}} native_exp %[[ZERO_F16]]
      %exp_f16 = llvm.call @_Z22__spirv_ocl_native_expDh(%c0_f16) : (f16) -> f16
      // CHECK: OpExtInst %[[F32T]] %{{.+}} native_exp %[[ZERO_F32]]
      %exp_f32 = llvm.call @_Z22__spirv_ocl_native_expf(%c0_f32) : (f32) -> f32
      // CHECK: OpExtInst %[[F64T]] %{{.+}} native_exp %[[ZERO_F64]]
      %exp_f64 = llvm.call @_Z22__spirv_ocl_native_expd(%c0_f64) : (f64) -> f64

      // CHECK: OpExtInst %[[V2F64T]] %{{.+}} native_exp %[[V2_ZERO_F64]]
      %exp_v2_f64 = llvm.call @_Z22__spirv_ocl_native_expDv2_f64(%v2_c0_f64) : (vector<2xf64>) -> vector<2xf64>
      // CHECK: OpExtInst %[[V3F32T]] %{{.+}} native_exp %[[V3_ZERO_F32]]
      %exp_v3_f32 = llvm.call @_Z22__spirv_ocl_native_expDv3_f32(%v3_c0_f32) : (vector<3xf32>) -> vector<3xf32>
      // CHECK: OpExtInst %[[V4F64T]] %{{.+}} native_exp %[[V4_ZERO_F64]]
      %exp_v4_f64 = llvm.call @_Z22__spirv_ocl_native_expDv4_f64(%v4_c0_f64) : (vector<4xf64>) -> vector<4xf64>
      // CHECK: OpExtInst %[[V8F64T]] %{{.+}} native_exp %[[V8_ZERO_F64]]
      %exp_v8_f64 = llvm.call @_Z22__spirv_ocl_native_expDv8_f64(%v8_c0_f64) : (vector<8xf64>) -> vector<8xf64>
      // CHECK: OpExtInst %[[V16F16T]] %{{.+}} native_exp %[[V16_ZERO_F16]]
      %exp_v16_f16 = llvm.call @_Z22__spirv_ocl_native_expDv16_f16(%v16_c0_f16) : (vector<16xf16>) -> vector<16xf16>

      // SPIRV backend does not currently handle fastmath flags: The SPIRV
      // backend would need to generate OpDecorate calls to decorate math ops
      // with FPFastMathMode/FPFastMathModeINTEL decorations.
      //
      // FIXME: When support for fastmath flags in the SPIRV backend is added, 
      // add tests here to ensure fastmath flags are converted to the correct
      // OpDecorate calls.
      // 
      // See:
      // - https://registry.khronos.org/SPIR-V/specs/unified1/OpenCL.ExtendedInstructionSet.100.html#_math_extended_instructions
      // - https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate

      // CHECK: OpExtInst %[[F16T]] %{{.+}} native_cos %[[ZERO_F16]]
      %cos_afn_f16 = llvm.call @_Z22__spirv_ocl_native_cosDh(%c0_f16) {fastmathFlags = #llvm.fastmath<afn>} : (f16) -> f16
      // CHECK: OpExtInst %[[F32T]] %{{.+}} native_exp2 %[[ZERO_F32]]
      %exp2_afn_f32 = llvm.call @_Z23__spirv_ocl_native_exp2f(%c0_f32) {fastmathFlags = #llvm.fastmath<afn>} : (f32) -> f32
      // CHECK: OpExtInst %[[F16T]] %{{.+}} native_log %[[ZERO_F16]]
      %log_afn_f16 = llvm.call @_Z22__spirv_ocl_native_logDh(%c0_f16) {fastmathFlags = #llvm.fastmath<afn>} : (f16) -> f16
      // CHECK: OpExtInst %[[F32T]] %{{.+}} native_log2 %[[ZERO_F32]]
      %log2_afn_f32 = llvm.call @_Z23__spirv_ocl_native_log2f(%c0_f32) {fastmathFlags = #llvm.fastmath<afn>} : (f32) -> f32
      // CHECK: OpExtInst %[[V8F64T]] %{{.+}} native_log10 %[[V8_ZERO_F64]]
      %log10_afn_f64 = llvm.call @_Z24__spirv_ocl_native_log10Dv8_d(%v8_c0_f64) {fastmathFlags = #llvm.fastmath<afn>} : (vector<8xf64>) -> vector<8xf64>
      // CHECK: OpExtInst %[[V16F16T]] %{{.+}} native_powr %[[V16_ZERO_F16]] %[[V16_ZERO_F16]]
      %powr_afn_f16 = llvm.call @_Z23__spirv_ocl_native_powrDv16_DhS_(%v16_c0_f16, %v16_c0_f16) {fastmathFlags = #llvm.fastmath<afn>} : (vector<16xf16>, vector<16xf16>) -> vector<16xf16>
      // CHECK: OpExtInst %[[F64T]] %{{.+}} native_rsqrt %[[ZERO_F64]]
      %rsqrt_afn_f64 = llvm.call @_Z24__spirv_ocl_native_rsqrtd(%c0_f64) {fastmathFlags = #llvm.fastmath<afn>} : (f64) -> f64
      // CHECK: OpExtInst %[[F16T]] %{{.+}} native_sin %[[ZERO_F16]]
      %sin_afn_f16 = llvm.call @_Z22__spirv_ocl_native_sinDh(%c0_f16) {fastmathFlags = #llvm.fastmath<afn>} : (f16) -> f16
      // CHECK: OpExtInst %[[F32T]] %{{.+}} native_sqrt %[[ZERO_F32]]
      %sqrt_afn_f32 = llvm.call @_Z23__spirv_ocl_native_sqrtf(%c0_f32) {fastmathFlags = #llvm.fastmath<afn>} : (f32) -> f32
      // CHECK: OpExtInst %[[F64T]] %{{.+}} native_tan %[[ZERO_F64]]
      %tan_afn_f64 = llvm.call @_Z22__spirv_ocl_native_tand(%c0_f64) {fastmathFlags = #llvm.fastmath<afn>} : (f64) -> f64
      // CHECK: OpExtInst %[[F32T]] %{{.+}} native_divide %[[ZERO_F32]] %[[ZERO_F32]]
      %divide_afn_f32 = llvm.call @_Z25__spirv_ocl_native_divideff(%c0_f32, %c0_f32) {fastmathFlags = #llvm.fastmath<afn>} : (f32, f32) -> f32

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
    llvm.func @_Z22__spirv_ocl_native_cosDh(f16) -> f16
    llvm.func @_Z23__spirv_ocl_native_exp2f(f32) -> f32
    llvm.func @_Z22__spirv_ocl_native_logDh(f16) -> f16
    llvm.func @_Z23__spirv_ocl_native_log2f(f32) -> f32
    llvm.func @_Z24__spirv_ocl_native_log10Dv8_d(vector<8xf64>) -> vector<8xf64>
    llvm.func @_Z23__spirv_ocl_native_powrDv16_DhS_(vector<16xf16>, vector<16xf16>) -> vector<16xf16>
    llvm.func @_Z24__spirv_ocl_native_rsqrtd(f64) -> f64
    llvm.func @_Z22__spirv_ocl_native_sinDh(f16) -> f16
    llvm.func @_Z23__spirv_ocl_native_sqrtf(f32) -> f32
    llvm.func @_Z22__spirv_ocl_native_tand(f64) -> f64
    llvm.func @_Z25__spirv_ocl_native_divideff(f32, f32) -> f32
  }
}
