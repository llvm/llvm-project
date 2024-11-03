; RUN: llc -O0 -opaque-pointers=0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK: %[[#ExtInstSetId:]] = OpExtInstImport "OpenCL.std"
; CHECK: %[[#Float:]] = OpTypeFloat 32
; CHECK: %[[#Double:]] = OpTypeFloat 64
; CHECK: %[[#Double4:]] = OpTypeVector %[[#Double]] 4
; CHECK: %[[#FloatArg:]] = OpConstant %[[#Float]]
; CHECK: %[[#DoubleArg:]] = OpConstant %[[#Double]]
; CHECK: %[[#Double4Arg:]] = OpConstantComposite %[[#Double4]]

;; We need to store sqrt results, otherwise isel does not emit sqrts as dead insts.
define spir_func void @test_sqrt(float* %x, double* %y, <4 x double>* %z) {
entry:
  %0 = call float @llvm.sqrt.f32(float 0x40091EB860000000)
  store float %0, float* %x
  %1 = call double @llvm.sqrt.f64(double 2.710000e+00)
  store double %1, double* %y
  %2 = call <4 x double> @llvm.sqrt.v4f64(<4 x double> <double 5.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01>)
  store <4 x double> %2, <4 x double>* %z
; CHECK: %[[#]] = OpExtInst %[[#Float]] %[[#ExtInstSetId]] sqrt %[[#FloatArg]]
; CHECK: %[[#]] = OpExtInst %[[#Double]] %[[#ExtInstSetId]] sqrt %[[#DoubleArg]]
; CHECK: %[[#]] = OpExtInst %[[#Double4]] %[[#ExtInstSetId]] sqrt %[[#Double4Arg]]
  ret void
}

declare float @llvm.sqrt.f32(float)

declare double @llvm.sqrt.f64(double)

declare <4 x double> @llvm.sqrt.v4f64(<4 x double>)
