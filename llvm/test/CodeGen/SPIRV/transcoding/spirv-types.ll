;; Test SPIR-V opaque types

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: OpCapability Float16
; CHECK-SPIRV-DAG: OpCapability ImageReadWrite
; CHECK-SPIRV-DAG: OpCapability Pipes
; CHECK-SPIRV-DAG: OpCapability DeviceEnqueue

; CHECK-SPIRV-DAG: %[[#VOID:]] = OpTypeVoid
; CHECK-SPIRV-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#HALF:]] = OpTypeFloat 16
; CHECK-SPIRV-DAG: %[[#FLOAT:]] = OpTypeFloat 32
; CHECK-SPIRV-DAG: %[[#PIPE_RD:]] = OpTypePipe ReadOnly
; CHECK-SPIRV-DAG: %[[#PIPE_WR:]] = OpTypePipe WriteOnly
; CHECK-SPIRV-DAG: %[[#IMG1D_RD:]] = OpTypeImage %[[#VOID]] 1D 0 0 0 0 Unknown ReadOnly
; CHECK-SPIRV-DAG: %[[#IMG2D_RD:]] = OpTypeImage %[[#INT]] 2D 0 0 0 0 Unknown ReadOnly
; CHECK-SPIRV-DAG: %[[#IMG3D_RD:]] = OpTypeImage %[[#INT]] 3D 0 0 0 0 Unknown ReadOnly
; CHECK-SPIRV-DAG: %[[#IMG2DD_RD:]] = OpTypeImage %[[#FLOAT]] 2D 1 0 0 0 Unknown ReadOnly
; CHECK-SPIRV-DAG: %[[#IMG2DA_RD:]] = OpTypeImage %[[#HALF]] 2D 0 1 0 0 Unknown ReadOnly
; CHECK-SPIRV-DAG: %[[#IMG1DB_RD:]] = OpTypeImage %[[#FLOAT]] Buffer 0 0 0 0 Unknown ReadOnly
; CHECK-SPIRV-DAG: %[[#IMG1D_WR:]] = OpTypeImage %[[#VOID]] 1D 0 0 0 0 Unknown WriteOnly
; CHECK-SPIRV-DAG: %[[#IMG2D_RW:]] = OpTypeImage %[[#VOID]] 2D 0 0 0 0 Unknown ReadWrite
; CHECK-SPIRV-DAG: %[[#DEVEVENT:]] = OpTypeDeviceEvent
; CHECK-SPIRV-DAG: %[[#EVENT:]] = OpTypeEvent
; CHECK-SPIRV-DAG: %[[#QUEUE:]] = OpTypeQueue
; CHECK-SPIRV-DAG: %[[#RESID:]] = OpTypeReserveId
; CHECK-SPIRV-DAG: %[[#SAMP:]] = OpTypeSampler
; CHECK-SPIRV-DAG: %[[#SAMPIMG:]] = OpTypeSampledImage %[[#IMG2DD_RD]]

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#PIPE_RD]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#PIPE_WR]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#IMG1D_RD]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#IMG2D_RD]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#IMG3D_RD]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#IMG2DA_RD]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#IMG1DB_RD]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#IMG1D_WR]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#IMG2D_RW]]

define spir_kernel void @foo(
  target("spirv.Pipe", 0) %a,
  target("spirv.Pipe", 1) %b,
  target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %c1,
  target("spirv.Image", i32, 1, 0, 0, 0, 0, 0, 0) %d1,
  target("spirv.Image", i32, 2, 0, 0, 0, 0, 0, 0) %e1,
  target("spirv.Image", half, 1, 0, 1, 0, 0, 0, 0) %f1,
  target("spirv.Image", float, 5, 0, 0, 0, 0, 0, 0) %g1,
  target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 1) %c2,
  target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 2) %d3) {
entry:
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#DEVEVENT]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#EVENT]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#QUEUE]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#RESID]]

define spir_func void @bar(
  target("spirv.DeviceEvent") %a,
  target("spirv.Event") %b,
  target("spirv.Queue") %c,
  target("spirv.ReserveId") %d) {
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#IMG_ARG:]] = OpFunctionParameter %[[#IMG2DD_RD]]
; CHECK-SPIRV: %[[#SAMP_ARG:]] = OpFunctionParameter %[[#SAMP]]
; CHECK-SPIRV: %[[#SAMPIMG_VAR:]] = OpSampledImage %[[#SAMPIMG]] %[[#IMG_ARG]] %[[#SAMP_ARG]]
; CHECK-SPIRV: %[[#]] = OpImageSampleExplicitLod %[[#]] %[[#SAMPIMG_VAR]]

define spir_func void @test_sampler(target("spirv.Image", float, 1, 1, 0, 0, 0, 0, 0) %srcimg.coerce,
                                    target("spirv.Sampler") %s.coerce) {
  %1 = tail call spir_func target("spirv.Image", float, 1, 1, 0, 0, 0, 0, 0) @_Z20__spirv_SampledImagePU3AS1K34__spirv_Image__float_1_1_0_0_0_0_0PU3AS1K15__spirv_Sampler(target("spirv.Image", float, 1, 1, 0, 0, 0, 0, 0) %srcimg.coerce, target("spirv.Sampler") %s.coerce)
  %2 = tail call spir_func <4 x float> @_Z38__spirv_ImageSampleExplicitLod_Rfloat4PU3AS120__spirv_SampledImageDv4_iif(target("spirv.Image", float, 1, 1, 0, 0, 0, 0, 0) %1, <4 x i32> zeroinitializer, i32 2, float 1.000000e+00)
  ret void
}

declare spir_func target("spirv.Image", float, 1, 1, 0, 0, 0, 0, 0) @_Z20__spirv_SampledImagePU3AS1K34__spirv_Image__float_1_1_0_0_0_0_0PU3AS1K15__spirv_Sampler(target("spirv.Image", float, 1, 1, 0, 0, 0, 0, 0), target("spirv.Sampler"))

declare spir_func <4 x float> @_Z38__spirv_ImageSampleExplicitLod_Rfloat4PU3AS120__spirv_SampledImageDv4_iif(target("spirv.Image", float, 1, 1, 0, 0, 0, 0, 0), <4 x i32>, i32, float)
