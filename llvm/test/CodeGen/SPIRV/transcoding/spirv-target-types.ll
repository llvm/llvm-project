; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability Float16
; CHECK-DAG: OpCapability ImageBasic
; CHECK-DAG: OpCapability ImageReadWrite
; CHECK-DAG: OpCapability Pipes
; CHECK-DAG: OpCapability DeviceEnqueue

; CHECK-DAG: %[[#VOID:]] = OpTypeVoid
; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#HALF:]] = OpTypeFloat 16
; CHECK-DAG: %[[#FLOAT:]] = OpTypeFloat 32
; CHECK-DAG: %[[#PIPE_RD:]] = OpTypePipe ReadOnly
; CHECK-DAG: %[[#PIPE_WR:]] = OpTypePipe WriteOnly
; CHECK-DAG: %[[#IMG1D_RD:]] = OpTypeImage %[[#VOID]] 1D 0 0 0 0 Unknown ReadOnly
; CHECK-DAG: %[[#IMG2D_RD:]] = OpTypeImage %[[#INT]] 2D 0 0 0 0
; CHECK-DAG: %[[#IMG3D_RD:]] = OpTypeImage %[[#INT]] 3D 0 0 0 0
; CHECK-DAG: %[[#IMG2DA_RD:]] = OpTypeImage %[[#HALF]] 2D 0 1 0 0
; CHECK-DAG: %[[#IMG2DD_RD:]] = OpTypeImage %[[#FLOAT]] Buffer 0 0 0
; CHECK-DAG: %[[#IMG1D_WR:]] = OpTypeImage %[[#VOID]] 1D 0 0 0 0 Unknown WriteOnly
; CHECK-DAG: %[[#IMG2D_RW:]] = OpTypeImage %[[#VOID]] 2D 0 0 0 0 Unknown ReadWrite
; CHECK-DAG: %[[#IMG1DB_RD:]] = OpTypeImage %[[#FLOAT]] 2D 1 0 0 0

; CHECK-DAG: %[[#DEVEVENT:]] = OpTypeDeviceEvent
; CHECK-DAG: %[[#EVENT:]] = OpTypeEvent
; CHECK-DAG: %[[#QUEUE:]] = OpTypeQueue
; CHECK-DAG: %[[#RESID:]] = OpTypeReserveId
; CHECK-DAG: %[[#SAMP:]] = OpTypeSampler
; CHECK-DAG: %[[#SAMPIMG:]] = OpTypeSampledImage %[[#IMG1DB_RD]]

; CHECK-DAG: %[[#]] = OpFunction %[[#VOID]]
; CHECK-DAG: %[[#]] = OpFunctionParameter %[[#PIPE_RD]]
; CHECK-DAG: %[[#]] = OpFunctionParameter %[[#PIPE_WR]]
; CHECK-DAG: %[[#]] = OpFunctionParameter %[[#IMG1D_RD]]
; CHECK-DAG: %[[#]] = OpFunctionParameter %[[#IMG2D_RD]]
; CHECK-DAG: %[[#]] = OpFunctionParameter %[[#IMG3D_RD]]
; CHECK-DAG: %[[#]] = OpFunctionParameter %[[#IMG2DA_RD]]
; CHECK-DAG: %[[#]] = OpFunctionParameter %[[#IMG2DD_RD]]
; CHECK-DAG: %[[#]] = OpFunctionParameter %[[#IMG1D_WR]]
; CHECK-DAG: %[[#]] = OpFunctionParameter %[[#IMG2D_RW]]

define spir_kernel void @foo(
  target("spirv.Pipe", 0) %a,
  target("spirv.Pipe", 1) %b,
  target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %c1,
  target("spirv.Image", i32, 1, 0, 0, 0, 0, 0, 0) %d1,
  target("spirv.Image", i32, 2, 0, 0, 0, 0, 0, 0) %e1,
  target("spirv.Image", half, 1, 0, 1, 0, 0, 0, 0) %f1,
  target("spirv.Image", float, 5, 0, 0, 0, 0, 0, 0) %g1,
  target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 1) %c2,
  target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 2) %d3) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  ret void
}

; CHECK-DAG: %[[#]] = OpFunction
; CHECK-DAG: %[[#]] = OpFunctionParameter %[[#DEVEVENT]]
; CHECK-DAG: %[[#]] = OpFunctionParameter %[[#EVENT]]
; CHECK-DAG: %[[#]] = OpFunctionParameter %[[#QUEUE]]
; CHECK-DAG: %[[#]] = OpFunctionParameter %[[#RESID]]

; CHECK-DAG: %[[#IMARG:]] = OpFunctionParameter %[[#IMG1DB_RD]]
; CHECK-DAG: %[[#SAMARG:]] = OpFunctionParameter %[[#SAMP]]
; CHECK-DAG: %[[#SAMPIMVAR:]] = OpSampledImage %[[#SAMPIMG]] %[[#IMARG]] %[[#SAMARG]]
; CHECK-DAG: %[[#]] = OpImageSampleExplicitLod %[[#]] %[[#SAMPIMVAR]]

define spir_func void @bar(
  target("spirv.DeviceEvent") %a,
  target("spirv.Event") %b,
  target("spirv.Queue") %c,
  target("spirv.ReserveId") %d) {
  ret void
}

define spir_func void @test_sampler(target("spirv.Image", float, 1, 1, 0, 0, 0, 0, 0) %srcimg.coerce,
                                    target("spirv.Sampler") %s.coerce) {
  %1 = tail call spir_func target("spirv.SampledImage", float, 1, 1, 0, 0, 0, 0, 0) @_Z20__spirv_SampledImagePU3AS1K34__spirv_Image__float_1_1_0_0_0_0_0PU3AS1K15__spirv_Sampler(target("spirv.Image", float, 1, 1, 0, 0, 0, 0, 0) %srcimg.coerce, target("spirv.Sampler") %s.coerce) #1
  %2 = tail call spir_func <4 x float> @_Z38__spirv_ImageSampleExplicitLod_Rfloat4PU3AS120__spirv_SampledImageDv4_iif(target("spirv.SampledImage", float, 1, 1, 0, 0, 0, 0, 0) %1, <4 x i32> zeroinitializer, i32 2, float 1.000000e+00) #1
  ret void
}

declare spir_func target("spirv.SampledImage", float, 1, 1, 0, 0, 0, 0, 0) @_Z20__spirv_SampledImagePU3AS1K34__spirv_Image__float_1_1_0_0_0_0_0PU3AS1K15__spirv_Sampler(target("spirv.Image", float, 1, 1, 0, 0, 0, 0, 0), target("spirv.Sampler"))

declare spir_func <4 x float> @_Z38__spirv_ImageSampleExplicitLod_Rfloat4PU3AS120__spirv_SampledImageDv4_iif(target("spirv.SampledImage", float, 1, 1, 0, 0, 0, 0, 0), <4 x i32>, i32, float)

attributes #0 = { nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!9}
!opencl.compiler.options = !{!8}

!1 = !{i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1}
!2 = !{!"read_only", !"write_only", !"read_only", !"read_only", !"read_only", !"read_only", !"read_only", !"write_only", !"read_write"}
!3 = !{!"int", !"int", !"image1d_t", !"image2d_t", !"image3d_t", !"image2d_array_t", !"image1d_buffer_t", !"image1d_t", !"image2d_t"}
!4 = !{!"int", !"int", !"image1d_t", !"image2d_t", !"image3d_t", !"image2d_array_t", !"image1d_buffer_t", !"image1d_t", !"image2d_t"}
!5 = !{!"pipe", !"pipe", !"", !"", !"", !"", !"", !"", !""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{!"cl_khr_fp16"}
!9 = !{!"cl_images"}
