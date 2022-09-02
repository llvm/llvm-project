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

%spirv.Pipe._0 = type opaque ; read_only pipe
%spirv.Pipe._1 = type opaque ; write_only pipe
%spirv.Image._void_0_0_0_0_0_0_0 = type opaque ; read_only image1d_ro_t
%spirv.Image._int_1_0_0_0_0_0_0 = type opaque ; read_only image2d_ro_t
%spirv.Image._uint_2_0_0_0_0_0_0 = type opaque ; read_only image3d_ro_t
%spirv.Image._float_1_1_0_0_0_0_0 = type opaque; read_only image2d_depth_ro_t
%spirv.Image._half_1_0_1_0_0_0_0 = type opaque ; read_only image2d_array_ro_t
%spirv.Image._float_5_0_0_0_0_0_0 = type opaque ; read_only image1d_buffer_ro_t
%spirv.Image._void_0_0_0_0_0_0_1 = type opaque ; write_only image1d_wo_t
%spirv.Image._void_1_0_0_0_0_0_2 = type opaque ; read_write image2d_rw_t
%spirv.DeviceEvent          = type opaque ; clk_event_t
%spirv.Event                = type opaque ; event_t
%spirv.Queue                = type opaque ; queue_t
%spirv.ReserveId            = type opaque ; reserve_id_t
%spirv.Sampler              = type opaque ; sampler_t
%spirv.SampledImage._float_1_1_0_0_0_0_0 = type opaque

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
  %spirv.Pipe._0 addrspace(1)* nocapture %a,
  %spirv.Pipe._1 addrspace(1)* nocapture %b,
  %spirv.Image._void_0_0_0_0_0_0_0 addrspace(1)* nocapture %c1,
  %spirv.Image._int_1_0_0_0_0_0_0 addrspace(1)* nocapture %d1,
  %spirv.Image._uint_2_0_0_0_0_0_0 addrspace(1)* nocapture %e1,
  %spirv.Image._half_1_0_1_0_0_0_0 addrspace(1)* nocapture %f1,
  %spirv.Image._float_5_0_0_0_0_0_0 addrspace(1)* nocapture %g1,
  %spirv.Image._void_0_0_0_0_0_0_1 addrspace(1)* nocapture %c2,
  %spirv.Image._void_1_0_0_0_0_0_2 addrspace(1)* nocapture %d3) {
entry:
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#DEVEVENT]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#EVENT]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#QUEUE]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#RESID]]

define spir_func void @bar(
  %spirv.DeviceEvent * %a,
  %spirv.Event * %b,
  %spirv.Queue * %c,
  %spirv.ReserveId * %d) {
  ret void
}

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#IMG_ARG:]] = OpFunctionParameter %[[#IMG2DD_RD]]
; CHECK-SPIRV: %[[#SAMP_ARG:]] = OpFunctionParameter %[[#SAMP]]
; CHECK-SPIRV: %[[#SAMPIMG_VAR:]] = OpSampledImage %[[#SAMPIMG]] %[[#IMG_ARG]] %[[#SAMP_ARG]]
; CHECK-SPIRV: %[[#]] = OpImageSampleExplicitLod %[[#]] %[[#SAMPIMG_VAR]]

define spir_func void @test_sampler(%spirv.Image._float_1_1_0_0_0_0_0 addrspace(1)* %srcimg.coerce,
                                    %spirv.Sampler addrspace(1)* %s.coerce) {
  %1 = tail call spir_func %spirv.SampledImage._float_1_1_0_0_0_0_0 addrspace(1)* @_Z20__spirv_SampledImagePU3AS1K34__spirv_Image__float_1_1_0_0_0_0_0PU3AS1K15__spirv_Sampler(%spirv.Image._float_1_1_0_0_0_0_0 addrspace(1)* %srcimg.coerce, %spirv.Sampler addrspace(1)* %s.coerce)
  %2 = tail call spir_func <4 x float> @_Z38__spirv_ImageSampleExplicitLod_Rfloat4PU3AS120__spirv_SampledImageDv4_iif(%spirv.SampledImage._float_1_1_0_0_0_0_0 addrspace(1)* %1, <4 x i32> zeroinitializer, i32 2, float 1.000000e+00)
  ret void
}

declare spir_func %spirv.SampledImage._float_1_1_0_0_0_0_0 addrspace(1)* @_Z20__spirv_SampledImagePU3AS1K34__spirv_Image__float_1_1_0_0_0_0_0PU3AS1K15__spirv_Sampler(%spirv.Image._float_1_1_0_0_0_0_0 addrspace(1)*, %spirv.Sampler addrspace(1)*)

declare spir_func <4 x float> @_Z38__spirv_ImageSampleExplicitLod_Rfloat4PU3AS120__spirv_SampledImageDv4_iif(%spirv.SampledImage._float_1_1_0_0_0_0_0 addrspace(1)*, <4 x i32>, i32, float)
