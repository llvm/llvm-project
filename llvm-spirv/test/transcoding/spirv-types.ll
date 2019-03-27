;; Test SPIR-V opaque types
;;
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spv.txt
; RUN: FileCheck < %t.spv.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.from-llvm.spv
; RUN: llvm-spirv -to-binary %t.spv.txt -o %t.from-text.spv
; RUN: cmp %t.from-llvm.spv %t.from-text.spv
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: 2 Capability Float16
; CHECK-SPIRV: 2 Capability ImageBasic
; CHECK-SPIRV: 2 Capability ImageReadWrite
; CHECK-SPIRV: 2 Capability Pipes
; CHECK-SPIRV: 2 Capability DeviceEnqueue

; CHECK-SPIRV-DAG: 2 TypeVoid [[VOID:[0-9]+]]
; CHECK-SPIRV-DAG: 4 TypeInt [[INT:[0-9]+]] 32 0
; CHECK-SPIRV-DAG: 3 TypeFloat [[HALF:[0-9]+]] 16
; CHECK-SPIRV-DAG: 3 TypeFloat [[FLOAT:[0-9]+]] 32
; CHECK-SPIRV-DAG: 3 TypePipe [[PIPE_RD:[0-9]+]] 0
; CHECK-SPIRV-DAG: 3 TypePipe [[PIPE_WR:[0-9]+]] 1
; CHECK-SPIRV-DAG: 10 TypeImage [[IMG1D_RD:[0-9]+]] [[VOID]] 0 0 0 0 0 0 0
; CHECK-SPIRV-DAG: 10 TypeImage [[IMG2D_RD:[0-9]+]] [[INT]] 1 0 0 0 0 0 0
; CHECK-SPIRV-DAG: 10 TypeImage [[IMG3D_RD:[0-9]+]] [[INT]] 2 0 0 0 0 0 0
; CHECK-SPIRV-DAG: 10 TypeImage [[IMG2DD_RD:[0-9]+]] [[FLOAT]] 1 1 0 0 0 0 0
; CHECK-SPIRV-DAG: 10 TypeImage [[IMG2DA_RD:[0-9]+]] [[HALF]] 1 0 1 0 0 0 0
; CHECK-SPIRV-DAG: 10 TypeImage [[IMG1DB_RD:[0-9]+]] [[FLOAT]] 5 0 0 0 0 0 0
; CHECK-SPIRV-DAG: 10 TypeImage [[IMG1D_WR:[0-9]+]] [[VOID]] 0 0 0 0 0 0 1
; CHECK-SPIRV-DAG: 10 TypeImage [[IMG2D_RW:[0-9]+]] [[VOID]] 1 0 0 0 0 0 2
; CHECK-SPIRV-DAG: 2 TypeDeviceEvent [[DEVEVENT:[0-9]+]]
; CHECK-SPIRV-DAG: 2 TypeEvent [[EVENT:[0-9]+]]
; CHECK-SPIRV-DAG: 2 TypeQueue [[QUEUE:[0-9]+]]
; CHECK-SPIRV-DAG: 2 TypeReserveId [[RESID:[0-9]+]]
; CHECK-SPIRV-DAG: 2 TypeSampler [[SAMP:[0-9]+]]
; CHECK-SPIRV-DAG: 3 TypeSampledImage [[SAMPIMG:[0-9]+]] [[IMG2DD_RD]]

; ModuleID = 'cl-types.cl'
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-LLVM-DAG: %opencl.pipe_ro_t = type opaque
; CHECK-LLVM-DAG: %opencl.pipe_wo_t = type opaque
; CHECK-LLVM-DAG: %opencl.image3d_ro_t = type opaque
; CHECK-LLVM_DAG: %opencl.image2d_depth_ro_t = type opaque
; CHECK-LLVM-DAG: %opencl.image2d_array_ro_t = type opaque
; CHECK-LLVM-DAG: %opencl.image1d_buffer_ro_t = type opaque
; CHECK-LLVM-DAG: %opencl.image1d_ro_t = type opaque
; CHECK-LLVM-DAG: %opencl.image1d_wo_t = type opaque
; CHECK-LLVM-DAG: %opencl.image2d_ro_t = type opaque
; CHECK-LLVM-DAG: %opencl.image2d_rw_t = type opaque
; CHECK-LLVM-DAG: %opencl.clk_event_t = type opaque
; CHECK-LLVM-DAG: %opencl.event_t = type opaque
; CHECK-LLVM-DAG: %opencl.queue_t = type opaque
; CHECK-LLVM-DAG: %opencl.reserve_id_t = type opaque

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

; CHECK-SPIRV: {{[0-9]+}} Function
; CHECK-SPIRV: 3 FunctionParameter [[PIPE_RD]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[PIPE_WR]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[IMG1D_RD]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[IMG2D_RD]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[IMG3D_RD]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[IMG2DA_RD]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[IMG1DB_RD]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[IMG1D_WR]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[IMG2D_RW]] {{[0-9]+}}

; CHECK-LLVM:        define spir_kernel void @foo(
; CHECK-LLVM-SAME:     %opencl.pipe_ro_t addrspace(1)* nocapture %a,
; CHECK-LLVM-SAME:     %opencl.pipe_wo_t addrspace(1)* nocapture %b,
; CHECK-LLVM-SAME:     %opencl.image1d_ro_t addrspace(1)* nocapture %c1,
; CHECK-LLVM-SAME:     %opencl.image2d_ro_t addrspace(1)* nocapture %d1,
; CHECK-LLVM-SAME:     %opencl.image3d_ro_t addrspace(1)* nocapture %e1,
; CHECK-LLVM-SAME:     %opencl.image2d_array_ro_t addrspace(1)* nocapture %f1,
; CHECK-LLVM-SAME:     %opencl.image1d_buffer_ro_t addrspace(1)* nocapture %g1,
; CHECK-LLVM-SAME:     %opencl.image1d_wo_t addrspace(1)* nocapture %c2,
; CHECK-LLVM-SAME:     %opencl.image2d_rw_t addrspace(1)* nocapture %d3)
; CHECK-LLVM-SAME:     !kernel_arg_addr_space [[AS:![0-9]+]]
; CHECK-LLVM-SAME:     !kernel_arg_access_qual [[AQ:![0-9]+]]
; CHECK-LLVM-SAME:     !kernel_arg_type [[TYPE:![0-9]+]]
; CHECK-LLVM-SAME:     !kernel_arg_type_qual [[TQ:![0-9]+]]
; CHECK-LLVM-SAME:     !kernel_arg_base_type [[BT:![0-9]+]]

; Function Attrs: nounwind readnone
define spir_kernel void @foo(
  %spirv.Pipe._0 addrspace(1)* nocapture %a,
  %spirv.Pipe._1 addrspace(1)* nocapture %b,
  %spirv.Image._void_0_0_0_0_0_0_0 addrspace(1)* nocapture %c1,
  %spirv.Image._int_1_0_0_0_0_0_0 addrspace(1)* nocapture %d1,
  %spirv.Image._uint_2_0_0_0_0_0_0 addrspace(1)* nocapture %e1,
  %spirv.Image._half_1_0_1_0_0_0_0 addrspace(1)* nocapture %f1,
  %spirv.Image._float_5_0_0_0_0_0_0 addrspace(1)* nocapture %g1,
  %spirv.Image._void_0_0_0_0_0_0_1 addrspace(1)* nocapture %c2,
  %spirv.Image._void_1_0_0_0_0_0_2 addrspace(1)* nocapture %d3) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  ret void
}

; CHECK-SPIRV: {{[0-9]+}} Function
; CHECK-SPIRV: 3 FunctionParameter [[DEVEVENT]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[EVENT]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[QUEUE]] {{[0-9]+}}
; CHECK-SPIRV: 3 FunctionParameter [[RESID]] {{[0-9]+}}

; CHECK-LLVM: define spir_func void @bar(
; CHECK-LLVM:  %opencl.clk_event_t* %a,
; CHECK-LLVM:  %opencl.event_t* %b,
; CHECK-LLVM:  %opencl.queue_t* %c,
; CHECK-LLVM:  %opencl.reserve_id_t* %d)

define spir_func void @bar(
  %spirv.DeviceEvent * %a,
  %spirv.Event * %b,
  %spirv.Queue * %c,
  %spirv.ReserveId * %d) {
  ret void
}

; CHECK-SPIRV: {{[0-9]+}} Function
; CHECK-SPIRV: 3 FunctionParameter [[IMG2DD_RD]] [[IMG_ARG:[0-9]+]]
; CHECK-SPIRV: 3 FunctionParameter [[SAMP]] [[SAMP_ARG:[0-9]+]]
; CHECK-SPIRV: 5 SampledImage [[SAMPIMG]] [[SAMPIMG_VAR:[0-9]+]] [[IMG_ARG]] [[SAMP_ARG]]
; CHECK-SPIRV: 7 ImageSampleExplicitLod {{[0-9]+}} {{[0-9]+}} [[SAMPIMG_VAR]]

; CHECK-LLVM: define spir_func void @test_sampler(
; CHECK-LLVM:  %opencl.image2d_depth_ro_t addrspace(1)* %srcimg.coerce,
; CHECK-LLVM:  %opencl.sampler_t* %s.coerce)
; CHECK-LLVM:  call spir_func float @_Z11read_imagef20ocl_image2d_depth_ro11ocl_samplerDv4_if(%opencl.image2d_depth_ro_t addrspace(1)* %srcimg.coerce, %opencl.sampler_t* %s.coerce, <4 x i32> zeroinitializer, float 1.000000e+00)

define spir_func void @test_sampler(%spirv.Image._float_1_1_0_0_0_0_0 addrspace(1)* %srcimg.coerce,
                                    %spirv.Sampler addrspace(1)* %s.coerce) {
  %1 = tail call spir_func %spirv.SampledImage._float_1_1_0_0_0_0_0 addrspace(1)* @_Z20__spirv_SampledImagePU3AS1K34__spirv_Image__float_1_1_0_0_0_0_0PU3AS1K15__spirv_Sampler(%spirv.Image._float_1_1_0_0_0_0_0 addrspace(1)* %srcimg.coerce, %spirv.Sampler addrspace(1)* %s.coerce) #1
  %2 = tail call spir_func <4 x float> @_Z38__spirv_ImageSampleExplicitLod_Rfloat4PU3AS120__spirv_SampledImageDv4_iif(%spirv.SampledImage._float_1_1_0_0_0_0_0 addrspace(1)* %1, <4 x i32> zeroinitializer, i32 2, float 1.000000e+00) #1
  ret void
}

declare spir_func %spirv.SampledImage._float_1_1_0_0_0_0_0 addrspace(1)* @_Z20__spirv_SampledImagePU3AS1K34__spirv_Image__float_1_1_0_0_0_0_0PU3AS1K15__spirv_Sampler(%spirv.Image._float_1_1_0_0_0_0_0 addrspace(1)*, %spirv.Sampler addrspace(1)*)

declare spir_func <4 x float> @_Z38__spirv_ImageSampleExplicitLod_Rfloat4PU3AS120__spirv_SampledImageDv4_iif(%spirv.SampledImage._float_1_1_0_0_0_0_0 addrspace(1)*, <4 x i32>, i32, float)

attributes #0 = { nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!9}
!opencl.compiler.options = !{!8}

; CHECK-LLVM-DAG: [[AS]] = !{i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1}
; CHECK-LLVM-DAG: [[AQ]] = !{!"read_only", !"write_only", !"read_only", !"read_only", !"read_only", !"read_only", !"read_only", !"write_only", !"read_write"}
; CHECK-LLVM-DAG: [[TYPE]] = !{!"int", !"int", !"image1d_t", !"image2d_t", !"image3d_t", !"image2d_array_t", !"image1d_buffer_t", !"image1d_t", !"image2d_t"}
; CHECK-LLVM-DAG: [[BT]] = !{!"pipe", !"pipe", !"image1d_t", !"image2d_t", !"image3d_t", !"image2d_array_t", !"image1d_buffer_t", !"image1d_t", !"image2d_t"}
; CHECK-LLVM-DAG: [[TQ]] = !{!"pipe", !"pipe", !"", !"", !"", !"", !"", !"", !""}

!1 = !{i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1}
!2 = !{!"read_only", !"write_only", !"read_only", !"read_only", !"read_only", !"read_only", !"read_only", !"write_only", !"read_write"}
!3 = !{!"int", !"int", !"image1d_t", !"image2d_t", !"image3d_t", !"image2d_array_t", !"image1d_buffer_t", !"image1d_t", !"image2d_t"}
!4 = !{!"int", !"int", !"image1d_t", !"image2d_t", !"image3d_t", !"image2d_array_t", !"image1d_buffer_t", !"image1d_t", !"image2d_t"}
!5 = !{!"pipe", !"pipe", !"", !"", !"", !"", !"", !"", !""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{!"cl_khr_fp16"}
!9 = !{!"cl_images"}
