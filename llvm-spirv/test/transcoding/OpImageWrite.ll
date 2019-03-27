;;OpImageWrite.cl
;;void sample_kernel_write(float4 input, write_only image2d_t output, int2 coord)
;;{
;;   write_imagef( output, coord , 5, input);
;;   write_imagef( output, coord , input);
;;}
;;clang -cc1 -O0 -triple spir-unknown-unknown -cl-std=CL2.0 -x cl OpImageWrite.cl -include opencl-20.h -emit-llvm -o - | opt -mem2reg -S > OpImageWrite.ll

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Constant {{[0-9]+}} [[lod:[0-9]+]] 5
; CHECK-SPIRV: ImageWrite [[image:[0-9]+]] [[coord:[0-9]+]] [[texel:[0-9]+]] 2 [[lod]]
; CHECK-SPIRV: ImageWrite [[image]] [[coord]] [[texel]]


target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%opencl.image2d_wo_t = type opaque

; Function Attrs: nounwind
define spir_func void @sample_kernel_write(<4 x float> %input, %opencl.image2d_wo_t addrspace(1)* %output, <2 x i32> %coord) #0 {
entry:
  call spir_func void @_Z12write_imagef14ocl_image2d_woDv2_iiDv4_f(%opencl.image2d_wo_t addrspace(1)* %output, <2 x i32> %coord, i32 5, <4 x float> %input)
; CHECK-LLVM: call spir_func void @_Z12write_imagef14ocl_image2d_woDv2_iiDv4_f(%opencl.image2d_wo_t addrspace(1)* %output, <2 x i32> %coord, i32 5, <4 x float> %input)

  call spir_func void @_Z12write_imagef14ocl_image2d_woDv2_iDv4_f(%opencl.image2d_wo_t addrspace(1)* %output, <2 x i32> %coord, <4 x float> %input)
; CHECK-LLVM: call spir_func void @_Z12write_imagef14ocl_image2d_woDv2_iDv4_f(%opencl.image2d_wo_t addrspace(1)* %output, <2 x i32> %coord, <4 x float> %input)

  ret void
}

declare spir_func void @_Z12write_imagef14ocl_image2d_woDv2_iiDv4_f(%opencl.image2d_wo_t addrspace(1)*, <2 x i32>, i32, <4 x float>) #1

declare spir_func void @_Z12write_imagef14ocl_image2d_woDv2_iDv4_f(%opencl.image2d_wo_t addrspace(1)*, <2 x i32>, <4 x float>) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!3}
!opencl.compiler.options = !{!2}

!0 = !{i32 1, i32 2}
!1 = !{i32 2, i32 0}
!2 = !{}
!3 = !{!"cl_images"}

