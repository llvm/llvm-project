; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t
; RUN: FileCheck < %t %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; Image types may be represented in two ways while translating to SPIR-V:
; - OpenCL form, for example, '%opencl.image2d_ro_t',
; - SPIR-V form, for example, '%spirv.Image._void_1_0_0_0_0_0_0',
; but it is still one type which should be translated to one SPIR-V type.
;
; The test checks that the code below is successfully translated and only one
; SPIR-V type for images is generated.

; CHECK:     10 TypeImage
; CHECK-NOT: 10 TypeImage

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

%opencl.image2d_ro_t = type opaque
%spirv.Image._void_1_0_0_0_0_0_0 = type opaque

; Function Attrs: convergent noinline nounwind optnone
define spir_kernel void @read_image(%opencl.image2d_ro_t addrspace(1)* %srcimg) #0 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_base_type !8 !kernel_arg_type_qual !9 {
entry:
  %srcimg.addr = alloca %opencl.image2d_ro_t addrspace(1)*, align 8
  %spirvimg.addr = alloca %spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)*, align 8
  store %opencl.image2d_ro_t addrspace(1)* %srcimg, %opencl.image2d_ro_t addrspace(1)** %srcimg.addr, align 8
  ret void
}

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!3}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{}
!3 = !{!"cl_images"}
!4 = !{!"clang version 6.0.0"}
!5 = !{i32 1}
!6 = !{!"read_only"}
!7 = !{!"image2d_t"}
!8 = !{!"image2d_t"}
!9 = !{!""}
