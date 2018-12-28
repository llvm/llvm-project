; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: TypeInt [[IntTyID:[0-9]+]]
; CHECK-SPIRV: TypeVoid [[VoidTyID:[0-9]+]]
; CHECK-SPIRV: TypeImage [[ImageTyID:[0-9]+]] [[VoidTyID]] 1 0 1 0 0 0 0
; CHECK-SPIRV: TypeVector [[VectorTyID:[0-9]+]] [[IntTyID]] {{[0-9]+}}
; CHECK-SPIRV: FunctionParameter [[ImageTyID]] [[ImageArgID:[0-9]+]]
; CHECK-SPIRV: ImageQuerySizeLod [[VectorTyID]] {{[0-9]+}} [[ImageArgID]]

; CHECK-LLVM: call spir_func <2 x i32> @_Z13get_image_dim16ocl_image2darray(%opencl.image2d_array_t addrspace(1)*

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

%opencl.image2d_array_ro_t = type opaque

; Function Attrs: nounwind
define spir_kernel void @sample_kernel(%opencl.image2d_array_ro_t addrspace(1)* %input) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %call = call spir_func i32 @_Z15get_image_width20ocl_image2d_array_ro(%opencl.image2d_array_ro_t addrspace(1)* %input) #2
  ret void
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z15get_image_width20ocl_image2d_array_ro(%opencl.image2d_array_ro_t addrspace(1)*) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!6}
!opencl.used.extensions = !{!7}
!opencl.used.optional.core.features = !{!7}
!opencl.compiler.options = !{!7}
!llvm.ident = !{!8}

!1 = !{i32 1}
!2 = !{!"read_only"}
!3 = !{!"__read_only image2d_array_t"}
!4 = !{!"__read_only image2d_array_t"}
!5 = !{!""}
!6 = !{i32 1, i32 2}
!7 = !{}
!8 = !{!"clang version 3.6.1 "}
