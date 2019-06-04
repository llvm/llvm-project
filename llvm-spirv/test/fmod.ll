; __kernel void fmod_kernel( float out, float in1, float in2 )
; { out = fmod( in1, in2 ); }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: 7 ExtInst {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} fmod {{[0-9]+}} {{[0-9]+}}
; CHECK-LLVM: call spir_func float @_Z4fmodff
; CHECK-LLVM: declare spir_func float @_Z4fmodff(float, float)

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @fmod_kernel(float %out, float %in1, float %in2) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %call = call spir_func float @_Z4fmodff(float %in1, float %in2) #2
  ret void
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4fmodff(float, float) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!8}
!llvm.ident = !{!9}

!1 = !{i32 0, i32 0, i32 0}
!2 = !{!"none", !"none", !"none"}
!3 = !{!"float", !"float", !"float"}
!4 = !{!"float", !"float", !"float"}
!5 = !{!"", !"", !""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{}
!9 = !{!"clang version 3.6.1"}
