; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t
; RUN: FileCheck < %t %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; check for magic number followed by version 1.0
; CHECK: 119734787 65536

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @worker() #0 {
entry:
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!8}
!llvm.ident = !{!9}
!spirv.Source = !{!10}

!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 2}
!8 = !{}
!9 = !{!"clang version 3.6.1 "}
!10 = !{i32 4, i32 202000}
