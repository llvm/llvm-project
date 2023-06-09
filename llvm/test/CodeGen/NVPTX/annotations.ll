; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

@texture = internal addrspace(1) global i64 0, align 8
; CHECK: .global .texref texture
@surface = internal addrspace(1) global i64 0, align 8
; CHECK: .global .surfref surface

; CHECK: .entry kernel_func_maxntid
define void @kernel_func_maxntid(ptr %a) {
; CHECK: .maxntid 10, 20, 30
; CHECK: ret
  ret void
}

; CHECK: .entry kernel_func_reqntid
define void @kernel_func_reqntid(ptr %a) {
; CHECK: .reqntid 11, 22, 33
; CHECK: ret
  ret void
}

; CHECK: .entry kernel_func_minctasm
define void @kernel_func_minctasm(ptr %a) {
; CHECK: .minnctapersm 42
; CHECK: ret
  ret void
}

; CHECK-LABEL: .entry kernel_func_maxnreg
define void @kernel_func_maxnreg() {
; CHECK: .maxnreg 1234
; CHECK: ret
  ret void
}

!nvvm.annotations = !{!1, !2, !3, !4, !5, !6, !7, !8, !9, !10}

!1 = !{ptr @kernel_func_maxntid, !"kernel", i32 1}
!2 = !{ptr @kernel_func_maxntid, !"maxntidx", i32 10, !"maxntidy", i32 20, !"maxntidz", i32 30}

!3 = !{ptr @kernel_func_reqntid, !"kernel", i32 1}
!4 = !{ptr @kernel_func_reqntid, !"reqntidx", i32 11, !"reqntidy", i32 22, !"reqntidz", i32 33}

!5 = !{ptr @kernel_func_minctasm, !"kernel", i32 1}
!6 = !{ptr @kernel_func_minctasm, !"minctasm", i32 42}

!7 = !{ptr @kernel_func_maxnreg, !"kernel", i32 1}
!8 = !{ptr @kernel_func_maxnreg, !"maxnreg", i32 1234}

!9 = !{ptr addrspace(1) @texture, !"texture", i32 1}
!10 = !{ptr addrspace(1) @surface, !"surface", i32 1}
