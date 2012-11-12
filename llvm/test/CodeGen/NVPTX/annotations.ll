; RUN: llc < %s -march=nvptx -mcpu=sm_10 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_10 | FileCheck %s
; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s


@texture = internal addrspace(1) global i64 0, align 8
; CHECK: .global .texref texture
@surface = internal addrspace(1) global i64 0, align 8
; CHECK: .global .surfref surface


; CHECK: .entry kernel_func_maxntid
define void @kernel_func_maxntid(float* %a) {
; CHECK: .maxntid 10, 20, 30
; CHECK: ret
  ret void
}

; CHECK: .entry kernel_func_reqntid
define void @kernel_func_reqntid(float* %a) {
; CHECK: .reqntid 11, 22, 33
; CHECK: ret
  ret void
}

; CHECK: .entry kernel_func_minctasm
define void @kernel_func_minctasm(float* %a) {
; CHECK: .minnctapersm 42
; CHECK: ret
  ret void
}



!nvvm.annotations = !{!1, !2, !3, !4, !5, !6, !7, !8}

!1 = metadata !{void (float*)* @kernel_func_maxntid, metadata !"kernel", i32 1}
!2 = metadata !{void (float*)* @kernel_func_maxntid,
                metadata !"maxntidx", i32 10,
                metadata !"maxntidy", i32 20,
                metadata !"maxntidz", i32 30}

!3 = metadata !{void (float*)* @kernel_func_reqntid, metadata !"kernel", i32 1}
!4 = metadata !{void (float*)* @kernel_func_reqntid,
                metadata !"reqntidx", i32 11,
                metadata !"reqntidy", i32 22,
                metadata !"reqntidz", i32 33}

!5 = metadata !{void (float*)* @kernel_func_minctasm, metadata !"kernel", i32 1}
!6 = metadata !{void (float*)* @kernel_func_minctasm,
                metadata !"minctasm", i32 42}

!7 = metadata !{i64 addrspace(1)* @texture, metadata !"texture", i32 1}
!8 = metadata !{i64 addrspace(1)* @surface, metadata !"surface", i32 1}
