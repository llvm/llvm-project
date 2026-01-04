; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s --check-prefix=CHECK

define dso_local void @normal_call(ptr noundef readonly %func_ptr) local_unnamed_addr section "nc_sect" {
entry:
  call void %func_ptr()
  ret void
}
; CHECK-LABEL:  normal_call:
; CHECK:        .Limpcall0:
; CHECK-NEXT:     callq   *__guard_dispatch_icall_fptr(%rip)

define dso_local void @tail_call_fp(ptr noundef readonly %func_ptr) local_unnamed_addr section "tc_sect" {
entry:
  tail call void %func_ptr()
  ret void
}
; CHECK-LABEL:  tail_call_fp:
; CHECK:        .Limpcall1:
; CHECK-NEXT:     rex64 jmpq      *__guard_dispatch_icall_fptr(%rip)

; CHECK-LABEL  .section   .retplne,"yi"
; CHECK-NEXT   .asciz  "RetpolineV1"
; CHECK-NEXT   .long   16
; CHECK-NEXT   .secnum tc_sect
; CHECK-NEXT   .long   10
; CHECK-NEXT   .secoffset      .Limpcall1
; CHECK-NEXT   .long   16
; CHECK-NEXT   .secnum nc_sect
; CHECK-NEXT   .long   9
; CHECK-NEXT   .secoffset      .Limpcall0

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"import-call-optimization", i32 1}
!1 = !{i32 2, !"cfguard", i32 2}
