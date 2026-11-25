; RUN: llc < %s -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -enable-new-pm -verify-machineinstrs | FileCheck %s

target triple = "x86_64-pc-windows-msvc"

@env = dso_local global [256 x i8] zeroinitializer, align 16

declare dllimport i32 @_setjmp(ptr, ptr) returns_twice
declare ptr @llvm.frameaddress.p0(i32)
declare void @cleanup()
declare i32 @__CxxFrameHandler3(...)

; The longjmp label goes after the loader-rewritable ICO sequence but before
; the separate WinEH NOP and the end of the invoke's EH range.
define i32 @f() personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: f:
; CHECK:       [[ICO:\.Limpcall[0-9]+]]:
; CHECK-NEXT:  rex64
; CHECK-NEXT:  callq __imp__setjmp
; CHECK-NEXT:  nopl 8(%rax,%rax)
; CHECK-NEXT:  $cfgsj_f0:
; CHECK-NEXT:  nop
; CHECK-NEXT:  .Ltmp{{[0-9]+}}:{{.*}}# EH_LABEL
entry:
  %fp = call ptr @llvm.frameaddress.p0(i32 0)
  %r = invoke i32 @_setjmp(ptr @env, ptr %fp) returns_twice
          to label %cont unwind label %ehcleanup

cont:
  ret i32 %r

ehcleanup:
  %cp = cleanuppad within none []
  call void @cleanup() [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller
}

; CHECK:       .section .gljmp$y,"dr"
; CHECK-NEXT:  .symidx $cfgsj_f0
; CHECK:       .section .retplne,"yi"
; CHECK-NEXT:  .asciz "RetpolineV1"
; CHECK-NEXT:  .long 16
; CHECK-NEXT:  .secnum .text
; CHECK-NEXT:  .long 3
; CHECK-NEXT:  .secoffset [[ICO]]

!llvm.module.flags = !{!0, !1}
!0 = !{i32 2, !"cfguard", i32 2}
!1 = !{i32 1, !"import-call-optimization", i32 1}
