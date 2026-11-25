; RUN: llc < %s -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -enable-new-pm -verify-machineinstrs | FileCheck %s
; RUN: sed 's/^declare i32 @_setjmp/declare dllimport i32 @_setjmp/' %s > %t.ll
; RUN: not llc %t.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not llc -enable-new-pm %t.ll -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ERROR

target triple = "x86_64-pc-windows-msvc"

@env = dso_local global [256 x i8] zeroinitializer, align 16

declare i32 @_setjmp(ptr, ptr) returns_twice
declare ptr @llvm.frameaddress.p0(i32)
declare void @cleanup()
declare i32 @__CxxFrameHandler3(...)

; The CRT declares _setjmp without dllimport, so enabling ICO does not change
; the call sequence. The longjmp label must still point to the WinEH NOP.
define i32 @f() personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: f:
; CHECK-NOT:   .Limpcall
; CHECK:       callq _setjmp
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
; CHECK-NOT:   .secnum

; ERROR: LLVM ERROR: import call optimization does not support dllimport _setjmp

!llvm.module.flags = !{!0, !1}
!0 = !{i32 2, !"cfguard", i32 2}
!1 = !{i32 1, !"import-call-optimization", i32 1}
