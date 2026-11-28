; RUN: llc < %s -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -enable-new-pm -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -stop-after=x86-avoid-trailing-call -o - | FileCheck %s --check-prefix=MIR

target triple = "x86_64-pc-windows-msvc"

@env = dso_local global [256 x i8] zeroinitializer, align 16

declare i32 @_setjmp(ptr, ptr) returns_twice
declare ptr @llvm.frameaddress.p0(i32)
declare void @cleanup()
declare i32 @__CxxFrameHandler3(...)

; Windows x64 unwinding may insert padding after a call to ensure that the
; return address is in the same IP2State as the call instruction itself. If this
; NOP is inserted, then we need to make sure that the longjmp target is the
; call's return address (that is, the inserted NOP) and not the instruction
; after the NOP.
; CHECK-LABEL: f:
; CHECK:       callq _setjmp
; CHECK-NEXT:  $cfgsj_f0:
; CHECK-NEXT:  nop
; CHECK-NEXT:  {{.*}} # EH_LABEL
; CHECK:       .section .gljmp$y,"dr"
; CHECK-NEXT:  .symidx $cfgsj_f0
; MIR:         CALL64pcrel32 @_setjmp, {{.*}}post-instr-symbol <mcsymbol $cfgsj_f0>
; MIR-NEXT:    NOOP
; MIR-NEXT:    EH_LABEL

define i32 @f() personality ptr @__CxxFrameHandler3 {
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

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 2}
