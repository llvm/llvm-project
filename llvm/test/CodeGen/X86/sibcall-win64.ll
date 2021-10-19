; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck %s

declare dso_local win64cc void @win64_callee(i32)
declare dso_local win64cc void (i32)* @win64_indirect()
declare dso_local win64cc void @win64_other(i32)
declare dso_local void @sysv_callee(i32)
declare dso_local void (i32)* @sysv_indirect()
declare dso_local void @sysv_other(i32)

define void @sysv_caller(i32 %p1) {
entry:
  tail call win64cc void @win64_callee(i32 %p1)
  ret void
}

; CHECK-LABEL: sysv_caller:
; CHECK: subq $40, %rsp
; CHECK: callq win64_callee
; CHECK: addq $40, %rsp
; CHECK: retq

define win64cc void @win64_caller(i32 %p1) {
entry:
  tail call void @sysv_callee(i32 %p1)
  ret void
}

; CHECK-LABEL: win64_caller:
; CHECK: callq sysv_callee
; CHECK: retq

define void @sysv_matched(i32 %p1) {
  tail call void @sysv_callee(i32 %p1)
  ret void
}

; CHECK-LABEL: sysv_matched:
; CHECK: jmp sysv_callee # TAILCALL

define win64cc void @win64_matched(i32 %p1) {
  tail call win64cc void @win64_callee(i32 %p1)
  ret void
}

; CHECK-LABEL: win64_matched:
; CHECK: jmp win64_callee # TAILCALL

define win64cc void @win64_indirect_caller(i32 %p1) {
  %1 = call win64cc void (i32)* @win64_indirect()
  call win64cc void @win64_other(i32 0)
  tail call win64cc void %1(i32 %p1)
  ret void
}

; CHECK-LABEL: win64_indirect_caller:
; CHECK: jmpq *%{{rax|rcx|rdx|r8|r9|r11}} # TAILCALL

define void @sysv_indirect_caller(i32 %p1) {
  %1 = call void (i32)* @sysv_indirect()
  call void @sysv_other(i32 0)
  tail call void %1(i32 %p1)
  ret void
}

; CHECK-LABEL: sysv_indirect_caller:
; CHECK: jmpq *%{{rax|rcx|rdx|rsi|rdi|r8|r9|r11}} # TAILCALL
