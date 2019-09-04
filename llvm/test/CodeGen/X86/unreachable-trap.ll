; RUN: llc -o - %s -mtriple=x86_64-windows-msvc | FileCheck %s --check-prefixes=CHECK,TRAP_AFTER_NORETURN
; RUN: llc -o - %s -mtriple=x86_64-apple-darwin | FileCheck %s --check-prefixes=CHECK,NO_TRAP_AFTER_NORETURN

; CHECK-LABEL: call_exit:
; CHECK: callq {{_?}}exit
; TRAP_AFTER_NORETURN: ud2
; NO_TRAP_AFTER_NORETURN-NOT: ud2
define i32 @call_exit() noreturn nounwind {
  tail call void @exit(i32 0)
  unreachable
}

; CHECK-LABEL: trap:
; CHECK: ud2
; TRAP_AFTER_NORETURN: ud2
; NO_TRAP_AFTER_NORETURN-NOT: ud2
define i32 @trap() noreturn nounwind {
  tail call void @llvm.trap()
  unreachable
}

; CHECK-LABEL: unreachable:
; CHECK: ud2
define i32 @unreachable() noreturn nounwind {
  unreachable
}

declare void @llvm.trap() nounwind noreturn
declare void @exit(i32 %rc) nounwind noreturn
