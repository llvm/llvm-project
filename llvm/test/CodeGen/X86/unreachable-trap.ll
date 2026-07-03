; RUN: llc -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK
; RUN: llc -o - %s -mtriple=x86_64-windows-msvc | FileCheck %s --check-prefixes=CHECK
; RUN: llc -o - %s -mtriple=x86_64-apple-darwin | FileCheck %s --check-prefixes=CHECK,NO_TRAP_AFTER_NORETURN

; On PS4/PS5, always emit trap instructions regardless of of trap-unreachable or no-trap-after-noreturn.
; RUN: llc -o - %s -mtriple=x86_64-scei-ps4 -trap-unreachable | FileCheck %s --check-prefixes=CHECK,TRAP_AFTER_NORETURN
; RUN: llc -o - %s -mtriple=x86_64-sie-ps5 -trap-unreachable | FileCheck %s --check-prefixes=CHECK,TRAP_AFTER_NORETURN
; RUN: llc -o - %s -mtriple=x86_64-scei-ps4 -trap-unreachable=false | FileCheck %s --check-prefixes=CHECK,TRAP_AFTER_NORETURN
; RUN: llc -o - %s -mtriple=x86_64-sie-ps5 -trap-unreachable=false | FileCheck %s --check-prefixes=CHECK,TRAP_AFTER_NORETURN
; RUN: llc -o - %s -mtriple=x86_64-scei-ps4 -trap-unreachable -no-trap-after-noreturn=false | FileCheck %s --check-prefixes=CHECK,TRAP_AFTER_NORETURN
; RUN: llc -o - %s -mtriple=x86_64-sie-ps5 -trap-unreachable -no-trap-after-noreturn=false | FileCheck %s --check-prefixes=CHECK,TRAP_AFTER_NORETURN

; RUN: llc --trap-unreachable -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,TRAP_AFTER_NORETURN
; RUN: llc --trap-unreachable -global-isel -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,TRAP_AFTER_NORETURN
; RUN: llc --trap-unreachable -fast-isel -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,TRAP_AFTER_NORETURN

; CHECK-LABEL: call_exit:
; CHECK: callq {{_?}}exit
; TRAP_AFTER_NORETURN: ud2
; CHECK-NOT: ud2
define i32 @call_exit() noreturn nounwind {
  tail call void @exit(i32 0)
  unreachable
}

; CHECK-LABEL: trap:
; CHECK: ud2
; CHECK-NOT: ud2
define i32 @trap() noreturn nounwind {
  tail call void @llvm.trap()
  unreachable
}

; CHECK-LABEL: trap_fn_attr:
; CHECK: callq {{_?}}trap_func
; TRAP_AFTER_NORETURN: ud2
; CHECK-NOT: ud2
define i32 @trap_fn_attr() noreturn nounwind {
  tail call void @llvm.trap() "trap-func-name"="trap_func"
  unreachable
}

; CHECK-LABEL: noreturn_indirect:
; CHECK: callq *%r{{.+}}
; TRAP_AFTER_NORETURN: ud2
; CHECK-NOT: ud2
define i32 @noreturn_indirect(ptr %fptr) noreturn nounwind {
  tail call void (...) %fptr() noreturn nounwind
  unreachable
}

; CHECK-LABEL: unreachable:
; TRAP_AFTER_NORETURN: ud2
; NO_TRAP_AFTER_NORETURN: ud2
; CHECK-NOT: ud2
; CHECK: # -- End function
define i32 @unreachable() noreturn nounwind {
  unreachable
}

declare void @llvm.trap() nounwind noreturn
declare void @exit(i32 %rc) nounwind noreturn
