; RUN: llc --trap-unreachable -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,NO_TRAP_AFTER_NORET
; RUN: llc --trap-unreachable -global-isel -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,NO_TRAP_AFTER_NORET
; RUN: llc --trap-unreachable -fast-isel -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,NO_TRAP_AFTER_NORET

; RUN: llc --trap-unreachable=false -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,NO_TRAP_AFTER_NORET
; RUN: llc --trap-unreachable=false -global-isel -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,NO_TRAP_AFTER_NORET
; RUN: llc --trap-unreachable=false -fast-isel -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,NO_TRAP_AFTER_NORET

; RUN: llc --trap-unreachable --no-trap-after-noreturn=false -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,TRAP_AFTER_NORET
; RUN: llc --trap-unreachable --no-trap-after-noreturn=false -global-isel -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,TRAP_AFTER_NORET
; RUN: llc --trap-unreachable --no-trap-after-noreturn=false -fast-isel -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,TRAP_AFTER_NORET

; CHECK-LABEL: ubsantrap:
; CHECK:         ud1l 12(%eax), %eax
; CHECK-NOT:     ud2
define i32 @ubsantrap() noreturn nounwind {
  call void @llvm.ubsantrap(i8 12)
  unreachable
}

; CHECK-LABEL:      ubsantrap_fn_attr:
; CHECK:              callq {{_?}}ubsantrap_func
; TRAP_AFTER_NORET:        ud2
; NO_TRAP_AFTER_NORET-NOT: ud2
define i32 @ubsantrap_fn_attr() noreturn nounwind {
  call void @llvm.ubsantrap(i8 12) "trap-func-name"="ubsantrap_func"
  unreachable
}

declare void @llvm.ubsantrap(i8) cold noreturn nounwind
