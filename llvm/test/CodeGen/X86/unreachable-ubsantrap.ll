; RUN: llc -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK
; RUN: llc -global-isel -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK
; RUN: llc -fast-isel -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK
; RUN: llc --trap-unreachable -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,TRAP_UNREACHABLE
; RUN: llc --trap-unreachable -global-isel -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,TRAP_UNREACHABLE
; RUN: llc --trap-unreachable -fast-isel -o - %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,TRAP_UNREACHABLE

; CHECK-LABEL: ubsantrap:
; CHECK:         ud1l 12(%eax), %eax
; CHECK-NOT:     ud2
define i32 @ubsantrap() noreturn nounwind {
  call void @llvm.ubsantrap(i8 12)
  unreachable
}

; CHECK-LABEL:      ubsantrap_fn_attr:
; CHECK:              callq {{_?}}ubsantrap_func
; TRAP_UNREACHABLE:   ud2
; CHECK-NOT:          ud2
define i32 @ubsantrap_fn_attr() noreturn nounwind {
  call void @llvm.ubsantrap(i8 12) "trap-func-name"="ubsantrap_func"
  unreachable
}

declare void @llvm.ubsantrap(i8) cold noreturn nounwind
