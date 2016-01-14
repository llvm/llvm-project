; RUN: llc -march=ppc64le -code-model=large < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

@number64 = global i64 10, align 8

; CHECK: .abiversion 2

define i64 @use_toc(i64 %a) nounwind {
entry:
; CHECK: .Lfunc_toc[[FN:[0-9]+]]:
; CHECK-NEXT: .quad .TOC.-.Lfunc_gep[[FN]]
; CHECK: use_toc:
; CHECK-NEXT: .L{{.*}}:
; CHECK-NEXT: .Lfunc_gep[[FN]]:
; CHECK-NEXT: ld 2, .Lfunc_toc[[FN]]-.Lfunc_gep[[FN]](12)
; CHECK-NEXT: add 2, 2, 12
; CHECK-NEXT: .Lfunc_lep[[FN]]:
; CHECK-NEXT: .localentry use_toc, .Lfunc_lep[[FN]]-.Lfunc_gep[[FN]]
; CHECK-NEXT: %entry
  %0 = load i64, i64* @number64, align 8
  %cmp = icmp eq i64 %0, %a
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}

