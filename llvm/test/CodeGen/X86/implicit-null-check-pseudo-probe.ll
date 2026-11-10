; RUN: llc -enable-implicit-null-checks -O2 -mtriple=x86_64-unknown-linux-gnu \
; RUN:   -imp-null-max-insts-to-consider=1 < %s | FileCheck %s

; A pseudo probe has no execution semantics but is marked as having side
; effects.  It must not stop the load behind it from becoming an implicit null
; check, and must not use up the scan budget.

define i32 @imp_null_check_with_pseudo_probe(ptr %p) {
; CHECK-LABEL: imp_null_check_with_pseudo_probe:
; CHECK-NOT:   testq
; CHECK:       movl ({{.*}}), %eax # on-fault:
entry:
  call void @llvm.pseudoprobe(i64 1234, i64 1, i32 1, i64 0)
  %isnull = icmp eq ptr %p, null
  br i1 %isnull, label %null, label %notnull, !make.implicit !0

null:
  ret i32 42

notnull:
  call void @llvm.pseudoprobe(i64 1234, i64 2, i32 1, i64 0)
  %value = load i32, ptr %p, align 4
  ret i32 %value
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)

!0 = !{}
