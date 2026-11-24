; RUN: llc < %s -mtriple=aarch64-linux-gnu -O2 | FileCheck %s

define ptr @zext_nneg_load_signed_select(ptr %p) {
; CHECK-LABEL: zext_nneg_load_signed_select:
; CHECK:       ldrsb x[[BYTE:[0-9]+]],
; CHECK-NOT:   sxtb
; CHECK:       cmp {{[wx]}}[[BYTE]], #95
; CHECK-NOT:   sxtb
; CHECK:       csel {{.*}}, gt
  %op = load i8, ptr %p, align 1
  %cmp = icmp sgt i8 %op, 95
  %z = zext nneg i8 %op to i64
  %q = getelementptr i8, ptr %p, i64 %z
  %skip = getelementptr i8, ptr %q, i64 -94
  %next = getelementptr inbounds nuw i8, ptr %p, i64 1
  %r = select i1 %cmp, ptr %skip, ptr %next
  ret ptr %r
}
