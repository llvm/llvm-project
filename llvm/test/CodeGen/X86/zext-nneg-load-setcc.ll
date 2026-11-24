; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -O2 | FileCheck %s

define ptr @zext_nneg_load_signed_cmp(ptr %p) {
; CHECK-LABEL: zext_nneg_load_signed_cmp:
; CHECK:       movzbl (%rdi), %eax
; CHECK:       cmpb $96, %al
  %op = load i8, ptr %p, align 1
  %cmp = icmp sgt i8 %op, 95
  %z = zext nneg i8 %op to i64
  %q = getelementptr i8, ptr %p, i64 %z
  %skip = getelementptr i8, ptr %q, i64 -94
  %next = getelementptr inbounds nuw i8, ptr %p, i64 1
  %r = select i1 %cmp, ptr %skip, ptr %next
  ret ptr %r
}
