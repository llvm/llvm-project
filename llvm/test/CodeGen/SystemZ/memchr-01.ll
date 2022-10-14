; Test memchr using SRST, with the correct prototype.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -no-integrated-as | FileCheck %s

declare ptr@memchr(ptr %src, i32 %char, i64 %len)

; Test a simple forwarded call.
define ptr@f1(i64 %len, ptr %src, i32 %char) {
; CHECK-LABEL: f1:
; CHECK-DAG: agr %r2, %r3
; CHECK-DAG: llcr %r0, %r4
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: srst %r2, %r3
; CHECK-NEXT: jo [[LABEL]]
; CHECK: blr %r14
; CHECK: lghi %r2, 0
; CHECK: br %r14
  %res = call ptr@memchr(ptr %src, i32 %char, i64 %len)
  ret ptr %res
}

; Test a doubled call with no use of %r0 in between.  There should be a
; single load of %r0.
define ptr@f2(ptr %src, ptr %charptr, i64 %len) {
; CHECK-LABEL: f2:
; CHECK: llc %r0, 0(%r3)
; CHECK-NOT: %r0
; CHECK: srst [[RES1:%r[1-5]]], %r2
; CHECK-NOT: %r0
; CHECK: srst %r2, [[RES1]]
; CHECK: br %r14
  %char = load volatile i8, ptr %charptr
  %charext = zext i8 %char to i32
  %res1 = call ptr@memchr(ptr %src, i32 %charext, i64 %len)
  %res2 = call ptr@memchr(ptr %res1, i32 %charext, i64 %len)
  ret ptr %res2
}

; Test a doubled call with a use of %r0 in between.  %r0 must be loaded
; for each loop.
define ptr@f3(ptr %src, ptr %charptr, i64 %len) {
; CHECK-LABEL: f3:
; CHECK: llc [[CHAR:%r[1-5]]], 0(%r3)
; CHECK: lr %r0, [[CHAR]]
; CHECK: srst [[RES1:%r[1-5]]], %r2
; CHECK: lhi %r0, 0
; CHECK: blah %r0
; CHECK: lr %r0, [[CHAR]]
; CHECK: srst %r2, [[RES1]]
; CHECK: br %r14
  %char = load volatile i8, ptr %charptr
  %charext = zext i8 %char to i32
  %res1 = call ptr@memchr(ptr %src, i32 %charext, i64 %len)
  call void asm sideeffect "blah $0", "{r0}" (i32 0)
  %res2 = call ptr@memchr(ptr %res1, i32 %charext, i64 %len)
  ret ptr %res2
}
