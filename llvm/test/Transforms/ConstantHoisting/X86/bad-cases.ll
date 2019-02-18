; RUN: opt -consthoist -S < %s | FileCheck %s
target triple = "x86_64--"

; We don't want to convert constant divides because the benefit from converting
; them to a mul in the backend is larget than constant materialization savings.
define void @signed_const_division(i64 %in1, i64 %in2, i64* %addr) {
; CHECK-LABEL: @signed_const_division
; CHECK: %res1 = sdiv i64 %l1, 4294967296
; CHECK: %res2 = srem i64 %l2, 4294967296
entry:
  br label %loop

loop:
  %l1 = phi i64 [%res1, %loop], [%in1, %entry]
  %l2 = phi i64 [%res2, %loop], [%in2, %entry]
  %res1 = sdiv i64 %l1, 4294967296
  store volatile i64 %res1, i64* %addr
  %res2 = srem i64 %l2, 4294967296
  store volatile i64 %res2, i64* %addr
  %again = icmp eq i64 %res1, %res2
  br i1 %again, label %loop, label %end

end:
  ret void
}

define void @unsigned_const_division(i64 %in1, i64 %in2, i64* %addr) {
; CHECK-LABEL: @unsigned_const_division
; CHECK: %res1 = udiv i64 %l1, 4294967296
; CHECK: %res2 = urem i64 %l2, 4294967296

entry:
  br label %loop

loop:
  %l1 = phi i64 [%res1, %loop], [%in1, %entry]
  %l2 = phi i64 [%res2, %loop], [%in2, %entry]
  %res1 = udiv i64 %l1, 4294967296
  store volatile i64 %res1, i64* %addr
  %res2 = urem i64 %l2, 4294967296
  store volatile i64 %res2, i64* %addr
  %again = icmp eq i64 %res1, %res2
  br i1 %again, label %loop, label %end

end:
  ret void
}
