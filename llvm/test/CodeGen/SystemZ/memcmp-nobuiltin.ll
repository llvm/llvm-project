; Test that memcmp won't be converted to CLC if calls are
; marked with nobuiltin, eg. for sanitizers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare signext i32 @memcmp(ptr %src1, ptr %src2, i64 %size)

; Zero-length comparisons should be optimized away.
define i32 @f1(ptr %src1, ptr %src2) {
; CHECK-LABEL: f1:
; CHECK-NOT: clc
; CHECK: brasl %r14, memcmp
; CHECK: br %r14
  %res = call i32 @memcmp(ptr %src1, ptr %src2, i64 0) nobuiltin
  ret i32 %res
}

; Check a case where the result is used as an integer.
define i32 @f2(ptr %src1, ptr %src2) {
; CHECK-LABEL: f2:
; CHECK-NOT: clc
; CHECK: brasl %r14, memcmp
; CHECK: br %r14
  %res = call i32 @memcmp(ptr %src1, ptr %src2, i64 2) nobuiltin
  ret i32 %res
}

; Check a case where the result is tested for equality.
define void @f3(ptr %src1, ptr %src2, ptr %dest) {
; CHECK-LABEL: f3:
; CHECK-NOT: clc
; CHECK: brasl %r14, memcmp
; CHECK: br %r14
  %res = call i32 @memcmp(ptr %src1, ptr %src2, i64 3) nobuiltin
  %cmp = icmp eq i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, ptr %dest
  br label %exit

exit:
  ret void
}

; Check a case where the result is tested for inequality.
define void @f4(ptr %src1, ptr %src2, ptr %dest) {
; CHECK-LABEL: f4:
; CHECK-NOT: clc
; CHECK: brasl %r14, memcmp
; CHECK: br %r14
entry:
  %res = call i32 @memcmp(ptr %src1, ptr %src2, i64 4) nobuiltin
  %cmp = icmp ne i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, ptr %dest
  br label %exit

exit:
  ret void
}

; Check a case where the result is tested via slt.
define void @f5(ptr %src1, ptr %src2, ptr %dest) {
; CHECK-LABEL: f5:
; CHECK-NOT: clc
; CHECK: brasl %r14, memcmp
; CHECK: br %r14
entry:
  %res = call i32 @memcmp(ptr %src1, ptr %src2, i64 5) nobuiltin
  %cmp = icmp slt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, ptr %dest
  br label %exit

exit:
  ret void
}

; Check a case where the result is tested for sgt.
define void @f6(ptr %src1, ptr %src2, ptr %dest) {
; CHECK-LABEL: f6:
; CHECK-NOT: clc
; CHECK: brasl %r14, memcmp
; CHECK: br %r14
entry:
  %res = call i32 @memcmp(ptr %src1, ptr %src2, i64 6) nobuiltin
  %cmp = icmp sgt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, ptr %dest
  br label %exit

exit:
  ret void
}

; Check the upper end of the CLC range.  Here the result is used both as
; an integer and for branching.
define i32 @f7(ptr %src1, ptr %src2, ptr %dest) {
; CHECK-LABEL: f7:
; CHECK-NOT: clc
; CHECK: brasl %r14, memcmp
; CHECK: br %r14
entry:
  %res = call i32 @memcmp(ptr %src1, ptr %src2, i64 256) nobuiltin
  %cmp = icmp slt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, ptr %dest
  br label %exit

exit:
  ret i32 %res
}

; 257 bytes needs two CLCs.
define i32 @f8(ptr %src1, ptr %src2) {
; CHECK-LABEL: f8:
; CHECK-NOT: clc
; CHECK: brasl %r14, memcmp
; CHECK: br %r14
  %res = call i32 @memcmp(ptr %src1, ptr %src2, i64 257) nobuiltin
  ret i32 %res
}

; Test a comparison of 258 bytes in which the CC result can be used directly.
define void @f9(ptr %src1, ptr %src2, ptr %dest) {
; CHECK-LABEL: f9:
; CHECK-NOT: clc
; CHECK: brasl %r14, memcmp
; CHECK: br %r14
entry:
  %res = call i32 @memcmp(ptr %src1, ptr %src2, i64 257) nobuiltin
  %cmp = icmp slt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, ptr %dest
  br label %exit

exit:
  ret void
}

; Test the largest size that can use two CLCs.
define i32 @f10(ptr %src1, ptr %src2) {
; CHECK-LABEL: f10:
; CHECK-NOT: clc
; CHECK: brasl %r14, memcmp
; CHECK: br %r14
  %res = call i32 @memcmp(ptr %src1, ptr %src2, i64 512) nobuiltin
  ret i32 %res
}

; Test the smallest size that needs 3 CLCs.
define i32 @f11(ptr %src1, ptr %src2) {
; CHECK-LABEL: f11:
; CHECK-NOT: clc
; CHECK: brasl %r14, memcmp
; CHECK: br %r14
  %res = call i32 @memcmp(ptr %src1, ptr %src2, i64 513) nobuiltin
  ret i32 %res
}

; Test the largest size than can use 3 CLCs.
define i32 @f12(ptr %src1, ptr %src2) {
; CHECK-LABEL: f12:
; CHECK-NOT: clc
; CHECK: brasl %r14, memcmp
; CHECK: br %r14
  %res = call i32 @memcmp(ptr %src1, ptr %src2, i64 768) nobuiltin
  ret i32 %res
}

; The next size up uses a loop instead.  We leave the more complicated
; loop tests to memcpy-01.ll, which shares the same form.
define i32 @f13(ptr %src1, ptr %src2) {
; CHECK-LABEL: f13:
; CHECK-NOT: clc
; CHECK: brasl %r14, memcmp
; CHECK: br %r14
  %res = call i32 @memcmp(ptr %src1, ptr %src2, i64 769) nobuiltin
  ret i32 %res
}
