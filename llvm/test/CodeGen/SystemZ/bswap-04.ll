; Test 32-bit byteswaps from registers to memory.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @llvm.bswap.i32(i32 %a)

; Check STRV with no displacement.
define void @f1(i32 *%dst, i32 %a) {
; CHECK-LABEL: f1:
; CHECK: strv %r3, 0(%r2)
; CHECK: br %r14
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  store i32 %swapped, i32 *%dst
  ret void
}

; Check the high end of the aligned STRV range.
define void @f2(i32 *%dst, i32 %a) {
; CHECK-LABEL: f2:
; CHECK: strv %r3, 524284(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%dst, i64 131071
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  store i32 %swapped, i32 *%ptr
  ret void
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f3(i32 *%dst, i32 %a) {
; CHECK-LABEL: f3:
; CHECK: agfi %r2, 524288
; CHECK: strv %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%dst, i64 131072
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  store i32 %swapped, i32 *%ptr
  ret void
}

; Check the high end of the negative aligned STRV range.
define void @f4(i32 *%dst, i32 %a) {
; CHECK-LABEL: f4:
; CHECK: strv %r3, -4(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%dst, i64 -1
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  store i32 %swapped, i32 *%ptr
  ret void
}

; Check the low end of the STRV range.
define void @f5(i32 *%dst, i32 %a) {
; CHECK-LABEL: f5:
; CHECK: strv %r3, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%dst, i64 -131072
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  store i32 %swapped, i32 *%ptr
  ret void
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f6(i32 *%dst, i32 %a) {
; CHECK-LABEL: f6:
; CHECK: agfi %r2, -524292
; CHECK: strv %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%dst, i64 -131073
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  store i32 %swapped, i32 *%ptr
  ret void
}

; Check that STRV allows an index.
define void @f7(i64 %src, i64 %index, i32 %a) {
; CHECK-LABEL: f7:
; CHECK: strv %r4, 524287({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i32 *
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  store i32 %swapped, i32 *%ptr
  ret void
}

; Check that volatile stores do not use STRV, which might access the
; storage multple times.
define void @f8(i32 *%dst, i32 %a) {
; CHECK-LABEL: f8:
; CHECK: lrvr [[REG:%r[0-5]]], %r3
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  store volatile i32 %swapped, i32 *%dst
  ret void
}
