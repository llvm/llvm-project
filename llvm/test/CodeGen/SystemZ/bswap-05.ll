; Test 64-bit byteswaps from registers to memory.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @llvm.bswap.i64(i64 %a)

; Check STRVG with no displacement.
define void @f1(i64 *%dst, i64 %a) {
; CHECK-LABEL: f1:
; CHECK: strvg %r3, 0(%r2)
; CHECK: br %r14
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  store i64 %swapped, i64 *%dst
  ret void
}

; Check the high end of the aligned STRVG range.
define void @f2(i64 *%dst, i64 %a) {
; CHECK-LABEL: f2:
; CHECK: strvg %r3, 524280(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%dst, i64 65535
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  store i64 %swapped, i64 *%ptr
  ret void
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f3(i64 *%dst, i64 %a) {
; CHECK-LABEL: f3:
; CHECK: agfi %r2, 524288
; CHECK: strvg %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%dst, i64 65536
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  store i64 %swapped, i64 *%ptr
  ret void
}

; Check the high end of the negative aligned STRVG range.
define void @f4(i64 *%dst, i64 %a) {
; CHECK-LABEL: f4:
; CHECK: strvg %r3, -8(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%dst, i64 -1
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  store i64 %swapped, i64 *%ptr
  ret void
}

; Check the low end of the STRVG range.
define void @f5(i64 *%dst, i64 %a) {
; CHECK-LABEL: f5:
; CHECK: strvg %r3, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%dst, i64 -65536
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  store i64 %swapped, i64 *%ptr
  ret void
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f6(i64 *%dst, i64 %a) {
; CHECK-LABEL: f6:
; CHECK: agfi %r2, -524296
; CHECK: strvg %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%dst, i64 -65537
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  store i64 %swapped, i64 *%ptr
  ret void
}

; Check that STRVG allows an index.
define void @f7(i64 %src, i64 %index, i64 %a) {
; CHECK-LABEL: f7:
; CHECK: strvg %r4, 524287({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i64 *
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  store i64 %swapped, i64 *%ptr
  ret void
}

