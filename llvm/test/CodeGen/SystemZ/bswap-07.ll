; Test 32-bit byteswaps from registers to memory.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i16 @llvm.bswap.i16(i16 %a)

; Check STRVH with no displacement.
define void @f1(ptr %dst, i16 %a) {
; CHECK-LABEL: f1:
; CHECK: strvh %r3, 0(%r2)
; CHECK: br %r14
  %swapped = call i16 @llvm.bswap.i16(i16 %a)
  store i16 %swapped, ptr %dst
  ret void
}

; Check the high end of the aligned STRVH range.
define void @f2(ptr %dst, i16 %a) {
; CHECK-LABEL: f2:
; CHECK: strvh %r3, 524286(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %dst, i64 262143
  %swapped = call i16 @llvm.bswap.i16(i16 %a)
  store i16 %swapped, ptr %ptr
  ret void
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f3(ptr %dst, i16 %a) {
; CHECK-LABEL: f3:
; CHECK: agfi %r2, 524288
; CHECK: strvh %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %dst, i64 262144
  %swapped = call i16 @llvm.bswap.i16(i16 %a)
  store i16 %swapped, ptr %ptr
  ret void
}

; Check the high end of the negative aligned STRVH range.
define void @f4(ptr %dst, i16 %a) {
; CHECK-LABEL: f4:
; CHECK: strvh %r3, -2(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %dst, i64 -1
  %swapped = call i16 @llvm.bswap.i16(i16 %a)
  store i16 %swapped, ptr %ptr
  ret void
}

; Check the low end of the STRVH range.
define void @f5(ptr %dst, i16 %a) {
; CHECK-LABEL: f5:
; CHECK: strvh %r3, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %dst, i64 -262144
  %swapped = call i16 @llvm.bswap.i16(i16 %a)
  store i16 %swapped, ptr %ptr
  ret void
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f6(ptr %dst, i16 %a) {
; CHECK-LABEL: f6:
; CHECK: agfi %r2, -524290
; CHECK: strvh %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %dst, i64 -262145
  %swapped = call i16 @llvm.bswap.i16(i16 %a)
  store i16 %swapped, ptr %ptr
  ret void
}

; Check that STRVH allows an index.
define void @f7(i64 %src, i64 %index, i16 %a) {
; CHECK-LABEL: f7:
; CHECK: strvh %r4, 524287({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to ptr
  %swapped = call i16 @llvm.bswap.i16(i16 %a)
  store i16 %swapped, ptr %ptr
  ret void
}

