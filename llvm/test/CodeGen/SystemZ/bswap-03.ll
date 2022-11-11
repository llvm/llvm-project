; Test 64-bit byteswaps from memory to registers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @llvm.bswap.i64(i64 %a)

; Check LRVG with no displacement.
define i64 @f1(ptr %src) {
; CHECK-LABEL: f1:
; CHECK: lrvg %r2, 0(%r2)
; CHECK: br %r14
  %a = load i64, ptr %src
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  ret i64 %swapped
}

; Check the high end of the aligned LRVG range.
define i64 @f2(ptr %src) {
; CHECK-LABEL: f2:
; CHECK: lrvg %r2, 524280(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 65535
  %a = load i64, ptr %ptr
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  ret i64 %swapped
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f3(ptr %src) {
; CHECK-LABEL: f3:
; CHECK: agfi %r2, 524288
; CHECK: lrvg %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 65536
  %a = load i64, ptr %ptr
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  ret i64 %swapped
}

; Check the high end of the negative aligned LRVG range.
define i64 @f4(ptr %src) {
; CHECK-LABEL: f4:
; CHECK: lrvg %r2, -8(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 -1
  %a = load i64, ptr %ptr
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  ret i64 %swapped
}

; Check the low end of the LRVG range.
define i64 @f5(ptr %src) {
; CHECK-LABEL: f5:
; CHECK: lrvg %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 -65536
  %a = load i64, ptr %ptr
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  ret i64 %swapped
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f6(ptr %src) {
; CHECK-LABEL: f6:
; CHECK: agfi %r2, -524296
; CHECK: lrvg %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 -65537
  %a = load i64, ptr %ptr
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  ret i64 %swapped
}

; Check that LRVG allows an index.
define i64 @f7(i64 %src, i64 %index) {
; CHECK-LABEL: f7:
; CHECK: lrvg %r2, 524287({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to ptr
  %a = load i64, ptr %ptr
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  ret i64 %swapped
}

; Test a case where we spill the source of at least one LRVGR.  We want
; to use LRVG if possible.
define i64 @f8(ptr %ptr) {
; CHECK-LABEL: f8:
; CHECK: lrvg {{%r[0-9]+}}, 160(%r15)
; CHECK: br %r14

  %val0 = call i64 @foo()
  %val1 = call i64 @foo()
  %val2 = call i64 @foo()
  %val3 = call i64 @foo()
  %val4 = call i64 @foo()
  %val5 = call i64 @foo()
  %val6 = call i64 @foo()
  %val7 = call i64 @foo()
  %val8 = call i64 @foo()
  %val9 = call i64 @foo()

  %swapped0 = call i64 @llvm.bswap.i64(i64 %val0)
  %swapped1 = call i64 @llvm.bswap.i64(i64 %val1)
  %swapped2 = call i64 @llvm.bswap.i64(i64 %val2)
  %swapped3 = call i64 @llvm.bswap.i64(i64 %val3)
  %swapped4 = call i64 @llvm.bswap.i64(i64 %val4)
  %swapped5 = call i64 @llvm.bswap.i64(i64 %val5)
  %swapped6 = call i64 @llvm.bswap.i64(i64 %val6)
  %swapped7 = call i64 @llvm.bswap.i64(i64 %val7)
  %swapped8 = call i64 @llvm.bswap.i64(i64 %val8)
  %swapped9 = call i64 @llvm.bswap.i64(i64 %val9)

  %ret1 = add i64 %swapped0, %swapped1
  %ret2 = add i64 %ret1, %swapped2
  %ret3 = add i64 %ret2, %swapped3
  %ret4 = add i64 %ret3, %swapped4
  %ret5 = add i64 %ret4, %swapped5
  %ret6 = add i64 %ret5, %swapped6
  %ret7 = add i64 %ret6, %swapped7
  %ret8 = add i64 %ret7, %swapped8
  %ret9 = add i64 %ret8, %swapped9

  ret i64 %ret9
}

declare i64 @foo()
