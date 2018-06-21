; Test 32-bit byteswaps from memory to registers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @llvm.bswap.i32(i32 %a)

; Check LRV with no displacement.
define i32 @f1(i32 *%src) {
; CHECK-LABEL: f1:
; CHECK: lrv %r2, 0(%r2)
; CHECK: br %r14
  %a = load i32 , i32 *%src
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check the high end of the aligned LRV range.
define i32 @f2(i32 *%src) {
; CHECK-LABEL: f2:
; CHECK: lrv %r2, 524284(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131071
  %a = load i32 , i32 *%ptr
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f3(i32 *%src) {
; CHECK-LABEL: f3:
; CHECK: agfi %r2, 524288
; CHECK: lrv %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131072
  %a = load i32 , i32 *%ptr
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check the high end of the negative aligned LRV range.
define i32 @f4(i32 *%src) {
; CHECK-LABEL: f4:
; CHECK: lrv %r2, -4(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -1
  %a = load i32 , i32 *%ptr
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check the low end of the LRV range.
define i32 @f5(i32 *%src) {
; CHECK-LABEL: f5:
; CHECK: lrv %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -131072
  %a = load i32 , i32 *%ptr
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f6(i32 *%src) {
; CHECK-LABEL: f6:
; CHECK: agfi %r2, -524292
; CHECK: lrv %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -131073
  %a = load i32 , i32 *%ptr
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check that LRV allows an index.
define i32 @f7(i64 %src, i64 %index) {
; CHECK-LABEL: f7:
; CHECK: lrv %r2, 524287({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i32 *
  %a = load i32 , i32 *%ptr
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check that volatile accesses do not use LRV, which might access the
; storage multple times.
define i32 @f8(i32 *%src) {
; CHECK-LABEL: f8:
; CHECK: l [[REG:%r[0-5]]], 0(%r2)
; CHECK: lrvr %r2, [[REG]]
; CHECK: br %r14
  %a = load volatile i32 , i32 *%src
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Test a case where we spill the source of at least one LRVR.  We want
; to use LRV if possible.
define void @f9(i32 *%ptr) {
; CHECK-LABEL: f9:
; CHECK: lrv {{%r[0-9]+}}, 16{{[04]}}(%r15)
; CHECK: br %r14
  %val0 = load volatile i32 , i32 *%ptr
  %val1 = load volatile i32 , i32 *%ptr
  %val2 = load volatile i32 , i32 *%ptr
  %val3 = load volatile i32 , i32 *%ptr
  %val4 = load volatile i32 , i32 *%ptr
  %val5 = load volatile i32 , i32 *%ptr
  %val6 = load volatile i32 , i32 *%ptr
  %val7 = load volatile i32 , i32 *%ptr
  %val8 = load volatile i32 , i32 *%ptr
  %val9 = load volatile i32 , i32 *%ptr
  %val10 = load volatile i32 , i32 *%ptr
  %val11 = load volatile i32 , i32 *%ptr
  %val12 = load volatile i32 , i32 *%ptr
  %val13 = load volatile i32 , i32 *%ptr
  %val14 = load volatile i32 , i32 *%ptr
  %val15 = load volatile i32 , i32 *%ptr

  %swapped0 = call i32 @llvm.bswap.i32(i32 %val0)
  %swapped1 = call i32 @llvm.bswap.i32(i32 %val1)
  %swapped2 = call i32 @llvm.bswap.i32(i32 %val2)
  %swapped3 = call i32 @llvm.bswap.i32(i32 %val3)
  %swapped4 = call i32 @llvm.bswap.i32(i32 %val4)
  %swapped5 = call i32 @llvm.bswap.i32(i32 %val5)
  %swapped6 = call i32 @llvm.bswap.i32(i32 %val6)
  %swapped7 = call i32 @llvm.bswap.i32(i32 %val7)
  %swapped8 = call i32 @llvm.bswap.i32(i32 %val8)
  %swapped9 = call i32 @llvm.bswap.i32(i32 %val9)
  %swapped10 = call i32 @llvm.bswap.i32(i32 %val10)
  %swapped11 = call i32 @llvm.bswap.i32(i32 %val11)
  %swapped12 = call i32 @llvm.bswap.i32(i32 %val12)
  %swapped13 = call i32 @llvm.bswap.i32(i32 %val13)
  %swapped14 = call i32 @llvm.bswap.i32(i32 %val14)
  %swapped15 = call i32 @llvm.bswap.i32(i32 %val15)

  store volatile i32 %val0, i32 *%ptr
  store volatile i32 %val1, i32 *%ptr
  store volatile i32 %val2, i32 *%ptr
  store volatile i32 %val3, i32 *%ptr
  store volatile i32 %val4, i32 *%ptr
  store volatile i32 %val5, i32 *%ptr
  store volatile i32 %val6, i32 *%ptr
  store volatile i32 %val7, i32 *%ptr
  store volatile i32 %val8, i32 *%ptr
  store volatile i32 %val9, i32 *%ptr
  store volatile i32 %val10, i32 *%ptr
  store volatile i32 %val11, i32 *%ptr
  store volatile i32 %val12, i32 *%ptr
  store volatile i32 %val13, i32 *%ptr
  store volatile i32 %val14, i32 *%ptr
  store volatile i32 %val15, i32 *%ptr

  store volatile i32 %swapped0, i32 *%ptr
  store volatile i32 %swapped1, i32 *%ptr
  store volatile i32 %swapped2, i32 *%ptr
  store volatile i32 %swapped3, i32 *%ptr
  store volatile i32 %swapped4, i32 *%ptr
  store volatile i32 %swapped5, i32 *%ptr
  store volatile i32 %swapped6, i32 *%ptr
  store volatile i32 %swapped7, i32 *%ptr
  store volatile i32 %swapped8, i32 *%ptr
  store volatile i32 %swapped9, i32 *%ptr
  store volatile i32 %swapped10, i32 *%ptr
  store volatile i32 %swapped11, i32 *%ptr
  store volatile i32 %swapped12, i32 *%ptr
  store volatile i32 %swapped13, i32 *%ptr
  store volatile i32 %swapped14, i32 *%ptr
  store volatile i32 %swapped15, i32 *%ptr

  ret void
}
