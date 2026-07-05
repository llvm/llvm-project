; Test 32-bit byteswaps from memory to registers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @llvm.bswap.i32(i32 %a)

; Check LRV with no displacement.
define i32 @f1(ptr %src) {
; CHECK-LABEL: f1:
; CHECK: lrv %r2, 0(%r2)
; CHECK: br %r14
  %a = load i32, ptr %src
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check the high end of the aligned LRV range.
define i32 @f2(ptr %src) {
; CHECK-LABEL: f2:
; CHECK: lrv %r2, 524284(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 131071
  %a = load i32, ptr %ptr
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f3(ptr %src) {
; CHECK-LABEL: f3:
; CHECK: agfi %r2, 524288
; CHECK: lrv %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 131072
  %a = load i32, ptr %ptr
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check the high end of the negative aligned LRV range.
define i32 @f4(ptr %src) {
; CHECK-LABEL: f4:
; CHECK: lrv %r2, -4(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -1
  %a = load i32, ptr %ptr
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check the low end of the LRV range.
define i32 @f5(ptr %src) {
; CHECK-LABEL: f5:
; CHECK: lrv %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -131072
  %a = load i32, ptr %ptr
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f6(ptr %src) {
; CHECK-LABEL: f6:
; CHECK: agfi %r2, -524292
; CHECK: lrv %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -131073
  %a = load i32, ptr %ptr
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
  %ptr = inttoptr i64 %add2 to ptr
  %a = load i32, ptr %ptr
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Test a case where we spill the source of at least one LRVR.  We want
; to use LRV if possible.
define i32 @f8(ptr %ptr0) {
; CHECK-LABEL: f8:
; CHECK: lrv {{%r[0-9]+}}, 16{{[04]}}(%r15)
; CHECK: br %r14

  %val0 = call i32 @foo()
  %val1 = call i32 @foo()
  %val2 = call i32 @foo()
  %val3 = call i32 @foo()
  %val4 = call i32 @foo()
  %val5 = call i32 @foo()
  %val6 = call i32 @foo()
  %val7 = call i32 @foo()
  %val8 = call i32 @foo()
  %val9 = call i32 @foo()

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

  %ret1 = add i32 %swapped0, %swapped1
  %ret2 = add i32 %ret1, %swapped2
  %ret3 = add i32 %ret2, %swapped3
  %ret4 = add i32 %ret3, %swapped4
  %ret5 = add i32 %ret4, %swapped5
  %ret6 = add i32 %ret5, %swapped6
  %ret7 = add i32 %ret6, %swapped7
  %ret8 = add i32 %ret7, %swapped8
  %ret9 = add i32 %ret8, %swapped9

  ret i32 %ret9
}

declare i32 @foo()
