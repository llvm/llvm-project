; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s
;
; Test that a reload of a LOCGR/SELGR operand can be folded into a LOC
; instruction.

declare i64 @foo()
declare i32 @foo32()

; Check that conditional loads of spilled values can use LOCG rather than LOCGR.
define void @f0(i64 *%ptr0, i64 *%dstPtr) {
; CHECK-LABEL: f0:
; CHECK: brasl %r14, foo@PLT
; CHECK: locglh {{.*}}           # 8-byte Folded Reload
; CHECK: br %r14
  %ptr1 = getelementptr i64, i64 *%ptr0, i64 2
  %ptr2 = getelementptr i64, i64 *%ptr0, i64 4
  %ptr3 = getelementptr i64, i64 *%ptr0, i64 6
  %ptr4 = getelementptr i64, i64 *%ptr0, i64 8
  %ptr5 = getelementptr i64, i64 *%ptr0, i64 10
  %ptr6 = getelementptr i64, i64 *%ptr0, i64 12
  %ptr7 = getelementptr i64, i64 *%ptr0, i64 14
  %ptr8 = getelementptr i64, i64 *%ptr0, i64 16
  %ptr9 = getelementptr i64, i64 *%ptr0, i64 18

  %val0 = load i64, i64 *%ptr0
  %val1 = load i64, i64 *%ptr1
  %val2 = load i64, i64 *%ptr2
  %val3 = load i64, i64 *%ptr3
  %val4 = load i64, i64 *%ptr4
  %val5 = load i64, i64 *%ptr5
  %val6 = load i64, i64 *%ptr6
  %val7 = load i64, i64 *%ptr7
  %val8 = load i64, i64 *%ptr8
  %val9 = load i64, i64 *%ptr9

  %ret = call i64 @foo()

  %add0 = add i64 %ret, %val0
  %add1 = add i64 %add0, %val1
  %add2 = add i64 %add1, %val2
  %add3 = add i64 %add2, %val3
  %add4 = add i64 %add3, %val4
  %add5 = add i64 %add4, %val5
  %add6 = add i64 %add5, %val6
  %add7 = add i64 %add6, %val7
  %add8 = add i64 %add7, %val8

  %cond = icmp eq i64 %add7, %add8
  %res = select i1 %cond, i64 %add8, i64 %val9

  store i64 %res, i64* %dstPtr
  ret void
}

; Check that conditional loads of spilled values can use LOC rather than LOCR.
define void @f1(i32 *%ptr0, i32 *%dstPtr) {
; CHECK-LABEL: f1:
; CHECK: brasl %r14, foo32@PLT
; CHECK: loclh {{.*}}            # 4-byte Folded Reload
; CHECK: br %r14
  %ptr1 = getelementptr i32, i32 *%ptr0, i32 2
  %ptr2 = getelementptr i32, i32 *%ptr0, i32 4
  %ptr3 = getelementptr i32, i32 *%ptr0, i32 6
  %ptr4 = getelementptr i32, i32 *%ptr0, i32 8
  %ptr5 = getelementptr i32, i32 *%ptr0, i32 10
  %ptr6 = getelementptr i32, i32 *%ptr0, i32 12
  %ptr7 = getelementptr i32, i32 *%ptr0, i32 14
  %ptr8 = getelementptr i32, i32 *%ptr0, i32 16
  %ptr9 = getelementptr i32, i32 *%ptr0, i32 18

  %val0 = load i32, i32 *%ptr0
  %val1 = load i32, i32 *%ptr1
  %val2 = load i32, i32 *%ptr2
  %val3 = load i32, i32 *%ptr3
  %val4 = load i32, i32 *%ptr4
  %val5 = load i32, i32 *%ptr5
  %val6 = load i32, i32 *%ptr6
  %val7 = load i32, i32 *%ptr7
  %val8 = load i32, i32 *%ptr8
  %val9 = load i32, i32 *%ptr9

  %ret = call i32 @foo32()

  %add0 = add i32 %ret, %val0
  %add1 = add i32 %add0, %val1
  %add2 = add i32 %add1, %val2
  %add3 = add i32 %add2, %val3
  %add4 = add i32 %add3, %val4
  %add5 = add i32 %add4, %val5
  %add6 = add i32 %add5, %val6
  %add7 = add i32 %add6, %val7
  %add8 = add i32 %add7, %val8

  %cond = icmp eq i32 %add7, %add8
  %res = select i1 %cond, i32 %add8, i32 %val9

  store i32 %res, i32* %dstPtr
  ret void
}
