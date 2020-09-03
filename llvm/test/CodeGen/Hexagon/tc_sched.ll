; RUN: llc -march=hexagon -mcpu=hexagonv67t < %s | FileCheck %s

; A simple test case for the tiny core instruction latency information.

; CHECK-LABEL: test
; CHECK-DAG: [[REG1:r([0-9]+)]] = memw([[REG0:r[0-9]+]]+#0)
; CHECK-DAG: [[REG2:r([0-9]+)]] = memw([[REG0]]+#4)
; CHECK-NEXT: }
; CHECK: {
; CHECK: {
; CHECK-NEXT: = add([[REG2]],[[REG1]])

define i32 @test(i32* nocapture readonly %p) local_unnamed_addr #0 {
entry:
  %incdec.ptr = getelementptr inbounds i32, i32* %p, i32 1
  %0 = load i32, i32* %p, align 4
  %incdec.ptr1 = getelementptr inbounds i32, i32* %p, i32 2
  %1 = load i32, i32* %incdec.ptr, align 4
  %incdec.ptr2 = getelementptr inbounds i32, i32* %p, i32 3
  %2 = load i32, i32* %incdec.ptr1, align 4
  %3 = load i32, i32* %incdec.ptr2, align 4
  %add = add nsw i32 %1, %0
  %add4 = add nsw i32 %3, %2
  %mul = mul nsw i32 %add4, %add
  ret i32 %mul
}

; CHECK-LABEL: test1
; CHECK-DAG: [[REG4:r([0-9]+)]] = memw([[REG3:r[0-9]+]]+#0)
; CHECK-DAG: [[REG5:r([0-9]+)]] = memw([[REG3]]+#4)
; CHECK-NEXT: }
; CHECK: {
; CHECK: {
; CHECK-NEXT: [[REG7:r([0-9]+)]] = add([[REG5]],[[REG4]])
; CHECK: }
; CHECK-NEXT: {
; CHECK-NEXT: = sub([[REG7]]

define i32 @test1(i32* nocapture readonly %p) local_unnamed_addr #0 {
entry:
  %incdec.ptr = getelementptr inbounds i32, i32* %p, i32 1
  %0 = load i32, i32* %p, align 4
  %incdec.ptr1 = getelementptr inbounds i32, i32* %p, i32 2
  %1 = load i32, i32* %incdec.ptr, align 4
  %incdec.ptr2 = getelementptr inbounds i32, i32* %p, i32 3
  %2 = load i32, i32* %incdec.ptr1, align 4
  %3 = load i32, i32* %incdec.ptr2, align 4
  %add4.neg = add i32 %1, %0
  %add = sub i32 %add4.neg, %2
  %sub = sub i32 %add, %3
  ret i32 %sub
}

; Test that multiplies are not placed in the same packet.
; CHECK-LABEL: test2
; CHECK: = mpyi
; CHECK: }
; CHECK: = mpyi
; CHECK: }
; CHECK: = mpyi
; CHECK: }
; CHECK: = mpyi

define i32 @test2(i32* nocapture readonly %p) local_unnamed_addr #1 {
entry:
  %incdec.ptr = getelementptr inbounds i32, i32* %p, i32 1
  %0 = load i32, i32* %p, align 4
  %incdec.ptr1 = getelementptr inbounds i32, i32* %p, i32 2
  %1 = load i32, i32* %incdec.ptr, align 4
  %incdec.ptr2 = getelementptr inbounds i32, i32* %p, i32 3
  %2 = load i32, i32* %incdec.ptr1, align 4
  %3 = load i32, i32* %incdec.ptr2, align 4
  %mul = mul nsw i32 %1, %0
  %mul4 = mul nsw i32 %3, %2
  %mul5 = mul nsw i32 %3, %0
  %mul6 = mul nsw i32 %2, %1
  %call = tail call i32 @foo(i32 %mul, i32 %mul4, i32 %mul5, i32 %mul6) #3
  ret i32 %call
}

declare i32 @foo(i32, i32, i32, i32) local_unnamed_addr #2

