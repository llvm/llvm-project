; RUN: opt -loop-unroll -S -mtriple arm-none-eabi -mcpu=cortex-m7 %s | FileCheck %s -check-prefix=UNROLL

; This test is meant to check that this loop is unrolled into three iterations.

; UNROLL-LABEL: @test
; UNROLL: load i32, i32*
; UNROLL: load i32, i32*
; UNROLL: load i32, i32*
; UNROLL-NOT: load i32, i32*

define void @test(i32* %x, i32 %n) {
entry:
  %sub = add nsw i32 %n, -1
  %rem = srem i32 %sub, 4
  %cmp7 = icmp sgt i32 %rem, 0
  br i1 %cmp7, label %while.body, label %while.end

while.body:                                       ; preds = %entry, %if.end
  %x.addr.09 = phi i32* [ %incdec.ptr, %if.end ], [ %x, %entry ]
  %n.addr.08 = phi i32 [ %dec, %if.end ], [ %rem, %entry ]
  %0 = load i32, i32* %x.addr.09, align 4
  %cmp1 = icmp slt i32 %0, 10
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %while.body
  store i32 0, i32* %x.addr.09, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %while.body
  %incdec.ptr = getelementptr inbounds i32, i32* %x.addr.09, i32 1
  %dec = add nsw i32 %n.addr.08, -1
  %cmp = icmp sgt i32 %dec, 0
  br i1 %cmp, label %while.body, label %while.end

while.end:                                        ; preds = %if.end, %entry
  ret void
}

