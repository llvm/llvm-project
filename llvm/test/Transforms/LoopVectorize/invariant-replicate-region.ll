; RUN: opt -p loop-vectorize -force-vector-width=4 -S %s | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx14.0.0"

; Test for https://github.com/llvm/llvm-project/issues/109510.
define i32 @test_invariant_replicate_region(i32 %x, i1 %c) {
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  br i1 %c, label %then, label %loop.latch

then:
  %rem.1 = urem i32 10, %x
  br label %loop.latch

loop.latch:
  %res = phi i32 [ 0, %loop.header ], [ %rem.1, %then ]
  %iv.next = add i32 %iv, 1
  %ec = icmp eq i32 %iv, 99
  br i1 %ec, label %exit, label %loop.header

exit:
  ret i32 %res
}
