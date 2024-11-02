; RUN: opt -loop-idiom -S < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; Don't crash inside DependenceAnalysis
; PR14219
define void @test1(i64* %iwork, i64 %x)  {
bb0:
  %mul116 = mul nsw i64 %x, %x
  %incdec.ptr6.sum175 = add i64 42, %x
  %arrayidx135 = getelementptr inbounds i64, i64* %iwork, i64 %incdec.ptr6.sum175
  br label %bb1
bb1:
  %storemerge4226 = phi i64 [ 0, %bb0 ], [ %inc139, %bb1 ]
  store i64 1, i64* %arrayidx135, align 8
  %incdec.ptr6.sum176 = add i64 %mul116, %storemerge4226
  %arrayidx137 = getelementptr inbounds i64, i64* %iwork, i64 %incdec.ptr6.sum176
  store i64 1, i64* %arrayidx137, align 8
  %inc139 = add nsw i64 %storemerge4226, 1
  %cmp131 = icmp sgt i64 %storemerge4226, 42
  br i1 %cmp131, label %bb2, label %bb1
bb2:
  ret void
}

