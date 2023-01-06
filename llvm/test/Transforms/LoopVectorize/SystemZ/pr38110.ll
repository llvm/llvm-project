; RUN: opt -passes='loop-vectorize' -mcpu=z13 -force-vector-width=2 -S < %s | FileCheck %s
;
; Forcing VF=2 to trigger vector code gen
;
; This is a test case to exercise more cases in truncateToMinimalBitWidths().
; Test passes if vector code is generated w/o hitting llvm_unreachable().
;
; Performing minimal check in the output to ensure the loop is actually
; vectorized.
;
; CHECK: vector.body

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

define void @test(i32 zeroext %width, ptr nocapture %row, i16 zeroext %src, ptr nocapture readonly %dst) {
entry:
  %cmp10 = icmp eq i32 %width, 0
  br i1 %cmp10, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %conv1 = zext i16 %src to i32
  br label %for.body

for.body:                                         ; preds = %for.inc, %for.body.lr.ph
  %i.012 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
  %sp.011 = phi ptr [ %row, %for.body.lr.ph ], [ %incdec.ptr, %for.inc ]
  %0 = load i8, ptr %sp.011, align 1
  %conv = zext i8 %0 to i32
  %cmp2 = icmp eq i32 %conv, %conv1
  br i1 %cmp2, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %1 = load i16, ptr %dst, align 2
  %conv4 = trunc i16 %1 to i8
  store i8 %conv4, ptr %sp.011, align 1
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nuw i32 %i.012, 1
  %incdec.ptr = getelementptr inbounds i8, ptr %sp.011, i64 1
  %exitcond = icmp eq i32 %inc, %width
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.inc
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}
