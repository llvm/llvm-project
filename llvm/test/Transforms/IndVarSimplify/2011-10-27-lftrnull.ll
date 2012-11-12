; RUN: opt < %s -indvars -S | FileCheck %s
; rdar://10359193: assert "IndVar type must match IVInit type"

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-darwin"

; CHECK: @test
; CHECK: if.end.i126:
; CHECK: %exitcond = icmp ne i8* %incdec.ptr.i, getelementptr (i8* null, i32 undef)
define void @test() nounwind {
entry:
  br label %while.cond

while.cond:
  br i1 undef, label %while.end, label %while.body

while.body:                                       ; preds = %while.cond
  br i1 undef, label %if.then165, label %while.cond

if.then165:                                       ; preds = %while.body
  br i1 undef, label %while.cond, label %for.body.lr.ph.i81

for.body.lr.ph.i81:                               ; preds = %if.then165
  br label %for.body.i86

for.body.i86:                                     ; preds = %for.end.i129, %for.body.lr.ph.i81
  %cmp196.i = icmp ult i32 0, undef
  br i1 %cmp196.i, label %for.body21.lr.ph.i, label %for.end.i129

for.body21.lr.ph.i:                               ; preds = %for.body.i86
  br label %for.body21.i

for.body21.i:
  %destYPixelPtr.010.i = phi i8* [ null, %for.body21.lr.ph.i ], [ %incdec.ptr.i, %if.end.i126 ]
  %x.09.i = phi i32 [ 0, %for.body21.lr.ph.i ], [ %inc.i125, %if.end.i126 ]
  br i1 undef, label %if.end.i126, label %if.else.i124

if.else.i124:                                     ; preds = %for.body21.i
  store i8 undef, i8* %destYPixelPtr.010.i, align 1
  br label %if.end.i126

if.end.i126:                                      ; preds = %if.else.i124, %for.body21.i
  %incdec.ptr.i = getelementptr inbounds i8* %destYPixelPtr.010.i, i32 1
  %inc.i125 = add i32 %x.09.i, 1
  %cmp19.i = icmp ult i32 %inc.i125, undef
  br i1 %cmp19.i, label %for.body21.i, label %for.end.i129

for.end.i129:                                     ; preds = %if.end.i126, %for.body.i86
  br i1 undef, label %for.body.i86, label %while.cond

while.end:                                        ; preds = %while.cond
  br label %bail

bail:                                             ; preds = %while.end, %lor.lhs.false44, %lor.lhs.false41, %if.end29, %if.end
  unreachable

return:                                           ; preds = %lor.lhs.false20, %lor.lhs.false12, %lor.lhs.false, %entry
  ret void
}
