; RUN: opt < %s -loop-simplify -S
; PR8702
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-freebsd9.0"

declare void @foo(i32 %x)

define fastcc void @inm_merge() nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %while.cond36.i, %entry
  br i1 undef, label %do.body, label %for.body

for.body:                                         ; preds = %for.cond
  br i1 undef, label %while.cond36.i, label %if.end44

if.end44:                                         ; preds = %for.body
  %call49 = call fastcc i32 @inm_get_source()
  br i1 undef, label %if.end54, label %for.cond64

if.end54:                                         ; preds = %if.end44
  br label %while.cond36.i

while.cond36.i:                                   ; preds = %if.end54, %for.body
  br label %for.cond

for.cond64:                                       ; preds = %if.end88, %for.cond64, %if.end44
  %error.161 = phi i32 [ %error.161, %for.cond64 ], [ %error.161, %if.end88 ], [ %call49, %if.end44 ]
  call void @foo(i32 %error.161)
  br i1 undef, label %for.cond64, label %if.end88

if.end88:                                         ; preds = %for.cond64
  br i1 undef, label %for.cond64, label %if.end98

if.end98:                                         ; preds = %if.end88
  unreachable

do.body:                                          ; preds = %for.cond
  unreachable
}

declare fastcc i32 @inm_get_source() nounwind
