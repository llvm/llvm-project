; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1
; RUN: opt < %s -analyze -basicaa -da
;
; CHECK: da analyze - consistent input [S S]!
; CHECK: da analyze - confused!
; CHECK: da analyze - input [* *]!
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n8:16:32-S64"
target triple = "thumbv7--linux-gnueabi"

define void @f(i32** %a, i32 %n) align 2 {
for.preheader:
  %t.0 = ashr exact i32 %n, 3
  br label %for.body.1

for.body.1:
  %i.1 = phi i32 [ %t.5, %for.inc ], [ 0, %for.preheader ]
  %i.2 = phi i32 [ %i.5, %for.inc ], [ %t.0, %for.preheader ]
  br i1 undef, label %for.inc, label %for.body.2

for.body.2:
  %i.3 = phi i32 [ %t.1, %for.body.2 ], [ %i.1, %for.body.1 ]
  %t.1 = add i32 %i.3, 1
  %t.2 = load i32*, i32** %a, align 4
  %t.3 = getelementptr inbounds i32, i32* %t.2, i32 %i.3
  %t.4 = load i32, i32* %t.3, align 4
  br i1 undef, label %for.inc, label %for.body.2

for.inc:
  %i.4 = phi i32 [ %i.2, %for.body.1 ], [ %i.2, %for.body.2 ]
  %t.5 = add i32 %i.1, %i.4
  %i.5 = add i32 %i.2, -1
  br i1 undef, label %for.exit, label %for.body.1

for.exit:
  ret void
}
