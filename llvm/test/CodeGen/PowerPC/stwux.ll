target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"
; RUN: llc < %s | FileCheck %s

@multvec_i = external unnamed_addr global [100 x i32], align 4

define fastcc void @subs_STMultiExceptIntern() nounwind {
entry:
  br i1 undef, label %while.body.lr.ph, label %return

while.body.lr.ph:                                 ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %if.end12, %while.body.lr.ph
  %i.0240 = phi i32 [ -1, %while.body.lr.ph ], [ %i.1, %if.end12 ]
  br i1 undef, label %if.end12, label %if.then

if.then:                                          ; preds = %while.body
  br label %if.end12

if.end12:                                         ; preds = %if.then, %while.body
  %i.1 = phi i32 [ %i.0240, %while.body ], [ undef, %if.then ]
  br i1 undef, label %while.body, label %while.end

while.end:                                        ; preds = %if.end12
  br i1 undef, label %return, label %if.end15

if.end15:                                         ; preds = %while.end
  %idxprom.i.i230 = sext i32 %i.1 to i64
  %arrayidx18 = getelementptr inbounds [100 x i32]* @multvec_i, i64 0, i64 %idxprom.i.i230
  store i32 0, i32* %arrayidx18, align 4
  br i1 undef, label %while.body21, label %while.end90

while.body21:                                     ; preds = %if.end15
  unreachable

while.end90:                                      ; preds = %if.end15
  store i32 0, i32* %arrayidx18, align 4
  br label %return

return:                                           ; preds = %while.end90, %while.end, %entry
  ret void

; CHECK: @subs_STMultiExceptIntern
; CHECK: stwux
}

