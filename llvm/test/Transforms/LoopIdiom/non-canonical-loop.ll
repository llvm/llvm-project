; RUN: opt -S -loop-idiom < %s
; Don't crash
; PR13892

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test(i32* %currMB) nounwind uwtable {
entry:
  br i1 undef, label %start.exit, label %if.then.i

if.then.i:                                        ; preds = %entry
  unreachable

start.exit:                       ; preds = %entry
  indirectbr i8* undef, [label %0, label %for.bodyprime]

; <label>:0                                       ; preds = %start.exit
  unreachable

for.bodyprime:                                    ; preds = %for.bodyprime, %start.exit
  %i.057375 = phi i32 [ 0, %start.exit ], [ %1, %for.bodyprime ]
  %arrayidx8prime = getelementptr inbounds i32, i32* %currMB, i32 %i.057375
  store i32 0, i32* %arrayidx8prime, align 4
  %1 = add i32 %i.057375, 1
  %cmp5prime = icmp slt i32 %1, 4
  br i1 %cmp5prime, label %for.bodyprime, label %for.endprime

for.endprime:                                     ; preds = %for.bodyprime
  br label %for.body23prime

for.body23prime:                                  ; preds = %for.body23prime, %for.endprime
  br label %for.body23prime
}
