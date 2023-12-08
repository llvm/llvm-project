; RUN: opt %loadPolly -polly-codegen < %s

; We just check that this compilation does not crash.

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-a0:0-n32"

define void @init() nounwind {
entry:
  %hi.129.reg2mem = alloca i64
  br label %for.body

for.cond5.preheader:                              ; preds = %for.body
  br label %for.body7

for.body:                                         ; preds = %for.body, %entry
  br i1 undef, label %for.body, label %for.cond5.preheader

for.body7:                                        ; preds = %for.body7, %for.cond5.preheader
  %i.128 = phi i64 [ 0, %for.cond5.preheader ], [ %inc07, %for.body7 ]
  %inc07 = add nsw i64 %i.128, 1
  store i64 undef, ptr %hi.129.reg2mem
  br i1 false, label %for.body7, label %for.end18

for.end18:                                        ; preds = %for.body7
  ret void
}
