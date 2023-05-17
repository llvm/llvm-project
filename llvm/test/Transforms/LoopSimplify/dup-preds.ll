; RUN: opt -passes=loop-simplify -S %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64le-unknown-linux"

define fastcc void @do_update_md(ptr nocapture readonly %x) #0 {
entry:
  br i1 undef, label %if.end365, label %lor.lhs.false134

lor.lhs.false134:                                 ; preds = %entry
  br i1 undef, label %lor.lhs.false138, label %if.end365

lor.lhs.false138:                                 ; preds = %lor.lhs.false134
  br i1 undef, label %lor.lhs.false142, label %if.end365

lor.lhs.false142:                                 ; preds = %lor.lhs.false138
  br i1 undef, label %for.body276.lr.ph, label %if.end365

for.body276.lr.ph:                                ; preds = %lor.lhs.false142
  switch i16 undef, label %if.then288 [
    i16 4, label %for.body344
    i16 2, label %for.body344
  ]

if.then288:                                       ; preds = %for.body276.lr.ph
  br label %for.body305

for.body305:                                      ; preds = %for.body305, %if.then288
  br label %for.body305

for.body344:                                      ; preds = %for.body344, %for.body276.lr.ph, %for.body276.lr.ph
  %indvar = phi i64 [ %indvar.next, %for.body344 ], [ 0, %for.body276.lr.ph ], [ 0, %for.body276.lr.ph ]
  %indvars.iv552 = phi i64 [ %indvars.iv.next553, %for.body344 ], [ 0, %for.body276.lr.ph ], [ 0, %for.body276.lr.ph ]
  %indvars.iv.next553 = add nuw nsw i64 %indvars.iv552, 1
  %indvar.next = add i64 %indvar, 1
  br label %for.body344

; CHECK-LABEL: @do_update_md
; CHECK: %indvars.iv552 = phi i64 [ %indvars.iv.next553, %for.body344 ], [ 0, %for.body344.preheader ]
; CHECK: ret

if.end365:                                        ; preds = %lor.lhs.false142, %lor.lhs.false138, %lor.lhs.false134, %entry
  ret void
}

attributes #0 = { nounwind }

