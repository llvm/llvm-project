; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s
;
; This caused an lnt crash at some point, just verify it will run through.
;
; CHECK-LABEL: polly.merge_new_and_old:
; CHECK-NEXT:    br label %for.body.6
;
; CHECK-LABEL: for.body.6:
; CHECK-NEXT:    %i.14 = phi i32 [ undef, %for.body.6 ], [ 0, %polly.merge_new_and_old ]
;
@recd = external hidden global [255 x i32], align 16

define void @rsdec_204(i8* %data_in) {
entry:
  br i1 undef, label %if.then, label %for.body

if.then:                                          ; preds = %entry
  unreachable

for.body:                                         ; preds = %for.body, %entry
  %i.05 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i8, i8* %data_in, i64 0
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %arrayidx2 = getelementptr inbounds [255 x i32], [255 x i32]* @recd, i64 0, i64 0
  store i32 %conv, i32* %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.05, 1
  br i1 false, label %for.body, label %for.body.6

for.body.6:                                       ; preds = %for.body.6, %for.body
  %i.14 = phi i32 [ undef, %for.body.6 ], [ 0, %for.body ]
  br i1 undef, label %for.body.6, label %for.body.16

for.body.16:                                      ; preds = %for.body.16, %for.body.6
  br i1 undef, label %for.body.16, label %for.body.29

for.body.29:                                      ; preds = %for.body.29, %for.body.16
  br i1 undef, label %for.body.29, label %for.end.38

for.end.38:                                       ; preds = %for.body.29
  unreachable
}
