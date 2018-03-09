; RUN: opt -loop-vectorize -S < %s | FileCheck %s

; These tests check that we don't crash if vectorizer decides to cast
; a double value to be stored into a pointer type or vice-versa.

; This test checks when a double value is stored into a pointer type.

; ModuleID = 'bugpoint-reduced-simplified.bc'
source_filename = "bugpoint-output-26dbd81.bc"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

%struct.CvNode1D = type { double, %struct.CvNode1D* }

; CHECK-LABEL: @cvCalcEMD2
; CHECK: vector.body
; CHECK: store <{{[0-9]+}} x %struct.CvNode1D*>
define void @cvCalcEMD2() local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  br label %for.body14.i.i

for.body14.i.i:                                   ; preds = %for.body14.i.i, %entry
  %i.1424.i.i = phi i32 [ %inc21.i.i, %for.body14.i.i ], [ 0, %entry ]
  %arrayidx15.i.i1427 = getelementptr inbounds %struct.CvNode1D, %struct.CvNode1D* undef, i32 %i.1424.i.i
  %val.i.i = getelementptr inbounds %struct.CvNode1D, %struct.CvNode1D* %arrayidx15.i.i1427, i32 0, i32 0
  store double 0xC415AF1D80000000, double* %val.i.i, align 4
  %next19.i.i = getelementptr inbounds %struct.CvNode1D, %struct.CvNode1D* undef, i32 %i.1424.i.i, i32 1
  store %struct.CvNode1D* undef, %struct.CvNode1D** %next19.i.i, align 4
  %inc21.i.i = add nuw nsw i32 %i.1424.i.i, 1
  %exitcond438.i.i = icmp eq i32 %inc21.i.i, 0
  br i1 %exitcond438.i.i, label %for.end22.i.i, label %for.body14.i.i

for.end22.i.i:                                    ; preds = %for.body14.i.i
  unreachable
}

; This test checks when a pointer value is stored into a double type.

%struct.CvNode1D2 = type { %struct.CvNode1D2*, double }

; CHECK-LABEL: @cvCalcEMD2_2
; CHECK: vector.body
; CHECK: store <{{[0-9]+}} x double>
define void @cvCalcEMD2_2() local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  br label %for.body14.i.i

for.body14.i.i:                                   ; preds = %for.body14.i.i, %entry
  %i.1424.i.i = phi i32 [ %inc21.i.i, %for.body14.i.i ], [ 0, %entry ]
  %next19.i.i = getelementptr inbounds %struct.CvNode1D2, %struct.CvNode1D2* undef, i32 %i.1424.i.i, i32 0
  store %struct.CvNode1D2* undef, %struct.CvNode1D2** %next19.i.i, align 4
  %arrayidx15.i.i1427 = getelementptr inbounds %struct.CvNode1D2, %struct.CvNode1D2* undef, i32 %i.1424.i.i
  %val.i.i = getelementptr inbounds %struct.CvNode1D2, %struct.CvNode1D2* %arrayidx15.i.i1427, i32 0, i32 1
  store double 0xC415AF1D80000000, double* %val.i.i, align 4
  %inc21.i.i = add nuw nsw i32 %i.1424.i.i, 1
  %exitcond438.i.i = icmp eq i32 %inc21.i.i, 0
  br i1 %exitcond438.i.i, label %for.end22.i.i, label %for.body14.i.i

for.end22.i.i:                                    ; preds = %for.body14.i.i
  unreachable
}

; This test check for integer to pointer casting with load instructions.

; CHECK-LABEL: @cvCalcEMD3
; CHECK: vector.body
; CHECK: inttoptr <{{[0-9]+}} x i64>
define void @cvCalcEMD3() local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  br label %for.body14.i.i

for.body14.i.i:                                   ; preds = %for.body14.i.i, %entry
  %i.1424.i.i = phi i32 [ %inc21.i.i, %for.body14.i.i ], [ 0, %entry ]
  %arrayidx15.i.i1427 = getelementptr inbounds %struct.CvNode1D, %struct.CvNode1D* undef, i32 %i.1424.i.i
  %val.i.i = getelementptr inbounds %struct.CvNode1D, %struct.CvNode1D* %arrayidx15.i.i1427, i32 0, i32 0
  %load_d = load double, double* %val.i.i, align 4
  %next19.i.i = getelementptr inbounds %struct.CvNode1D, %struct.CvNode1D* undef, i32 %i.1424.i.i, i32 1
  %load_p = load %struct.CvNode1D*, %struct.CvNode1D** %next19.i.i, align 4
  %inc21.i.i = add nuw nsw i32 %i.1424.i.i, 1
  %exitcond438.i.i = icmp eq i32 %inc21.i.i, 0
  br i1 %exitcond438.i.i, label %for.end22.i.i, label %for.body14.i.i

for.end22.i.i:                                    ; preds = %for.body14.i.i
  unreachable
}

; This test check for pointer to integer casting with load instructions.

; CHECK-LABEL: @cvCalcEMD3_2
; CHECK: vector.body
; CHECK: ptrtoint <{{[0-9]+}} x %struct.CvNode1D2*>
define void @cvCalcEMD3_2() local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  br label %for.body14.i.i

for.body14.i.i:                                   ; preds = %for.body14.i.i, %entry
  %i.1424.i.i = phi i32 [ %inc21.i.i, %for.body14.i.i ], [ 0, %entry ]
  %next19.i.i = getelementptr inbounds %struct.CvNode1D2, %struct.CvNode1D2* undef, i32 %i.1424.i.i, i32 0
  %load_p = load %struct.CvNode1D2*, %struct.CvNode1D2** %next19.i.i, align 4
  %arrayidx15.i.i1427 = getelementptr inbounds %struct.CvNode1D2, %struct.CvNode1D2* undef, i32 %i.1424.i.i
  %val.i.i = getelementptr inbounds %struct.CvNode1D2, %struct.CvNode1D2* %arrayidx15.i.i1427, i32 0, i32 1
  %load_d = load double, double* %val.i.i, align 4
  %inc21.i.i = add nuw nsw i32 %i.1424.i.i, 1
  %exitcond438.i.i = icmp eq i32 %inc21.i.i, 0
  br i1 %exitcond438.i.i, label %for.end22.i.i, label %for.body14.i.i

for.end22.i.i:                                    ; preds = %for.body14.i.i
  unreachable
}

declare i32 @__gxx_personality_v0(...)

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }

