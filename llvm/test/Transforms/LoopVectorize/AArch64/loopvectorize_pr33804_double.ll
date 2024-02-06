; RUN: opt -passes=loop-vectorize -S < %s | FileCheck %s

; These tests check that we don't crash if vectorizer decides to cast
; a double value to be stored into a pointer type or vice-versa.

; This test checks when a double value is stored into a pointer type.

; ModuleID = 'bugpoint-reduced-simplified.bc'
source_filename = "bugpoint-output-26dbd81.bc"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

%struct.CvNode1D = type { double, ptr }

; CHECK-LABEL: @cvCalcEMD2
; CHECK: vector.body
; CHECK: store <{{[0-9]+}} x ptr>
define void @cvCalcEMD2(ptr %dst) {
entry:
  br label %for.body14.i.i

for.body14.i.i:                                   ; preds = %for.body14.i.i, %entry
  %i.1424.i.i = phi i32 [ %inc21.i.i, %for.body14.i.i ], [ 0, %entry ]
  %arrayidx15.i.i1427 = getelementptr inbounds %struct.CvNode1D, ptr %dst, i32 %i.1424.i.i
  store double 0xC415AF1D80000000, ptr %arrayidx15.i.i1427, align 4
  %next19.i.i = getelementptr inbounds %struct.CvNode1D, ptr %dst, i32 %i.1424.i.i, i32 1
  store ptr %dst, ptr %next19.i.i, align 4
  %inc21.i.i = add nuw nsw i32 %i.1424.i.i, 1
  %exitcond438.i.i = icmp eq i32 %inc21.i.i, 0
  br i1 %exitcond438.i.i, label %for.end22.i.i, label %for.body14.i.i

for.end22.i.i:                                    ; preds = %for.body14.i.i
  unreachable
}

; This test checks when a pointer value is stored into a double type.

%struct.CvNode1D2 = type { ptr, double }

; CHECK-LABEL: @cvCalcEMD2_2
; CHECK: vector.body
; CHECK: store <{{[0-9]+}} x double>
define void @cvCalcEMD2_2(ptr %dst) {
entry:
  br label %for.body14.i.i

for.body14.i.i:                                   ; preds = %for.body14.i.i, %entry
  %i.1424.i.i = phi i32 [ %inc21.i.i, %for.body14.i.i ], [ 0, %entry ]
  %next19.i.i = getelementptr inbounds %struct.CvNode1D2, ptr %dst, i32 %i.1424.i.i, i32 0
  store ptr %dst, ptr %next19.i.i, align 4
  %arrayidx15.i.i1427 = getelementptr inbounds %struct.CvNode1D2, ptr %dst, i32 %i.1424.i.i
  %val.i.i = getelementptr inbounds %struct.CvNode1D2, ptr %arrayidx15.i.i1427, i32 0, i32 1
  store double 0xC415AF1D80000000, ptr %val.i.i, align 4
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
define void @cvCalcEMD3(ptr %src, ptr %dst) {
entry:
  br label %for.body14.i.i

for.body14.i.i:                                   ; preds = %for.body14.i.i, %entry
  %i.1424.i.i = phi i32 [ %inc21.i.i, %for.body14.i.i ], [ 0, %entry ]
  %arrayidx15.i.i1427 = getelementptr inbounds %struct.CvNode1D, ptr %src, i32 %i.1424.i.i
  %load_d = load double, ptr %arrayidx15.i.i1427, align 4
  %next19.i.i = getelementptr inbounds %struct.CvNode1D, ptr %src, i32 %i.1424.i.i, i32 1
  %load_p = load ptr, ptr %next19.i.i, align 4
  %dst.ptr = getelementptr inbounds %struct.CvNode1D, ptr %dst, i32 %i.1424.i.i
  %dst.ptr.1 = getelementptr inbounds %struct.CvNode1D, ptr %dst, i32 %i.1424.i.i, i32 1
  store double %load_d, ptr %dst.ptr, align 4
  store ptr %load_p, ptr %dst.ptr.1, align 4
  %inc21.i.i = add nuw nsw i32 %i.1424.i.i, 1
  %exitcond438.i.i = icmp eq i32 %inc21.i.i, 0
  br i1 %exitcond438.i.i, label %for.end22.i.i, label %for.body14.i.i

for.end22.i.i:                                    ; preds = %for.body14.i.i
  unreachable
}

; This test check for pointer to integer casting with load instructions.

; CHECK-LABEL: @cvCalcEMD3_2
; CHECK: vector.body
; CHECK: ptrtoint <{{[0-9]+}} x ptr>
define void @cvCalcEMD3_2(ptr %src, ptr %dst) {
entry:
  br label %for.body14.i.i

for.body14.i.i:                                   ; preds = %for.body14.i.i, %entry
  %i.1424.i.i = phi i32 [ %inc21.i.i, %for.body14.i.i ], [ 0, %entry ]
  %next19.i.i = getelementptr inbounds %struct.CvNode1D2, ptr %src, i32 %i.1424.i.i, i32 0
  %load_p = load ptr, ptr %next19.i.i, align 4
  %arrayidx15.i.i1427 = getelementptr inbounds %struct.CvNode1D2, ptr %src, i32 %i.1424.i.i
  %val.i.i = getelementptr inbounds %struct.CvNode1D2, ptr %arrayidx15.i.i1427, i32 0, i32 1
  %load_d = load double, ptr %val.i.i, align 4
  %dst.ptr = getelementptr inbounds %struct.CvNode1D, ptr %dst, i32 %i.1424.i.i
  %dst.ptr.1 = getelementptr inbounds %struct.CvNode1D, ptr %dst, i32 %i.1424.i.i, i32 1
  store double %load_d, ptr %dst.ptr, align 4
  store ptr %load_p, ptr %dst.ptr.1, align 4
  %inc21.i.i = add nuw nsw i32 %i.1424.i.i, 1
  %exitcond438.i.i = icmp eq i32 %inc21.i.i, 0
  br i1 %exitcond438.i.i, label %for.end22.i.i, label %for.body14.i.i

for.end22.i.i:                                    ; preds = %for.body14.i.i
  unreachable
}
