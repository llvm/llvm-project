; RUN: opt -passes=loop-vectorize -S < %s | FileCheck %s

; These tests check that we don't crash if vectorizer decides to cast
; a float value to be stored into a pointer type or vice-versa.

; This test checks when a float value is stored into a pointer type.

; ModuleID = 'bugpoint-reduced-simplified.bc'
source_filename = "bugpoint-output-26dbd81.bc"
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-unknown-linux-gnueabihf"

%struct.CvNode1D = type { float, ptr }

; CHECK-LABEL: @cvCalcEMD2
; CHECK: vector.body
; CHECK: store <{{[0-9]+}} x ptr>
define void @cvCalcEMD2(ptr %dst) {
entry:
  br label %for.body14.i.i

for.body14.i.i:                                   ; preds = %for.body14.i.i, %entry
  %i.1424.i.i = phi i32 [ %inc21.i.i, %for.body14.i.i ], [ 0, %entry ]
  %arrayidx15.i.i1427 = getelementptr inbounds %struct.CvNode1D, ptr %dst, i32 %i.1424.i.i
  store float 0xC415AF1D80000000, ptr %arrayidx15.i.i1427, align 4
  %next19.i.i = getelementptr inbounds %struct.CvNode1D, ptr %dst, i32 %i.1424.i.i, i32 1
  store ptr %dst, ptr %next19.i.i, align 4
  %inc21.i.i = add nuw nsw i32 %i.1424.i.i, 1
  %exitcond438.i.i = icmp eq i32 %inc21.i.i, 0
  br i1 %exitcond438.i.i, label %for.end22.i.i, label %for.body14.i.i

for.end22.i.i:                                    ; preds = %for.body14.i.i
  unreachable
}

; This test checks when a pointer value is stored into a float type.

%struct.CvNode1D2 = type { ptr, float }

; CHECK-LABEL: @cvCalcEMD2_2
; CHECK: vector.body
; CHECK: store <{{[0-9]+}} x float>
define void @cvCalcEMD2_2(ptr %dst) {
entry:
  br label %for.body14.i.i

for.body14.i.i:                                   ; preds = %for.body14.i.i, %entry
  %i.1424.i.i = phi i32 [ %inc21.i.i, %for.body14.i.i ], [ 0, %entry ]
  %next19.i.i = getelementptr inbounds %struct.CvNode1D2, ptr %dst, i32 %i.1424.i.i, i32 0
  store ptr %dst, ptr %next19.i.i, align 4
  %arrayidx15.i.i1427 = getelementptr inbounds %struct.CvNode1D2, ptr %dst, i32 %i.1424.i.i
  %val.i.i = getelementptr inbounds %struct.CvNode1D2, ptr %arrayidx15.i.i1427, i32 0, i32 1
  store float 0xC415AF1D80000000, ptr %val.i.i, align 4
  %inc21.i.i = add nuw nsw i32 %i.1424.i.i, 1
  %exitcond438.i.i = icmp eq i32 %inc21.i.i, 0
  br i1 %exitcond438.i.i, label %for.end22.i.i, label %for.body14.i.i

for.end22.i.i:                                    ; preds = %for.body14.i.i
  unreachable
}

; This test checks for the intoptr conversions with load instructions.

; CHECK-LABEL: @cvCalcEMD3
; CHECK: vector.body
; CHECK: inttoptr <{{[0-9]+}} x i32>
define void @cvCalcEMD3(ptr %src, ptr %dst) {
entry:
  br label %for.body14.i.i

for.body14.i.i:                                   ; preds = %for.body14.i.i, %entry
  %i.1424.i.i = phi i32 [ %inc21.i.i, %for.body14.i.i ], [ 0, %entry ]
  %arrayidx15.i.i1427 = getelementptr inbounds %struct.CvNode1D, ptr %src, i32 %i.1424.i.i
  %loadf = load float, ptr %arrayidx15.i.i1427, align 4
  %next19.i.i = getelementptr inbounds %struct.CvNode1D, ptr %src, i32 %i.1424.i.i, i32 1
  %loadp = load ptr, ptr %next19.i.i, align 4
  %dst.ptr = getelementptr inbounds %struct.CvNode1D, ptr %dst, i32 %i.1424.i.i
  %dst.ptr.1 = getelementptr inbounds %struct.CvNode1D, ptr %dst, i32 %i.1424.i.i, i32 1
  store float %loadf, ptr %dst.ptr, align 4
  store ptr %loadp, ptr %dst.ptr.1, align 4
  %inc21.i.i = add nuw nsw i32 %i.1424.i.i, 1
  %exitcond438.i.i = icmp eq i32 %inc21.i.i, 0
  br i1 %exitcond438.i.i, label %for.end22.i.i, label %for.body14.i.i

for.end22.i.i:                                    ; preds = %for.body14.i.i
  unreachable
}

; This test checks for ptrtoint conversions with load instructions.

; CHECK-LABEL: @cvCalcEMD3_2
; CHECK: vector.body
; CHECK: ptrtoint <{{[0-9]+}} x ptr>
define void @cvCalcEMD3_2(ptr %src, ptr %dst) {
entry:
  br label %for.body14.i.i

for.body14.i.i:                                   ; preds = %for.body14.i.i, %entry
  %i.1424.i.i = phi i32 [ %inc21.i.i, %for.body14.i.i ], [ 0, %entry ]
  %next19.i.i = getelementptr inbounds %struct.CvNode1D2, ptr %src, i32 %i.1424.i.i, i32 0
  %loadp = load ptr, ptr %next19.i.i, align 4
  %arrayidx15.i.i1427 = getelementptr inbounds %struct.CvNode1D2, ptr %src, i32 %i.1424.i.i
  %val.i.i = getelementptr inbounds %struct.CvNode1D2, ptr %arrayidx15.i.i1427, i32 0, i32 1
  %loadf = load float, ptr %val.i.i, align 4
  %dst.ptr = getelementptr inbounds %struct.CvNode1D, ptr %dst, i32 %i.1424.i.i
  %dst.ptr.1 = getelementptr inbounds %struct.CvNode1D, ptr %dst, i32 %i.1424.i.i, i32 1
  store float %loadf, ptr %dst.ptr, align 4
  store ptr %loadp, ptr %dst.ptr.1, align 4
  %inc21.i.i = add nuw nsw i32 %i.1424.i.i, 1
  %exitcond438.i.i = icmp eq i32 %inc21.i.i, 0
  br i1 %exitcond438.i.i, label %for.end22.i.i, label %for.body14.i.i

for.end22.i.i:                                    ; preds = %for.body14.i.i
  unreachable
}
