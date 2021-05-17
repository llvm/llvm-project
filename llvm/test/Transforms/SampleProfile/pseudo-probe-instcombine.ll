; RUN: opt -passes=instcombine -available-load-scan-limit=2 -S < %s | FileCheck %s

%struct.nonbonded = type { [2 x %struct.CompAtom*], [2 x %struct.CompAtomExt*], [2 x %struct.CompAtom*], [2 x %class.Vector*], [2 x %class.Vector*], [2 x i32], %class.Vector, double*, double*, %class.ComputeNonbondedWorkArrays*, %class.Pairlists*, i32, i32, double, double, i32, i32, i32, i32 }
%struct.CompAtomExt = type { i32 }
%struct.CompAtom = type { %class.Vector, float, i16, i8, i8 }
%class.Vector = type { double, double, double }
%class.ComputeNonbondedWorkArrays = type { %class.ResizeArray, %class.ResizeArray.0, %class.ResizeArray, %class.ResizeArray, %class.ResizeArray, %class.ResizeArray, %class.ResizeArray, %class.ResizeArray, %class.ResizeArray, %class.ResizeArray, %class.ResizeArray, %class.ResizeArray, %class.ResizeArray, %class.ResizeArray, %class.ResizeArray, %class.ResizeArray, %class.ResizeArray, %class.ResizeArray.2, %class.ResizeArray.2 }
%class.ResizeArray.0 = type { i32 (...)**, %class.ResizeArrayRaw.1* }
%class.ResizeArrayRaw.1 = type <{ double*, i8*, i32, i32, i32, float, i32, [4 x i8] }>
%class.ResizeArray = type { i32 (...)**, %class.ResizeArrayRaw* }
%class.ResizeArrayRaw = type <{ i16*, i8*, i32, i32, i32, float, i32, [4 x i8] }>
%class.ResizeArray.2 = type { i32 (...)**, %class.ResizeArrayRaw.3* }
%class.ResizeArrayRaw.3 = type <{ %class.Vector*, i8*, i32, i32, i32, float, i32, [4 x i8] }>
%class.Pairlists = type { i16*, i32, i32 }

define dso_local void @merge(%struct.nonbonded* nocapture readonly %params) local_unnamed_addr align 2 {
;; Check the minPart4 and minPart assignments are merged.
; CHECK-LABEL: @merge(
; CHECK-COUNT-1: getelementptr inbounds %struct.nonbonded, %struct.nonbonded* %params, i64 0, i32 16
; CHECK-NOT: getelementptr inbounds %struct.nonbonded, %struct.nonbonded* %params, i64 0, i32 16
entry:
  %savePairlists3 = getelementptr inbounds %struct.nonbonded, %struct.nonbonded* %params, i64 0, i32 11
  %0 = load i32, i32* %savePairlists3, align 8
  %usePairlists4 = getelementptr inbounds %struct.nonbonded, %struct.nonbonded* %params, i64 0, i32 12
  %1 = load i32, i32* %usePairlists4, align 4
  %tobool54.not = icmp eq i32 %0, 0
  br i1 %tobool54.not, label %lor.lhs.false55, label %if.end109

lor.lhs.false55:                                  ; preds = %entry
  %tobool56.not = icmp eq i32 %1, 0
  br i1 %tobool56.not, label %if.end109, label %if.end109.thread

if.end109.thread:                                 ; preds = %lor.lhs.false55
  %minPart4 = getelementptr inbounds %struct.nonbonded, %struct.nonbonded* %params, i64 0, i32 16
  %2 = load i32, i32* %minPart4, align 4
  call void @llvm.pseudoprobe(i64 -6172701105289426098, i64 2, i32 0, i64 -1)
  br label %if.then138

if.end109:                                        ; preds = %lor.lhs.false55, %entry
  %minPart = getelementptr inbounds %struct.nonbonded, %struct.nonbonded* %params, i64 0, i32 16
  %3 = load i32, i32* %minPart, align 4
  call void @llvm.pseudoprobe(i64 -6172701105289426098, i64 3, i32 0, i64 -1)
  %tobool116.not = icmp eq i32 %1, 0
  br i1 %tobool116.not, label %if.then117, label %if.then138

if.then117:                                       ; preds = %if.end109
  ret void

if.then138:                                       ; preds = %if.end109.thread, %if.end109
  %4 = phi i32 [ %2, %if.end109.thread ], [ %3, %if.end109 ]
  %tobool139.not = icmp eq i32 %4, 0
  br i1 %tobool139.not, label %if.else147, label %if.then140

if.then140:                                       ; preds = %if.then138
  ret void

if.else147:                                       ; preds = %if.then138
  ret void
}

define i32 @load(i32* nocapture %a, i32* nocapture %b) {
;; Check the last store is deleted.
; CHECK-LABEL: @load(
; CHECK-NEXT:  %1 = getelementptr inbounds i32, i32* %a, i64 1
; CHECK-NEXT:  %2 = load i32, i32* %1, align 8
; CHECK-NEXT:  %3 = getelementptr inbounds i32, i32* %b, i64 1
; CHECK-NEXT:       store i32 %2, i32* %3, align 8
; CHECK-NEXT:    call void @llvm.pseudoprobe(i64 5116412291814990879, i64 1, i32 0, i64 -1)
; CHECK-NEXT:    ret i32 %[[#]]
  %1 = getelementptr inbounds i32, i32* %a, i32 1
  %2 = load i32, i32* %1, align 8
  %3 = getelementptr inbounds i32, i32* %b, i32 1
       store i32 %2, i32* %3, align 8
  %4 = getelementptr inbounds i32, i32* %b, i32 1
  call void @llvm.pseudoprobe(i64 5116412291814990879, i64 1, i32 0, i64 -1)
  %5 = load i32, i32* %4, align 8
  ret i32 %5
}

define void @dse(i32* %p) {
;; Check the first store is deleted.
; CHECK-LABEL: @dse(
; CHECK-NEXT:    call void @llvm.pseudoprobe(i64 5116412291814990879, i64 1, i32 0, i64 -1)
; CHECK-NEXT:    store i32 0, i32* [[P:%.*]], align 4
; CHECK-NEXT:    ret void
  store i32 0, i32* %p
  call void @llvm.pseudoprobe(i64 5116412291814990879, i64 1, i32 0, i64 -1)
  store i32 0, i32* %p
  ret void
}

; Function Attrs: inaccessiblememonly nounwind willreturn
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #0

attributes #0 = { inaccessiblememonly nounwind willreturn }
