; RUN: opt < %s -function-attrs -S | FileCheck %s
; RUN: opt < %s -passes=function-attrs -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: define void @_Z4foo1Pi(i32* nocapture readnone %a) local_unnamed_addr #0 {
define void @_Z4foo1Pi(i32* nocapture readnone %a) local_unnamed_addr #0 {
entry:
  tail call void @_Z3extv()
  ret void
}

declare void @_Z3extv() local_unnamed_addr

; CHECK: define void @_Z4foo2Pi(i32* nocapture %a) local_unnamed_addr #1 {
define void @_Z4foo2Pi(i32* nocapture %a) local_unnamed_addr #1 {
entry:
  %0 = bitcast i32* %a to i8*
  tail call void @free(i8* %0) #2
  ret void
}

declare void @free(i8* nocapture) local_unnamed_addr #2

; CHECK: define i32 @_Z4foo3Pi(i32* nocapture readonly %a) local_unnamed_addr #3 {
define i32 @_Z4foo3Pi(i32* nocapture readonly %a) local_unnamed_addr #3 {
entry:
  %0 = load i32, i32* %a, align 4
  ret i32 %0
}

; CHECK: define double @_Z4foo4Pd(double* nocapture readonly %a) local_unnamed_addr #1 {
define double @_Z4foo4Pd(double* nocapture readonly %a) local_unnamed_addr #1 {
entry:
  %0 = load double, double* %a, align 8
  %call = tail call double @cos(double %0) #2
  ret double %call
}

declare double @cos(double) local_unnamed_addr #2

; CHECK: define noalias i32* @_Z4foo5Pm(i64* nocapture readonly %a) local_unnamed_addr #1 {
define noalias i32* @_Z4foo5Pm(i64* nocapture readonly %a) local_unnamed_addr #1 {
entry:
  %0 = load i64, i64* %a, align 8
  %call = tail call noalias i8* @malloc(i64 %0) #2
  %1 = bitcast i8* %call to i32*
  ret i32* %1
}

declare noalias i8* @malloc(i64) local_unnamed_addr #2

; CHECK: define noalias i64* @_Z4foo6Pm(i64* nocapture %a) local_unnamed_addr #1 {
define noalias i64* @_Z4foo6Pm(i64* nocapture %a) local_unnamed_addr #1 {
entry:
  %0 = bitcast i64* %a to i8*
  %1 = load i64, i64* %a, align 8
  %call = tail call i8* @realloc(i8* %0, i64 %1) #2
  %2 = bitcast i8* %call to i64*
  ret i64* %2
}

declare noalias i8* @realloc(i8* nocapture, i64) local_unnamed_addr #2

; CHECK: define void @_Z4foo7Pi(i32* %a) local_unnamed_addr #1 {
define void @_Z4foo7Pi(i32* %a) local_unnamed_addr #1 {
entry:
  %isnull = icmp eq i32* %a, null
  br i1 %isnull, label %delete.end, label %delete.notnull

delete.notnull:                                   ; preds = %entry
  %0 = bitcast i32* %a to i8*
  tail call void @_ZdlPv(i8* %0) #5
  br label %delete.end

delete.end:                                       ; preds = %delete.notnull, %entry
  ret void
}

declare void @_ZdlPv(i8*) local_unnamed_addr #4

; CHECK: define void @_Z4foo8Pi(i32* %a) local_unnamed_addr #1 {
define void @_Z4foo8Pi(i32* %a) local_unnamed_addr #1 {
entry:
  %isnull = icmp eq i32* %a, null
  br i1 %isnull, label %delete.end, label %delete.notnull

delete.notnull:                                   ; preds = %entry
  %0 = bitcast i32* %a to i8*
  tail call void @_ZdaPv(i8* %0) #5
  br label %delete.end

delete.end:                                       ; preds = %delete.notnull, %entry
  ret void
}

declare void @_ZdaPv(i8*) local_unnamed_addr #4

attributes #0 = { uwtable }
attributes #1 = { nounwind uwtable }
attributes #2 = { nounwind }
attributes #3 = { norecurse nounwind readonly uwtable }
attributes #4 = { nobuiltin nounwind }
attributes #5 = { builtin nounwind }

; CHECK: attributes #0 = { uwtable }
; CHECK: attributes #1 = { nounwind uwtable }
; CHECK: attributes #2 = { nounwind }
; CHECK: attributes #3 = { norecurse nounwind readonly uwtable willreturn }
; CHECK: attributes #4 = { nobuiltin nounwind }
; CHECK: attributes #5 = { builtin nounwind }

