; RUN: llc < %s | FileCheck %s

; CHECK-LABEL: pr33172
; CHECK: ldp
; CHECK: stp

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios10.3.0"

@main.b = external global [200 x float], align 8
@main.x = external global [200 x float], align 8

; Function Attrs: nounwind ssp
define void @pr33172() local_unnamed_addr  {
entry:
  %wide.load8281058.3 = load i64, ptr getelementptr inbounds ([200 x float], ptr @main.b, i64 0, i64 12), align 8
  %wide.load8291059.3 = load i64, ptr getelementptr inbounds ([200 x float], ptr @main.b, i64 0, i64 14), align 8
  store i64 %wide.load8281058.3, ptr getelementptr inbounds ([200 x float], ptr @main.x, i64 0, i64 12), align 8
  store i64 %wide.load8291059.3, ptr getelementptr inbounds ([200 x float], ptr @main.x, i64 0, i64 14), align 8
  %wide.load8281058.4 = load i64, ptr getelementptr inbounds ([200 x float], ptr @main.b, i64 0, i64 16), align 8
  %wide.load8291059.4 = load i64, ptr getelementptr inbounds ([200 x float], ptr @main.b, i64 0, i64 18), align 8
  store i64 %wide.load8281058.4, ptr getelementptr inbounds ([200 x float], ptr @main.x, i64 0, i64 16), align 8
  store i64 %wide.load8291059.4, ptr getelementptr inbounds ([200 x float], ptr @main.x, i64 0, i64 18), align 8
  tail call void @llvm.memset.p0.i64(ptr align 8 @main.b, i8 0, i64 undef, i1 false) #2
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1) #1

attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }
