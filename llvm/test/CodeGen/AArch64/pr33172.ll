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
  %wide.load8281058.3 = load i64, i64* bitcast (float* getelementptr inbounds ([200 x float], [200 x float]* @main.b, i64 0, i64 12) to i64*), align 8
  %wide.load8291059.3 = load i64, i64* bitcast (float* getelementptr inbounds ([200 x float], [200 x float]* @main.b, i64 0, i64 14) to i64*), align 8
  store i64 %wide.load8281058.3, i64* bitcast (float* getelementptr inbounds ([200 x float], [200 x float]* @main.x, i64 0, i64 12) to i64*), align 8
  store i64 %wide.load8291059.3, i64* bitcast (float* getelementptr inbounds ([200 x float], [200 x float]* @main.x, i64 0, i64 14) to i64*), align 8
  %wide.load8281058.4 = load i64, i64* bitcast (float* getelementptr inbounds ([200 x float], [200 x float]* @main.b, i64 0, i64 16) to i64*), align 8
  %wide.load8291059.4 = load i64, i64* bitcast (float* getelementptr inbounds ([200 x float], [200 x float]* @main.b, i64 0, i64 18) to i64*), align 8
  store i64 %wide.load8281058.4, i64* bitcast (float* getelementptr inbounds ([200 x float], [200 x float]* @main.x, i64 0, i64 16) to i64*), align 8
  store i64 %wide.load8291059.4, i64* bitcast (float* getelementptr inbounds ([200 x float], [200 x float]* @main.x, i64 0, i64 18) to i64*), align 8
  tail call void @llvm.memset.p0i8.i64(i8* align 8 bitcast ([200 x float]* @main.b to i8*), i8 0, i64 undef, i1 false) #2
  unreachable
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }
