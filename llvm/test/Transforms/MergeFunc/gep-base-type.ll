; RUN: opt -passes=mergefunc -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; These should not be merged, the type of the GEP pointer argument does not have
; the same stride.

%"struct1" = type <{ i8*, i32, [4 x i8] }>
%"struct2" = type { i8*, { i64, i64 } }

define internal %struct2* @Ffunc(%struct2* %P, i64 %i) {
; CHECK-LABEL: @Ffunc(
; CHECK-NEXT: getelementptr
; CHECK-NEXT: getelementptr
; CHECK-NEXT: getelementptr
; CHECK-NEXT: getelementptr
; CHECK-NEXT: getelementptr
; CHECK-NEXT: getelementptr
; CHECK-NEXT: ret
  %1 = getelementptr inbounds %"struct2", %"struct2"* %P, i64 %i
  %2 = getelementptr inbounds %"struct2", %"struct2"* %P, i64 %i
  %3 = getelementptr inbounds %"struct2", %"struct2"* %P, i64 %i
  %4 = getelementptr inbounds %"struct2", %"struct2"* %P, i64 %i
  %5 = getelementptr inbounds %"struct2", %"struct2"* %P, i64 %i
  %6 = getelementptr inbounds %"struct2", %"struct2"* %P, i64 %i
  ret %struct2* %6
}


define internal %struct1* @Gfunc(%struct1* %P, i64 %i) {
; CHECK-LABEL: @Gfunc(
; CHECK-NEXT: getelementptr
; CHECK-NEXT: getelementptr
; CHECK-NEXT: getelementptr
; CHECK-NEXT: getelementptr
; CHECK-NEXT: getelementptr
; CHECK-NEXT: getelementptr
; CHECK-NEXT: ret
  %1 = getelementptr inbounds %"struct1", %"struct1"* %P, i64 %i
  %2 = getelementptr inbounds %"struct1", %"struct1"* %P, i64 %i
  %3 = getelementptr inbounds %"struct1", %"struct1"* %P, i64 %i
  %4 = getelementptr inbounds %"struct1", %"struct1"* %P, i64 %i
  %5 = getelementptr inbounds %"struct1", %"struct1"* %P, i64 %i
  %6 = getelementptr inbounds %"struct1", %"struct1"* %P, i64 %i
  ret %struct1* %6
}

