; RUN: opt %loadPolly -polly-allow-differing-element-types -polly-codegen -S \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s

; CHECK: %polly.access.cast.global.load = bitcast %struct.hoge* %global.load to i32*
; CHECK: %polly.access.global.load = getelementptr i32, i32* %polly.access.cast.global.load, i64 0
; CHECK: %polly.access.global.load.load = load i32, i32* %polly.access.global.load

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.hoge = type { i32, double }

@global = external global %struct.hoge*, align 8

; Function Attrs: nounwind uwtable
define void @widget(double* %A) #0 {
bb:
  br label %bb4

bb4:
  %tmp = load %struct.hoge*, %struct.hoge** @global
  %tmp5 = getelementptr inbounds %struct.hoge, %struct.hoge* %tmp, i64 0, i32 0
  %tmp6 = load i32, i32* %tmp5
  %tmp7 = getelementptr inbounds %struct.hoge, %struct.hoge* %tmp, i64 0, i32 1
  %tmp8 = load double, double* %tmp7
  store double %tmp8, double* %A
  br i1 false, label %bb11, label %bb12

bb11:
  br label %bb12

bb12:
  %tmp13 = phi float [ undef, %bb11 ], [ 1.000000e+00, %bb4 ]
  ret void
}

