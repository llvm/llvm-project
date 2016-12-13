; RUN: opt -instcombine -S -o - %s | FileCheck %s
; Tests that we don't crash upon encountering a vector GEP

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Dual = type { %Dual.72, %Partials.73 }
%Dual.72 = type { double, %Partials }
%Partials = type { [2 x double] }
%Partials.73 = type { [2 x %Dual.72] }

; Function Attrs: sspreq
define <8 x i64*> @"julia_axpy!_65480"(%Dual* %arg1, <8 x i64> %arg2) {
top:
; CHECK: %VectorGep14 = getelementptr inbounds %Dual, %Dual* %arg1, <8 x i64> %arg2, i32 1, i32 0, i64 0, i32 1, i32 0, i64 0
  %VectorGep14 = getelementptr inbounds %Dual, %Dual* %arg1, <8 x i64> %arg2, i32 1, i32 0, i64 0, i32 1, i32 0, i64 0
  %0 = bitcast <8 x double*> %VectorGep14 to <8 x i64*>
  ret <8 x i64*> %0
}
