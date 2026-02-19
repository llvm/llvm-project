; RUN: opt -S -scalarizer -dxil-op-lower %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64-v48:16:16-v96:32:32-v192:64:64"
target triple = "dxilv1.3-pc-shadermodel6.3-compute"

; The definition of the custom type should be added
; CHECK: %dx.types.fouri32 = type { i32, i32, i32, i32 }

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define hidden noundef <4 x i32> @_Z4testb(i1 noundef %p1) {
entry:
  %p1.addr = alloca i32, align 4
  %storedv = zext i1 %p1 to i32
  store i32 %storedv, ptr %p1.addr, align 4
  %0 = load i32, ptr %p1.addr, align 4
  %loadedv = trunc i32 %0 to i1
  %1 = load i32, ptr %p1.addr, align 4
  %loadedv1 = trunc i32 %1 to i1

  ; CHECK: call %dx.types.fouri32 @dx.op.waveActiveBallot(i32 116, i1 %loadedv1)

  %2 = call { i32, i32, i32, i32 } @llvm.dx.wave.ballot.i32(i1 %loadedv1)
  %3 = extractvalue { i32, i32, i32, i32 } %2, 0
  %4 = insertelement <4 x i32> poison, i32 %3, i32 0
  %5 = extractvalue { i32, i32, i32, i32 } %2, 1
  %6 = insertelement <4 x i32> %4, i32 %5, i32 1
  %7 = extractvalue { i32, i32, i32, i32 } %2, 2
  %8 = insertelement <4 x i32> %6, i32 %7, i32 2
  %9 = extractvalue { i32, i32, i32, i32 } %2, 3
  %10 = insertelement <4 x i32> %8, i32 %9, i32 3

  ; CHECK-NOT: ret %dx.types.fouri32
  ; CHECK: ret <4 x i32>
  ret <4 x i32> %10
}

declare { i32, i32, i32, i32 } @llvm.dx.wave.ballot.i32(i1)
