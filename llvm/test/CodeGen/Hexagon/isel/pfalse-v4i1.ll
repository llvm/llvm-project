; RUN: llc -march=hexagon -debug-only=isel 2>&1 < %s - | FileCheck %s

; CHECK: [[R0:%[0-9]+]]:intregs = A2_tfrsi 0
; CHECK-NEXT: predregs = C2_tfrrp killed [[R0]]:intregs

define fastcc i16 @test(ptr %0, { <4 x i32>, <4 x i1> } %1, <4 x i1> %2) {
Entry:
  %3 = alloca [16 x i8], i32 0, align 16
  %4 = alloca [16 x i8], i32 0, align 16
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %4, align 16
  store <4 x i32> <i32 5, i32 6, i32 7, i32 8>, ptr %3, align 16
  %5 = load <4 x i32>, ptr %4, align 16
  %6 = load <4 x i32>, ptr %3, align 16
  %7 = call { <4 x i32>, <4 x i1> } @llvm.sadd.with.overflow.v4i32(<4 x i32> %5, <4 x i32> %6)
  %8 = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> %2)
  br i1 %8, label %OverflowFail, label %OverflowOk

OverflowFail:                                     ; preds = %Entry
  store volatile i32 0, ptr null, align 4
    unreachable

OverflowOk:                                       ; preds = %Entry
  %9 = extractvalue { <4 x i32>, <4 x i1> } %7, 0
    store <4 x i32> %9, ptr %0, align 16
      ret i16 0
      }

declare { <4 x i32>, <4 x i1> } @llvm.sadd.with.overflow.v4i32(<4 x i32>, <4 x i32>) #0
declare i1 @llvm.vector.reduce.or.v4i1(<4 x i1>) #0
