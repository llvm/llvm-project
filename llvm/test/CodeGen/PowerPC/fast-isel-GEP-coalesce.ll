; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s --check-prefix=PPC64
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-ibm-aix-xcoff -mcpu=pwr7 | FileCheck %s --check-prefix=PPC64

%struct.A = type { i32, [2 x [2 x i32]], i8, [3 x [3 x [3 x i32]]] }
%struct.B = type { i32, [2 x [2 x [2 x %struct.A]]] }

@arr = common global [2 x [2 x [2 x [2 x [2 x i32]]]]] zeroinitializer, align 4
@A = common global [3 x [3 x %struct.A]] zeroinitializer, align 4
@B = common global [2 x [2 x [2 x %struct.B]]] zeroinitializer, align 4

define ptr @t1() nounwind {
entry:
; PPC64: t1
  %addr = alloca ptr, align 4
  store ptr getelementptr inbounds ([2 x [2 x [2 x [2 x [2 x i32]]]]], ptr @arr, i32 0, i32 1, i32 1, i32 1, i32 1, i32 1), ptr %addr, align 4
; PPC64: addi {{[0-9]+}}, {{[0-9]+}}, 124
  %0 = load ptr, ptr %addr, align 4
  ret ptr %0
}

define ptr @t2() nounwind {
entry:
; PPC64: t2
  %addr = alloca ptr, align 4
  store ptr getelementptr inbounds ([3 x [3 x %struct.A]], ptr @A, i32 0, i32 2, i32 2, i32 3, i32 1, i32 2, i32 2), ptr %addr, align 4
; PPC64: addi {{[0-9]+}}, {{[0-9]+}}, 1148
  %0 = load ptr, ptr %addr, align 4
  ret ptr %0
}

define ptr @t3() nounwind {
entry:
; PPC64: t3
  %addr = alloca ptr, align 4
  store ptr getelementptr inbounds ([3 x [3 x %struct.A]], ptr @A, i32 0, i32 0, i32 1, i32 1, i32 0, i32 1), ptr %addr, align 4
; PPC64: addi {{[0-9]+}}, {{[0-9]+}}, 140
  %0 = load ptr, ptr %addr, align 4
  ret ptr %0
}

define ptr @t4() nounwind {
entry:
; PPC64: t4
  %addr = alloca ptr, align 4
  store ptr getelementptr inbounds ([2 x [2 x [2 x %struct.B]]], ptr @B, i32 0, i32 0, i32 0, i32 1, i32 1, i32 0, i32 0, i32 1, i32 3, i32 1, i32 2, i32 1), ptr %addr, align 4
; PPC64: addi {{[0-9]+}}, {{[0-9]+}}, 1284
  %0 = load ptr, ptr %addr, align 4
  ret ptr %0
}
