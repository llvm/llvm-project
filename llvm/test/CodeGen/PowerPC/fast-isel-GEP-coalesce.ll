; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s --check-prefix=PPC64
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-ibm-aix-xcoff -mcpu=pwr7 | FileCheck %s --check-prefix=PPC64

%struct.A = type { i32, [2 x [2 x i32]], i8, [3 x [3 x [3 x i32]]] }
%struct.B = type { i32, [2 x [2 x [2 x %struct.A]]] }

@arr = common global [2 x [2 x [2 x [2 x [2 x i32]]]]] zeroinitializer, align 4
@A = common global [3 x [3 x %struct.A]] zeroinitializer, align 4
@B = common global [2 x [2 x [2 x %struct.B]]] zeroinitializer, align 4

define i32* @t1() nounwind {
entry:
; PPC64: t1
  %addr = alloca i32*, align 4
  store i32* getelementptr inbounds ([2 x [2 x [2 x [2 x [2 x i32]]]]], [2 x [2 x [2 x [2 x [2 x i32]]]]]* @arr, i32 0, i32 1, i32 1, i32 1, i32 1, i32 1), i32** %addr, align 4
; PPC64: addi {{[0-9]+}}, {{[0-9]+}}, 124
  %0 = load i32*, i32** %addr, align 4
  ret i32* %0
}

define i32* @t2() nounwind {
entry:
; PPC64: t2
  %addr = alloca i32*, align 4
  store i32* getelementptr inbounds ([3 x [3 x %struct.A]], [3 x [3 x %struct.A]]* @A, i32 0, i32 2, i32 2, i32 3, i32 1, i32 2, i32 2), i32** %addr, align 4
; PPC64: addi {{[0-9]+}}, {{[0-9]+}}, 1148
  %0 = load i32*, i32** %addr, align 4
  ret i32* %0
}

define i32* @t3() nounwind {
entry:
; PPC64: t3
  %addr = alloca i32*, align 4
  store i32* getelementptr inbounds ([3 x [3 x %struct.A]], [3 x [3 x %struct.A]]* @A, i32 0, i32 0, i32 1, i32 1, i32 0, i32 1), i32** %addr, align 4
; PPC64: addi {{[0-9]+}}, {{[0-9]+}}, 140
  %0 = load i32*, i32** %addr, align 4
  ret i32* %0
}

define i32* @t4() nounwind {
entry:
; PPC64: t4
  %addr = alloca i32*, align 4
  store i32* getelementptr inbounds ([2 x [2 x [2 x %struct.B]]], [2 x [2 x [2 x %struct.B]]]* @B, i32 0, i32 0, i32 0, i32 1, i32 1, i32 0, i32 0, i32 1, i32 3, i32 1, i32 2, i32 1), i32** %addr, align 4
; PPC64: addi {{[0-9]+}}, {{[0-9]+}}, 1284
  %0 = load i32*, i32** %addr, align 4
  ret i32* %0
}
