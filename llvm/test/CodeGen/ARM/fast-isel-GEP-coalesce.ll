; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=armv7-apple-darwin | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-darwin | FileCheck %s --check-prefix=THUMB

%struct.A = type { i32, [2 x [2 x i32]], i8, [3 x [3 x [3 x i32]]] }
%struct.B = type { i32, [2 x [2 x [2 x %struct.A]]] }

@arr = common global [2 x [2 x [2 x [2 x [2 x i32]]]]] zeroinitializer, align 4
@A = common global [3 x [3 x %struct.A]] zeroinitializer, align 4
@B = common global [2 x [2 x [2 x %struct.B]]] zeroinitializer, align 4

define i32* @t1() nounwind {
entry:
; ARM: t1
; THUMB: t1
  %addr = alloca i32*, align 4
  store i32* getelementptr inbounds ([2 x [2 x [2 x [2 x [2 x i32]]]]]* @arr, i32 0, i32 1, i32 1, i32 1, i32 1, i32 1), i32** %addr, align 4
; ARM: add r0, r0, #124
; THUMB: adds r0, #124
  %0 = load i32** %addr, align 4
  ret i32* %0
}

define i32* @t2() nounwind {
entry:
; ARM: t2
; THUMB: t2
  %addr = alloca i32*, align 4
  store i32* getelementptr inbounds ([3 x [3 x %struct.A]]* @A, i32 0, i32 2, i32 2, i32 3, i32 1, i32 2, i32 2), i32** %addr, align 4
; ARM: movw r1, #1148
; ARM: add r0, r0, r1
; THUMB: addw r0, r0, #1148
  %0 = load i32** %addr, align 4
  ret i32* %0
}

define i32* @t3() nounwind {
entry:
; ARM: t3
; THUMB: t3
  %addr = alloca i32*, align 4
  store i32* getelementptr inbounds ([3 x [3 x %struct.A]]* @A, i32 0, i32 0, i32 1, i32 1, i32 0, i32 1), i32** %addr, align 4
; ARM: add r0, r0, #140
; THUMB: adds r0, #140
  %0 = load i32** %addr, align 4
  ret i32* %0
}

define i32* @t4() nounwind {
entry:
; ARM: t4
; THUMB: t4
  %addr = alloca i32*, align 4
  store i32* getelementptr inbounds ([2 x [2 x [2 x %struct.B]]]* @B, i32 0, i32 0, i32 0, i32 1, i32 1, i32 0, i32 0, i32 1, i32 3, i32 1, i32 2, i32 1), i32** %addr, align 4
; ARM-NOT: movw r{{[0-9]}}, #1060
; ARM-NOT: add r{{[0-9]}}, r{{[0-9]}}, #4
; ARM-NOT: add r{{[0-9]}}, r{{[0-9]}}, #132
; ARM-NOT: add r{{[0-9]}}, r{{[0-9]}}, #24
; ARM-NOT: add r{{[0-9]}}, r{{[0-9]}}, #36
; ARM-NOT: add r{{[0-9]}}, r{{[0-9]}}, #24
; ARM-NOT: add r{{[0-9]}}, r{{[0-9]}}, #4
; ARM: movw r{{[0-9]}}, #1284
; THUMB: addw r{{[0-9]}}, r{{[0-9]}}, #1284
  %0 = load i32** %addr, align 4
  ret i32* %0
}
