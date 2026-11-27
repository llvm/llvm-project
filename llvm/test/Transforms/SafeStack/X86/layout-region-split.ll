; Regression test for safestack layout. Used to fail with asan.
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -passes='require<libcall-lowering-info>,safe-stack' -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

define void @f() safestack {
; CHECK-LABEL: define void @f
entry:
; CHECK:  %[[USP:.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr
; CHECK:   getelementptr i8, ptr %[[USP]], i64 -224

  %x0 = alloca i8, align 16
  %x1 = alloca i8, align 16
  %x2 = alloca i8, align 16
  %x3 = alloca i8, align 16
  %x4 = alloca i8, align 16
  %x5 = alloca i8, align 16
  %x6 = alloca i8, align 16
  %x7 = alloca i8, align 16
  %x8 = alloca i8, align 16
  %x9 = alloca i8, align 16
  %x10 = alloca i8, align 16
  %x11 = alloca i8, align 16
  %x12 = alloca i8, align 16
  %x13 = alloca i8, align 16
  %y0 = alloca i8, align 2
  %y1 = alloca i8, align 2
  %y2 = alloca i8, align 2
  %y3 = alloca i8, align 2
  %y4 = alloca i8, align 2
  %y5 = alloca i8, align 2
  %y6 = alloca i8, align 2
  %y7 = alloca i8, align 2
  %y8 = alloca i8, align 2

; CHECK: getelementptr i8, ptr %[[USP]], i64 -16
  call void @capture8(ptr %x0)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -32
  call void @capture8(ptr %x1)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -48
  call void @capture8(ptr %x2)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -64
  call void @capture8(ptr %x3)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -80
  call void @capture8(ptr %x4)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -96
  call void @capture8(ptr %x5)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -112
  call void @capture8(ptr %x6)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -128
  call void @capture8(ptr %x7)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -144
  call void @capture8(ptr %x8)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -160
  call void @capture8(ptr %x9)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -176
  call void @capture8(ptr %x10)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -192
  call void @capture8(ptr %x11)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -208
  call void @capture8(ptr %x12)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -224
  call void @capture8(ptr %x13)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -2
  call void @capture8(ptr %y0)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -4
  call void @capture8(ptr %y1)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -6
  call void @capture8(ptr %y2)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -8
  call void @capture8(ptr %y3)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -10
  call void @capture8(ptr %y4)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -12
  call void @capture8(ptr %y5)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -14
  call void @capture8(ptr %y6)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -18
  call void @capture8(ptr %y7)
; CHECK: getelementptr i8, ptr %[[USP]], i64 -20
  call void @capture8(ptr %y8)

  ret void
}

declare void @capture8(ptr)
