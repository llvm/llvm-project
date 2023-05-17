; RUN: llc -mtriple=thumbv7-windows-msvc -frame-pointer=all %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7-windows-msvc -fast-isel -frame-pointer=all %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7-windows-msvc %s -o - | FileCheck %s --check-prefix=NOFP
; RUN: llc -mtriple=thumbv7-windows-msvc -fast-isel %s -o - | FileCheck %s --check-prefix=NOFP

@env2 = common dso_local global ptr null, align 8

define dso_local void @bar() {
  %1 = call ptr @llvm.sponentry()
  %2 = load ptr, ptr @env2, align 8
  %3 = call i32 @_setjmpex(ptr %2, ptr %1) #2
  ret void
}

; CHECK: bar:
; CHECK: push.w  {r11, lr}
; CHECK: mov     r11, sp
; CHECK: add.w   r1, r11, #8
; CHECK: bl      _setjmpex

; NOFP: bar:
; NOFP: push.w  {r11, lr}
; NOFP: add     r1, sp, #8
; NOFP: bl      _setjmpex

define dso_local void @foo(ptr) {
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca [100 x i32], align 4
  store ptr %0, ptr %2, align 8
  %5 = call ptr @llvm.sponentry()
  %6 = load ptr, ptr %2, align 8
  %7 = call i32 @_setjmpex(ptr %6, ptr %5)
  store i32 %7, ptr %3, align 4
  ret void
}

; CHECK: foo:
; CHECK: push.w  {r11, lr}
; CHECK: mov     r11, sp
; CHECK: sub     sp, #416
; CHECK: add.w   r1, r11, #8
; CHECK: bl      _setjmpex

; NOFP: foo:
; NOFP: push.w  {r11, lr}
; NOFP: sub     sp, #416
; NOFP: add     r1, sp, #424
; NOFP: bl      _setjmpex

define dso_local void @var_args(ptr, ...) {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  call void @llvm.va_start(ptr %3)
  %4 = load ptr, ptr %3, align 8
  %5 = getelementptr inbounds i8, ptr %4, i64 8
  store ptr %5, ptr %3, align 8
  %6 = load i32, ptr %4, align 8
  call void @llvm.va_end(ptr %3)
  %7 = call ptr @llvm.sponentry()
  %8 = load ptr, ptr @env2, align 8
  %9 = call i32 @_setjmpex(ptr %8, ptr %7) #3
  ret void
}

; CHECK: var_args:
; CHECK: sub     sp, #12
; CHECK: push.w  {r11, lr}
; CHECK: mov     r11, sp
; CHECK: add.w   r1, r11, #20
; CHECK: bl      _setjmpex

; NOFP: var_args:
; NOFP: sub     sp, #12
; NOFP: push.w  {r11, lr}
; NOFP: sub     sp, #12
; NOFP: add     r1, sp, #32
; NOFP: bl      _setjmpex

define dso_local void @manyargs(i64 %x1, i64 %x2, i64 %x3, i64 %x4, i64 %x5, i64 %x6, i64 %x7, i64 %x8, i64 %x9, i64 %x10) {
  %1 = call ptr @llvm.sponentry()
  %2 = load ptr, ptr @env2, align 8
  %3 = call i32 @_setjmpex(ptr %2, ptr %1) #2
  ret void
}

; CHECK: manyargs:
; CHECK: push.w  {r11, lr}
; CHECK: mov     r11, sp
; CHECK: add.w   r1, r11, #8
; CHECK: bl      _setjmpex

; NOFP: manyargs:
; NOFP: push.w  {r11, lr}
; NOFP: add     r1, sp, #8
; NOFP: bl      _setjmpex

; Function Attrs: nounwind readnone
declare ptr @llvm.sponentry()

; Function Attrs: returns_twice
declare dso_local i32 @_setjmpex(ptr, ptr)

; Function Attrs: nounwind
declare void @llvm.va_start(ptr) #1

; Function Attrs: nounwind
declare void @llvm.va_end(ptr) #1
