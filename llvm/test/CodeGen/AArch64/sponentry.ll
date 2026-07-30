; RUN: llc -mtriple=aarch64-windows-msvc -frame-pointer=all %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-windows-msvc -fast-isel -frame-pointer=all %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-windows-msvc %s -o - | FileCheck %s --check-prefix=NOFP
; RUN: llc -mtriple=aarch64-windows-msvc -fast-isel %s -o - | FileCheck %s --check-prefix=NOFP

@env2 = common dso_local global ptr null, align 8

define dso_local void @bar() {
  %1 = call ptr @llvm.sponentry()
  %2 = load ptr, ptr @env2, align 8
  %3 = call i32 @_setjmpex(ptr %2, ptr %1) #2
  ret void
}

; CHECK: bar:
; CHECK: mov     x29, sp
; CHECK: add     x1, x29, #16
; CHECK: bl      _setjmpex

; NOFP: str     x30, [sp, #-16]!
; NOFP: add     x1, sp, #16

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
; CHECK: sub     sp, sp, #448
; CHECK: add     x29, sp, #424
; CHECK: add     x1, x29, #24
; CHECK: bl      _setjmpex

; NOFP: sub     sp, sp, #432
; NOFP: add     x1, sp, #432

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
; CHECK: sub     sp, sp, #96
; CHECK: add     x29, sp, #16
; CHECK: add     x1, x29, #80
; CHECK: bl      _setjmpex

; NOFP: sub     sp, sp, #96
; NOFP: add     x1, sp, #96

define dso_local void @manyargs(i64 %x1, i64 %x2, i64 %x3, i64 %x4, i64 %x5, i64 %x6, i64 %x7, i64 %x8, i64 %x9, i64 %x10) {
  %1 = call ptr @llvm.sponentry()
  %2 = load ptr, ptr @env2, align 8
  %3 = call i32 @_setjmpex(ptr %2, ptr %1) #2
  ret void
}

; CHECK: manyargs:
; CHECK: stp     x29, x30, [sp, #-16]!
; CHECK: add     x1, x29, #16

; NOFP: str     x30, [sp, #-16]!
; NOFP: add     x1, sp, #16

; Function Attrs: nounwind readnone
declare ptr @llvm.sponentry()

; Function Attrs: returns_twice
declare dso_local i32 @_setjmpex(ptr, ptr)

; Function Attrs: nounwind
declare void @llvm.va_start(ptr) #1

; Function Attrs: nounwind
declare void @llvm.va_end(ptr) #1
