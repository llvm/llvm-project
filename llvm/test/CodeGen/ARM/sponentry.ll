; RUN: llc -mtriple=thumbv7-windows-msvc -frame-pointer=all %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7-windows-msvc -fast-isel -frame-pointer=all %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7-windows-msvc %s -o - | FileCheck %s --check-prefix=NOFP
; RUN: llc -mtriple=thumbv7-windows-msvc -fast-isel %s -o - | FileCheck %s --check-prefix=NOFP

@env2 = common dso_local global [24 x i64]* null, align 8

define dso_local void @bar() {
  %1 = call i8* @llvm.sponentry()
  %2 = load [24 x i64]*, [24 x i64]** @env2, align 8
  %3 = getelementptr inbounds [24 x i64], [24 x i64]* %2, i32 0, i32 0
  %4 = bitcast i64* %3 to i8*
  %5 = call i32 @_setjmpex(i8* %4, i8* %1) #2
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

define dso_local void @foo([24 x i64]*) {
  %2 = alloca [24 x i64]*, align 8
  %3 = alloca i32, align 4
  %4 = alloca [100 x i32], align 4
  store [24 x i64]* %0, [24 x i64]** %2, align 8
  %5 = call i8* @llvm.sponentry()
  %6 = load [24 x i64]*, [24 x i64]** %2, align 8
  %7 = getelementptr inbounds [24 x i64], [24 x i64]* %6, i32 0, i32 0
  %8 = bitcast i64* %7 to i8*
  %9 = call i32 @_setjmpex(i8* %8, i8* %5)
  store i32 %9, i32* %3, align 4
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

define dso_local void @var_args(i8*, ...) {
  %2 = alloca i8*, align 8
  %3 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %4 = bitcast i8** %3 to i8*
  call void @llvm.va_start(i8* %4)
  %5 = load i8*, i8** %3, align 8
  %6 = getelementptr inbounds i8, i8* %5, i64 8
  store i8* %6, i8** %3, align 8
  %7 = bitcast i8* %5 to i32*
  %8 = load i32, i32* %7, align 8
  %9 = bitcast i8** %3 to i8*
  call void @llvm.va_end(i8* %9)
  %10 = call i8* @llvm.sponentry()
  %11 = load [24 x i64]*, [24 x i64]** @env2, align 8
  %12 = getelementptr inbounds [24 x i64], [24 x i64]* %11, i32 0, i32 0
  %13 = bitcast i64* %12 to i8*
  %14 = call i32 @_setjmpex(i8* %13, i8* %10) #3
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
  %1 = call i8* @llvm.sponentry()
  %2 = load [24 x i64]*, [24 x i64]** @env2, align 8
  %3 = getelementptr inbounds [24 x i64], [24 x i64]* %2, i32 0, i32 0
  %4 = bitcast i64* %3 to i8*
  %5 = call i32 @_setjmpex(i8* %4, i8* %1) #2
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
declare i8* @llvm.sponentry()

; Function Attrs: returns_twice
declare dso_local i32 @_setjmpex(i8*, i8*)

; Function Attrs: nounwind
declare void @llvm.va_start(i8*) #1

; Function Attrs: nounwind
declare void @llvm.va_end(i8*) #1
