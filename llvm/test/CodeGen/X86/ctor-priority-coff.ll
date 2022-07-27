; RUN: llc < %s | FileCheck %s

; Check that we come up with appropriate section names that link.exe sorts
; well.

; CHECK: .section        .CRT$XCA00042,"dr"
; CHECK: .p2align        3
; CHECK: .quad   f
; CHECK: .section        .CRT$XCT12345,"dr"
; CHECK: .p2align        3
; CHECK: .quad   g
; CHECK: .section        .CRT$XCT23456,"dr",associative,h
; CHECK: .p2align        3
; CHECK: .quad   init_h

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.14.26433"

$h = comdat any

@h = linkonce_odr global i8 55, comdat, align 1

@str0 = private dso_local unnamed_addr constant [6 x i8] c"later\00", align 1
@str1 = private dso_local unnamed_addr constant [6 x i8] c"first\00", align 1
@str2 = private dso_local unnamed_addr constant [5 x i8] c"main\00", align 1

@llvm.global_ctors = appending global [3 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 12345, ptr @g, ptr null },
  { i32, ptr, ptr } { i32 42, ptr @f, ptr null },
  { i32, ptr, ptr } { i32 23456, ptr @init_h, ptr @h }
]

declare dso_local i32 @puts(ptr nocapture readonly) local_unnamed_addr

define dso_local void @g() {
entry:
  %call = tail call i32 @puts(ptr @str0)
  ret void
}

define dso_local void @f() {
entry:
  %call = tail call i32 @puts(ptr @str1)
  ret void
}

define dso_local void @init_h() {
entry:
  store i8 42, ptr @h
  ret void
}


; Function Attrs: nounwind uwtable
define dso_local i32 @main() local_unnamed_addr {
entry:
  %call = tail call i32 @puts(ptr @str2)
  ret i32 0
}
