; RUN: llc < %s -mtriple=x86_64-apple-darwin10 | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mtriple=x86_64-pc-win32 | FileCheck %s -check-prefix=WIN64

@.str = private unnamed_addr constant [5 x i8] c"%ld\0A\00"
@sel = external global ptr
@sel3 = external global ptr
@sel4 = external global ptr
@sel5 = external global ptr
@sel6 = external global ptr
@sel7 = external global ptr

; X64: @foo
; X64: jmp
; WIN64: @foo
; WIN64: callq
define void @foo(i64 %arg) nounwind optsize ssp noredzone {
entry:
  %call = tail call i32 (ptr, ...) @printf(ptr @.str, i64 %arg) nounwind optsize noredzone
  ret void
}

declare i32 @printf(ptr, ...) optsize noredzone

; X64: @bar
; X64: jmp
; WIN64: @bar
; WIN64: jmp
define void @bar(i64 %arg) nounwind optsize ssp noredzone {
entry:
  tail call void @bar2(ptr @.str, i64 %arg) nounwind optsize noredzone
  ret void
}

declare void @bar2(ptr, i64) optsize noredzone

; X64: @foo2
; X64: jmp
; WIN64: @foo2
; WIN64: callq
define ptr @foo2(ptr %arg) nounwind optsize ssp noredzone {
entry:
  %tmp1 = load ptr, ptr @sel, align 8
  %call = tail call ptr (ptr, ptr, ...) @x2(ptr %arg, ptr %tmp1) nounwind optsize noredzone
  ret ptr %call
}

declare ptr @x2(ptr, ptr, ...) optsize noredzone

; X64: @foo6
; X64: jmp
; WIN64: @foo6
; WIN64: callq
define ptr @foo6(ptr %arg1, ptr %arg2) nounwind optsize ssp noredzone {
entry:
  %tmp2 = load ptr, ptr @sel3, align 8
  %tmp3 = load ptr, ptr @sel4, align 8
  %tmp4 = load ptr, ptr @sel5, align 8
  %tmp5 = load ptr, ptr @sel6, align 8
  %call = tail call ptr (ptr, ptr, ptr, ...) @x3(ptr %arg1, ptr %arg2, ptr %tmp2, ptr %tmp3, ptr %tmp4, ptr %tmp5) nounwind optsize noredzone
  ret ptr %call
}

declare ptr @x3(ptr, ptr, ptr, ...) optsize noredzone

; X64: @foo7
; X64: callq
; WIN64: @foo7
; WIN64: callq
define ptr @foo7(ptr %arg1, ptr %arg2) nounwind optsize ssp noredzone {
entry:
  %tmp2 = load ptr, ptr @sel3, align 8
  %tmp3 = load ptr, ptr @sel4, align 8
  %tmp4 = load ptr, ptr @sel5, align 8
  %tmp5 = load ptr, ptr @sel6, align 8
  %tmp6 = load ptr, ptr @sel7, align 8
  %call = tail call ptr (ptr, ptr, ptr, ptr, ptr, ptr, ptr, ...) @x7(ptr %arg1, ptr %arg2, ptr %tmp2, ptr %tmp3, ptr %tmp4, ptr %tmp5, ptr %tmp6) nounwind optsize noredzone
  ret ptr %call
}

declare ptr @x7(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ...) optsize noredzone

; X64: @foo8
; X64: callq
; WIN64: @foo8
; WIN64: callq
define ptr @foo8(ptr %arg1, ptr %arg2) nounwind optsize ssp noredzone {
entry:
  %tmp2 = load ptr, ptr @sel3, align 8
  %tmp3 = load ptr, ptr @sel4, align 8
  %tmp4 = load ptr, ptr @sel5, align 8
  %tmp5 = load ptr, ptr @sel6, align 8
  %call = tail call ptr (ptr, ptr, ptr, ...) @x3(ptr %arg1, ptr %arg2, ptr %tmp2, ptr %tmp3, ptr %tmp4, ptr %tmp5, i32 48879, i32 48879) nounwind optsize noredzone
  ret ptr %call
}
