; RUN: llc -march=mips < %s | FileCheck %s -check-prefix=O32
; RUN: llc -march=mips64 -target-abi=n32 < %s | FileCheck %s -check-prefix=N32
; RUN: llc -march=mips64 < %s | FileCheck %s -check-prefix=N64

; We only use the '$' prefix on O32. The others use the ELF convention.
; O32: $JTI0_0
; N32: .LJTI0_0
; N64: .LJTI0_0

; Check basic block labels while we're at it.
; O32: $BB0_2:
; N32: .LBB0_2:
; N64: .LBB0_2:

@.str = private unnamed_addr constant [2 x i8] c"A\00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"B\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"C\00", align 1
@.str.3 = private unnamed_addr constant [2 x i8] c"D\00", align 1
@.str.4 = private unnamed_addr constant [2 x i8] c"E\00", align 1
@.str.5 = private unnamed_addr constant [2 x i8] c"F\00", align 1
@.str.6 = private unnamed_addr constant [2 x i8] c"G\00", align 1
@.str.7 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1

define ptr @_Z3fooi(i32 signext %Letter) {
entry:
  %retval = alloca ptr, align 8
  %Letter.addr = alloca i32, align 4
  store i32 %Letter, ptr %Letter.addr, align 4
  %0 = load i32, ptr %Letter.addr, align 4
  switch i32 %0, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
    i32 4, label %sw.bb4
    i32 5, label %sw.bb5
    i32 6, label %sw.bb6
  ]

sw.bb:
  store ptr @.str, ptr %retval, align 8
  br label %return

sw.bb1:
  store ptr @.str.1, ptr %retval, align 8
  br label %return

sw.bb2:
  store ptr @.str.2, ptr %retval, align 8
  br label %return

sw.bb3:
  store ptr @.str.3, ptr %retval, align 8
  br label %return

sw.bb4:
  store ptr @.str.4, ptr %retval, align 8
  br label %return

sw.bb5:
  store ptr @.str.5, ptr %retval, align 8
  br label %return

sw.bb6:
  store ptr @.str.6, ptr %retval, align 8
  br label %return

sw.epilog:
  store ptr @.str.7, ptr %retval, align 8
  br label %return

return:
  %1 = load ptr, ptr %retval, align 8
  ret ptr %1
}
