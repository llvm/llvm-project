; RUN: llc < %s -mtriple=arm64-apple-ios -relocation-model=pic -frame-pointer=all | FileCheck %s

@__stack_chk_guard = external global i64*

; PR20558

; CHECK: adrp [[R0:x[0-9]+]], ___stack_chk_guard@GOTPAGE
; CHECK: ldr  [[R1:x[0-9]+]], {{\[}}[[R0]], ___stack_chk_guard@GOTPAGEOFF{{\]}}
; Load the stack guard for the second time, just in case the previous value gets spilled.
; CHECK: adrp [[GUARD_PAGE:x[0-9]+]], ___stack_chk_guard@GOTPAGE
; CHECK: ldr  [[R2:x[0-9]+]], {{\[}}[[R1]]{{\]}}
; CHECK: stur [[R2]], {{\[}}x29, [[SLOT0:[0-9#\-]+]]{{\]}}
; CHECK: ldur [[R3:x[0-9]+]], {{\[}}x29, [[SLOT0]]{{\]}}
; CHECK: ldr  [[GUARD_ADDR:x[0-9]+]], {{\[}}[[GUARD_PAGE]], ___stack_chk_guard@GOTPAGEOFF{{\]}}
; CHECK: ldr  [[GUARD:x[0-9]+]], {{\[}}[[GUARD_ADDR]]{{\]}}
; CHECK: cmp  [[GUARD]], [[R3]]
; CHECK: b.ne LBB

define i32 @test_stack_guard_remat2() {
entry:
  %StackGuardSlot = alloca i8*
  %StackGuard = load i8*, i8** bitcast (i64** @__stack_chk_guard to i8**)
  call void @llvm.stackprotector(i8* %StackGuard, i8** %StackGuardSlot)
  %container = alloca [32 x i8], align 1
  call void @llvm.stackprotectorcheck(i8** bitcast (i64** @__stack_chk_guard to i8**))
  ret i32 -1
}

declare void @llvm.stackprotector(i8*, i8**)
declare void @llvm.stackprotectorcheck(i8**)
