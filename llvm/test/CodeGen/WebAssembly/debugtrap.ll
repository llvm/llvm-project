; RUN: llc < %s -asm-verbose=false -verify-machineinstrs | FileCheck %s

; Test lowering of __builtin_debugtrap in cases where lowering it via
; the normal UNREACHABLE instruction would yield invalid
; MachineFunctions.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32"

declare void @llvm.debugtrap()

; CHECK-LABEL: foo:
; CHECK-NEXT: .functype       foo (i32) -> ()
; CHECK-NEXT: .LBB0_1:
; CHECK-NEXT: loop
; CHECK-NEXT: unreachable
; CHECK-NEXT: i32.const       0
; CHECK-NEXT: br_if           0
; CHECK-NEXT: end_loop
; CHECK-NEXT: end_function
define void @foo(i32 %g) {
entry:
  br label %for.body

for.body:
  call void @llvm.debugtrap()
  %exitcond = icmp eq i32 undef, %g
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

; CHECK-LABEL: middle_of_block:
; CHECK-NEXT: .functype       middle_of_block (i32, i32) -> (i32)
; CHECK-NEXT: unreachable
; CHECK-NEXT: local.get       0
; CHECK-NEXT: local.get       1
; CHECK-NEXT: i32.add
; CHECK-NEXT: end_function
define i32 @middle_of_block(i32 %x, i32 %y) {
  %r = add i32 %x, %y
  call void @llvm.debugtrap()
  ret i32 %r
}

; CHECK-LABEL: really_middle_of_block:
; CHECK-NEXT: .functype       really_middle_of_block () -> (i32)
; CHECK-NEXT: call    bar
; CHECK-NEXT: drop
; CHECK-NEXT: unreachable
; CHECK-NEXT: call    bar
; CHECK-NEXT: end_function
declare i32 @bar()
define i32 @really_middle_of_block() {
  %x = call i32 @bar()
  call void @llvm.debugtrap()
  %r = call i32 @bar()
  ret i32 %r
}
