; RUN: llc -verify-machineinstrs -filetype=asm -o - -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -filetype=asm -o - -mtriple=x86_64-darwin-unknown    < %s | FileCheck %s

define i32 @foo(i32 %i) nounwind noinline uwtable "xray-instruction-threshold"="10" "xray-ignore-loops" {
entry:
  br label %Test
Test:
  %indvar = phi i32 [0, %entry], [%nextindvar, %Inc]
  %cond = icmp eq i32 %indvar, %i
  br i1 %cond, label %Exit, label %Inc
Inc:
  %nextindvar = add i32 %indvar, 1
  br label %Test
Exit:
  %retval = phi i32 [%indvar, %Test]
  ret i32 %retval
}

; CHECK-NOT: xray_sled_0:
