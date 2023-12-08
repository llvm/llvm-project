; RUN: llc < %s -mcpu=generic -mtriple=x86_64-pc-linux -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-pc-linux-gnux32 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mcpu=generic -mtriple=i686-pc-linux -verify-machineinstrs | FileCheck %s -check-prefix=IA32

; trivial test for correct call suffix

define i32 @far() nounwind uwtable {
entry:
; CHECK: callq foo
; IA32: calll foo
  tail call void @foo() nounwind
  ret i32 0
}

declare void @foo()
