; RUN: llvm-as < %s | llvm-bcanalyzer -dump | FileCheck %s -check-prefix=BC
; Check for VST forward declaration record and VST function index records.

; BC: <VSTOFFSET
; BC: <FNENTRY
; BC: <FNENTRY

; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; Check that this round-trips correctly.

; ModuleID = '<stdin>'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: define i32 @foo()

; Function Attrs: nounwind uwtable
define i32 @foo() #0 {
entry:
  ret i32 1
}

; CHECK: define i32 @bar(i32 %x)

; Function Attrs: nounwind uwtable
define i32 @bar(i32 %x) #0 {
entry:
  ret i32 %x
}
