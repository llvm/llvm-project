; RUN: llc -mtriple x86_64-apple-macosx10.8.0 < %s 2>&1 >/dev/null | FileCheck %s
; Check the internal option that warns when the stack frame size exceeds the
; given amount.
; <rdar://13987214>

; CHECK-NOT: nowarn
define void @nowarn() nounwind ssp "warn-stack-size"="80" {
entry:
  %buffer = alloca [12 x i8], align 1
  call void @doit(ptr %buffer) nounwind
  ret void
}

; CHECK: warning: <unknown>:0:0: stack frame size ([[STCK:[0-9]+]]) exceeds limit (80) in function 'warn'
; CHECK: {{[0-9]+}}/[[STCK]] ({{.*}}%) spills, {{[0-9]+}}/[[STCK]] ({{.*}}%) variables
define void @warn() nounwind ssp "warn-stack-size"="80" {
entry:
  %buffer = alloca [80 x i8], align 1
  call void @doit(ptr %buffer) nounwind
  ret void
}

; Ensure that warn-stack-size also considers the size of the unsafe stack.
; With safestack enabled the machine stack size is well below 80, but the
; combined stack size of the machine stack and unsafe stack will exceed the
; warning threshold

; CHECK: warning: <unknown>:0:0: stack frame size ([[STCK:[0-9]+]]) exceeds limit (80) in function 'warn_safestack'
; CHECK: {{[0-9]+}}/[[STCK]] ({{.*}}%) spills, {{[0-9]+}}/[[STCK]] ({{.*}}%) variables, {{[0-9]+}}/[[STCK]] ({{.*}}%) unsafe stack
define i32 @warn_safestack() nounwind ssp safestack "warn-stack-size"="80" {
entry:
  %var = alloca i32, align 4
  %buffer = alloca [80 x i8], align 1
  call void @doit(ptr %buffer) nounwind
  call void @doit(ptr %var) nounwind
  %val = load i32, ptr %var
  ret i32 %val
}
declare void @doit(ptr)
