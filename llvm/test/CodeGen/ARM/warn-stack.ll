; RUN: llc -mtriple thumbv7-apple-ios3.0.0 < %s 2>&1 >/dev/null | FileCheck %s
; Check the internal option that warns when the stack frame size exceeds the
; given amount.
; <rdar://13987214>

; CHECK-NOT: nowarn
define void @nowarn() nounwind ssp "frame-pointer"="all" "warn-stack-size"="80" {
entry:
  %buffer = alloca [12 x i8], align 1
  call void @doit(ptr %buffer) nounwind
  ret void
}

; CHECK: warning: <unknown>:0:0: stack frame size ([[STCK:[0-9]+]]) exceeds limit (80) in function 'warn'
; CHECK: {{[0-9]+}}/[[STCK]] ({{.*}}%) spills, {{[0-9]+}}/[[STCK]] ({{.*}}%) variables
define i32 @warn() nounwind ssp "frame-pointer"="all" "warn-stack-size"="80" {
entry:
  %var = alloca i32, align 4
  %buffer = alloca [80 x i8], align 1
  call void @doit(ptr %buffer) nounwind
  call void @doit(ptr %var) nounwind
  %val = load i32, ptr %var
  ret i32 %val
}

; CHECK: warning: stack frame size ([[STCK:[0-9]+]]) exceeds limit (80) in function 'warn_safestack'
; CHECK: {{[0-9]+}}/[[STCK]] ({{.*}}%) spills, {{[0-9]+}}/[[STCK]] ({{.*}}%) variables, {{[0-9]+}}/[[STCK]] ({{.*}}%) unsafe stack
define i32 @warn_safestack() nounwind ssp safestack "warn-stack-size"="80" {
entry:
  %var = alloca i32, align 4
  %a = alloca i32, align 4
  %buffer = alloca [80 x i8], align 1
  call void @doit(ptr %buffer) nounwind
  call void @doit(ptr %var) nounwind
  %val = load i32, ptr %var
  ret i32 %val
}

declare void @doit(ptr)
