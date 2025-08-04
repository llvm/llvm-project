; RUN: llc < %s -mtriple=x86_64-windows-gnu | FileCheck %s

define void @foo() unnamed_addr #0 {
start:
  %b = alloca i64, align 8
  %c = alloca [4294967295 x i8], align 1
  ret void
}

attributes #0 = { nonlazybind uwtable "probe-stack"="probe_stack" "target-cpu"="x86-64" }

; CHECK-LABEL: foo:
; CHECK: movabsq $4294967304, %rax
; CHECK-NEXT: callq probe_stack
