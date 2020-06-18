; RUN: llc -verify-machineinstrs < %s -mcpu=pwr4 \
; RUN: -mtriple=powerpc-ibm-aix-xcoff 2>&1 | FileCheck %s

; RUN: llc -verify-machineinstrs < %s -mcpu=pwr4 \
; RUN: -mtriple=powerpc-unknown-linux-gnu 2>&1 | FileCheck %s

; When we convert an `i64` to `f32` on 32-bit PPC target, a `setcc` will be
; generated. And this testcase verifies that the operand expansion of `setcc`
; will not crash.

%struct.A = type { float }

@ll = external local_unnamed_addr global i64
@a = external local_unnamed_addr global %struct.A

define void @foo() local_unnamed_addr {
entry:
  %0 = load i64, i64* @ll
  %conv = sitofp i64 %0 to float
  store float %conv, float* getelementptr inbounds (%struct.A, %struct.A* @a, i32 0, i32 0)
  ret void
}

; CHECK-NOT: Unexpected setcc expansion!
