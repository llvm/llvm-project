; RUN: opt -safepoint-ir-verifier-print-only -verify-safepoint-ir -S %s 2>&1 | FileCheck %s

; This test checks that StatepointIRVerifier does not crash on
; a CFG with unreachable blocks.

%jObject = type { [8 x i8] }

define %jObject addrspace(1)* @test(%jObject addrspace(1)* %arg) gc "statepoint-example" {
; CHECK-LABEL: Verifying gc pointers in function: test
; CHECK-NEXT:  No illegal uses found by SafepointIRVerifier in: test
  %safepoint_token3 = tail call token (i64, i32, double (double)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_f64f64f(i64 0, i32 0, double (double)* undef, i32 1, i32 0, double undef, i32 0, i32 0) ["gc-live"(%jObject addrspace(1)* %arg)]
  %arg2.relocated4 = call coldcc %jObject addrspace(1)* @llvm.experimental.gc.relocate.p1jObject(token %safepoint_token3, i32 0, i32 0)
  ret %jObject addrspace(1)* %arg2.relocated4

unreachable:
  ret %jObject addrspace(1)* null
}

; Function Attrs: nounwind
declare %jObject addrspace(1)* @llvm.experimental.gc.relocate.p1jObject(token, i32, i32) #3

declare token @llvm.experimental.gc.statepoint.p0f_f64f64f(i64, i32, double (double)*, i32, i32, ...)

; In %merge %val.unrelocated, %ptr and %arg should be unrelocated.
define void @test2(i8 addrspace(1)* %arg) gc "statepoint-example" {
; CHECK-LABEL: Verifying gc pointers in function: test2
; CHECK: No illegal uses found by SafepointIRVerifier in: test2
 bci_0:
  %ptr = getelementptr i8, i8 addrspace(1)* %arg, i64 4
  br label %right

 left:
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 0)
  br label %merge

 right:
  br label %merge

 merge:
  %val.unrelocated = phi i8 addrspace(1)* [ %arg, %left ], [ %ptr, %right ]
  %c = icmp eq i8 addrspace(1)* %val.unrelocated, %arg
  ret void
}

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
