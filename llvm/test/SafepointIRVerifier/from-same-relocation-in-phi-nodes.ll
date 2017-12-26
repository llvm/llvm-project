; XFAIL: *
; RUN: opt -safepoint-ir-verifier-print-only -verify-safepoint-ir -S %s 2>&1 | FileCheck %s

; In %merge %val.unrelocated, %ptr and %arg should be unrelocated.
; FIXME: if this test fails it is a false-positive alarm. IR is correct.
define void @test.unrelocated-phi.ok(i8 addrspace(1)* %arg) gc "statepoint-example" {
; CHECK-LABEL: Verifying gc pointers in function: test.unrelocated-phi.ok
 bci_0:
  %ptr = getelementptr i8, i8 addrspace(1)* %arg, i64 4
  br i1 undef, label %left, label %right

 left:
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  br label %merge

 right:
  br label %merge

 merge:
; CHECK: No illegal uses found by SafepointIRVerifier in: test.unrelocated-phi.ok
  %val.unrelocated = phi i8 addrspace(1)* [ %arg, %left ], [ %ptr, %right ]
  %c = icmp eq i8 addrspace(1)* %val.unrelocated, %arg
  ret void
}

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
