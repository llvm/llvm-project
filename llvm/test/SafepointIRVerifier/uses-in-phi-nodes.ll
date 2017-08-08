; RUN: opt -safepoint-ir-verifier-print-only -verify-safepoint-ir -S %s 2>&1 | FileCheck %s

define i8 addrspace(1)* @test.not.ok.0(i8 addrspace(1)* %arg) gc "statepoint-example" {
; CHECK-LABEL: Verifying gc pointers in function: test.not.ok.0
 bci_0:
  br i1 undef, label %left, label %right

 left:
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  br label %merge

 right:
  br label %merge

 merge:
; CHECK: Illegal use of unrelocated value found!
; CHECK-NEXT: Def: i8 addrspace(1)* %arg
; CHECK-NEXT: Use:   %val = phi i8 addrspace(1)* [ %arg, %left ], [ %arg, %right ]
  %val = phi i8 addrspace(1)* [ %arg, %left ], [ %arg, %right]
  ret i8 addrspace(1)* %val
}

define i8 addrspace(1)* @test.not.ok.1(i8 addrspace(1)* %arg) gc "statepoint-example" {
; CHECK-LABEL: Verifying gc pointers in function: test.not.ok.1
 bci_0:
  br i1 undef, label %left, label %right

 left:
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  br label %merge

 right:
  br label %merge

 merge:
; CHECK: Illegal use of unrelocated value found!
; CHECK-NEXT: Def: i8 addrspace(1)* %arg
; CHECK-NEXT: Use:   %val = phi i8 addrspace(1)* [ %arg, %left ], [ null, %right ]
  %val = phi i8 addrspace(1)* [ %arg, %left ], [ null, %right]
  ret i8 addrspace(1)* %val
}

define i8 addrspace(1)* @test.ok.0(i8 addrspace(1)* %arg) gc "statepoint-example" {
; CHECK: No illegal uses found by SafepointIRVerifier in: test.ok.0
 bci_0:
  br i1 undef, label %left, label %right

 left:
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  br label %merge

 right:
  br label %merge

 merge:
  %val = phi i8 addrspace(1)* [ null, %left ], [ null, %right]
  ret i8 addrspace(1)* %val
}

define i8 addrspace(1)* @test.ok.1(i8 addrspace(1)* %arg) gc "statepoint-example" {
; CHECK: No illegal uses found by SafepointIRVerifier in: test.ok.1
 bci_0:
  br i1 undef, label %left, label %right

 left:
  call void @not_statepoint()
  br label %merge

 right:
  br label %merge

 merge:
  %val = phi i8 addrspace(1)* [ %arg, %left ], [ %arg, %right]
  ret i8 addrspace(1)* %val
}

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
declare void @not_statepoint()
