; RUN: opt -safepoint-ir-verifier-print-only -verify-safepoint-ir -S %s 2>&1 | FileCheck %s

define i8 addrspace(1)* @test1(i64 %arg) gc "statepoint-example" {
; CHECK: No illegal uses found by SafepointIRVerifier in: test1
entry:
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* null
}

define i8 addrspace(1)* @test2(i64 %arg) gc "statepoint-example" {
; CHECK: No illegal uses found by SafepointIRVerifier in: test2
entry:
  %load_addr = getelementptr i8, i8 addrspace(1)* inttoptr (i64 15 to i8 addrspace(1)*), i64 %arg
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* %load_addr
}

define i8 addrspace(1)* @test3(i64 %arg) gc "statepoint-example" {
; CHECK: No illegal uses found by SafepointIRVerifier in: test3
entry:
  %load_addr = getelementptr i32, i32 addrspace(1)* inttoptr (i64 15 to i32 addrspace(1)*), i64 %arg
  %load_addr.cast = bitcast i32 addrspace(1)* %load_addr to i8 addrspace(1)*
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* %load_addr.cast
}

define i8 addrspace(1)* @test4(i64 %arg, i1 %cond) gc "statepoint-example" {
; CHECK: No illegal uses found by SafepointIRVerifier in: test4
entry:
  %load_addr.1 = getelementptr i8, i8 addrspace(1)* inttoptr (i64 15 to i8 addrspace(1)*), i64 %arg
  br i1 %cond, label %split, label %join

split:
  %load_addr.2 = getelementptr i8, i8 addrspace(1)* inttoptr (i64 30 to i8 addrspace(1)*), i64 %arg
  br label %join

join:
  %load_addr = phi i8 addrspace(1)* [%load_addr.1, %entry], [%load_addr.2, %split]
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* %load_addr
}

define i8 addrspace(1)* @test5(i64 %arg, i1 %cond) gc "statepoint-example" {
; CHECK: No illegal uses found by SafepointIRVerifier in: test5
entry:
  %load_addr.1 = getelementptr i8, i8 addrspace(1)* inttoptr (i64 15 to i8 addrspace(1)*), i64 %arg
  %load_addr.2 = getelementptr i8, i8 addrspace(1)* inttoptr (i64 30 to i8 addrspace(1)*), i64 %arg
  %load_addr = select i1 %cond, i8 addrspace(1)* %load_addr.1, i8 addrspace(1)* %load_addr.2
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* %load_addr
}

define i8 addrspace(1)* @test6(i64 %arg, i1 %cond, i8 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: Verifying gc pointers in function: test6
; CHECK: Illegal use of unrelocated value found!
entry:
  %load_addr.1 = getelementptr i8, i8 addrspace(1)* %base, i64 %arg
  br i1 %cond, label %split, label %join

split:
  %load_addr.2 = getelementptr i8, i8 addrspace(1)* inttoptr (i64 30 to i8 addrspace(1)*), i64 %arg
  br label %join

join:
  %load_addr = phi i8 addrspace(1)* [%load_addr.1, %entry], [%load_addr.2, %split]
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* %load_addr
}

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
