; RUN: opt -safepoint-ir-verifier-print-only -verify-safepoint-ir -S %s 2>&1 | FileCheck %s

; In some cases, it is valid to have unrelocated pointers used as compare
; operands. Make sure the verifier knows to spot these exceptions.


; comparison against null.
define i8 addrspace(1)* @test1(i64 %arg, i8 addrspace(1)* %addr) gc "statepoint-example" {
; CHECK: No illegal uses found by SafepointIRVerifier in: test1
entry:
  %load_addr = getelementptr i8, i8 addrspace(1)* %addr, i64 %arg
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* elementtype(void ()) undef, i32 0, i32 0, i32 0, i32 0)
  %cmp = icmp eq i8 addrspace(1)* %load_addr, null
  ret i8 addrspace(1)* null
}

; comparison against exclusively derived null.
define void @test2(i64 %arg, i1 %cond, i8 addrspace(1)* %addr) gc "statepoint-example" {
; CHECK: No illegal uses found by SafepointIRVerifier in: test2
  %load_addr = getelementptr i8, i8 addrspace(1)* null, i64 %arg
  %load_addr_sel = select i1 %cond, i8 addrspace(1)* null, i8 addrspace(1)* %load_addr
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* elementtype(void ()) undef, i32 0, i32 0, i32 0, i32 0)
  %cmp = icmp eq i8 addrspace(1)* %addr, %load_addr_sel
  ret void
}

; comparison against a constant non-null pointer. This is unrelocated use, since
; that pointer bits may mean something in a VM.
define void @test3(i64 %arg, i32 addrspace(1)* %addr) gc "statepoint-example" {
; CHECK-LABEL: Verifying gc pointers in function: test3
; CHECK: Illegal use of unrelocated value found!
entry:
  %load_addr = getelementptr i32, i32 addrspace(1)* %addr, i64 %arg
  %load_addr_const = getelementptr i32, i32 addrspace(1)* inttoptr (i64 15 to i32 addrspace(1)*), i64 %arg
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* elementtype(void ()) undef, i32 0, i32 0, i32 0, i32 0)
  %cmp = icmp eq i32 addrspace(1)* %load_addr, %load_addr_const
  ret void
}

; comparison against a derived pointer that is *not* exclusively derived from
; null. An unrelocated use since the derived pointer could be from the constant
; non-null pointer (load_addr.2).
define void @test4(i64 %arg, i1 %cond, i8 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: Verifying gc pointers in function: test4
; CHECK: Illegal use of unrelocated value found!
entry:
  %load_addr.1 = getelementptr i8, i8 addrspace(1)* null, i64 %arg
  br i1 %cond, label %split, label %join

split:
  %load_addr.2 = getelementptr i8, i8 addrspace(1)* inttoptr (i64 30 to i8 addrspace(1)*), i64 %arg
  br label %join

join:
  %load_addr = phi i8 addrspace(1)* [%load_addr.1, %entry], [%load_addr.2, %split]
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* elementtype(void ()) undef, i32 0, i32 0, i32 0, i32 0)
  %cmp = icmp eq i8 addrspace(1)* %load_addr, %base
  ret void
}

; comparison between 2 unrelocated base pointers.
; Since the cmp can be reordered legally before the safepoint, these are correct
; unrelocated uses of the pointers.
define void @test5(i64 %arg, i8 addrspace(1)* %base1, i8 addrspace(1)* %base2) gc "statepoint-example" {
; CHECK: No illegal uses found by SafepointIRVerifier in: test5
  %load_addr1 = getelementptr i8, i8 addrspace(1)* %base1, i64 %arg
  %load_addr2 = getelementptr i8, i8 addrspace(1)* %base2, i64 %arg
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* elementtype(void ()) undef, i32 0, i32 0, i32 0, i32 0)
  %cmp = icmp eq i8 addrspace(1)* %load_addr1, %load_addr2
  ret void
}

; comparison between a relocated and an unrelocated pointer.
; this is invalid use of the unrelocated pointer.
define void @test6(i64 %arg, i8 addrspace(1)* %base1, i8 addrspace(1)* %base2) gc "statepoint-example" {
; CHECK-LABEL: Verifying gc pointers in function: test6
; CHECK: Illegal use of unrelocated value found!
  %load_addr1 = getelementptr i8, i8 addrspace(1)* %base1, i64 %arg
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* elementtype(void ()) undef, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i8 addrspace(1)* %base2)]
  %ptr2.relocated = call i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %safepoint_token, i32 0, i32 0) ; base2, base2
  %cmp = icmp eq i8 addrspace(1)* %load_addr1, %ptr2.relocated
  ret void
}
declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
declare i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token, i32, i32)
