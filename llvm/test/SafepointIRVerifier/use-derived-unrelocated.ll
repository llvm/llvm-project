; RUN: opt -safepoint-ir-verifier-print-only -verify-safepoint-ir -S %s 2>&1 | FileCheck %s

; Checking if verifier accepts chain of GEPs/bitcasts.
define void @test.deriving.ok(i32, i8 addrspace(1)* %base1, i8 addrspace(1)* %base2) gc "statepoint-example" {
; CHECK-LABEL: Verifying gc pointers in function: test.deriving.ok
; CHECK-NEXT: No illegal uses found by SafepointIRVerifier in: test.deriving.ok
  %ptr = getelementptr i8, i8 addrspace(1)* %base1, i64 4
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i8 addrspace(1)* %base1)]
  %ptr2 = getelementptr i8, i8 addrspace(1)* %base2, i64 8
  %ptr.i32 = bitcast i8 addrspace(1)* %ptr to i32 addrspace(1)*
  %ptr2.i32 = bitcast i8 addrspace(1)* %ptr2 to i32 addrspace(1)*
  ret void
}

; Checking if verifier accepts cmp of two derived pointers when one defined
; before safepoint and one after and both have unrelocated base.
define void @test.cmp.ok(i32, i8 addrspace(1)* %base1, i8 addrspace(1)* %base2) gc "statepoint-example" {
; CHECK-LABEL: Verifying gc pointers in function: test.cmp.ok
; CHECK-NEXT: No illegal uses found by SafepointIRVerifier in: test.cmp.ok
  %ptr = getelementptr i8, i8 addrspace(1)* %base1, i64 4
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i8 addrspace(1)* %base1)]
  %ptr2 = getelementptr i8, i8 addrspace(1)* %base2, i64 8
  %c2 = icmp sgt i8 addrspace(1)* %ptr2, %ptr
  ret void
}

; Checking if verifier accepts cmp of two derived pointers when one defined
; before safepoint and one after and both have unrelocated base. One of pointers
; defined as a long chain of geps/bitcasts.
define void @test.cmp-long_chain.ok(i32, i8 addrspace(1)* %base1, i8 addrspace(1)* %base2) gc "statepoint-example" {
; CHECK-LABEL: Verifying gc pointers in function: test.cmp-long_chain.ok
; CHECK-NEXT: No illegal uses found by SafepointIRVerifier in: test.cmp-long_chain.ok
  %ptr = getelementptr i8, i8 addrspace(1)* %base1, i64 4
  %ptr.i32 = bitcast i8 addrspace(1)* %ptr to i32 addrspace(1)*
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i8 addrspace(1)* %base1)]
  %ptr2 = getelementptr i8, i8 addrspace(1)* %base2, i64 8
  %ptr2.i32 = bitcast i8 addrspace(1)* %ptr2 to i32 addrspace(1)*
  %ptr2.i32.2 = getelementptr i32, i32 addrspace(1)* %ptr2.i32, i64 4
  %ptr2.i32.3 = getelementptr i32, i32 addrspace(1)* %ptr2.i32.2, i64 8
  %ptr2.i32.4 = getelementptr i32, i32 addrspace(1)* %ptr2.i32.3, i64 8
  %ptr2.i32.5 = getelementptr i32, i32 addrspace(1)* %ptr2.i32.4, i64 8
  %ptr2.i32.6 = getelementptr i32, i32 addrspace(1)* %ptr2.i32.5, i64 8
  %ptr2.i32.6.i8 = bitcast i32 addrspace(1)* %ptr2.i32.6 to i8 addrspace(1)*
  %ptr2.i32.6.i8.i32 = bitcast i8 addrspace(1)* %ptr2.i32.6.i8 to i32 addrspace(1)*
  %ptr2.i32.6.i8.i32.2 = getelementptr i32, i32 addrspace(1)* %ptr2.i32.6.i8.i32, i64 8
  %c2 = icmp sgt i32 addrspace(1)* %ptr2.i32.6.i8.i32.2, %ptr.i32
  ret void
}

; GEP and bitcast of unrelocated pointer is acceptable, but load by resulting
; pointer should be reported.
define void @test.load.fail(i32, i8 addrspace(1)* %base) gc "statepoint-example" {
; CHECK-LABEL: Verifying gc pointers in function: test.load.fail
  %ptr = getelementptr i8, i8 addrspace(1)* %base, i64 4
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i8 addrspace(1)* %base)]
  %ptr.i32 = bitcast i8 addrspace(1)* %ptr to i32 addrspace(1)* ; it's ok
; CHECK-NEXT: Illegal use of unrelocated value found!
; CHECK-NEXT: Def:   %ptr.i32 = bitcast i8 addrspace(1)* %ptr to i32 addrspace(1)*
; CHECK-NEXT: Use:   %ptr.val = load i32, i32 addrspace(1)* %ptr.i32
  %ptr.val = load i32, i32 addrspace(1)* %ptr.i32
  ret void
}

; Comparison between pointer derived from unrelocated one (though defined after
; safepoint) and relocated pointer should be reported.
define void @test.cmp.fail(i64 %arg, i8 addrspace(1)* %base1, i8 addrspace(1)* %base2) gc "statepoint-example" {
; CHECK-LABEL: Verifying gc pointers in function: test.cmp.fail
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i8 addrspace(1)* %base2)]
  %base2.relocated = call i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %safepoint_token, i32 0, i32 0) ; base2, base2
  %addr1 = getelementptr i8, i8 addrspace(1)* %base1, i64 %arg
; CHECK-NEXT: Illegal use of unrelocated value found!
; CHECK-NEXT: Def:   %addr1 = getelementptr i8, i8 addrspace(1)* %base1, i64 %arg
; CHECK-NEXT: Use:   %cmp = icmp eq i8 addrspace(1)* %addr1, %base2.relocated
  %cmp = icmp eq i8 addrspace(1)* %addr1, %base2.relocated
  ret void
}

; Same as test.cmp.fail but splitted into two BBs.
define void @test.cmp2.fail(i64 %arg, i8 addrspace(1)* %base1, i8 addrspace(1)* %base2) gc "statepoint-example" {
.b0:
; CHECK-LABEL: Verifying gc pointers in function: test.cmp2.fail
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i8 addrspace(1)* %base2)]
  %base2.relocated = call i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %safepoint_token, i32 0, i32 0) ; base2, base2
  %addr1 = getelementptr i8, i8 addrspace(1)* %base1, i64 %arg
  br label %.b1

.b1:
; CHECK-NEXT: Illegal use of unrelocated value found!
; CHECK-NEXT: Def:   %addr1 = getelementptr i8, i8 addrspace(1)* %base1, i64 %arg
; CHECK-NEXT: Use:   %cmp = icmp eq i8 addrspace(1)* %addr1, %base2.relocated
  %cmp = icmp eq i8 addrspace(1)* %addr1, %base2.relocated
  ret void
}

; Checking that cmp of two unrelocated pointers is OK and load is not.
define void @test.cmp-load.fail(i64 %arg, i8 addrspace(1)* %base1, i8 addrspace(1)* %base2) gc "statepoint-example" {
; CHECK-LABEL: Verifying gc pointers in function: test.cmp-load.fail
  %addr1 = getelementptr i8, i8 addrspace(1)* %base1, i64 %arg
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i8 addrspace(1)* %base2)]
  %addr2 = getelementptr i8, i8 addrspace(1)* %base2, i64 8
  %cmp = icmp eq i8 addrspace(1)* %addr1, %addr2
; CHECK-NEXT: Illegal use of unrelocated value found!
; CHECK-NEXT: Def:   %addr2 = getelementptr i8, i8 addrspace(1)* %base2, i64 8
; CHECK-NEXT: Use:   %val = load i8, i8 addrspace(1)* %addr2
  %val = load i8, i8 addrspace(1)* %addr2
  ret void
}

; Same as test.cmp-load.fail but splitted into thee BBs.
define void @test.cmp-load2.fail(i64 %arg, i8 addrspace(1)* %base1, i8 addrspace(1)* %base2) gc "statepoint-example" {
.b0:
; CHECK-LABEL: Verifying gc pointers in function: test.cmp-load2.fail
  %addr1 = getelementptr i8, i8 addrspace(1)* %base1, i64 %arg
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i8 addrspace(1)* %base2)]
  br label %.b1

.b1:
  %addr2 = getelementptr i8, i8 addrspace(1)* %base2, i64 8
  br label %.b2

.b2:
  %cmp = icmp eq i8 addrspace(1)* %addr1, %addr2
; CHECK-NEXT: Illegal use of unrelocated value found!
; CHECK-NEXT: Def:   %addr2 = getelementptr i8, i8 addrspace(1)* %base2, i64 8
; CHECK-NEXT: Use:   %val = load i8, i8 addrspace(1)* %addr2
  %val = load i8, i8 addrspace(1)* %addr2
  ret void
}

; Same as test.cmp.ok but with multiple safepoints within one BB. And the last
; one is in the very end of BB so that Contribution of this BB is empty.
define void @test.cmp.multi-sp.ok(i64 %arg, i8 addrspace(1)* %base1, i8 addrspace(1)* %base2) gc "statepoint-example" {
; CHECK-LABEL: Verifying gc pointers in function: test.cmp.multi-sp.ok
; CHECK-NEXT: No illegal uses found by SafepointIRVerifier in: test.cmp.multi-sp.ok
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i8 addrspace(1)* %base2)]
  %base2.relocated = call i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %safepoint_token, i32 0, i32 0) ; base2, base2
  %addr1 = getelementptr i8, i8 addrspace(1)* %base1, i64 %arg
  %safepoint_token2 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i8 addrspace(1)* %base2.relocated)]
  %base2.relocated2 = call i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %safepoint_token2, i32 0, i32 0) ; base2.relocated, base2.relocated
  %addr2 = getelementptr i8, i8 addrspace(1)* %base2, i64 %arg
  %cmp = icmp eq i8 addrspace(1)* %addr1, %addr2
  %safepoint_token3 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i8 addrspace(1)* %base2.relocated2)]
  ret void
}

; Function Attrs: nounwind
declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
declare i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token, i32, i32)

