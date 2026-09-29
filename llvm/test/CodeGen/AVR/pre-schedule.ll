; RUN: llc < %s -mtriple=avr | FileCheck %s
target triple = "avr-unknown-unknown"

; The case illustrate DAG schedular may pre-schedule the node has
; ADJCALLSTACKDOWN parent, so ADJCALLSTACKUP may hold CallResource too long
; and make other calls can't be scheduled. If there's no other available node
; to schedule, the scheduler will try to rename the register by creating the
; copy to avoid the conflict which will fail because CallResource is not a real
; physical register.
;
; The issue is found by Tim on https://github.com/avr-rust/rust/issues/111 and
; discuss in http://lists.llvm.org/pipermail/llvm-dev/2018-October/127083.html.

define void @"main"() addrspace(1) {
start:
  %0 = or i64 undef, undef
   br i1 undef, label %mul_and_call, label %fail

  mul_and_call:
  %1 = mul i64 %0, %0
  call addrspace(1) void @"three_ints"(i64 undef, i64 %1, i64 %0)
; The CHECK line only want to make sure the following assertion message
; won't trigger due to create copy of artificial CallResource register.
; llc: llvm/lib/CodeGen/TargetRegisterInfo.cpp:203: const llvm::TargetRegisterClass* llvm::TargetRegisterInfo::getMinimalPhysRegClass(unsigned int, llvm::MVT) const: Assertion `BestRC && "Couldn't find the register class"' failed.
; CHECK: call    __muldi3
  ret void

  fail:
    ret void
}

declare void @"three_ints"(i64, i64, i64) addrspace(1)
