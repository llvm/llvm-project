; RUN: opt -passes='sccp,instcombine' -S %s | FileCheck %s

define i32 @src(i32 %v0_u8, i32 %v1_u8) {
; CHECK-LABEL: define {{.*}}i32 @src(
; CHECK-NOT: lshr
; CHECK-NOT: xor
; CHECK: ret i32 %v1_u8
entry:
  %v0.lo = icmp sge i32 %v0_u8, 0
  %v0.hi = icmp sle i32 %v0_u8, 1
  %v0.ok = and i1 %v0.lo, %v0.hi
  call void @llvm.assume(i1 %v0.ok)
  %v1.lo = icmp sge i32 %v1_u8, 1
  %v1.hi = icmp sle i32 %v1_u8, 2
  %v1.ok = and i1 %v1.lo, %v1.hi
  call void @llvm.assume(i1 %v1.ok)
  %shr = lshr i32 %v0_u8, %v1_u8
  %xor = xor i32 %v1_u8, %shr
  ret i32 %xor
}
