; The Metal pipeline legalizes agx-raise-to-air's hints (llvm/IR/AIR.h) that
; Metal has no form of: generic pointers become device pointers, integers of
; 65 to 128 bits become two i64 limbs, and 32-bit read-modify-writes without
; an AIR function become compare-exchange loops.
; RUN: opt -mtriple=air -passes=air-legalize-unsupported-ir -S %s | FileCheck %s
; RUN: opt -mtriple=air -passes=air-lower-atomic-rmw -S %s | FileCheck %s --check-prefix=RMW
; RUN: llc -mtriple=air -filetype=obj %s -o /dev/null

; CHECK-LABEL: define void @generic(
; (a GEP of an inttoptr address is integer arithmetic, gep-of-inttoptr.ll)
; CHECK: %p = inttoptr i64 %a to ptr addrspace(1)
; CHECK: [[A:%.*]] = add i64 %a, 4
; CHECK: %q = inttoptr i64 [[A]] to ptr addrspace(1)
; CHECK: load i32, ptr addrspace(1) %q
; CHECK: store i32 %v, ptr addrspace(1) %p
; CHECK-NOT: addrspace(4)
; CHECK: ret void
define void @generic(i64 %a) "air-kernel" {
  %p = inttoptr i64 %a to ptr addrspace(4)
  %q = getelementptr i32, ptr addrspace(4) %p, i64 1
  %v = load i32, ptr addrspace(4) %q
  %d = addrspacecast ptr addrspace(4) %p to ptr addrspace(1)
  store i32 %v, ptr addrspace(1) %d
  ret void
}

; CHECK-LABEL: define void @wide(
; CHECK-NOT: i128
; The high limb: the low one's carry into it.
; CHECK: %[[LO:[0-9]+]] = add i64
; CHECK: icmp ult i64 %[[LO]]
; CHECK: ret void
define void @wide(ptr addrspace(1) %out, i64 %x, i64 %y) "air-kernel" {
  %a = zext i64 %x to i128
  %b = shl i128 %a, 40
  %c = sext i64 %y to i128
  %s = add i128 %b, %c
  %abs = call i128 @llvm.abs.i128(i128 %s, i1 true)
  %neg = icmp slt i128 %s, 0
  %h = lshr i128 %abs, 64
  %ht = trunc i128 %h to i64
  %lt = trunc i128 %abs to i64
  %r = select i1 %neg, i64 %ht, i64 %lt
  store i64 %r, ptr addrspace(1) %out
  ret void
}

; RMW-LABEL: define void @inc(
; RMW: load volatile i32, ptr addrspace(1) %p
; RMW: atomicrmw.loop:
; RMW: icmp uge i32 %{{[0-9]+}}, %v
; RMW: call i32 @air.atomic.global.cmpxchg.weak.i32(
; RMW: br i1 %{{[0-9]+}}, label %{{.*}}, label %atomicrmw.loop
define void @inc(ptr addrspace(1) %p, i32 %v) "air-kernel" {
  %o = atomicrmw uinc_wrap ptr addrspace(1) %p, i32 %v monotonic
  store i32 %o, ptr addrspace(1) %p
  ret void
}

declare i128 @llvm.abs.i128(i128, i1)
