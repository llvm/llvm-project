; RUN: llvm-as < %s | llvm-dis | FileCheck %s


define void @atomic_inc(ptr %ptr0, ptr addrspace(1) %ptr1, ptr addrspace(3) %ptr3) {
  ; CHECK: atomicrmw uinc_wrap ptr %ptr0, i32 42 syncscope("agent") seq_cst, align 4
  %result0 = call i32 @llvm.amdgcn.atomic.inc.i32.p0(ptr %ptr0, i32 42, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw uinc_wrap ptr addrspace(1) %ptr1, i32 43 syncscope("agent") seq_cst, align 4
  %result1 = call i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) %ptr1, i32 43, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw uinc_wrap ptr addrspace(3) %ptr3, i32 46 syncscope("agent") seq_cst, align 4
  %result2 = call i32 @llvm.amdgcn.atomic.inc.i32.p3(ptr addrspace(3) %ptr3, i32 46, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw uinc_wrap ptr %ptr0, i64 48 syncscope("agent") seq_cst, align 8
  %result3 = call i64 @llvm.amdgcn.atomic.inc.i64.p0(ptr %ptr0, i64 48, i32 0, i32 0, i1 false)

 ; CHECK: atomicrmw uinc_wrap ptr addrspace(1) %ptr1, i64 45 syncscope("agent") seq_cst, align 8
  %result4 = call i64 @llvm.amdgcn.atomic.inc.i64.p1(ptr addrspace(1) %ptr1, i64 45, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw uinc_wrap ptr addrspace(3) %ptr3, i64 4345 syncscope("agent") seq_cst, align 8
  %result5 = call i64 @llvm.amdgcn.atomic.inc.i64.p3(ptr addrspace(3) %ptr3, i64 4345, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw volatile uinc_wrap ptr addrspace(3) %ptr3, i64 4345 syncscope("agent") seq_cst, align 8
  %result6 = call i64 @llvm.amdgcn.atomic.inc.i64.p3(ptr addrspace(3) %ptr3, i64 4345, i32 0, i32 0, i1 true)
  ret void
}

define void @atomic_dec(ptr %ptr0, ptr addrspace(1) %ptr1, ptr addrspace(3) %ptr3) {
  ; CHECK: atomicrmw udec_wrap ptr %ptr0, i32 42 syncscope("agent") seq_cst, align 4
  %result0 = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %ptr0, i32 42, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw udec_wrap ptr addrspace(1) %ptr1, i32 43 syncscope("agent") seq_cst, align 4
  %result1 = call i32 @llvm.amdgcn.atomic.dec.i32.p1(ptr addrspace(1) %ptr1, i32 43, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw udec_wrap ptr addrspace(3) %ptr3, i32 46 syncscope("agent") seq_cst, align 4
  %result2 = call i32 @llvm.amdgcn.atomic.dec.i32.p3(ptr addrspace(3) %ptr3, i32 46, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw udec_wrap ptr %ptr0, i64 48 syncscope("agent") seq_cst, align 8
  %result3 = call i64 @llvm.amdgcn.atomic.dec.i64.p0(ptr %ptr0, i64 48, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw udec_wrap ptr addrspace(1) %ptr1, i64 45 syncscope("agent") seq_cst, align 8
  %result4 = call i64 @llvm.amdgcn.atomic.dec.i64.p1(ptr addrspace(1) %ptr1, i64 45, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw udec_wrap ptr addrspace(3) %ptr3, i64 4345 syncscope("agent") seq_cst, align 8
  %result5 = call i64 @llvm.amdgcn.atomic.dec.i64.p3(ptr addrspace(3) %ptr3, i64 4345, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw volatile udec_wrap ptr addrspace(3) %ptr3, i64 4345 syncscope("agent") seq_cst, align 8
  %result6 = call i64 @llvm.amdgcn.atomic.dec.i64.p3(ptr addrspace(3) %ptr3, i64 4345, i32 0, i32 0, i1 true)
  ret void
}

; Test some invalid ordering handling
define void @ordering(ptr %ptr0, ptr addrspace(1) %ptr1, ptr addrspace(3) %ptr3) {
  ; CHECK: atomicrmw volatile uinc_wrap ptr %ptr0, i32 42 syncscope("agent") seq_cst, align 4
  %result0 = call i32 @llvm.amdgcn.atomic.inc.i32.p0(ptr %ptr0, i32 42, i32 -1, i32 0, i1 true)

  ; CHECK: atomicrmw volatile uinc_wrap ptr addrspace(1) %ptr1, i32 43 syncscope("agent") seq_cst, align 4
  %result1 = call i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) %ptr1, i32 43, i32 0, i32 0, i1 true)

  ; CHECK: atomicrmw uinc_wrap ptr addrspace(1) %ptr1, i32 43 syncscope("agent") seq_cst, align 4
  %result2 = call i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) %ptr1, i32 43, i32 1, i32 0, i1 false)

  ; CHECK: atomicrmw volatile uinc_wrap ptr addrspace(1) %ptr1, i32 43 syncscope("agent") monotonic, align 4
  %result3 = call i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) %ptr1, i32 43, i32 2, i32 0, i1 true)

  ; CHECK: atomicrmw uinc_wrap ptr addrspace(1) %ptr1, i32 43 syncscope("agent") seq_cst, align 4
  %result4 = call i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) %ptr1, i32 43, i32 3, i32 0, i1 false)

  ; CHECK: atomicrmw volatile udec_wrap ptr %ptr0, i32 42 syncscope("agent") seq_cst, align 4
  %result5 = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %ptr0, i32 42, i32 0, i32 4, i1 true)

  ; CHECK: atomicrmw udec_wrap ptr %ptr0, i32 42 syncscope("agent") seq_cst, align 4
  %result6 = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %ptr0, i32 42, i32 0, i32 5, i1 false)

  ; CHECK: atomicrmw volatile udec_wrap ptr %ptr0, i32 42 syncscope("agent") seq_cst, align 4
  %result7 = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %ptr0, i32 42, i32 0, i32 6, i1 true)

  ; CHECK: atomicrmw udec_wrap ptr %ptr0, i32 42 syncscope("agent") seq_cst, align 4
  %result8 = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %ptr0, i32 42, i32 0, i32 7, i1 false)

  ; CHECK:= atomicrmw volatile udec_wrap ptr %ptr0, i32 42 syncscope("agent") seq_cst, align 4
  %result9 = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %ptr0, i32 42, i32 0, i32 8, i1 true)

  ; CHECK:= atomicrmw volatile udec_wrap ptr addrspace(1) %ptr1, i32 43 syncscope("agent") seq_cst, align 4
  %result10 = call i32 @llvm.amdgcn.atomic.dec.i32.p1(ptr addrspace(1) %ptr1, i32 43, i32 3, i32 0, i1 true)
  ret void
}

define void @immarg_violations(ptr %ptr0, i32 %val32, i1 %val1) {
  ; CHECK: atomicrmw udec_wrap ptr %ptr0, i32 42 syncscope("agent") seq_cst, align 4
  %result0 = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %ptr0, i32 42, i32 %val32, i32 0, i1 false)

; CHECK: atomicrmw udec_wrap ptr %ptr0, i32 42 syncscope("agent") monotonic, align 4
  %result1 = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %ptr0, i32 42, i32 2, i32 %val32, i1 false)

  ; CHECK: atomicrmw volatile udec_wrap ptr %ptr0, i32 42 syncscope("agent") monotonic, align 4
  %result2 = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %ptr0, i32 42, i32 2, i32 0, i1 %val1)
  ret void
}

declare i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) nocapture, i32, i32 immarg, i32 immarg, i1 immarg) #0
declare i32 @llvm.amdgcn.atomic.inc.i32.p3(ptr addrspace(3) nocapture, i32, i32 immarg, i32 immarg, i1 immarg) #0
declare i32 @llvm.amdgcn.atomic.inc.i32.p0(ptr nocapture, i32, i32 immarg, i32 immarg, i1 immarg) #0
declare i64 @llvm.amdgcn.atomic.inc.i64.p1(ptr addrspace(1) nocapture, i64, i32 immarg, i32 immarg, i1 immarg) #0
declare i64 @llvm.amdgcn.atomic.inc.i64.p3(ptr addrspace(3) nocapture, i64, i32 immarg, i32 immarg, i1 immarg) #0
declare i64 @llvm.amdgcn.atomic.inc.i64.p0(ptr nocapture, i64, i32 immarg, i32 immarg, i1 immarg) #0

declare i32 @llvm.amdgcn.atomic.dec.i32.p1(ptr addrspace(1) nocapture, i32, i32 immarg, i32 immarg, i1 immarg) #0
declare i32 @llvm.amdgcn.atomic.dec.i32.p3(ptr addrspace(3) nocapture, i32, i32 immarg, i32 immarg, i1 immarg) #0
declare i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr nocapture, i32, i32 immarg, i32 immarg, i1 immarg) #0
declare i64 @llvm.amdgcn.atomic.dec.i64.p1(ptr addrspace(1) nocapture, i64, i32 immarg, i32 immarg, i1 immarg) #0
declare i64 @llvm.amdgcn.atomic.dec.i64.p3(ptr addrspace(3) nocapture, i64, i32 immarg, i32 immarg, i1 immarg) #0
declare i64 @llvm.amdgcn.atomic.dec.i64.p0(ptr nocapture, i64, i32 immarg, i32 immarg, i1 immarg) #0

attributes #0 = { argmemonly nounwind willreturn }
