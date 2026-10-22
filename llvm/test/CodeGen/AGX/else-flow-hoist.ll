; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s

; Both switches test n == 2 in a leaf block of an else side. MachineCSE's PRE
; hoists the two compares into their common dominator, the flow block that
; ends in the else of the first switch, where only the threads of the then
; side run: the threads with n == 2 then read a stale register and take the
; byte path. The compare must end up in the if block, before the if.

; CHECK-LABEL: k:
; CHECK-NOT: if_icmp
; CHECK: icmpsel ueq, {{r[0-9]+}}, {{r[0-9]+}}, 2, 0, 1
; CHECK: if_icmp r0l, nueq
; CHECK-NOT: icmpsel ueq, {{r[0-9]+}}, {{r[0-9]+}}, 2, 0, 1
; CHECK: stop

declare i32 @llvm.agx.thread.index.in.threadgroup()
define void @k(ptr addrspace(1) %p, ptr addrspace(1) %src) {
entry:
  %tid = call i32 @llvm.agx.thread.index.in.threadgroup()
  %ep = getelementptr inbounds i32, ptr addrspace(1) %src, i32 %tid
  %n = load i32, ptr addrspace(1) %ep, align 4
  %q = getelementptr inbounds i64, ptr addrspace(1) %p, i32 %tid
  switch i32 %n, label %a1 [ i32 2, label %a2 i32 4, label %a4 i32 8, label %a8 ]
a1:
  %v1 = load i8, ptr addrspace(1) %src, align 1
  %w1 = zext i8 %v1 to i64
  br label %mid
a2:
  %v2 = load i16, ptr addrspace(1) %src, align 2
  %w2 = zext i16 %v2 to i64
  br label %mid
a4:
  %v4 = load i32, ptr addrspace(1) %src, align 4
  %w4 = zext i32 %v4 to i64
  br label %mid
a8:
  %w8 = load i64, ptr addrspace(1) %src, align 8
  br label %mid
mid:
  %w = phi i64 [ %w1, %a1 ], [ %w2, %a2 ], [ %w4, %a4 ], [ %w8, %a8 ]
  switch i32 %n, label %b1 [ i32 2, label %b2 i32 4, label %b4 i32 8, label %b8 ]
b1:
  %t1 = trunc i64 %w to i8
  store i8 %t1, ptr addrspace(1) %q, align 1
  ret void
b2:
  %t2 = trunc i64 %w to i16
  store i16 %t2, ptr addrspace(1) %q, align 2
  ret void
b4:
  %t4 = trunc i64 %w to i32
  store i32 %t4, ptr addrspace(1) %q, align 4
  ret void
b8:
  store i64 %w, ptr addrspace(1) %q, align 8
  ret void
}
