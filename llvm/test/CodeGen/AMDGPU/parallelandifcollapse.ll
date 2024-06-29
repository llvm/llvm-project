; RUN: llc -mtriple=r600 -mcpu=redwood -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck %s
;
; CFG flattening should use parallel-and mode to generate branch conditions and
; then merge if-regions with the same bodies.
;
; CHECK: AND_INT
; CHECK-NEXT: AND_INT
; CHECK-NEXT: OR_INT

; FIXME: For some reason having the allocas here allowed the flatten cfg pass
; to do its transformation, however now that we are using local memory for
; allocas, the transformation isn't happening.

define amdgpu_kernel void @_Z9chk1D_512v() #0 {
entry:
  %a0 = alloca i32, align 4, addrspace(5)
  %b0 = alloca i32, align 4, addrspace(5)
  %c0 = alloca i32, align 4, addrspace(5)
  %d0 = alloca i32, align 4, addrspace(5)
  %a1 = alloca i32, align 4, addrspace(5)
  %b1 = alloca i32, align 4, addrspace(5)
  %c1 = alloca i32, align 4, addrspace(5)
  %d1 = alloca i32, align 4, addrspace(5)
  %data = alloca i32, align 4, addrspace(5)
  %0 = load i32, ptr addrspace(5) %a0, align 4
  %1 = load i32, ptr addrspace(5) %b0, align 4
  %cmp = icmp ne i32 %0, %1
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  %2 = load i32, ptr addrspace(5) %c0, align 4
  %3 = load i32, ptr addrspace(5) %d0, align 4
  %cmp1 = icmp ne i32 %2, %3
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  store i32 1, ptr addrspace(5) %data, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %entry
  %4 = load i32, ptr addrspace(5) %a1, align 4
  %5 = load i32, ptr addrspace(5) %b1, align 4
  %cmp2 = icmp ne i32 %4, %5
  br i1 %cmp2, label %land.lhs.true3, label %if.end6

land.lhs.true3:                                   ; preds = %if.end
  %6 = load i32, ptr addrspace(5) %c1, align 4
  %7 = load i32, ptr addrspace(5) %d1, align 4
  %cmp4 = icmp ne i32 %6, %7
  br i1 %cmp4, label %if.then5, label %if.end6

if.then5:                                         ; preds = %land.lhs.true3
  store i32 1, ptr addrspace(5) %data, align 4
  br label %if.end6

if.end6:                                          ; preds = %if.then5, %land.lhs.true3, %if.end
  ret void
}
