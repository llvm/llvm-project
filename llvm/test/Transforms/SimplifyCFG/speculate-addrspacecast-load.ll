; RUN: opt -S -passes=simplifycfg < %s | FileCheck %s

; SimplifyCFG may only speculate a load through an addrspacecast when a
; dereferenceable pointer in the source address space is guaranteed to remain
; dereferenceable in the destination address space. Whether that holds is
; target knowledge exposed by Triple::isValidAddrSpaceCast().
;
; On AMDGPU, casts among flat (0), global (1), and constant (4) preserve
; dereferenceability, but a cast into local (3) does not. So a load reached
; through addrspacecast 4->1 can be speculated, while one reached through 4->3
; must not be.

target triple = "amdgcn-amd-amdhsa"

; The local cast (4->3) is not a dereferenceability-preserving cast on AMDGPU,
; so the dereferenceable(4) fact on the constant-AS pointer does not carry over
; and neither load may be speculated. The conditional branch must be preserved.
define i32 @no_speculate_local(ptr addrspace(4) dereferenceable(4) %p, i1 %c) {
; CHECK-LABEL: define i32 @no_speculate_local(
; CHECK:       entry:
; CHECK-NOT:     load
; CHECK:         br i1
; CHECK:       then:
; CHECK:         load i32, ptr addrspace(3)
; CHECK:       else:
; CHECK:         load i32, ptr addrspace(1)
entry:
  br i1 %c, label %then, label %else

then:
  %as3 = addrspacecast ptr addrspace(4) %p to ptr addrspace(3)
  %v1 = load i32, ptr addrspace(3) %as3, align 1
  br label %exit

else:
  %as1 = addrspacecast ptr addrspace(4) %p to ptr addrspace(1)
  %v2 = load i32, ptr addrspace(1) %as1, align 1
  br label %exit

exit:
  %res = phi i32 [ %v1, %then ], [ %v2, %else ]
  ret i32 %res
}

; The global cast (4->1) preserves dereferenceability on AMDGPU, so the load is
; known dereferenceable and SimplifyCFG can speculate it, folding the branch
; into a select.
define i32 @speculate_global(ptr addrspace(4) dereferenceable(4) %p, i1 %c) {
; CHECK-LABEL: define i32 @speculate_global(
; CHECK:         [[AS1:%.*]] = addrspacecast ptr addrspace(4) %p to ptr addrspace(1)
; CHECK:         [[V:%.*]] = load i32, ptr addrspace(1) [[AS1]]
; CHECK:         [[RES:%.*]] = select i1 %c, i32 [[V]], i32 0
; CHECK:         ret i32 [[RES]]
; CHECK-NOT:     br i1
entry:
  br i1 %c, label %load, label %exit

load:
  %as1 = addrspacecast ptr addrspace(4) %p to ptr addrspace(1)
  %v = load i32, ptr addrspace(1) %as1, align 1
  br label %exit

exit:
  %res = phi i32 [ %v, %load ], [ 0, %entry ]
  ret i32 %res
}
