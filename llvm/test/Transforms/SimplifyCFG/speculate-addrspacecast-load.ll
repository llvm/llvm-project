; RUN: opt -S -passes=simplifycfg < %s | FileCheck %s

; SimplifyCFG may only speculate a load through an addrspacecast when the cast
; is a no-op, i.e. it preserves the represented address. Whether a cast is a
; no-op is encoded in the data layout via the "as:<as>:<as>..." specifier,
; which lists address spaces whose mutual addrspacecasts are no-ops.
;
; Here generic(4) and global(1) are in the same no-op group, but local(3) is
; not. So a load reached through addrspacecast 4->1 is dereferenceable and can
; be speculated, while one reached through 4->3 must not be.

target datalayout = "e-p:64:64-p1:64:64-p3:64:64-p4:64:64-as:1:4"

; The local cast (4->3) is not a no-op, so the dereferenceable(4) fact on the
; generic pointer does not carry over and neither load may be speculated. The
; conditional branch must be preserved.
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

; The global cast (4->1) is a no-op per the data layout, so the load is known
; dereferenceable and SimplifyCFG can speculate it, folding the branch into a
; select.
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
