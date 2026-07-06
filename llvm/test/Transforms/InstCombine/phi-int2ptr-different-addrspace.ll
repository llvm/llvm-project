; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; Verify that foldIntegerTypedPHI does not crash when a phi incoming value
; is a ptrtoint from a different address space than the inttoptr user targets.

define float @phi_ptrtoint_different_addrspace(ptr addrspace(1) noundef %ptr, i1 %cond) {
; CHECK-LABEL: @phi_ptrtoint_different_addrspace(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ALLOCA:%.*]] = alloca ptr addrspace(1), align 8, addrspace(4)
; CHECK-NEXT:    [[PTR_INT:%.*]] = ptrtoint ptr addrspace(1) [[PTR:%.*]] to i64
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[PHI:%.*]] = phi i64 [ [[PTR_INT]], [[ENTRY:%.*]] ], [ [[ALLOCA_INT:%.*]], [[LOOP_LATCH:%.*]] ]
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[LOOP_LATCH]], label [[EXIT:%.*]]
; CHECK:       loop.latch:
; CHECK-NEXT:    [[ALLOCA_INT]] = ptrtoint ptr addrspace(4) [[ALLOCA]] to i64
; CHECK-NEXT:    br label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    [[PHI_PTR:%.*]] = inttoptr i64 [[PHI]] to ptr addrspace(1)
; CHECK-NEXT:    [[VAL:%.*]] = load float, ptr addrspace(1) [[PHI_PTR]], align 4
; CHECK-NEXT:    ret float [[VAL]]
;
entry:
  %alloca = alloca ptr addrspace(1), align 8, addrspace(4)
  %ptr.int = ptrtoint ptr addrspace(1) %ptr to i64
  br label %loop

loop:
  %phi = phi i64 [ %ptr.int, %entry ], [ %alloca.int, %loop.latch ]
  br i1 %cond, label %loop.latch, label %exit

loop.latch:
  %alloca.int = ptrtoint ptr addrspace(4) %alloca to i64
  br label %loop

exit:
  %phi.ptr = inttoptr i64 %phi to ptr addrspace(1)
  %val = load float, ptr addrspace(1) %phi.ptr, align 4
  ret float %val
}
