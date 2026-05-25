; RUN: opt -S -passes=simplifycfg < %s | FileCheck %s

; Verify SimplifyCFG does not speculate loads through non-noop addrspacecasts.

define i32 @no_speculate_addrspacecast_load(ptr addrspace(4) noalias nocapture readonly dereferenceable(4) %p) {
; CHECK-LABEL: define i32 @no_speculate_addrspacecast_load(
; CHECK:         entry:
; CHECK-NOT:     load
; CHECK:       then:
; CHECK:         load i32, ptr addrspace(3)
; CHECK:       else:
; CHECK:         load i32, ptr addrspace(1)
entry:
  %i = ptrtoint ptr addrspace(4) %p to i64
  %tag = lshr i64 %i, 61
  switch i64 %tag, label %else [
  i64 2, label %then
  ]

then:
  %as3 = addrspacecast ptr addrspace(4) %p to ptr addrspace(3)
  %load1 = load i32, ptr addrspace(3) %as3, align 1
  br label %exit

else:
  %as1 = addrspacecast ptr addrspace(4) %p to ptr addrspace(1)
  %load2 = load i32, ptr addrspace(1) %as1, align 1
  br label %exit

exit:
  %res = phi i32 [ %load1, %then ], [ %load2, %else ]
  %a = add nsw i32 %res, 1
  ret i32 %a
}
