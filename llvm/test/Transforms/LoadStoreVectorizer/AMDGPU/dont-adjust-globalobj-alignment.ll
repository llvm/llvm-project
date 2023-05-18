; RUN: opt -S -passes=load-store-vectorizer --mcpu=hawaii -mattr=+unaligned-access-mode,+unaligned-scratch-access,+max-private-element-size-16 < %s | FileCheck --match-full-lines %s

target triple = "amdgcn--"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

@G = internal addrspace(5) global [8 x i16] undef, align 1

; Verify that the alignment of the global remains at 1, even if we vectorize
; the stores.
;
; CHECK: @G = internal addrspace(5) global [8 x i16] undef, align 1

define void @private_store_2xi16_align2_not_alloca(ptr addrspace(5) %p, ptr addrspace(5) %r) {
; CHECK: define void @private_store_2xi16_align2_not_alloca(ptr addrspace(5) [[P:%.*]], ptr addrspace(5) [[R:%.*]]) #0 {
; CHECK-NEXT:    [[GEP0:%.*]] = getelementptr i16, ptr addrspace(5) @G, i32 0
; CHECK-NEXT:    store <2 x i16> <i16 1, i16 2>, ptr addrspace(5) [[GEP0]], align 1
; CHECK-NEXT:    ret void
;
  %gep0 = getelementptr i16, ptr addrspace(5) @G, i32 0
  %gep1 = getelementptr i16, ptr addrspace(5) @G, i32 1
  store i16 1, ptr addrspace(5) %gep0, align 1
  store i16 2, ptr addrspace(5) %gep1, align 1
  ret void
}
