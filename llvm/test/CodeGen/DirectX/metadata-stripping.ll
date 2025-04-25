; RUN: opt -S --dxil-prepare %s | FileCheck %s

; This test tests the whitelist inside of DxilPrepare.cpp.
; It ensures that certain metadata nodes are removed that aren't
; in the whitelist, and that certain nodes may remain that
; are on the whitelist.

target triple = "dxilv1.0-unknown-shadermodel6.0-compute"

; Function Attrs: noinline nounwind memory(readwrite, inaccessiblemem: none)
define void @main(i32* %ptr) {
entry:  
  ; metadata ID changes to 0 once the current !0 and !1 are removed
  ; since they aren't in the whitelist. range needs a payload.
  ; CHECK: %val = load i32, ptr %ptr, align 4, !range !0
  %val = load i32, ptr %ptr, align 4, !range !2

  ; dx.nonuniform is a valid metadata node kind on the whitelist,
  ; so give it a bogus payload and ensure it sticks around
  ; CHECK-next: %cmp.i1.not = icmp eq i32 1, 0, !dx.nonuniform !0
  %cmp.i1.not = icmp eq i32 1, 0, !dx.nonuniform !2
  br i1 %cmp.i1.not, label %_Z4mainDv3_j.exit, label %for.body.i

for.body.i:                                       ; preds = %entry
  %cmp.i = icmp ult i32 1, 2
  br i1 %cmp.i, label %for.body.i, label %_Z4mainDv3_j.exit, !llvm.loop !0

_Z4mainDv3_j.exit:                                ; preds = %for.body.i, %entry
  ret void
}

; CHECK: !0 = !{i32 1, i32 5}
; this next check line checks that nothing comes after the above check line. 
; No more metadata should be necessary, the rest (the current 0 and 1)
; should be removed.
; CHECK-NOT: !{!"llvm.loop.mustprogress"}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{i32 1, i32 5}
