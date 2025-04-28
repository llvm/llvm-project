; RUN: opt -S --dxil-prepare %s | FileCheck %s

; Test that only metadata nodes that are valid in DXIL are allowed through

target triple = "dxilv1.0-unknown-shadermodel6.0-compute"

; Function Attrs: noinline nounwind memory(readwrite, inaccessiblemem: none)
define void @main(i32* %ptr) {
entry:  
  ; metadata ID changes to 0 once the current !0 and !1 are removed
  ; since they aren't in the whitelist. range needs a payload.
  ; CHECK: %val = load i32, ptr %ptr, align 4, !range [[RANGEMD:![0-9]+]]
  %val = load i32, ptr %ptr, align 4, !range !2

  %cmp.i = icmp ult i32 1, 2
  br i1 %cmp.i, label %_Z4mainDv3_j.exit, label %_Z4mainDv3_j.exit, !llvm.loop !0

_Z4mainDv3_j.exit:                                ; preds = %for.body.i, %entry
  ret void
}

; CHECK: [[RANGEMD]] = !{i32 1, i32 5}
; this next check line checks that nothing comes after the above check line. 
; No more metadata should be necessary, the rest (the current 0 and 1)
; should be removed.
; CHECK-NOT: !{!"llvm.loop.mustprogress"}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{i32 1, i32 5}
