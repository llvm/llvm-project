; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder < %s

; Function Attrs: nounwind
define void @_Z4testPiPf(i32* nocapture %pI, float* nocapture %pF) #0 {
entry:
  store i32 0, i32* %pI, align 4, !tbaa !{!"int", !0}
  ; CHECK: store i32 0, ptr %pI, align 4, !tbaa [[TAG_INT:!.*]]
  store float 1.000000e+00, float* %pF, align 4, !tbaa !2
  ; CHECK: store float 1.000000e+00, ptr %pF, align 4, !tbaa [[TAG_FLOAT:!.*]]
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = !{!"omnipotent char", !1}
!1 = !{!"Simple C/C++ TBAA"}
!2 = !{!"float", !0}

; CHECK: [[TAG_INT]] = !{[[TYPE_INT:!.*]], [[TYPE_INT]], i64 0}
; CHECK: [[TYPE_INT]] = !{!"int", [[TYPE_CHAR:!.*]]}
; CHECK: [[TYPE_CHAR]] = !{!"omnipotent char", !{{.*}}
; CHECK: [[TAG_FLOAT]] = !{[[TYPE_FLOAT:!.*]], [[TYPE_FLOAT]], i64 0}
; CHECK: [[TYPE_FLOAT]] = !{!"float", [[TYPE_CHAR]]}
