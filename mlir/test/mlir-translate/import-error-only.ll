; RUN: not mlir-translate %s -import-llvm -split-input-file -error-diagnostics-only 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not mlir-translate %s -import-llvm -split-input-file 2>&1 | FileCheck %s --check-prefix=ALL

; ERROR-NOT: warning:
; ALL:       warning:
define void @warning(i64 %n, ptr %A) {
entry:
  br label %end, !llvm.loop !0
end:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.disable_nonforced"}
!2 = !{!"llvm.loop.typo"}

; // -----

; ERROR: error:
; ALL:   error:
define i32 @error(ptr %dst) {
  indirectbr ptr %dst, [label %bb1, label %bb2]
bb1:
  ret i32 0
bb2:
  ret i32 1
}
