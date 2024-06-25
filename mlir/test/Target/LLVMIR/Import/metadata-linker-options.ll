; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK: llvm.linker_options ["DEFAULTLIB:", "libcmt"]
!llvm.linker.options = !{!0}
!0 = !{!"DEFAULTLIB:", !"libcmt"}

; // -----

!llvm.linker.options = !{!0, !1, !2}
; CHECK: llvm.linker_options ["DEFAULTLIB:", "libcmt"]
!0 = !{!"DEFAULTLIB:", !"libcmt"}
; CHECK: llvm.linker_options ["DEFAULTLIB:", "libcmtd"]
!1 = !{!"DEFAULTLIB:", !"libcmtd"}
; CHECK: llvm.linker_options ["-lm"]
!2 = !{!"-lm"}
