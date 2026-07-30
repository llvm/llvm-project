; RUN: llvm-link %S/Inputs/commandline.a.ll %S/Inputs/commandline.b.ll -S | FileCheck %s

; Verify that multiple input llvm.commandline metadata are linked together.

; CHECK-DAG: !llvm.commandline = !{!0, !1, !2}
; CHECK-DAG: !{{[0-2]}} = !{!"compiler -v1"}
; CHECK-DAG: !{{[0-2]}} = !{!"compiler -v2"}
; CHECK-DAG: !{{[0-2]}} = !{!"compiler -v3"}
