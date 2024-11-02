; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s
target triple = "dxil-unknown-shadermodel6.7-library"

!llvm.used = !{!0}

!0 = !DISubroutineType(types: !1)
!1 = !{!2, !2, !2, !2}
!2 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)

; CHECK: !0 = !DISubroutineType(types: !1)
; CHECK: !1 = !{!2, !2, !2, !2}
; CHECK: !2 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
