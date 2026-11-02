; RUN: opt -disable-output -passes=print-dx-shader-flags %s 2>&1 | FileCheck %s --check-prefix=LIB
; RUN: opt -S -passes=dxil-translate-metadata %s | FileCheck %s --check-prefix=MD

; An unused declaration requires Barycentrics, but must not affect unrelated
; library entries or helper functions.
target triple = "dxil-pc-shadermodel6.8-library"
define void @barycentric() #0 { ret void }
define void @ordinary() #0 { ret void }
define void @vertex() #1 { ret void }
define void @helper() { ret void }
attributes #0 = { "hlsl.shader"="pixel" }
attributes #1 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0, !4, !5}
!0 = !{ptr @barycentric, !1, null}
!1 = !{!2}
!2 = !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !3, i32 3, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 0}
!4 = !{ptr @ordinary, null, null}
!5 = !{ptr @vertex, null, null}

; LIB: Combined Shader Flags for Module
; LIB-NEXT: ; Shader Flags Value: 0x00400000
; LIB: ;       Barycentrics
; LIB: Function barycentric : 0x00400000
; LIB: Function ordinary : 0x00000000
; LIB: Function vertex : 0x00000000
; LIB: Function helper : 0x00000000
; MD: !dx.entryPoints = !{![[LIBREC:[0-9]+]], ![[BARY:[0-9]+]], ![[ORD:[0-9]+]], ![[VS:[0-9]+]]}
; MD-DAG: ![[LIBREC]] = !{null, !"", null, null, ![[LPROPS:[0-9]+]]}
; MD-DAG: ![[LPROPS]] = !{i32 0, i64 4194304}
; MD-DAG: ![[BARY]] = !{ptr @barycentric, !"barycentric", !{{[0-9]+}}, null, ![[BPROPS:[0-9]+]]}
; MD-DAG: ![[ORD]] = !{ptr @ordinary, !"ordinary", null, null, ![[BPROPS]]}
; MD-DAG: ![[VS]] = !{ptr @vertex, !"vertex", null, null, ![[VPROPS:[0-9]+]]}
; MD-DAG: ![[BPROPS]] = !{i32 8, i32 0}
; MD-DAG: ![[VPROPS]] = !{i32 8, i32 1}
