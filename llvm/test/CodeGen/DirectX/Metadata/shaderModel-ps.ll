; RUN: opt -S -dxil-metadata-emit %s | FileCheck %s
target triple = "dxil-pc-shadermodel5.0-pixel"

; CHECK: !dx.shaderModel = !{![[SM:[0-9]+]]}
; CHECK: ![[SM]] = !{!"ps", i32 5, i32 0}
