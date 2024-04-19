; RUN: opt -S -dxil-metadata-emit %s | FileCheck %s
target triple = "dxilv1.3-pc-shadermodel6.3-library"

; CHECK: !dx.shaderModel = !{![[SM:[0-9]+]]}
; CHECK: ![[SM]] = !{!"lib", i32 6, i32 3}
