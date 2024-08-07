; RUN: opt -S -dxil-metadata-emit %s | FileCheck %s

target triple = "dxilv1.3-unknown-shadermodel6.3-library"

define void @"?f1@@YAXXZ"() #0 {
entry:
  ret void
}

define void @"?f2@MyNamespace@@YAXXZ"() #0 {
entry:
  ret void
}

attributes #0 = { convergent noinline nounwind optnone "hlsl.export" }

; CHECK: !dx.exports = !{[[Exp1:![0-9]+]], [[Exp2:![0-9]+]]}
; CHECK: [[Exp1]] = !{ptr @"?f1@@YAXXZ", !"?f1@@YAXXZ"}
; CHECK: [[Exp2]] = !{ptr @"?f2@MyNamespace@@YAXXZ", !"?f2@MyNamespace@@YAXXZ"}
