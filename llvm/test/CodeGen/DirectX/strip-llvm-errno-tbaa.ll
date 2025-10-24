; RUN: opt -S -dxil-prepare < %s | FileCheck %s

; Ensures that dxil-prepare will remove the llvm.errno.tbaa metadata

target triple = "dxil-unknown-shadermodel6.0-compute"

define void @main() {
entry:
  ret void
}

; CHECK-NOT: !llvm.errno.tbaa
; CHECK-NOT: {{^!}}

!llvm.errno.tbaa = !{!0}

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
