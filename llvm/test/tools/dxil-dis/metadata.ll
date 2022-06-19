; RUN: llc --filetype=obj %s -o - | dxil-dis 
target triple = "dxil-unknown-unknown"

!llvm.foo = !{!0}
!llvm.bar = !{!1}

!0 = !{i32 42}
!1 = !{!"Some MDString"}

; CHECK: !llvm.foo = !{!0}
; CHECK: !llvm.bar = !{!1}
; CHECK: !0 = !{i32 42}
; CHECK: !1 = !{!"Some MDString"}
