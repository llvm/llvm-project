; RUN: llc --filetype=obj %s -o - | dxil-dis
target triple = "dxil-unknown-shadermodel6.7-library"

define void @kernel(ptr addrspace(1)) {
    ret void
}

!llvm.foo = !{!0}
!llvm.bar = !{!1}
!llvm.baz = !{!2}

!0 = !{i32 42}
!1 = !{!"Some MDString"}
!2 = !{ptr @kernel}

; CHECK: !llvm.foo = !{!0}
; CHECK: !llvm.bar = !{!1}
; CHECK: !llvm.baz = !{!2}
; CHECK: !0 = !{i32 42}
; CHECK: !1 = !{!"Some MDString"}
; CHECK: !2 = !{void (i8 addrspace(1)*)* @kernel}
