; Test sramecc module flag linking with no flag (any)
; RUN: llvm-link -S %s %S/Inputs/amdgpu-sramecc-module-flag-0.ll -o - | FileCheck --check-prefix=OTHER-OFF %s
; RUN: llvm-link -S %s %S/Inputs/amdgpu-sramecc-module-flag-1.ll -o - | FileCheck --check-prefix=OTHER-ON %s
; RUN: llvm-link -S %s %S/Inputs/amdgpu-sramecc-module-flag-any.ll -o - | FileCheck --check-prefix=BOTH-ANY %s

; Test any + disabled = disabled
; OTHER-OFF: !llvm.module.flags = !{!0, !1}
; OTHER-OFF-DAG: !{{[0-9]}} = !{i32 {{[0-9]+}}, !"PIC Level", i32 {{[0-9]+}}}
; OTHER-OFF-DAG: !{{[0-9]}} = !{i32 1, !"amdgpu.sramecc", i32 0}

; Test any + enabled = enabled
; OTHER-ON: !llvm.module.flags = !{!0, !1}
; OTHER-ON-DAG: !{{[0-9]}} = !{i32 {{[0-9]+}}, !"PIC Level", i32 {{[0-9]+}}}
; OTHER-ON-DAG: !{{[0-9]}} = !{i32 1, !"amdgpu.sramecc", i32 1}

; Test any + any = any (no flag)
; BOTH-ANY: !llvm.module.flags = !{!0}
; BOTH-ANY: !0 = !{i32 {{[0-9]+}}, !"PIC Level", i32 2}

define void @baz() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 7, !"PIC Level", i32 2}
