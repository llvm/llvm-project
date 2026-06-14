; Test xnack module flag linking with enabled flag
; RUN: not llvm-link -S %s %S/Inputs/amdgpu-xnack-module-flag-0.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CONFLICT %s
; RUN: llvm-link -S %s %S/Inputs/amdgpu-xnack-module-flag-1.ll -o - | FileCheck --check-prefix=BOTH-ON %s
; RUN: llvm-link -S %s %S/Inputs/amdgpu-xnack-module-flag-any.ll -o - | FileCheck --check-prefix=ONE-ON %s

; Test enabled + disabled = error
; CONFLICT: linking module flags 'amdgpu.xnack': IDs have conflicting values

; Test enabled + enabled = enabled
; BOTH-ON: !llvm.module.flags = !{!0}
; BOTH-ON: !0 = !{i32 1, !"amdgpu.xnack", i32 1}

; Test enabled + any = enabled
; ONE-ON: !llvm.module.flags = !{!0, !1}
; ONE-ON-DAG: !{{[0-9]}} = !{i32 1, !"amdgpu.xnack", i32 1}
; ONE-ON-DAG: !{{[0-9]}} = !{i32 {{[0-9]+}}, !"PIC Level", i32 {{[0-9]+}}}

define void @bar() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu.xnack", i32 1}
