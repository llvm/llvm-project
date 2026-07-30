; Test that sramecc module flags are linked correctly with Module::Error behavior

; RUN: llvm-link -S %s %S/Inputs/amdgpu-sramecc-module-flag-0.ll -o - | FileCheck --check-prefix=BOTH-OFF %s
; RUN: not llvm-link -S %s %S/Inputs/amdgpu-sramecc-module-flag-1.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CONFLICT %s
; RUN: llvm-link -S %s %S/Inputs/amdgpu-sramecc-module-flag-any.ll -o - | FileCheck --check-prefix=ONE-OFF %s

; Test disabled + disabled = disabled
; BOTH-OFF: !llvm.module.flags = !{!0}
; BOTH-OFF: !0 = !{i32 1, !"amdgpu.sramecc", i32 0}

; Test disabled + enabled = error
; CONFLICT: linking module flags 'amdgpu.sramecc': IDs have conflicting values

; Test disabled + any = disabled
; ONE-OFF: !llvm.module.flags = !{!0, !1}
; ONE-OFF-DAG: !{{[0-9]}} = !{i32 1, !"amdgpu.sramecc", i32 0}
; ONE-OFF-DAG: !{{[0-9]}} = !{i32 {{[0-9]+}}, !"PIC Level", i32 {{[0-9]+}}}

define void @foo() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu.sramecc", i32 0}
