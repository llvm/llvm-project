; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-default.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefixes=NOCHANGE,COMMON %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-empty.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefixes=NOCHANGE,COMMON %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-sram-ecc-off-xnack-on.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefixes=BOTH,COMMON %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-xnack-off.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefixes=XNACK,COMMON %s

; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-invalid.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=INVALID %s

; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-diff-triple.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=DIFFTRIPLE %s

; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-diff-cpu.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=DIFFCPU %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-none.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=NONE %s

; Test target id module flags.

; COMMON: !llvm.module.flags = !{!0}
; NOCHANGE: !0 = !{i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908"}
; BOTH: !0 = !{i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908:sram-ecc-:xnack+"}
; XNACK: !0 = !{i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908:xnack-"}
; NONE: !llvm.module.flags = !{!0, !1}
; NONE: !0 = !{i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908"}
; NONE: !1 = !{i32 1, !"foo", i32 37}

; INVALID: error: invalid module flag 'target-id': incorrect format ('amdgcn-amd-amdhsa--gfx908:xnack'
; DIFFTRIPLE: error: linking module flags 'target-id': IDs have conflicting values ('amdgcn-amd-amdpal--gfx908' from '{{.*}}' with 'amdgcn-amd-amdhsa--gfx908' from '{{.*}}'
; DIFFCPU: error: linking module flags 'target-id': IDs have conflicting values ('amdgcn-amd-amdhsa--gfx900' from '{{.*}}' with 'amdgcn-amd-amdhsa--gfx908' from '{{.*}}'

!llvm.module.flags = !{ !0 }
!0 = !{ i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908" }
