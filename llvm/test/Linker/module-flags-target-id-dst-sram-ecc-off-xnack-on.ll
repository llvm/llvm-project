; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-default.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=NOCHANGE %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-empty.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefixes=NOCHANGE %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-sram-ecc-off-xnack-on.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=NOCHANGE %s

; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-diff-triple.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=DIFFTRIPLE %s

; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-diff-cpu.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=DIFFCPU %s

; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-xnack-off.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=NOXNACK %s

; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-invalid.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=INVALID %s

; Test target id module flags.

; NOCHANGE: !llvm.module.flags = !{!0}
; NOCHANGE: !0 = !{i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908:sram-ecc-:xnack+"}

; DIFFTRIPLE: error: linking module flags 'target-id': IDs have conflicting values ('amdgcn-amd-amdpal--gfx908' from '{{.*}}' with 'amdgcn-amd-amdhsa--gfx908:sram-ecc-:xnack+' from '{{.*}}'
; DIFFCPU: error: linking module flags 'target-id': IDs have conflicting values ('amdgcn-amd-amdhsa--gfx900' from '{{.*}}' with 'amdgcn-amd-amdhsa--gfx908:sram-ecc-:xnack+' from '{{.*}}'
; NOXNACK: error: linking module flags 'target-id': IDs have conflicting values ('amdgcn-amd-amdhsa--gfx908:xnack-' from '{{.*}}' with 'amdgcn-amd-amdhsa--gfx908:sram-ecc-:xnack+' from '{{.*}}'

; INVALID: error: invalid module flag 'target-id': incorrect format ('amdgcn-amd-amdhsa--gfx908:xnack'

!llvm.module.flags = !{ !0 }
!0 = !{ i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908:sram-ecc-:xnack+" }
