; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-default.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefixes=DEFAULT,COMMON %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-empty.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefixes=EMPTY,COMMON %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-sram-ecc-off-xnack-on.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefixes=BOTH,COMMON %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-xnack-off.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefixes=XNACK,COMMON %s

; RUN: not llvm-link %s %p/Inputs/module-flags-target-id-src-invalid.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=INVALID %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-diff-triple.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefixes=DIFFTRIPLE,COMMON %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-diff-cpu.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefixes=DIFFCPU,COMMON %s

; RUN: llvm-link %s %p/Inputs/module-flags-target-id-src-none.ll -S -o - \
; RUN:  2>&1 | FileCheck -check-prefix=NONE %s

; Test target id module flags.

; COMMON: !llvm.module.flags = !{!0}
; DEFAULT: !0 = !{i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908"}
; EMPTY: !0 = !{i32 8, !"target-id", !""}
; BOTH: !0 = !{i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908:sram-ecc-:xnack+"}
; XNACK: !0 = !{i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908:xnack-"}
; DIFFTRIPLE: !0 = !{i32 8, !"target-id", !"amdgcn-amd-amdpal--gfx908"}
; DIFFCPU: !0 = !{i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx900"}
; NONE: !llvm.module.flags = !{!0, !1}
; NONE: !0 = !{i32 8, !"target-id", !""}
; NONE: !1 = !{i32 1, !"foo", i32 37}

; INVALID: error: invalid module flag 'target-id': incorrect format ('amdgcn-amd-amdhsa--gfx908:xnack'

!llvm.module.flags = !{ !0 }
!0 = !{ i32 8, !"target-id", !"" }
