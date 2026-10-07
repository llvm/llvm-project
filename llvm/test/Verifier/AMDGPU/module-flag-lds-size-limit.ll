; Tests for IR verifier enforcement of the "amdgpu.lds.size.limit" module flag.
; The flag must use Module::Error (i32 1) merge behavior and its value must be
; i32 65536 or i32 131072.

; RUN: split-file %s %t

; RUN: llvm-as %t/valid-64k.ll --disable-output
; RUN: llvm-as %t/valid-128k.ll --disable-output

; RUN: not llvm-as %t/wrong-behavior.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=WRONG-BEHAVIOR

; RUN: not llvm-as %t/non-integer.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NON-INT

; RUN: not llvm-as %t/zero.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=VALUE

; RUN: not llvm-as %t/other-value.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=VALUE

; RUN: not llvm-as %t/i64.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=VALUE

; WRONG-BEHAVIOR: 'amdgpu.lds.size.limit' module flag must use 'error' merge behaviour
; NON-INT:        'amdgpu.lds.size.limit' module flag must have a constant integer value
; VALUE:          'amdgpu.lds.size.limit' module flag must be i32 65536 or 131072

;--- valid-64k.ll
!0 = !{i32 1, !"amdgpu.lds.size.limit", i32 65536}
!llvm.module.flags = !{!0}

;--- valid-128k.ll
!0 = !{i32 1, !"amdgpu.lds.size.limit", i32 131072}
!llvm.module.flags = !{!0}

;--- wrong-behavior.ll
; Max (i32 7) is not Error (i32 1).
!0 = !{i32 7, !"amdgpu.lds.size.limit", i32 131072}
!llvm.module.flags = !{!0}

;--- non-integer.ll
!0 = !{i32 1, !"amdgpu.lds.size.limit", float 1.0}
!llvm.module.flags = !{!0}

;--- zero.ll
!0 = !{i32 1, !"amdgpu.lds.size.limit", i32 0}
!llvm.module.flags = !{!0}

;--- other-value.ll
!0 = !{i32 1, !"amdgpu.lds.size.limit", i32 100000}
!llvm.module.flags = !{!0}

;--- i64.ll
!0 = !{i32 1, !"amdgpu.lds.size.limit", i64 131072}
!llvm.module.flags = !{!0}
