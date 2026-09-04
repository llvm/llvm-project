; Tests for IR verifier enforcement of the "amdgpu.sramecc" module flag.
; The flag must use Module::Error (i32 1) merge behavior, carry a constant
; integer value, and be 0 or 1.

; RUN: split-file %s %t

; --- Negative: wrong merge behavior (Max=7 instead of Error=1) ---
; RUN: not llvm-as %t/wrong-behavior.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=WRONG-BEHAVIOR

; --- Negative: non-integer value ---
; RUN: not llvm-as %t/non-integer.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NON-INT

; --- Negative: missing value ---
; RUN: not llvm-as %t/missing-value.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=MISSING-VALUE

; --- Negative: value out of range (2 is not 0 or 1) ---
; RUN: not llvm-as %t/out-of-range.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=RANGE

; WRONG-BEHAVIOR: 'amdgpu.sramecc' module flag must use 'error' merge behaviour
; NON-INT:        'amdgpu.sramecc' module flag must have a constant integer value
; MISSING-VALUE:  incorrect number of operands in module flag
; RANGE:          'amdgpu.sramecc' module flag must be 0 or 1

;--- wrong-behavior.ll
; Max (i32 7) is not Error (i32 1).
!0 = !{i32 7, !"amdgpu.sramecc", i32 1}
!llvm.module.flags = !{!0}

;--- non-integer.ll
; Error behavior but float value instead of integer.
!0 = !{i32 1, !"amdgpu.sramecc", float 1.0}
!llvm.module.flags = !{!0}

;--- missing-value.ll
; Missing value field.
!0 = !{i32 1, !"amdgpu.sramecc"}
!llvm.module.flags = !{!0}

;--- out-of-range.ll
; Value 2 is out of range (must be 0 or 1).
!0 = !{i32 1, !"amdgpu.sramecc", i32 2}
!llvm.module.flags = !{!0}
