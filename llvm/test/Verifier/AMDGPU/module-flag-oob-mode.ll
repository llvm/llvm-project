; Tests for IR verifier enforcement of the "amdgpu.oob.mode" module flag.
; The flag must use Module::Min (i32 8) merge behaviour, carry a constant
; integer value, and have no bits set outside the currently defined mask (0x3).

; RUN: split-file %s %t

; --- Negative: wrong merge behaviour (Override=4 instead of Min=8) ---
; RUN: not llvm-as %t/wrong-behavior.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=WRONG-BEHAVIOR

; --- Negative: non-integer value ---
; RUN: not llvm-as %t/non-integer.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NON-INTEGER

; --- Negative: unknown bits set ---
; RUN: not llvm-as %t/unknown-bits.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=UNKNOWN-BITS

; --- Positive: absent flag (no error expected) ---
; RUN: llvm-as %t/absent.ll --disable-output 2>&1 | count 0

; --- Positive: valid relaxed value 0x1 ---
; RUN: llvm-as %t/valid-0x1.ll --disable-output 2>&1 | count 0

; --- Positive: valid relaxed value 0x3 ---
; RUN: llvm-as %t/valid-0x3.ll --disable-output 2>&1 | count 0

; --- Positive: explicit strict value 0x0 ---
; RUN: llvm-as %t/valid-0x0.ll --disable-output 2>&1 | count 0

; WRONG-BEHAVIOR: 'amdgpu.oob.mode' module flag must use 'min' merge behaviour
; NON-INTEGER:    invalid value for 'min' module flag (expected constant non-negative integer)
; UNKNOWN-BITS:   'amdgpu.oob.mode' module flag has unknown bits set

;--- wrong-behavior.ll
; Override (i32 4) is not Min (i32 8).
!0 = !{i32 4, !"amdgpu.oob.mode", i32 1}
!llvm.module.flags = !{!0}

;--- non-integer.ll
; Min behaviour but float value instead of integer.
!0 = !{i32 8, !"amdgpu.oob.mode", float 1.0}
!llvm.module.flags = !{!0}

;--- unknown-bits.ll
; Bit 2 (0x4) is not defined in AMDGPUOOBMode.
!0 = !{i32 8, !"amdgpu.oob.mode", i32 4}
!llvm.module.flags = !{!0}

;--- absent.ll
; No "amdgpu.oob.mode" flag at all -- should be accepted.
define void @f() { ret void }

;--- valid-0x1.ll
; UntypedBuffer bit only.
!0 = !{i32 8, !"amdgpu.oob.mode", i32 1}
!llvm.module.flags = !{!0}

;--- valid-0x3.ll
; Both UntypedBuffer and TypedBuffer bits.
!0 = !{i32 8, !"amdgpu.oob.mode", i32 3}
!llvm.module.flags = !{!0}

;--- valid-0x0.ll
; Explicit strict mode.
!0 = !{i32 8, !"amdgpu.oob.mode", i32 0}
!llvm.module.flags = !{!0}
