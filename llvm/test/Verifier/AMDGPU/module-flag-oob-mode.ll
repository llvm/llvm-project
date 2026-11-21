; Tests for IR verifier enforcement of the "amdgpu.buffer.oob.mode" and
; "amdgpu.tbuffer.oob.mode" module flags.  Each flag must use Module::Max
; (i32 7) merge behaviour, carry a constant integer value, and be 0, 1, or 2.

; RUN: split-file %s %t

; --- Negative: wrong merge behaviour (Override=4 instead of Max=7) ---
; RUN: not llvm-as %t/wrong-behavior-buffer.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=WRONG-BUF
; RUN: not llvm-as %t/wrong-behavior-tbuffer.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=WRONG-TBUF

; --- Negative: non-integer value ---
; RUN: not llvm-as %t/non-integer-buffer.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NON-INT-BUF
; RUN: not llvm-as %t/non-integer-tbuffer.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NON-INT-TBUF

; --- Negative: value out of range (3 is not 0, 1, or 2) ---
; RUN: not llvm-as %t/out-of-range-buffer.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=RANGE-BUF
; RUN: not llvm-as %t/out-of-range-tbuffer.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=RANGE-TBUF

; --- Positive: absent flags (no error expected) ---
; RUN: llvm-as %t/absent.ll --disable-output 2>&1 | count 0

; --- Positive: valid any value 0 ---
; RUN: llvm-as %t/valid-both-0.ll --disable-output 2>&1 | count 0

; --- Positive: valid relaxed value 1 for buffer ---
; RUN: llvm-as %t/valid-buffer-1.ll --disable-output 2>&1 | count 0

; --- Positive: valid relaxed value 1 for tbuffer ---
; RUN: llvm-as %t/valid-tbuffer-1.ll --disable-output 2>&1 | count 0

; --- Positive: valid strict value 2 ---
; RUN: llvm-as %t/valid-both-2.ll --disable-output 2>&1 | count 0

; WRONG-BUF:    'amdgpu.buffer.oob.mode' module flag must use 'max' merge behaviour
; WRONG-TBUF:   'amdgpu.tbuffer.oob.mode' module flag must use 'max' merge behaviour
; NON-INT-BUF:  invalid value for 'max' module flag (expected constant integer)
; NON-INT-TBUF: invalid value for 'max' module flag (expected constant integer)
; RANGE-BUF:    'amdgpu.buffer.oob.mode' module flag must be 0, 1, or 2
; RANGE-TBUF:   'amdgpu.tbuffer.oob.mode' module flag must be 0, 1, or 2

;--- wrong-behavior-buffer.ll
; Override (i32 4) is not Max (i32 7).
!0 = !{i32 4, !"amdgpu.buffer.oob.mode", i32 1}
!llvm.module.flags = !{!0}

;--- wrong-behavior-tbuffer.ll
!0 = !{i32 4, !"amdgpu.tbuffer.oob.mode", i32 1}
!llvm.module.flags = !{!0}

;--- non-integer-buffer.ll
; Max behaviour but float value instead of integer.
!0 = !{i32 7, !"amdgpu.buffer.oob.mode", float 1.0}
!llvm.module.flags = !{!0}

;--- non-integer-tbuffer.ll
!0 = !{i32 7, !"amdgpu.tbuffer.oob.mode", float 1.0}
!llvm.module.flags = !{!0}

;--- out-of-range-buffer.ll
; Value 3 is out of range (must be 0, 1, or 2).
!0 = !{i32 7, !"amdgpu.buffer.oob.mode", i32 3}
!llvm.module.flags = !{!0}

;--- out-of-range-tbuffer.ll
!0 = !{i32 7, !"amdgpu.tbuffer.oob.mode", i32 3}
!llvm.module.flags = !{!0}

;--- absent.ll
; No OOB flags at all - should be accepted.
define void @f() { ret void }

;--- valid-both-0.ll
; Both flags explicitly any.
!0 = !{i32 7, !"amdgpu.buffer.oob.mode", i32 0}
!1 = !{i32 7, !"amdgpu.tbuffer.oob.mode", i32 0}
!llvm.module.flags = !{!0, !1}

;--- valid-buffer-1.ll
; Relaxed untyped buffer OOB.
!0 = !{i32 7, !"amdgpu.buffer.oob.mode", i32 1}
!llvm.module.flags = !{!0}

;--- valid-tbuffer-1.ll
; Relaxed typed buffer OOB.
!0 = !{i32 7, !"amdgpu.tbuffer.oob.mode", i32 1}
!llvm.module.flags = !{!0}

;--- valid-both-2.ll
; Both flags strict.
!0 = !{i32 7, !"amdgpu.buffer.oob.mode", i32 2}
!1 = !{i32 7, !"amdgpu.tbuffer.oob.mode", i32 2}
!llvm.module.flags = !{!0, !1}
