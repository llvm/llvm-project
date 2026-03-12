; Tests for IR verifier enforcement of the "amdgpu.buffer.oob.relaxed" and
; "amdgpu.tbuffer.oob.relaxed" module flags.  Each flag must use Module::Min
; (i32 8) merge behaviour, carry a constant integer value, and be 0 or 1.

; RUN: split-file %s %t

; --- Negative: wrong merge behaviour (Override=4 instead of Min=8) ---
; RUN: not llvm-as %t/wrong-behavior-buffer.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=WRONG-BUF
; RUN: not llvm-as %t/wrong-behavior-tbuffer.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=WRONG-TBUF

; --- Negative: non-integer value ---
; RUN: not llvm-as %t/non-integer-buffer.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NON-INT-BUF
; RUN: not llvm-as %t/non-integer-tbuffer.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NON-INT-TBUF

; --- Negative: value out of range (2 is not 0 or 1) ---
; RUN: not llvm-as %t/out-of-range-buffer.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=RANGE-BUF
; RUN: not llvm-as %t/out-of-range-tbuffer.ll --disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=RANGE-TBUF

; --- Positive: absent flags (no error expected) ---
; RUN: llvm-as %t/absent.ll --disable-output 2>&1 | count 0

; --- Positive: valid relaxed value 1 for buffer ---
; RUN: llvm-as %t/valid-buffer-1.ll --disable-output 2>&1 | count 0

; --- Positive: valid relaxed value 1 for tbuffer ---
; RUN: llvm-as %t/valid-tbuffer-1.ll --disable-output 2>&1 | count 0

; --- Positive: both flags set to 1 ---
; RUN: llvm-as %t/valid-both-1.ll --disable-output 2>&1 | count 0

; --- Positive: explicit strict value 0 ---
; RUN: llvm-as %t/valid-both-0.ll --disable-output 2>&1 | count 0

; --- Linker BUG: absent + relaxed(1) currently preserves relaxed(1) ---
; NOTE: This documents current IRMover behavior. For Module::Min, absent should
; ideally behave as 0 (strict), but early-return paths in IRMover bypass the
; absent->0 fixup.
; RUN: llvm-link %t/absent.ll %t/valid-buffer-1.ll -S -o - 2>&1 \
; RUN:   | FileCheck %s --check-prefix=BUG-LINK-ABSENT-PASSTHRU

; --- Linker: strict(0) + relaxed(1) -> strict(0) via Min ---
; RUN: llvm-link %t/valid-both-0.ll %t/valid-buffer-1.ll -S -o - 2>&1 \
; RUN:   | FileCheck %s --check-prefix=LINK-STRICT

; WRONG-BUF:    'amdgpu.buffer.oob.relaxed' module flag must use 'min' merge behaviour
; WRONG-TBUF:   'amdgpu.tbuffer.oob.relaxed' module flag must use 'min' merge behaviour
; NON-INT-BUF:  invalid value for 'min' module flag (expected constant non-negative integer)
; NON-INT-TBUF: invalid value for 'min' module flag (expected constant non-negative integer)
; RANGE-BUF:    'amdgpu.buffer.oob.relaxed' module flag must be 0 or 1
; RANGE-TBUF:   'amdgpu.tbuffer.oob.relaxed' module flag must be 0 or 1

; BUG-LINK-ABSENT-PASSTHRU: !{i32 8, !"amdgpu.buffer.oob.relaxed", i32 1}
; LINK-STRICT: !{i32 8, !"amdgpu.buffer.oob.relaxed", i32 0}

;--- wrong-behavior-buffer.ll
; Override (i32 4) is not Min (i32 8).
!0 = !{i32 4, !"amdgpu.buffer.oob.relaxed", i32 1}
!llvm.module.flags = !{!0}

;--- wrong-behavior-tbuffer.ll
!0 = !{i32 4, !"amdgpu.tbuffer.oob.relaxed", i32 1}
!llvm.module.flags = !{!0}

;--- non-integer-buffer.ll
; Min behaviour but float value instead of integer.
!0 = !{i32 8, !"amdgpu.buffer.oob.relaxed", float 1.0}
!llvm.module.flags = !{!0}

;--- non-integer-tbuffer.ll
!0 = !{i32 8, !"amdgpu.tbuffer.oob.relaxed", float 1.0}
!llvm.module.flags = !{!0}

;--- out-of-range-buffer.ll
; Value 2 is not a valid boolean (must be 0 or 1).
!0 = !{i32 8, !"amdgpu.buffer.oob.relaxed", i32 2}
!llvm.module.flags = !{!0}

;--- out-of-range-tbuffer.ll
!0 = !{i32 8, !"amdgpu.tbuffer.oob.relaxed", i32 2}
!llvm.module.flags = !{!0}

;--- absent.ll
; No OOB flags at all - should be accepted.
define void @f() { ret void }

;--- valid-buffer-1.ll
; Relaxed untyped buffer OOB.
!0 = !{i32 8, !"amdgpu.buffer.oob.relaxed", i32 1}
!llvm.module.flags = !{!0}

;--- valid-tbuffer-1.ll
; Relaxed typed buffer OOB.
!0 = !{i32 8, !"amdgpu.tbuffer.oob.relaxed", i32 1}
!llvm.module.flags = !{!0}

;--- valid-both-1.ll
; Both flags relaxed.
!0 = !{i32 8, !"amdgpu.buffer.oob.relaxed", i32 1}
!1 = !{i32 8, !"amdgpu.tbuffer.oob.relaxed", i32 1}
!llvm.module.flags = !{!0, !1}

;--- valid-both-0.ll
; Both flags explicitly strict.
!0 = !{i32 8, !"amdgpu.buffer.oob.relaxed", i32 0}
!1 = !{i32 8, !"amdgpu.tbuffer.oob.relaxed", i32 0}
!llvm.module.flags = !{!0, !1}
