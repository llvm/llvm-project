; RUN: llc < %s -mcpu=cortex-a8 -align-neon-spills=0 | FileCheck %s
; RUN: llc < %s -mcpu=cortex-a8 -align-neon-spills=1 | FileCheck %s --check-prefix=NEON
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios"

; CHECK: f
; This function is forced to spill a double.
; Verify that the spill slot is properly aligned.
;
; The caller-saved r4 is used as a scratch register for stack realignment.
; CHECK: push {r4, r7, lr}
; CHECK: bfc r4, #0, #3
; CHECK: mov sp, r4
define void @f(double* nocapture %p) nounwind ssp "no-frame-pointer-elim"="true" {
entry:
  %0 = load double, double* %p, align 4
  tail call void asm sideeffect "", "~{d8},~{d9},~{d10},~{d11},~{d12},~{d13},~{d14},~{d15}"() nounwind
  tail call void @g() nounwind
  store double %0, double* %p, align 4
  ret void
}

; NEON: f
; NEON: push {r4, r7, lr}
; NEON: sub.w r4, sp, #64
; NEON: bfc r4, #0, #4
; Stack pointer must be updated before the spills.
; NEON: mov sp, r4
; NEON: vst1.64 {d8, d9, d10, d11}, [r4:128]!
; NEON: vst1.64 {d12, d13, d14, d15}, [r4:128]
; Stack pointer adjustment for the stack frame contents.
; This could legally happen before the spills.
; Since the spill slot is only 8 bytes, technically it would be fine to only
; subtract #8 here. That would leave sp less aligned than some stack slots,
; and would probably blow MFI's mind.
; NEON: sub sp, #16
; The epilog is free to use another scratch register than r4.
; NEON: add r[[R4:[0-9]+]], sp, #16
; NEON: vld1.64 {d8, d9, d10, d11}, [r[[R4]]:128]!
; NEON: vld1.64 {d12, d13, d14, d15}, [r[[R4]]:128]
; The stack pointer restore must happen after the reloads.
; NEON: mov sp,
; NEON: pop

declare void @g()

; Spill 7 d-registers.
define void @f7(double* nocapture %p) nounwind ssp "no-frame-pointer-elim"="true" {
entry:
  tail call void asm sideeffect "", "~{d8},~{d9},~{d10},~{d11},~{d12},~{d13},~{d14}"() nounwind
  ret void
}

; NEON: f7
; NEON: push {r4, r7, lr}
; NEON: sub.w r4, sp, #56
; NEON: bfc r4, #0, #4
; Stack pointer must be updated before the spills.
; NEON: mov sp, r4
; NEON: vst1.64 {d8, d9, d10, d11}, [r4:128]!
; NEON: vst1.64 {d12, d13}, [r4:128]
; NEON: vstr d14, [r4, #16]
; Epilog
; NEON: vld1.64 {d8, d9, d10, d11},
; NEON: vld1.64 {d12, d13},
; NEON: vldr d14,
; The stack pointer restore must happen after the reloads.
; NEON: mov sp,
; NEON: pop

; Spill 7 d-registers, leave a hole.
define void @f3plus4(double* nocapture %p) nounwind ssp "no-frame-pointer-elim"="true" {
entry:
  tail call void asm sideeffect "", "~{d8},~{d9},~{d10},~{d12},~{d13},~{d14},~{d15}"() nounwind
  ret void
}

; Aligned spilling only works for contiguous ranges starting from d8.
; The rest goes to the standard vpush instructions.
; NEON: f3plus4
; NEON: push {r4, r7, lr}
; NEON: vpush {d12, d13, d14, d15}
; NEON: sub.w r4, sp, #24
; NEON: bfc r4, #0, #4
; Stack pointer must be updated before the spills.
; NEON: mov sp, r4
; NEON: vst1.64 {d8, d9}, [r4:128]
; NEON: vstr d10, [r4, #16]
; Epilog
; NEON: vld1.64 {d8, d9},
; NEON: vldr d10, [{{.*}}, #16]
; The stack pointer restore must happen after the reloads.
; NEON: mov sp,
; NEON: vpop {d12, d13, d14, d15}
; NEON: pop
