; RUN: llc <%s -mtriple=arm64ec-pc-windows-msvc 2>&1 | FileCheck %s

; Check that we warn that x13, x14, x23, x24, x28 and float regs v16-v31
; will be clobbered on Arm64EC. With a note explaining the reason.

; CHECK: warning: inline asm clobber list contains reserved registers: X13, X14, X23, X24, X28
; CHECK-NEXT: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.
; CHECK-NEXT: note: x13 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: x14 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: x23 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: x24 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: x28 is clobbered by asynchronous signals when using Arm64EC.

define void @fn_x() {
entry:
  call void asm sideeffect "nop", "~{x13},~{x14},~{x23},~{x24},~{x28}"()
  ret void
}

; CHECK: warning: inline asm clobber list contains reserved registers: W13, W14, W23, W24, W28
; CHECK-NEXT: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.
; CHECK-NEXT: note: w13 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: w14 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: w23 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: w24 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: w28 is clobbered by asynchronous signals when using Arm64EC.

define void @fn_w() {
entry:
  call void asm sideeffect "nop", "~{w13},~{w14},~{w23},~{w24},~{w28}"()
  ret void
}

; CHECK: warning: inline asm clobber list contains reserved registers: Q16, Q17, Q18, Q19, Q20, Q21, Q22, Q23, Q24, Q25, Q26, Q27, Q28, Q29, Q30, Q31
; CHECK-NEXT: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.
; CHECK-NEXT: note: q16 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: q17 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: q18 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: q19 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: q20 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: q21 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: q22 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: q23 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: q24 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: q25 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: q26 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: q27 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: q28 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: q29 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: q30 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: q31 is clobbered by asynchronous signals when using Arm64EC.

; Aka "Q" registers.
define void @fn_float_v() {
entry:
  call void asm sideeffect "nop", "~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
  ret void
}

; CHECK: warning: inline asm clobber list contains reserved registers: D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26, D27, D28, D29, D30, D31
; CHECK-NEXT: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.
; CHECK-NEXT: note: d16 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: d17 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: d18 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: d19 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: d20 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: d21 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: d22 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: d23 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: d24 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: d25 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: d26 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: d27 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: d28 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: d29 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: d30 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: d31 is clobbered by asynchronous signals when using Arm64EC.

define void @fn_float_d() {
entry:
  call void asm sideeffect "nop", "~{d16},~{d17},~{d18},~{d19},~{d20},~{d21},~{d22},~{d23},~{d24},~{d25},~{d26},~{d27},~{d28},~{d29},~{d30},~{d31}"()
  ret void
}

; CHECK: warning: inline asm clobber list contains reserved registers: S16, S17, S18, S19, S20, S21, S22, S23, S24, S25, S26, S27, S28, S29, S30, S31
; CHECK-NEXT: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.
; CHECK-NEXT: note: s16 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: s17 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: s18 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: s19 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: s20 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: s21 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: s22 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: s23 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: s24 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: s25 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: s26 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: s27 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: s28 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: s29 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: s30 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: s31 is clobbered by asynchronous signals when using Arm64EC.

define void @fn_float_s() {
entry:
  call void asm sideeffect "nop", "~{s16},~{s17},~{s18},~{s19},~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29},~{s30},~{s31}"()
  ret void
}

; CHECK: warning: inline asm clobber list contains reserved registers: H16, H17, H18, H19, H20, H21, H22, H23, H24, H25, H26, H27, H28, H29, H30, H31
; CHECK-NEXT: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.
; CHECK-NEXT: note: h16 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: h17 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: h18 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: h19 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: h20 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: h21 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: h22 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: h23 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: h24 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: h25 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: h26 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: h27 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: h28 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: h29 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: h30 is clobbered by asynchronous signals when using Arm64EC.
; CHECK-NEXT: note: h31 is clobbered by asynchronous signals when using Arm64EC.

define void @fn_float_h() {
entry:
  call void asm sideeffect "nop", "~{h16},~{h17},~{h18},~{h19},~{h20},~{h21},~{h22},~{h23},~{h24},~{h25},~{h26},~{h27},~{h28},~{h29},~{h30},~{h31}"()
  ret void
}

; llvm does not currently handle the B registers so they are not tested.
