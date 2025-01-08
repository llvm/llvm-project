;; Check that an unknown --target-abi is ignored and the triple-implied ABI is
;; used.
; RUN: llc --mtriple=loongarch32-linux-gnu --target-abi=foo --mattr=+d < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=ILP32D,UNKNOWN
; RUN: llc --mtriple=loongarch64-linux-gnu --target-abi=foo --mattr=+d < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=LP64D,UNKNOWN

; UNKNOWN: warning: the 'foo' is not a recognized ABI for this target, ignoring and using triple-implied ABI

;; Check that --target-abi takes precedence over triple-supplied ABI modifiers.
; RUN: llc --mtriple=loongarch32-linux-gnusf --target-abi=ilp32d --mattr=+d < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=ILP32D,CONFLICT-ILP32D
; RUN: llc --mtriple=loongarch64-linux-gnusf --target-abi=lp64d --mattr=+d < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=LP64D,CONFLICT-LP64D

; CONFLICT-ILP32D: warning: triple-implied ABI conflicts with provided target-abi 'ilp32d', using target-abi
; CONFLICT-LP64D:  warning: triple-implied ABI conflicts with provided target-abi 'lp64d', using target-abi

;; Check that no warning is reported when there is no environment component in
;; triple-supplied ABI modifiers and --target-abi is used.
; RUN: llc --mtriple=loongarch64-linux --target-abi=lp64d --mattr=+d < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=LP64D,NO-WARNING

; NO-WARNING-NOT:  warning: triple-implied ABI conflicts with provided target-abi 'lp64d', using target-abi

;; Check that ILP32-on-LA64 and LP64-on-LA32 combinations are handled properly.
; RUN: llc --mtriple=loongarch64 --target-abi=ilp32d --mattr=+d < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=LP64D,32ON64
; RUN: llc --mtriple=loongarch32 --target-abi=lp64d --mattr=+d < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=ILP32D,64ON32

; 32ON64: warning: 32-bit ABIs are not supported for 64-bit targets, ignoring and using triple-implied ABI
; 64ON32: warning: 64-bit ABIs are not supported for 32-bit targets, ignoring and using triple-implied ABI

;; Check that target-abi is invalid but triple-implied ABI is valid, use triple-implied ABI
; RUN: llc --mtriple=loongarch64-linux-gnusf --target-abi=lp64f --mattr=-f < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=LP64S,LP64S-LP64F-NOF
; RUN: llc --mtriple=loongarch64-linux-gnusf --target-abi=lp64d --mattr=-d < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=LP64S,LP64S-LP64D-NOD

; LP64S-LP64F-NOF: warning: the 'lp64f' ABI can't be used for a target that doesn't support the 'F' instruction set, ignoring and using triple-implied ABI
; LP64S-LP64D-NOD: warning: the 'lp64d' ABI can't be used for a target that doesn't support the 'D' instruction set, ignoring and using triple-implied ABI

;; Check that both target-abi and triple-implied ABI are invalid, use feature-implied ABI
; RUN: llc --mtriple=loongarch64-linux-gnuf64 --target-abi=lp64f --mattr=-f < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=LP64S,LP64D-LP64F-NOF
; RUN: llc --mtriple=loongarch64 --target-abi=lp64f --mattr=-f < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=LP64S,LP64D-LP64F-NOF

; LP64D-LP64F-NOF: warning: both target-abi and the triple-implied ABI are invalid, ignoring and using feature-implied ABI

;; Check that triple-implied ABI are invalid, use feature-implied ABI
; RUN: llc --mtriple=loongarch64 --mattr=-f < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=LP64S,LP64D-NONE-NOF

; LP64D-NONE-NOF: warning: the triple-implied ABI is invalid, ignoring and using feature-implied ABI

define float @f(float %a) {
; ILP32D-LABEL: f:
; ILP32D:       # %bb.0:
; ILP32D-NEXT:    addi.w $a0, $zero, 1
; ILP32D-NEXT:    movgr2fr.w $fa1, $a0
; ILP32D-NEXT:    ffint.s.w $fa1, $fa1
; ILP32D-NEXT:    fadd.s $fa0, $fa0, $fa1
; ILP32D-NEXT:    ret
;
; LP64D-LABEL: f:
; LP64D:       # %bb.0:
; LP64D-NEXT:    vldi $vr1, -1168
; LP64D-NEXT:    fadd.s $fa0, $fa0, $fa1
; LP64D-NEXT:    ret
;
; LP64S-LP64F-NOF-LABEL: f:
; LP64S-LP64F-NOF:    bl %plt(__addsf3)
;
; LP64S-LP64D-NOD-LABEL: f:
; LP64S-LP64D-NOD:       # %bb.0:
; LP64S-LP64D-NOD-NEXT:    movgr2fr.w $fa0, $a0
; LP64S-LP64D-NOD-NEXT:    addi.w $a0, $zero, 1
; LP64S-LP64D-NOD-NEXT:    movgr2fr.w $fa1, $a0
; LP64S-LP64D-NOD-NEXT:    ffint.s.w $fa1, $fa1
; LP64S-LP64D-NOD-NEXT:    fadd.s $fa0, $fa0, $fa1
; LP64S-LP64D-NOD-NEXT:    movfr2gr.s $a0, $fa0
; LP64S-LP64D-NOD-NEXT:    ret
;
; LP64D-LP64F-NOF-LABEL: f:
; LP64D-LP64F-NOF:    bl %plt(__addsf3)
;
; LP64D-NONE-NOF-LABEL: f:
; LP64D-NONE-NOF:    bl %plt(__addsf3)
  %1 = fadd float %a, 1.0
  ret float %1
}

define double @g(double %a) {
; ILP32D-LABEL: g:
; ILP32D:       # %bb.0:
; ILP32D-NEXT:    addi.w $a0, $zero, 1
; ILP32D-NEXT:    movgr2fr.w $fa1, $a0
; ILP32D-NEXT:    ffint.s.w $fa1, $fa1
; ILP32D-NEXT:    fcvt.d.s $fa1, $fa1
; ILP32D-NEXT:    fadd.d $fa0, $fa0, $fa1
; ILP32D-NEXT:    ret
;
; LP64D-LABEL: g:
; LP64D:       # %bb.0:
; LP64D-NEXT:    vldi $vr1, -912
; LP64D-NEXT:    fadd.d $fa0, $fa0, $fa1
; LP64D-NEXT:    ret
;
; LP64S-LABEL: g:
; LP64S:         bl %plt(__adddf3)
  %1 = fadd double %a, 1.0
  ret double %1
}
