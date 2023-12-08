;; Check that an unknown --target-abi is ignored and the triple-implied ABI is
;; used.
; RUN: llc --mtriple=loongarch32-linux-gnu --target-abi=foo --mattr=+d < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=ILP32D,UNKNOWN
; RUN: llc --mtriple=loongarch64-linux-gnu --target-abi=foo --mattr=+d < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=LP64D,UNKNOWN

; UNKNOWN: 'foo' is not a recognized ABI for this target, ignoring and using triple-implied ABI

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

; 32ON64: 32-bit ABIs are not supported for 64-bit targets, ignoring target-abi and using triple-implied ABI
; 64ON32: 64-bit ABIs are not supported for 32-bit targets, ignoring target-abi and using triple-implied ABI

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
; LP64D-NEXT:    addi.w $a0, $zero, 1
; LP64D-NEXT:    movgr2fr.w $fa1, $a0
; LP64D-NEXT:    ffint.s.w $fa1, $fa1
; LP64D-NEXT:    fadd.s $fa0, $fa0, $fa1
; LP64D-NEXT:    ret
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
; LP64D-NEXT:    addi.d $a0, $zero, 1
; LP64D-NEXT:    movgr2fr.d $fa1, $a0
; LP64D-NEXT:    ffint.d.l $fa1, $fa1
; LP64D-NEXT:    fadd.d $fa0, $fa0, $fa1
; LP64D-NEXT:    ret
  %1 = fadd double %a, 1.0
  ret double %1
}
