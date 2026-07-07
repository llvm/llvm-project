; RUN: llc -mtriple=aarch64              < %s | FileCheck --check-prefixes=CHECK,COMPAT %s
; RUN: llc -mtriple=aarch64 -mattr=v8.3a < %s | FileCheck --check-prefixes=CHECK,V83A %s
; RUN: llc -mtriple=aarch64 -mattr=v9a -mattr=pauth-lr < %s | FileCheck --check-prefixes=CHECK,V9A %s
; RUN: sed 's/"branch-protection-pauth-lr" //g' %s \
; RUN:   | llc -mtriple=aarch64 -mattr=v9a \
; RUN:   | FileCheck --check-prefixes=CHECK,PAUTH %s

; These tests cover the interaction between swifttailcc and return-address
; signing. The key invariant is that AUTIASP/AUTIBSP uses the current SP as the
; discriminator, which must equal the entry SP at the point of signing. When a
; tail call involves a stack-argument size mismatch (FPDiff != 0) this invariant
; can be violated if we're not careful.

declare swifttailcc void @callee_stack0()
declare swifttailcc void @callee_stack8([8 x i64], i64)

; FPDiff == 0: SP is unchanged at epilogue. autiasp is safe.
define swifttailcc void @caller_to0_from0() "branch-protection-pauth-lr" "sign-return-address"="all" "frame-pointer"="all" nounwind uwtable(async) {
; CHECK-LABEL: caller_to0_from0:
; CHECK:       // %bb.0:

; COMPAT-NEXT:   hint #39
; COMPAT-NEXT: .Ltmp0:
; COMPAT-NEXT:   hint #25
; COMPAT-NEXT:   .cfi_set_ra_state 2, .Ltmp0

; V83A-NEXT:     hint #39
; V83A-NEXT:   .Ltmp0:
; V83A-NEXT:     paciasp
; V83A-NEXT:     .cfi_set_ra_state 2, .Ltmp0

; V9A-NEXT:    .Ltmp0:
; V9A-NEXT:      paciasppc
; V9A-NEXT:      .cfi_set_ra_state 2, .Ltmp0

; PAUTH-NEXT:    paciasp
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK-NEXT:    stp x29, x30, [sp, #-16]! // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    mov x29, sp
; CHECK-NEXT:    .cfi_def_cfa w29, 16
; CHECK-NEXT:    .cfi_offset w30, -8
; CHECK-NEXT:    .cfi_offset w29, -16
; CHECK-NEXT:    .cfi_def_cfa wsp, 16
; CHECK-NEXT:    ldp x29, x30, [sp], #16 // 16-byte Folded Reload
; CHECK-NEXT:    .cfi_def_cfa_offset 0
; CHECK-NEXT:    .cfi_restore w30
; CHECK-NEXT:    .cfi_restore w29

; COMPAT-NEXT:   adrp x16, .Ltmp0
; COMPAT-NEXT:   add x16, x16, :lo12:.Ltmp0
; COMPAT-NEXT:   hint #39
; COMPAT-NEXT:   .cfi_set_ra_state 0, 0
; COMPAT-NEXT:   hint #29

; V83A-NEXT:     adrp x16, .Ltmp0
; V83A-NEXT:     add x16, x16, :lo12:.Ltmp0
; V83A-NEXT:     hint #39
; V83A-NEXT:     .cfi_set_ra_state 0, 0
; V83A-NEXT:     autiasp

; V9A-NEXT:      .cfi_set_ra_state 0, 0
; V9A-NEXT:      autiasppc .Ltmp0

; PAUTH-NEXT:    autiasp
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK-NEXT:    b callee_stack0
  tail call swifttailcc void @callee_stack0()
  ret void
}

; FPDiff > 0: caller received 8 bytes of stack args, callee receives 0.
; SP has been bumped by the post-index LDP. We must authenticate with the
; entry SP discriminator *before* popping the leftover argument space.
; The correct sequence is: autiasp (SP == entry SP here), then add sp, #16.
;
; Key point: we must NOT bump SP before autiasp, because that leaves the
; live argument space below SP and potentially outside the red-zone.
define swifttailcc void @caller_to0_from8([8 x i64], i64) "branch-protection-pauth-lr" "sign-return-address"="all" "frame-pointer"="all" uwtable(async) {
; CHECK-LABEL: caller_to0_from8:
; CHECK:       // %bb.0:

; COMPAT-NEXT:   hint #39
; COMPAT-NEXT: .Ltmp1:
; COMPAT-NEXT:   hint #25
; COMPAT-NEXT:   .cfi_set_ra_state 2, .Ltmp1

; V83A-NEXT:     hint #39
; V83A-NEXT:   .Ltmp1:
; V83A-NEXT:     paciasp
; V83A-NEXT:     .cfi_set_ra_state 2, .Ltmp1

; V9A-NEXT:    .Ltmp1:
; V9A-NEXT:      paciasppc
; V9A-NEXT:      .cfi_set_ra_state 2, .Ltmp1

; PAUTH-NEXT:    paciasp
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK-NEXT:    stp x29, x30, [sp, #-16]! // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    mov x29, sp
; CHECK-NEXT:    .cfi_def_cfa w29, 16
; CHECK-NEXT:    .cfi_offset w30, -8
; CHECK-NEXT:    .cfi_offset w29, -16
; CHECK-NEXT:    .cfi_def_cfa wsp, 16
; CHECK-NEXT:    ldp x29, x30, [sp], #16 // 16-byte Folded Reload
; CHECK-NEXT:    .cfi_def_cfa_offset 0
; CHECK-NEXT:    .cfi_def_cfa_offset -16
; CHECK-NEXT:    .cfi_restore w30
; CHECK-NEXT:    .cfi_restore w29

; COMPAT-NEXT:   adrp x16, .Ltmp1
; COMPAT-NEXT:   add x16, x16, :lo12:.Ltmp1
; COMPAT-NEXT:   hint #39
; COMPAT-NEXT:   .cfi_set_ra_state 0, 0
; COMPAT-NEXT:   hint #29

; V83A-NEXT:     adrp x16, .Ltmp1
; V83A-NEXT:     add x16, x16, :lo12:.Ltmp1
; V83A-NEXT:     hint #39
; V83A-NEXT:     .cfi_set_ra_state 0, 0
; V83A-NEXT:     autiasp

; V9A-NEXT:      .cfi_set_ra_state 0, 0
; V9A-NEXT:      autiasppc .Ltmp1

; PAUTH-NEXT:    autiasp
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK-NEXT:    add sp, sp, #16
; CHECK-NEXT:    b callee_stack0
  tail call swifttailcc void @callee_stack0()
  ret void
}

; FPDiff < 0: callee receives 8 bytes of stack args, caller received 0.
; Entry SP is above current SP; reconstruct it in x16 and use autia1716.
define swifttailcc void @caller_to8_from0() "branch-protection-pauth-lr" "sign-return-address"="all" "frame-pointer"="all" uwtable(async) {
; CHECK-LABEL: caller_to8_from0:
; CHECK:       // %bb.0:

; COMPAT-NEXT:   hint #39
; COMPAT-NEXT: .Ltmp2:
; COMPAT-NEXT:   hint #25
; COMPAT-NEXT:   .cfi_set_ra_state 2, .Ltmp2

; V83A-NEXT:     hint #39
; V83A-NEXT:   .Ltmp2:
; V83A-NEXT:     paciasp
; V83A-NEXT:     .cfi_set_ra_state 2, .Ltmp2

; V9A-NEXT:    .Ltmp2:
; V9A-NEXT:      paciasppc
; V9A-NEXT:      .cfi_set_ra_state 2, .Ltmp2

; PAUTH-NEXT:    paciasp
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK-NEXT:    stp x29, x30, [sp, #-32]! // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 32
; CHECK-NEXT:    mov x29, sp
; CHECK-NEXT:    .cfi_def_cfa w29, 32
; CHECK-NEXT:    .cfi_offset w30, -24
; CHECK-NEXT:    .cfi_offset w29, -32
; CHECK-NEXT:    mov w8, #42 // =0x2a
; CHECK-NEXT:    str x8, [x29, #16]
; CHECK-NEXT:    .cfi_def_cfa wsp, 32
; CHECK-NEXT:    ldp x29, x30, [sp], #16 // 16-byte Folded Reload
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    .cfi_restore w30
; CHECK-NEXT:    .cfi_restore w29
; CHECK-NEXT:    add x16, sp, #16
; CHECK-NEXT:    mov x17, x30

; COMPAT-NEXT:   adrp x15, .Ltmp2
; COMPAT-NEXT:   add x15, x15, :lo12:.Ltmp2
; COMPAT-NEXT:   hint #39
; COMPAT-NEXT:   .cfi_set_ra_state 0, 0
; COMPAT-NEXT:   hint #12

; V83A-NEXT:     adrp x15, .Ltmp2
; V83A-NEXT:     add x15, x15, :lo12:.Ltmp2
; V83A-NEXT:     hint #39
; V83A-NEXT:     .cfi_set_ra_state 0, 0
; V83A-NEXT:     autia1716

; V9A-NEXT:      adrp x15, .Ltmp2
; V9A-NEXT:      add x15, x15, :lo12:.Ltmp2
; V9A-NEXT:      .cfi_set_ra_state 0, 0
; V9A-NEXT:      autia171615

; PAUTH-NEXT:    autia1716
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK-NEXT:    mov x30, x17
; CHECK-NEXT:    b callee_stack8
  tail call swifttailcc void @callee_stack8([8 x i64] poison, i64 42)
  ret void
}
