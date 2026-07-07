; RUN: llc -mtriple=aarch64              < %s | FileCheck --check-prefixes=CHECK,COMPAT %s
; RUN: llc -mtriple=aarch64 -mattr=v8.3a < %s | FileCheck --check-prefixes=CHECK,V83A %s
; RUN: llc -mtriple=aarch64 -mattr=v9a -mattr=pauth-lr < %s | FileCheck --check-prefixes=CHECK,V9A %s
; RUN: sed 's/"branch-protection-pauth-lr" //g' %s \
; RUN:   | llc -mtriple=aarch64 -mattr=v9a \
; RUN:   | FileCheck --check-prefixes=CHECK,PAUTH %s
;
; Test tail calls with return address signing across all three modes
; (COMPAT, V83A, V9A) and both FPDiff==0 and FPDiff!=0 cases,
; with A-key and B-key variants, for both direct and indirect tail calls.

declare swifttailcc void @callee_stack_args(ptr swiftasync %ctx, i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare swifttailcc void @callee_no_stack_args(ptr swiftasync %ctx)

; FPDiff != 0, A-key:     callee has stack args that this function doesn't.
define swifttailcc void @tail_call_fpdiff_a_key(ptr swiftasync %ctx) "branch-protection-pauth-lr" "sign-return-address"="all" "frame-pointer"="all" uwtable(async) {
; CHECK-LABEL: tail_call_fpdiff_a_key:
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

; CHECK:         orr x29, x29, #0x1000000000000000
; CHECK-NEXT:    sub sp, sp, #48
; CHECK-NEXT:    .cfi_def_cfa_offset 48
; CHECK-NEXT:    stp x29, x30, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    str x22, [sp, #8]
; CHECK-NEXT:    add x29, sp, #16
; CHECK-NEXT:    .cfi_def_cfa w29, 32
; CHECK-NEXT:    .cfi_offset w30, -24
; CHECK-NEXT:    .cfi_offset w29, -32
; CHECK-NEXT:    mov w8, #9 // =0x9
; CHECK-NEXT:    mov w0, #1 // =0x1
; CHECK-NEXT:    mov w1, #2 // =0x2
; CHECK-NEXT:    str x8, [x29, #16]
; CHECK-NEXT:    mov w2, #3 // =0x3
; CHECK-NEXT:    mov w3, #4 // =0x4
; CHECK-NEXT:    mov w4, #5 // =0x5
; CHECK-NEXT:    mov w5, #6 // =0x6
; CHECK-NEXT:    mov w6, #7 // =0x7
; CHECK-NEXT:    mov w7, #8 // =0x8
; CHECK-NEXT:    .cfi_def_cfa wsp, 48
; CHECK-NEXT:    ldp x29, x30, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    and x29, x29, #0xefffffffffffffff
; CHECK-NEXT:    add sp, sp, #32
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    .cfi_restore w30
; CHECK-NEXT:    .cfi_restore w29
; CHECK-NEXT:    add x16, sp, #16
; CHECK-NEXT:    mov x17, x30

; COMPAT-NEXT:   adrp x15, .Ltmp0
; COMPAT-NEXT:   add x15, x15, :lo12:.Ltmp0
; COMPAT-NEXT:   hint #39
; COMPAT-NEXT:   .cfi_set_ra_state 0, 0
; COMPAT-NEXT:   hint #12

; V83A-NEXT:     adrp x15, .Ltmp0
; V83A-NEXT:     add x15, x15, :lo12:.Ltmp0
; V83A-NEXT:     hint #39
; V83A-NEXT:     .cfi_set_ra_state 0, 0
; V83A-NEXT:     autia1716

; V9A-NEXT:      adrp x15, .Ltmp0
; V9A-NEXT:      add x15, x15, :lo12:.Ltmp0
; V9A-NEXT:      .cfi_set_ra_state 0, 0
; V9A-NEXT:      autia171615

; PAUTH-NEXT:    autia1716
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK-NEXT:    mov x30, x17
; CHECK-NEXT:    b callee_stack_args
  musttail call swifttailcc void @callee_stack_args(ptr swiftasync %ctx, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9)
  ret void
}

; FPDiff != 0, B-key:     callee has stack args that this function doesn't.
define swifttailcc void @tail_call_fpdiff_b_key(ptr swiftasync %ctx) "branch-protection-pauth-lr" "sign-return-address"="all" "sign-return-address-key"="b_key" "frame-pointer"="all" uwtable(async) {
; CHECK-LABEL: tail_call_fpdiff_b_key:
; CHECK:         // %bb.0:
; CHECK-NEXT:    .cfi_b_key_frame

; COMPAT-NEXT:   hint #39
; COMPAT-NEXT: .Ltmp1:
; COMPAT-NEXT:   hint #27
; COMPAT-NEXT:   .cfi_set_ra_state 2, .Ltmp1

; V83A-NEXT:     hint #39
; V83A-NEXT:   .Ltmp1:
; V83A-NEXT:     pacibsp
; V83A-NEXT:     .cfi_set_ra_state 2, .Ltmp1

; V9A-NEXT:    .Ltmp1:
; V9A-NEXT:      pacibsppc
; V9A-NEXT:      .cfi_set_ra_state 2, .Ltmp1

; PAUTH-NEXT:    pacibsp
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK-NEXT:    orr x29, x29, #0x1000000000000000
; CHECK-NEXT:    sub sp, sp, #48
; CHECK-NEXT:    .cfi_def_cfa_offset 48
; CHECK-NEXT:    stp x29, x30, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    str x22, [sp, #8]
; CHECK-NEXT:    add x29, sp, #16
; CHECK-NEXT:    .cfi_def_cfa w29, 32
; CHECK-NEXT:    .cfi_offset w30, -24
; CHECK-NEXT:    .cfi_offset w29, -32
; CHECK-NEXT:    mov w8, #9 // =0x9
; CHECK-NEXT:    mov w0, #1 // =0x1
; CHECK-NEXT:    mov w1, #2 // =0x2
; CHECK-NEXT:    str x8, [x29, #16]
; CHECK-NEXT:    mov w2, #3 // =0x3
; CHECK-NEXT:    mov w3, #4 // =0x4
; CHECK-NEXT:    mov w4, #5 // =0x5
; CHECK-NEXT:    mov w5, #6 // =0x6
; CHECK-NEXT:    mov w6, #7 // =0x7
; CHECK-NEXT:    mov w7, #8 // =0x8
; CHECK-NEXT:    .cfi_def_cfa wsp, 48
; CHECK-NEXT:    ldp x29, x30, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    and x29, x29, #0xefffffffffffffff
; CHECK-NEXT:    add sp, sp, #32
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    .cfi_restore w30
; CHECK-NEXT:    .cfi_restore w29
; CHECK-NEXT:    add x16, sp, #16
; CHECK-NEXT:    mov x17, x30

; COMPAT-NEXT:   adrp x15, .Ltmp1
; COMPAT-NEXT:   add x15, x15, :lo12:.Ltmp1
; COMPAT-NEXT:   hint #39
; COMPAT-NEXT:   .cfi_set_ra_state 0, 0
; COMPAT-NEXT:   hint #14

; V83A-NEXT:     adrp x15, .Ltmp1
; V83A-NEXT:     add x15, x15, :lo12:.Ltmp1
; V83A-NEXT:     hint #39
; V83A-NEXT:     .cfi_set_ra_state 0, 0
; V83A-NEXT:     autib1716

; V9A-NEXT:      adrp x15, .Ltmp1
; V9A-NEXT:      add x15, x15, :lo12:.Ltmp1
; V9A-NEXT:      .cfi_set_ra_state 0, 0
; V9A-NEXT:      autib171615

; PAUTH-NEXT:    autib1716
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK-NEXT:    mov x30, x17
; CHECK-NEXT:    b callee_stack_args
  musttail call swifttailcc void @callee_stack_args(ptr swiftasync %ctx, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9)
  ret void
}

; FPDiff == 0, A-key:     callee has same calling convention, no extra stack args.
define swifttailcc void @tail_call_no_fpdiff_a_key(ptr swiftasync %ctx) "branch-protection-pauth-lr" "sign-return-address"="all" "frame-pointer"="all" uwtable(async) {
; CHECK-LABEL: tail_call_no_fpdiff_a_key:
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

; CHECK:         orr x29, x29, #0x1000000000000000
; CHECK-NEXT:    sub sp, sp, #32
; CHECK-NEXT:    .cfi_def_cfa_offset 32
; CHECK-NEXT:    stp x29, x30, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    str x22, [sp, #8]
; CHECK-NEXT:    add x29, sp, #16
; CHECK-NEXT:    .cfi_def_cfa w29, 16
; CHECK-NEXT:    .cfi_offset w30, -8
; CHECK-NEXT:    .cfi_offset w29, -16
; CHECK-NEXT:    .cfi_def_cfa wsp, 32
; CHECK-NEXT:    ldp x29, x30, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    and x29, x29, #0xefffffffffffffff
; CHECK-NEXT:    add sp, sp, #32
; CHECK-NEXT:    .cfi_def_cfa_offset 0
; CHECK-NEXT:    .cfi_restore w30
; CHECK-NEXT:    .cfi_restore w29

; COMPAT-NEXT:   adrp x16, .Ltmp2
; COMPAT-NEXT:   add x16, x16, :lo12:.Ltmp2
; COMPAT-NEXT:   hint #39
; COMPAT-NEXT:   .cfi_set_ra_state 0, 0
; COMPAT-NEXT:   hint #29

; V83A-NEXT:     adrp x16, .Ltmp2
; V83A-NEXT:     add x16, x16, :lo12:.Ltmp2
; V83A-NEXT:     hint #39
; V83A-NEXT:     .cfi_set_ra_state 0, 0
; V83A-NEXT:     autiasp

; V9A-NEXT:      .cfi_set_ra_state 0, 0
; V9A-NEXT:      autiasppc .Ltmp2

; PAUTH-NEXT:    autiasp
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK-NEXT:    b callee_no_stack_args
  musttail call swifttailcc void @callee_no_stack_args(ptr swiftasync %ctx)
  ret void
}

; FPDiff == 0, B-key:     callee has same calling convention, no extra stack args.
define swifttailcc void @tail_call_no_fpdiff_b_key(ptr swiftasync %ctx) "branch-protection-pauth-lr" "sign-return-address"="all" "sign-return-address-key"="b_key" "frame-pointer"="all" uwtable(async) {
; CHECK-LABEL: tail_call_no_fpdiff_b_key:
; CHECK:       // %bb.0:
; CHECK-NEXT:    .cfi_b_key_frame

; COMPAT-NEXT:   hint #39
; COMPAT-NEXT: .Ltmp3:
; COMPAT-NEXT:   hint #27
; COMPAT-NEXT:   .cfi_set_ra_state 2, .Ltmp3

; V83A-NEXT:     hint #39
; V83A-NEXT:   .Ltmp3:
; V83A-NEXT:     pacibsp
; V83A-NEXT:     .cfi_set_ra_state 2, .Ltmp3

; V9A-NEXT:    .Ltmp3:
; V9A-NEXT:      pacibsppc
; V9A-NEXT:      .cfi_set_ra_state 2, .Ltmp3

; PAUTH-NEXT:    pacibsp
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK-NEXT:    orr x29, x29, #0x1000000000000000
; CHECK-NEXT:    sub sp, sp, #32
; CHECK-NEXT:    .cfi_def_cfa_offset 32
; CHECK-NEXT:    stp x29, x30, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    str x22, [sp, #8]
; CHECK-NEXT:    add x29, sp, #16
; CHECK-NEXT:    .cfi_def_cfa w29, 16
; CHECK-NEXT:    .cfi_offset w30, -8
; CHECK-NEXT:    .cfi_offset w29, -16
; CHECK-NEXT:    .cfi_def_cfa wsp, 32
; CHECK-NEXT:    ldp x29, x30, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    and x29, x29, #0xefffffffffffffff
; CHECK-NEXT:    add sp, sp, #32
; CHECK-NEXT:    .cfi_def_cfa_offset 0
; CHECK-NEXT:    .cfi_restore w30
; CHECK-NEXT:    .cfi_restore w29

; COMPAT-NEXT:   adrp x16, .Ltmp3
; COMPAT-NEXT:   add x16, x16, :lo12:.Ltmp3
; COMPAT-NEXT:   hint #39
; COMPAT-NEXT:   .cfi_set_ra_state 0, 0
; COMPAT-NEXT:   hint #31

; V83A-NEXT:     adrp x16, .Ltmp3
; V83A-NEXT:     add x16, x16, :lo12:.Ltmp3
; V83A-NEXT:     hint #39
; V83A-NEXT:     .cfi_set_ra_state 0, 0
; V83A-NEXT:     autibsp

; V9A-NEXT:      .cfi_set_ra_state 0, 0
; V9A-NEXT:      autibsppc .Ltmp3

; PAUTH-NEXT:    autibsp
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK-NEXT:    b callee_no_stack_args
  musttail call swifttailcc void @callee_no_stack_args(ptr swiftasync %ctx)
  ret void
}

; FPDiff != 0, A-key, indirect call: callee ptr passed as argument.
define swifttailcc void @indirect_tail_call_fpdiff_a_key(ptr swiftasync %ctx, ptr %callee) "branch-protection-pauth-lr" "sign-return-address"="all" "frame-pointer"="all" uwtable(async) {
; CHECK-LABEL: indirect_tail_call_fpdiff_a_key:
; CHECK:       // %bb.0:

; COMPAT-NEXT:   hint #39
; COMPAT-NEXT: .Ltmp4:
; COMPAT-NEXT:   hint #25
; COMPAT-NEXT:   .cfi_set_ra_state 2, .Ltmp4

; V83A-NEXT:     hint #39
; V83A-NEXT:   .Ltmp4:
; V83A-NEXT:     paciasp
; V83A-NEXT:     .cfi_set_ra_state 2, .Ltmp4

; V9A-NEXT:    .Ltmp4:
; V9A-NEXT:      paciasppc
; V9A-NEXT:      .cfi_set_ra_state 2, .Ltmp4

; PAUTH-NEXT:    paciasp
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK:         orr x29, x29, #0x1000000000000000
; CHECK-NEXT:    sub sp, sp, #48
; CHECK-NEXT:    .cfi_def_cfa_offset 48
; CHECK-NEXT:    stp x29, x30, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    str x22, [sp, #8]
; CHECK-NEXT:    add x29, sp, #16
; CHECK-NEXT:    .cfi_def_cfa w29, 32
; CHECK-NEXT:    .cfi_offset w30, -24
; CHECK-NEXT:    .cfi_offset w29, -32
; CHECK-NEXT:    mov w9, #9 // =0x9
; CHECK-NEXT:    mov x8, x0
; CHECK-NEXT:    mov w0, #1 // =0x1
; CHECK-NEXT:    str x9, [x29, #16]
; CHECK-NEXT:    mov w1, #2 // =0x2
; CHECK-NEXT:    mov w2, #3 // =0x3
; CHECK-NEXT:    mov w3, #4 // =0x4
; CHECK-NEXT:    mov w4, #5 // =0x5
; CHECK-NEXT:    mov w5, #6 // =0x6
; CHECK-NEXT:    mov w6, #7 // =0x7
; CHECK-NEXT:    mov w7, #8 // =0x8
; CHECK-NEXT:    .cfi_def_cfa wsp, 48
; CHECK-NEXT:    ldp x29, x30, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    and x29, x29, #0xefffffffffffffff
; CHECK-NEXT:    add sp, sp, #32
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    .cfi_restore w30
; CHECK-NEXT:    .cfi_restore w29
; CHECK-NEXT:    add x16, sp, #16
; CHECK-NEXT:    mov x17, x30

; COMPAT-NEXT:   adrp x15, .Ltmp4
; COMPAT-NEXT:   add x15, x15, :lo12:.Ltmp4
; COMPAT-NEXT:   hint #39
; COMPAT-NEXT:   .cfi_set_ra_state 0, 0
; COMPAT-NEXT:   hint #12

; V83A-NEXT:     adrp x15, .Ltmp4
; V83A-NEXT:     add x15, x15, :lo12:.Ltmp4
; V83A-NEXT:     hint #39
; V83A-NEXT:     .cfi_set_ra_state 0, 0
; V83A-NEXT:     autia1716

; V9A-NEXT:      adrp x15, .Ltmp4
; V9A-NEXT:      add x15, x15, :lo12:.Ltmp4
; V9A-NEXT:      .cfi_set_ra_state 0, 0
; V9A-NEXT:      autia171615

; PAUTH-NEXT:    autia1716
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK-NEXT:    mov x30, x17
; CHECK-NEXT:    br x8
  musttail call swifttailcc void %callee(ptr swiftasync %ctx, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9)
  ret void
}

; FPDiff != 0, B-key, indirect call: callee ptr passed as argument.
define swifttailcc void @indirect_tail_call_fpdiff_b_key(ptr swiftasync %ctx, ptr %callee) "branch-protection-pauth-lr" "sign-return-address"="all" "sign-return-address-key"="b_key" "frame-pointer"="all" uwtable(async) {
; CHECK-LABEL: indirect_tail_call_fpdiff_b_key:
; CHECK:         // %bb.0:
; CHECK-NEXT:    .cfi_b_key_frame

; COMPAT-NEXT:   hint #39
; COMPAT-NEXT: .Ltmp5:
; COMPAT-NEXT:   hint #27
; COMPAT-NEXT:   .cfi_set_ra_state 2, .Ltmp5

; V83A-NEXT:     hint #39
; V83A-NEXT:   .Ltmp5:
; V83A-NEXT:     pacibsp
; V83A-NEXT:     .cfi_set_ra_state 2, .Ltmp5

; V9A-NEXT:    .Ltmp5:
; V9A-NEXT:      pacibsppc
; V9A-NEXT:      .cfi_set_ra_state 2, .Ltmp5

; PAUTH-NEXT:    pacibsp
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK:         orr x29, x29, #0x1000000000000000
; CHECK-NEXT:    sub sp, sp, #48
; CHECK-NEXT:    .cfi_def_cfa_offset 48
; CHECK-NEXT:    stp x29, x30, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    str x22, [sp, #8]
; CHECK-NEXT:    add x29, sp, #16
; CHECK-NEXT:    .cfi_def_cfa w29, 32
; CHECK-NEXT:    .cfi_offset w30, -24
; CHECK-NEXT:    .cfi_offset w29, -32
; CHECK-NEXT:    mov w9, #9 // =0x9
; CHECK-NEXT:    mov x8, x0
; CHECK-NEXT:    mov w0, #1 // =0x1
; CHECK-NEXT:    str x9, [x29, #16]
; CHECK-NEXT:    mov w1, #2 // =0x2
; CHECK-NEXT:    mov w2, #3 // =0x3
; CHECK-NEXT:    mov w3, #4 // =0x4
; CHECK-NEXT:    mov w4, #5 // =0x5
; CHECK-NEXT:    mov w5, #6 // =0x6
; CHECK-NEXT:    mov w6, #7 // =0x7
; CHECK-NEXT:    mov w7, #8 // =0x8
; CHECK-NEXT:    .cfi_def_cfa wsp, 48
; CHECK-NEXT:    ldp x29, x30, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    and x29, x29, #0xefffffffffffffff
; CHECK-NEXT:    add sp, sp, #32
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    .cfi_restore w30
; CHECK-NEXT:    .cfi_restore w29
; CHECK-NEXT:    add x16, sp, #16
; CHECK-NEXT:    mov x17, x30

; COMPAT-NEXT:   adrp x15, .Ltmp5
; COMPAT-NEXT:   add x15, x15, :lo12:.Ltmp5
; COMPAT-NEXT:   hint #39
; COMPAT-NEXT:   .cfi_set_ra_state 0, 0
; COMPAT-NEXT:   hint #14

; V83A-NEXT:     adrp x15, .Ltmp5
; V83A-NEXT:     add x15, x15, :lo12:.Ltmp5
; V83A-NEXT:     hint #39
; V83A-NEXT:     .cfi_set_ra_state 0, 0
; V83A-NEXT:     autib1716

; V9A-NEXT:      adrp x15, .Ltmp5
; V9A-NEXT:      add x15, x15, :lo12:.Ltmp5
; V9A-NEXT:      .cfi_set_ra_state 0, 0
; V9A-NEXT:      autib171615

; PAUTH-NEXT:    autib1716
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK-NEXT:    mov x30, x17
; CHECK-NEXT:    br x8
  musttail call swifttailcc void %callee(ptr swiftasync %ctx, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9)
  ret void
}

; FPDiff == 0, A-key, indirect call: callee ptr passed as argument, no extra stack args.
define swifttailcc void @indirect_tail_call_no_fpdiff_a_key(ptr swiftasync %ctx, ptr %callee) "branch-protection-pauth-lr" "sign-return-address"="all" "frame-pointer"="all" uwtable(async) {
; CHECK-LABEL: indirect_tail_call_no_fpdiff_a_key:
; CHECK:       // %bb.0:

; COMPAT-NEXT:   hint #39
; COMPAT-NEXT: .Ltmp6:
; COMPAT-NEXT:   hint #25
; COMPAT-NEXT:   .cfi_set_ra_state 2, .Ltmp6

; V83A-NEXT:     hint #39
; V83A-NEXT:   .Ltmp6:
; V83A-NEXT:     paciasp
; V83A-NEXT:     .cfi_set_ra_state 2, .Ltmp6

; V9A-NEXT:    .Ltmp6:
; V9A-NEXT:      paciasppc
; V9A-NEXT:      .cfi_set_ra_state 2, .Ltmp6

; PAUTH-NEXT:    paciasp
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK:         orr x29, x29, #0x1000000000000000
; CHECK-NEXT:    sub sp, sp, #32
; CHECK-NEXT:    .cfi_def_cfa_offset 32
; CHECK-NEXT:    stp x29, x30, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    str x22, [sp, #8]
; CHECK-NEXT:    add x29, sp, #16
; CHECK-NEXT:    .cfi_def_cfa w29, 16
; CHECK-NEXT:    .cfi_offset w30, -8
; CHECK-NEXT:    .cfi_offset w29, -16
; CHECK-NEXT:    .cfi_def_cfa wsp, 32
; CHECK-NEXT:    ldp x29, x30, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    and x29, x29, #0xefffffffffffffff
; CHECK-NEXT:    add sp, sp, #32
; CHECK-NEXT:    .cfi_def_cfa_offset 0
; CHECK-NEXT:    .cfi_restore w30
; CHECK-NEXT:    .cfi_restore w29

; COMPAT-NEXT:   adrp x16, .Ltmp6
; COMPAT-NEXT:   add x16, x16, :lo12:.Ltmp6
; COMPAT-NEXT:   hint #39
; COMPAT-NEXT:   .cfi_set_ra_state 0, 0
; COMPAT-NEXT:   hint #29

; V83A-NEXT:     adrp x16, .Ltmp6
; V83A-NEXT:     add x16, x16, :lo12:.Ltmp6
; V83A-NEXT:     hint #39
; V83A-NEXT:     .cfi_set_ra_state 0, 0
; V83A-NEXT:     autiasp

; V9A-NEXT:      .cfi_set_ra_state 0, 0
; V9A-NEXT:      autiasppc .Ltmp6

; PAUTH-NEXT:    autiasp
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK-NEXT:    br x0
  musttail call swifttailcc void %callee(ptr swiftasync %ctx)
  ret void
}

; FPDiff == 0, B-key, indirect call: callee ptr passed as argument, no extra stack args.
define swifttailcc void @indirect_tail_call_no_fpdiff_b_key(ptr swiftasync %ctx, ptr %callee) "branch-protection-pauth-lr" "sign-return-address"="all" "sign-return-address-key"="b_key" "frame-pointer"="all" uwtable(async) {
; CHECK-LABEL: indirect_tail_call_no_fpdiff_b_key:
; CHECK:       // %bb.0:
; CHECK-NEXT:    .cfi_b_key_frame

; COMPAT-NEXT:   hint #39
; COMPAT-NEXT: .Ltmp7:
; COMPAT-NEXT:   hint #27
; COMPAT-NEXT:   .cfi_set_ra_state 2, .Ltmp7

; V83A-NEXT:     hint #39
; V83A-NEXT:   .Ltmp7:
; V83A-NEXT:     pacibsp
; V83A-NEXT:     .cfi_set_ra_state 2, .Ltmp7

; V9A-NEXT:    .Ltmp7:
; V9A-NEXT:      pacibsppc
; V9A-NEXT:      .cfi_set_ra_state 2, .Ltmp7

; PAUTH-NEXT:    pacibsp
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK-NEXT:    orr x29, x29, #0x1000000000000000
; CHECK-NEXT:    sub sp, sp, #32
; CHECK-NEXT:    .cfi_def_cfa_offset 32
; CHECK-NEXT:    stp x29, x30, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    str x22, [sp, #8]
; CHECK-NEXT:    add x29, sp, #16
; CHECK-NEXT:    .cfi_def_cfa w29, 16
; CHECK-NEXT:    .cfi_offset w30, -8
; CHECK-NEXT:    .cfi_offset w29, -16
; CHECK-NEXT:    .cfi_def_cfa wsp, 32
; CHECK-NEXT:    ldp x29, x30, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    and x29, x29, #0xefffffffffffffff
; CHECK-NEXT:    add sp, sp, #32
; CHECK-NEXT:    .cfi_def_cfa_offset 0
; CHECK-NEXT:    .cfi_restore w30
; CHECK-NEXT:    .cfi_restore w29

; COMPAT-NEXT:   adrp x16, .Ltmp7
; COMPAT-NEXT:   add x16, x16, :lo12:.Ltmp7
; COMPAT-NEXT:   hint #39
; COMPAT-NEXT:   .cfi_set_ra_state 0, 0
; COMPAT-NEXT:   hint #31

; V83A-NEXT:     adrp x16, .Ltmp7
; V83A-NEXT:     add x16, x16, :lo12:.Ltmp7
; V83A-NEXT:     hint #39
; V83A-NEXT:     .cfi_set_ra_state 0, 0
; V83A-NEXT:     autibsp

; V9A-NEXT:      .cfi_set_ra_state 0, 0
; V9A-NEXT:      autibsppc .Ltmp7

; PAUTH-NEXT:    autibsp
; PAUTH-NEXT:    .cfi_negate_ra_state

; CHECK-NEXT:    br x0
  musttail call swifttailcc void %callee(ptr swiftasync %ctx)
  ret void
}
