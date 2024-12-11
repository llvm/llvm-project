; RUN: llc -mtriple=aarch64-darwin -mattr=+sve -mattr=+sme -enable-aarch64-sme-peephole-opt=false -verify-machineinstrs < %s | FileCheck %s

declare void @normal_callee();

define void @locally_streaming_fn() #0 {
; CHECK-LABEL: locally_streaming_fn:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    stp d15, d14, [sp, #-96]! ; 16-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 96
; CHECK-NEXT:    rdsvl x9, #1
; CHECK-NEXT:    stp d13, d12, [sp, #16] ; 16-byte Folded Spill
; CHECK-NEXT:    lsr x9, x9, #3
; CHECK-NEXT:    stp d11, d10, [sp, #32] ; 16-byte Folded Spill
; CHECK-NEXT:    stp d9, d8, [sp, #48] ; 16-byte Folded Spill
; CHECK-NEXT:    stp x30, x9, [sp, #64] ; 16-byte Folded Spill
; CHECK-NEXT:    cntd x9
; CHECK-NEXT:    str x9, [sp, #80] ; 8-byte Folded Spill
; CHECK-NEXT:    .cfi_offset vg, -16
; CHECK-NEXT:    .cfi_offset w30, -32
; CHECK-NEXT:    .cfi_offset b8, -40
; CHECK-NEXT:    .cfi_offset b9, -48
; CHECK-NEXT:    .cfi_offset b10, -56
; CHECK-NEXT:    .cfi_offset b11, -64
; CHECK-NEXT:    .cfi_offset b12, -72
; CHECK-NEXT:    .cfi_offset b13, -80
; CHECK-NEXT:    .cfi_offset b14, -88
; CHECK-NEXT:    .cfi_offset b15, -96
; CHECK-NEXT:    smstart sm
; CHECK-NEXT:    .cfi_offset vg, -24
; CHECK-NEXT:    smstop sm
; CHECK-NEXT:    bl _normal_callee
; CHECK-NEXT:    smstart sm
; CHECK-NEXT:    .cfi_restore vg
; CHECK-NEXT:    smstop sm
; CHECK-NEXT:    ldp d9, d8, [sp, #48] ; 16-byte Folded Reload
; CHECK-NEXT:    ldr x30, [sp, #64] ; 8-byte Folded Reload
; CHECK-NEXT:    ldp d11, d10, [sp, #32] ; 16-byte Folded Reload
; CHECK-NEXT:    ldp d13, d12, [sp, #16] ; 16-byte Folded Reload
; CHECK-NEXT:    ldp d15, d14, [sp], #96 ; 16-byte Folded Reload
; CHECK-NEXT:    .cfi_def_cfa_offset 0
; CHECK-NEXT:    .cfi_restore w30
; CHECK-NEXT:    .cfi_restore b8
; CHECK-NEXT:    .cfi_restore b9
; CHECK-NEXT:    .cfi_restore b10
; CHECK-NEXT:    .cfi_restore b11
; CHECK-NEXT:    .cfi_restore b12
; CHECK-NEXT:    .cfi_restore b13
; CHECK-NEXT:    .cfi_restore b14
; CHECK-NEXT:    .cfi_restore b15
; CHECK-NEXT:    ret
  call void @normal_callee()
  ret void
}

attributes #0 = { "aarch64_pstate_sm_body" uwtable(async) }
