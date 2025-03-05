; RUN: llc -mtriple=aarch64-linux-gnu -aarch64-streaming-hazard-size=0 -mattr=+sve -mattr=+sme2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -aarch64-streaming-hazard-size=0 -mattr=+sve -mattr=+sme2 -frame-pointer=non-leaf -verify-machineinstrs < %s | FileCheck %s --check-prefix=FP-CHECK
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme2 -frame-pointer=non-leaf -verify-machineinstrs < %s | FileCheck %s --check-prefix=NO-SVE-CHECK
; RUN: llc -mtriple=aarch64-linux-gnu -aarch64-streaming-hazard-size=0 -mattr=+sve -mattr=+sme2 -verify-machineinstrs -enable-machine-outliner < %s | FileCheck %s --check-prefix=OUTLINER-CHECK

declare void @callee();
declare void @fixed_callee(<4 x i32>);
declare void @scalable_callee(<vscale x 2 x i64>);

declare void @streaming_callee() #0;
declare void @streaming_callee_with_arg(i32) #0;

; Simple example of a function with one call requiring a streaming mode change
;
define void @vg_unwind_simple() #0 {
; CHECK-LABEL: vg_unwind_simple:
; CHECK:       // %bb.0:
; CHECK-NEXT:    stp d15, d14, [sp, #-80]! // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 80
; CHECK-NEXT:    cntd x9
; CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; CHECK-NEXT:    stp x30, x9, [sp, #64] // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_offset w30, -16
; CHECK-NEXT:    .cfi_offset b8, -24
; CHECK-NEXT:    .cfi_offset b9, -32
; CHECK-NEXT:    .cfi_offset b10, -40
; CHECK-NEXT:    .cfi_offset b11, -48
; CHECK-NEXT:    .cfi_offset b12, -56
; CHECK-NEXT:    .cfi_offset b13, -64
; CHECK-NEXT:    .cfi_offset b14, -72
; CHECK-NEXT:    .cfi_offset b15, -80
; CHECK-NEXT:    .cfi_offset vg, -8
; CHECK-NEXT:    smstop sm
; CHECK-NEXT:    bl callee
; CHECK-NEXT:    smstart sm
; CHECK-NEXT:    .cfi_restore vg
; CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; CHECK-NEXT:    ldr x30, [sp, #64] // 8-byte Folded Reload
; CHECK-NEXT:    ldp d11, d10, [sp, #32] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d13, d12, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d15, d14, [sp], #80 // 16-byte Folded Reload
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
;
; FP-CHECK-LABEL: vg_unwind_simple:
; FP-CHECK:       // %bb.0:
; FP-CHECK-NEXT:    stp d15, d14, [sp, #-96]! // 16-byte Folded Spill
; FP-CHECK-NEXT:    .cfi_def_cfa_offset 96
; FP-CHECK-NEXT:    cntd x9
; FP-CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp x29, x30, [sp, #64] // 16-byte Folded Spill
; FP-CHECK-NEXT:    str x9, [sp, #80] // 8-byte Folded Spill
; FP-CHECK-NEXT:    add x29, sp, #64
; FP-CHECK-NEXT:    .cfi_def_cfa w29, 32
; FP-CHECK-NEXT:    .cfi_offset w30, -24
; FP-CHECK-NEXT:    .cfi_offset w29, -32
; FP-CHECK-NEXT:    .cfi_offset b8, -40
; FP-CHECK-NEXT:    .cfi_offset b9, -48
; FP-CHECK-NEXT:    .cfi_offset b10, -56
; FP-CHECK-NEXT:    .cfi_offset b11, -64
; FP-CHECK-NEXT:    .cfi_offset b12, -72
; FP-CHECK-NEXT:    .cfi_offset b13, -80
; FP-CHECK-NEXT:    .cfi_offset b14, -88
; FP-CHECK-NEXT:    .cfi_offset b15, -96
; FP-CHECK-NEXT:    .cfi_offset vg, -16
; FP-CHECK-NEXT:    smstop sm
; FP-CHECK-NEXT:    bl callee
; FP-CHECK-NEXT:    smstart sm
; FP-CHECK-NEXT:    .cfi_restore vg
; FP-CHECK-NEXT:    .cfi_def_cfa wsp, 96
; FP-CHECK-NEXT:    ldp x29, x30, [sp, #64] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d11, d10, [sp, #32] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d13, d12, [sp, #16] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d15, d14, [sp], #96 // 16-byte Folded Reload
; FP-CHECK-NEXT:    .cfi_def_cfa_offset 0
; FP-CHECK-NEXT:    .cfi_restore w30
; FP-CHECK-NEXT:    .cfi_restore w29
; FP-CHECK-NEXT:    .cfi_restore b8
; FP-CHECK-NEXT:    .cfi_restore b9
; FP-CHECK-NEXT:    .cfi_restore b10
; FP-CHECK-NEXT:    .cfi_restore b11
; FP-CHECK-NEXT:    .cfi_restore b12
; FP-CHECK-NEXT:    .cfi_restore b13
; FP-CHECK-NEXT:    .cfi_restore b14
; FP-CHECK-NEXT:    .cfi_restore b15
; FP-CHECK-NEXT:    ret
;
; OUTLINER-CHECK-LABEL: vg_unwind_simple:
; OUTLINER-CHECK-NOT: OUTLINED_FUNCTION_
;
  call void @callee();
  ret void;
}

; As above, with an extra register clobbered by the inline asm call which
; changes NeedsGapToAlignStack to false
;
define void @vg_unwind_needs_gap() #0 {
; CHECK-LABEL: vg_unwind_needs_gap:
; CHECK:       // %bb.0:
; CHECK-NEXT:    stp d15, d14, [sp, #-96]! // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 96
; CHECK-NEXT:    cntd x9
; CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; CHECK-NEXT:    stp x30, x9, [sp, #64] // 16-byte Folded Spill
; CHECK-NEXT:    str x20, [sp, #80] // 8-byte Folded Spill
; CHECK-NEXT:    .cfi_offset w20, -16
; CHECK-NEXT:    .cfi_offset w30, -32
; CHECK-NEXT:    .cfi_offset b8, -40
; CHECK-NEXT:    .cfi_offset b9, -48
; CHECK-NEXT:    .cfi_offset b10, -56
; CHECK-NEXT:    .cfi_offset b11, -64
; CHECK-NEXT:    .cfi_offset b12, -72
; CHECK-NEXT:    .cfi_offset b13, -80
; CHECK-NEXT:    .cfi_offset b14, -88
; CHECK-NEXT:    .cfi_offset b15, -96
; CHECK-NEXT:    //APP
; CHECK-NEXT:    //NO_APP
; CHECK-NEXT:    .cfi_offset vg, -24
; CHECK-NEXT:    smstop sm
; CHECK-NEXT:    bl callee
; CHECK-NEXT:    smstart sm
; CHECK-NEXT:    .cfi_restore vg
; CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; CHECK-NEXT:    ldr x20, [sp, #80] // 8-byte Folded Reload
; CHECK-NEXT:    ldp d11, d10, [sp, #32] // 16-byte Folded Reload
; CHECK-NEXT:    ldr x30, [sp, #64] // 8-byte Folded Reload
; CHECK-NEXT:    ldp d13, d12, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d15, d14, [sp], #96 // 16-byte Folded Reload
; CHECK-NEXT:    .cfi_def_cfa_offset 0
; CHECK-NEXT:    .cfi_restore w20
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
;
; FP-CHECK-LABEL: vg_unwind_needs_gap:
; FP-CHECK:       // %bb.0:
; FP-CHECK-NEXT:    stp d15, d14, [sp, #-96]! // 16-byte Folded Spill
; FP-CHECK-NEXT:    .cfi_def_cfa_offset 96
; FP-CHECK-NEXT:    cntd x9
; FP-CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp x29, x30, [sp, #64] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp x9, x20, [sp, #80] // 16-byte Folded Spill
; FP-CHECK-NEXT:    add x29, sp, #64
; FP-CHECK-NEXT:    .cfi_def_cfa w29, 32
; FP-CHECK-NEXT:    .cfi_offset w20, -8
; FP-CHECK-NEXT:    .cfi_offset w30, -24
; FP-CHECK-NEXT:    .cfi_offset w29, -32
; FP-CHECK-NEXT:    .cfi_offset b8, -40
; FP-CHECK-NEXT:    .cfi_offset b9, -48
; FP-CHECK-NEXT:    .cfi_offset b10, -56
; FP-CHECK-NEXT:    .cfi_offset b11, -64
; FP-CHECK-NEXT:    .cfi_offset b12, -72
; FP-CHECK-NEXT:    .cfi_offset b13, -80
; FP-CHECK-NEXT:    .cfi_offset b14, -88
; FP-CHECK-NEXT:    .cfi_offset b15, -96
; FP-CHECK-NEXT:    //APP
; FP-CHECK-NEXT:    //NO_APP
; FP-CHECK-NEXT:    .cfi_offset vg, -16
; FP-CHECK-NEXT:    smstop sm
; FP-CHECK-NEXT:    bl callee
; FP-CHECK-NEXT:    smstart sm
; FP-CHECK-NEXT:    .cfi_restore vg
; FP-CHECK-NEXT:    .cfi_def_cfa wsp, 96
; FP-CHECK-NEXT:    ldp x29, x30, [sp, #64] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldr x20, [sp, #88] // 8-byte Folded Reload
; FP-CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d11, d10, [sp, #32] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d13, d12, [sp, #16] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d15, d14, [sp], #96 // 16-byte Folded Reload
; FP-CHECK-NEXT:    .cfi_def_cfa_offset 0
; FP-CHECK-NEXT:    .cfi_restore w20
; FP-CHECK-NEXT:    .cfi_restore w30
; FP-CHECK-NEXT:    .cfi_restore w29
; FP-CHECK-NEXT:    .cfi_restore b8
; FP-CHECK-NEXT:    .cfi_restore b9
; FP-CHECK-NEXT:    .cfi_restore b10
; FP-CHECK-NEXT:    .cfi_restore b11
; FP-CHECK-NEXT:    .cfi_restore b12
; FP-CHECK-NEXT:    .cfi_restore b13
; FP-CHECK-NEXT:    .cfi_restore b14
; FP-CHECK-NEXT:    .cfi_restore b15
; FP-CHECK-NEXT:    ret
;
; OUTLINER-CHECK-LABEL: vg_unwind_needs_gap:
; OUTLINER-CHECK-NOT: OUTLINED_FUNCTION_
;
  call void asm sideeffect "", "~{x20}"()
  call void @callee();
  ret void;
}

define void @vg_unwind_with_fixed_args(<4 x i32> %x) #0 {
; CHECK-LABEL: vg_unwind_with_fixed_args:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sub sp, sp, #96
; CHECK-NEXT:    .cfi_def_cfa_offset 96
; CHECK-NEXT:    cntd x9
; CHECK-NEXT:    stp d15, d14, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    stp d13, d12, [sp, #32] // 16-byte Folded Spill
; CHECK-NEXT:    stp d11, d10, [sp, #48] // 16-byte Folded Spill
; CHECK-NEXT:    stp d9, d8, [sp, #64] // 16-byte Folded Spill
; CHECK-NEXT:    stp x30, x9, [sp, #80] // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_offset w30, -16
; CHECK-NEXT:    .cfi_offset b8, -24
; CHECK-NEXT:    .cfi_offset b9, -32
; CHECK-NEXT:    .cfi_offset b10, -40
; CHECK-NEXT:    .cfi_offset b11, -48
; CHECK-NEXT:    .cfi_offset b12, -56
; CHECK-NEXT:    .cfi_offset b13, -64
; CHECK-NEXT:    .cfi_offset b14, -72
; CHECK-NEXT:    .cfi_offset b15, -80
; CHECK-NEXT:    str q0, [sp] // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_offset vg, -8
; CHECK-NEXT:    smstop sm
; CHECK-NEXT:    ldr q0, [sp] // 16-byte Folded Reload
; CHECK-NEXT:    bl fixed_callee
; CHECK-NEXT:    smstart sm
; CHECK-NEXT:    .cfi_restore vg
; CHECK-NEXT:    ldp d9, d8, [sp, #64] // 16-byte Folded Reload
; CHECK-NEXT:    ldr x30, [sp, #80] // 8-byte Folded Reload
; CHECK-NEXT:    ldp d11, d10, [sp, #48] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d13, d12, [sp, #32] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d15, d14, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    add sp, sp, #96
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
;
; FP-CHECK-LABEL: vg_unwind_with_fixed_args:
; FP-CHECK:       // %bb.0:
; FP-CHECK-NEXT:    sub sp, sp, #112
; FP-CHECK-NEXT:    .cfi_def_cfa_offset 112
; FP-CHECK-NEXT:    cntd x9
; FP-CHECK-NEXT:    stp d15, d14, [sp, #16] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp d13, d12, [sp, #32] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp d11, d10, [sp, #48] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp d9, d8, [sp, #64] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp x29, x30, [sp, #80] // 16-byte Folded Spill
; FP-CHECK-NEXT:    str x9, [sp, #96] // 8-byte Folded Spill
; FP-CHECK-NEXT:    add x29, sp, #80
; FP-CHECK-NEXT:    .cfi_def_cfa w29, 32
; FP-CHECK-NEXT:    .cfi_offset w30, -24
; FP-CHECK-NEXT:    .cfi_offset w29, -32
; FP-CHECK-NEXT:    .cfi_offset b8, -40
; FP-CHECK-NEXT:    .cfi_offset b9, -48
; FP-CHECK-NEXT:    .cfi_offset b10, -56
; FP-CHECK-NEXT:    .cfi_offset b11, -64
; FP-CHECK-NEXT:    .cfi_offset b12, -72
; FP-CHECK-NEXT:    .cfi_offset b13, -80
; FP-CHECK-NEXT:    .cfi_offset b14, -88
; FP-CHECK-NEXT:    .cfi_offset b15, -96
; FP-CHECK-NEXT:    str q0, [sp] // 16-byte Folded Spill
; FP-CHECK-NEXT:    .cfi_offset vg, -16
; FP-CHECK-NEXT:    smstop sm
; FP-CHECK-NEXT:    ldr q0, [sp] // 16-byte Folded Reload
; FP-CHECK-NEXT:    bl fixed_callee
; FP-CHECK-NEXT:    smstart sm
; FP-CHECK-NEXT:    .cfi_restore vg
; FP-CHECK-NEXT:    .cfi_def_cfa wsp, 112
; FP-CHECK-NEXT:    ldp x29, x30, [sp, #80] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d9, d8, [sp, #64] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d11, d10, [sp, #48] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d13, d12, [sp, #32] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d15, d14, [sp, #16] // 16-byte Folded Reload
; FP-CHECK-NEXT:    add sp, sp, #112
; FP-CHECK-NEXT:    .cfi_def_cfa_offset 0
; FP-CHECK-NEXT:    .cfi_restore w30
; FP-CHECK-NEXT:    .cfi_restore w29
; FP-CHECK-NEXT:    .cfi_restore b8
; FP-CHECK-NEXT:    .cfi_restore b9
; FP-CHECK-NEXT:    .cfi_restore b10
; FP-CHECK-NEXT:    .cfi_restore b11
; FP-CHECK-NEXT:    .cfi_restore b12
; FP-CHECK-NEXT:    .cfi_restore b13
; FP-CHECK-NEXT:    .cfi_restore b14
; FP-CHECK-NEXT:    .cfi_restore b15
; FP-CHECK-NEXT:    ret
;
; OUTLINER-CHECK-LABEL: vg_unwind_with_fixed_args:
; OUTLINER-CHECK-NOT: OUTLINED_FUNCTION_
;
  call void @fixed_callee(<4 x i32> %x);
  ret void;
}

define void @vg_unwind_with_sve_args(<vscale x 2 x i64> %x) #0 {
; CHECK-LABEL: vg_unwind_with_sve_args:
; CHECK:       // %bb.0:
; CHECK-NEXT:    stp x29, x30, [sp, #-32]! // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 32
; CHECK-NEXT:    cntd x9
; CHECK-NEXT:    stp x9, x28, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_offset w28, -8
; CHECK-NEXT:    .cfi_offset w30, -24
; CHECK-NEXT:    .cfi_offset w29, -32
; CHECK-NEXT:    addvl sp, sp, #-18
; CHECK-NEXT:    .cfi_escape 0x0f, 0x0d, 0x8f, 0x00, 0x11, 0x20, 0x22, 0x11, 0x90, 0x01, 0x92, 0x2e, 0x00, 0x1e, 0x22 // sp + 32 + 144 * VG
; CHECK-NEXT:    str p8, [sp, #11, mul vl] // 2-byte Folded Spill
; CHECK-NEXT:    ptrue pn8.b
; CHECK-NEXT:    str p15, [sp, #4, mul vl] // 2-byte Folded Spill
; CHECK-NEXT:    st1b { z22.b, z23.b }, pn8, [sp, #2, mul vl] // 32-byte Folded Spill
; CHECK-NEXT:    st1b { z20.b, z21.b }, pn8, [sp, #4, mul vl] // 32-byte Folded Spill
; CHECK-NEXT:    str p14, [sp, #5, mul vl] // 2-byte Folded Spill
; CHECK-NEXT:    st1b { z18.b, z19.b }, pn8, [sp, #6, mul vl] // 32-byte Folded Spill
; CHECK-NEXT:    st1b { z16.b, z17.b }, pn8, [sp, #8, mul vl] // 32-byte Folded Spill
; CHECK-NEXT:    str p13, [sp, #6, mul vl] // 2-byte Folded Spill
; CHECK-NEXT:    st1b { z14.b, z15.b }, pn8, [sp, #10, mul vl] // 32-byte Folded Spill
; CHECK-NEXT:    st1b { z12.b, z13.b }, pn8, [sp, #12, mul vl] // 32-byte Folded Spill
; CHECK-NEXT:    str p12, [sp, #7, mul vl] // 2-byte Folded Spill
; CHECK-NEXT:    st1b { z10.b, z11.b }, pn8, [sp, #14, mul vl] // 32-byte Folded Spill
; CHECK-NEXT:    str p11, [sp, #8, mul vl] // 2-byte Folded Spill
; CHECK-NEXT:    str p10, [sp, #9, mul vl] // 2-byte Folded Spill
; CHECK-NEXT:    str p9, [sp, #10, mul vl] // 2-byte Folded Spill
; CHECK-NEXT:    str p7, [sp, #12, mul vl] // 2-byte Folded Spill
; CHECK-NEXT:    str p6, [sp, #13, mul vl] // 2-byte Folded Spill
; CHECK-NEXT:    str p5, [sp, #14, mul vl] // 2-byte Folded Spill
; CHECK-NEXT:    str p4, [sp, #15, mul vl] // 2-byte Folded Spill
; CHECK-NEXT:    str z9, [sp, #16, mul vl] // 16-byte Folded Spill
; CHECK-NEXT:    str z8, [sp, #17, mul vl] // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_escape 0x10, 0x48, 0x0a, 0x11, 0x60, 0x22, 0x11, 0x78, 0x92, 0x2e, 0x00, 0x1e, 0x22 // $d8 @ cfa - 32 - 8 * VG
; CHECK-NEXT:    .cfi_escape 0x10, 0x49, 0x0a, 0x11, 0x60, 0x22, 0x11, 0x70, 0x92, 0x2e, 0x00, 0x1e, 0x22 // $d9 @ cfa - 32 - 16 * VG
; CHECK-NEXT:    .cfi_escape 0x10, 0x4a, 0x0a, 0x11, 0x60, 0x22, 0x11, 0x68, 0x92, 0x2e, 0x00, 0x1e, 0x22 // $d10 @ cfa - 32 - 24 * VG
; CHECK-NEXT:    .cfi_escape 0x10, 0x4b, 0x0a, 0x11, 0x60, 0x22, 0x11, 0x60, 0x92, 0x2e, 0x00, 0x1e, 0x22 // $d11 @ cfa - 32 - 32 * VG
; CHECK-NEXT:    .cfi_escape 0x10, 0x4c, 0x0a, 0x11, 0x60, 0x22, 0x11, 0x58, 0x92, 0x2e, 0x00, 0x1e, 0x22 // $d12 @ cfa - 32 - 40 * VG
; CHECK-NEXT:    .cfi_escape 0x10, 0x4d, 0x0a, 0x11, 0x60, 0x22, 0x11, 0x50, 0x92, 0x2e, 0x00, 0x1e, 0x22 // $d13 @ cfa - 32 - 48 * VG
; CHECK-NEXT:    .cfi_escape 0x10, 0x4e, 0x0a, 0x11, 0x60, 0x22, 0x11, 0x48, 0x92, 0x2e, 0x00, 0x1e, 0x22 // $d14 @ cfa - 32 - 56 * VG
; CHECK-NEXT:    .cfi_escape 0x10, 0x4f, 0x0a, 0x11, 0x60, 0x22, 0x11, 0x40, 0x92, 0x2e, 0x00, 0x1e, 0x22 // $d15 @ cfa - 32 - 64 * VG
; CHECK-NEXT:    addvl sp, sp, #-1
; CHECK-NEXT:    .cfi_escape 0x0f, 0x0d, 0x8f, 0x00, 0x11, 0x20, 0x22, 0x11, 0x98, 0x01, 0x92, 0x2e, 0x00, 0x1e, 0x22 // sp + 32 + 152 * VG
; CHECK-NEXT:    str z0, [sp] // 16-byte Folded Spill
; CHECK-NEXT:    //APP
; CHECK-NEXT:    //NO_APP
; CHECK-NEXT:    .cfi_offset vg, -16
; CHECK-NEXT:    smstop sm
; CHECK-NEXT:    ldr z0, [sp] // 16-byte Folded Reload
; CHECK-NEXT:    bl scalable_callee
; CHECK-NEXT:    smstart sm
; CHECK-NEXT:    .cfi_restore vg
; CHECK-NEXT:    addvl sp, sp, #1
; CHECK-NEXT:    .cfi_escape 0x0f, 0x0d, 0x8f, 0x00, 0x11, 0x20, 0x22, 0x11, 0x90, 0x01, 0x92, 0x2e, 0x00, 0x1e, 0x22 // sp + 32 + 144 * VG
; CHECK-NEXT:    ptrue pn8.b
; CHECK-NEXT:    ldr z9, [sp, #16, mul vl] // 16-byte Folded Reload
; CHECK-NEXT:    ldr z8, [sp, #17, mul vl] // 16-byte Folded Reload
; CHECK-NEXT:    ld1b { z22.b, z23.b }, pn8/z, [sp, #2, mul vl] // 32-byte Folded Reload
; CHECK-NEXT:    ld1b { z20.b, z21.b }, pn8/z, [sp, #4, mul vl] // 32-byte Folded Reload
; CHECK-NEXT:    ld1b { z18.b, z19.b }, pn8/z, [sp, #6, mul vl] // 32-byte Folded Reload
; CHECK-NEXT:    ld1b { z16.b, z17.b }, pn8/z, [sp, #8, mul vl] // 32-byte Folded Reload
; CHECK-NEXT:    ld1b { z14.b, z15.b }, pn8/z, [sp, #10, mul vl] // 32-byte Folded Reload
; CHECK-NEXT:    ld1b { z12.b, z13.b }, pn8/z, [sp, #12, mul vl] // 32-byte Folded Reload
; CHECK-NEXT:    ld1b { z10.b, z11.b }, pn8/z, [sp, #14, mul vl] // 32-byte Folded Reload
; CHECK-NEXT:    ldr p15, [sp, #4, mul vl] // 2-byte Folded Reload
; CHECK-NEXT:    ldr p14, [sp, #5, mul vl] // 2-byte Folded Reload
; CHECK-NEXT:    ldr p13, [sp, #6, mul vl] // 2-byte Folded Reload
; CHECK-NEXT:    ldr p12, [sp, #7, mul vl] // 2-byte Folded Reload
; CHECK-NEXT:    ldr p11, [sp, #8, mul vl] // 2-byte Folded Reload
; CHECK-NEXT:    ldr p10, [sp, #9, mul vl] // 2-byte Folded Reload
; CHECK-NEXT:    ldr p9, [sp, #10, mul vl] // 2-byte Folded Reload
; CHECK-NEXT:    ldr p8, [sp, #11, mul vl] // 2-byte Folded Reload
; CHECK-NEXT:    ldr p7, [sp, #12, mul vl] // 2-byte Folded Reload
; CHECK-NEXT:    ldr p6, [sp, #13, mul vl] // 2-byte Folded Reload
; CHECK-NEXT:    ldr p5, [sp, #14, mul vl] // 2-byte Folded Reload
; CHECK-NEXT:    ldr p4, [sp, #15, mul vl] // 2-byte Folded Reload
; CHECK-NEXT:    addvl sp, sp, #18
; CHECK-NEXT:    .cfi_def_cfa wsp, 32
; CHECK-NEXT:    .cfi_restore z8
; CHECK-NEXT:    .cfi_restore z9
; CHECK-NEXT:    .cfi_restore z10
; CHECK-NEXT:    .cfi_restore z11
; CHECK-NEXT:    .cfi_restore z12
; CHECK-NEXT:    .cfi_restore z13
; CHECK-NEXT:    .cfi_restore z14
; CHECK-NEXT:    .cfi_restore z15
; CHECK-NEXT:    ldr x28, [sp, #24] // 8-byte Folded Reload
; CHECK-NEXT:    ldp x29, x30, [sp], #32 // 16-byte Folded Reload
; CHECK-NEXT:    .cfi_def_cfa_offset 0
; CHECK-NEXT:    .cfi_restore w28
; CHECK-NEXT:    .cfi_restore w30
; CHECK-NEXT:    .cfi_restore w29
; CHECK-NEXT:    ret
;
; FP-CHECK-LABEL: vg_unwind_with_sve_args:
; FP-CHECK:       // %bb.0:
; FP-CHECK-NEXT:    stp x29, x30, [sp, #-48]! // 16-byte Folded Spill
; FP-CHECK-NEXT:    .cfi_def_cfa_offset 48
; FP-CHECK-NEXT:    cntd x9
; FP-CHECK-NEXT:    stp x28, x27, [sp, #32] // 16-byte Folded Spill
; FP-CHECK-NEXT:    str x9, [sp, #16] // 8-byte Folded Spill
; FP-CHECK-NEXT:    mov x29, sp
; FP-CHECK-NEXT:    .cfi_def_cfa w29, 48
; FP-CHECK-NEXT:    .cfi_offset w27, -8
; FP-CHECK-NEXT:    .cfi_offset w28, -16
; FP-CHECK-NEXT:    .cfi_offset w30, -40
; FP-CHECK-NEXT:    .cfi_offset w29, -48
; FP-CHECK-NEXT:    addvl sp, sp, #-18
; FP-CHECK-NEXT:    str p8, [sp, #11, mul vl] // 2-byte Folded Spill
; FP-CHECK-NEXT:    ptrue pn8.b
; FP-CHECK-NEXT:    str p15, [sp, #4, mul vl] // 2-byte Folded Spill
; FP-CHECK-NEXT:    st1b { z22.b, z23.b }, pn8, [sp, #2, mul vl] // 32-byte Folded Spill
; FP-CHECK-NEXT:    st1b { z20.b, z21.b }, pn8, [sp, #4, mul vl] // 32-byte Folded Spill
; FP-CHECK-NEXT:    str p14, [sp, #5, mul vl] // 2-byte Folded Spill
; FP-CHECK-NEXT:    st1b { z18.b, z19.b }, pn8, [sp, #6, mul vl] // 32-byte Folded Spill
; FP-CHECK-NEXT:    st1b { z16.b, z17.b }, pn8, [sp, #8, mul vl] // 32-byte Folded Spill
; FP-CHECK-NEXT:    str p13, [sp, #6, mul vl] // 2-byte Folded Spill
; FP-CHECK-NEXT:    st1b { z14.b, z15.b }, pn8, [sp, #10, mul vl] // 32-byte Folded Spill
; FP-CHECK-NEXT:    st1b { z12.b, z13.b }, pn8, [sp, #12, mul vl] // 32-byte Folded Spill
; FP-CHECK-NEXT:    str p12, [sp, #7, mul vl] // 2-byte Folded Spill
; FP-CHECK-NEXT:    st1b { z10.b, z11.b }, pn8, [sp, #14, mul vl] // 32-byte Folded Spill
; FP-CHECK-NEXT:    str p11, [sp, #8, mul vl] // 2-byte Folded Spill
; FP-CHECK-NEXT:    str p10, [sp, #9, mul vl] // 2-byte Folded Spill
; FP-CHECK-NEXT:    str p9, [sp, #10, mul vl] // 2-byte Folded Spill
; FP-CHECK-NEXT:    str p7, [sp, #12, mul vl] // 2-byte Folded Spill
; FP-CHECK-NEXT:    str p6, [sp, #13, mul vl] // 2-byte Folded Spill
; FP-CHECK-NEXT:    str p5, [sp, #14, mul vl] // 2-byte Folded Spill
; FP-CHECK-NEXT:    str p4, [sp, #15, mul vl] // 2-byte Folded Spill
; FP-CHECK-NEXT:    str z9, [sp, #16, mul vl] // 16-byte Folded Spill
; FP-CHECK-NEXT:    str z8, [sp, #17, mul vl] // 16-byte Folded Spill
; FP-CHECK-NEXT:    .cfi_escape 0x10, 0x48, 0x0a, 0x11, 0x50, 0x22, 0x11, 0x78, 0x92, 0x2e, 0x00, 0x1e, 0x22 // $d8 @ cfa - 48 - 8 * VG
; FP-CHECK-NEXT:    .cfi_escape 0x10, 0x49, 0x0a, 0x11, 0x50, 0x22, 0x11, 0x70, 0x92, 0x2e, 0x00, 0x1e, 0x22 // $d9 @ cfa - 48 - 16 * VG
; FP-CHECK-NEXT:    .cfi_escape 0x10, 0x4a, 0x0a, 0x11, 0x50, 0x22, 0x11, 0x68, 0x92, 0x2e, 0x00, 0x1e, 0x22 // $d10 @ cfa - 48 - 24 * VG
; FP-CHECK-NEXT:    .cfi_escape 0x10, 0x4b, 0x0a, 0x11, 0x50, 0x22, 0x11, 0x60, 0x92, 0x2e, 0x00, 0x1e, 0x22 // $d11 @ cfa - 48 - 32 * VG
; FP-CHECK-NEXT:    .cfi_escape 0x10, 0x4c, 0x0a, 0x11, 0x50, 0x22, 0x11, 0x58, 0x92, 0x2e, 0x00, 0x1e, 0x22 // $d12 @ cfa - 48 - 40 * VG
; FP-CHECK-NEXT:    .cfi_escape 0x10, 0x4d, 0x0a, 0x11, 0x50, 0x22, 0x11, 0x50, 0x92, 0x2e, 0x00, 0x1e, 0x22 // $d13 @ cfa - 48 - 48 * VG
; FP-CHECK-NEXT:    .cfi_escape 0x10, 0x4e, 0x0a, 0x11, 0x50, 0x22, 0x11, 0x48, 0x92, 0x2e, 0x00, 0x1e, 0x22 // $d14 @ cfa - 48 - 56 * VG
; FP-CHECK-NEXT:    .cfi_escape 0x10, 0x4f, 0x0a, 0x11, 0x50, 0x22, 0x11, 0x40, 0x92, 0x2e, 0x00, 0x1e, 0x22 // $d15 @ cfa - 48 - 64 * VG
; FP-CHECK-NEXT:    addvl sp, sp, #-1
; FP-CHECK-NEXT:    str z0, [x29, #-19, mul vl] // 16-byte Folded Spill
; FP-CHECK-NEXT:    //APP
; FP-CHECK-NEXT:    //NO_APP
; FP-CHECK-NEXT:    .cfi_offset vg, -32
; FP-CHECK-NEXT:    smstop sm
; FP-CHECK-NEXT:    ldr z0, [x29, #-19, mul vl] // 16-byte Folded Reload
; FP-CHECK-NEXT:    bl scalable_callee
; FP-CHECK-NEXT:    smstart sm
; FP-CHECK-NEXT:    .cfi_restore vg
; FP-CHECK-NEXT:    addvl sp, sp, #1
; FP-CHECK-NEXT:    ptrue pn8.b
; FP-CHECK-NEXT:    ldr z9, [sp, #16, mul vl] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldr z8, [sp, #17, mul vl] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ld1b { z22.b, z23.b }, pn8/z, [sp, #2, mul vl] // 32-byte Folded Reload
; FP-CHECK-NEXT:    ld1b { z20.b, z21.b }, pn8/z, [sp, #4, mul vl] // 32-byte Folded Reload
; FP-CHECK-NEXT:    ld1b { z18.b, z19.b }, pn8/z, [sp, #6, mul vl] // 32-byte Folded Reload
; FP-CHECK-NEXT:    ld1b { z16.b, z17.b }, pn8/z, [sp, #8, mul vl] // 32-byte Folded Reload
; FP-CHECK-NEXT:    ld1b { z14.b, z15.b }, pn8/z, [sp, #10, mul vl] // 32-byte Folded Reload
; FP-CHECK-NEXT:    ld1b { z12.b, z13.b }, pn8/z, [sp, #12, mul vl] // 32-byte Folded Reload
; FP-CHECK-NEXT:    ld1b { z10.b, z11.b }, pn8/z, [sp, #14, mul vl] // 32-byte Folded Reload
; FP-CHECK-NEXT:    ldr p15, [sp, #4, mul vl] // 2-byte Folded Reload
; FP-CHECK-NEXT:    ldr p14, [sp, #5, mul vl] // 2-byte Folded Reload
; FP-CHECK-NEXT:    ldr p13, [sp, #6, mul vl] // 2-byte Folded Reload
; FP-CHECK-NEXT:    ldr p12, [sp, #7, mul vl] // 2-byte Folded Reload
; FP-CHECK-NEXT:    ldr p11, [sp, #8, mul vl] // 2-byte Folded Reload
; FP-CHECK-NEXT:    ldr p10, [sp, #9, mul vl] // 2-byte Folded Reload
; FP-CHECK-NEXT:    ldr p9, [sp, #10, mul vl] // 2-byte Folded Reload
; FP-CHECK-NEXT:    ldr p8, [sp, #11, mul vl] // 2-byte Folded Reload
; FP-CHECK-NEXT:    ldr p7, [sp, #12, mul vl] // 2-byte Folded Reload
; FP-CHECK-NEXT:    ldr p6, [sp, #13, mul vl] // 2-byte Folded Reload
; FP-CHECK-NEXT:    ldr p5, [sp, #14, mul vl] // 2-byte Folded Reload
; FP-CHECK-NEXT:    ldr p4, [sp, #15, mul vl] // 2-byte Folded Reload
; FP-CHECK-NEXT:    addvl sp, sp, #18
; FP-CHECK-NEXT:    .cfi_restore z8
; FP-CHECK-NEXT:    .cfi_restore z9
; FP-CHECK-NEXT:    .cfi_restore z10
; FP-CHECK-NEXT:    .cfi_restore z11
; FP-CHECK-NEXT:    .cfi_restore z12
; FP-CHECK-NEXT:    .cfi_restore z13
; FP-CHECK-NEXT:    .cfi_restore z14
; FP-CHECK-NEXT:    .cfi_restore z15
; FP-CHECK-NEXT:    .cfi_def_cfa wsp, 48
; FP-CHECK-NEXT:    ldp x28, x27, [sp, #32] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp x29, x30, [sp], #48 // 16-byte Folded Reload
; FP-CHECK-NEXT:    .cfi_def_cfa_offset 0
; FP-CHECK-NEXT:    .cfi_restore w27
; FP-CHECK-NEXT:    .cfi_restore w28
; FP-CHECK-NEXT:    .cfi_restore w30
; FP-CHECK-NEXT:    .cfi_restore w29
; FP-CHECK-NEXT:    ret
;
; OUTLINER-CHECK-LABEL: vg_unwind_with_sve_args:
; OUTLINER-CHECK-NOT: OUTLINED_FUNCTION_
;
  call void asm sideeffect "", "~{x28}"()
  call void @scalable_callee(<vscale x 2 x i64> %x);
  ret void;
}

; This test was based on stack-probing-64k.ll and tries to test multiple uses of
; findScratchNonCalleeSaveRegister.
;
define void @vg_unwind_multiple_scratch_regs(ptr %out) #1 {
; CHECK-LABEL: vg_unwind_multiple_scratch_regs:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    stp d15, d14, [sp, #-96]! // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 96
; CHECK-NEXT:    cntd x9
; CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; CHECK-NEXT:    stp x29, x30, [sp, #64] // 16-byte Folded Spill
; CHECK-NEXT:    str x9, [sp, #80] // 8-byte Folded Spill
; CHECK-NEXT:    .cfi_offset w30, -24
; CHECK-NEXT:    .cfi_offset w29, -32
; CHECK-NEXT:    .cfi_offset b8, -40
; CHECK-NEXT:    .cfi_offset b9, -48
; CHECK-NEXT:    .cfi_offset b10, -56
; CHECK-NEXT:    .cfi_offset b11, -64
; CHECK-NEXT:    .cfi_offset b12, -72
; CHECK-NEXT:    .cfi_offset b13, -80
; CHECK-NEXT:    .cfi_offset b14, -88
; CHECK-NEXT:    .cfi_offset b15, -96
; CHECK-NEXT:    sub x9, sp, #80, lsl #12 // =327680
; CHECK-NEXT:    .cfi_def_cfa w9, 327776
; CHECK-NEXT:  .LBB4_1: // %entry
; CHECK-NEXT:    // =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    sub sp, sp, #1, lsl #12 // =4096
; CHECK-NEXT:    cmp sp, x9
; CHECK-NEXT:    str xzr, [sp]
; CHECK-NEXT:    b.ne .LBB4_1
; CHECK-NEXT:  // %bb.2: // %entry
; CHECK-NEXT:    .cfi_def_cfa_register wsp
; CHECK-NEXT:    mov x8, sp
; CHECK-NEXT:    str x8, [x0]
; CHECK-NEXT:    .cfi_offset vg, -16
; CHECK-NEXT:    smstop sm
; CHECK-NEXT:    bl callee
; CHECK-NEXT:    smstart sm
; CHECK-NEXT:    .cfi_restore vg
; CHECK-NEXT:    add sp, sp, #80, lsl #12 // =327680
; CHECK-NEXT:    .cfi_def_cfa_offset 96
; CHECK-NEXT:    ldp x29, x30, [sp, #64] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d11, d10, [sp, #32] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d13, d12, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d15, d14, [sp], #96 // 16-byte Folded Reload
; CHECK-NEXT:    .cfi_def_cfa_offset 0
; CHECK-NEXT:    .cfi_restore w30
; CHECK-NEXT:    .cfi_restore w29
; CHECK-NEXT:    .cfi_restore b8
; CHECK-NEXT:    .cfi_restore b9
; CHECK-NEXT:    .cfi_restore b10
; CHECK-NEXT:    .cfi_restore b11
; CHECK-NEXT:    .cfi_restore b12
; CHECK-NEXT:    .cfi_restore b13
; CHECK-NEXT:    .cfi_restore b14
; CHECK-NEXT:    .cfi_restore b15
; CHECK-NEXT:    ret
;
; FP-CHECK-LABEL: vg_unwind_multiple_scratch_regs:
; FP-CHECK:       // %bb.0: // %entry
; FP-CHECK-NEXT:    stp d15, d14, [sp, #-96]! // 16-byte Folded Spill
; FP-CHECK-NEXT:    .cfi_def_cfa_offset 96
; FP-CHECK-NEXT:    cntd x9
; FP-CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp x29, x30, [sp, #64] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp x9, x28, [sp, #80] // 16-byte Folded Spill
; FP-CHECK-NEXT:    add x29, sp, #64
; FP-CHECK-NEXT:    .cfi_def_cfa w29, 32
; FP-CHECK-NEXT:    .cfi_offset w28, -8
; FP-CHECK-NEXT:    .cfi_offset w30, -24
; FP-CHECK-NEXT:    .cfi_offset w29, -32
; FP-CHECK-NEXT:    .cfi_offset b8, -40
; FP-CHECK-NEXT:    .cfi_offset b9, -48
; FP-CHECK-NEXT:    .cfi_offset b10, -56
; FP-CHECK-NEXT:    .cfi_offset b11, -64
; FP-CHECK-NEXT:    .cfi_offset b12, -72
; FP-CHECK-NEXT:    .cfi_offset b13, -80
; FP-CHECK-NEXT:    .cfi_offset b14, -88
; FP-CHECK-NEXT:    .cfi_offset b15, -96
; FP-CHECK-NEXT:    sub x9, sp, #80, lsl #12 // =327680
; FP-CHECK-NEXT:  .LBB4_1: // %entry
; FP-CHECK-NEXT:    // =>This Inner Loop Header: Depth=1
; FP-CHECK-NEXT:    sub sp, sp, #1, lsl #12 // =4096
; FP-CHECK-NEXT:    cmp sp, x9
; FP-CHECK-NEXT:    str xzr, [sp]
; FP-CHECK-NEXT:    b.ne .LBB4_1
; FP-CHECK-NEXT:  // %bb.2: // %entry
; FP-CHECK-NEXT:    mov x8, sp
; FP-CHECK-NEXT:    str x8, [x0]
; FP-CHECK-NEXT:    .cfi_offset vg, -16
; FP-CHECK-NEXT:    smstop sm
; FP-CHECK-NEXT:    bl callee
; FP-CHECK-NEXT:    smstart sm
; FP-CHECK-NEXT:    .cfi_restore vg
; FP-CHECK-NEXT:    add sp, sp, #80, lsl #12 // =327680
; FP-CHECK-NEXT:    .cfi_def_cfa wsp, 96
; FP-CHECK-NEXT:    ldp x29, x30, [sp, #64] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldr x28, [sp, #88] // 8-byte Folded Reload
; FP-CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d11, d10, [sp, #32] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d13, d12, [sp, #16] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d15, d14, [sp], #96 // 16-byte Folded Reload
; FP-CHECK-NEXT:    .cfi_def_cfa_offset 0
; FP-CHECK-NEXT:    .cfi_restore w28
; FP-CHECK-NEXT:    .cfi_restore w30
; FP-CHECK-NEXT:    .cfi_restore w29
; FP-CHECK-NEXT:    .cfi_restore b8
; FP-CHECK-NEXT:    .cfi_restore b9
; FP-CHECK-NEXT:    .cfi_restore b10
; FP-CHECK-NEXT:    .cfi_restore b11
; FP-CHECK-NEXT:    .cfi_restore b12
; FP-CHECK-NEXT:    .cfi_restore b13
; FP-CHECK-NEXT:    .cfi_restore b14
; FP-CHECK-NEXT:    .cfi_restore b15
; FP-CHECK-NEXT:    ret
;
; OUTLINER-CHECK-LABEL: vg_unwind_multiple_scratch_regs:
; OUTLINER-CHECK-NOT: OUTLINED_FUNCTION_
;
entry:
  %v = alloca i8, i64 327680, align 1
  store ptr %v, ptr %out, align 8
  call void @callee()
  ret void
}

; Locally streaming functions require storing both the streaming and
; non-streaming values of VG.
;
define void @vg_locally_streaming_fn() #3 {
; CHECK-LABEL: vg_locally_streaming_fn:
; CHECK:       // %bb.0:
; CHECK-NEXT:    stp d15, d14, [sp, #-96]! // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 96
; CHECK-NEXT:    rdsvl x9, #1
; CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    lsr x9, x9, #3
; CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; CHECK-NEXT:    stp x30, x9, [sp, #64] // 16-byte Folded Spill
; CHECK-NEXT:    cntd x9
; CHECK-NEXT:    str x9, [sp, #80] // 8-byte Folded Spill
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
; CHECK-NEXT:    bl callee
; CHECK-NEXT:    smstart sm
; CHECK-NEXT:    .cfi_restore vg
; CHECK-NEXT:    bl streaming_callee
; CHECK-NEXT:    .cfi_offset vg, -24
; CHECK-NEXT:    smstop sm
; CHECK-NEXT:    bl callee
; CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; CHECK-NEXT:    ldr x30, [sp, #64] // 8-byte Folded Reload
; CHECK-NEXT:    ldp d11, d10, [sp, #32] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d13, d12, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d15, d14, [sp], #96 // 16-byte Folded Reload
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
;
; FP-CHECK-LABEL: vg_locally_streaming_fn:
; FP-CHECK:       // %bb.0:
; FP-CHECK-NEXT:    stp d15, d14, [sp, #-96]! // 16-byte Folded Spill
; FP-CHECK-NEXT:    .cfi_def_cfa_offset 96
; FP-CHECK-NEXT:    rdsvl x9, #1
; FP-CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; FP-CHECK-NEXT:    lsr x9, x9, #3
; FP-CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; FP-CHECK-NEXT:    str x9, [sp, #80] // 8-byte Folded Spill
; FP-CHECK-NEXT:    cntd x9
; FP-CHECK-NEXT:    stp x29, x30, [sp, #64] // 16-byte Folded Spill
; FP-CHECK-NEXT:    str x9, [sp, #88] // 8-byte Folded Spill
; FP-CHECK-NEXT:    add x29, sp, #64
; FP-CHECK-NEXT:    .cfi_def_cfa w29, 32
; FP-CHECK-NEXT:    .cfi_offset vg, -8
; FP-CHECK-NEXT:    .cfi_offset w30, -24
; FP-CHECK-NEXT:    .cfi_offset w29, -32
; FP-CHECK-NEXT:    .cfi_offset b8, -40
; FP-CHECK-NEXT:    .cfi_offset b9, -48
; FP-CHECK-NEXT:    .cfi_offset b10, -56
; FP-CHECK-NEXT:    .cfi_offset b11, -64
; FP-CHECK-NEXT:    .cfi_offset b12, -72
; FP-CHECK-NEXT:    .cfi_offset b13, -80
; FP-CHECK-NEXT:    .cfi_offset b14, -88
; FP-CHECK-NEXT:    .cfi_offset b15, -96
; FP-CHECK-NEXT:    bl callee
; FP-CHECK-NEXT:    smstart sm
; FP-CHECK-NEXT:    .cfi_restore vg
; FP-CHECK-NEXT:    bl streaming_callee
; FP-CHECK-NEXT:    .cfi_offset vg, -16
; FP-CHECK-NEXT:    smstop sm
; FP-CHECK-NEXT:    bl callee
; FP-CHECK-NEXT:    .cfi_def_cfa wsp, 96
; FP-CHECK-NEXT:    ldp x29, x30, [sp, #64] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d11, d10, [sp, #32] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d13, d12, [sp, #16] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d15, d14, [sp], #96 // 16-byte Folded Reload
; FP-CHECK-NEXT:    .cfi_def_cfa_offset 0
; FP-CHECK-NEXT:    .cfi_restore w30
; FP-CHECK-NEXT:    .cfi_restore w29
; FP-CHECK-NEXT:    .cfi_restore b8
; FP-CHECK-NEXT:    .cfi_restore b9
; FP-CHECK-NEXT:    .cfi_restore b10
; FP-CHECK-NEXT:    .cfi_restore b11
; FP-CHECK-NEXT:    .cfi_restore b12
; FP-CHECK-NEXT:    .cfi_restore b13
; FP-CHECK-NEXT:    .cfi_restore b14
; FP-CHECK-NEXT:    .cfi_restore b15
; FP-CHECK-NEXT:    ret
;
; OUTLINER-CHECK-LABEL: vg_locally_streaming_fn:
; OUTLINER-CHECK-NOT: OUTLINED_FUNCTION_
;
  call void @callee()
  call void @streaming_callee()
  call void @callee()
  ret void
}

define void @streaming_compatible_to_streaming() #4 {
; CHECK-LABEL: streaming_compatible_to_streaming:
; CHECK:       // %bb.0:
; CHECK-NEXT:    stp d15, d14, [sp, #-96]! // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 96
; CHECK-NEXT:    cntd x9
; CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; CHECK-NEXT:    stp x30, x9, [sp, #64] // 16-byte Folded Spill
; CHECK-NEXT:    str x19, [sp, #80] // 8-byte Folded Spill
; CHECK-NEXT:    .cfi_offset w19, -16
; CHECK-NEXT:    .cfi_offset w30, -32
; CHECK-NEXT:    .cfi_offset b8, -40
; CHECK-NEXT:    .cfi_offset b9, -48
; CHECK-NEXT:    .cfi_offset b10, -56
; CHECK-NEXT:    .cfi_offset b11, -64
; CHECK-NEXT:    .cfi_offset b12, -72
; CHECK-NEXT:    .cfi_offset b13, -80
; CHECK-NEXT:    .cfi_offset b14, -88
; CHECK-NEXT:    .cfi_offset b15, -96
; CHECK-NEXT:    bl __arm_sme_state
; CHECK-NEXT:    and x19, x0, #0x1
; CHECK-NEXT:    .cfi_offset vg, -24
; CHECK-NEXT:    tbnz w19, #0, .LBB6_2
; CHECK-NEXT:  // %bb.1:
; CHECK-NEXT:    smstart sm
; CHECK-NEXT:  .LBB6_2:
; CHECK-NEXT:    bl streaming_callee
; CHECK-NEXT:    tbnz w19, #0, .LBB6_4
; CHECK-NEXT:  // %bb.3:
; CHECK-NEXT:    smstop sm
; CHECK-NEXT:  .LBB6_4:
; CHECK-NEXT:    .cfi_restore vg
; CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; CHECK-NEXT:    ldr x19, [sp, #80] // 8-byte Folded Reload
; CHECK-NEXT:    ldp d11, d10, [sp, #32] // 16-byte Folded Reload
; CHECK-NEXT:    ldr x30, [sp, #64] // 8-byte Folded Reload
; CHECK-NEXT:    ldp d13, d12, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d15, d14, [sp], #96 // 16-byte Folded Reload
; CHECK-NEXT:    .cfi_def_cfa_offset 0
; CHECK-NEXT:    .cfi_restore w19
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
;
; FP-CHECK-LABEL: streaming_compatible_to_streaming:
; FP-CHECK:       // %bb.0:
; FP-CHECK-NEXT:    stp d15, d14, [sp, #-96]! // 16-byte Folded Spill
; FP-CHECK-NEXT:    .cfi_def_cfa_offset 96
; FP-CHECK-NEXT:    cntd x9
; FP-CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp x29, x30, [sp, #64] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp x9, x19, [sp, #80] // 16-byte Folded Spill
; FP-CHECK-NEXT:    add x29, sp, #64
; FP-CHECK-NEXT:    .cfi_def_cfa w29, 32
; FP-CHECK-NEXT:    .cfi_offset w19, -8
; FP-CHECK-NEXT:    .cfi_offset w30, -24
; FP-CHECK-NEXT:    .cfi_offset w29, -32
; FP-CHECK-NEXT:    .cfi_offset b8, -40
; FP-CHECK-NEXT:    .cfi_offset b9, -48
; FP-CHECK-NEXT:    .cfi_offset b10, -56
; FP-CHECK-NEXT:    .cfi_offset b11, -64
; FP-CHECK-NEXT:    .cfi_offset b12, -72
; FP-CHECK-NEXT:    .cfi_offset b13, -80
; FP-CHECK-NEXT:    .cfi_offset b14, -88
; FP-CHECK-NEXT:    .cfi_offset b15, -96
; FP-CHECK-NEXT:    bl __arm_sme_state
; FP-CHECK-NEXT:    and x19, x0, #0x1
; FP-CHECK-NEXT:    .cfi_offset vg, -16
; FP-CHECK-NEXT:    tbnz w19, #0, .LBB6_2
; FP-CHECK-NEXT:  // %bb.1:
; FP-CHECK-NEXT:    smstart sm
; FP-CHECK-NEXT:  .LBB6_2:
; FP-CHECK-NEXT:    bl streaming_callee
; FP-CHECK-NEXT:    tbnz w19, #0, .LBB6_4
; FP-CHECK-NEXT:  // %bb.3:
; FP-CHECK-NEXT:    smstop sm
; FP-CHECK-NEXT:  .LBB6_4:
; FP-CHECK-NEXT:    .cfi_restore vg
; FP-CHECK-NEXT:    .cfi_def_cfa wsp, 96
; FP-CHECK-NEXT:    ldp x29, x30, [sp, #64] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldr x19, [sp, #88] // 8-byte Folded Reload
; FP-CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d11, d10, [sp, #32] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d13, d12, [sp, #16] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d15, d14, [sp], #96 // 16-byte Folded Reload
; FP-CHECK-NEXT:    .cfi_def_cfa_offset 0
; FP-CHECK-NEXT:    .cfi_restore w19
; FP-CHECK-NEXT:    .cfi_restore w30
; FP-CHECK-NEXT:    .cfi_restore w29
; FP-CHECK-NEXT:    .cfi_restore b8
; FP-CHECK-NEXT:    .cfi_restore b9
; FP-CHECK-NEXT:    .cfi_restore b10
; FP-CHECK-NEXT:    .cfi_restore b11
; FP-CHECK-NEXT:    .cfi_restore b12
; FP-CHECK-NEXT:    .cfi_restore b13
; FP-CHECK-NEXT:    .cfi_restore b14
; FP-CHECK-NEXT:    .cfi_restore b15
; FP-CHECK-NEXT:    ret
;
; OUTLINER-CHECK-LABEL: streaming_compatible_to_streaming:
; OUTLINER-CHECK-NOT: OUTLINED_FUNCTION_
;
  call void @streaming_callee()
  ret void
}

define void @streaming_compatible_to_non_streaming() #4 {
; CHECK-LABEL: streaming_compatible_to_non_streaming:
; CHECK:       // %bb.0:
; CHECK-NEXT:    stp d15, d14, [sp, #-96]! // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 96
; CHECK-NEXT:    cntd x9
; CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; CHECK-NEXT:    stp x30, x9, [sp, #64] // 16-byte Folded Spill
; CHECK-NEXT:    str x19, [sp, #80] // 8-byte Folded Spill
; CHECK-NEXT:    .cfi_offset w19, -16
; CHECK-NEXT:    .cfi_offset w30, -32
; CHECK-NEXT:    .cfi_offset b8, -40
; CHECK-NEXT:    .cfi_offset b9, -48
; CHECK-NEXT:    .cfi_offset b10, -56
; CHECK-NEXT:    .cfi_offset b11, -64
; CHECK-NEXT:    .cfi_offset b12, -72
; CHECK-NEXT:    .cfi_offset b13, -80
; CHECK-NEXT:    .cfi_offset b14, -88
; CHECK-NEXT:    .cfi_offset b15, -96
; CHECK-NEXT:    bl __arm_sme_state
; CHECK-NEXT:    and x19, x0, #0x1
; CHECK-NEXT:    .cfi_offset vg, -24
; CHECK-NEXT:    tbz w19, #0, .LBB7_2
; CHECK-NEXT:  // %bb.1:
; CHECK-NEXT:    smstop sm
; CHECK-NEXT:  .LBB7_2:
; CHECK-NEXT:    bl callee
; CHECK-NEXT:    tbz w19, #0, .LBB7_4
; CHECK-NEXT:  // %bb.3:
; CHECK-NEXT:    smstart sm
; CHECK-NEXT:  .LBB7_4:
; CHECK-NEXT:    .cfi_restore vg
; CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; CHECK-NEXT:    ldr x19, [sp, #80] // 8-byte Folded Reload
; CHECK-NEXT:    ldp d11, d10, [sp, #32] // 16-byte Folded Reload
; CHECK-NEXT:    ldr x30, [sp, #64] // 8-byte Folded Reload
; CHECK-NEXT:    ldp d13, d12, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d15, d14, [sp], #96 // 16-byte Folded Reload
; CHECK-NEXT:    .cfi_def_cfa_offset 0
; CHECK-NEXT:    .cfi_restore w19
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
;
; FP-CHECK-LABEL: streaming_compatible_to_non_streaming:
; FP-CHECK:       // %bb.0:
; FP-CHECK-NEXT:    stp d15, d14, [sp, #-96]! // 16-byte Folded Spill
; FP-CHECK-NEXT:    .cfi_def_cfa_offset 96
; FP-CHECK-NEXT:    cntd x9
; FP-CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp x29, x30, [sp, #64] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp x9, x19, [sp, #80] // 16-byte Folded Spill
; FP-CHECK-NEXT:    add x29, sp, #64
; FP-CHECK-NEXT:    .cfi_def_cfa w29, 32
; FP-CHECK-NEXT:    .cfi_offset w19, -8
; FP-CHECK-NEXT:    .cfi_offset w30, -24
; FP-CHECK-NEXT:    .cfi_offset w29, -32
; FP-CHECK-NEXT:    .cfi_offset b8, -40
; FP-CHECK-NEXT:    .cfi_offset b9, -48
; FP-CHECK-NEXT:    .cfi_offset b10, -56
; FP-CHECK-NEXT:    .cfi_offset b11, -64
; FP-CHECK-NEXT:    .cfi_offset b12, -72
; FP-CHECK-NEXT:    .cfi_offset b13, -80
; FP-CHECK-NEXT:    .cfi_offset b14, -88
; FP-CHECK-NEXT:    .cfi_offset b15, -96
; FP-CHECK-NEXT:    bl __arm_sme_state
; FP-CHECK-NEXT:    and x19, x0, #0x1
; FP-CHECK-NEXT:    .cfi_offset vg, -16
; FP-CHECK-NEXT:    tbz w19, #0, .LBB7_2
; FP-CHECK-NEXT:  // %bb.1:
; FP-CHECK-NEXT:    smstop sm
; FP-CHECK-NEXT:  .LBB7_2:
; FP-CHECK-NEXT:    bl callee
; FP-CHECK-NEXT:    tbz w19, #0, .LBB7_4
; FP-CHECK-NEXT:  // %bb.3:
; FP-CHECK-NEXT:    smstart sm
; FP-CHECK-NEXT:  .LBB7_4:
; FP-CHECK-NEXT:    .cfi_restore vg
; FP-CHECK-NEXT:    .cfi_def_cfa wsp, 96
; FP-CHECK-NEXT:    ldp x29, x30, [sp, #64] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldr x19, [sp, #88] // 8-byte Folded Reload
; FP-CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d11, d10, [sp, #32] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d13, d12, [sp, #16] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d15, d14, [sp], #96 // 16-byte Folded Reload
; FP-CHECK-NEXT:    .cfi_def_cfa_offset 0
; FP-CHECK-NEXT:    .cfi_restore w19
; FP-CHECK-NEXT:    .cfi_restore w30
; FP-CHECK-NEXT:    .cfi_restore w29
; FP-CHECK-NEXT:    .cfi_restore b8
; FP-CHECK-NEXT:    .cfi_restore b9
; FP-CHECK-NEXT:    .cfi_restore b10
; FP-CHECK-NEXT:    .cfi_restore b11
; FP-CHECK-NEXT:    .cfi_restore b12
; FP-CHECK-NEXT:    .cfi_restore b13
; FP-CHECK-NEXT:    .cfi_restore b14
; FP-CHECK-NEXT:    .cfi_restore b15
; FP-CHECK-NEXT:    ret
;
; OUTLINER-CHECK-LABEL: streaming_compatible_to_non_streaming:
; OUTLINER-CHECK-NOT: OUTLINED_FUNCTION_
;
  call void @callee()
  ret void
}

; If the target does not have SVE, do not emit cntd in the prologue and
; instead spill the result returned by __arm_get_current_vg.
; This requires preserving the argument %x as the vg value is returned
; in X0.
;
define void @streaming_compatible_no_sve(i32 noundef %x) #4 {
; NO-SVE-CHECK-LABEL: streaming_compatible_no_sve:
; NO-SVE-CHECK:       // %bb.0:
; NO-SVE-CHECK-NEXT:    stp d15, d14, [sp, #-96]! // 16-byte Folded Spill
; NO-SVE-CHECK-NEXT:    .cfi_def_cfa_offset 96
; NO-SVE-CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; NO-SVE-CHECK-NEXT:    mov x9, x0
; NO-SVE-CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; NO-SVE-CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; NO-SVE-CHECK-NEXT:    stp x29, x30, [sp, #64] // 16-byte Folded Spill
; NO-SVE-CHECK-NEXT:    bl __arm_get_current_vg
; NO-SVE-CHECK-NEXT:    stp x0, x19, [sp, #80] // 16-byte Folded Spill
; NO-SVE-CHECK-NEXT:    mov x0, x9
; NO-SVE-CHECK-NEXT:    add x29, sp, #64
; NO-SVE-CHECK-NEXT:    .cfi_def_cfa w29, 32
; NO-SVE-CHECK-NEXT:    .cfi_offset w19, -8
; NO-SVE-CHECK-NEXT:    .cfi_offset w30, -24
; NO-SVE-CHECK-NEXT:    .cfi_offset w29, -32
; NO-SVE-CHECK-NEXT:    .cfi_offset b8, -40
; NO-SVE-CHECK-NEXT:    .cfi_offset b9, -48
; NO-SVE-CHECK-NEXT:    .cfi_offset b10, -56
; NO-SVE-CHECK-NEXT:    .cfi_offset b11, -64
; NO-SVE-CHECK-NEXT:    .cfi_offset b12, -72
; NO-SVE-CHECK-NEXT:    .cfi_offset b13, -80
; NO-SVE-CHECK-NEXT:    .cfi_offset b14, -88
; NO-SVE-CHECK-NEXT:    .cfi_offset b15, -96
; NO-SVE-CHECK-NEXT:    mov w8, w0
; NO-SVE-CHECK-NEXT:    bl __arm_sme_state
; NO-SVE-CHECK-NEXT:    and x19, x0, #0x1
; NO-SVE-CHECK-NEXT:    .cfi_offset vg, -16
; NO-SVE-CHECK-NEXT:    tbnz w19, #0, .LBB8_2
; NO-SVE-CHECK-NEXT:  // %bb.1:
; NO-SVE-CHECK-NEXT:    smstart sm
; NO-SVE-CHECK-NEXT:  .LBB8_2:
; NO-SVE-CHECK-NEXT:    mov w0, w8
; NO-SVE-CHECK-NEXT:    bl streaming_callee_with_arg
; NO-SVE-CHECK-NEXT:    tbnz w19, #0, .LBB8_4
; NO-SVE-CHECK-NEXT:  // %bb.3:
; NO-SVE-CHECK-NEXT:    smstop sm
; NO-SVE-CHECK-NEXT:  .LBB8_4:
; NO-SVE-CHECK-NEXT:    .cfi_restore vg
; NO-SVE-CHECK-NEXT:    .cfi_def_cfa wsp, 96
; NO-SVE-CHECK-NEXT:    ldp x29, x30, [sp, #64] // 16-byte Folded Reload
; NO-SVE-CHECK-NEXT:    ldr x19, [sp, #88] // 8-byte Folded Reload
; NO-SVE-CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; NO-SVE-CHECK-NEXT:    ldp d11, d10, [sp, #32] // 16-byte Folded Reload
; NO-SVE-CHECK-NEXT:    ldp d13, d12, [sp, #16] // 16-byte Folded Reload
; NO-SVE-CHECK-NEXT:    ldp d15, d14, [sp], #96 // 16-byte Folded Reload
; NO-SVE-CHECK-NEXT:    .cfi_def_cfa_offset 0
; NO-SVE-CHECK-NEXT:    .cfi_restore w19
; NO-SVE-CHECK-NEXT:    .cfi_restore w30
; NO-SVE-CHECK-NEXT:    .cfi_restore w29
; NO-SVE-CHECK-NEXT:    .cfi_restore b8
; NO-SVE-CHECK-NEXT:    .cfi_restore b9
; NO-SVE-CHECK-NEXT:    .cfi_restore b10
; NO-SVE-CHECK-NEXT:    .cfi_restore b11
; NO-SVE-CHECK-NEXT:    .cfi_restore b12
; NO-SVE-CHECK-NEXT:    .cfi_restore b13
; NO-SVE-CHECK-NEXT:    .cfi_restore b14
; NO-SVE-CHECK-NEXT:    .cfi_restore b15
; NO-SVE-CHECK-NEXT:    ret
;
; OUTLINER-CHECK-LABEL: streaming_compatible_no_sve:
; OUTLINER-CHECK-NOT: OUTLINED_FUNCTION_
;
  call void @streaming_callee_with_arg(i32 %x)
  ret void
}

; The algorithm that fixes up the offsets of the callee-save/restore
; instructions must jump over the instructions that instantiate the current
; 'VG' value. We must make sure that it doesn't consider any RDSVL in
; user-code as if it is part of the frame-setup when doing so.
define void @test_rdsvl_right_after_prologue(i64 %x0) nounwind {
; NO-SVE-CHECK-LABEL: test_rdsvl_right_after_prologue:
; NO-SVE-CHECK:     // %bb.0:
; NO-SVE-CHECK-NEXT: stp     d15, d14, [sp, #-96]!           // 16-byte Folded Spill
; NO-SVE-CHECK-NEXT: stp     d13, d12, [sp, #16]             // 16-byte Folded Spill
; NO-SVE-CHECK-NEXT: mov     x9, x0
; NO-SVE-CHECK-NEXT: stp     d11, d10, [sp, #32]             // 16-byte Folded Spill
; NO-SVE-CHECK-NEXT: stp     d9, d8, [sp, #48]               // 16-byte Folded Spill
; NO-SVE-CHECK-NEXT: stp     x29, x30, [sp, #64]             // 16-byte Folded Spill
; NO-SVE-CHECK-NEXT: bl      __arm_get_current_vg
; NO-SVE-CHECK-NEXT: str     x0, [sp, #80]                   // 8-byte Folded Spill
; NO-SVE-CHECK-NEXT: mov     x0, x9
; NO-SVE-CHECK-NEXT: rdsvl   x8, #1
; NO-SVE-CHECK-NEXT: add     x29, sp, #64
; NO-SVE-CHECK-NEXT: lsr     x8, x8, #3
; NO-SVE-CHECK-NEXT: mov     x1, x0
; NO-SVE-CHECK-NEXT: smstart sm
; NO-SVE-CHECK-NEXT: mov     x0, x8
; NO-SVE-CHECK-NEXT: bl      bar
; NO-SVE-CHECK-NEXT: smstop  sm
; NO-SVE-CHECK-NEXT: ldp     x29, x30, [sp, #64]             // 16-byte Folded Reload
; NO-SVE-CHECK-NEXT: ldp     d9, d8, [sp, #48]               // 16-byte Folded Reload
; NO-SVE-CHECK-NEXT: ldp     d11, d10, [sp, #32]             // 16-byte Folded Reload
; NO-SVE-CHECK-NEXT: ldp     d13, d12, [sp, #16]             // 16-byte Folded Reload
; NO-SVE-CHECK-NEXT: ldp     d15, d14, [sp], #96             // 16-byte Folded Reload
; NO-SVE-CHECK-NEXT: ret
  %some_alloc = alloca i64, align 8
  %rdsvl = tail call i64 @llvm.aarch64.sme.cntsd()
  call void @bar(i64 %rdsvl, i64 %x0) "aarch64_pstate_sm_enabled"
  ret void
}

declare void @bar(i64, i64)

; Ensure we still emit async unwind information with -fno-asynchronous-unwind-tables
; if the function contains a streaming-mode change.

define void @vg_unwind_noasync() #5 {
; CHECK-LABEL: vg_unwind_noasync:
; CHECK:       // %bb.0:
; CHECK-NEXT:    stp d15, d14, [sp, #-80]! // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 80
; CHECK-NEXT:    cntd x9
; CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; CHECK-NEXT:    stp x30, x9, [sp, #64] // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_offset w30, -16
; CHECK-NEXT:    .cfi_offset b8, -24
; CHECK-NEXT:    .cfi_offset b9, -32
; CHECK-NEXT:    .cfi_offset b10, -40
; CHECK-NEXT:    .cfi_offset b11, -48
; CHECK-NEXT:    .cfi_offset b12, -56
; CHECK-NEXT:    .cfi_offset b13, -64
; CHECK-NEXT:    .cfi_offset b14, -72
; CHECK-NEXT:    .cfi_offset b15, -80
; CHECK-NEXT:    .cfi_offset vg, -8
; CHECK-NEXT:    smstop sm
; CHECK-NEXT:    bl callee
; CHECK-NEXT:    smstart sm
; CHECK-NEXT:    .cfi_restore vg
; CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; CHECK-NEXT:    ldr x30, [sp, #64] // 8-byte Folded Reload
; CHECK-NEXT:    ldp d11, d10, [sp, #32] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d13, d12, [sp, #16] // 16-byte Folded Reload
; CHECK-NEXT:    ldp d15, d14, [sp], #80 // 16-byte Folded Reload
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
;
; FP-CHECK-LABEL: vg_unwind_noasync:
; FP-CHECK:       // %bb.0:
; FP-CHECK-NEXT:    stp d15, d14, [sp, #-96]! // 16-byte Folded Spill
; FP-CHECK-NEXT:    .cfi_def_cfa_offset 96
; FP-CHECK-NEXT:    cntd x9
; FP-CHECK-NEXT:    stp d13, d12, [sp, #16] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp d11, d10, [sp, #32] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp d9, d8, [sp, #48] // 16-byte Folded Spill
; FP-CHECK-NEXT:    stp x29, x30, [sp, #64] // 16-byte Folded Spill
; FP-CHECK-NEXT:    str x9, [sp, #80] // 8-byte Folded Spill
; FP-CHECK-NEXT:    add x29, sp, #64
; FP-CHECK-NEXT:    .cfi_def_cfa w29, 32
; FP-CHECK-NEXT:    .cfi_offset w30, -24
; FP-CHECK-NEXT:    .cfi_offset w29, -32
; FP-CHECK-NEXT:    .cfi_offset b8, -40
; FP-CHECK-NEXT:    .cfi_offset b9, -48
; FP-CHECK-NEXT:    .cfi_offset b10, -56
; FP-CHECK-NEXT:    .cfi_offset b11, -64
; FP-CHECK-NEXT:    .cfi_offset b12, -72
; FP-CHECK-NEXT:    .cfi_offset b13, -80
; FP-CHECK-NEXT:    .cfi_offset b14, -88
; FP-CHECK-NEXT:    .cfi_offset b15, -96
; FP-CHECK-NEXT:    .cfi_offset vg, -16
; FP-CHECK-NEXT:    smstop sm
; FP-CHECK-NEXT:    bl callee
; FP-CHECK-NEXT:    smstart sm
; FP-CHECK-NEXT:    .cfi_restore vg
; FP-CHECK-NEXT:    .cfi_def_cfa wsp, 96
; FP-CHECK-NEXT:    ldp x29, x30, [sp, #64] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d9, d8, [sp, #48] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d11, d10, [sp, #32] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d13, d12, [sp, #16] // 16-byte Folded Reload
; FP-CHECK-NEXT:    ldp d15, d14, [sp], #96 // 16-byte Folded Reload
; FP-CHECK-NEXT:    .cfi_def_cfa_offset 0
; FP-CHECK-NEXT:    .cfi_restore w30
; FP-CHECK-NEXT:    .cfi_restore w29
; FP-CHECK-NEXT:    .cfi_restore b8
; FP-CHECK-NEXT:    .cfi_restore b9
; FP-CHECK-NEXT:    .cfi_restore b10
; FP-CHECK-NEXT:    .cfi_restore b11
; FP-CHECK-NEXT:    .cfi_restore b12
; FP-CHECK-NEXT:    .cfi_restore b13
; FP-CHECK-NEXT:    .cfi_restore b14
; FP-CHECK-NEXT:    .cfi_restore b15
; FP-CHECK-NEXT:    ret
; OUTLINER-CHECK-LABEL: vg_unwind_noasync:
; OUTLINER-CHECK-NOT: OUTLINED_FUNCTION_
;
  call void @callee();
  ret void;
}

attributes #0 = { "aarch64_pstate_sm_enabled" uwtable(async) }
attributes #1 = { "probe-stack"="inline-asm" "aarch64_pstate_sm_enabled" uwtable(async) }
attributes #3 = { "aarch64_pstate_sm_body" uwtable(async) }
attributes #4 = { "aarch64_pstate_sm_compatible" uwtable(async) }
attributes #5 = { "aarch64_pstate_sm_enabled" }
