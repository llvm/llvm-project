; DEFINE: %{compile} = llc -mtriple=aarch64-linux-gnu -aarch64-streaming-hazard-size=0 -mattr=+sme -mattr=+sve -verify-machineinstrs -enable-aarch64-sme-peephole-opt=false < %s
; RUN: %{compile} | FileCheck %s
; RUN: %{compile} -filetype=obj -o %t
; RUN: llvm-objdump --dwarf=frames %t | FileCheck %s --check-prefix=UNWINDINFO

; This tests that functions with streaming mode changes use explicitly use the
; "IncomingVG" (the value of VG on entry to the function) in SVE unwind information.
;
; [ ] N  ->  S    (Normal -> Streaming, mode change)
; [ ] S  ->  N    (Streaming -> Normal, mode change)
; [ ] N  ->  N    (Normal -> Normal, no mode change)
; [ ] S  ->  S    (Streaming -> Streaming, no mode change)
; [ ] LS ->  S    (Locally-streaming -> Streaming, mode change)
; [ ] SC ->  S    (Streaming-compatible -> Streaming, mode change)

declare void @normal_callee()
declare void @streaming_callee() "aarch64_pstate_sm_enabled"

; [x] N  ->  S
; [ ] S  ->  N
; [ ] N  ->  N
; [ ] S  ->  S
; [ ] LS ->  S
; [ ] SC ->  S
define aarch64_sve_vector_pcs void @normal_caller_streaming_callee() {
; CHECK-LABEL: normal_caller_streaming_callee:
; CHECK:    stp x29, x30, [sp, #-32]! // 16-byte Folded Spill
; CHECK:    .cfi_def_cfa_offset 32
; CHECK:    cntd x9
; CHECK:    stp x9, x28, [sp, #16] // 16-byte Folded Spill
; CHECK:    mov x29, sp
; CHECK:    .cfi_def_cfa w29, 32
; CHECK:    .cfi_offset vg, -16
; CHECK:    addvl sp, sp, #-18
; CHECK:    str z15, [sp, #10, mul vl] // 16-byte Folded Spill
; CHECK:    str z14, [sp, #11, mul vl] // 16-byte Folded Spill
; CHECK:    str z13, [sp, #12, mul vl] // 16-byte Folded Spill
; CHECK:    str z12, [sp, #13, mul vl] // 16-byte Folded Spill
; CHECK:    str z11, [sp, #14, mul vl] // 16-byte Folded Spill
; CHECK:    str z10, [sp, #15, mul vl] // 16-byte Folded Spill
; CHECK:    str z9, [sp, #16, mul vl] // 16-byte Folded Spill
; CHECK:    str z8, [sp, #17, mul vl] // 16-byte Folded Spill
; CHECK:    .cfi_escape 0x10, 0x48, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x78, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d8 @ cfa - 8 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x49, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x70, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d9 @ cfa - 16 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x4a, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x68, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d10 @ cfa - 24 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x4b, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x60, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d11 @ cfa - 32 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x4c, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x58, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d12 @ cfa - 40 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x4d, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x50, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d13 @ cfa - 48 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x4e, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x48, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d14 @ cfa - 56 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x4f, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x40, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d15 @ cfa - 64 * IncomingVG - 32
; CHECK:    smstart sm
; CHECK:    bl streaming_callee
; CHECK:    smstop sm
;
; UNWINDINFO:      DW_CFA_def_cfa: reg29 +32
; UNWINDINFO:      DW_CFA_offset: reg46 -16
; UNWINDINFO:      DW_CFA_expression: reg72 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -8, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg73 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -16, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg74 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -24, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg75 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -32, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg76 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -40, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg77 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -48, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg78 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -56, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg79 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -64, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
  call void @streaming_callee()
  ret void
}

; [ ] N  ->  S
; [x] S  ->  N
; [ ] N  ->  N
; [ ] S  ->  S
; [ ] LS ->  S
; [ ] SC ->  S
define aarch64_sve_vector_pcs void @streaming_caller_normal_callee() "aarch64_pstate_sm_enabled" {
; CHECK-LABEL: streaming_caller_normal_callee:
; CHECK:    stp x29, x30, [sp, #-32]! // 16-byte Folded Spill
; CHECK:    .cfi_def_cfa_offset 32
; CHECK:    cntd x9
; CHECK:    stp x9, x28, [sp, #16] // 16-byte Folded Spill
; CHECK:    mov x29, sp
; CHECK:    .cfi_def_cfa w29, 32
; CHECK:    .cfi_offset vg, -16
; CHECK:    addvl sp, sp, #-18
; CHECK:    str z15, [sp, #10, mul vl] // 16-byte Folded Spill
; CHECK:    str z14, [sp, #11, mul vl] // 16-byte Folded Spill
; CHECK:    str z13, [sp, #12, mul vl] // 16-byte Folded Spill
; CHECK:    str z12, [sp, #13, mul vl] // 16-byte Folded Spill
; CHECK:    str z11, [sp, #14, mul vl] // 16-byte Folded Spill
; CHECK:    str z10, [sp, #15, mul vl] // 16-byte Folded Spill
; CHECK:    str z9, [sp, #16, mul vl] // 16-byte Folded Spill
; CHECK:    str z8, [sp, #17, mul vl] // 16-byte Folded Spill
; CHECK:    .cfi_escape 0x10, 0x48, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x78, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d8 @ cfa - 8 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x49, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x70, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d9 @ cfa - 16 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x4a, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x68, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d10 @ cfa - 24 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x4b, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x60, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d11 @ cfa - 32 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x4c, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x58, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d12 @ cfa - 40 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x4d, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x50, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d13 @ cfa - 48 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x4e, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x48, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d14 @ cfa - 56 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x4f, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x40, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d15 @ cfa - 64 * IncomingVG - 32
; CHECK:    smstop sm
; CHECK:    bl normal_callee
; CHECK:    smstart sm
;
; UNWINDINFO:      DW_CFA_def_cfa: reg29 +32
; UNWINDINFO:      DW_CFA_offset: reg46 -16
; UNWINDINFO:      DW_CFA_expression: reg72 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -8, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg73 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -16, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg74 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -24, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg75 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -32, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg76 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -40, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg77 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -48, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg78 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -56, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg79 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -64, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
  call void @normal_callee()
  ret void
}

; [ ] N  ->  S
; [ ] S  ->  N
; [x] N  ->  N
; [ ] S  ->  S
; [ ] LS ->  S
; [ ] SC ->  S
define aarch64_sve_vector_pcs void @normal_caller_normal_callee() {
; CHECK-LABEL: normal_caller_normal_callee:
; CHECK:    stp x29, x30, [sp, #-16]! // 16-byte Folded Spill
; CHECK:    addvl sp, sp, #-18
; CHECK:    str z15, [sp, #10, mul vl] // 16-byte Folded Spill
; CHECK:    str z14, [sp, #11, mul vl] // 16-byte Folded Spill
; CHECK:    str z13, [sp, #12, mul vl] // 16-byte Folded Spill
; CHECK:    str z12, [sp, #13, mul vl] // 16-byte Folded Spill
; CHECK:    str z11, [sp, #14, mul vl] // 16-byte Folded Spill
; CHECK:    str z10, [sp, #15, mul vl] // 16-byte Folded Spill
; CHECK:    str z9, [sp, #16, mul vl] // 16-byte Folded Spill
; CHECK:    str z8, [sp, #17, mul vl] // 16-byte Folded Spill
; CHECK:    .cfi_escape 0x0f, 0x0a, 0x8f, 0x10, 0x92, 0x2e, 0x00, 0x11, 0x90, 0x01, 0x1e, 0x22 // sp + 16 + 144 * VG
; CHECK:    .cfi_escape 0x10, 0x48, 0x09, 0x92, 0x2e, 0x00, 0x11, 0x78, 0x1e, 0x22, 0x40, 0x1c // $d8 @ cfa - 8 * VG - 16
; CHECK:    .cfi_escape 0x10, 0x49, 0x09, 0x92, 0x2e, 0x00, 0x11, 0x70, 0x1e, 0x22, 0x40, 0x1c // $d9 @ cfa - 16 * VG - 16
; CHECK:    .cfi_escape 0x10, 0x4a, 0x09, 0x92, 0x2e, 0x00, 0x11, 0x68, 0x1e, 0x22, 0x40, 0x1c // $d10 @ cfa - 24 * VG - 16
; CHECK:    .cfi_escape 0x10, 0x4b, 0x09, 0x92, 0x2e, 0x00, 0x11, 0x60, 0x1e, 0x22, 0x40, 0x1c // $d11 @ cfa - 32 * VG - 16
; CHECK:    .cfi_escape 0x10, 0x4c, 0x09, 0x92, 0x2e, 0x00, 0x11, 0x58, 0x1e, 0x22, 0x40, 0x1c // $d12 @ cfa - 40 * VG - 16
; CHECK:    .cfi_escape 0x10, 0x4d, 0x09, 0x92, 0x2e, 0x00, 0x11, 0x50, 0x1e, 0x22, 0x40, 0x1c // $d13 @ cfa - 48 * VG - 16
; CHECK:    .cfi_escape 0x10, 0x4e, 0x09, 0x92, 0x2e, 0x00, 0x11, 0x48, 0x1e, 0x22, 0x40, 0x1c // $d14 @ cfa - 56 * VG - 16
; CHECK:    .cfi_escape 0x10, 0x4f, 0x09, 0x92, 0x2e, 0x00, 0x11, 0x40, 0x1e, 0x22, 0x40, 0x1c // $d15 @ cfa - 64 * VG - 16
; CHECK:    bl normal_callee
;
; UNWINDINFO:      DW_CFA_def_cfa_expression: DW_OP_breg31 +16, DW_OP_bregx 0x2e +0, DW_OP_consts +144, DW_OP_mul, DW_OP_plus
; UNWINDINFO:      DW_CFA_expression: reg72 DW_OP_bregx 0x2e +0, DW_OP_consts -8, DW_OP_mul, DW_OP_plus, DW_OP_lit16, DW_OP_minus
; UNWINDINFO-NEXT: DW_CFA_expression: reg73 DW_OP_bregx 0x2e +0, DW_OP_consts -16, DW_OP_mul, DW_OP_plus, DW_OP_lit16, DW_OP_minus
; UNWINDINFO-NEXT: DW_CFA_expression: reg74 DW_OP_bregx 0x2e +0, DW_OP_consts -24, DW_OP_mul, DW_OP_plus, DW_OP_lit16, DW_OP_minus
; UNWINDINFO-NEXT: DW_CFA_expression: reg75 DW_OP_bregx 0x2e +0, DW_OP_consts -32, DW_OP_mul, DW_OP_plus, DW_OP_lit16, DW_OP_minus
; UNWINDINFO-NEXT: DW_CFA_expression: reg76 DW_OP_bregx 0x2e +0, DW_OP_consts -40, DW_OP_mul, DW_OP_plus, DW_OP_lit16, DW_OP_minus
; UNWINDINFO-NEXT: DW_CFA_expression: reg77 DW_OP_bregx 0x2e +0, DW_OP_consts -48, DW_OP_mul, DW_OP_plus, DW_OP_lit16, DW_OP_minus
; UNWINDINFO-NEXT: DW_CFA_expression: reg78 DW_OP_bregx 0x2e +0, DW_OP_consts -56, DW_OP_mul, DW_OP_plus, DW_OP_lit16, DW_OP_minus
; UNWINDINFO-NEXT: DW_CFA_expression: reg79 DW_OP_bregx 0x2e +0, DW_OP_consts -64, DW_OP_mul, DW_OP_plus, DW_OP_lit16, DW_OP_minus
  call void @normal_callee()
  ret void
}

; [ ] N  ->  S
; [ ] S  ->  N
; [ ] N  ->  N
; [x] S  ->  S
; [ ] LS ->  S
; [ ] SC ->  S
define aarch64_sve_vector_pcs void @streaming_caller_streaming_callee() "aarch64_pstate_sm_enabled" {
; CHECK-LABEL: streaming_caller_streaming_callee:
; CHECK:    stp x29, x30, [sp, #-16]! // 16-byte Folded Spill
; CHECK:    addvl sp, sp, #-18
; CHECK:    str z15, [sp, #10, mul vl] // 16-byte Folded Spill
; CHECK:    str z14, [sp, #11, mul vl] // 16-byte Folded Spill
; CHECK:    str z13, [sp, #12, mul vl] // 16-byte Folded Spill
; CHECK:    str z12, [sp, #13, mul vl] // 16-byte Folded Spill
; CHECK:    str z11, [sp, #14, mul vl] // 16-byte Folded Spill
; CHECK:    str z10, [sp, #15, mul vl] // 16-byte Folded Spill
; CHECK:    str z9, [sp, #16, mul vl] // 16-byte Folded Spill
; CHECK:    str z8, [sp, #17, mul vl] // 16-byte Folded Spill
; CHECK:    .cfi_escape 0x0f, 0x0a, 0x8f, 0x10, 0x92, 0x2e, 0x00, 0x11, 0x90, 0x01, 0x1e, 0x22 // sp + 16 + 144 * VG
; CHECK:    .cfi_escape 0x10, 0x48, 0x09, 0x92, 0x2e, 0x00, 0x11, 0x78, 0x1e, 0x22, 0x40, 0x1c // $d8 @ cfa - 8 * VG - 16
; CHECK:    .cfi_escape 0x10, 0x49, 0x09, 0x92, 0x2e, 0x00, 0x11, 0x70, 0x1e, 0x22, 0x40, 0x1c // $d9 @ cfa - 16 * VG - 16
; CHECK:    .cfi_escape 0x10, 0x4a, 0x09, 0x92, 0x2e, 0x00, 0x11, 0x68, 0x1e, 0x22, 0x40, 0x1c // $d10 @ cfa - 24 * VG - 16
; CHECK:    .cfi_escape 0x10, 0x4b, 0x09, 0x92, 0x2e, 0x00, 0x11, 0x60, 0x1e, 0x22, 0x40, 0x1c // $d11 @ cfa - 32 * VG - 16
; CHECK:    .cfi_escape 0x10, 0x4c, 0x09, 0x92, 0x2e, 0x00, 0x11, 0x58, 0x1e, 0x22, 0x40, 0x1c // $d12 @ cfa - 40 * VG - 16
; CHECK:    .cfi_escape 0x10, 0x4d, 0x09, 0x92, 0x2e, 0x00, 0x11, 0x50, 0x1e, 0x22, 0x40, 0x1c // $d13 @ cfa - 48 * VG - 16
; CHECK:    .cfi_escape 0x10, 0x4e, 0x09, 0x92, 0x2e, 0x00, 0x11, 0x48, 0x1e, 0x22, 0x40, 0x1c // $d14 @ cfa - 56 * VG - 16
; CHECK:    .cfi_escape 0x10, 0x4f, 0x09, 0x92, 0x2e, 0x00, 0x11, 0x40, 0x1e, 0x22, 0x40, 0x1c // $d15 @ cfa - 64 * VG - 16
; CHECK:    bl streaming_callee
;
; UNWINDINFO:      DW_CFA_def_cfa_expression: DW_OP_breg31 +16, DW_OP_bregx 0x2e +0, DW_OP_consts +144, DW_OP_mul, DW_OP_plus
; UNWINDINFO:      DW_CFA_expression: reg72 DW_OP_bregx 0x2e +0, DW_OP_consts -8, DW_OP_mul, DW_OP_plus, DW_OP_lit16, DW_OP_minus
; UNWINDINFO-NEXT: DW_CFA_expression: reg73 DW_OP_bregx 0x2e +0, DW_OP_consts -16, DW_OP_mul, DW_OP_plus, DW_OP_lit16, DW_OP_minus
; UNWINDINFO-NEXT: DW_CFA_expression: reg74 DW_OP_bregx 0x2e +0, DW_OP_consts -24, DW_OP_mul, DW_OP_plus, DW_OP_lit16, DW_OP_minus
; UNWINDINFO-NEXT: DW_CFA_expression: reg75 DW_OP_bregx 0x2e +0, DW_OP_consts -32, DW_OP_mul, DW_OP_plus, DW_OP_lit16, DW_OP_minus
; UNWINDINFO-NEXT: DW_CFA_expression: reg76 DW_OP_bregx 0x2e +0, DW_OP_consts -40, DW_OP_mul, DW_OP_plus, DW_OP_lit16, DW_OP_minus
; UNWINDINFO-NEXT: DW_CFA_expression: reg77 DW_OP_bregx 0x2e +0, DW_OP_consts -48, DW_OP_mul, DW_OP_plus, DW_OP_lit16, DW_OP_minus
; UNWINDINFO-NEXT: DW_CFA_expression: reg78 DW_OP_bregx 0x2e +0, DW_OP_consts -56, DW_OP_mul, DW_OP_plus, DW_OP_lit16, DW_OP_minus
; UNWINDINFO-NEXT: DW_CFA_expression: reg79 DW_OP_bregx 0x2e +0, DW_OP_consts -64, DW_OP_mul, DW_OP_plus, DW_OP_lit16, DW_OP_minus
  call void @streaming_callee()
  ret void
}

; [ ] N  ->  S
; [ ] S  ->  N
; [ ] N  ->  N
; [ ] S  ->  S
; [x] LS ->  S
; [ ] SC ->  S
define aarch64_sve_vector_pcs void @locally_streaming() "aarch64_pstate_sm_body" {
; CHECK-LABEL: locally_streaming:
; CHECK:    stp x29, x30, [sp, #-32]! // 16-byte Folded Spill
; CHECK:    .cfi_def_cfa_offset 32
; CHECK:    cntd x9
; CHECK:    stp x9, x28, [sp, #16] // 16-byte Folded Spill
; CHECK:    mov x29, sp
; CHECK:    .cfi_def_cfa w29, 32
; CHECK:    .cfi_offset vg, -16
; CHECK:    addsvl sp, sp, #-18
; CHECK:    str z15, [sp, #10, mul vl] // 16-byte Folded Spill
; CHECK:    str z14, [sp, #11, mul vl] // 16-byte Folded Spill
; CHECK:    str z13, [sp, #12, mul vl] // 16-byte Folded Spill
; CHECK:    str z12, [sp, #13, mul vl] // 16-byte Folded Spill
; CHECK:    str z11, [sp, #14, mul vl] // 16-byte Folded Spill
; CHECK:    str z10, [sp, #15, mul vl] // 16-byte Folded Spill
; CHECK:    str z9, [sp, #16, mul vl] // 16-byte Folded Spill
; CHECK:    str z8, [sp, #17, mul vl] // 16-byte Folded Spill
; CHECK:    .cfi_escape 0x10, 0x48, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x78, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d8 @ cfa - 8 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x49, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x70, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d9 @ cfa - 16 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x4a, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x68, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d10 @ cfa - 24 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x4b, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x60, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d11 @ cfa - 32 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x4c, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x58, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d12 @ cfa - 40 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x4d, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x50, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d13 @ cfa - 48 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x4e, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x48, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d14 @ cfa - 56 * IncomingVG - 32
; CHECK:    .cfi_escape 0x10, 0x4f, 0x0b, 0x12, 0x40, 0x1c, 0x06, 0x11, 0x40, 0x1e, 0x22, 0x11, 0x60, 0x22 // $d15 @ cfa - 64 * IncomingVG - 32
; CHECK:    smstart sm
; CHECK:    bl streaming_callee
; CHECK:    smstop sm
;
; UNWINDINFO:      DW_CFA_def_cfa: reg29 +32
; UNWINDINFO:      DW_CFA_offset: reg46 -16
; UNWINDINFO:      DW_CFA_expression: reg72 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -8, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg73 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -16, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg74 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -24, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg75 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -32, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg76 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -40, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg77 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -48, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg78 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -56, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg79 DW_OP_dup, DW_OP_lit16, DW_OP_minus, DW_OP_deref, DW_OP_consts -64, DW_OP_mul, DW_OP_plus, DW_OP_consts -32, DW_OP_plus
  call void @streaming_callee()
  ret void
}

; [ ] N  ->  S
; [ ] S  ->  N
; [ ] N  ->  N
; [ ] S  ->  S
; [ ] LS ->  S
; [x] SC ->  S
define aarch64_sve_vector_pcs void @streaming_compatible_caller_conditional_mode_switch() "aarch64_pstate_sm_compatible" {
; CHECK-LABEL: streaming_compatible_caller_conditional_mode_switch:
; CHECK:    stp x29, x30, [sp, #-48]! // 16-byte Folded Spill
; CHECK:    .cfi_def_cfa_offset 48
; CHECK:    cntd x9
; CHECK:    stp x28, x19, [sp, #32] // 16-byte Folded Spill
; CHECK:    str x9, [sp, #16] // 8-byte Folded Spill
; CHECK:    mov x29, sp
; CHECK:    .cfi_def_cfa w29, 48
; CHECK:    .cfi_offset vg, -32
; CHECK:    addvl sp, sp, #-18
; CHECK:    str z15, [sp, #10, mul vl] // 16-byte Folded Spill
; CHECK:    str z14, [sp, #11, mul vl] // 16-byte Folded Spill
; CHECK:    str z13, [sp, #12, mul vl] // 16-byte Folded Spill
; CHECK:    str z12, [sp, #13, mul vl] // 16-byte Folded Spill
; CHECK:    str z11, [sp, #14, mul vl] // 16-byte Folded Spill
; CHECK:    str z10, [sp, #15, mul vl] // 16-byte Folded Spill
; CHECK:    str z9, [sp, #16, mul vl] // 16-byte Folded Spill
; CHECK:    str z8, [sp, #17, mul vl] // 16-byte Folded Spill
; CHECK:    .cfi_escape 0x10, 0x48, 0x0c, 0x12, 0x11, 0x60, 0x22, 0x06, 0x11, 0x78, 0x1e, 0x22, 0x11, 0x50, 0x22 // $d8 @ cfa - 8 * IncomingVG - 48
; CHECK:    .cfi_escape 0x10, 0x49, 0x0c, 0x12, 0x11, 0x60, 0x22, 0x06, 0x11, 0x70, 0x1e, 0x22, 0x11, 0x50, 0x22 // $d9 @ cfa - 16 * IncomingVG - 48
; CHECK:    .cfi_escape 0x10, 0x4a, 0x0c, 0x12, 0x11, 0x60, 0x22, 0x06, 0x11, 0x68, 0x1e, 0x22, 0x11, 0x50, 0x22 // $d10 @ cfa - 24 * IncomingVG - 48
; CHECK:    .cfi_escape 0x10, 0x4b, 0x0c, 0x12, 0x11, 0x60, 0x22, 0x06, 0x11, 0x60, 0x1e, 0x22, 0x11, 0x50, 0x22 // $d11 @ cfa - 32 * IncomingVG - 48
; CHECK:    .cfi_escape 0x10, 0x4c, 0x0c, 0x12, 0x11, 0x60, 0x22, 0x06, 0x11, 0x58, 0x1e, 0x22, 0x11, 0x50, 0x22 // $d12 @ cfa - 40 * IncomingVG - 48
; CHECK:    .cfi_escape 0x10, 0x4d, 0x0c, 0x12, 0x11, 0x60, 0x22, 0x06, 0x11, 0x50, 0x1e, 0x22, 0x11, 0x50, 0x22 // $d13 @ cfa - 48 * IncomingVG - 48
; CHECK:    .cfi_escape 0x10, 0x4e, 0x0c, 0x12, 0x11, 0x60, 0x22, 0x06, 0x11, 0x48, 0x1e, 0x22, 0x11, 0x50, 0x22 // $d14 @ cfa - 56 * IncomingVG - 48
; CHECK:    .cfi_escape 0x10, 0x4f, 0x0c, 0x12, 0x11, 0x60, 0x22, 0x06, 0x11, 0x40, 0x1e, 0x22, 0x11, 0x50, 0x22 // $d15 @ cfa - 64 * IncomingVG - 48
; CHECK:    mrs x19, SVCR
; CHECK:    tbnz w19, #0, .LBB5_2
; CHECK:    smstart sm
; CHECK:  .LBB5_2:
; CHECK:    bl streaming_callee
; CHECK:    tbnz w19, #0, .LBB5_4
; CHECK:    smstop sm
; CHECK:  .LBB5_4:
;
; UNWINDINFO:      DW_CFA_def_cfa: reg29 +48
; UNWINDINFO:      DW_CFA_offset: reg46 -32
; UNWINDINFO:      DW_CFA_expression: reg72 DW_OP_dup, DW_OP_consts -32, DW_OP_plus, DW_OP_deref, DW_OP_consts -8, DW_OP_mul, DW_OP_plus, DW_OP_consts -48, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg73 DW_OP_dup, DW_OP_consts -32, DW_OP_plus, DW_OP_deref, DW_OP_consts -16, DW_OP_mul, DW_OP_plus, DW_OP_consts -48, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg74 DW_OP_dup, DW_OP_consts -32, DW_OP_plus, DW_OP_deref, DW_OP_consts -24, DW_OP_mul, DW_OP_plus, DW_OP_consts -48, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg75 DW_OP_dup, DW_OP_consts -32, DW_OP_plus, DW_OP_deref, DW_OP_consts -32, DW_OP_mul, DW_OP_plus, DW_OP_consts -48, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg76 DW_OP_dup, DW_OP_consts -32, DW_OP_plus, DW_OP_deref, DW_OP_consts -40, DW_OP_mul, DW_OP_plus, DW_OP_consts -48, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg77 DW_OP_dup, DW_OP_consts -32, DW_OP_plus, DW_OP_deref, DW_OP_consts -48, DW_OP_mul, DW_OP_plus, DW_OP_consts -48, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg78 DW_OP_dup, DW_OP_consts -32, DW_OP_plus, DW_OP_deref, DW_OP_consts -56, DW_OP_mul, DW_OP_plus, DW_OP_consts -48, DW_OP_plus
; UNWINDINFO-NEXT: DW_CFA_expression: reg79 DW_OP_dup, DW_OP_consts -32, DW_OP_plus, DW_OP_deref, DW_OP_consts -64, DW_OP_mul, DW_OP_plus, DW_OP_consts -48, DW_OP_plus
  call void @streaming_callee()
  ret void
}
