; RUN: not llvm-mc -triple arm64-apple-darwin -show-encoding < %s 2> %t | FileCheck %s
; RUN: not llvm-mc -triple arm64-apple-darwin -mattr=+v8.3a -show-encoding < %s 2> %t | FileCheck %s --check-prefix=CHECK-V83
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

foo:

;-----------------------------------------------------------------------------
; Simple encodings (instructions w/ no operands)
;-----------------------------------------------------------------------------

  nop
  sev
  sevl
  wfe
  wfi
  yield

; CHECK: nop                             ; encoding: [0x1f,0x20,0x03,0xd5]
; CHECK: sev                             ; encoding: [0x9f,0x20,0x03,0xd5]
; CHECK: sevl                            ; encoding: [0xbf,0x20,0x03,0xd5]
; CHECK: wfe                             ; encoding: [0x5f,0x20,0x03,0xd5]
; CHECK: wfi                             ; encoding: [0x7f,0x20,0x03,0xd5]
; CHECK: yield                           ; encoding: [0x3f,0x20,0x03,0xd5]

;-----------------------------------------------------------------------------
; Single-immediate operand instructions
;-----------------------------------------------------------------------------

  clrex #10
; CHECK: clrex #10  ; encoding: [0x5f,0x3a,0x03,0xd5]
  isb #15
  isb sy
; CHECK: isb     ; encoding: [0xdf,0x3f,0x03,0xd5]
; CHECK: isb     ; encoding: [0xdf,0x3f,0x03,0xd5]
  dmb #3
  dmb osh
; CHECK: dmb osh    ; encoding: [0xbf,0x33,0x03,0xd5]
; CHECK: dmb osh    ; encoding: [0xbf,0x33,0x03,0xd5]
  dsb #7
  dsb nsh
; CHECK: dsb nsh    ; encoding: [0x9f,0x37,0x03,0xd5]
; CHECK: dsb nsh    ; encoding: [0x9f,0x37,0x03,0xd5]

;-----------------------------------------------------------------------------
; Generic system instructions
;-----------------------------------------------------------------------------
  sys #2, c0, c5, #7
; CHECK: encoding: [0xff,0x05,0x0a,0xd5]
  sys #7, C6, c10, #7, x7
; CHECK: encoding: [0xe7,0x6a,0x0f,0xd5]
  sysl  x20, #6, c3, C15, #7
; CHECK: encoding: [0xf4,0x3f,0x2e,0xd5]

; Check for error on invalid 'C' operand value.
  sys #2, c16, c5, #7
; CHECK-ERRORS: error: Expected cN operand where 0 <= N <= 15

;-----------------------------------------------------------------------------
; MSR/MRS instructions
;-----------------------------------------------------------------------------
  msr ACTLR_EL1, x3
  msr ACTLR_EL2, x3
  msr ACTLR_EL3, x3
  msr AFSR0_EL1, x3
  msr AFSR0_EL2, x3
  msr AFSR0_EL3, x3
  msr AFSR1_EL1, x3
  msr AFSR1_EL2, x3
  msr AFSR1_EL3, x3
  msr AMAIR_EL1, x3
  msr AMAIR_EL2, x3
  msr AMAIR_EL3, x3
  msr CNTFRQ_EL0, x3
  msr CNTHCTL_EL2, x3
  msr CNTHP_CTL_EL2, x3
  msr CNTHP_CVAL_EL2, x3
  msr CNTHP_TVAL_EL2, x3
  msr CNTKCTL_EL1, x3
  msr CNTP_CTL_EL0, x3
  msr CNTP_CVAL_EL0, x3
  msr CNTP_TVAL_EL0, x3
  msr CNTVOFF_EL2, x3
  msr CNTV_CTL_EL0, x3
  msr CNTV_CVAL_EL0, x3
  msr CNTV_TVAL_EL0, x3
  msr CONTEXTIDR_EL1, x3
  msr CPACR_EL1, x3
  msr CPTR_EL2, x3
  msr CPTR_EL3, x3
  msr CSSELR_EL1, x3
  msr DACR32_EL2, x3
  msr ESR_EL1, x3
  msr ESR_EL2, x3
  msr ESR_EL3, x3
  msr FAR_EL1, x3
  msr FAR_EL2, x3
  msr FAR_EL3, x3
  msr FPEXC32_EL2, x3
  msr HACR_EL2, x3
  msr HCR_EL2, x3
  msr HPFAR_EL2, x3
  msr HSTR_EL2, x3
  msr IFSR32_EL2, x3
  msr MAIR_EL1, x3
  msr MAIR_EL2, x3
  msr MAIR_EL3, x3
  msr MDCR_EL2, x3
  msr MDCR_EL3, x3
  msr PAR_EL1, x3
  msr SCR_EL3, x3
  msr SCTLR_EL1, x3
  msr SCTLR_EL2, x3
  msr SCTLR_EL3, x3
  msr SDER32_EL3, x3
  msr TCR_EL1, x3
  msr TCR_EL2, x3
  msr TCR_EL3, x3
  msr TEECR32_EL1, x3
  msr TEEHBR32_EL1, x3
  msr TPIDRRO_EL0, x3
  msr TPIDR_EL0, x3
  msr TPIDR_EL1, x3
  msr TPIDR_EL2, x3
  msr TPIDR_EL3, x3
  msr TTBR0_EL1, x3
  msr TTBR0_EL2, x3
  msr TTBR0_EL3, x3
  msr TTBR1_EL1, x3
  msr VBAR_EL1, x3
  msr VBAR_EL2, x3
  msr VBAR_EL3, x3
  msr VMPIDR_EL2, x3
  msr VPIDR_EL2, x3
  msr VTCR_EL2, x3
  msr VTTBR_EL2, x3
  msr SPSel, x3
  msr S3_2_C11_C6_4, x1
  msr  S0_0_C0_C0_0, x0
  msr  S1_2_C3_C4_5, x2
; CHECK: msr ACTLR_EL1, x3              ; encoding: [0x23,0x10,0x18,0xd5]
; CHECK: msr ACTLR_EL2, x3              ; encoding: [0x23,0x10,0x1c,0xd5]
; CHECK: msr ACTLR_EL3, x3              ; encoding: [0x23,0x10,0x1e,0xd5]
; CHECK: msr AFSR0_EL1, x3              ; encoding: [0x03,0x51,0x18,0xd5]
; CHECK: msr AFSR0_EL2, x3              ; encoding: [0x03,0x51,0x1c,0xd5]
; CHECK: msr AFSR0_EL3, x3              ; encoding: [0x03,0x51,0x1e,0xd5]
; CHECK: msr AFSR1_EL1, x3              ; encoding: [0x23,0x51,0x18,0xd5]
; CHECK: msr AFSR1_EL2, x3              ; encoding: [0x23,0x51,0x1c,0xd5]
; CHECK: msr AFSR1_EL3, x3              ; encoding: [0x23,0x51,0x1e,0xd5]
; CHECK: msr AMAIR_EL1, x3              ; encoding: [0x03,0xa3,0x18,0xd5]
; CHECK: msr AMAIR_EL2, x3              ; encoding: [0x03,0xa3,0x1c,0xd5]
; CHECK: msr AMAIR_EL3, x3              ; encoding: [0x03,0xa3,0x1e,0xd5]
; CHECK: msr CNTFRQ_EL0, x3             ; encoding: [0x03,0xe0,0x1b,0xd5]
; CHECK: msr CNTHCTL_EL2, x3            ; encoding: [0x03,0xe1,0x1c,0xd5]
; CHECK: msr CNTHP_CTL_EL2, x3          ; encoding: [0x23,0xe2,0x1c,0xd5]
; CHECK: msr CNTHP_CVAL_EL2, x3         ; encoding: [0x43,0xe2,0x1c,0xd5]
; CHECK: msr CNTHP_TVAL_EL2, x3         ; encoding: [0x03,0xe2,0x1c,0xd5]
; CHECK: msr CNTKCTL_EL1, x3            ; encoding: [0x03,0xe1,0x18,0xd5]
; CHECK: msr CNTP_CTL_EL0, x3           ; encoding: [0x23,0xe2,0x1b,0xd5]
; CHECK: msr CNTP_CVAL_EL0, x3          ; encoding: [0x43,0xe2,0x1b,0xd5]
; CHECK: msr CNTP_TVAL_EL0, x3          ; encoding: [0x03,0xe2,0x1b,0xd5]
; CHECK: msr CNTVOFF_EL2, x3            ; encoding: [0x63,0xe0,0x1c,0xd5]
; CHECK: msr CNTV_CTL_EL0, x3           ; encoding: [0x23,0xe3,0x1b,0xd5]
; CHECK: msr CNTV_CVAL_EL0, x3          ; encoding: [0x43,0xe3,0x1b,0xd5]
; CHECK: msr CNTV_TVAL_EL0, x3          ; encoding: [0x03,0xe3,0x1b,0xd5]
; CHECK: msr CONTEXTIDR_EL1, x3         ; encoding: [0x23,0xd0,0x18,0xd5]
; CHECK: msr CPACR_EL1, x3              ; encoding: [0x43,0x10,0x18,0xd5]
; CHECK: msr CPTR_EL2, x3               ; encoding: [0x43,0x11,0x1c,0xd5]
; CHECK: msr CPTR_EL3, x3               ; encoding: [0x43,0x11,0x1e,0xd5]
; CHECK: msr CSSELR_EL1, x3             ; encoding: [0x03,0x00,0x1a,0xd5]
; CHECK: msr DACR32_EL2, x3             ; encoding: [0x03,0x30,0x1c,0xd5]
; CHECK: msr ESR_EL1, x3                ; encoding: [0x03,0x52,0x18,0xd5]
; CHECK: msr ESR_EL2, x3                ; encoding: [0x03,0x52,0x1c,0xd5]
; CHECK: msr ESR_EL3, x3                ; encoding: [0x03,0x52,0x1e,0xd5]
; CHECK: msr FAR_EL1, x3                ; encoding: [0x03,0x60,0x18,0xd5]
; CHECK: msr FAR_EL2, x3                ; encoding: [0x03,0x60,0x1c,0xd5]
; CHECK: msr FAR_EL3, x3                ; encoding: [0x03,0x60,0x1e,0xd5]
; CHECK: msr FPEXC32_EL2, x3            ; encoding: [0x03,0x53,0x1c,0xd5]
; CHECK: msr HACR_EL2, x3               ; encoding: [0xe3,0x11,0x1c,0xd5]
; CHECK: msr HCR_EL2, x3                ; encoding: [0x03,0x11,0x1c,0xd5]
; CHECK: msr HPFAR_EL2, x3              ; encoding: [0x83,0x60,0x1c,0xd5]
; CHECK: msr HSTR_EL2, x3               ; encoding: [0x63,0x11,0x1c,0xd5]
; CHECK: msr IFSR32_EL2, x3             ; encoding: [0x23,0x50,0x1c,0xd5]
; CHECK: msr MAIR_EL1, x3               ; encoding: [0x03,0xa2,0x18,0xd5]
; CHECK: msr MAIR_EL2, x3               ; encoding: [0x03,0xa2,0x1c,0xd5]
; CHECK: msr MAIR_EL3, x3               ; encoding: [0x03,0xa2,0x1e,0xd5]
; CHECK: msr MDCR_EL2, x3               ; encoding: [0x23,0x11,0x1c,0xd5]
; CHECK: msr MDCR_EL3, x3               ; encoding: [0x23,0x13,0x1e,0xd5]
; CHECK: msr PAR_EL1, x3                ; encoding: [0x03,0x74,0x18,0xd5]
; CHECK: msr SCR_EL3, x3                ; encoding: [0x03,0x11,0x1e,0xd5]
; CHECK: msr SCTLR_EL1, x3              ; encoding: [0x03,0x10,0x18,0xd5]
; CHECK: msr SCTLR_EL2, x3              ; encoding: [0x03,0x10,0x1c,0xd5]
; CHECK: msr SCTLR_EL3, x3              ; encoding: [0x03,0x10,0x1e,0xd5]
; CHECK: msr SDER32_EL3, x3             ; encoding: [0x23,0x11,0x1e,0xd5]
; CHECK: msr TCR_EL1, x3                ; encoding: [0x43,0x20,0x18,0xd5]
; CHECK: msr TCR_EL2, x3                ; encoding: [0x43,0x20,0x1c,0xd5]
; CHECK: msr TCR_EL3, x3                ; encoding: [0x43,0x20,0x1e,0xd5]
; CHECK: msr TEECR32_EL1, x3            ; encoding: [0x03,0x00,0x12,0xd5]
; CHECK: msr TEEHBR32_EL1, x3           ; encoding: [0x03,0x10,0x12,0xd5]
; CHECK: msr TPIDRRO_EL0, x3            ; encoding: [0x63,0xd0,0x1b,0xd5]
; CHECK: msr TPIDR_EL0, x3              ; encoding: [0x43,0xd0,0x1b,0xd5]
; CHECK: msr TPIDR_EL1, x3              ; encoding: [0x83,0xd0,0x18,0xd5]
; CHECK: msr TPIDR_EL2, x3              ; encoding: [0x43,0xd0,0x1c,0xd5]
; CHECK: msr TPIDR_EL3, x3              ; encoding: [0x43,0xd0,0x1e,0xd5]
; CHECK: msr TTBR0_EL1, x3              ; encoding: [0x03,0x20,0x18,0xd5]
; CHECK: msr TTBR0_EL2, x3              ; encoding: [0x03,0x20,0x1c,0xd5]
; CHECK: msr TTBR0_EL3, x3              ; encoding: [0x03,0x20,0x1e,0xd5]
; CHECK: msr TTBR1_EL1, x3              ; encoding: [0x23,0x20,0x18,0xd5]
; CHECK: msr VBAR_EL1, x3               ; encoding: [0x03,0xc0,0x18,0xd5]
; CHECK: msr VBAR_EL2, x3               ; encoding: [0x03,0xc0,0x1c,0xd5]
; CHECK: msr VBAR_EL3, x3               ; encoding: [0x03,0xc0,0x1e,0xd5]
; CHECK: msr VMPIDR_EL2, x3             ; encoding: [0xa3,0x00,0x1c,0xd5]
; CHECK: msr VPIDR_EL2, x3              ; encoding: [0x03,0x00,0x1c,0xd5]
; CHECK: msr VTCR_EL2, x3               ; encoding: [0x43,0x21,0x1c,0xd5]
; CHECK: msr VTTBR_EL2, x3              ; encoding: [0x03,0x21,0x1c,0xd5]
; CHECK: msr  SPSel, x3                 ; encoding: [0x03,0x42,0x18,0xd5]
; CHECK: msr  S3_2_C11_C6_4, x1         ; encoding: [0x81,0xb6,0x1a,0xd5]
; CHECK: msr  S0_0_C0_C0_0, x0          ; encoding: [0x00,0x00,0x00,0xd5]
; CHECK: msr  S1_2_C3_C4_5, x2          ; encoding: [0xa2,0x34,0x0a,0xd5]

// Readonly system registers: writing to them gives an error
  msr CURRENTEL, x3
; CHECK-ERRORS: :[[@LINE-1]]:7: error: expected writable system register or pstate

  mrs x3, ACTLR_EL1
  mrs x3, ACTLR_EL2
  mrs x3, ACTLR_EL3
  mrs x3, AFSR0_EL1
  mrs x3, AFSR0_EL2
  mrs x3, AFSR0_EL3
  mrs x3, AIDR_EL1
  mrs x3, AFSR1_EL1
  mrs x3, AFSR1_EL2
  mrs x3, AFSR1_EL3
  mrs x3, AMAIR_EL1
  mrs x3, AMAIR_EL2
  mrs x3, AMAIR_EL3
  mrs x3, CCSIDR_EL1
  mrs x3, CLIDR_EL1
  mrs x3, CCSIDR2_EL1
  mrs x3, CNTFRQ_EL0
  mrs x3, CNTHCTL_EL2
  mrs x3, CNTHP_CTL_EL2
  mrs x3, CNTHP_CVAL_EL2
  mrs x3, CNTHP_TVAL_EL2
  mrs x3, CNTKCTL_EL1
  mrs x3, CNTPCT_EL0
  mrs x3, CNTP_CTL_EL0
  mrs x3, CNTP_CVAL_EL0
  mrs x3, CNTP_TVAL_EL0
  mrs x3, CNTVCT_EL0
  mrs x3, CNTVOFF_EL2
  mrs x3, CNTV_CTL_EL0
  mrs x3, CNTV_CVAL_EL0
  mrs x3, CNTV_TVAL_EL0
  mrs x3, CONTEXTIDR_EL1
  mrs x3, CPACR_EL1
  mrs x3, CPTR_EL2
  mrs x3, CPTR_EL3
  mrs x3, CSSELR_EL1
  mrs x3, CTR_EL0
  mrs x3, CURRENTEL
  mrs x3, DACR32_EL2
  mrs x3, DCZID_EL0
  mrs x3, REVIDR_EL1
  mrs x3, ESR_EL1
  mrs x3, ESR_EL2
  mrs x3, ESR_EL3
  mrs x3, FAR_EL1
  mrs x3, FAR_EL2
  mrs x3, FAR_EL3
  mrs x3, FPEXC32_EL2
  mrs x3, HACR_EL2
  mrs x3, HCR_EL2
  mrs x3, HPFAR_EL2
  mrs x3, HSTR_EL2
  mrs x3, ID_AA64DFR0_EL1
  mrs x3, ID_AA64DFR1_EL1
  mrs x3, ID_AA64ISAR0_EL1
  mrs x3, ID_AA64ISAR1_EL1
  mrs x3, ID_AA64ISAR2_EL1
  mrs x3, ID_AA64MMFR0_EL1
  mrs x3, ID_AA64MMFR1_EL1
  mrs x3, ID_AA64PFR0_EL1
  mrs x3, ID_AA64PFR1_EL1
  mrs x3, IFSR32_EL2
  mrs x3, ISR_EL1
  mrs x3, MAIR_EL1
  mrs x3, MAIR_EL2
  mrs x3, MAIR_EL3
  mrs x3, MDCR_EL2
  mrs x3, MDCR_EL3
  mrs x3, MIDR_EL1
  mrs x3, MPIDR_EL1
  mrs x3, MVFR0_EL1
  mrs x3, MVFR1_EL1
  mrs x3, PAR_EL1
  mrs x3, RVBAR_EL1
  mrs x3, RVBAR_EL2
  mrs x3, RVBAR_EL3
  mrs x3, SCR_EL3
  mrs x3, SCTLR_EL1
  mrs x3, SCTLR_EL2
  mrs x3, SCTLR_EL3
  mrs x3, SDER32_EL3
  mrs x3, TCR_EL1
  mrs x3, TCR_EL2
  mrs x3, TCR_EL3
  mrs x3, TEECR32_EL1
  mrs x3, TEEHBR32_EL1
  mrs x3, TPIDRRO_EL0
  mrs x3, TPIDR_EL0
  mrs x3, TPIDR_EL1
  mrs x3, TPIDR_EL2
  mrs x3, TPIDR_EL3
  mrs x3, TTBR0_EL1
  mrs x3, TTBR0_EL2
  mrs x3, TTBR0_EL3
  mrs x3, TTBR1_EL1
  mrs x3, VBAR_EL1
  mrs x3, VBAR_EL2
  mrs x3, VBAR_EL3
  mrs x3, VMPIDR_EL2
  mrs x3, VPIDR_EL2
  mrs x3, VTCR_EL2
  mrs x3, VTTBR_EL2

  mrs x3, MDCCSR_EL0
  mrs x3, MDCCINT_EL1
  mrs x3, DBGDTR_EL0
  mrs x3, DBGDTRRX_EL0
  mrs x3, DBGVCR32_EL2
  mrs x3, OSDTRRX_EL1
  mrs x3, MDSCR_EL1
  mrs x3, OSDTRTX_EL1
  mrs x3, OSECCR_EL1
  mrs x3, DBGBVR0_EL1
  mrs x3, DBGBVR1_EL1
  mrs x3, DBGBVR2_EL1
  mrs x3, DBGBVR3_EL1
  mrs x3, DBGBVR4_EL1
  mrs x3, DBGBVR5_EL1
  mrs x3, DBGBVR6_EL1
  mrs x3, DBGBVR7_EL1
  mrs x3, DBGBVR8_EL1
  mrs x3, DBGBVR9_EL1
  mrs x3, DBGBVR10_EL1
  mrs x3, DBGBVR11_EL1
  mrs x3, DBGBVR12_EL1
  mrs x3, DBGBVR13_EL1
  mrs x3, DBGBVR14_EL1
  mrs x3, DBGBVR15_EL1
  mrs x3, DBGBCR0_EL1
  mrs x3, DBGBCR1_EL1
  mrs x3, DBGBCR2_EL1
  mrs x3, DBGBCR3_EL1
  mrs x3, DBGBCR4_EL1
  mrs x3, DBGBCR5_EL1
  mrs x3, DBGBCR6_EL1
  mrs x3, DBGBCR7_EL1
  mrs x3, DBGBCR8_EL1
  mrs x3, DBGBCR9_EL1
  mrs x3, DBGBCR10_EL1
  mrs x3, DBGBCR11_EL1
  mrs x3, DBGBCR12_EL1
  mrs x3, DBGBCR13_EL1
  mrs x3, DBGBCR14_EL1
  mrs x3, DBGBCR15_EL1
  mrs x3, DBGWVR0_EL1
  mrs x3, DBGWVR1_EL1
  mrs x3, DBGWVR2_EL1
  mrs x3, DBGWVR3_EL1
  mrs x3, DBGWVR4_EL1
  mrs x3, DBGWVR5_EL1
  mrs x3, DBGWVR6_EL1
  mrs x3, DBGWVR7_EL1
  mrs x3, DBGWVR8_EL1
  mrs x3, DBGWVR9_EL1
  mrs x3, DBGWVR10_EL1
  mrs x3, DBGWVR11_EL1
  mrs x3, DBGWVR12_EL1
  mrs x3, DBGWVR13_EL1
  mrs x3, DBGWVR14_EL1
  mrs x3, DBGWVR15_EL1
  mrs x3, DBGWCR0_EL1
  mrs x3, DBGWCR1_EL1
  mrs x3, DBGWCR2_EL1
  mrs x3, DBGWCR3_EL1
  mrs x3, DBGWCR4_EL1
  mrs x3, DBGWCR5_EL1
  mrs x3, DBGWCR6_EL1
  mrs x3, DBGWCR7_EL1
  mrs x3, DBGWCR8_EL1
  mrs x3, DBGWCR9_EL1
  mrs x3, DBGWCR10_EL1
  mrs x3, DBGWCR11_EL1
  mrs x3, DBGWCR12_EL1
  mrs x3, DBGWCR13_EL1
  mrs x3, DBGWCR14_EL1
  mrs x3, DBGWCR15_EL1
  mrs x3, MDRAR_EL1
  mrs x3, OSLSR_EL1
  mrs x3, OSDLR_EL1
  mrs x3, DBGPRCR_EL1
  mrs x3, DBGCLAIMSET_EL1
  mrs x3, DBGCLAIMCLR_EL1
  mrs x3, DBGAUTHSTATUS_EL1
  mrs x1, S3_2_C15_C6_4
  mrs x3, s3_3_c11_c1_4
  mrs x3, S3_3_c11_c1_4

; CHECK: mrs x3, ACTLR_EL1              ; encoding: [0x23,0x10,0x38,0xd5]
; CHECK: mrs x3, ACTLR_EL2              ; encoding: [0x23,0x10,0x3c,0xd5]
; CHECK: mrs x3, ACTLR_EL3              ; encoding: [0x23,0x10,0x3e,0xd5]
; CHECK: mrs x3, AFSR0_EL1              ; encoding: [0x03,0x51,0x38,0xd5]
; CHECK: mrs x3, AFSR0_EL2              ; encoding: [0x03,0x51,0x3c,0xd5]
; CHECK: mrs x3, AFSR0_EL3              ; encoding: [0x03,0x51,0x3e,0xd5]
; CHECK: mrs x3, AIDR_EL1               ; encoding: [0xe3,0x00,0x39,0xd5]
; CHECK: mrs x3, AFSR1_EL1              ; encoding: [0x23,0x51,0x38,0xd5]
; CHECK: mrs x3, AFSR1_EL2              ; encoding: [0x23,0x51,0x3c,0xd5]
; CHECK: mrs x3, AFSR1_EL3              ; encoding: [0x23,0x51,0x3e,0xd5]
; CHECK: mrs x3, AMAIR_EL1              ; encoding: [0x03,0xa3,0x38,0xd5]
; CHECK: mrs x3, AMAIR_EL2              ; encoding: [0x03,0xa3,0x3c,0xd5]
; CHECK: mrs x3, AMAIR_EL3              ; encoding: [0x03,0xa3,0x3e,0xd5]
; CHECK: mrs x3, CCSIDR_EL1             ; encoding: [0x03,0x00,0x39,0xd5]
; CHECK: mrs x3, CLIDR_EL1              ; encoding: [0x23,0x00,0x39,0xd5]
; CHECK-V83: mrs x3, CCSIDR2_EL1        ; encoding: [0x43,0x00,0x39,0xd5]
; CHECK: mrs x3, CNTFRQ_EL0             ; encoding: [0x03,0xe0,0x3b,0xd5]
; CHECK: mrs x3, CNTHCTL_EL2            ; encoding: [0x03,0xe1,0x3c,0xd5]
; CHECK: mrs x3, CNTHP_CTL_EL2          ; encoding: [0x23,0xe2,0x3c,0xd5]
; CHECK: mrs x3, CNTHP_CVAL_EL2         ; encoding: [0x43,0xe2,0x3c,0xd5]
; CHECK: mrs x3, CNTHP_TVAL_EL2         ; encoding: [0x03,0xe2,0x3c,0xd5]
; CHECK: mrs x3, CNTKCTL_EL1            ; encoding: [0x03,0xe1,0x38,0xd5]
; CHECK: mrs x3, CNTPCT_EL0             ; encoding: [0x23,0xe0,0x3b,0xd5]
; CHECK: mrs x3, CNTP_CTL_EL0           ; encoding: [0x23,0xe2,0x3b,0xd5]
; CHECK: mrs x3, CNTP_CVAL_EL0          ; encoding: [0x43,0xe2,0x3b,0xd5]
; CHECK: mrs x3, CNTP_TVAL_EL0          ; encoding: [0x03,0xe2,0x3b,0xd5]
; CHECK: mrs x3, CNTVCT_EL0             ; encoding: [0x43,0xe0,0x3b,0xd5]
; CHECK: mrs x3, CNTVOFF_EL2            ; encoding: [0x63,0xe0,0x3c,0xd5]
; CHECK: mrs x3, CNTV_CTL_EL0           ; encoding: [0x23,0xe3,0x3b,0xd5]
; CHECK: mrs x3, CNTV_CVAL_EL0          ; encoding: [0x43,0xe3,0x3b,0xd5]
; CHECK: mrs x3, CNTV_TVAL_EL0          ; encoding: [0x03,0xe3,0x3b,0xd5]
; CHECK: mrs x3, CONTEXTIDR_EL1         ; encoding: [0x23,0xd0,0x38,0xd5]
; CHECK: mrs x3, CPACR_EL1              ; encoding: [0x43,0x10,0x38,0xd5]
; CHECK: mrs x3, CPTR_EL2               ; encoding: [0x43,0x11,0x3c,0xd5]
; CHECK: mrs x3, CPTR_EL3               ; encoding: [0x43,0x11,0x3e,0xd5]
; CHECK: mrs x3, CSSELR_EL1             ; encoding: [0x03,0x00,0x3a,0xd5]
; CHECK: mrs x3, CTR_EL0                ; encoding: [0x23,0x00,0x3b,0xd5]
; CHECK: mrs x3, CurrentEL              ; encoding: [0x43,0x42,0x38,0xd5]
; CHECK: mrs x3, DACR32_EL2             ; encoding: [0x03,0x30,0x3c,0xd5]
; CHECK: mrs x3, DCZID_EL0              ; encoding: [0xe3,0x00,0x3b,0xd5]
; CHECK: mrs x3, REVIDR_EL1             ; encoding: [0xc3,0x00,0x38,0xd5]
; CHECK: mrs x3, ESR_EL1                ; encoding: [0x03,0x52,0x38,0xd5]
; CHECK: mrs x3, ESR_EL2                ; encoding: [0x03,0x52,0x3c,0xd5]
; CHECK: mrs x3, ESR_EL3                ; encoding: [0x03,0x52,0x3e,0xd5]
; CHECK: mrs x3, FAR_EL1                ; encoding: [0x03,0x60,0x38,0xd5]
; CHECK: mrs x3, FAR_EL2                ; encoding: [0x03,0x60,0x3c,0xd5]
; CHECK: mrs x3, FAR_EL3                ; encoding: [0x03,0x60,0x3e,0xd5]
; CHECK: mrs x3, FPEXC32_EL2            ; encoding: [0x03,0x53,0x3c,0xd5]
; CHECK: mrs x3, HACR_EL2               ; encoding: [0xe3,0x11,0x3c,0xd5]
; CHECK: mrs x3, HCR_EL2                ; encoding: [0x03,0x11,0x3c,0xd5]
; CHECK: mrs x3, HPFAR_EL2              ; encoding: [0x83,0x60,0x3c,0xd5]
; CHECK: mrs x3, HSTR_EL2               ; encoding: [0x63,0x11,0x3c,0xd5]
; CHECK: mrs x3, ID_AA64DFR0_EL1        ; encoding: [0x03,0x05,0x38,0xd5]
; CHECK: mrs x3, ID_AA64DFR1_EL1        ; encoding: [0x23,0x05,0x38,0xd5]
; CHECK: mrs x3, ID_AA64ISAR0_EL1       ; encoding: [0x03,0x06,0x38,0xd5]
; CHECK: mrs x3, ID_AA64ISAR1_EL1       ; encoding: [0x23,0x06,0x38,0xd5]
; CHECK: mrs x3, ID_AA64ISAR2_EL1       ; encoding: [0x43,0x06,0x38,0xd5]
; CHECK: mrs x3, ID_AA64MMFR0_EL1       ; encoding: [0x03,0x07,0x38,0xd5]
; CHECK: mrs x3, ID_AA64MMFR1_EL1       ; encoding: [0x23,0x07,0x38,0xd5]
; CHECK: mrs x3, ID_AA64PFR0_EL1        ; encoding: [0x03,0x04,0x38,0xd5]
; CHECK: mrs x3, ID_AA64PFR1_EL1        ; encoding: [0x23,0x04,0x38,0xd5]
; CHECK: mrs x3, IFSR32_EL2             ; encoding: [0x23,0x50,0x3c,0xd5]
; CHECK: mrs x3, ISR_EL1                ; encoding: [0x03,0xc1,0x38,0xd5]
; CHECK: mrs x3, MAIR_EL1               ; encoding: [0x03,0xa2,0x38,0xd5]
; CHECK: mrs x3, MAIR_EL2               ; encoding: [0x03,0xa2,0x3c,0xd5]
; CHECK: mrs x3, MAIR_EL3               ; encoding: [0x03,0xa2,0x3e,0xd5]
; CHECK: mrs x3, MDCR_EL2               ; encoding: [0x23,0x11,0x3c,0xd5]
; CHECK: mrs x3, MDCR_EL3               ; encoding: [0x23,0x13,0x3e,0xd5]
; CHECK: mrs x3, MIDR_EL1               ; encoding: [0x03,0x00,0x38,0xd5]
; CHECK: mrs x3, MPIDR_EL1              ; encoding: [0xa3,0x00,0x38,0xd5]
; CHECK: mrs x3, MVFR0_EL1              ; encoding: [0x03,0x03,0x38,0xd5]
; CHECK: mrs x3, MVFR1_EL1              ; encoding: [0x23,0x03,0x38,0xd5]
; CHECK: mrs x3, PAR_EL1                ; encoding: [0x03,0x74,0x38,0xd5]
; CHECK: mrs x3, RVBAR_EL1              ; encoding: [0x23,0xc0,0x38,0xd5]
; CHECK: mrs x3, RVBAR_EL2              ; encoding: [0x23,0xc0,0x3c,0xd5]
; CHECK: mrs x3, RVBAR_EL3              ; encoding: [0x23,0xc0,0x3e,0xd5]
; CHECK: mrs x3, SCR_EL3                ; encoding: [0x03,0x11,0x3e,0xd5]
; CHECK: mrs x3, SCTLR_EL1              ; encoding: [0x03,0x10,0x38,0xd5]
; CHECK: mrs x3, SCTLR_EL2              ; encoding: [0x03,0x10,0x3c,0xd5]
; CHECK: mrs x3, SCTLR_EL3              ; encoding: [0x03,0x10,0x3e,0xd5]
; CHECK: mrs x3, SDER32_EL3             ; encoding: [0x23,0x11,0x3e,0xd5]
; CHECK: mrs x3, TCR_EL1                ; encoding: [0x43,0x20,0x38,0xd5]
; CHECK: mrs x3, TCR_EL2                ; encoding: [0x43,0x20,0x3c,0xd5]
; CHECK: mrs x3, TCR_EL3                ; encoding: [0x43,0x20,0x3e,0xd5]
; CHECK: mrs x3, TEECR32_EL1            ; encoding: [0x03,0x00,0x32,0xd5]
; CHECK: mrs x3, TEEHBR32_EL1           ; encoding: [0x03,0x10,0x32,0xd5]
; CHECK: mrs x3, TPIDRRO_EL0            ; encoding: [0x63,0xd0,0x3b,0xd5]
; CHECK: mrs x3, TPIDR_EL0              ; encoding: [0x43,0xd0,0x3b,0xd5]
; CHECK: mrs x3, TPIDR_EL1              ; encoding: [0x83,0xd0,0x38,0xd5]
; CHECK: mrs x3, TPIDR_EL2              ; encoding: [0x43,0xd0,0x3c,0xd5]
; CHECK: mrs x3, TPIDR_EL3              ; encoding: [0x43,0xd0,0x3e,0xd5]
; CHECK: mrs x3, TTBR0_EL1              ; encoding: [0x03,0x20,0x38,0xd5]
; CHECK: mrs x3, TTBR0_EL2              ; encoding: [0x03,0x20,0x3c,0xd5]
; CHECK: mrs x3, TTBR0_EL3              ; encoding: [0x03,0x20,0x3e,0xd5]
; CHECK: mrs x3, TTBR1_EL1              ; encoding: [0x23,0x20,0x38,0xd5]
; CHECK: mrs x3, VBAR_EL1               ; encoding: [0x03,0xc0,0x38,0xd5]
; CHECK: mrs x3, VBAR_EL2               ; encoding: [0x03,0xc0,0x3c,0xd5]
; CHECK: mrs x3, VBAR_EL3               ; encoding: [0x03,0xc0,0x3e,0xd5]
; CHECK: mrs x3, VMPIDR_EL2             ; encoding: [0xa3,0x00,0x3c,0xd5]
; CHECK: mrs x3, VPIDR_EL2              ; encoding: [0x03,0x00,0x3c,0xd5]
; CHECK: mrs x3, VTCR_EL2               ; encoding: [0x43,0x21,0x3c,0xd5]
; CHECK: mrs x3, VTTBR_EL2              ; encoding: [0x03,0x21,0x3c,0xd5]
; CHECK: mrs	x3, MDCCSR_EL0          ; encoding: [0x03,0x01,0x33,0xd5]
; CHECK: mrs	x3, MDCCINT_EL1         ; encoding: [0x03,0x02,0x30,0xd5]
; CHECK: mrs	x3, DBGDTR_EL0          ; encoding: [0x03,0x04,0x33,0xd5]
; CHECK: mrs	x3, DBGDTRRX_EL0        ; encoding: [0x03,0x05,0x33,0xd5]
; CHECK: mrs	x3, DBGVCR32_EL2        ; encoding: [0x03,0x07,0x34,0xd5]
; CHECK: mrs	x3, OSDTRRX_EL1         ; encoding: [0x43,0x00,0x30,0xd5]
; CHECK: mrs	x3, MDSCR_EL1           ; encoding: [0x43,0x02,0x30,0xd5]
; CHECK: mrs	x3, OSDTRTX_EL1         ; encoding: [0x43,0x03,0x30,0xd5]
; CHECK: mrs	x3, OSECCR_EL1          ; encoding: [0x43,0x06,0x30,0xd5]
; CHECK: mrs	x3, DBGBVR0_EL1         ; encoding: [0x83,0x00,0x30,0xd5]
; CHECK: mrs	x3, DBGBVR1_EL1         ; encoding: [0x83,0x01,0x30,0xd5]
; CHECK: mrs	x3, DBGBVR2_EL1         ; encoding: [0x83,0x02,0x30,0xd5]
; CHECK: mrs	x3, DBGBVR3_EL1         ; encoding: [0x83,0x03,0x30,0xd5]
; CHECK: mrs	x3, DBGBVR4_EL1         ; encoding: [0x83,0x04,0x30,0xd5]
; CHECK: mrs	x3, DBGBVR5_EL1         ; encoding: [0x83,0x05,0x30,0xd5]
; CHECK: mrs	x3, DBGBVR6_EL1         ; encoding: [0x83,0x06,0x30,0xd5]
; CHECK: mrs	x3, DBGBVR7_EL1         ; encoding: [0x83,0x07,0x30,0xd5]
; CHECK: mrs	x3, DBGBVR8_EL1         ; encoding: [0x83,0x08,0x30,0xd5]
; CHECK: mrs	x3, DBGBVR9_EL1         ; encoding: [0x83,0x09,0x30,0xd5]
; CHECK: mrs	x3, DBGBVR10_EL1        ; encoding: [0x83,0x0a,0x30,0xd5]
; CHECK: mrs	x3, DBGBVR11_EL1        ; encoding: [0x83,0x0b,0x30,0xd5]
; CHECK: mrs	x3, DBGBVR12_EL1        ; encoding: [0x83,0x0c,0x30,0xd5]
; CHECK: mrs	x3, DBGBVR13_EL1        ; encoding: [0x83,0x0d,0x30,0xd5]
; CHECK: mrs	x3, DBGBVR14_EL1        ; encoding: [0x83,0x0e,0x30,0xd5]
; CHECK: mrs	x3, DBGBVR15_EL1        ; encoding: [0x83,0x0f,0x30,0xd5]
; CHECK: mrs	x3, DBGBCR0_EL1         ; encoding: [0xa3,0x00,0x30,0xd5]
; CHECK: mrs	x3, DBGBCR1_EL1         ; encoding: [0xa3,0x01,0x30,0xd5]
; CHECK: mrs	x3, DBGBCR2_EL1         ; encoding: [0xa3,0x02,0x30,0xd5]
; CHECK: mrs	x3, DBGBCR3_EL1         ; encoding: [0xa3,0x03,0x30,0xd5]
; CHECK: mrs	x3, DBGBCR4_EL1         ; encoding: [0xa3,0x04,0x30,0xd5]
; CHECK: mrs	x3, DBGBCR5_EL1         ; encoding: [0xa3,0x05,0x30,0xd5]
; CHECK: mrs	x3, DBGBCR6_EL1         ; encoding: [0xa3,0x06,0x30,0xd5]
; CHECK: mrs	x3, DBGBCR7_EL1         ; encoding: [0xa3,0x07,0x30,0xd5]
; CHECK: mrs	x3, DBGBCR8_EL1         ; encoding: [0xa3,0x08,0x30,0xd5]
; CHECK: mrs	x3, DBGBCR9_EL1         ; encoding: [0xa3,0x09,0x30,0xd5]
; CHECK: mrs	x3, DBGBCR10_EL1        ; encoding: [0xa3,0x0a,0x30,0xd5]
; CHECK: mrs	x3, DBGBCR11_EL1        ; encoding: [0xa3,0x0b,0x30,0xd5]
; CHECK: mrs	x3, DBGBCR12_EL1        ; encoding: [0xa3,0x0c,0x30,0xd5]
; CHECK: mrs	x3, DBGBCR13_EL1        ; encoding: [0xa3,0x0d,0x30,0xd5]
; CHECK: mrs	x3, DBGBCR14_EL1        ; encoding: [0xa3,0x0e,0x30,0xd5]
; CHECK: mrs	x3, DBGBCR15_EL1        ; encoding: [0xa3,0x0f,0x30,0xd5]
; CHECK: mrs	x3, DBGWVR0_EL1         ; encoding: [0xc3,0x00,0x30,0xd5]
; CHECK: mrs	x3, DBGWVR1_EL1         ; encoding: [0xc3,0x01,0x30,0xd5]
; CHECK: mrs	x3, DBGWVR2_EL1         ; encoding: [0xc3,0x02,0x30,0xd5]
; CHECK: mrs	x3, DBGWVR3_EL1         ; encoding: [0xc3,0x03,0x30,0xd5]
; CHECK: mrs	x3, DBGWVR4_EL1         ; encoding: [0xc3,0x04,0x30,0xd5]
; CHECK: mrs	x3, DBGWVR5_EL1         ; encoding: [0xc3,0x05,0x30,0xd5]
; CHECK: mrs	x3, DBGWVR6_EL1         ; encoding: [0xc3,0x06,0x30,0xd5]
; CHECK: mrs	x3, DBGWVR7_EL1         ; encoding: [0xc3,0x07,0x30,0xd5]
; CHECK: mrs	x3, DBGWVR8_EL1         ; encoding: [0xc3,0x08,0x30,0xd5]
; CHECK: mrs	x3, DBGWVR9_EL1         ; encoding: [0xc3,0x09,0x30,0xd5]
; CHECK: mrs	x3, DBGWVR10_EL1        ; encoding: [0xc3,0x0a,0x30,0xd5]
; CHECK: mrs	x3, DBGWVR11_EL1        ; encoding: [0xc3,0x0b,0x30,0xd5]
; CHECK: mrs	x3, DBGWVR12_EL1        ; encoding: [0xc3,0x0c,0x30,0xd5]
; CHECK: mrs	x3, DBGWVR13_EL1        ; encoding: [0xc3,0x0d,0x30,0xd5]
; CHECK: mrs	x3, DBGWVR14_EL1        ; encoding: [0xc3,0x0e,0x30,0xd5]
; CHECK: mrs	x3, DBGWVR15_EL1        ; encoding: [0xc3,0x0f,0x30,0xd5]
; CHECK: mrs	x3, DBGWCR0_EL1         ; encoding: [0xe3,0x00,0x30,0xd5]
; CHECK: mrs	x3, DBGWCR1_EL1         ; encoding: [0xe3,0x01,0x30,0xd5]
; CHECK: mrs	x3, DBGWCR2_EL1         ; encoding: [0xe3,0x02,0x30,0xd5]
; CHECK: mrs	x3, DBGWCR3_EL1         ; encoding: [0xe3,0x03,0x30,0xd5]
; CHECK: mrs	x3, DBGWCR4_EL1         ; encoding: [0xe3,0x04,0x30,0xd5]
; CHECK: mrs	x3, DBGWCR5_EL1         ; encoding: [0xe3,0x05,0x30,0xd5]
; CHECK: mrs	x3, DBGWCR6_EL1         ; encoding: [0xe3,0x06,0x30,0xd5]
; CHECK: mrs	x3, DBGWCR7_EL1         ; encoding: [0xe3,0x07,0x30,0xd5]
; CHECK: mrs	x3, DBGWCR8_EL1         ; encoding: [0xe3,0x08,0x30,0xd5]
; CHECK: mrs	x3, DBGWCR9_EL1         ; encoding: [0xe3,0x09,0x30,0xd5]
; CHECK: mrs	x3, DBGWCR10_EL1        ; encoding: [0xe3,0x0a,0x30,0xd5]
; CHECK: mrs	x3, DBGWCR11_EL1        ; encoding: [0xe3,0x0b,0x30,0xd5]
; CHECK: mrs	x3, DBGWCR12_EL1        ; encoding: [0xe3,0x0c,0x30,0xd5]
; CHECK: mrs	x3, DBGWCR13_EL1        ; encoding: [0xe3,0x0d,0x30,0xd5]
; CHECK: mrs	x3, DBGWCR14_EL1        ; encoding: [0xe3,0x0e,0x30,0xd5]
; CHECK: mrs	x3, DBGWCR15_EL1        ; encoding: [0xe3,0x0f,0x30,0xd5]
; CHECK: mrs	x3, MDRAR_EL1           ; encoding: [0x03,0x10,0x30,0xd5]
; CHECK: mrs	x3, OSLSR_EL1           ; encoding: [0x83,0x11,0x30,0xd5]
; CHECK: mrs	x3, OSDLR_EL1           ; encoding: [0x83,0x13,0x30,0xd5]
; CHECK: mrs	x3, DBGPRCR_EL1         ; encoding: [0x83,0x14,0x30,0xd5]
; CHECK: mrs	x3, DBGCLAIMSET_EL1     ; encoding: [0xc3,0x78,0x30,0xd5]
; CHECK: mrs	x3, DBGCLAIMCLR_EL1     ; encoding: [0xc3,0x79,0x30,0xd5]
; CHECK: mrs	x3, DBGAUTHSTATUS_EL1   ; encoding: [0xc3,0x7e,0x30,0xd5]
; CHECK: mrs    x1, S3_2_C15_C6_4       ; encoding: [0x81,0xf6,0x3a,0xd5]
; CHECK: mrs	x3, S3_3_C11_C1_4       ; encoding: [0x83,0xb1,0x3b,0xd5]
; CHECK: mrs	x3, S3_3_C11_C1_4       ; encoding: [0x83,0xb1,0x3b,0xd5]

  msr RMR_EL3, x0
  msr RMR_EL2, x0
  msr RMR_EL1, x0
  msr OSLAR_EL1, x3
  msr DBGDTRTX_EL0, x3
        
; CHECK: msr	RMR_EL3, x0             ; encoding: [0x40,0xc0,0x1e,0xd5]
; CHECK: msr	RMR_EL2, x0             ; encoding: [0x40,0xc0,0x1c,0xd5]
; CHECK: msr	RMR_EL1, x0             ; encoding: [0x40,0xc0,0x18,0xd5]
; CHECK: msr	OSLAR_EL1, x3           ; encoding: [0x83,0x10,0x10,0xd5]
; CHECK: msr	DBGDTRTX_EL0, x3        ; encoding: [0x03,0x05,0x13,0xd5]
        
 mrs x0, ID_PFR0_EL1
 mrs x0, ID_PFR1_EL1
 mrs x0, ID_DFR0_EL1
 mrs x0, ID_AFR0_EL1
 mrs x0, ID_ISAR0_EL1
 mrs x0, ID_ISAR1_EL1
 mrs x0, ID_ISAR2_EL1
 mrs x0, ID_ISAR3_EL1
 mrs x0, ID_ISAR4_EL1
 mrs x0, ID_ISAR5_EL1
 mrs x0, AFSR1_EL1
 mrs x0, AFSR0_EL1
 mrs x0, REVIDR_EL1
; CHECK: mrs	x0, ID_PFR0_EL1         ; encoding: [0x00,0x01,0x38,0xd5]
; CHECK: mrs	x0, ID_PFR1_EL1         ; encoding: [0x20,0x01,0x38,0xd5]
; CHECK: mrs	x0, ID_DFR0_EL1         ; encoding: [0x40,0x01,0x38,0xd5]
; CHECK: mrs	x0, ID_AFR0_EL1         ; encoding: [0x60,0x01,0x38,0xd5]
; CHECK: mrs	x0, ID_ISAR0_EL1        ; encoding: [0x00,0x02,0x38,0xd5]
; CHECK: mrs	x0, ID_ISAR1_EL1        ; encoding: [0x20,0x02,0x38,0xd5]
; CHECK: mrs	x0, ID_ISAR2_EL1        ; encoding: [0x40,0x02,0x38,0xd5]
; CHECK: mrs	x0, ID_ISAR3_EL1        ; encoding: [0x60,0x02,0x38,0xd5]
; CHECK: mrs	x0, ID_ISAR4_EL1        ; encoding: [0x80,0x02,0x38,0xd5]
; CHECK: mrs	x0, ID_ISAR5_EL1        ; encoding: [0xa0,0x02,0x38,0xd5]
; CHECK: mrs	x0, AFSR1_EL1           ; encoding: [0x20,0x51,0x38,0xd5]
; CHECK: mrs	x0, AFSR0_EL1           ; encoding: [0x00,0x51,0x38,0xd5]
; CHECK: mrs	x0, REVIDR_EL1          ; encoding: [0xc0,0x00,0x38,0xd5]
