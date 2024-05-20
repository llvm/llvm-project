# REQUIRES: aarch64

# This checks that lldb's disassembler enables every extension that an AArch64
# target could have.

# RUN: llvm-mc -filetype=obj -triple aarch64-linux-gnueabihf %s -o %t --mattr=+all
# RUN: %lldb %t -o "disassemble -n fn" -o exit 2>&1 | FileCheck %s

.globl  fn
.type   fn, @function
fn:
  // These are in alphabetical order by extension name
  aesd v0.16b, v0.16b                   // AEK_AES
  bfadd z23.h, p3/m, z23.h, z13.h       // AEK_B16B16
  bfdot v2.2s, v3.4h, v4.4h             // AEK_BF16
  brb iall                              // AEK_BRBE
  crc32b w0, w0, w0                     // AEK_CRC
  // AEK_CRYPTO enables a combination of other features
  smin x0, x0, #0                       // AEK_CSSC
  sysp	#0, c2, c0, #0, x0, x1          // AEK_D128
  sdot v0.2s, v1.8b, v2.8b              // AEK_DOTPROD
  fmmla z0.s, z1.s, z2.s                // AEK_F32MM
  fmmla z0.d, z1.d, z2.d                // AEK_F64MM
  cfinv                                 // AEK_FLAGM
  fcvt d0, s0                           // AEK_FP
  fabs h1, h2                           // AEK_FP16
  fmlal v0.2s, v1.2h, v2.2h             // AEK_FP16FML
  bc.eq lbl                             // AEK_HBC
  smmla v1.4s, v16.16b, v31.16b         // AEK_I8MM
  ld64b x0, [x13]                       // AEK_LS64
  ldaddab w0, w0, [sp]                  // AEK_LSE
  ldclrp x1, x2, [x11]                  // AEK_LSE128
  irg x0, x0                            // AEK_MTE
  cpyfp [x0]!, [x1]!, x2!               // AEK_MOPS
  pacia x0, x1                          // AEK_PAUTH
  mrs x0, pmccntr_el0                   // AEK_PERFMON
  cfp rctx, x0                          // AEK_PREDRES
  psb csync                             // AEK_PROFILE/SPE
  msr erxpfgctl_el1, x0                 // AEK_RAS
  ldaprb w0, [x0, #0]                   // AEK_RCPC
  stilp w26, w2, [x18]                  // AEK_RCPC3
  sqrdmlah v0.4h, v1.4h, v2.4h          // AEK_RDM
  mrs x0, rndr                          // AEK_RAND
  sb                                    // AEK_SB
  sha256h q0, q0, v0.4s                 // AEK_SHA2
  bcax v0.16b, v0.16b, v0.16b, v0.16b   // AEK_SHA3
  addp v0.4s, v0.4s, v0.4s              // AEK_SIMD (neon)
  sm4e v0.4s, v0.4s                     // AEK_SM4
  addha za0.s, p0/m, p0/m, z0.s         // AEK_SME
  fadd za.h[w11, 7], {z12.h - z13.h}    // AEK_SMEF16F16
  fmopa za0.d, p0/m, p0/m, z0.d, z0.d   // AEK_SMEF64F64
  addha za0.d, p0/m, p0/m, z0.d         // AEK_SMEI16I64
  add {z0.h, z1.h}, {z0.h, z1.h}, z0.h  // AEK_SME2
  // AEK_SME2P1: see AEK_SVE2P1
  mrs x2, ssbs                          // AEK_SSBS
  abs z31.h, p7/m, z31.h                // AEK_SVE
  sqdmlslbt z0.d, z1.s, z31.s           // AEK_SVE2
  aesd z0.b, z0.b, z31.b                // AEK_SVE2AES
  bdep z0.b, z1.b, z31.b                // AEK_SVE2BITPERM
  rax1 z0.d, z0.d, z0.d                 // AEK_SVE2SHA3
  sm4e z0.s, z0.s, z0.s                 // AEK_SVE2SM4
  addqv   v0.8h, p0, z0.h               // AEK_SVE2P1 / AEK_SME2P1
  rcwswp x0, x1, [x2]                   // AEK_THE
  tcommit                               // AEK_TME
lbl:
.fn_end:
  .size   fn, .fn_end-fn

# CHECK: command-disassemble-aarch64-extensions.s.tmp`fn:
# CHECK-NEXT: aesd   v0.16b, v0.16b
# CHECK-NEXT: bfadd  z23.h, p3/m, z23.h, z13.h
# CHECK-NEXT: bfdot  v2.2s, v3.4h, v4.4h
# CHECK-NEXT: brb    iall
# CHECK-NEXT: crc32b w0, w0, w0
# CHECK-NEXT: smin   x0, x0, #0
# CHECK-NEXT: sysp   #0x0, c2, c0, #0x0, x0, x1
# CHECK-NEXT: sdot   v0.2s, v1.8b, v2.8b
# CHECK-NEXT: fmmla  z0.s, z1.s, z2.s
# CHECK-NEXT: fmmla  z0.d, z1.d, z2.d
# CHECK-NEXT: cfinv
# CHECK-NEXT: fcvt   d0, s0
# CHECK-NEXT: fabs   h1, h2
# CHECK-NEXT: fmlal  v0.2s, v1.2h, v2.2h
# CHECK-NEXT: bc.eq 0xc8
# CHECK-NEXT: smmla  v1.4s, v16.16b, v31.16b
# CHECK-NEXT: ld64b  x0, [x13]
# CHECK-NEXT: ldaddab w0, w0, [sp]
# CHECK-NEXT: ldclrp  x1, x2, [x11]
# CHECK-NEXT: irg    x0, x0
# CHECK-NEXT: cpyfp  [x0]!, [x1]!, x2!
# CHECK-NEXT: pacia  x0, x1
# CHECK-NEXT: mrs    x0, PMCCNTR_EL0
# CHECK-NEXT: cfp    rctx, x0
# CHECK-NEXT: psb    csync
# CHECK-NEXT: msr    ERXPFGCTL_EL1, x0
# CHECK-NEXT: ldaprb w0, [x0]
# CHECK-NEXT: stilp w26, w2, [x18]
# CHECK-NEXT: sqrdmlah v0.4h, v1.4h, v2.4h
# CHECK-NEXT: mrs    x0, RNDR
# CHECK-NEXT: sb
# CHECK-NEXT: sha256h q0, q0, v0.4s
# CHECK-NEXT: bcax   v0.16b, v0.16b, v0.16b, v0.16b
# CHECK-NEXT: addp   v0.4s, v0.4s, v0.4s
# CHECK-NEXT: sm4e   v0.4s, v0.4s
# CHECK-NEXT: addha  za0.s, p0/m, p0/m, z0.s
# CHECK-NEXT: fadd   za.h[w11, 7, vgx2], { z12.h, z13.h }
# CHECK-NEXT: fmopa  za0.d, p0/m, p0/m, z0.d, z0.d
# CHECK-NEXT: addha  za0.d, p0/m, p0/m, z0.d
# CHECK-NEXT: add    { z0.h, z1.h }, { z0.h, z1.h }, z0.h
# CHECK-NEXT: mrs    x2, SSBS
# CHECK-NEXT: abs    z31.h, p7/m, z31.h
# CHECK-NEXT: sqdmlslbt z0.d, z1.s, z31.s
# CHECK-NEXT: aesd   z0.b, z0.b, z31.b
# CHECK-NEXT: bdep   z0.b, z1.b, z31.b
# CHECK-NEXT: rax1   z0.d, z0.d, z0.d
# CHECK-NEXT: sm4e   z0.s, z0.s, z0.s
# CHECK-NEXT: addqv  v0.8h, p0, z0.h
# CHECK-NEXT: rcwswp x0, x1, [x2]
# CHECK-NEXT: tcommit
