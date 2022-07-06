# REQUIRES: aarch64

# This checks that lldb's disassembler enables every extension that an AArch64
# target could have.

# RUN: llvm-mc -filetype=obj -triple aarch64-linux-gnueabihf %s -o %t --mattr=+all
# RUN: %lldb %t -o "disassemble -n fn" -o exit 2>&1 | FileCheck %s

.globl  fn
.type   fn, @function
fn:
  // These are in the same order as llvm/include/llvm/Support/AArch64TargetParser.def
  crc32b w0, w0, w0                   // CRC
  ldaddab w0, w0, [sp]                // LSE
  sqrdmlah v0.4h, v1.4h, v2.4h        // RDM
  // CRYPTO enables a combination of other features
  sm4e v0.4s, v0.4s                   // SM4
  bcax v0.16b, v0.16b, v0.16b, v0.16b // SHA3
  sha256h q0, q0, v0.4s               // SHA256
  aesd v0.16b, v0.16b                 // AES
  sdot v0.2s, v1.8b, v2.8b            // DOTPROD
  fcvt d0, s0                         // FP
  addp v0.4s, v0.4s, v0.4s            // SIMD (neon)
  fabs h1, h2                         // FP16
  fmlal v0.2s, v1.2h, v2.2h           // FP16FML
  psb csync                           // PROFILE/SPE
  msr erxpfgctl_el1, x0               // RAS
  abs z31.h, p7/m, z31.h              // SVE
  sqdmlslbt z0.d, z1.s, z31.s         // SVE2
  aesd z0.b, z0.b, z31.b              // SVE2AES
  sm4e z0.s, z0.s, z0.s               // SVE2SM4
  rax1 z0.d, z0.d, z0.d               // SVE2SHA3
  bdep z0.b, z1.b, z31.b              // SVE2BITPERM
  ldaprb w0, [x0, #0]                 // RCPC
  mrs x0, rndr                        // RAND
  irg x0, x0                          // MTE
  mrs x2, ssbs                        // SSBS
  sb                                  // SB
  cfp rctx, x0                        // PREDRES
  bfdot v2.2s, v3.4h, v4.4h           // BF16
  smmla v1.4s, v16.16b, v31.16b       // I8MM
  fmmla z0.s, z1.s, z2.s              // F32MM
  fmmla z0.d, z1.d, z2.d              // F64MM
  tcommit                             // TME
  ld64b x0, [x13]                     // LS64
  brb iall                            // BRBE
  pacia x0, x1                        // PAUTH
  cfinv                               // FLAGM
  addha za0.s, p0/m, p0/m, z0.s       // SME
  fmopa za0.d, p0/m, p0/m, z0.d, z0.d // SMEF64
  addha za0.d, p0/m, p0/m, z0.d       // SMEI64
lbl:
  bc.eq lbl                           // HBC
  cpyfp [x0]!, [x1]!, x2!             // MOPS
  mrs x0, pmccntr_el0                 // PERFMON
.fn_end:
  .size   fn, .fn_end-fn

# CHECK: command-disassemble-aarch64-extensions.s.tmp`fn:
# CHECK: crc32b w0, w0, w0
# CHECK: ldaddab w0, w0, [sp]
# CHECK: sqrdmlah v0.4h, v1.4h, v2.4h
# CHECK: sm4e   v0.4s, v0.4s
# CHECK: bcax   v0.16b, v0.16b, v0.16b, v0.16b
# CHECK: sha256h q0, q0, v0.4s
# CHECK: aesd   v0.16b, v0.16b
# CHECK: sdot   v0.2s, v1.8b, v2.8b
# CHECK: fcvt   d0, s0
# CHECK: addp   v0.4s, v0.4s, v0.4s
# CHECK: fabs   h1, h2
# CHECK: fmlal  v0.2s, v1.2h, v2.2h
# CHECK: psb    csync
# CHECK: msr    ERXPFGCTL_EL1, x0
# CHECK: abs    z31.h, p7/m, z31.h
# CHECK: sqdmlslbt z0.d, z1.s, z31.s
# CHECK: aesd   z0.b, z0.b, z31.b
# CHECK: sm4e   z0.s, z0.s, z0.s
# CHECK: rax1   z0.d, z0.d, z0.d
# CHECK: bdep   z0.b, z1.b, z31.b
# CHECK: ldaprb w0, [x0]
# CHECK: mrs    x0, RNDR
# CHECK: irg    x0, x0
# CHECK: mrs    x2, SSBS
# CHECK: sb
# CHECK: cfp    rctx, x0
# CHECK: bfdot  v2.2s, v3.4h, v4.4h
# CHECK: smmla  v1.4s, v16.16b, v31.16b
# CHECK: fmmla  z0.s, z1.s, z2.s
# CHECK: fmmla  z0.d, z1.d, z2.d
# CHECK: tcommit
# CHECK: ld64b  x0, [x13]
# CHECK: brb    iall
# CHECK: pacia  x0, x1
# CHECK: cfinv
# CHECK: addha  za0.s, p0/m, p0/m, z0.s
# CHECK: fmopa  za0.d, p0/m, p0/m, z0.d, z0.d
# CHECK: addha  za0.d, p0/m, p0/m, z0.d
# CHECK: bc.eq  0x98
# CHECK: cpyfp  [x0]!, [x1]!, x2!
# CHECK: mrs    x0, PMCCNTR_EL0
