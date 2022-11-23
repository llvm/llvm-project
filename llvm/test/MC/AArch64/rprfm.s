// RPRFM is now a v8.0a optional instruction, and overlaps with PRFM. This test
// checks we can assemble as PRFM, and we always disassemble as RPRFM.

// RUN: llvm-mc -triple=aarch64 -show-encoding --print-imm-hex=false < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj < %s \
// RUN:        | llvm-objdump -d --print-imm-hex=false - | FileCheck %s --check-prefix=CHECK-INST
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -disassemble -show-encoding --print-imm-hex=false \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

rprfm #0, x0, [x0]
// CHECK-INST: rprfm pldkeep, x0, [x0]
// CHECK-ENCODING: [0x18,0x48,0xa0,0xf8]

rprfm #1, x0, [x0]
// CHECK-INST: rprfm pstkeep, x0, [x0]
// CHECK-ENCODING: [0x19,0x48,0xa0,0xf8]

rprfm #2, x0, [x0]
// CHECK-INST: rprfm #2, x0, [x0]
// CHECK-ENCODING: [0x1a,0x48,0xa0,0xf8]

rprfm #3, x0, [x0]
// CHECK-INST: rprfm #3, x0, [x0]
// CHECK-ENCODING: [0x1b,0x48,0xa0,0xf8]

rprfm #4, x0, [x0]
// CHECK-INST: rprfm pldstrm, x0, [x0]
// CHECK-ENCODING: [0x1c,0x48,0xa0,0xf8]

rprfm #5, x0, [x0]
// CHECK-INST: rprfm pststrm, x0, [x0]
// CHECK-ENCODING: [0x1d,0x48,0xa0,0xf8]

rprfm #6, x0, [x0]
// CHECK-INST: rprfm #6, x0, [x0]
// CHECK-ENCODING: [0x1e,0x48,0xa0,0xf8]

rprfm #7, x0, [x0]
// CHECK-INST: rprfm #7, x0, [x0]
// CHECK-ENCODING: [0x1f,0x48,0xa0,0xf8]

rprfm #8, x0, [x0]
// CHECK-INST: rprfm #8, x0, [x0]
// CHECK-ENCODING: [0x18,0x58,0xa0,0xf8]

rprfm #9, x0, [x0]
// CHECK-INST: rprfm #9, x0, [x0]
// CHECK-ENCODING: [0x19,0x58,0xa0,0xf8]

rprfm #10, x0, [x0]
// CHECK-INST: rprfm #10, x0, [x0]
// CHECK-ENCODING: [0x1a,0x58,0xa0,0xf8]

rprfm #11, x0, [x0]
// CHECK-INST: rprfm #11, x0, [x0]
// CHECK-ENCODING: [0x1b,0x58,0xa0,0xf8]

rprfm #12, x0, [x0]
// CHECK-INST: rprfm #12, x0, [x0]
// CHECK-ENCODING: [0x1c,0x58,0xa0,0xf8]

rprfm #13, x0, [x0]
// CHECK-INST: rprfm #13, x0, [x0]
// CHECK-ENCODING: [0x1d,0x58,0xa0,0xf8]

rprfm #14, x0, [x0]
// CHECK-INST: rprfm #14, x0, [x0]
// CHECK-ENCODING: [0x1e,0x58,0xa0,0xf8]

rprfm #15, x0, [x0]
// CHECK-INST: rprfm #15, x0, [x0]
// CHECK-ENCODING: [0x1f,0x58,0xa0,0xf8]

rprfm #16, x0, [x0]
// CHECK-INST: rprfm #16, x0, [x0]
// CHECK-ENCODING: [0x18,0x68,0xa0,0xf8]

rprfm #17, x0, [x0]
// CHECK-INST: rprfm #17, x0, [x0]
// CHECK-ENCODING: [0x19,0x68,0xa0,0xf8]

rprfm #18, x0, [x0]
// CHECK-INST: rprfm #18, x0, [x0]
// CHECK-ENCODING: [0x1a,0x68,0xa0,0xf8]

rprfm #19, x0, [x0]
// CHECK-INST: rprfm #19, x0, [x0]
// CHECK-ENCODING: [0x1b,0x68,0xa0,0xf8]

rprfm #20, x0, [x0]
// CHECK-INST: rprfm #20, x0, [x0]
// CHECK-ENCODING: [0x1c,0x68,0xa0,0xf8]

rprfm #21, x0, [x0]
// CHECK-INST: rprfm #21, x0, [x0]
// CHECK-ENCODING: [0x1d,0x68,0xa0,0xf8]

rprfm #22, x0, [x0]
// CHECK-INST: rprfm #22, x0, [x0]
// CHECK-ENCODING: [0x1e,0x68,0xa0,0xf8]

rprfm #23, x0, [x0]
// CHECK-INST: rprfm #23, x0, [x0]
// CHECK-ENCODING: [0x1f,0x68,0xa0,0xf8]

rprfm #24, x0, [x0]
// CHECK-INST: rprfm #24, x0, [x0]
// CHECK-ENCODING: [0x18,0x78,0xa0,0xf8]

rprfm #25, x0, [x0]
// CHECK-INST: rprfm #25, x0, [x0]
// CHECK-ENCODING: [0x19,0x78,0xa0,0xf8]

rprfm #26, x0, [x0]
// CHECK-INST: rprfm #26, x0, [x0]
// CHECK-ENCODING: [0x1a,0x78,0xa0,0xf8]

rprfm #27, x0, [x0]
// CHECK-INST: rprfm #27, x0, [x0]
// CHECK-ENCODING: [0x1b,0x78,0xa0,0xf8]

rprfm #28, x0, [x0]
// CHECK-INST: rprfm #28, x0, [x0]
// CHECK-ENCODING: [0x1c,0x78,0xa0,0xf8]

rprfm #29, x0, [x0]
// CHECK-INST: rprfm #29, x0, [x0]
// CHECK-ENCODING: [0x1d,0x78,0xa0,0xf8]

rprfm #30, x0, [x0]
// CHECK-INST: rprfm #30, x0, [x0]
// CHECK-ENCODING: [0x1e,0x78,0xa0,0xf8]

rprfm #31, x0, [x0]
// CHECK-INST: rprfm #31, x0, [x0]
// CHECK-ENCODING: [0x1f,0x78,0xa0,0xf8]

rprfm #32, x0, [x0]
// CHECK-INST: rprfm #32, x0, [x0]
// CHECK-ENCODING: [0x18,0xc8,0xa0,0xf8]

rprfm #33, x0, [x0]
// CHECK-INST: rprfm #33, x0, [x0]
// CHECK-ENCODING: [0x19,0xc8,0xa0,0xf8]

rprfm #34, x0, [x0]
// CHECK-INST: rprfm #34, x0, [x0]
// CHECK-ENCODING: [0x1a,0xc8,0xa0,0xf8]

rprfm #35, x0, [x0]
// CHECK-INST: rprfm #35, x0, [x0]
// CHECK-ENCODING: [0x1b,0xc8,0xa0,0xf8]

rprfm #36, x0, [x0]
// CHECK-INST: rprfm #36, x0, [x0]
// CHECK-ENCODING: [0x1c,0xc8,0xa0,0xf8]

rprfm #37, x0, [x0]
// CHECK-INST: rprfm #37, x0, [x0]
// CHECK-ENCODING: [0x1d,0xc8,0xa0,0xf8]

rprfm #38, x0, [x0]
// CHECK-INST: rprfm #38, x0, [x0]
// CHECK-ENCODING: [0x1e,0xc8,0xa0,0xf8]

rprfm #39, x0, [x0]
// CHECK-INST: rprfm #39, x0, [x0]
// CHECK-ENCODING: [0x1f,0xc8,0xa0,0xf8]

rprfm #40, x0, [x0]
// CHECK-INST: rprfm #40, x0, [x0]
// CHECK-ENCODING: [0x18,0xd8,0xa0,0xf8]

rprfm #41, x0, [x0]
// CHECK-INST: rprfm #41, x0, [x0]
// CHECK-ENCODING: [0x19,0xd8,0xa0,0xf8]

rprfm #42, x0, [x0]
// CHECK-INST: rprfm #42, x0, [x0]
// CHECK-ENCODING: [0x1a,0xd8,0xa0,0xf8]

rprfm #43, x0, [x0]
// CHECK-INST: rprfm #43, x0, [x0]
// CHECK-ENCODING: [0x1b,0xd8,0xa0,0xf8]

rprfm #44, x0, [x0]
// CHECK-INST: rprfm #44, x0, [x0]
// CHECK-ENCODING: [0x1c,0xd8,0xa0,0xf8]

rprfm #45, x0, [x0]
// CHECK-INST: rprfm #45, x0, [x0]
// CHECK-ENCODING: [0x1d,0xd8,0xa0,0xf8]

rprfm #46, x0, [x0]
// CHECK-INST: rprfm #46, x0, [x0]
// CHECK-ENCODING: [0x1e,0xd8,0xa0,0xf8]

rprfm #47, x0, [x0]
// CHECK-INST: rprfm #47, x0, [x0]
// CHECK-ENCODING: [0x1f,0xd8,0xa0,0xf8]

rprfm #48, x0, [x0]
// CHECK-INST: rprfm #48, x0, [x0]
// CHECK-ENCODING: [0x18,0xe8,0xa0,0xf8]

rprfm #49, x0, [x0]
// CHECK-INST: rprfm #49, x0, [x0]
// CHECK-ENCODING: [0x19,0xe8,0xa0,0xf8]

rprfm #50, x0, [x0]
// CHECK-INST: rprfm #50, x0, [x0]
// CHECK-ENCODING: [0x1a,0xe8,0xa0,0xf8]

rprfm #51, x0, [x0]
// CHECK-INST: rprfm #51, x0, [x0]
// CHECK-ENCODING: [0x1b,0xe8,0xa0,0xf8]

rprfm #52, x0, [x0]
// CHECK-INST: rprfm #52, x0, [x0]
// CHECK-ENCODING: [0x1c,0xe8,0xa0,0xf8]

rprfm #53, x0, [x0]
// CHECK-INST: rprfm #53, x0, [x0]
// CHECK-ENCODING: [0x1d,0xe8,0xa0,0xf8]

rprfm #54, x0, [x0]
// CHECK-INST: rprfm #54, x0, [x0]
// CHECK-ENCODING: [0x1e,0xe8,0xa0,0xf8]

rprfm #55, x0, [x0]
// CHECK-INST: rprfm #55, x0, [x0]
// CHECK-ENCODING: [0x1f,0xe8,0xa0,0xf8]

rprfm #56, x0, [x0]
// CHECK-INST: rprfm #56, x0, [x0]
// CHECK-ENCODING: [0x18,0xf8,0xa0,0xf8]

rprfm #57, x0, [x0]
// CHECK-INST: rprfm #57, x0, [x0]
// CHECK-ENCODING: [0x19,0xf8,0xa0,0xf8]

rprfm #58, x0, [x0]
// CHECK-INST: rprfm #58, x0, [x0]
// CHECK-ENCODING: [0x1a,0xf8,0xa0,0xf8]

rprfm #59, x0, [x0]
// CHECK-INST: rprfm #59, x0, [x0]
// CHECK-ENCODING: [0x1b,0xf8,0xa0,0xf8]

rprfm #60, x0, [x0]
// CHECK-INST: rprfm #60, x0, [x0]
// CHECK-ENCODING: [0x1c,0xf8,0xa0,0xf8]

rprfm #61, x0, [x0]
// CHECK-INST: rprfm #61, x0, [x0]
// CHECK-ENCODING: [0x1d,0xf8,0xa0,0xf8]

rprfm #62, x0, [x0]
// CHECK-INST: rprfm #62, x0, [x0]
// CHECK-ENCODING: [0x1e,0xf8,0xa0,0xf8]

rprfm #63, x0, [x0]
// CHECK-INST: rprfm #63, x0, [x0]
// CHECK-ENCODING: [0x1f,0xf8,0xa0,0xf8]

// Aliases
// -----------------------------------------------------------------------------

prfm #24, [x0, w0, uxtw]
// CHECK-INST: rprfm pldkeep, x0, [x0]
// CHECK-ENCODING: [0x18,0x48,0xa0,0xf8]

prfm #25, [x0, w0, uxtw]
// CHECK-INST: rprfm pstkeep, x0, [x0]
// CHECK-ENCODING: [0x19,0x48,0xa0,0xf8]

prfm #26, [x0, w0, uxtw]
// CHECK-INST: rprfm #2, x0, [x0]
// CHECK-ENCODING: [0x1a,0x48,0xa0,0xf8]

prfm #27, [x0, w0, uxtw]
// CHECK-INST: rprfm #3, x0, [x0]
// CHECK-ENCODING: [0x1b,0x48,0xa0,0xf8]

prfm #28, [x0, w0, uxtw]
// CHECK-INST: rprfm pldstrm, x0, [x0]
// CHECK-ENCODING: [0x1c,0x48,0xa0,0xf8]

prfm #29, [x0, w0, uxtw]
// CHECK-INST: rprfm pststrm, x0, [x0]
// CHECK-ENCODING: [0x1d,0x48,0xa0,0xf8]

prfm #30, [x0, w0, uxtw]
// CHECK-INST: rprfm #6, x0, [x0]
// CHECK-ENCODING: [0x1e,0x48,0xa0,0xf8]

prfm #31, [x0, w0, uxtw]
// CHECK-INST: rprfm #7, x0, [x0]
// CHECK-ENCODING: [0x1f,0x48,0xa0,0xf8]

prfm #24, [x0, w0, uxtw #3]
// CHECK-INST: rprfm #8, x0, [x0]
// CHECK-ENCODING: [0x18,0x58,0xa0,0xf8]

prfm #25, [x0, w0, uxtw #3]
// CHECK-INST: rprfm #9, x0, [x0]
// CHECK-ENCODING: [0x19,0x58,0xa0,0xf8]

prfm #26, [x0, w0, uxtw #3]
// CHECK-INST: rprfm #10, x0, [x0]
// CHECK-ENCODING: [0x1a,0x58,0xa0,0xf8]

prfm #27, [x0, w0, uxtw #3]
// CHECK-INST: rprfm #11, x0, [x0]
// CHECK-ENCODING: [0x1b,0x58,0xa0,0xf8]

prfm #28, [x0, w0, uxtw #3]
// CHECK-INST: rprfm #12, x0, [x0]
// CHECK-ENCODING: [0x1c,0x58,0xa0,0xf8]

prfm #29, [x0, w0, uxtw #3]
// CHECK-INST: rprfm #13, x0, [x0]
// CHECK-ENCODING: [0x1d,0x58,0xa0,0xf8]

prfm #30, [x0, w0, uxtw #3]
// CHECK-INST: rprfm #14, x0, [x0]
// CHECK-ENCODING: [0x1e,0x58,0xa0,0xf8]

prfm #31, [x0, w0, uxtw #3]
// CHECK-INST: rprfm #15, x0, [x0]
// CHECK-ENCODING: [0x1f,0x58,0xa0,0xf8]

prfm #24, [x0, x0]
// CHECK-INST: rprfm #16, x0, [x0]
// CHECK-ENCODING: [0x18,0x68,0xa0,0xf8]

prfm #25, [x0, x0]
// CHECK-INST: rprfm #17, x0, [x0]
// CHECK-ENCODING: [0x19,0x68,0xa0,0xf8]

prfm #26, [x0, x0]
// CHECK-INST: rprfm #18, x0, [x0]
// CHECK-ENCODING: [0x1a,0x68,0xa0,0xf8]

prfm #27, [x0, x0]
// CHECK-INST: rprfm #19, x0, [x0]
// CHECK-ENCODING: [0x1b,0x68,0xa0,0xf8]

prfm #28, [x0, x0]
// CHECK-INST: rprfm #20, x0, [x0]
// CHECK-ENCODING: [0x1c,0x68,0xa0,0xf8]

prfm #29, [x0, x0]
// CHECK-INST: rprfm #21, x0, [x0]
// CHECK-ENCODING: [0x1d,0x68,0xa0,0xf8]

prfm #30, [x0, x0]
// CHECK-INST: rprfm #22, x0, [x0]
// CHECK-ENCODING: [0x1e,0x68,0xa0,0xf8]

prfm #31, [x0, x0]
// CHECK-INST: rprfm #23, x0, [x0]
// CHECK-ENCODING: [0x1f,0x68,0xa0,0xf8]

prfm #24, [x0, x0, lsl #3]
// CHECK-INST: rprfm #24, x0, [x0]
// CHECK-ENCODING: [0x18,0x78,0xa0,0xf8]

prfm #25, [x0, x0, lsl #3]
// CHECK-INST: rprfm #25, x0, [x0]
// CHECK-ENCODING: [0x19,0x78,0xa0,0xf8]

prfm #26, [x0, x0, lsl #3]
// CHECK-INST: rprfm #26, x0, [x0]
// CHECK-ENCODING: [0x1a,0x78,0xa0,0xf8]

prfm #27, [x0, x0, lsl #3]
// CHECK-INST: rprfm #27, x0, [x0]
// CHECK-ENCODING: [0x1b,0x78,0xa0,0xf8]

prfm #28, [x0, x0, lsl #3]
// CHECK-INST: rprfm #28, x0, [x0]
// CHECK-ENCODING: [0x1c,0x78,0xa0,0xf8]

prfm #29, [x0, x0, lsl #3]
// CHECK-INST: rprfm #29, x0, [x0]
// CHECK-ENCODING: [0x1d,0x78,0xa0,0xf8]

prfm #30, [x0, x0, lsl #3]
// CHECK-INST: rprfm #30, x0, [x0]
// CHECK-ENCODING: [0x1e,0x78,0xa0,0xf8]

prfm #31, [x0, x0, lsl #3]
// CHECK-INST: rprfm #31, x0, [x0]
// CHECK-ENCODING: [0x1f,0x78,0xa0,0xf8]

prfm #24, [x0, w0, sxtw]
// CHECK-INST: rprfm #32, x0, [x0]
// CHECK-ENCODING: [0x18,0xc8,0xa0,0xf8]

prfm #25, [x0, w0, sxtw]
// CHECK-INST: rprfm #33, x0, [x0]
// CHECK-ENCODING: [0x19,0xc8,0xa0,0xf8]

prfm #26, [x0, w0, sxtw]
// CHECK-INST: rprfm #34, x0, [x0]
// CHECK-ENCODING: [0x1a,0xc8,0xa0,0xf8]

prfm #27, [x0, w0, sxtw]
// CHECK-INST: rprfm #35, x0, [x0]
// CHECK-ENCODING: [0x1b,0xc8,0xa0,0xf8]

prfm #28, [x0, w0, sxtw]
// CHECK-INST: rprfm #36, x0, [x0]
// CHECK-ENCODING: [0x1c,0xc8,0xa0,0xf8]

prfm #29, [x0, w0, sxtw]
// CHECK-INST: rprfm #37, x0, [x0]
// CHECK-ENCODING: [0x1d,0xc8,0xa0,0xf8]

prfm #30, [x0, w0, sxtw]
// CHECK-INST: rprfm #38, x0, [x0]
// CHECK-ENCODING: [0x1e,0xc8,0xa0,0xf8]

prfm #31, [x0, w0, sxtw]
// CHECK-INST: rprfm #39, x0, [x0]
// CHECK-ENCODING: [0x1f,0xc8,0xa0,0xf8]

prfm #24, [x0, w0, sxtw #3]
// CHECK-INST: rprfm #40, x0, [x0]
// CHECK-ENCODING: [0x18,0xd8,0xa0,0xf8]

prfm #25, [x0, w0, sxtw #3]
// CHECK-INST: rprfm #41, x0, [x0]
// CHECK-ENCODING: [0x19,0xd8,0xa0,0xf8]

prfm #26, [x0, w0, sxtw #3]
// CHECK-INST: rprfm #42, x0, [x0]
// CHECK-ENCODING: [0x1a,0xd8,0xa0,0xf8]

prfm #27, [x0, w0, sxtw #3]
// CHECK-INST: rprfm #43, x0, [x0]
// CHECK-ENCODING: [0x1b,0xd8,0xa0,0xf8]

prfm #28, [x0, w0, sxtw #3]
// CHECK-INST: rprfm #44, x0, [x0]
// CHECK-ENCODING: [0x1c,0xd8,0xa0,0xf8]

prfm #29, [x0, w0, sxtw #3]
// CHECK-INST: rprfm #45, x0, [x0]
// CHECK-ENCODING: [0x1d,0xd8,0xa0,0xf8]

prfm #30, [x0, w0, sxtw #3]
// CHECK-INST: rprfm #46, x0, [x0]
// CHECK-ENCODING: [0x1e,0xd8,0xa0,0xf8]

prfm #31, [x0, w0, sxtw #3]
// CHECK-INST: rprfm #47, x0, [x0]
// CHECK-ENCODING: [0x1f,0xd8,0xa0,0xf8]

prfm #24, [x0, x0, sxtx]
// CHECK-INST: rprfm #48, x0, [x0]
// CHECK-ENCODING: [0x18,0xe8,0xa0,0xf8]

prfm #25, [x0, x0, sxtx]
// CHECK-INST: rprfm #49, x0, [x0]
// CHECK-ENCODING: [0x19,0xe8,0xa0,0xf8]

prfm #26, [x0, x0, sxtx]
// CHECK-INST: rprfm #50, x0, [x0]
// CHECK-ENCODING: [0x1a,0xe8,0xa0,0xf8]

prfm #27, [x0, x0, sxtx]
// CHECK-INST: rprfm #51, x0, [x0]
// CHECK-ENCODING: [0x1b,0xe8,0xa0,0xf8]

prfm #28, [x0, x0, sxtx]
// CHECK-INST: rprfm #52, x0, [x0]
// CHECK-ENCODING: [0x1c,0xe8,0xa0,0xf8]

prfm #29, [x0, x0, sxtx]
// CHECK-INST: rprfm #53, x0, [x0]
// CHECK-ENCODING: [0x1d,0xe8,0xa0,0xf8]

prfm #30, [x0, x0, sxtx]
// CHECK-INST: rprfm #54, x0, [x0]
// CHECK-ENCODING: [0x1e,0xe8,0xa0,0xf8]

prfm #31, [x0, x0, sxtx]
// CHECK-INST: rprfm #55, x0, [x0]
// CHECK-ENCODING: [0x1f,0xe8,0xa0,0xf8]

prfm #24, [x0, x0, sxtx #3]
// CHECK-INST: rprfm #56, x0, [x0]
// CHECK-ENCODING: [0x18,0xf8,0xa0,0xf8]

prfm #25, [x0, x0, sxtx #3]
// CHECK-INST: rprfm #57, x0, [x0]
// CHECK-ENCODING: [0x19,0xf8,0xa0,0xf8]

prfm #26, [x0, x0, sxtx #3]
// CHECK-INST: rprfm #58, x0, [x0]
// CHECK-ENCODING: [0x1a,0xf8,0xa0,0xf8]

prfm #27, [x0, x0, sxtx #3]
// CHECK-INST: rprfm #59, x0, [x0]
// CHECK-ENCODING: [0x1b,0xf8,0xa0,0xf8]

prfm #28, [x0, x0, sxtx #3]
// CHECK-INST: rprfm #60, x0, [x0]
// CHECK-ENCODING: [0x1c,0xf8,0xa0,0xf8]

prfm #29, [x0, x0, sxtx #3]
// CHECK-INST: rprfm #61, x0, [x0]
// CHECK-ENCODING: [0x1d,0xf8,0xa0,0xf8]

prfm #30, [x0, x0, sxtx #3]
// CHECK-INST: rprfm #62, x0, [x0]
// CHECK-ENCODING: [0x1e,0xf8,0xa0,0xf8]

prfm #31, [x0, x0, sxtx #3]
// CHECK-INST: rprfm #63, x0, [x0]
// CHECK-ENCODING: [0x1f,0xf8,0xa0,0xf8]

