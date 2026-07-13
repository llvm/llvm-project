# REQUIRES: aarch64-registered-target



// Test for omitting OperandType::OPERAND_SHIFT_MSL

// MOVIv2s_msl: MOVI vd, #imm{, shift}
# RUN: llvm-exegesis --mtriple=aarch64 --mcpu=neoverse-v2 --mode=latency  --benchmark-phase=prepare-and-assemble-snippet --opcode-name=MOVIv4s_msl 2>&1 | FileCheck %s --check-prefix=MOVIv4s_msl_latency
# RUN: llvm-exegesis --mtriple=aarch64 --mcpu=neoverse-v2 --mode=inverse_throughput --benchmark-phase=prepare-and-assemble-snippet --opcode-name=MOVIv4s_msl 2>&1 | FileCheck %s --check-prefix=MOVIv4s_msl_throughput
# MOVIv4s_msl_latency-NOT: Not all operands were initialized by the snippet generator for MOVIv4s_msl opcode

// TODO: Add test to check if the immediate value is correct when serial execution strategy is added for MOVIv4s_msl


# MOVIv4s_msl_throughput-NOT: Not all operands were initialized by the snippet generator for MOVIv4s_msl opcode
# MOVIv4s_msl_throughput: ---
# MOVIv4s_msl_throughput-NEXT: mode: inverse_throughput
# MOVIv4s_msl_throughput-NEXT: key: 
# MOVIv4s_msl_throughput-NEXT:   instructions:
# MOVIv4s_msl_throughput-NEXT:     MOVIv4s_msl [[REG1:Q[0-9]+|LR]] i_0x1 i_0x108
# MOVIv4s_msl_throughput: ...

// MOVIv2s_msl: MOVI vd, #imm{, shift}
# RUN: llvm-exegesis --mtriple=aarch64 --mcpu=neoverse-v2 --mode=latency  --benchmark-phase=prepare-and-assemble-snippet --opcode-name=MOVIv2s_msl 2>&1 | FileCheck %s --check-prefix=MOVIv2s_msl_latency
# RUN: llvm-exegesis --mtriple=aarch64 --mcpu=neoverse-v2 --mode=inverse_throughput --benchmark-phase=prepare-and-assemble-snippet --opcode-name=MOVIv2s_msl 2>&1 | FileCheck %s --check-prefix=MOVIv2s_msl_throughput
# MOVIv2s_msl_latency-NOT: Not all operands were initialized by the snippet generator for MOVIv2s_msl opcode

// TODO: Add test to check if the immediate value is correct when serial execution strategy is added for MOVIv2s_msl


# MOVIv2s_msl_throughput-NOT: Not all operands were initialized by the snippet generator for MOVIv2s_msl opcode
# MOVIv2s_msl_throughput: ---
# MOVIv2s_msl_throughput-NEXT: mode: inverse_throughput
# MOVIv2s_msl_throughput-NEXT: key: 
# MOVIv2s_msl_throughput-NEXT:   instructions:
# MOVIv2s_msl_throughput-NEXT:     MOVIv2s_msl [[REG1:D[0-9]+|LR]] i_0x1 i_0x108
# MOVIv2s_msl_throughput: ...



// Test for omitting OperandType::OPERAND_PCREL
// LDRDl: LDRD ldr1, ldr2, [pc, #imm]
# RUN: llvm-exegesis --mtriple=aarch64 --mcpu=neoverse-v2 --mode=latency  --benchmark-phase=prepare-and-assemble-snippet --opcode-name=LDRDl 2>&1 | FileCheck %s --check-prefix=LDRDl_latency
# RUN: llvm-exegesis --mtriple=aarch64 --mcpu=neoverse-v2 --mode=inverse_throughput --benchmark-phase=prepare-and-assemble-snippet --opcode-name=LDRDl 2>&1 | FileCheck %s --check-prefix=LDRDl_throughput

# LDRDl_latency-NOT: Not all operands were initialized by the snippet generator for LDRDl opcodes
# LDRDl_throughput-NOT: Not all operands were initialized by the snippet generator for LDRDl opcodes

# LDRDl_throughput:      ---
# LDRDl_throughput-NEXT: mode: inverse_throughput
# LDRDl_throughput-NEXT: key:
# LDRDl_throughput-NEXT:   instructions:
# LDRDl_throughput-NEXT:     LDRDl [[REG1:D[0-9]+|LR]] i_0x8
# LDRDl_throughput: ...



// Test for omitting OperandType::OPERAND_IMPLICIT_IMM_0

// UMOVvi16_idx0: UMOV wd, vn.h[index]
# RUN: llvm-exegesis --mtriple=aarch64 --mcpu=neoverse-v2 --mode=latency --benchmark-phase=prepare-and-assemble-snippet --opcode-name=UMOVvi16_idx0 2>&1 | FileCheck %s --check-prefix=UMOVvi16_idx0_latency
# RUN: llvm-exegesis --mtriple=aarch64 --mcpu=neoverse-v2 --mode=inverse_throughput --benchmark-phase=prepare-and-assemble-snippet --opcode-name=UMOVvi16_idx0 2>&1 | FileCheck %s --check-prefix=UMOVvi16_idx0_throughput

# UMOVvi16_idx0_latency-NOT: UMOVvi16_idx0: Not all operands were initialized by the snippet generator for UMOVvi16_idx0 opcode.

# UMOVvi16_idx0_throughput-NOT: UMOVvi16_idx0: Not all operands were initialized by the snippet generator for UMOVvi16_idx0 opcode.
# UMOVvi16_idx0_throughput:      ---
# UMOVvi16_idx0_throughput-NEXT: mode: inverse_throughput
# UMOVvi16_idx0_throughput-NEXT: key:
# UMOVvi16_idx0_throughput-NEXT:   instructions:
# UMOVvi16_idx0_throughput-NEXT:     UMOVvi16_idx0 [[REG1:W[0-9]+|LR]] [[REG2:Q[0-9]+|LR]] i_0x0
# UMOVvi16_idx0_throughput: ...
