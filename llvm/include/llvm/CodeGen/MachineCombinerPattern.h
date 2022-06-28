//===-- llvm/CodeGen/MachineCombinerPattern.h - Instruction pattern supported by
// combiner  ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines instruction pattern supported by combiner
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINECOMBINERPATTERN_H
#define LLVM_CODEGEN_MACHINECOMBINERPATTERN_H

namespace llvm {

/// These are instruction patterns matched by the machine combiner pass.
enum class MachineCombinerPattern {
  // These are commutative variants for reassociating a computation chain. See
  // the comments before getMachineCombinerPatterns() in TargetInstrInfo.cpp.
  REASSOC_AX_BY,
  REASSOC_AX_YB,
  REASSOC_XA_BY,
  REASSOC_XA_YB,

  // These are patterns matched by the PowerPC to reassociate FMA chains.
  REASSOC_XY_AMM_BMM,
  REASSOC_XMM_AMM_BMM,

  // These are patterns matched by the PowerPC to reassociate FMA and FSUB to
  // reduce register pressure.
  REASSOC_XY_BCA,
  REASSOC_XY_BAC,

  // These are patterns used to reduce the length of dependence chain.
  SUBADD_OP1,
  SUBADD_OP2,

  // These are multiply-add patterns matched by the AArch64 machine combiner.
  MULADDW_OP1,
  MULADDW_OP2,
  MULSUBW_OP1,
  MULSUBW_OP2,
  MULADDWI_OP1,
  MULSUBWI_OP1,
  MULADDX_OP1,
  MULADDX_OP2,
  MULSUBX_OP1,
  MULSUBX_OP2,
  MULADDXI_OP1,
  MULSUBXI_OP1,
  // NEON integers vectors
  MULADDv8i8_OP1,
  MULADDv8i8_OP2,
  MULADDv16i8_OP1,
  MULADDv16i8_OP2,
  MULADDv4i16_OP1,
  MULADDv4i16_OP2,
  MULADDv8i16_OP1,
  MULADDv8i16_OP2,
  MULADDv2i32_OP1,
  MULADDv2i32_OP2,
  MULADDv4i32_OP1,
  MULADDv4i32_OP2,

  MULSUBv8i8_OP1,
  MULSUBv8i8_OP2,
  MULSUBv16i8_OP1,
  MULSUBv16i8_OP2,
  MULSUBv4i16_OP1,
  MULSUBv4i16_OP2,
  MULSUBv8i16_OP1,
  MULSUBv8i16_OP2,
  MULSUBv2i32_OP1,
  MULSUBv2i32_OP2,
  MULSUBv4i32_OP1,
  MULSUBv4i32_OP2,

  MULADDv4i16_indexed_OP1,
  MULADDv4i16_indexed_OP2,
  MULADDv8i16_indexed_OP1,
  MULADDv8i16_indexed_OP2,
  MULADDv2i32_indexed_OP1,
  MULADDv2i32_indexed_OP2,
  MULADDv4i32_indexed_OP1,
  MULADDv4i32_indexed_OP2,

  MULSUBv4i16_indexed_OP1,
  MULSUBv4i16_indexed_OP2,
  MULSUBv8i16_indexed_OP1,
  MULSUBv8i16_indexed_OP2,
  MULSUBv2i32_indexed_OP1,
  MULSUBv2i32_indexed_OP2,
  MULSUBv4i32_indexed_OP1,
  MULSUBv4i32_indexed_OP2,

  // Floating Point
  FMULADDH_OP1,
  FMULADDH_OP2,
  FMULSUBH_OP1,
  FMULSUBH_OP2,
  FMULADDS_OP1,
  FMULADDS_OP2,
  FMULSUBS_OP1,
  FMULSUBS_OP2,
  FMULADDD_OP1,
  FMULADDD_OP2,
  FMULSUBD_OP1,
  FMULSUBD_OP2,
  FNMULSUBH_OP1,
  FNMULSUBS_OP1,
  FNMULSUBD_OP1,
  FMLAv1i32_indexed_OP1,
  FMLAv1i32_indexed_OP2,
  FMLAv1i64_indexed_OP1,
  FMLAv1i64_indexed_OP2,
  FMLAv4f16_OP1,
  FMLAv4f16_OP2,
  FMLAv8f16_OP1,
  FMLAv8f16_OP2,
  FMLAv2f32_OP2,
  FMLAv2f32_OP1,
  FMLAv2f64_OP1,
  FMLAv2f64_OP2,
  FMLAv4i16_indexed_OP1,
  FMLAv4i16_indexed_OP2,
  FMLAv8i16_indexed_OP1,
  FMLAv8i16_indexed_OP2,
  FMLAv2i32_indexed_OP1,
  FMLAv2i32_indexed_OP2,
  FMLAv2i64_indexed_OP1,
  FMLAv2i64_indexed_OP2,
  FMLAv4f32_OP1,
  FMLAv4f32_OP2,
  FMLAv4i32_indexed_OP1,
  FMLAv4i32_indexed_OP2,
  FMLSv1i32_indexed_OP2,
  FMLSv1i64_indexed_OP2,
  FMLSv4f16_OP1,
  FMLSv4f16_OP2,
  FMLSv8f16_OP1,
  FMLSv8f16_OP2,
  FMLSv2f32_OP1,
  FMLSv2f32_OP2,
  FMLSv2f64_OP1,
  FMLSv2f64_OP2,
  FMLSv4i16_indexed_OP1,
  FMLSv4i16_indexed_OP2,
  FMLSv8i16_indexed_OP1,
  FMLSv8i16_indexed_OP2,
  FMLSv2i32_indexed_OP1,
  FMLSv2i32_indexed_OP2,
  FMLSv2i64_indexed_OP1,
  FMLSv2i64_indexed_OP2,
  FMLSv4f32_OP1,
  FMLSv4f32_OP2,
  FMLSv4i32_indexed_OP1,
  FMLSv4i32_indexed_OP2,

  FMULv2i32_indexed_OP1,
  FMULv2i32_indexed_OP2,
  FMULv2i64_indexed_OP1,
  FMULv2i64_indexed_OP2,
  FMULv4i16_indexed_OP1,
  FMULv4i16_indexed_OP2,
  FMULv4i32_indexed_OP1,
  FMULv4i32_indexed_OP2,
  FMULv8i16_indexed_OP1,
  FMULv8i16_indexed_OP2,
};

} // end namespace llvm

#endif
