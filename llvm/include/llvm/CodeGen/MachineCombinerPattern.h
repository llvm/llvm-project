//===-- llvm/CodeGen/MachineCombinerPattern.h - Instruction pattern supported by
// combiner  ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  MULSUBXI_OP1
};

} // end namespace llvm

#endif
