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

/// The combiner's goal may differ based on which pattern it is attempting
/// to optimize.
enum class CombinerObjective {
  MustReduceDepth,            // The data dependency chain must be improved.
  MustReduceRegisterPressure, // The register pressure must be reduced.
  Default                     // The critical path must not be lengthened.
};

/// These are instruction patterns matched by the machine combiner pass.
enum MachineCombinerPattern : unsigned {
  // These are commutative variants for reassociating a computation chain. See
  // the comments before getMachineCombinerPatterns() in TargetInstrInfo.cpp.
  REASSOC_AX_BY,
  REASSOC_AX_YB,
  REASSOC_XA_BY,
  REASSOC_XA_YB,

  TARGET_PATTERN_START
  // SystemZ patterns.  (EXPERIMENTAL)
  FMA2_P1P0,
  FMA2_P0P1,
  FMA2,
  FMA1_Add_L,
  FMA1_Add_R,
  FMA3,     // These are inspired by PPC
  FMA2_Add, //

};

} // end namespace llvm

#endif
