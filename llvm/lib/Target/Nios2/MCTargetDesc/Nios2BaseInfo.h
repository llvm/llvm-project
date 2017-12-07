//===-- Nios2BaseInfo.h - Top level definitions for NIOS2 MC ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the Nios2 target useful for the compiler back-end and the MC libraries.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_NIOS2_MCTARGETDESC_NIOS2BASEINFO_H
#define LLVM_LIB_TARGET_NIOS2_MCTARGETDESC_NIOS2BASEINFO_H

namespace llvm {

/// Nios2FG - This namespace holds all of the target specific flags that
/// instruction info tracks.
namespace Nios2FG {
/// Target Operand Flag enum.
enum TOF {
  //===------------------------------------------------------------------===//
  // Nios2 Specific MachineOperand flags.

  MO_NO_FLAG,

  /// MO_ABS_HI/LO - Represents the hi or low part of an absolute symbol
  /// address.
  MO_ABS_HI,
  MO_ABS_LO,

};
} // namespace Nios2FG
} // namespace llvm

#endif
