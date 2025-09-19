//===-- llvm/MC/MCInstrInfo.h - Target Instruction Info ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the target machine instruction set.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCINSTRINFO_H
#define LLVM_MC_MCINSTRINFO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/Compiler.h"
#include <cassert>

namespace llvm {

class MCSubtargetInfo;

//---------------------------------------------------------------------------
/// Interface to description of machine instruction set.
class MCInstrInfo {
public:
  using ComplexDeprecationPredicate = bool (*)(MCInst &,
                                               const MCSubtargetInfo &,
                                               std::string &);

private:
  const MCInstrDesc *LastDesc;      // Raw array to allow static init'n
  const unsigned *InstrNameIndices; // Array for name indices in InstrNameData
  const char *InstrNameData;        // Instruction name string pool
  // Subtarget feature that an instruction is deprecated on, if any
  // -1 implies this is not deprecated by any single feature. It may still be
  // deprecated due to a "complex" reason, below.
  const uint8_t *DeprecatedFeatures;
  // A complex method to determine if a certain instruction is deprecated or
  // not, and return the reason for deprecation.
  const ComplexDeprecationPredicate *ComplexDeprecationInfos;
  unsigned NumOpcodes;              // Number of entries in the desc array

protected:
  // Pointer to 2d array [NumHwModes][NumRegClassByHwModes]
  const int16_t *RegClassByHwModeTables;
  int16_t NumRegClassByHwModes;

public:
  /// Initialize MCInstrInfo, called by TableGen auto-generated routines.
  /// *DO NOT USE*.
  void InitMCInstrInfo(const MCInstrDesc *D, const unsigned *NI, const char *ND,
                       const uint8_t *DF,
                       const ComplexDeprecationPredicate *CDI, unsigned NO,
                       const int16_t *RCHWTables = nullptr,
                       int16_t NumRegClassByHwMode = 0) {
    LastDesc = D + NO - 1;
    InstrNameIndices = NI;
    InstrNameData = ND;
    DeprecatedFeatures = DF;
    ComplexDeprecationInfos = CDI;
    NumOpcodes = NO;
    RegClassByHwModeTables = RCHWTables;
    NumRegClassByHwModes = NumRegClassByHwMode;
  }

  unsigned getNumOpcodes() const { return NumOpcodes; }

  const int16_t *getRegClassByHwModeTable(unsigned ModeId) const {
    assert(RegClassByHwModeTables && NumRegClassByHwModes != 0 &&
           "MCInstrInfo not properly initialized");
    return &RegClassByHwModeTables[ModeId * NumRegClassByHwModes];
  }

  /// Return the ID of the register class to use for \p OpInfo, for the active
  /// HwMode \p HwModeId. In general TargetInstrInfo's version which is already
  /// specialized to the subtarget should be used.
  int16_t getOpRegClassID(const MCOperandInfo &OpInfo,
                          unsigned HwModeId) const {
    int16_t RegClass = OpInfo.RegClass;
    if (OpInfo.isLookupRegClassByHwMode())
      RegClass = getRegClassByHwModeTable(HwModeId)[RegClass];
    return RegClass;
  }

  /// Return the machine instruction descriptor that corresponds to the
  /// specified instruction opcode.
  const MCInstrDesc &get(unsigned Opcode) const {
    assert(Opcode < NumOpcodes && "Invalid opcode!");
    // The table is indexed backwards from the last entry.
    return *(LastDesc - Opcode);
  }

  /// Returns the name for the instructions with the given opcode.
  StringRef getName(unsigned Opcode) const {
    assert(Opcode < NumOpcodes && "Invalid opcode!");
    return StringRef(&InstrNameData[InstrNameIndices[Opcode]]);
  }

  /// Returns true if a certain instruction is deprecated and if so
  /// returns the reason in \p Info.
  LLVM_ABI bool getDeprecatedInfo(MCInst &MI, const MCSubtargetInfo &STI,
                                  std::string &Info) const;
};

} // End llvm namespace

#endif
