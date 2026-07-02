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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/Compiler.h"
#include <atomic>
#include <cassert>
#include <cstdint>

namespace llvm {

class MCSubtargetInfo;

/// Lazily decodes a front-coded instruction name table.
///
/// Each block stores its decoded byte size followed by one-byte common-prefix
/// and suffix lengths and the suffix bytes for each name.
///
/// Names are decoded in small opcode blocks. Most users never request
/// instruction names, so they avoid both the decode cost and the memory for
/// unused blocks.
class MCInstrNameTable {
  ArrayRef<uint8_t> CompressedData;
  ArrayRef<uint32_t> BlockOffsets;
  std::atomic<const uint16_t *> *DecodedBlocks;
  unsigned NumNames;

  LLVM_ABI const uint16_t *decodeBlock(unsigned Block) const;

public:
  static constexpr unsigned BlockSize = 128;

  constexpr MCInstrNameTable(ArrayRef<uint8_t> CompressedData,
                             ArrayRef<uint32_t> BlockOffsets,
                             std::atomic<const uint16_t *> *DecodedBlocks,
                             unsigned NumNames)
      : CompressedData(CompressedData), BlockOffsets(BlockOffsets),
        DecodedBlocks(DecodedBlocks), NumNames(NumNames) {}

  StringRef getName(unsigned Opcode) const {
    assert(Opcode < NumNames && "Invalid opcode!");
    unsigned Block = Opcode / BlockSize;
    const uint16_t *Indices =
        DecodedBlocks[Block].load(std::memory_order_acquire);
    if (!Indices)
      Indices = decodeBlock(Block);

    unsigned Remaining = NumNames - Block * BlockSize;
    unsigned NumBlockNames = Remaining < BlockSize ? Remaining : BlockSize;
    const uint8_t *Lengths =
        reinterpret_cast<const uint8_t *>(Indices + NumBlockNames);
    const char *Data = reinterpret_cast<const char *>(Lengths + NumBlockNames);
    unsigned Index = Opcode % BlockSize;
    return StringRef(Data + Indices[Index], Lengths[Index]);
  }
};

//---------------------------------------------------------------------------
/// Interface to description of machine instruction set.
class MCInstrInfo {
public:
  using ComplexDeprecationPredicate = bool (*)(MCInst &,
                                               const MCSubtargetInfo &,
                                               std::string &);

private:
  const MCInstrDesc *LastDesc; // Raw array to allow static init'n
  // Instruction name indices, or an MCInstrNameTable when InstrNameData is
  // null.
  const void *InstrNameIndices;
  const char *InstrNameData; // Instruction name string pool
  // Subtarget feature that an instruction is deprecated on, if any
  // -1 implies this is not deprecated by any single feature. It may still be
  // deprecated due to a "complex" reason, below.
  const uint8_t *DeprecatedFeatures;
  // A complex method to determine if a certain instruction is deprecated or
  // not, and return the reason for deprecation.
  const ComplexDeprecationPredicate *ComplexDeprecationInfos;
  unsigned NumOpcodes; // Number of entries in the desc array

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

  /// Initialize MCInstrInfo with a compressed instruction name table.
  /// *DO NOT USE*.
  void InitMCInstrInfo(const MCInstrDesc *D, const MCInstrNameTable *Names,
                       const uint8_t *DF,
                       const ComplexDeprecationPredicate *CDI, unsigned NO,
                       const int16_t *RCHWTables = nullptr,
                       int16_t NumRegClassByHwMode = 0) {
    LastDesc = D + NO - 1;
    InstrNameIndices = Names;
    InstrNameData = nullptr;
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
    if (!InstrNameData)
      return static_cast<const MCInstrNameTable *>(InstrNameIndices)
          ->getName(Opcode);
    return StringRef(&InstrNameData[static_cast<const unsigned *>(
        InstrNameIndices)[Opcode]]);
  }

  /// Returns true if a certain instruction is deprecated and if so
  /// returns the reason in \p Info.
  LLVM_ABI bool getDeprecatedInfo(MCInst &MI, const MCSubtargetInfo &STI,
                                  std::string &Info) const;
};

} // namespace llvm

#endif
