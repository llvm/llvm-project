//===- GCNLaneMaskUtils.h ----------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Various utility functions for dealing with lane masks during code
/// generation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNLANEMASKUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_GCNLANEMASKUTILS_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineSSAUpdater.h"

namespace llvm {

class GCNLaneMaskAnalysis;
class MachineFunction;

/// \brief Wavefront-size dependent constants.
struct GCNLaneMaskConstants {
  Register RegExec;                    // EXEC / EXEC_LO
  Register RegVcc;                     // VCC / VCC_LO
  const TargetRegisterClass *RegClass; // SReg_nnRegClass
  unsigned OpMov;                      // S_MOV_Bnn
  unsigned OpMovTerm;                  // S_MOV_Bnn_term
  unsigned OpAnd;                      // S_AND_Bnn
  unsigned OpOr;                       // S_OR_Bnn
  unsigned OpXor;                      // S_XOR_Bnn
  unsigned OpAndN2;                    // S_ANDN2_Bnn
  unsigned OpOrN2;                     // S_ORN2_Bnn
  unsigned OpCSelect;                  // S_CSELECT_Bnn
};

/// \brief Helper class for lane-mask related tasks.
class GCNLaneMaskUtils {
private:
  MachineFunction *MF = nullptr;
  const GCNLaneMaskConstants *Constants = nullptr;

public:
  static const GCNLaneMaskConstants *getConsts(unsigned WavefrontSize);
  static const GCNLaneMaskConstants *getConsts(MachineFunction &MF);

  GCNLaneMaskUtils() = default;
  explicit GCNLaneMaskUtils(MachineFunction &MF) { setFunction(MF); }

  MachineFunction *function() const { return MF; }
  void setFunction(MachineFunction &Func) {
    MF = &Func;
    Constants = getConsts(Func);
  }

  const GCNLaneMaskConstants &consts() const {
    assert(Constants);
    return *Constants;
  }

  bool maybeLaneMask(Register Reg) const;
  bool isConstantLaneMask(Register Reg, bool &Val) const;

  Register createLaneMaskReg() const;
  void buildMergeLaneMasks(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I, const DebugLoc &DL,
                           Register DstReg, Register PrevReg, Register CurReg,
                           GCNLaneMaskAnalysis *LMA = nullptr,
                           bool Accumulating = false) const;
};

/// Lazy analyses of lane masks.
class GCNLaneMaskAnalysis {
private:
  GCNLaneMaskUtils LMU;

  DenseMap<Register, bool> SubsetOfExec;

public:
  GCNLaneMaskAnalysis(MachineFunction &MF) : LMU(MF) {}

  bool isSubsetOfExec(Register Reg, MachineBasicBlock &UseBlock,
                      unsigned RemainingDepth = 5);
};

/// \brief SSA-updater for lane masks.
///
/// The updater operates in one of two modes: "default" and "accumulating".
///
/// Default mode is the analog to regular SSA construction and suitable for the
/// lowering of normal per-lane boolean values to lane masks: the mask can be
/// (re-)written multiple times for each lane. In each basic block, only the
/// lanes enabled by that block's EXEC mask are updated. Bits for lanes that
/// never contributed with an available value are undefined.
///
/// Accumulating mode is used for some aspects of control flow lowering. In
/// this mode, each lane is assumed to provide a "true" available value only
/// once, and to never attempt to change the value back to "false" -- except
/// that all lanes are reset to false in "reset blocks" as explained below.
/// In accumulating mode, the bits for lanes that never contributed with an
/// available value are 0.
///
/// In accumulating mode, all lanes are reset to 0 at certain points in "reset
/// blocks" which are added via \ref addReset. The reset happens in one or both
/// of two modes:
///  - ResetInMiddle: Reset logically happens after the point queried by
///    \ref getValueInMiddleOfBlock and before the contribution of the block's
///    available value ("merge").
///  - ResetAtEnd: Reset logically happens after the contribution of the
///    block's available value, but before the point queried by
///    \ref getValueAtEndOfBlock. Use \ref getValueAfterMerge to query the
///    value just after contribution of the reset block's available value.
///
class GCNLaneMaskUpdater {
public:
  enum ResetFlags {
    ResetInMiddle = (1 << 0),
    ResetAtEnd = (1 << 1),
  };

private:
  GCNLaneMaskUtils LMU;
  GCNLaneMaskAnalysis *LMA = nullptr;
  MachineSSAUpdater SSAUpdater;

  bool Accumulating = false;

  bool Processed = false;

  struct BlockInfo {
    MachineBasicBlock *Block;
    unsigned Flags = 0; // ResetFlags
    Register Value;
    Register Merged;

    explicit BlockInfo(MachineBasicBlock *Block) : Block(Block) {}
  };

  SmallVector<BlockInfo, 4> Blocks;

  Register ZeroReg;
  DenseSet<MachineInstr *> PotentiallyDead;

public:
  GCNLaneMaskUpdater(MachineFunction &MF) : LMU(MF), SSAUpdater(MF) {}

  void setLaneMaskAnalysis(GCNLaneMaskAnalysis *Analysis) { LMA = Analysis; }

  void init(Register Reg);
  void cleanup();

  void setAccumulating(bool Val) { Accumulating = Val; }

  void addReset(MachineBasicBlock &Block, ResetFlags Flags);
  void addAvailable(MachineBasicBlock &Block, Register Value);

  Register getValueInMiddleOfBlock(MachineBasicBlock &Block);
  Register getValueAtEndOfBlock(MachineBasicBlock &Block);
  Register getValueAfterMerge(MachineBasicBlock &Block);

private:
  void process();
  SmallVectorImpl<BlockInfo>::iterator findBlockInfo(MachineBasicBlock &Block);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNLANEMASKUTILS_H
