//===----------------- HexagonLiveVariables.h ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Hexagon Live Variable Analysis
// This file implements the Hexagon specific LiveVariables analysis pass.
// 1. Computes the live variables by analyzing the use-defs.
//      - The use-def specifiers are 'assumed' to be correct for each operand.
// 2. Re-calculates the MBB numbers to that they are in sequence.
// TODO: Mark dead instructions.
// TODO: Provide APIs like the target independent Liveness Analysis so that
//       other passes can reuse the liveness information.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_LIVEVARIABLES_H
#define HEXAGON_LIVEVARIABLES_H

#include "Hexagon.h"
#include "HexagonInstrInfo.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include <algorithm>
#include <cstdint>
#include <list>

class HexagonLiveVariablesImpl;

namespace llvm {

typedef std::pair<BitVector, BitVector> UseDef_t; // (Use, Def)
typedef DenseMap<MachineBasicBlock *, UseDef_t> MBBUseDef_t;
typedef DenseMap<const MachineInstr *, UseDef_t> MIUseDef_t;

// List of intervals [From, To).
typedef std::list<std::pair<int64_t, int64_t>> IntervalList_t;
// Intervals stored in indexed form.
typedef SmallVector<IntervalList_t, 0> IndexedLiveIntervals_t;

class HexagonLiveVariables : public MachineFunctionPass {
public:
  typedef MachineBasicBlock::const_instr_iterator MICInstIterType;

  static char ID; // Pass identification, replacement for typeid
  bool HLVComplete;
  HexagonLiveVariables();

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  StringRef getPassName() const override {
    return "Hexagon Live Variables Analysis";
  }

  /// recalculate - recalculates the liveness from scratch. It is like
  /// calling the runOnMachineFunction.
  void recalculate(MachineFunction &MF);

  /// updateLocalLiveness - update only kill flags of operands.
  /// Assumes that global liveness is correct.
  bool updateLocalLiveness(MachineFunction &Fn);

  /// updateLocalLiveness - update only kill flags of operands in MBB.
  /// Assumes that global liveness is correct.
  /// This is useful when a local transformation modifies MIs,
  /// which only changes the local liveness.
  bool updateLocalLiveness(MachineBasicBlock *MBB, bool updateBundle);

  /// incrementalUpdate - update the liveness when \p MIDelta is moved from
  /// \p From to \p To.
  /// @note: This is extremely fragile now. It 'assumes' that the other
  /// successor(s) of \p To do not use Defs of MIDelta.
  bool incrementalUpdate(MICInstIterType MIDelta, MachineBasicBlock *From,
                         MachineBasicBlock *To);
  // addNewMI - update internal data-structures of Live Variable Analysis.
  void addNewMI(MachineInstr *MI, MachineBasicBlock *MBB);

  /// addNewMBB - inform the LiveVariable Analysis that new MBB has been added.
  /// update the liveness of this new MBB.
  /// @note MBB should be empty. If we want to add an MI, add it after calling
  /// this function.
  void addNewMBB(MachineBasicBlock *MBB);

  /// @brief Constructs use-defs of \p MBB by analyzing each MachineOperand.
  /// Collects relevant information so that global liveness can be updated.
  void constructUseDef(MachineBasicBlock *MBB);

  bool isLiveOut(const MachineBasicBlock *MBB, unsigned Reg) const;
  const BitVector &getLiveOuts(const MachineBasicBlock *MBB) const;

  // Returns true when \p Reg is used within [MIBegin, MIEnd)
  // @note: MIBegin and MIEnd should be from same MBB
  // @note: It returns just the first use found in the range.
  // The Use is closest to MIEnd.
  // Takes care of aliases as well.
  bool
  isUsedWithin(MICInstIterType MIBegin, MICInstIterType MIEnd, unsigned Reg,
               MICInstIterType &Use,
               SmallPtrSet<MachineInstr *, 2> *ExceptionsList = nullptr) const;
  // Returns true when \p Reg id defined within [MIBegin, MIEnd)
  // @note: MIBegin and MIEnd should be from same MBB
  // The Def is closest to MIEnd.
  // Takes care of aliases as well.
  bool isDefinedWithin(MICInstIterType MIBegin, MICInstIterType MIEnd,
                       unsigned Reg, MICInstIterType &Def) const;
  bool isDefLiveIn(const MachineInstr *MI, const MachineBasicBlock *MBB) const;
  MBBUseDef_t &getMBBUseDefs();
  MIUseDef_t &getMIUseDefs();

  /// Returns the linear distance (as per layout) of \p MI from the Function.
  /// \p BufferPerMBB is to allow some room for .falign (if added later).
  unsigned getDistanceBetween(const MachineBasicBlock *From,
                              const MachineBasicBlock *To,
                              unsigned BufferPerMBB = HEXAGON_INSTR_SIZE) const;

  // recalculate the distance map.
  void regenerateDistanceMap(const MachineFunction &Fn);

private:
  std::unique_ptr<HexagonLiveVariablesImpl> HLV;
};

} // namespace llvm

#endif
