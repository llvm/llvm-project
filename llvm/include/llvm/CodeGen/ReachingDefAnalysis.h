//==--- llvm/CodeGen/ReachingDefAnalysis.h - Reaching Def Analysis -*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Reaching Defs Analysis pass.
///
/// This pass tracks for each instruction what is the "closest" reaching def of
/// a given register. It is used by BreakFalseDeps (for clearance calculation)
/// and ExecutionDomainFix (for arbitrating conflicting domains).
///
/// Note that this is different from the usual definition notion of liveness.
/// The CPU doesn't care whether or not we consider a register killed.
///
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REACHINGDEFANALYSIS_H
#define LLVM_CODEGEN_REACHINGDEFANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LoopTraversal.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

namespace llvm {

class MachineBasicBlock;
class MachineInstr;

// An implementation of multimap from (MBBNumber, Unit) to reaching definitions.
//
// This implementation only supports modification operations just enough
// to serve our needs:
//
// - addDef
// - prependDef
// - replaceFront
//
// Internally, the multimap is implemented as a collection of singly linked
// lists represented on top of a single array.  Each singly-linked list
// contains reaching definitions for a given pair of MBBNumber and Unit.
//
// This design has the following highlights:
//
// - Unlike SparseMultiset or other maps, we do not store keys as part of values
//   or anywhere else in the data structure.
//
// - The single array design minimizes malloc traffic.
//
// - Reaching definitions share one array.  This means that if one pair of
//   (MBBNumber, Unit) has multiple reaching definitions while another pair of
//   (MBBNumber, Unit) has none, they cancel each other to some extent.
class MBBReachingDefsInfo {
public:
  MBBReachingDefsInfo() = default;
  MBBReachingDefsInfo(const MBBReachingDefsInfo &) = delete;
  MBBReachingDefsInfo &operator=(const MBBReachingDefsInfo &) = delete;

  // Initialize the multimap with the number of basic blocks and the number of
  // register units.
  void init(unsigned BBs, unsigned Regs) {
    assert(NumBlockIDs == 0 && "can initialize only once");
    assert(NumRegUnits == 0 && "can initialize only once");
    assert(Storage.empty() && "can initialize only once");
    NumBlockIDs = BBs;
    NumRegUnits = Regs;
    unsigned NumIndexes = NumBlockIDs * NumRegUnits;
    // Reserve space for reaching definitions.  Note that the first NumIndexes
    // elements are used for indexes to various chains.  The second half
    // accommodates up to one reaching def per (MBBNumber, Unit) pair on
    // average.
    Storage.reserve(NumIndexes * 2);
    Storage.assign(NumIndexes, std::make_pair(0, 0));
  }

  // Clear the entire data structure.
  void clear() {
    NumBlockIDs = 0;
    NumRegUnits = 0;
    Storage.clear();
  }

  // Add a reaching definition Def to the end of the singly-linked list of
  // definitions for (MBBNumber, Unit).
  void addDef(unsigned MBBNumber, unsigned Unit, int Def) {
    unsigned Key = computeKey(MBBNumber, Unit);
    unsigned NewIndex = Storage.size();
    Storage.emplace_back(Def, 0);
    if (Storage[Key].first == 0) {
      // Update the index of the first element.
      Storage[Key].first = NewIndex;
      // Update the index of the last element.
      Storage[Key].second = NewIndex;
    } else {
      unsigned OldLastPos = Storage[Key].second;
      // The old last element now points to the new element.
      Storage[OldLastPos].second = NewIndex;
      // Update the index of the last element.
      Storage[Key].second = NewIndex;
    }
  }

  // Add a reaching definition Def to the beginning of the singly-linked list of
  // definitions for (MBBNumber, Unit).
  void prependDef(unsigned MBBNumber, unsigned Unit, int Def) {
    unsigned Key = computeKey(MBBNumber, Unit);
    unsigned NewIndex = Storage.size();
    Storage.emplace_back(Def, 0);
    if (Storage[Key].first == 0) {
      // Update the index of the first element.
      Storage[Key].first = NewIndex;
      // Update the index of the last element.
      Storage[Key].second = NewIndex;
    } else {
      // The new element now points to the old first element.
      Storage[NewIndex].second = Storage[Key].first;
      // Update the index of the first element.
      Storage[Key].first = NewIndex;
    }
  }

  // Replace the definition at the beginning of the singly-linked list of
  // definitions for (MBBNumber, Unit).
  void replaceFront(unsigned MBBNumber, unsigned Unit, int Def) {
    unsigned Key = computeKey(MBBNumber, Unit);
    assert(Storage[Key].first != 0);
    assert(Storage[Key].second != 0);
    unsigned FirstPos = Storage[Key].first;
    Storage[FirstPos].first = Def;
  }

  class def_iterator {
    ArrayRef<std::pair<int, int>> Storage;
    unsigned Pos;

  public:
    def_iterator(ArrayRef<std::pair<int, int>> Storage, unsigned Pos)
        : Storage(Storage), Pos(Pos) {}
    int operator*() { return Storage[Pos].first; }
    void operator++() { Pos = Storage[Pos].second; }
    bool operator==(const def_iterator &RHS) const {
      return Storage == RHS.Storage && Pos == RHS.Pos;
    }
    bool operator!=(const def_iterator &RHS) const { return !operator==(RHS); }
  };

  def_iterator def_begin(unsigned MBBNumber, unsigned Unit) const {
    unsigned Key = computeKey(MBBNumber, Unit);
    return {Storage, static_cast<unsigned>(Storage[Key].first)};
  }
  def_iterator def_end() const { return {Storage, 0}; }
  iterator_range<def_iterator> defs(unsigned MBBNumber, unsigned Unit) const {
    return llvm::make_range(def_begin(MBBNumber, Unit), def_end());
  }

private:
  // The number of reg units.
  unsigned NumRegUnits = 0;

  // The number of basic blocks.
  unsigned NumBlockIDs = 0;

  // The storage for definitions and various indexes.  The array has two parts:
  //
  // The first NumBlockIDs * NumRegUnits elements represent array indexes to
  // reaching definitions for all possible pairs of MBBNumber and Unit.  Each
  // pair represents the first and last index of a corresponding chain.  If the
  // chain is empty, both values are zero.
  //
  // The subsequent elements represent reaching definitions and indexes to their
  // next elements.  In each pair, the first is the reaching def, and the second
  // is the index to the next element.  The index is zero for the last element
  // of the chain.
  std::vector<std::pair<int, int>> Storage;

  unsigned computeKey(unsigned MBBNumber, unsigned Unit) const {
    assert(MBBNumber < NumBlockIDs);
    assert(Unit < NumRegUnits);
    return MBBNumber * NumRegUnits + Unit;
  }
};

/// This class provides the reaching def analysis.
class ReachingDefAnalysis : public MachineFunctionPass {
private:
  MachineFunction *MF = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  LoopTraversal::TraversalOrder TraversedMBBOrder;
  unsigned NumRegUnits = 0;
  unsigned NumBlockIDs = 0;
  /// Instruction that defined each register, relative to the beginning of the
  /// current basic block.  When a LiveRegsDefInfo is used to represent a
  /// live-out register, this value is relative to the end of the basic block,
  /// so it will be a negative number.
  using LiveRegsDefInfo = std::vector<int>;
  LiveRegsDefInfo LiveRegs;

  /// Keeps clearance information for all registers. Note that this
  /// is different from the usual definition notion of liveness. The CPU
  /// doesn't care whether or not we consider a register killed.
  using OutRegsInfoMap = SmallVector<LiveRegsDefInfo, 4>;
  OutRegsInfoMap MBBOutRegsInfos;

  /// Current instruction number.
  /// The first instruction in each basic block is 0.
  int CurInstr = -1;

  /// Maps instructions to their instruction Ids, relative to the beginning of
  /// their basic blocks.
  DenseMap<MachineInstr *, int> InstIds;

  /// All reaching defs of all reg units for a all MBBs
  MBBReachingDefsInfo MBBReachingDefs;

  /// Default values are 'nothing happened a long time ago'.
  const int ReachingDefDefaultVal = -(1 << 21);

  using InstSet = SmallPtrSetImpl<MachineInstr*>;
  using BlockSet = SmallPtrSetImpl<MachineBasicBlock*>;

public:
  static char ID; // Pass identification, replacement for typeid

  ReachingDefAnalysis() : MachineFunctionPass(ID) {
    initializeReachingDefAnalysisPass(*PassRegistry::getPassRegistry());
  }
  void releaseMemory() override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs).set(
          MachineFunctionProperties::Property::TracksLiveness);
  }

  /// Re-run the analysis.
  void reset();

  /// Initialize data structures.
  void init();

  /// Traverse the machine function, mapping definitions.
  void traverse();

  /// Provides the instruction id of the closest reaching def instruction of
  /// PhysReg that reaches MI, relative to the begining of MI's basic block.
  int getReachingDef(MachineInstr *MI, MCRegister PhysReg) const;

  /// Return whether A and B use the same def of PhysReg.
  bool hasSameReachingDef(MachineInstr *A, MachineInstr *B,
                          MCRegister PhysReg) const;

  /// Return whether the reaching def for MI also is live out of its parent
  /// block.
  bool isReachingDefLiveOut(MachineInstr *MI, MCRegister PhysReg) const;

  /// Return the local MI that produces the live out value for PhysReg, or
  /// nullptr for a non-live out or non-local def.
  MachineInstr *getLocalLiveOutMIDef(MachineBasicBlock *MBB,
                                     MCRegister PhysReg) const;

  /// If a single MachineInstr creates the reaching definition, then return it.
  /// Otherwise return null.
  MachineInstr *getUniqueReachingMIDef(MachineInstr *MI,
                                       MCRegister PhysReg) const;

  /// If a single MachineInstr creates the reaching definition, for MIs operand
  /// at Idx, then return it. Otherwise return null.
  MachineInstr *getMIOperand(MachineInstr *MI, unsigned Idx) const;

  /// If a single MachineInstr creates the reaching definition, for MIs MO,
  /// then return it. Otherwise return null.
  MachineInstr *getMIOperand(MachineInstr *MI, MachineOperand &MO) const;

  /// Provide whether the register has been defined in the same basic block as,
  /// and before, MI.
  bool hasLocalDefBefore(MachineInstr *MI, MCRegister PhysReg) const;

  /// Return whether the given register is used after MI, whether it's a local
  /// use or a live out.
  bool isRegUsedAfter(MachineInstr *MI, MCRegister PhysReg) const;

  /// Return whether the given register is defined after MI.
  bool isRegDefinedAfter(MachineInstr *MI, MCRegister PhysReg) const;

  /// Provides the clearance - the number of instructions since the closest
  /// reaching def instuction of PhysReg that reaches MI.
  int getClearance(MachineInstr *MI, MCRegister PhysReg) const;

  /// Provides the uses, in the same block as MI, of register that MI defines.
  /// This does not consider live-outs.
  void getReachingLocalUses(MachineInstr *MI, MCRegister PhysReg,
                            InstSet &Uses) const;

  /// Search MBB for a definition of PhysReg and insert it into Defs. If no
  /// definition is found, recursively search the predecessor blocks for them.
  void getLiveOuts(MachineBasicBlock *MBB, MCRegister PhysReg, InstSet &Defs,
                   BlockSet &VisitedBBs) const;
  void getLiveOuts(MachineBasicBlock *MBB, MCRegister PhysReg,
                   InstSet &Defs) const;

  /// For the given block, collect the instructions that use the live-in
  /// value of the provided register. Return whether the value is still
  /// live on exit.
  bool getLiveInUses(MachineBasicBlock *MBB, MCRegister PhysReg,
                     InstSet &Uses) const;

  /// Collect the users of the value stored in PhysReg, which is defined
  /// by MI.
  void getGlobalUses(MachineInstr *MI, MCRegister PhysReg, InstSet &Uses) const;

  /// Collect all possible definitions of the value stored in PhysReg, which is
  /// used by MI.
  void getGlobalReachingDefs(MachineInstr *MI, MCRegister PhysReg,
                             InstSet &Defs) const;

  /// Return whether From can be moved forwards to just before To.
  bool isSafeToMoveForwards(MachineInstr *From, MachineInstr *To) const;

  /// Return whether From can be moved backwards to just after To.
  bool isSafeToMoveBackwards(MachineInstr *From, MachineInstr *To) const;

  /// Assuming MI is dead, recursively search the incoming operands which are
  /// killed by MI and collect those that would become dead.
  void collectKilledOperands(MachineInstr *MI, InstSet &Dead) const;

  /// Return whether removing this instruction will have no effect on the
  /// program, returning the redundant use-def chain.
  bool isSafeToRemove(MachineInstr *MI, InstSet &ToRemove) const;

  /// Return whether removing this instruction will have no effect on the
  /// program, ignoring the possible effects on some instructions, returning
  /// the redundant use-def chain.
  bool isSafeToRemove(MachineInstr *MI, InstSet &ToRemove,
                      InstSet &Ignore) const;

  /// Return whether a MachineInstr could be inserted at MI and safely define
  /// the given register without affecting the program.
  bool isSafeToDefRegAt(MachineInstr *MI, MCRegister PhysReg) const;

  /// Return whether a MachineInstr could be inserted at MI and safely define
  /// the given register without affecting the program, ignoring any effects
  /// on the provided instructions.
  bool isSafeToDefRegAt(MachineInstr *MI, MCRegister PhysReg,
                        InstSet &Ignore) const;

private:
  /// Set up LiveRegs by merging predecessor live-out values.
  void enterBasicBlock(MachineBasicBlock *MBB);

  /// Update live-out values.
  void leaveBasicBlock(MachineBasicBlock *MBB);

  /// Process he given basic block.
  void processBasicBlock(const LoopTraversal::TraversedMBBInfo &TraversedMBB);

  /// Process block that is part of a loop again.
  void reprocessBasicBlock(MachineBasicBlock *MBB);

  /// Update def-ages for registers defined by MI.
  /// Also break dependencies on partial defs and undef uses.
  void processDefs(MachineInstr *);

  /// Utility function for isSafeToMoveForwards/Backwards.
  template<typename Iterator>
  bool isSafeToMove(MachineInstr *From, MachineInstr *To) const;

  /// Return whether removing this instruction will have no effect on the
  /// program, ignoring the possible effects on some instructions, returning
  /// the redundant use-def chain.
  bool isSafeToRemove(MachineInstr *MI, InstSet &Visited,
                      InstSet &ToRemove, InstSet &Ignore) const;

  /// Provides the MI, from the given block, corresponding to the Id or a
  /// nullptr if the id does not refer to the block.
  MachineInstr *getInstFromId(MachineBasicBlock *MBB, int InstId) const;

  /// Provides the instruction of the closest reaching def instruction of
  /// PhysReg that reaches MI, relative to the begining of MI's basic block.
  MachineInstr *getReachingLocalMIDef(MachineInstr *MI,
                                      MCRegister PhysReg) const;
};

} // namespace llvm

#endif // LLVM_CODEGEN_REACHINGDEFANALYSIS_H
