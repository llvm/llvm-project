//===---- MachineOutliner.h - Outliner data structures ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Contains all data structures shared between the outliner implemented in
/// MachineOutliner.cpp and target implementations of the outliner.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_MACHINEOUTLINER_H
#define LLVM_MACHINEOUTLINER_H

#include "llvm/CodeGen/MachineFunction.h"

namespace outliner {

using namespace llvm;

/// Represents how an instruction should be mapped by the outliner.
/// \p Legal instructions are those which are safe to outline.
/// \p LegalTerminator instructions are safe to outline, but only as the
/// last instruction in a sequence.
/// \p Illegal instructions are those which cannot be outlined.
/// \p Invisible instructions are instructions which can be outlined, but
/// shouldn't actually impact the outlining result.
enum InstrType { Legal, LegalTerminator, Illegal, Invisible };

/// Describes the number of instructions that it will take to call and
/// construct a frame for a given outlining candidate.
struct TargetCostInfo {
  /// Represents the size of a sequence in bytes. (Some instructions vary
  /// widely in size, so just counting the instructions isn't very useful.)
  unsigned SequenceSize;

  /// Number of instructions to call an outlined function for this candidate.
  unsigned CallOverhead;

  /// Number of instructions to construct an outlined function frame
  /// for this candidate.
  unsigned FrameOverhead;

  /// Represents the specific instructions that must be emitted to
  /// construct a call to this candidate.
  unsigned CallConstructionID;

  /// Represents the specific instructions that must be emitted to
  /// construct a frame for this candidate's outlined function.
  unsigned FrameConstructionID;

  TargetCostInfo() {}
  TargetCostInfo(unsigned SequenceSize, unsigned CallOverhead,
                 unsigned FrameOverhead, unsigned CallConstructionID,
                 unsigned FrameConstructionID)
      : SequenceSize(SequenceSize), CallOverhead(CallOverhead),
        FrameOverhead(FrameOverhead), CallConstructionID(CallConstructionID),
        FrameConstructionID(FrameConstructionID) {}
};

/// An individual sequence of instructions to be replaced with a call to
/// an outlined function.
struct Candidate {
private:
  /// The start index of this \p Candidate in the instruction list.
  unsigned StartIdx;

  /// The number of instructions in this \p Candidate.
  unsigned Len;

  // The first instruction in this \p Candidate.
  MachineBasicBlock::iterator FirstInst;

  // The last instruction in this \p Candidate.
  MachineBasicBlock::iterator LastInst;

  // The basic block that contains this Candidate.
  MachineBasicBlock *MBB;

public:
  /// The index of this \p Candidate's \p OutlinedFunction in the list of
  /// \p OutlinedFunctions.
  unsigned FunctionIdx;

  /// Set to false if the candidate overlapped with another candidate.
  bool InCandidateList = true;

  /// Contains all target-specific information for this \p Candidate.
  TargetCostInfo TCI;

  /// Return the number of instructions in this Candidate.
  unsigned getLength() const { return Len; }

  /// Return the start index of this candidate.
  unsigned getStartIdx() const { return StartIdx; }

  /// Return the end index of this candidate.
  unsigned getEndIdx() const { return StartIdx + Len - 1; }

  MachineBasicBlock::iterator &front() { return FirstInst; }
  MachineBasicBlock::iterator &back() { return LastInst; }
  MachineFunction *getMF() const { return MBB->getParent(); }
  MachineBasicBlock *getMBB() const { return MBB; }

  /// The number of instructions that would be saved by outlining every
  /// candidate of this type.
  ///
  /// This is a fixed value which is not updated during the candidate pruning
  /// process. It is only used for deciding which candidate to keep if two
  /// candidates overlap. The true benefit is stored in the OutlinedFunction
  /// for some given candidate.
  unsigned Benefit = 0;

  Candidate(unsigned StartIdx, unsigned Len,
            MachineBasicBlock::iterator &FirstInst,
            MachineBasicBlock::iterator &LastInst, MachineBasicBlock *MBB,
            unsigned FunctionIdx)
      : StartIdx(StartIdx), Len(Len), FirstInst(FirstInst), LastInst(LastInst),
        MBB(MBB), FunctionIdx(FunctionIdx) {}

  /// Used to ensure that \p Candidates are outlined in an order that
  /// preserves the start and end indices of other \p Candidates.
  bool operator<(const Candidate &RHS) const {
    return getStartIdx() > RHS.getStartIdx();
  }
};

/// The information necessary to create an outlined function for some
/// class of candidate.
struct OutlinedFunction {

private:
  /// The number of candidates for this \p OutlinedFunction.
  unsigned OccurrenceCount = 0;

public:
  std::vector<std::shared_ptr<Candidate>> Candidates;

  /// The actual outlined function created.
  /// This is initialized after we go through and create the actual function.
  MachineFunction *MF = nullptr;

  /// A number assigned to this function which appears at the end of its name.
  unsigned Name;

  /// The sequence of integers corresponding to the instructions in this
  /// function.
  std::vector<unsigned> Sequence;

  /// Contains all target-specific information for this \p OutlinedFunction.
  TargetCostInfo TCI;

  /// Return the number of candidates for this \p OutlinedFunction.
  unsigned getOccurrenceCount() { return OccurrenceCount; }

  /// Decrement the occurrence count of this OutlinedFunction and return the
  /// new count.
  unsigned decrement() {
    assert(OccurrenceCount > 0 && "Can't decrement an empty function!");
    OccurrenceCount--;
    return getOccurrenceCount();
  }

  /// Return the number of bytes it would take to outline this
  /// function.
  unsigned getOutliningCost() {
    return (OccurrenceCount * TCI.CallOverhead) + TCI.SequenceSize +
           TCI.FrameOverhead;
  }

  /// Return the size in bytes of the unoutlined sequences.
  unsigned getNotOutlinedCost() { return OccurrenceCount * TCI.SequenceSize; }

  /// Return the number of instructions that would be saved by outlining
  /// this function.
  unsigned getBenefit() {
    unsigned NotOutlinedCost = getNotOutlinedCost();
    unsigned OutlinedCost = getOutliningCost();
    return (NotOutlinedCost < OutlinedCost) ? 0
                                            : NotOutlinedCost - OutlinedCost;
  }

  OutlinedFunction(unsigned Name, unsigned OccurrenceCount,
                   const std::vector<unsigned> &Sequence, TargetCostInfo &TCI)
      : OccurrenceCount(OccurrenceCount), Name(Name), Sequence(Sequence),
        TCI(TCI) {}
};
} // namespace outliner

#endif
