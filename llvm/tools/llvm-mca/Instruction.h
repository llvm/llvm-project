//===--------------------- Instruction.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines abstractions used by the Backend to model register reads,
/// register writes and instructions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_INSTRUCTION_H
#define LLVM_TOOLS_LLVM_MCA_INSTRUCTION_H

#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <set>
#include <vector>

namespace mca {

struct WriteDescriptor;
struct ReadDescriptor;
class WriteState;
class ReadState;

constexpr int UNKNOWN_CYCLES = -512;

class Instruction;

/// An InstRef contains both a SourceMgr index and Instruction pair.  The index
/// is used as a unique identifier for the instruction.  MCA will make use of
/// this index as a key throughout MCA.
class InstRef : public std::pair<unsigned, Instruction *> {
public:
  InstRef() : std::pair<unsigned, Instruction *>(0, nullptr) {}
  InstRef(unsigned Index, Instruction *I)
      : std::pair<unsigned, Instruction *>(Index, I) {}

  unsigned getSourceIndex() const { return first; }
  Instruction *getInstruction() { return second; }
  const Instruction *getInstruction() const { return second; }

  /// Returns true if  this InstRef has been populated.
  bool isValid() const { return second != nullptr; }

#ifndef NDEBUG
  void print(llvm::raw_ostream &OS) const { OS << getSourceIndex(); }
#endif
};

#ifndef NDEBUG
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const InstRef &IR) {
  IR.print(OS);
  return OS;
}
#endif

/// A register write descriptor.
struct WriteDescriptor {
  // Operand index. -1 if this is an implicit write.
  int OpIndex;
  // Write latency. Number of cycles before write-back stage.
  int Latency;
  // This field is set to a value different than zero only if this
  // is an implicit definition.
  unsigned RegisterID;
  // True if this write generates a partial update of a super-registers.
  // On X86, this flag is set by byte/word writes on GPR registers. Also,
  // a write of an XMM register only partially updates the corresponding
  // YMM super-register if the write is associated to a legacy SSE instruction.
  bool FullyUpdatesSuperRegs;
  // Instruction itineraries would set this field to the SchedClass ID.
  // Otherwise, it defaults to the WriteResourceID from the MCWriteLatencyEntry
  // element associated to this write.
  // When computing read latencies, this value is matched against the
  // "ReadAdvance" information. The hardware backend may implement
  // dedicated forwarding paths to quickly propagate write results to dependent
  // instructions waiting in the reservation station (effectively bypassing the
  // write-back stage).
  unsigned SClassOrWriteResourceID;
  // True only if this is a write obtained from an optional definition.
  // Optional definitions are allowed to reference regID zero (i.e. "no
  // register").
  bool IsOptionalDef;
};

/// A register read descriptor.
struct ReadDescriptor {
  // A MCOperand index. This is used by the Dispatch logic to identify register
  // reads. This field defaults to -1 if this is an implicit read.
  int OpIndex;
  // The actual "UseIdx". This is used to query the ReadAdvance table. Explicit
  // uses always come first in the sequence of uses.
  unsigned UseIndex;
  // This field is only set if this is an implicit read.
  unsigned RegisterID;
  // Scheduling Class Index. It is used to query the scheduling model for the
  // MCSchedClassDesc object.
  unsigned SchedClassID;
  // True if there may be a local forwarding logic in hardware to serve a
  // write used by this read. This information, along with SchedClassID, is
  // used to dynamically check at Instruction creation time, if the input
  // operands can benefit from a ReadAdvance bonus.
  bool HasReadAdvanceEntries;
};

/// Tracks uses of a register definition (e.g. register write).
///
/// Each implicit/explicit register write is associated with an instance of
/// this class. A WriteState object tracks the dependent users of a
/// register write. It also tracks how many cycles are left before the write
/// back stage.
class WriteState {
  const WriteDescriptor &WD;
  // On instruction issue, this field is set equal to the write latency.
  // Before instruction issue, this field defaults to -512, a special
  // value that represents an "unknown" number of cycles.
  int CyclesLeft;

  // Actual register defined by this write. This field is only used
  // to speedup queries on the register file.
  // For implicit writes, this field always matches the value of
  // field RegisterID from WD.
  unsigned RegisterID;

  // A list of dependent reads. Users is a set of dependent
  // reads. A dependent read is added to the set only if CyclesLeft
  // is "unknown". As soon as CyclesLeft is 'known', each user in the set
  // gets notified with the actual CyclesLeft.

  // The 'second' element of a pair is a "ReadAdvance" number of cycles.
  std::set<std::pair<ReadState *, int>> Users;

public:
  WriteState(const WriteDescriptor &Desc, unsigned RegID)
      : WD(Desc), CyclesLeft(UNKNOWN_CYCLES), RegisterID(RegID) {}
  WriteState(const WriteState &Other) = delete;
  WriteState &operator=(const WriteState &Other) = delete;

  int getCyclesLeft() const { return CyclesLeft; }
  unsigned getWriteResourceID() const { return WD.SClassOrWriteResourceID; }
  unsigned getRegisterID() const { return RegisterID; }

  void addUser(ReadState *Use, int ReadAdvance);
  bool fullyUpdatesSuperRegs() const { return WD.FullyUpdatesSuperRegs; }

  // On every cycle, update CyclesLeft and notify dependent users.
  void cycleEvent();
  void onInstructionIssued();

#ifndef NDEBUG
  void dump() const;
#endif
};

/// Tracks register operand latency in cycles.
///
/// A read may be dependent on more than one write. This occurs when some
/// writes only partially update the register associated to this read.
class ReadState {
  const ReadDescriptor &RD;
  unsigned RegisterID;
  unsigned DependentWrites;
  int CyclesLeft;
  unsigned TotalCycles;

public:
  bool isReady() const {
    if (DependentWrites)
      return false;
    return (CyclesLeft == UNKNOWN_CYCLES || CyclesLeft == 0);
  }

  ReadState(const ReadDescriptor &Desc, unsigned RegID)
      : RD(Desc), RegisterID(RegID), DependentWrites(0),
        CyclesLeft(UNKNOWN_CYCLES), TotalCycles(0) {}
  ReadState(const ReadState &Other) = delete;
  ReadState &operator=(const ReadState &Other) = delete;

  const ReadDescriptor &getDescriptor() const { return RD; }
  unsigned getSchedClass() const { return RD.SchedClassID; }
  unsigned getRegisterID() const { return RegisterID; }
  void cycleEvent();
  void writeStartEvent(unsigned Cycles);
  void setDependentWrites(unsigned Writes) { DependentWrites = Writes; }
};

/// A sequence of cycles.
///
/// This class can be used as a building block to construct ranges of cycles.
class CycleSegment {
  unsigned Begin; // Inclusive.
  unsigned End;   // Exclusive.
  bool Reserved;  // Resources associated to this segment must be reserved.

public:
  CycleSegment(unsigned StartCycle, unsigned EndCycle, bool IsReserved = false)
      : Begin(StartCycle), End(EndCycle), Reserved(IsReserved) {}

  bool contains(unsigned Cycle) const { return Cycle >= Begin && Cycle < End; }
  bool startsAfter(const CycleSegment &CS) const { return End <= CS.Begin; }
  bool endsBefore(const CycleSegment &CS) const { return Begin >= CS.End; }
  bool overlaps(const CycleSegment &CS) const {
    return !startsAfter(CS) && !endsBefore(CS);
  }
  bool isExecuting() const { return Begin == 0 && End != 0; }
  bool isExecuted() const { return End == 0; }
  bool operator<(const CycleSegment &Other) const {
    return Begin < Other.Begin;
  }
  CycleSegment &operator--(void) {
    if (Begin)
      Begin--;
    if (End)
      End--;
    return *this;
  }

  bool isValid() const { return Begin <= End; }
  unsigned size() const { return End - Begin; };
  void Subtract(unsigned Cycles) {
    assert(End >= Cycles);
    End -= Cycles;
  }

  unsigned begin() const { return Begin; }
  unsigned end() const { return End; }
  void setEnd(unsigned NewEnd) { End = NewEnd; }
  bool isReserved() const { return Reserved; }
  void setReserved() { Reserved = true; }
};

/// Helper used by class InstrDesc to describe how hardware resources
/// are used.
///
/// This class describes how many resource units of a specific resource kind
/// (and how many cycles) are "used" by an instruction.
struct ResourceUsage {
  CycleSegment CS;
  unsigned NumUnits;
  ResourceUsage(CycleSegment Cycles, unsigned Units = 1)
      : CS(Cycles), NumUnits(Units) {}
  unsigned size() const { return CS.size(); }
  bool isReserved() const { return CS.isReserved(); }
  void setReserved() { CS.setReserved(); }
};

/// An instruction descriptor
struct InstrDesc {
  std::vector<WriteDescriptor> Writes; // Implicit writes are at the end.
  std::vector<ReadDescriptor> Reads;   // Implicit reads are at the end.

  // For every resource used by an instruction of this kind, this vector
  // reports the number of "consumed cycles".
  std::vector<std::pair<uint64_t, ResourceUsage>> Resources;

  // A list of buffered resources consumed by this instruction.
  std::vector<uint64_t> Buffers;
  unsigned MaxLatency;
  // Number of MicroOps for this instruction.
  unsigned NumMicroOps;

  bool MayLoad;
  bool MayStore;
  bool HasSideEffects;

  // A zero latency instruction doesn't consume any scheduler resources.
  bool isZeroLatency() const { return !MaxLatency && Resources.empty(); }
};

/// An instruction dispatched to the out-of-order backend.
///
/// This class is used to monitor changes in the internal state of instructions
/// that are dispatched by the DispatchUnit to the hardware schedulers.
class Instruction {
  const InstrDesc &Desc;

  enum InstrStage {
    IS_INVALID,   // Instruction in an invalid state.
    IS_AVAILABLE, // Instruction dispatched but operands are not ready.
    IS_READY,     // Instruction dispatched and operands ready.
    IS_EXECUTING, // Instruction issued.
    IS_EXECUTED,  // Instruction executed. Values are written back.
    IS_RETIRED    // Instruction retired.
  };

  // The current instruction stage.
  enum InstrStage Stage;

  // This value defaults to the instruction latency. This instruction is
  // considered executed when field CyclesLeft goes to zero.
  int CyclesLeft;

  // Retire Unit token ID for this instruction.
  unsigned RCUTokenID;

  using UniqueDef = std::unique_ptr<WriteState>;
  using UniqueUse = std::unique_ptr<ReadState>;
  using VecDefs = std::vector<UniqueDef>;
  using VecUses = std::vector<UniqueUse>;

  // Output dependencies.
  // One entry per each implicit and explicit register definition.
  VecDefs Defs;

  // Input dependencies.
  // One entry per each implicit and explicit register use.
  VecUses Uses;

public:
  Instruction(const InstrDesc &D)
      : Desc(D), Stage(IS_INVALID), CyclesLeft(-1) {}
  Instruction(const Instruction &Other) = delete;
  Instruction &operator=(const Instruction &Other) = delete;

  VecDefs &getDefs() { return Defs; }
  const VecDefs &getDefs() const { return Defs; }
  VecUses &getUses() { return Uses; }
  const VecUses &getUses() const { return Uses; }
  const InstrDesc &getDesc() const { return Desc; }
  unsigned getRCUTokenID() const { return RCUTokenID; }

  // Transition to the dispatch stage, and assign a RCUToken to this
  // instruction. The RCUToken is used to track the completion of every
  // register write performed by this instruction.
  void dispatch(unsigned RCUTokenID);

  // Instruction issued. Transition to the IS_EXECUTING state, and update
  // all the definitions.
  void execute();

  // Force a transition from the IS_AVAILABLE state to the IS_READY state if
  // input operands are all ready. State transitions normally occur at the
  // beginning of a new cycle (see method cycleEvent()). However, the scheduler
  // may decide to promote instructions from the wait queue to the ready queue
  // as the result of another issue event.  This method is called every time the
  // instruction might have changed in state.
  void update();

  bool isDispatched() const { return Stage == IS_AVAILABLE; }
  bool isReady() const { return Stage == IS_READY; }
  bool isExecuting() const { return Stage == IS_EXECUTING; }
  bool isExecuted() const { return Stage == IS_EXECUTED; }
  bool isRetired() const { return Stage == IS_RETIRED; }

  void retire() {
    assert(isExecuted() && "Instruction is in an invalid state!");
    Stage = IS_RETIRED;
  }

  void cycleEvent();
};
} // namespace mca

#endif
