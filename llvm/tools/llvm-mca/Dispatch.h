//===----------------------- Dispatch.h -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements classes that are used to model register files,
/// reorder buffers and the hardware dispatch logic.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_DISPATCH_H
#define LLVM_TOOLS_LLVM_MCA_DISPATCH_H

#include "Instruction.h"
#include "llvm/MC/MCRegisterInfo.h"
#include <map>

namespace mca {

class WriteState;
class DispatchUnit;
class Scheduler;
class Backend;

/// \brief Keeps track of register definitions.
///
/// This class tracks register definitions, and performs register renaming
/// to break anti dependencies.
/// By default, there is no limit in the number of register aliases which
/// can be created for the purpose of register renaming. However, users can
/// specify at object construction time a limit in the number of temporary
/// registers which can be used by the register renaming logic.
class RegisterFile {
  const llvm::MCRegisterInfo &MRI;
  // Currently used mappings and maximum used mappings.
  // These are to generate statistics only.
  unsigned NumUsedMappings;
  unsigned MaxUsedMappings;
  // Total number of mappings created over time.
  unsigned TotalMappingsCreated;

  // The maximum number of register aliases which can be used by the
  // register renamer. Defaut value for this field is zero.
  // A value of zero for this field means that there is no limit in the
  // amount of register mappings which can be created. That is equivalent
  // to having a theoretically infinite number of temporary registers.
  unsigned TotalMappings;

  // This map contains an entry for every physical register.
  // A register index is used as a key value to access a WriteState.
  // This is how we track RAW dependencies for dispatched
  // instructions. For every register, we track the last seen write only.
  // This assumes that all writes fully update both super and sub registers.
  // We need a flag in MCInstrDesc to check if a write also updates super
  // registers. We can then have a extra tablegen flag to set for instructions.
  // This is a separate patch on its own.
  std::vector<WriteState *> RegisterMappings;
  // Assumptions are:
  //  a) a false dependencies is always removed by the register renamer.
  //  b) the register renamer can create an "infinite" number of mappings.
  // Since we track the number of mappings created, in future we may
  // introduce constraints on the number of mappings that can be created.
  // For example, the maximum number of registers that are available for
  // register renaming purposes may default to the size of the register file.

  // In future, we can extend this design to allow multiple register files, and
  // apply different restrictions on the register mappings and the number of
  // temporary registers used by mappings.

public:
  RegisterFile(const llvm::MCRegisterInfo &mri, unsigned Mappings = 0)
      : MRI(mri), NumUsedMappings(0), MaxUsedMappings(0),
        TotalMappingsCreated(0), TotalMappings(Mappings),
        RegisterMappings(MRI.getNumRegs(), nullptr) {}

  // Creates a new register mapping for RegID.
  // This reserves a temporary register in the register file.
  void addRegisterMapping(WriteState &WS);

  // Invalidates register mappings associated to the input WriteState object.
  // This releases temporary registers in the register file.
  void invalidateRegisterMapping(const WriteState &WS);

  bool isAvailable(unsigned NumRegWrites);
  void collectWrites(llvm::SmallVectorImpl<WriteState *> &Writes,
                     unsigned RegID) const;
  void updateOnRead(ReadState &RS, unsigned RegID);
  unsigned getMaxUsedRegisterMappings() const { return MaxUsedMappings; }
  unsigned getTotalRegisterMappingsCreated() const {
    return TotalMappingsCreated;
  }

#ifndef NDEBUG
  void dump() const;
#endif
};

/// \brief tracks which instructions are in-flight (i.e. dispatched but not
/// retired) in the OoO backend.
///
/// This class checks on every cycle if/which instructions can be retired.
/// Instructions are retired in program order.
/// In the event of instruction retired, the DispatchUnit object that owns
/// this RetireControlUnit gets notified.
/// On instruction retired, register updates are all architecturally
/// committed, and any temporary registers originally allocated for the
/// retired instruction are freed.
struct RetireControlUnit {
  // A "token" (object of class RUToken) is created by the retire unit for every
  // instruction dispatched to the schedulers.  Flag 'Executed' is used to
  // quickly check if an instruction has reached the write-back stage.  A token
  // also carries information related to the number of entries consumed by the
  // instruction in the reorder buffer. The idea is that those entries will
  // become available again once the instruction is retired.  On every cycle,
  // the RCU (Retire Control Unit) scans every token starting to search for
  // instructions that are ready to retire.  retired. Instructions are retired
  // in program order. Only 'Executed' instructions are eligible for retire.
  // Note that the size of the reorder buffer is defined by the scheduling model
  // via field 'NumMicroOpBufferSize'.
  struct RUToken {
    unsigned Index;    // Instruction index.
    unsigned NumSlots; // Slots reserved to this instruction.
    bool Executed;     // True if the instruction is past the WB stage.
  };

private:
  unsigned NextAvailableSlotIdx;
  unsigned CurrentInstructionSlotIdx;
  unsigned AvailableSlots;
  unsigned MaxRetirePerCycle; // 0 means no limit.
  std::vector<RUToken> Queue;
  DispatchUnit *Owner;

public:
  RetireControlUnit(unsigned NumSlots, unsigned RPC, DispatchUnit *DU)
      : NextAvailableSlotIdx(0), CurrentInstructionSlotIdx(0),
        AvailableSlots(NumSlots), MaxRetirePerCycle(RPC), Owner(DU) {
    assert(NumSlots && "Expected at least one slot!");
    Queue.resize(NumSlots);
  }

  bool isFull() const { return !AvailableSlots; }
  bool isEmpty() const { return AvailableSlots == Queue.size(); }
  bool isAvailable(unsigned Quantity = 1) const {
    // Some instructions may declare a number of uOps which exceedes the size
    // of the reorder buffer. To avoid problems, cap the amount of slots to
    // the size of the reorder buffer.
    Quantity = std::min(Quantity, static_cast<unsigned>(Queue.size()));
    return AvailableSlots >= Quantity;
  }

  // Reserves a number of slots, and returns a new token.
  unsigned reserveSlot(unsigned Index, unsigned NumMicroOps);

  /// Retires instructions in program order.
  void cycleEvent();

  void onInstructionExecuted(unsigned TokenID);

#ifndef NDEBUG
  void dump() const;
#endif
};

// \brief Implements the hardware dispatch logic.
//
// This class is responsible for the dispatch stage, in which instructions are
// dispatched in groups to the Scheduler.  An instruction can be dispatched if
// functional units are available.
// To be more specific, an instruction can be dispatched to the Scheduler if:
//  1) There are enough entries in the reorder buffer (implemented by class
//     RetireControlUnit) to accomodate all opcodes.
//  2) There are enough temporaries to rename output register operands.
//  3) There are enough entries available in the used buffered resource(s).
//
// The number of micro opcodes that can be dispatched in one cycle is limited by
// the value of field 'DispatchWidth'. A "dynamic dispatch stall" occurs when
// processor resources are not available (i.e. at least one of the
// abovementioned checks fails). Dispatch stall events are counted during the
// entire execution of the code, and displayed by the performance report when
// flag '-verbose' is specified.
//
// If the number of micro opcodes of an instruction is bigger than
// DispatchWidth, then it can only be dispatched at the beginning of one cycle.
// The DispatchUnit will still have to wait for a number of cycles (depending on
// the DispatchWidth and the number of micro opcodes) before it can serve other
// instructions.
class DispatchUnit {
  unsigned DispatchWidth;
  unsigned AvailableEntries;
  unsigned CarryOver;
  Scheduler *SC;

  std::unique_ptr<RegisterFile> RAT;
  std::unique_ptr<RetireControlUnit> RCU;
  Backend *Owner;

  /// Dispatch stall event identifiers.
  ///
  /// The naming convention is:
  /// * Event names starts with the "DS_" prefix
  /// * For dynamic dispatch stalls, the "DS_" prefix is followed by the
  ///   the unavailable resource/functional unit acronym (example: RAT)
  /// * The last substring is the event reason (example: REG_UNAVAILABLE means
  ///   that register renaming couldn't find enough spare registers in the
  ///   register file).
  ///
  /// List of acronyms used for processor resoures:
  /// RAT - Register Alias Table (used by the register renaming logic)
  /// RCU - Retire Control Unit
  /// SQ  - Scheduler's Queue
  /// LDQ - Load Queue
  /// STQ - Store Queue
  enum {
    DS_RAT_REG_UNAVAILABLE,
    DS_RCU_TOKEN_UNAVAILABLE,
    DS_SQ_TOKEN_UNAVAILABLE,
    DS_LDQ_TOKEN_UNAVAILABLE,
    DS_STQ_TOKEN_UNAVAILABLE,
    DS_DISPATCH_GROUP_RESTRICTION,
    DS_LAST
  };

  // The DispatchUnit track dispatch stall events caused by unavailable
  // of hardware resources. Events are classified based on the stall kind;
  // so we have a counter for every source of dispatch stall. Counters are
  // stored into a vector `DispatchStall` which is always of size DS_LAST.
  std::vector<unsigned> DispatchStalls;

  bool checkRAT(const InstrDesc &Desc);
  bool checkRCU(const InstrDesc &Desc);
  bool checkScheduler(const InstrDesc &Desc);

  void notifyInstructionDispatched(unsigned IID);

public:
  DispatchUnit(Backend *B, const llvm::MCRegisterInfo &MRI,
               unsigned MicroOpBufferSize, unsigned RegisterFileSize,
               unsigned MaxRetirePerCycle, unsigned MaxDispatchWidth,
               Scheduler *Sched)
      : DispatchWidth(MaxDispatchWidth), AvailableEntries(MaxDispatchWidth),
        CarryOver(0U), SC(Sched),
        RAT(llvm::make_unique<RegisterFile>(MRI, RegisterFileSize)),
        RCU(llvm::make_unique<RetireControlUnit>(MicroOpBufferSize,
                                                 MaxRetirePerCycle, this)),
        Owner(B), DispatchStalls(DS_LAST, 0) {}

  unsigned getDispatchWidth() const { return DispatchWidth; }

  bool isAvailable(unsigned NumEntries) const {
    return NumEntries <= AvailableEntries || AvailableEntries == DispatchWidth;
  }

  bool isRCUEmpty() const { return RCU->isEmpty(); }

  bool canDispatch(const InstrDesc &Desc) {
    assert(isAvailable(Desc.NumMicroOps));
    return checkRCU(Desc) && checkRAT(Desc) && checkScheduler(Desc);
  }

  unsigned dispatch(unsigned IID, Instruction *NewInst);

  void collectWrites(llvm::SmallVectorImpl<WriteState *> &Vec,
                     unsigned RegID) const {
    return RAT->collectWrites(Vec, RegID);
  }
  unsigned getNumRATStalls() const {
    return DispatchStalls[DS_RAT_REG_UNAVAILABLE];
  }
  unsigned getNumRCUStalls() const {
    return DispatchStalls[DS_RCU_TOKEN_UNAVAILABLE];
  }
  unsigned getNumSQStalls() const {
    return DispatchStalls[DS_SQ_TOKEN_UNAVAILABLE];
  }
  unsigned getNumLDQStalls() const {
    return DispatchStalls[DS_LDQ_TOKEN_UNAVAILABLE];
  }
  unsigned getNumSTQStalls() const {
    return DispatchStalls[DS_STQ_TOKEN_UNAVAILABLE];
  }
  unsigned getNumDispatchGroupStalls() const {
    return DispatchStalls[DS_DISPATCH_GROUP_RESTRICTION];
  }
  unsigned getMaxUsedRegisterMappings() const {
    return RAT->getMaxUsedRegisterMappings();
  }
  unsigned getTotalRegisterMappingsCreated() const {
    return RAT->getTotalRegisterMappingsCreated();
  }
  void addNewRegisterMapping(WriteState &WS) { RAT->addRegisterMapping(WS); }

  void cycleEvent(unsigned Cycle) {
    RCU->cycleEvent();
    AvailableEntries =
        CarryOver >= DispatchWidth ? 0 : DispatchWidth - CarryOver;
    CarryOver = CarryOver >= DispatchWidth ? CarryOver - DispatchWidth : 0U;
  }

  void notifyInstructionRetired(unsigned Index);

  void onInstructionExecuted(unsigned TokenID) {
    RCU->onInstructionExecuted(TokenID);
  }

  void invalidateRegisterMappings(const Instruction &Inst);
#ifndef NDEBUG
  void dump() const;
#endif
};

} // namespace mca

#endif
