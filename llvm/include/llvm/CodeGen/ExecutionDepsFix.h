//===- llvm/CodeGen/ExecutionDepsFix.h - Execution Dependency Fix -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file Execution Dependency Fix pass.
///
/// Some X86 SSE instructions like mov, and, or, xor are available in different
/// variants for different operand types. These variant instructions are
/// equivalent, but on Nehalem and newer cpus there is extra latency
/// transferring data between integer and floating point domains.  ARM cores
/// have similar issues when they are configured with both VFP and NEON
/// pipelines.
///
/// This pass changes the variant instructions to minimize domain crossings.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_CODEGEN_EXECUTIONDEPSFIX_H
#define LLVM_CODEGEN_EXECUTIONDEPSFIX_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/Support/Allocator.h"
#include <vector>

namespace llvm {

/// A DomainValue is a bit like LiveIntervals' ValNo, but it also keeps track
/// of execution domains.
///
/// An open DomainValue represents a set of instructions that can still switch
/// execution domain. Multiple registers may refer to the same open
/// DomainValue - they will eventually be collapsed to the same execution
/// domain.
///
/// A collapsed DomainValue represents a single register that has been forced
/// into one of more execution domains. There is a separate collapsed
/// DomainValue for each register, but it may contain multiple execution
/// domains. A register value is initially created in a single execution
/// domain, but if we were forced to pay the penalty of a domain crossing, we
/// keep track of the fact that the register is now available in multiple
/// domains.
struct DomainValue {
  // Basic reference counting.
  unsigned Refs;

  // Bitmask of available domains. For an open DomainValue, it is the still
  // possible domains for collapsing. For a collapsed DomainValue it is the
  // domains where the register is available for free.
  unsigned AvailableDomains;

  // Pointer to the next DomainValue in a chain.  When two DomainValues are
  // merged, Victim.Next is set to point to Victor, so old DomainValue
  // references can be updated by following the chain.
  DomainValue *Next;

  // Twiddleable instructions using or defining these registers.
  SmallVector<MachineInstr*, 8> Instrs;

  // A collapsed DomainValue has no instructions to twiddle - it simply keeps
  // track of the domains where the registers are already available.
  bool isCollapsed() const { return Instrs.empty(); }

  // Is domain available?
  bool hasDomain(unsigned domain) const {
    assert(domain <
               static_cast<unsigned>(std::numeric_limits<unsigned>::digits) &&
           "undefined behavior");
    return AvailableDomains & (1u << domain);
  }

  // Mark domain as available.
  void addDomain(unsigned domain) {
    AvailableDomains |= 1u << domain;
  }

  // Restrict to a single domain available.
  void setSingleDomain(unsigned domain) {
    AvailableDomains = 1u << domain;
  }

  // Return bitmask of domains that are available and in mask.
  unsigned getCommonDomains(unsigned mask) const {
    return AvailableDomains & mask;
  }

  // First domain available.
  unsigned getFirstDomain() const {
    return countTrailingZeros(AvailableDomains);
  }

  DomainValue() : Refs(0) { clear(); }

  // Clear this DomainValue and point to next which has all its data.
  void clear() {
    AvailableDomains = 0;
    Next = nullptr;
    Instrs.clear();
  }
};

/// Information about a live register.
struct LiveReg {
  /// Value currently in this register, or NULL when no value is being tracked.
  /// This counts as a DomainValue reference.
  DomainValue *Value;

  /// Instruction that defined this register, relative to the beginning of the
  /// current basic block.  When a LiveReg is used to represent a live-out
  /// register, this value is relative to the end of the basic block, so it
  /// will be a negative number.
  int Def;
};

class ExecutionDepsFix : public MachineFunctionPass {
  SpecificBumpPtrAllocator<DomainValue> Allocator;
  SmallVector<DomainValue*,16> Avail;

  const TargetRegisterClass *const RC;
  MachineFunction *MF;
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  RegisterClassInfo RegClassInfo;
  std::vector<SmallVector<int, 1>> AliasMap;
  const unsigned NumRegs;
  LiveReg *LiveRegs;
  struct MBBInfo {
    // Keeps clearance and domain information for all registers. Note that this
    // is different from the usual definition notion of liveness. The CPU
    // doesn't care whether or not we consider a register killed.
    LiveReg *OutRegs;

    // Whether we have gotten to this block in primary processing yet.
    bool PrimaryCompleted;

    // The number of predecessors for which primary processing has completed
    unsigned IncomingProcessed;

    // The value of `IncomingProcessed` at the start of primary processing
    unsigned PrimaryIncoming;

    // The number of predecessors for which all processing steps are done.
    unsigned IncomingCompleted;

    MBBInfo()
        : OutRegs(nullptr), PrimaryCompleted(false), IncomingProcessed(0),
          PrimaryIncoming(0), IncomingCompleted(0) {}
  };
  typedef DenseMap<MachineBasicBlock *, MBBInfo> MBBInfoMap;
  MBBInfoMap MBBInfos;

  /// List of undefined register reads in this block in forward order.
  std::vector<std::pair<MachineInstr*, unsigned> > UndefReads;

  /// Storage for register unit liveness.
  LivePhysRegs LiveRegSet;

  /// Current instruction number.
  /// The first instruction in each basic block is 0.
  int CurInstr;
public:
  ExecutionDepsFix(char &PassID, const TargetRegisterClass &RC)
    : MachineFunctionPass(PassID), RC(&RC), NumRegs(RC.getNumRegs()) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

private:
  iterator_range<SmallVectorImpl<int>::const_iterator>
  regIndices(unsigned Reg) const;
  // DomainValue allocation.
  DomainValue *alloc(int domain = -1);
  DomainValue *retain(DomainValue *DV) {
    if (DV) ++DV->Refs;
    return DV;
  }
  void release(DomainValue*);
  DomainValue *resolve(DomainValue*&);

  // LiveRegs manipulations.
  void setLiveReg(int rx, DomainValue *DV);
  void kill(int rx);
  void force(int rx, unsigned domain);
  void collapse(DomainValue *dv, unsigned domain);
  bool merge(DomainValue *A, DomainValue *B);

  void enterBasicBlock(MachineBasicBlock*);
  void leaveBasicBlock(MachineBasicBlock*);
  bool isBlockDone(MachineBasicBlock *);
  void processBasicBlock(MachineBasicBlock *MBB, bool PrimaryPass);
  void updateSuccessors(MachineBasicBlock *MBB, bool PrimaryPass);
  bool visitInstr(MachineInstr *);
  void processDefs(MachineInstr *, bool breakDependency, bool Kill);
  void visitSoftInstr(MachineInstr*, unsigned mask);
  void visitHardInstr(MachineInstr*, unsigned domain);
  bool pickBestRegisterForUndef(MachineInstr *MI, unsigned OpIdx,
                                unsigned Pref);
  bool shouldBreakDependence(MachineInstr*, unsigned OpIdx, unsigned Pref);
  void processUndefReads(MachineBasicBlock*);
};

} // end namepsace llvm

#endif
