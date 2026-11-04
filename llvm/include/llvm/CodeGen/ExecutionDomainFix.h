//==-- llvm/CodeGen/ExecutionDomainFix.h - Execution Domain Fix -*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Execution Domain Fix pass.
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

#ifndef LLVM_CODEGEN_EXECUTIONDOMAINFIX_H
#define LLVM_CODEGEN_EXECUTIONDOMAINFIX_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LoopTraversal.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/ReachingDefAnalysis.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class MachineInstr;
class TargetInstrInfo;

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
  /// Basic reference counting.
  unsigned Refs = 0;

  /// Bitmask of available domains. For an open DomainValue, it is the still
  /// possible domains for collapsing. For a collapsed DomainValue it is the
  /// domains where the register is available for free.
  unsigned AvailableDomains;

  /// Pointer to the next DomainValue in a chain.  When two DomainValues are
  /// merged, Victim.Next is set to point to Victor, so old DomainValue
  /// references can be updated by following the chain.
  DomainValue *Next;

  /// Twiddleable instructions using or defining these registers.
  SmallVector<MachineInstr *, 8> Instrs;

  DomainValue() { clear(); }

  /// A collapsed DomainValue has no instructions to twiddle - it simply keeps
  /// track of the domains where the registers are already available.
  bool isCollapsed() const { return Instrs.empty(); }

  /// Is domain available?
  bool hasDomain(unsigned domain) const {
    assert(domain <
               static_cast<unsigned>(std::numeric_limits<unsigned>::digits) &&
           "undefined behavior");
    return AvailableDomains & (1u << domain);
  }

  /// Mark domain as available.
  void addDomain(unsigned domain) {
    assert(domain <
               static_cast<unsigned>(std::numeric_limits<unsigned>::digits) &&
           "undefined behavior");
    AvailableDomains |= 1u << domain;
  }

  // Restrict to a single domain available.
  void setSingleDomain(unsigned domain) {
    assert(domain <
               static_cast<unsigned>(std::numeric_limits<unsigned>::digits) &&
           "undefined behavior");
    AvailableDomains = 1u << domain;
  }

  /// Return bitmask of domains that are available and in mask.
  unsigned getCommonDomains(unsigned mask) const {
    return AvailableDomains & mask;
  }

  /// First domain available.
  unsigned getFirstDomain() const {
    return llvm::countr_zero(AvailableDomains);
  }

  /// Clear this DomainValue and point to next which has all its data.
  void clear() {
    AvailableDomains = 0;
    Next = nullptr;
    Instrs.clear();
  }
};

class LLVM_ABI ExecutionDomainFixLegacy : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid

  ExecutionDomainFixLegacy() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<ReachingDefInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }

  StringRef getPassName() const override { return "Execution Domain Fix"; }
};

class ExecutionDomainFixPass
    : public OptionalPassInfoMixin<ExecutionDomainFixPass> {
public:
  LLVM_ABI PreservedAnalyses run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM);
  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().setNoVRegs();
  }
};

} // namespace llvm

#endif // LLVM_CODEGEN_EXECUTIONDOMAINFIX_H
