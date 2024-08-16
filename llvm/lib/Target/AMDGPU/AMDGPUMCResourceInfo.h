//===- AMDGPUMCResourceInfo.h ----- MC Resource Info --------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief MC infrastructure to propagate the function level resource usage
/// info.
///
//===----------------------------------------------------------------------===//

#include "AMDGPUResourceUsageAnalysis.h"
#include "MCTargetDesc/AMDGPUMCExpr.h"

namespace llvm {

class MCContext;
class MCSymbol;
class StringRef;
class MachineFunction;

class MCResourceInfo {
public:
  enum ResourceInfoKind {
    RIK_NumVGPR,
    RIK_NumAGPR,
    RIK_NumSGPR,
    RIK_PrivateSegSize,
    RIK_UsesVCC,
    RIK_UsesFlatScratch,
    RIK_HasDynSizedStack,
    RIK_HasRecursion,
    RIK_HasIndirectCall
  };

private:
  int32_t MaxVGPR = 0;
  int32_t MaxAGPR = 0;
  int32_t MaxSGPR = 0;

  MCContext &OutContext;
  bool finalized;

  void assignResourceInfoExpr(int64_t localValue, ResourceInfoKind RIK,
                              AMDGPUMCExpr::VariantKind Kind,
                              const MachineFunction &MF,
                              const SmallVectorImpl<const Function *> &Callees);

  // Assigns expression for Max S/V/A-GPRs to the referenced symbols.
  void assignMaxRegs();

public:
  MCResourceInfo(MCContext &OutContext)
      : OutContext(OutContext), finalized(false) {}
  void addMaxVGPRCandidate(int32_t candidate) {
    MaxVGPR = std::max(MaxVGPR, candidate);
  }
  void addMaxAGPRCandidate(int32_t candidate) {
    MaxAGPR = std::max(MaxAGPR, candidate);
  }
  void addMaxSGPRCandidate(int32_t candidate) {
    MaxSGPR = std::max(MaxSGPR, candidate);
  }

  MCSymbol *getSymbol(StringRef FuncName, ResourceInfoKind RIK);
  const MCExpr *getSymRefExpr(StringRef FuncName, ResourceInfoKind RIK,
                              MCContext &Ctx);

  // Resolves the final symbols that requires the inter-function resource info
  // to be resolved.
  void finalize();

  MCSymbol *getMaxVGPRSymbol();
  MCSymbol *getMaxAGPRSymbol();
  MCSymbol *getMaxSGPRSymbol();

  /// AMDGPUResourceUsageAnalysis gathers resource usage on a per-function
  /// granularity. However, some resource info has to be assigned the call
  /// transitive maximum or accumulative. For example, if A calls B and B's VGPR
  /// usage exceeds A's, A should be assigned B's VGPR usage. Furthermore,
  /// functions with indirect calls should be assigned the module level maximum.
  void gatherResourceInfo(
      const MachineFunction &MF,
      const AMDGPUResourceUsageAnalysis::SIFunctionResourceInfo &FRI);

  const MCExpr *createTotalNumVGPRs(const MachineFunction &MF, MCContext &Ctx);
  const MCExpr *createTotalNumSGPRs(const MachineFunction &MF, bool hasXnack,
                                    MCContext &Ctx);
};
} // namespace llvm
