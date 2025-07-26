//===- RISCVVectorConfigAnalysis ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This is the RISCV analysis of vector unit config.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVVCONFIGANALYSIS_H
#define LLVM_LIB_TARGET_RISCV_RISCVVCONFIGANALYSIS_H

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include <queue>
#include <vector>
using namespace llvm;

namespace llvm {
/// Which subfields of VL or VTYPE have values we need to preserve?
struct DemandedFields {
  // Some unknown property of VL is used.  If demanded, must preserve entire
  // value.
  bool VLAny = false;
  // Only zero vs non-zero is used. If demanded, can change non-zero values.
  bool VLZeroness = false;
  // What properties of SEW we need to preserve.
  enum : uint8_t {
    SEWEqual = 3, // The exact value of SEW needs to be preserved.
    SEWGreaterThanOrEqualAndLessThan64 =
        2, // SEW can be changed as long as it's greater
           // than or equal to the original value, but must be less
           // than 64.
    SEWGreaterThanOrEqual = 1, // SEW can be changed as long as it's greater
                               // than or equal to the original value.
    SEWNone = 0                // We don't need to preserve SEW at all.
  } SEW = SEWNone;
  enum : uint8_t {
    LMULEqual = 2, // The exact value of LMUL needs to be preserved.
    LMULLessThanOrEqualToM1 = 1, // We can use any LMUL <= M1.
    LMULNone = 0                 // We don't need to preserve LMUL at all.
  } LMUL = LMULNone;
  bool SEWLMULRatio = false;
  bool TailPolicy = false;
  bool MaskPolicy = false;
  // If this is true, we demand that VTYPE is set to some legal state, i.e. that
  // vill is unset.
  bool VILL = false;

  // Return true if any part of VTYPE was used
  bool usedVTYPE() const {
    return SEW || LMUL || SEWLMULRatio || TailPolicy || MaskPolicy || VILL;
  }

  // Return true if any property of VL was used
  bool usedVL() { return VLAny || VLZeroness; }

  // Mark all VTYPE subfields and properties as demanded
  void demandVTYPE();

  // Mark all VL properties as demanded
  void demandVL() {
    VLAny = true;
    VLZeroness = true;
  }

  static DemandedFields all() {
    DemandedFields DF;
    DF.demandVTYPE();
    DF.demandVL();
    return DF;
  }

  // Make this the result of demanding both the fields in this and B.
  void doUnion(const DemandedFields &B);

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Support for debugging, callable in GDB: V->dump()
  LLVM_DUMP_METHOD void dump() const {
    print(dbgs());
    dbgs() << "\n";
  }

  void print(raw_ostream &OS) const;
#endif
};

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_ATTRIBUTE_USED
inline raw_ostream &operator<<(raw_ostream &OS, const DemandedFields &DF) {
  DF.print(OS);
  return OS;
}
#endif

/// Defines the abstract state with which the forward dataflow models the
/// values of the VL and VTYPE registers after insertion.
class VSETVLIInfo {
  struct AVLDef {
    // Every AVLDef should have a VNInfo, unless we're running without
    // LiveIntervals in which case this will be nullptr.
    const VNInfo *ValNo;
    Register DefReg;
  };
  union {
    AVLDef AVLRegDef;
    unsigned AVLImm;
  };

  enum : uint8_t {
    Uninitialized,
    AVLIsReg,
    AVLIsImm,
    AVLIsVLMAX,
    Unknown, // AVL and VTYPE are fully unknown
  } State = Uninitialized;

  // Fields from VTYPE.
  RISCVVType::VLMUL VLMul = RISCVVType::LMUL_1;
  uint8_t SEW = 0;
  uint8_t TailAgnostic : 1;
  uint8_t MaskAgnostic : 1;
  uint8_t SEWLMULRatioOnly : 1;

public:
  VSETVLIInfo()
      : AVLImm(0), TailAgnostic(false), MaskAgnostic(false),
        SEWLMULRatioOnly(false) {}

  static VSETVLIInfo getUnknown() {
    VSETVLIInfo Info;
    Info.setUnknown();
    return Info;
  }

  bool isValid() const { return State != Uninitialized; }
  void setUnknown() { State = Unknown; }
  bool isUnknown() const { return State == Unknown; }

  void setAVLRegDef(const VNInfo *VNInfo, Register AVLReg) {
    assert(AVLReg.isVirtual());
    AVLRegDef.ValNo = VNInfo;
    AVLRegDef.DefReg = AVLReg;
    State = AVLIsReg;
  }

  void setAVLImm(unsigned Imm) {
    AVLImm = Imm;
    State = AVLIsImm;
  }

  void setAVLVLMAX() { State = AVLIsVLMAX; }

  bool hasAVLImm() const { return State == AVLIsImm; }
  bool hasAVLReg() const { return State == AVLIsReg; }
  bool hasAVLVLMAX() const { return State == AVLIsVLMAX; }
  Register getAVLReg() const {
    assert(hasAVLReg() && AVLRegDef.DefReg.isVirtual());
    return AVLRegDef.DefReg;
  }
  unsigned getAVLImm() const {
    assert(hasAVLImm());
    return AVLImm;
  }
  const VNInfo *getAVLVNInfo() const {
    assert(hasAVLReg());
    return AVLRegDef.ValNo;
  }
  // Most AVLIsReg infos will have a single defining MachineInstr, unless it was
  // a PHI node. In that case getAVLVNInfo()->def will point to the block
  // boundary slot and this will return nullptr.  If LiveIntervals isn't
  // available, nullptr is also returned.
  const MachineInstr *getAVLDefMI(const LiveIntervals *LIS) const;

  void setAVL(const VSETVLIInfo &Info);

  unsigned getSEW() const { return SEW; }
  RISCVVType::VLMUL getVLMUL() const { return VLMul; }
  bool getTailAgnostic() const { return TailAgnostic; }
  bool getMaskAgnostic() const { return MaskAgnostic; }

  bool hasNonZeroAVL(const LiveIntervals *LIS) const;

  bool hasEquallyZeroAVL(const VSETVLIInfo &Other,
                         const LiveIntervals *LIS) const {
    if (hasSameAVL(Other))
      return true;
    return (hasNonZeroAVL(LIS) && Other.hasNonZeroAVL(LIS));
  }

  bool hasSameAVLLatticeValue(const VSETVLIInfo &Other) const;
  // Return true if the two lattice values are guaranteed to have
  // the same AVL value at runtime.
  bool hasSameAVL(const VSETVLIInfo &Other) const;

  void setVTYPE(unsigned VType);

  void setVTYPE(RISCVVType::VLMUL L, unsigned S, bool TA, bool MA);

  void setVLMul(RISCVVType::VLMUL VLMul) { this->VLMul = VLMul; }

  unsigned encodeVTYPE() const {
    assert(isValid() && !isUnknown() && !SEWLMULRatioOnly &&
           "Can't encode VTYPE for uninitialized or unknown");
    return RISCVVType::encodeVTYPE(VLMul, SEW, TailAgnostic, MaskAgnostic);
  }

  bool hasSEWLMULRatioOnly() const { return SEWLMULRatioOnly; }

  bool hasSameVTYPE(const VSETVLIInfo &Other) const;

  unsigned getSEWLMULRatio() const {
    assert(isValid() && !isUnknown() &&
           "Can't use VTYPE for uninitialized or unknown");
    return RISCVVType::getSEWLMULRatio(SEW, VLMul);
  }

  // Check if the VTYPE for these two VSETVLIInfos produce the same VLMAX.
  // Note that having the same VLMAX ensures that both share the same
  // function from AVL to VL; that is, they must produce the same VL value
  // for any given AVL value.
  bool hasSameVLMAX(const VSETVLIInfo &Other) const {
    assert(isValid() && Other.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!isUnknown() && !Other.isUnknown() &&
           "Can't compare VTYPE in unknown state");
    return getSEWLMULRatio() == Other.getSEWLMULRatio();
  }

  bool hasCompatibleVTYPE(const DemandedFields &Used,
                          const VSETVLIInfo &Require) const;
  // Determine whether the vector instructions requirements represented by
  // Require are compatible with the previous vsetvli instruction represented
  // by this.  MI is the instruction whose requirements we're considering.
  bool isCompatible(const DemandedFields &Used, const VSETVLIInfo &Require,
                    const LiveIntervals *LIS) const;

  bool operator==(const VSETVLIInfo &Other) const;

  bool operator!=(const VSETVLIInfo &Other) const { return !(*this == Other); }

  // Calculate the VSETVLIInfo visible to a block assuming this and Other are
  // both predecessors.
  VSETVLIInfo intersect(const VSETVLIInfo &Other) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Support for debugging, callable in GDB: V->dump()
  LLVM_DUMP_METHOD void dump() const {
    print(dbgs());
    dbgs() << "\n";
  }

  void print(raw_ostream &OS) const;
#endif
};

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_ATTRIBUTE_USED
inline raw_ostream &operator<<(raw_ostream &OS, const VSETVLIInfo &V) {
  V.print(OS);
  return OS;
}
#endif

struct BlockData {
  // The VSETVLIInfo that represents the VL/VTYPE settings on exit from this
  // block. Calculated in Phase 2.
  VSETVLIInfo Exit;

  // The VSETVLIInfo that represents the VL/VTYPE settings from all predecessor
  // blocks. Calculated in Phase 2, and used by Phase 3.
  VSETVLIInfo Pred;

  // Keeps track of whether the block is already in the queue.
  bool InQueue = false;

  BlockData() = default;
};

class RISCVVectorConfigInfo {
  bool HaveVectorOp = false;
  const RISCVSubtarget *ST;
  // Possibly null!
  LiveIntervals *LIS;
  std::queue<const MachineBasicBlock *> WorkList;
  std::vector<BlockData> BlockInfo;

public:
  /// Return the fields and properties demanded by the provided instruction.
  static DemandedFields getDemanded(const MachineInstr &MI,
                                    const RISCVSubtarget *ST);

  /// Return true if moving from CurVType to NewVType is
  /// indistinguishable from the perspective of an instruction (or set
  /// of instructions) which use only the Used subfields and properties.

  static bool areCompatibleVTYPEs(uint64_t CurVType, uint64_t NewVType,
                                  const DemandedFields &Used);
  // Return a VSETVLIInfo representing the changes made by this VSETVLI or
  // VSETIVLI instruction.
  VSETVLIInfo getInfoForVSETVLI(const MachineInstr &MI) const;

  // Return true if we can mutate PrevMI to match MI without changing any the
  // fields which would be observed.
  bool canMutatePriorConfig(const MachineInstr &PrevMI, const MachineInstr &MI,
                            const DemandedFields &Used) const;
  RISCVVectorConfigInfo() {}
  RISCVVectorConfigInfo(const RISCVSubtarget *ST, LiveIntervals *LIS)
      : ST(ST), LIS(LIS) {}
  const std::vector<BlockData> &getInfo() const { return BlockInfo; }
  std::vector<BlockData> &getInfo() { return BlockInfo; }
  bool haveVectorOp();
  void compute(const MachineFunction &MF);
  void clear();
  // Given an incoming state reaching MI, minimally modifies that state so that
  // it is compatible with MI. The resulting state is guaranteed to be
  // semantically legal for MI, but may not be the state requested by MI.
  void transferBefore(VSETVLIInfo &Info, const MachineInstr &MI) const;
  // Given a state with which we evaluated MI (see transferBefore above for why
  // this might be different that the state MI requested), modify the state to
  // reflect the changes MI might make.
  void transferAfter(VSETVLIInfo &Info, const MachineInstr &MI) const;

private:
  static unsigned computeVLMAX(unsigned VLEN, unsigned SEW,
                               RISCVVType::VLMUL VLMul);
  // If we don't use LMUL or the SEW/LMUL ratio, then adjust LMUL so that we
  // maintain the SEW/LMUL ratio. This allows us to eliminate VL toggles in more
  // places.
  static VSETVLIInfo adjustIncoming(const VSETVLIInfo &PrevInfo,
                                    const VSETVLIInfo &NewInfo,
                                    DemandedFields &Demanded);
  /// Return true if a VSETVLI is required to transition from CurInfo to Require
  /// given a set of DemandedFields \p Used.
  bool needVSETVLI(const DemandedFields &Used, const VSETVLIInfo &Require,
                   const VSETVLIInfo &CurInfo) const;
  void computeIncomingVLVTYPE(const MachineBasicBlock &MBB);
  VSETVLIInfo computeInfoForInstr(const MachineInstr &MI) const;
  bool computeVLVTYPEChanges(const MachineBasicBlock &MBB,
                             VSETVLIInfo &Info) const;
  // If the AVL is defined by a vsetvli's output vl with the same VLMAX, we can
  // replace the AVL operand with the AVL of the defining vsetvli. E.g.
  //
  // %vl = PseudoVSETVLI %avl:gpr, SEW=32, LMUL=M1
  // $x0 = PseudoVSETVLI %vl:gpr, SEW=32, LMUL=M1
  // ->
  // %vl = PseudoVSETVLI %avl:gpr, SEW=32, LMUL=M1
  // $x0 = PseudoVSETVLI %avl:gpr, SEW=32, LMUL=M1
  void forwardVSETVLIAVL(VSETVLIInfo &Info) const;
};

class RISCVVectorConfigAnalysis
    : public AnalysisInfoMixin<RISCVVectorConfigAnalysis> {
  friend AnalysisInfoMixin<RISCVVectorConfigAnalysis>;
  static AnalysisKey Key;

public:
  using Result = RISCVVectorConfigInfo;
  Result run(MachineFunction &MF, MachineFunctionAnalysisManager &MFAM);
};

class RISCVVectorConfigWrapperPass : public MachineFunctionPass {
  RISCVVectorConfigInfo Result;

public:
  static char ID;

  RISCVVectorConfigWrapperPass();

  void getAnalysisUsage(AnalysisUsage &) const override;
  bool runOnMachineFunction(MachineFunction &) override;
  void releaseMemory() override { Result.clear(); }
  RISCVVectorConfigInfo &getResult() { return Result; }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVVCONFIGANALYSIS_H
