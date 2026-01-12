//===- RISCVVSETVLIInfoAnalysis.h - VSETVLI Info Analysis -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an analysis of the vtype/vl information that is needed
// by RISCVInsertVSETVLI pass and others.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveDebugVariables.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveStacks.h"

namespace llvm {
namespace RISCV {
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
  bool TWiden = false;
  bool AltFmt = false;

  // Return true if any part of VTYPE was used
  bool usedVTYPE() const {
    return SEW || LMUL || SEWLMULRatio || TailPolicy || MaskPolicy || VILL ||
           TWiden || AltFmt;
  }

  // Return true if any property of VL was used
  bool usedVL() { return VLAny || VLZeroness; }

  // Mark all VTYPE subfields and properties as demanded
  void demandVTYPE() {
    SEW = SEWEqual;
    LMUL = LMULEqual;
    SEWLMULRatio = true;
    TailPolicy = true;
    MaskPolicy = true;
    VILL = true;
    TWiden = true;
    AltFmt = true;
  }

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
  void doUnion(const DemandedFields &B) {
    VLAny |= B.VLAny;
    VLZeroness |= B.VLZeroness;
    SEW = std::max(SEW, B.SEW);
    LMUL = std::max(LMUL, B.LMUL);
    SEWLMULRatio |= B.SEWLMULRatio;
    TailPolicy |= B.TailPolicy;
    MaskPolicy |= B.MaskPolicy;
    VILL |= B.VILL;
    AltFmt |= B.AltFmt;
    TWiden |= B.TWiden;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Support for debugging, callable in GDB: V->dump()
  LLVM_DUMP_METHOD void dump() const {
    print(dbgs());
    dbgs() << "\n";
  }

  /// Implement operator<<.
  void print(raw_ostream &OS) const {
    OS << "{";
    OS << "VLAny=" << VLAny << ", ";
    OS << "VLZeroness=" << VLZeroness << ", ";
    OS << "SEW=";
    switch (SEW) {
    case SEWEqual:
      OS << "SEWEqual";
      break;
    case SEWGreaterThanOrEqual:
      OS << "SEWGreaterThanOrEqual";
      break;
    case SEWGreaterThanOrEqualAndLessThan64:
      OS << "SEWGreaterThanOrEqualAndLessThan64";
      break;
    case SEWNone:
      OS << "SEWNone";
      break;
    };
    OS << ", ";
    OS << "LMUL=";
    switch (LMUL) {
    case LMULEqual:
      OS << "LMULEqual";
      break;
    case LMULLessThanOrEqualToM1:
      OS << "LMULLessThanOrEqualToM1";
      break;
    case LMULNone:
      OS << "LMULNone";
      break;
    };
    OS << ", ";
    OS << "SEWLMULRatio=" << SEWLMULRatio << ", ";
    OS << "TailPolicy=" << TailPolicy << ", ";
    OS << "MaskPolicy=" << MaskPolicy << ", ";
    OS << "VILL=" << VILL << ", ";
    OS << "AltFmt=" << AltFmt << ", ";
    OS << "TWiden=" << TWiden;
    OS << "}";
  }
#endif
};

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_ATTRIBUTE_USED
inline raw_ostream &operator<<(raw_ostream &OS, const DemandedFields &DF) {
  DF.print(OS);
  return OS;
}
#endif

bool areCompatibleVTYPEs(uint64_t CurVType, uint64_t NewVType,
                         const DemandedFields &Used);

/// Return the fields and properties demanded by the provided instruction.
DemandedFields getDemanded(const MachineInstr &MI, const RISCVSubtarget *ST);

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

  enum class AVLState : uint8_t {
    Uninitialized,
    AVLIsReg,
    AVLIsImm,
    AVLIsVLMAX,
    Unknown, // AVL and VTYPE are fully unknown
  } State = AVLState::Uninitialized;

  // Fields from VTYPE.
  RISCVVType::VLMUL VLMul = RISCVVType::LMUL_1;
  uint8_t SEW = 0;
  uint8_t TailAgnostic : 1;
  uint8_t MaskAgnostic : 1;
  uint8_t SEWLMULRatioOnly : 1;
  uint8_t AltFmt : 1;
  uint8_t TWiden : 3;

public:
  VSETVLIInfo()
      : AVLImm(0), TailAgnostic(false), MaskAgnostic(false),
        SEWLMULRatioOnly(false), AltFmt(false), TWiden(0) {}

  static VSETVLIInfo getUnknown() {
    VSETVLIInfo Info;
    Info.setUnknown();
    return Info;
  }

  bool isValid() const { return State != AVLState::Uninitialized; }
  void setUnknown() { State = AVLState::Unknown; }
  bool isUnknown() const { return State == AVLState::Unknown; }

  void setAVLRegDef(const VNInfo *VNInfo, Register AVLReg) {
    assert(AVLReg.isVirtual());
    AVLRegDef.ValNo = VNInfo;
    AVLRegDef.DefReg = AVLReg;
    State = AVLState::AVLIsReg;
  }

  void setAVLImm(unsigned Imm) {
    AVLImm = Imm;
    State = AVLState::AVLIsImm;
  }

  void setAVLVLMAX() { State = AVLState::AVLIsVLMAX; }

  bool hasAVLImm() const { return State == AVLState::AVLIsImm; }
  bool hasAVLReg() const { return State == AVLState::AVLIsReg; }
  bool hasAVLVLMAX() const { return State == AVLState::AVLIsVLMAX; }
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
  const MachineInstr *getAVLDefMI(const LiveIntervals *LIS) const {
    assert(hasAVLReg());
    if (!LIS || getAVLVNInfo()->isPHIDef())
      return nullptr;
    auto *MI = LIS->getInstructionFromIndex(getAVLVNInfo()->def);
    assert(MI);
    return MI;
  }

  void setAVL(const VSETVLIInfo &Info) {
    assert(Info.isValid());
    if (Info.isUnknown())
      setUnknown();
    else if (Info.hasAVLReg())
      setAVLRegDef(Info.getAVLVNInfo(), Info.getAVLReg());
    else if (Info.hasAVLVLMAX())
      setAVLVLMAX();
    else {
      assert(Info.hasAVLImm());
      setAVLImm(Info.getAVLImm());
    }
  }

  bool hasSEWLMULRatioOnly() const { return SEWLMULRatioOnly; }

  unsigned getSEW() const {
    assert(isValid() && !isUnknown() && !hasSEWLMULRatioOnly() &&
           "Can't use VTYPE for uninitialized or unknown");
    return SEW;
  }
  RISCVVType::VLMUL getVLMUL() const {
    assert(isValid() && !isUnknown() && !hasSEWLMULRatioOnly() &&
           "Can't use VTYPE for uninitialized or unknown");
    return VLMul;
  }
  bool getTailAgnostic() const {
    assert(isValid() && !isUnknown() &&
           "Can't use VTYPE for uninitialized or unknown");
    return TailAgnostic;
  }
  bool getMaskAgnostic() const {
    assert(isValid() && !isUnknown() &&
           "Can't use VTYPE for uninitialized or unknown");
    return MaskAgnostic;
  }
  bool getAltFmt() const {
    assert(isValid() && !isUnknown() &&
           "Can't use VTYPE for uninitialized or unknown");
    return AltFmt;
  }
  unsigned getTWiden() const {
    assert(isValid() && !isUnknown() &&
           "Can't use VTYPE for uninitialized or unknown");
    return TWiden;
  }

  bool hasNonZeroAVL(const LiveIntervals *LIS) const {
    if (hasAVLImm())
      return getAVLImm() > 0;
    if (hasAVLReg()) {
      if (auto *DefMI = getAVLDefMI(LIS))
        return RISCVInstrInfo::isNonZeroLoadImmediate(*DefMI);
    }
    if (hasAVLVLMAX())
      return true;
    return false;
  }

  bool hasEquallyZeroAVL(const VSETVLIInfo &Other,
                         const LiveIntervals *LIS) const {
    if (hasSameAVL(Other))
      return true;
    return (hasNonZeroAVL(LIS) && Other.hasNonZeroAVL(LIS));
  }

  bool hasSameAVLLatticeValue(const VSETVLIInfo &Other) const {
    if (hasAVLReg() && Other.hasAVLReg()) {
      assert(!getAVLVNInfo() == !Other.getAVLVNInfo() &&
             "we either have intervals or we don't");
      if (!getAVLVNInfo())
        return getAVLReg() == Other.getAVLReg();
      return getAVLVNInfo()->id == Other.getAVLVNInfo()->id &&
             getAVLReg() == Other.getAVLReg();
    }

    if (hasAVLImm() && Other.hasAVLImm())
      return getAVLImm() == Other.getAVLImm();

    if (hasAVLVLMAX())
      return Other.hasAVLVLMAX() && hasSameVLMAX(Other);

    return false;
  }

  // Return true if the two lattice values are guaranteed to have
  // the same AVL value at runtime.
  bool hasSameAVL(const VSETVLIInfo &Other) const {
    // Without LiveIntervals, we don't know which instruction defines a
    // register.  Since a register may be redefined, this means all AVLIsReg
    // states must be treated as possibly distinct.
    if (hasAVLReg() && Other.hasAVLReg()) {
      assert(!getAVLVNInfo() == !Other.getAVLVNInfo() &&
             "we either have intervals or we don't");
      if (!getAVLVNInfo())
        return false;
    }
    return hasSameAVLLatticeValue(Other);
  }

  void setVTYPE(unsigned VType) {
    assert(isValid() && !isUnknown() &&
           "Can't set VTYPE for uninitialized or unknown");
    VLMul = RISCVVType::getVLMUL(VType);
    SEW = RISCVVType::getSEW(VType);
    TailAgnostic = RISCVVType::isTailAgnostic(VType);
    MaskAgnostic = RISCVVType::isMaskAgnostic(VType);
    AltFmt = RISCVVType::isAltFmt(VType);
    TWiden =
        RISCVVType::hasXSfmmWiden(VType) ? RISCVVType::getXSfmmWiden(VType) : 0;
  }
  void setVTYPE(RISCVVType::VLMUL L, unsigned S, bool TA, bool MA, bool Altfmt,
                unsigned W) {
    assert(isValid() && !isUnknown() &&
           "Can't set VTYPE for uninitialized or unknown");
    VLMul = L;
    SEW = S;
    TailAgnostic = TA;
    MaskAgnostic = MA;
    AltFmt = Altfmt;
    TWiden = W;
  }

  void setAltFmt(bool AF) { AltFmt = AF; }

  void setVLMul(RISCVVType::VLMUL VLMul) { this->VLMul = VLMul; }

  unsigned encodeVTYPE() const {
    assert(isValid() && !isUnknown() && !SEWLMULRatioOnly &&
           "Can't encode VTYPE for uninitialized or unknown");
    if (TWiden != 0)
      return RISCVVType::encodeXSfmmVType(SEW, TWiden, AltFmt);
    return RISCVVType::encodeVTYPE(VLMul, SEW, TailAgnostic, MaskAgnostic,
                                   AltFmt);
  }

  bool hasSameVTYPE(const VSETVLIInfo &Other) const {
    assert(isValid() && Other.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!isUnknown() && !Other.isUnknown() &&
           "Can't compare VTYPE in unknown state");
    assert(!SEWLMULRatioOnly && !Other.SEWLMULRatioOnly &&
           "Can't compare when only LMUL/SEW ratio is valid.");
    return std::tie(VLMul, SEW, TailAgnostic, MaskAgnostic, AltFmt, TWiden) ==
           std::tie(Other.VLMul, Other.SEW, Other.TailAgnostic,
                    Other.MaskAgnostic, Other.AltFmt, Other.TWiden);
  }

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
                    const LiveIntervals *LIS) const {
    assert(isValid() && Require.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    // Nothing is compatible with Unknown.
    if (isUnknown() || Require.isUnknown())
      return false;

    // If only our VLMAX ratio is valid, then this isn't compatible.
    if (SEWLMULRatioOnly || Require.SEWLMULRatioOnly)
      return false;

    if (Used.VLAny && !(hasSameAVL(Require) && hasSameVLMAX(Require)))
      return false;

    if (Used.VLZeroness && !hasEquallyZeroAVL(Require, LIS))
      return false;

    return hasCompatibleVTYPE(Used, Require);
  }

  bool operator==(const VSETVLIInfo &Other) const {
    // Uninitialized is only equal to another Uninitialized.
    if (!isValid())
      return !Other.isValid();
    if (!Other.isValid())
      return !isValid();

    // Unknown is only equal to another Unknown.
    if (isUnknown())
      return Other.isUnknown();
    if (Other.isUnknown())
      return isUnknown();

    if (!hasSameAVLLatticeValue(Other))
      return false;

    // If the SEWLMULRatioOnly bits are different, then they aren't equal.
    if (SEWLMULRatioOnly != Other.SEWLMULRatioOnly)
      return false;

    // If only the VLMAX is valid, check that it is the same.
    if (SEWLMULRatioOnly)
      return hasSameVLMAX(Other);

    // If the full VTYPE is valid, check that it is the same.
    return hasSameVTYPE(Other);
  }

  bool operator!=(const VSETVLIInfo &Other) const { return !(*this == Other); }

  // Calculate the VSETVLIInfo visible to a block assuming this and Other are
  // both predecessors.
  VSETVLIInfo intersect(const VSETVLIInfo &Other) const {
    // If the new value isn't valid, ignore it.
    if (!Other.isValid())
      return *this;

    // If this value isn't valid, this must be the first predecessor, use it.
    if (!isValid())
      return Other;

    // If either is unknown, the result is unknown.
    if (isUnknown() || Other.isUnknown())
      return VSETVLIInfo::getUnknown();

    // If we have an exact, match return this.
    if (*this == Other)
      return *this;

    // Not an exact match, but maybe the AVL and VLMAX are the same. If so,
    // return an SEW/LMUL ratio only value.
    if (hasSameAVL(Other) && hasSameVLMAX(Other)) {
      VSETVLIInfo MergeInfo = *this;
      MergeInfo.SEWLMULRatioOnly = true;
      return MergeInfo;
    }

    // Otherwise the result is unknown.
    return VSETVLIInfo::getUnknown();
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Support for debugging, callable in GDB: V->dump()
  LLVM_DUMP_METHOD void dump() const {
    print(dbgs());
    dbgs() << "\n";
  }

  /// Implement operator<<.
  /// @{
  void print(raw_ostream &OS) const {
    OS << '{';
    switch (State) {
    case AVLState::Uninitialized:
      OS << "Uninitialized";
      break;
    case AVLState::Unknown:
      OS << "unknown";
      break;
    case AVLState::AVLIsReg:
      OS << "AVLReg=" << llvm::printReg(getAVLReg());
      break;
    case AVLState::AVLIsImm:
      OS << "AVLImm=" << (unsigned)AVLImm;
      break;
    case AVLState::AVLIsVLMAX:
      OS << "AVLVLMAX";
      break;
    }
    if (isValid() && !isUnknown()) {
      OS << ", ";

      unsigned LMul;
      bool Fractional;
      std::tie(LMul, Fractional) = decodeVLMUL(VLMul);

      OS << "VLMul=m";
      if (Fractional)
        OS << 'f';
      OS << LMul << ", "
         << "SEW=e" << (unsigned)SEW << ", "
         << "TailAgnostic=" << (bool)TailAgnostic << ", "
         << "MaskAgnostic=" << (bool)MaskAgnostic << ", "
         << "SEWLMULRatioOnly=" << (bool)SEWLMULRatioOnly << ", "
         << "TWiden=" << (unsigned)TWiden << ", "
         << "AltFmt=" << (bool)AltFmt;
    }

    OS << '}';
  }
#endif
};

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_ATTRIBUTE_USED
inline raw_ostream &operator<<(raw_ostream &OS, const VSETVLIInfo &V) {
  V.print(OS);
  return OS;
}
#endif

class RISCVVSETVLIInfoAnalysis {
  const RISCVSubtarget *ST;
  // Possibly null!
  LiveIntervals *LIS;

public:
  RISCVVSETVLIInfoAnalysis() = default;
  RISCVVSETVLIInfoAnalysis(const RISCVSubtarget *ST, LiveIntervals *LIS)
      : ST(ST), LIS(LIS) {}

  VSETVLIInfo getInfoForVSETVLI(const MachineInstr &MI) const;
  VSETVLIInfo computeInfoForInstr(const MachineInstr &MI) const;

private:
  void forwardVSETVLIAVL(VSETVLIInfo &Info) const;
};

} // namespace RISCV
} // namespace llvm
