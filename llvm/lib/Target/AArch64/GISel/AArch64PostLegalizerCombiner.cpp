 //=== lib/CodeGen/GlobalISel/AArch64PostLegalizerCombiner.cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This performs post-legalization combines on generic MachineInstrs.
//
// Any combine that this pass performs must preserve instruction legality.
// Combines unconcerned with legality should be handled by the
// PreLegalizerCombiner instead.
//
//===----------------------------------------------------------------------===//

#include "AArch64TargetMachine.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aarch64-postlegalizer-combiner"

using namespace llvm;
using namespace MIPatternMatch;

/// Represents a pseudo instruction which replaces a G_SHUFFLE_VECTOR.
///
/// Used for matching target-supported shuffles before codegen.
struct ShuffleVectorPseudo {
  unsigned Opc; ///< Opcode for the instruction. (E.g. G_ZIP1)
  Register Dst; ///< Destination register.
  SmallVector<SrcOp, 2> SrcOps; ///< Source registers.
  ShuffleVectorPseudo(unsigned Opc, Register Dst,
                      std::initializer_list<SrcOp> SrcOps)
      : Opc(Opc), Dst(Dst), SrcOps(SrcOps){};
  ShuffleVectorPseudo() {}
};

/// \returns The splat index of a G_SHUFFLE_VECTOR \p MI when \p MI is a splat.
/// If \p MI is not a splat, returns None.
static Optional<int> getSplatIndex(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR &&
         "Only G_SHUFFLE_VECTOR can have a splat index!");
  ArrayRef<int> Mask = MI.getOperand(3).getShuffleMask();
  auto FirstDefinedIdx = find_if(Mask, [](int Elt) { return Elt >= 0; });

  // If all elements are undefined, this shuffle can be considered a splat.
  // Return 0 for better potential for callers to simplify.
  if (FirstDefinedIdx == Mask.end())
    return 0;

  // Make sure all remaining elements are either undef or the same
  // as the first non-undef value.
  int SplatValue = *FirstDefinedIdx;
  if (any_of(make_range(std::next(FirstDefinedIdx), Mask.end()),
             [&SplatValue](int Elt) { return Elt >= 0 && Elt != SplatValue; }))
    return None;

  return SplatValue;
}

/// Check if a vector shuffle corresponds to a REV instruction with the
/// specified blocksize.
static bool isREVMask(ArrayRef<int> M, unsigned EltSize, unsigned NumElts,
                      unsigned BlockSize) {
  assert((BlockSize == 16 || BlockSize == 32 || BlockSize == 64) &&
         "Only possible block sizes for REV are: 16, 32, 64");
  assert(EltSize != 64 && "EltSize cannot be 64 for REV mask.");

  unsigned BlockElts = M[0] + 1;

  // If the first shuffle index is UNDEF, be optimistic.
  if (M[0] < 0)
    BlockElts = BlockSize / EltSize;

  if (BlockSize <= EltSize || BlockSize != BlockElts * EltSize)
    return false;

  for (unsigned i = 0; i < NumElts; ++i) {
    // Ignore undef indices.
    if (M[i] < 0)
      continue;
    if (static_cast<unsigned>(M[i]) !=
        (i - i % BlockElts) + (BlockElts - 1 - i % BlockElts))
      return false;
  }

  return true;
}

/// Determines if \p M is a shuffle vector mask for a TRN of \p NumElts.
/// Whether or not G_TRN1 or G_TRN2 should be used is stored in \p WhichResult.
static bool isTRNMask(ArrayRef<int> M, unsigned NumElts,
                      unsigned &WhichResult) {
  if (NumElts % 2 != 0)
    return false;
  WhichResult = (M[0] == 0 ? 0 : 1);
  for (unsigned i = 0; i < NumElts; i += 2) {
    if ((M[i] >= 0 && static_cast<unsigned>(M[i]) != i + WhichResult) ||
        (M[i + 1] >= 0 &&
         static_cast<unsigned>(M[i + 1]) != i + NumElts + WhichResult))
      return false;
  }
  return true;
}

/// Check if a G_EXT instruction can handle a shuffle mask \p M when the vector
/// sources of the shuffle are different.
static Optional<std::pair<bool, uint64_t>> getExtMask(ArrayRef<int> M,
                                                      unsigned NumElts) {
  // Look for the first non-undef element.
  auto FirstRealElt = find_if(M, [](int Elt) { return Elt >= 0; });
  if (FirstRealElt == M.end())
    return None;

  // Use APInt to handle overflow when calculating expected element.
  unsigned MaskBits = APInt(32, NumElts * 2).logBase2();
  APInt ExpectedElt = APInt(MaskBits, *FirstRealElt + 1);

  // The following shuffle indices must be the successive elements after the
  // first real element.
  if (any_of(
          make_range(std::next(FirstRealElt), M.end()),
          [&ExpectedElt](int Elt) { return Elt != ExpectedElt++ && Elt >= 0; }))
    return None;

  // The index of an EXT is the first element if it is not UNDEF.
  // Watch out for the beginning UNDEFs. The EXT index should be the expected
  // value of the first element.  E.g.
  // <-1, -1, 3, ...> is treated as <1, 2, 3, ...>.
  // <-1, -1, 0, 1, ...> is treated as <2*NumElts-2, 2*NumElts-1, 0, 1, ...>.
  // ExpectedElt is the last mask index plus 1.
  uint64_t Imm = ExpectedElt.getZExtValue();
  bool ReverseExt = false;

  // There are two difference cases requiring to reverse input vectors.
  // For example, for vector <4 x i32> we have the following cases,
  // Case 1: shufflevector(<4 x i32>,<4 x i32>,<-1, -1, -1, 0>)
  // Case 2: shufflevector(<4 x i32>,<4 x i32>,<-1, -1, 7, 0>)
  // For both cases, we finally use mask <5, 6, 7, 0>, which requires
  // to reverse two input vectors.
  if (Imm < NumElts)
    ReverseExt = true;
  else
    Imm -= NumElts;
  return std::make_pair(ReverseExt, Imm);
}

/// Determines if \p M is a shuffle vector mask for a UZP of \p NumElts.
/// Whether or not G_UZP1 or G_UZP2 should be used is stored in \p WhichResult.
static bool isUZPMask(ArrayRef<int> M, unsigned NumElts,
                      unsigned &WhichResult) {
  WhichResult = (M[0] == 0 ? 0 : 1);
  for (unsigned i = 0; i != NumElts; ++i) {
    // Skip undef indices.
    if (M[i] < 0)
      continue;
    if (static_cast<unsigned>(M[i]) != 2 * i + WhichResult)
      return false;
  }
  return true;
}

/// \return true if \p M is a zip mask for a shuffle vector of \p NumElts.
/// Whether or not G_ZIP1 or G_ZIP2 should be used is stored in \p WhichResult.
static bool isZipMask(ArrayRef<int> M, unsigned NumElts,
                      unsigned &WhichResult) {
  if (NumElts % 2 != 0)
    return false;

  // 0 means use ZIP1, 1 means use ZIP2.
  WhichResult = (M[0] == 0 ? 0 : 1);
  unsigned Idx = WhichResult * NumElts / 2;
  for (unsigned i = 0; i != NumElts; i += 2) {
      if ((M[i] >= 0 && static_cast<unsigned>(M[i]) != Idx) ||
          (M[i + 1] >= 0 && static_cast<unsigned>(M[i + 1]) != Idx + NumElts))
        return false;
    Idx += 1;
  }
  return true;
}

/// \return true if a G_SHUFFLE_VECTOR instruction \p MI can be replaced with a
/// G_REV instruction. Returns the appropriate G_REV opcode in \p Opc.
static bool matchREV(MachineInstr &MI, MachineRegisterInfo &MRI,
                     ShuffleVectorPseudo &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR);
  ArrayRef<int> ShuffleMask = MI.getOperand(3).getShuffleMask();
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  LLT Ty = MRI.getType(Dst);
  unsigned EltSize = Ty.getScalarSizeInBits();

  // Element size for a rev cannot be 64.
  if (EltSize == 64)
    return false;

  unsigned NumElts = Ty.getNumElements();

  // Try to produce G_REV64
  if (isREVMask(ShuffleMask, EltSize, NumElts, 64)) {
    MatchInfo = ShuffleVectorPseudo(AArch64::G_REV64, Dst, {Src});
    return true;
  }

  // TODO: Produce G_REV32 and G_REV16 once we have proper legalization support.
  // This should be identical to above, but with a constant 32 and constant
  // 16.
  return false;
}

/// \return true if a G_SHUFFLE_VECTOR instruction \p MI can be replaced with
/// a G_TRN1 or G_TRN2 instruction.
static bool matchTRN(MachineInstr &MI, MachineRegisterInfo &MRI,
                     ShuffleVectorPseudo &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR);
  unsigned WhichResult;
  ArrayRef<int> ShuffleMask = MI.getOperand(3).getShuffleMask();
  Register Dst = MI.getOperand(0).getReg();
  unsigned NumElts = MRI.getType(Dst).getNumElements();
  if (!isTRNMask(ShuffleMask, NumElts, WhichResult))
    return false;
  unsigned Opc = (WhichResult == 0) ? AArch64::G_TRN1 : AArch64::G_TRN2;
  Register V1 = MI.getOperand(1).getReg();
  Register V2 = MI.getOperand(2).getReg();
  MatchInfo = ShuffleVectorPseudo(Opc, Dst, {V1, V2});
  return true;
}

/// \return true if a G_SHUFFLE_VECTOR instruction \p MI can be replaced with
/// a G_UZP1 or G_UZP2 instruction.
///
/// \param [in] MI - The shuffle vector instruction.
/// \param [out] MatchInfo - Either G_UZP1 or G_UZP2 on success.
static bool matchUZP(MachineInstr &MI, MachineRegisterInfo &MRI,
                     ShuffleVectorPseudo &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR);
  unsigned WhichResult;
  ArrayRef<int> ShuffleMask = MI.getOperand(3).getShuffleMask();
  Register Dst = MI.getOperand(0).getReg();
  unsigned NumElts = MRI.getType(Dst).getNumElements();
  if (!isUZPMask(ShuffleMask, NumElts, WhichResult))
    return false;
  unsigned Opc = (WhichResult == 0) ? AArch64::G_UZP1 : AArch64::G_UZP2;
  Register V1 = MI.getOperand(1).getReg();
  Register V2 = MI.getOperand(2).getReg();
  MatchInfo = ShuffleVectorPseudo(Opc, Dst, {V1, V2});
  return true;
}

static bool matchZip(MachineInstr &MI, MachineRegisterInfo &MRI,
                     ShuffleVectorPseudo &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR);
  unsigned WhichResult;
  ArrayRef<int> ShuffleMask = MI.getOperand(3).getShuffleMask();
  Register Dst = MI.getOperand(0).getReg();
  unsigned NumElts = MRI.getType(Dst).getNumElements();
  if (!isZipMask(ShuffleMask, NumElts, WhichResult))
    return false;
  unsigned Opc = (WhichResult == 0) ? AArch64::G_ZIP1 : AArch64::G_ZIP2;
  Register V1 = MI.getOperand(1).getReg();
  Register V2 = MI.getOperand(2).getReg();
  MatchInfo = ShuffleVectorPseudo(Opc, Dst, {V1, V2});
  return true;
}

static bool matchDup(MachineInstr &MI, MachineRegisterInfo &MRI,
                     ShuffleVectorPseudo &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR);
  auto Lane = getSplatIndex(MI);
  if (!Lane || *Lane != 0)
    return false;

  // Try to match a vector splat operation into a dup instruction.
  // We're looking for this pattern:
  //
  // %scalar:gpr(s64) = COPY $x0
  // %undef:fpr(<2 x s64>) = G_IMPLICIT_DEF
  // %cst0:gpr(s32) = G_CONSTANT i32 0
  // %zerovec:fpr(<2 x s32>) = G_BUILD_VECTOR %cst0(s32), %cst0(s32)
  // %ins:fpr(<2 x s64>) = G_INSERT_VECTOR_ELT %undef, %scalar(s64), %cst0(s32)
  // %splat:fpr(<2 x s64>) = G_SHUFFLE_VECTOR %ins(<2 x s64>), %undef, %zerovec(<2 x s32>)
  //
  // ...into:
  // %splat = G_DUP %scalar

  // Begin matching the insert.
  auto *InsMI = getOpcodeDef(TargetOpcode::G_INSERT_VECTOR_ELT,
                             MI.getOperand(1).getReg(), MRI);
  if (!InsMI)
    return false;

  // Match the undef vector operand.
  if (!getOpcodeDef(TargetOpcode::G_IMPLICIT_DEF,
                               InsMI->getOperand(1).getReg(), MRI))
    return false;

  // Match the index constant 0.
  int64_t Index = 0;
  if (!mi_match(InsMI->getOperand(3).getReg(), MRI, m_ICst(Index)) || Index)
    return false;

  Register Dst = MI.getOperand(0).getReg();
  MatchInfo =
      ShuffleVectorPseudo(AArch64::G_DUP, Dst, {InsMI->getOperand(2).getReg()});
  return true;
}

static bool matchEXT(MachineInstr &MI, MachineRegisterInfo &MRI,
                     ShuffleVectorPseudo &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR);
  Register Dst = MI.getOperand(0).getReg();
  auto ExtInfo = getExtMask(MI.getOperand(3).getShuffleMask(),
                            MRI.getType(Dst).getNumElements());
  if (!ExtInfo)
    return false;
  bool ReverseExt;
  uint64_t Imm;
  std::tie(ReverseExt, Imm) = *ExtInfo;
  Register V1 = MI.getOperand(1).getReg();
  Register V2 = MI.getOperand(2).getReg();
  if (ReverseExt)
    std::swap(V1, V2);
  uint64_t ExtFactor = MRI.getType(V1).getScalarSizeInBits() / 8;
  Imm *= ExtFactor;
  MatchInfo = ShuffleVectorPseudo(AArch64::G_EXT, Dst, {V1, V2, Imm});
  return true;
}

/// Replace a G_SHUFFLE_VECTOR instruction with a pseudo.
/// \p Opc is the opcode to use. \p MI is the G_SHUFFLE_VECTOR.
static bool applyShuffleVectorPseudo(MachineInstr &MI,
                                     ShuffleVectorPseudo &MatchInfo) {
  MachineIRBuilder MIRBuilder(MI);
  MIRBuilder.buildInstr(MatchInfo.Opc, {MatchInfo.Dst}, MatchInfo.SrcOps);
  MI.eraseFromParent();
  return true;
}

/// Replace a G_SHUFFLE_VECTOR instruction with G_EXT.
/// Special-cased because the constant operand must be emitted as a G_CONSTANT
/// for the imported tablegen patterns to work.
static bool applyEXT(MachineInstr &MI, ShuffleVectorPseudo &MatchInfo) {
  MachineIRBuilder MIRBuilder(MI);
  // Tablegen patterns expect an i32 G_CONSTANT as the final op.
  auto Cst =
      MIRBuilder.buildConstant(LLT::scalar(32), MatchInfo.SrcOps[2].getImm());
  MIRBuilder.buildInstr(MatchInfo.Opc, {MatchInfo.Dst},
                        {MatchInfo.SrcOps[0], MatchInfo.SrcOps[1], Cst});
  MI.eraseFromParent();
  return true;
}

#define AARCH64POSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS
#include "AArch64GenPostLegalizeGICombiner.inc"
#undef AARCH64POSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_DEPS

namespace {
#define AARCH64POSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H
#include "AArch64GenPostLegalizeGICombiner.inc"
#undef AARCH64POSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_H

class AArch64PostLegalizerCombinerInfo : public CombinerInfo {
  GISelKnownBits *KB;
  MachineDominatorTree *MDT;

public:
  AArch64GenPostLegalizerCombinerHelperRuleConfig GeneratedRuleCfg;

  AArch64PostLegalizerCombinerInfo(bool EnableOpt, bool OptSize, bool MinSize,
                                   GISelKnownBits *KB,
                                   MachineDominatorTree *MDT)
      : CombinerInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, EnableOpt, OptSize, MinSize),
        KB(KB), MDT(MDT) {
    if (!GeneratedRuleCfg.parseCommandLineOption())
      report_fatal_error("Invalid rule identifier");
  }

  virtual bool combine(GISelChangeObserver &Observer, MachineInstr &MI,
                       MachineIRBuilder &B) const override;
};

bool AArch64PostLegalizerCombinerInfo::combine(GISelChangeObserver &Observer,
                                               MachineInstr &MI,
                                               MachineIRBuilder &B) const {
  const auto *LI =
      MI.getParent()->getParent()->getSubtarget().getLegalizerInfo();
  CombinerHelper Helper(Observer, B, KB, MDT, LI);
  AArch64GenPostLegalizerCombinerHelper Generated(GeneratedRuleCfg);
  return Generated.tryCombineAll(Observer, MI, B, Helper);
}

#define AARCH64POSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP
#include "AArch64GenPostLegalizeGICombiner.inc"
#undef AARCH64POSTLEGALIZERCOMBINERHELPER_GENCOMBINERHELPER_CPP

class AArch64PostLegalizerCombiner : public MachineFunctionPass {
public:
  static char ID;

  AArch64PostLegalizerCombiner(bool IsOptNone = false);

  StringRef getPassName() const override {
    return "AArch64PostLegalizerCombiner";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  bool IsOptNone;
};
} // end anonymous namespace

void AArch64PostLegalizerCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  AU.addRequired<GISelKnownBitsAnalysis>();
  AU.addPreserved<GISelKnownBitsAnalysis>();
  if (!IsOptNone) {
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
  }
  MachineFunctionPass::getAnalysisUsage(AU);
}

AArch64PostLegalizerCombiner::AArch64PostLegalizerCombiner(bool IsOptNone)
    : MachineFunctionPass(ID), IsOptNone(IsOptNone) {
  initializeAArch64PostLegalizerCombinerPass(*PassRegistry::getPassRegistry());
}

bool AArch64PostLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  assert(MF.getProperties().hasProperty(
             MachineFunctionProperties::Property::Legalized) &&
         "Expected a legalized function?");
  auto *TPC = &getAnalysis<TargetPassConfig>();
  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOpt::None && !skipFunction(F);
  GISelKnownBits *KB = &getAnalysis<GISelKnownBitsAnalysis>().get(MF);
  MachineDominatorTree *MDT =
      IsOptNone ? nullptr : &getAnalysis<MachineDominatorTree>();
  AArch64PostLegalizerCombinerInfo PCInfo(EnableOpt, F.hasOptSize(),
                                          F.hasMinSize(), KB, MDT);
  Combiner C(PCInfo, TPC);
  return C.combineMachineInstrs(MF, /*CSEInfo*/ nullptr);
}

char AArch64PostLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(AArch64PostLegalizerCombiner, DEBUG_TYPE,
                      "Combine AArch64 MachineInstrs after legalization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelKnownBitsAnalysis)
INITIALIZE_PASS_END(AArch64PostLegalizerCombiner, DEBUG_TYPE,
                    "Combine AArch64 MachineInstrs after legalization", false,
                    false)

namespace llvm {
FunctionPass *createAArch64PostLegalizeCombiner(bool IsOptNone) {
  return new AArch64PostLegalizerCombiner(IsOptNone);
}
} // end namespace llvm
