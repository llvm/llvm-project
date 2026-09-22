//===-- lib/CodeGen/GlobalISel/PISAPreLegalizerCombiner.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PISAMCTargetDesc.h"
#include "PISA.h"
#include "PISALegalizerInfo.h"
#include "PISATargetMachine.h"
#include "PISAUtils.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/bit.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutor.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GISelChangeObserver.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IntrinsicsPISA.h"
#include "llvm/Target/TargetMachine.h"

#define GET_GICOMBINER_DEPS
#include "PISAGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_DEPS

#define DEBUG_TYPE "pisa-prelegalizer-combiner"

using namespace llvm;
using namespace llvm::MIPatternMatch;

namespace {

#define GET_GICOMBINER_TYPES
#include "PISAGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_TYPES

class PISAPreLegalizerCombinerImpl : public Combiner {
protected:
  const PISAPreLegalizerCombinerImplRuleConfig &RuleConfig;
  const PISASubtarget &STI;
  MachineDominatorTree *MDT;

  // TODO: Make CombinerHelper methods const.
  mutable CombinerHelper Helper;

public:
  PISAPreLegalizerCombinerImpl(
      MachineFunction &MF, CombinerInfo &CInfo, GISelValueTracking &KB,
      GISelCSEInfo *CSEInfo,
      const PISAPreLegalizerCombinerImplRuleConfig &RuleConfig,
      const PISASubtarget &STI, MachineDominatorTree *MDT,
      const LegalizerInfo *LI);

  static const char *getName() { return "PISAGenPreLegalizeGICombiner"; }

  bool tryCombineAllImpl(MachineInstr &MI) const;
  bool tryCombineAll(MachineInstr &I) const override;

  void applyTruncatedLoad(MachineInstr &MI) const;

  bool matchTruncatedStore(MachineInstr &MI) const;
  void applyTruncatedStore(MachineInstr &MI) const;

  bool matchExtendedLoad(MachineInstr &MI) const;
  void applyExtendedLoad(MachineInstr &MI) const;

  bool matchSimplifyNonPowerOf2LoadStoreChain(
      MachineInstr &MI,
      SmallVector<std::pair<MachineInstr *, unsigned>, 8> &Loads,
      MachineInstr *&SizeModificationOp) const;
  void applySimplifyNonPowerOf2LoadStoreChain(
      MachineInstr &MI,
      SmallVector<std::pair<MachineInstr *, unsigned>, 8> &Loads,
      MachineInstr *&SizeModificationOp) const;

  bool matchExpandNonPowerOf2LoadStore(MachineInstr &MI) const;
  void applyExpandNonPowerOf2LoadStore(MachineInstr &MI) const;

  bool matchTruncatedShift(MachineInstr &MI) const;
  void applyTruncatedShift(MachineInstr &MI) const;

  bool matchRedundantMovesPre(MachineInstr &MI) const;
  void applyRedundantMovesPre(MachineInstr &MI) const;

  bool
  matchRcpSqrtToRsqrt(MachineInstr &MI,
                      std::function<void(MachineIRBuilder &)> &MatchInfo) const;
  bool
  matchSubFloorToFrc(MachineInstr &MI,
                     std::function<void(MachineIRBuilder &)> &MatchInfo) const;

  bool matchLaneIdLeftShiftChain(
      MachineInstr &MI,
      std::function<void(MachineIRBuilder &)> &MatchInfo) const;

  bool matchZExtAndToAndZExt(
      MachineInstr &MI,
      std::function<void(MachineIRBuilder &)> &MatchInfo) const;

  bool matchExtractInsertToBitcast(MachineInstr &MI, Register &) const;
  void applyExtractInsertToBitcast(MachineInstr &MI, Register) const;

  bool matchExtractBuildVectorToBitcast(MachineInstr &MI, Register &) const;
  void applyExtractBuildVectorToBitcast(MachineInstr &MI, Register) const;

  bool matchReducePredicates(
      MachineInstr &MI,
      std::function<void(MachineIRBuilder &)> &MatchInfo) const;

  bool matchCmpInt1(MachineInstr &MI,
                    std::function<void(MachineIRBuilder &)> &MatchInfo) const;

  bool matchSelectTruncOneZero(
      MachineInstr &MI,
      std::function<void(MachineIRBuilder &)> &MatchInfo) const;

private:
#define GET_GICOMBINER_CLASS_MEMBERS
#include "PISAGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CLASS_MEMBERS
};

#define GET_GICOMBINER_IMPL
#include "PISAGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_IMPL

PISAPreLegalizerCombinerImpl::PISAPreLegalizerCombinerImpl(
    MachineFunction &MF, CombinerInfo &CInfo, GISelValueTracking &KB,
    GISelCSEInfo *CSEInfo,
    const PISAPreLegalizerCombinerImplRuleConfig &RuleConfig,
    const PISASubtarget &STI, MachineDominatorTree *MDT,
    const LegalizerInfo *LI)
    : Combiner(MF, CInfo, &KB, CSEInfo), RuleConfig(RuleConfig), STI(STI),
      MDT(MDT), Helper(Observer, B, /*IsPreLegalize=*/true, &KB, MDT, LI),
#define GET_GICOMBINER_CONSTRUCTOR_INITS
#include "PISAGenPreLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CONSTRUCTOR_INITS
{
}

bool PISAPreLegalizerCombinerImpl::tryCombineAll(MachineInstr &MI) const {
  // G_FPTRUNC to bf16 needs special constant folding because
  // getFltSemanticForLLT() doesn't support BFloat16 yet.
  if (MI.getOpcode() == TargetOpcode::G_FPTRUNC &&
      MRI.getType(MI.getOperand(0).getReg()).getScalarType().isBFloat16()) {
    const ConstantFP *Cst = nullptr;
    if (mi_match(MI.getOperand(1).getReg(), MRI, m_GFCst(Cst))) {
      APFloat Result(Cst->getValue());
      bool Unused;
      Result.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven, &Unused);
      const ConstantFP *NewCst = ConstantFP::get(B.getContext(), Result);
      MachineIRBuilder Builder(MI);
      Builder.buildFConstant(MI.getOperand(0), *NewCst);
      MI.eraseFromParent();
      return true;
    }
  }

  if (tryCombineAllImpl(MI))
    return true;

  return false;
}

void PISAPreLegalizerCombinerImpl::applyTruncatedLoad(MachineInstr &MI) const {
  llvm::MachineInstr *LoadMI = getDefIgnoringCopies(MI.getOperand(1).getReg(), MRI);
  llvm::MachineMemOperand *MMO = LoadMI->memoperands()[0];
  llvm::MachineOperand Dst = MI.getOperand(0);
  llvm::MachineOperand Addr = LoadMI->getOperand(1);
  llvm::MachineMemOperand *NewMMO = MI.getMF()->getMachineMemOperand(MMO, MMO->getOffset(),
                                                  MRI.getType(Dst.getReg()));
  B.buildLoad(Dst, Addr, *NewMMO);
  MI.eraseFromParent();
}

bool PISAPreLegalizerCombinerImpl::matchExpandNonPowerOf2LoadStore(
    MachineInstr &MI) const {
  assert(MI.getOpcode() == TargetOpcode::G_LOAD ||
         MI.getOpcode() == TargetOpcode::G_STORE);
  GLoadStore &LS = cast<GLoadStore>(MI);

  llvm::TypeSize Size = LS.getMemSizeInBits().getValue();

  if (isPowerOf2_32(Size))
    return false;

  if (LS.getMMO().getMemoryType().isVector())
    return false;

  // Check if this is a load and only has zext uses that are handled by the
  // extended load pattern - if yes, we want to handle it via said pattern.
  if (MI.getOpcode() == TargetOpcode::G_LOAD) {
    for (llvm::MachineOperand &Use : MRI.use_operands(MI.getOperand(0).getReg())) {
      llvm::MachineInstr *Inst = Use.getParent();
      if (Inst->getOpcode() != TargetOpcode::G_ZEXT ||
          !matchExtendedLoad(*Inst))
        return true;
    }
    return false;
  }

  return true;
}
void PISAPreLegalizerCombinerImpl::applyExpandNonPowerOf2LoadStore(
    MachineInstr &MI) const {
  assert(MI.getOpcode() == TargetOpcode::G_LOAD ||
         MI.getOpcode() == TargetOpcode::G_STORE);

  GLoadStore &LS = cast<GLoadStore>(MI);

  llvm::Register PointerReg = LS.getPointerReg();
  llvm::Register ValueReg = LS.getOperand(0).getReg();
  llvm::MachineMemOperand &MMO = LS.getMMO();

  // The remaining size in Bits that still has to be loaded/stored
  ssize_t Size = LS.getMemSizeInBits().getValue();
  // Support sizes that are not a multiple of 8 by "promoting" them to the next
  // multiple of 8. If the size is already a multiple of 8, it is not modified
  Size = (Size + 7) & ~7;

  unsigned Offset = 0;
  const llvm::LLT SizeTy = LLT::integer(Size);

  // If the size was changed to the next power of 8 and we are storing a value,
  // we need to modify the register's type as well.
  // A similar check is needed for loads, but this is done at the end
  if (MI.getOpcode() == TargetOpcode::G_STORE &&
      SizeTy != MRI.getType(ValueReg)) {
    llvm::Register NewValueReg = MRI.createGenericVirtualRegister(SizeTy);
    B.buildZExt(NewValueReg, ValueReg);
    ValueReg = NewValueReg;
  }

  // The register holding the loaded value at the end
  Register LoadRes;
  while (Size > 0) {
    uint64_t OpSize = bit_floor(static_cast<size_t>(Size));
    const unsigned ShiftAmount = Offset * 8;

    const LLT OpTy = LLT::integer(OpSize);

    llvm::MachineMemOperand *NewMMO =
        MI.getMF()->getMachineMemOperand(&MMO, MMO.getOffset() + Offset, OpTy);

    // Stores the (potentially modified) pointer register
    llvm::Register AddrReg = PointerReg;
    // We might need to change the store offset
    if (Offset != 0) {
      llvm::Register NewAddr = MRI.cloneVirtualRegister(AddrReg);

      // Get the pointer size from the pointer register type
      const LLT PtrTy = MRI.getType(AddrReg);
      const LLT IntTy = LLT::integer(PtrTy.getSizeInBits());
      llvm::Register Const = MRI.createGenericVirtualRegister(IntTy);
      B.buildConstant(Const, Offset);
      B.buildPtrAdd(NewAddr, AddrReg, Const);

      AddrReg = NewAddr;
    }

    if (MI.getOpcode() == TargetOpcode::G_STORE) {
      llvm::Register Res = ValueReg;
      // If we're not storing from the start, we need to shift our value first
      if (Offset != 0) {
        llvm::Register ShrRes = MRI.createGenericVirtualRegister(SizeTy);
        llvm::Register ShiftConst = MRI.createGenericVirtualRegister(SizeTy);
        B.buildConstant(ShiftConst, ShiftAmount);
        B.buildLShr(ShrRes, ValueReg, ShiftConst);
        Res = ShrRes;
      }

      // Next, truncate it to the correct size, if needed
      if (SizeTy != OpTy) {
        llvm::Register TruncRes = MRI.createGenericVirtualRegister(OpTy);
        B.buildTrunc(TruncRes, Res);
        Res = TruncRes;
      }

      B.buildStore(Res, AddrReg, *NewMMO);
    } else {
      // When loading, do the same thing but basically in reverse
      // First, load the value
      llvm::Register LoadedValReg = MRI.createGenericVirtualRegister(OpTy);
      B.buildLoad(LoadedValReg, AddrReg, *NewMMO);
      llvm::Register Res = LoadedValReg;

      // Extend it to the correct size, if needed
      if (SizeTy != OpTy) {
        llvm::Register ZextRes = MRI.createGenericVirtualRegister(SizeTy);
        B.buildZExt(ZextRes, LoadedValReg);
        Res = ZextRes;
      }

      // If we're not loading the first Bytes, then we need to shift it
      if (Offset != 0) {
        llvm::Register ShiftRes = MRI.createGenericVirtualRegister(SizeTy);

        llvm::Register ShiftConst = MRI.createGenericVirtualRegister(SizeTy);
        B.buildConstant(ShiftConst, ShiftAmount);

        B.buildShl(ShiftRes, Res, ShiftConst);
        Res = ShiftRes;
      }

      if (!LoadRes.isValid()) {
        LoadRes = MRI.createGenericVirtualRegister(SizeTy);
        B.buildConstant(LoadRes, 0);
      }

      llvm::Register NewLoadRes = MRI.createGenericVirtualRegister(SizeTy);
      B.buildOr(NewLoadRes, LoadRes, Res);
      LoadRes = NewLoadRes;
    }

    Offset += OpSize / 8;
    Size -= OpSize;
  }

  if (MI.getOpcode() == TargetOpcode::G_LOAD) {
    // If we modified the original size of the load (to support sizes that are
    // not a multiple of 8), we need to truncate it to the correct size again,
    // in order not to break any following instructions
    if (SizeTy != MRI.getType(ValueReg)) {
      llvm::Register TruncRes = MRI.createGenericVirtualRegister(MRI.getType(ValueReg));
      B.buildTrunc(TruncRes, LoadRes);
      LoadRes = TruncRes;
    }
    MRI.replaceRegWith(ValueReg, LoadRes);
  }

  MI.eraseFromParent();
}

/// If the value we store resulted from a G_LOAD that the rule above expanded,
/// then we can use the individual load values directly instead of merging into
/// one integer, and then splitting it again. Size modifications between the
/// loads and store (i.e. SEXT/ZEXT/TRUNC) are also supported by
/// truncating/extending the relevant load results (or, in the case of
/// truncation, ignoring some loads entirely)
///
/// Below is a simple example where we simply load and store an i56:
///
///  bb.1.entry:
///    %0:reg32b(p0) = functionParameter_32b 0
/// -> %3:_(s56) = G_ZEXTLOAD %0:reg32b(p0) ::
///      (load (s32) from %ir.dst)
///    %4:_(s56) = G_CONSTANT i56 0
///    %7:_(s32) = G_CONSTANT i32 4
///    %6:reg32b(p0) = G_PTR_ADD %0:reg32b, %7:_(s32)
/// -> %9:_(s56) = G_ZEXTLOAD %6:reg32b(p0) ::
///      (load (s16) from %ir.dst + 4, align 4)
///    %11:_(s56) = G_CONSTANT i56 32
///    %10:_(s56) = G_SHL %9:_, %11:_(s56)
///    %12:_(s56) = G_OR %3:_, %10:_
///    %14:_(s32) = G_CONSTANT i32 6
///    %13:reg32b(p0) = G_PTR_ADD %0:reg32b, %14:_(s32)
/// -> %16:_(s56) = G_ZEXTLOAD %13:reg32b(p0) ::
///      (load (s8) from %ir.dst + 6, align 2, basealign 4)
///    %18:_(s56) = G_CONSTANT i56 48
///    %17:_(s56) = G_SHL %16:_, %18:_(s56)
///    %19:_(s56) = G_OR %12:_, %17:_
///    G_STORE %19:_(s56), %0:reg32b(p0) ::
///      (store (s56) into %ir.dst, align 4)
///    ret
///
/// @param Loads stores all the load instructions that we find along the way.
/// The second parameter in the pair is the offset from which the value was
/// loaded
/// @param SizeModificationOp stores the opcode to the instruction that modified
/// the size of the integer after loading and before storing, if it exists.
/// Possible opcode values are null (none), G_SEXT, G_SEXT_INREG, G_ZEXT, and
/// G_TRUNC
bool PISAPreLegalizerCombinerImpl::matchSimplifyNonPowerOf2LoadStoreChain(
    MachineInstr &MI,
    SmallVector<std::pair<MachineInstr *, unsigned>, 8> &Loads,
    MachineInstr *&SizeModificationOp) const {
  GLoadStore &StoreInst = cast<GLoadStore>(MI);

  llvm::Register ValueReg = StoreInst.getOperand(0).getReg();

  // This has the integer size at the beginning, and each individual load
  // decreases the value by its load size. At the end, this must be zero.
  // NB: We use the size of the store here, promoted to the next multiple
  //  of 8. This size might be modified later if we find a G_*EXT/G_TRUNC.
  unsigned ValueSize = StoreInst.getMemSize().getValue() * 8;

  // This stores the size of the last load instruction in Bytes, s.t. we can
  // verify that the current load is larger than the previous one
  unsigned LastSize = 0;

  SizeModificationOp = nullptr;

  // First, we need to verify the entire "chain" of loads, s.t. we know that
  // there are no further modifications to the value that we want to store.
  // If there is a truncation/extension instruction in between, make note of
  // that, and continue while verifying the load chain using the load size
  // (i.e. size before *ext/trunc)
  MachineInstr *NextChainInst = MRI.getVRegDef(ValueReg);
  if (NextChainInst->getOpcode() == TargetOpcode::G_TRUNC ||
      NextChainInst->getOpcode() == TargetOpcode::G_ZEXT ||
      NextChainInst->getOpcode() == TargetOpcode::G_SEXT ||
      NextChainInst->getOpcode() == TargetOpcode::G_SEXT_INREG) {
    // The "actual" size that was used during loading
    unsigned LoadSize = MRI.getType(NextChainInst->getOperand(1).getReg())
                            .getScalarSizeInBits();

    // We also use G_TRUNC in case the loaded integer was not a multiple of 8.
    // In that case, ValueSize (which was already promoted to the next
    // multiple of 8) would be equal to the total number of Bytes loaded from
    // memory. If not, we need to remember this instruction, and change the
    // expected size to the one that is actually used during loading.
    if (ValueSize != LoadSize ||
        NextChainInst->getOpcode() != TargetOpcode::G_TRUNC) {
      SizeModificationOp = NextChainInst;
      ValueSize = LoadSize;
    }

    // Either way, skip this instruction
    NextChainInst = MRI.getVRegDef(NextChainInst->getOperand(1).getReg());

    // If we loaded an integer that's not a multiple of 8, after which an
    // *ext/trunc operation is done before storing, then there is another
    // G_TRUNC in the way. Simply ignore that one.
    // Example when doing load i38, zext -> i40, store i40:
    //  [...]
    //  %15:_(s38) = G_TRUNC %14:_(s40)
    //  %3:_(s40) = G_ZEXT %15:_(s38)
    //  G_STORE %3:_(s40), %1:reg32b(p0)
    //  [...]
    if (NextChainInst->getOpcode() == TargetOpcode::G_TRUNC) {
      if (LoadSize % 8 == 0)
        return false;
      ValueSize = MRI.getType(NextChainInst->getOperand(1).getReg())
                      .getScalarSizeInBits();
      NextChainInst = MRI.getVRegDef(NextChainInst->getOperand(1).getReg());
    }
  }

  while (ValueSize != 0) {
    // We either start at an G_OR, or at a G_LOAD/G_ZEXTLOAD (meaning we're
    // done)
    if (NextChainInst->getOpcode() == TargetOpcode::G_OR) {
      MachineInstr *NextOr, *ZextLoadMI;
      int64_t LoadShift;
      // First, match the Or-masking
      if (!mi_match(NextChainInst, MRI,
                    m_GOr(m_MInstr(NextOr),
                          m_GShl(m_MInstr(ZextLoadMI), m_ICst(LoadShift)))))
        return false;

      if (ZextLoadMI->getOpcode() != TargetOpcode::G_ZEXTLOAD)
        return false;

      // The shift amount matches the offset used in the load instruction
      MachineInstr *PtrAddInst =
          MRI.getVRegDef(ZextLoadMI->getOperand(1).getReg());
      if (PtrAddInst->getOpcode() != TargetOpcode::G_PTR_ADD)
        return false;

      std::optional<llvm::APInt> LoadOffset =
          getIConstantVRegVal(PtrAddInst->getOperand(2).getReg(), MRI);
      if (!LoadOffset.has_value() || LoadOffset.value() != LoadShift / 8)
        return false;

      // Verify the LLT that was loaded
      LLT LoadTy = cast<GZExtLoad>(ZextLoadMI)->getMMO().getType();
      if (!LoadTy.isScalar() || LoadTy.getScalarSizeInBits() <= LastSize ||
          ValueSize <= LastSize)
        return false;

      LastSize = LoadTy.getScalarSizeInBits();
      ValueSize -= LastSize;
      NextChainInst = NextOr;

      Loads.push_back({ZextLoadMI, static_cast<unsigned>(
                                       LoadOffset.value().getZExtValue())});
    } else if (NextChainInst->getOpcode() == TargetOpcode::G_ZEXTLOAD ||
               NextChainInst->getOpcode() == TargetOpcode::G_LOAD) {
      const llvm::GLoadStore &LoadInst = cast<GLoadStore>(*NextChainInst);
      const LLT LoadTy = LoadInst.getMMO().getType();
      if (!LoadTy.isScalar())
        return false;

      // This is the last load instruction of the chain, so it must match the
      // remaining value in ValueSize
      unsigned LoadSize = LoadTy.getScalarSizeInBits();
      if (ValueSize % 8 == 0 && LoadSize != ValueSize)
        return false;

      // To avoid endless loops with single loads, sort out the bad cases here
      if (Loads.empty()) {
        // We just load and store a value, there is nothing to optimize here
        if (!SizeModificationOp)
          return false;

        // There is no optimization potential for zero extensions or simple
        // truncations, only for sign extension
        if (SizeModificationOp->getOpcode() == TargetOpcode::G_ZEXT ||
            SizeModificationOp->getOpcode() == TargetOpcode::G_TRUNC)
          return false;

        unsigned SizeAfterModificationOp =
            MRI.getType(SizeModificationOp->getOperand(0).getReg())
                .getScalarSizeInBits();

        // If the size does not significantly change after sign extension, then
        // there's nothing to do for us here, the code is already optimal
        if ((isPowerOf2_32(SizeAfterModificationOp)
                 ? SizeAfterModificationOp
                 : NextPowerOf2(SizeAfterModificationOp)) == LoadSize)
          return false;

        // If the G_SEXT source size equals the load size and the source is
        // byte-aligned, applying this transformation would reconstruct the
        // same G_SEXT(G_LOAD) pattern, causing an infinite loop. When the
        // source size is not byte-aligned, the apply inserts in-place
        // shl/ashr to align it first, producing a different pattern.
        unsigned SextSrcSize =
            MRI.getType(SizeModificationOp->getOperand(1).getReg())
                .getScalarSizeInBits();
        if (LoadSize == SextSrcSize && SextSrcSize % 8 == 0)
          return false;

        // TODO: revisit
        // store(sext i63 (load i8)) results in infinite loop.
        if (!isPowerOf2_32(SizeAfterModificationOp) &&
            (SizeAfterModificationOp != LoadSize))
          return false;
      }

      // Done
      Loads.push_back({NextChainInst, 0});
      return true;
    } else {
      return false;
    }
  }
  return false;
}
void PISAPreLegalizerCombinerImpl::applySimplifyNonPowerOf2LoadStoreChain(
    MachineInstr &MI,
    SmallVector<std::pair<MachineInstr *, unsigned>, 8> &Loads,
    MachineInstr *&SizeModificationOp) const {
  // If the chain is valid, erase our single G_STORE and convert it into
  // multiple G_STOREs using the individually loaded values
  GLoadStore &StoreInst = cast<GLoadStore>(MI);

  llvm::Register PointerReg = StoreInst.getPointerReg();
  llvm::MachineMemOperand &MMO = StoreInst.getMMO();

  unsigned StoreSizeInBytes = StoreInst.getMemSize().getValue();

  for (size_t Index = 0; Index < Loads.size(); ++Index) {
    std::pair<MachineInstr *, unsigned> &Pair = Loads[Index];
    MachineInstr *LoadMI = Pair.first;
    unsigned Offset = Pair.second;

    Register Value = LoadMI->getOperand(0).getReg();
    LLT LoadTy = cast<GLoadStore>(LoadMI)->getMMO().getType();

    Register Res = Value;
    if (LoadTy.getSizeInBytes() + Offset <= StoreSizeInBytes) {
      if (LoadMI->getOpcode() == TargetOpcode::G_ZEXTLOAD) {
        // Since they are ZextLoad instructions, we need to change the size
        // back to the loaded size. This instruction will not be visible in
        // PISA later, but is required here for type correctness.
        Res = MRI.createGenericVirtualRegister(LoadTy);
        B.buildTrunc(Res, Value);
      }

      // This is the last load (i.e. the first entry in the vector) and the
      // load was extended, so we need to sext/zext this value before
      // storing
      if (Index == 0 && SizeModificationOp != nullptr) {
        unsigned Opcode = SizeModificationOp->getOpcode();
        assert(Opcode == TargetOpcode::G_ZEXT ||
               Opcode == TargetOpcode::G_SEXT ||
               Opcode == TargetOpcode::G_SEXT_INREG);

        if (Opcode == TargetOpcode::G_SEXT_INREG ||
            Opcode == TargetOpcode::G_SEXT) {
          // SEXT is a little trickier
          unsigned StoreSizeInBits = StoreInst.getMemSizeInBits().getValue();
          unsigned OriginalSize =
              Opcode == TargetOpcode::G_SEXT_INREG
                  ? (SizeModificationOp->getOperand(2).getImm())
                  : MRI.getType(SizeModificationOp->getOperand(1).getReg())
                        .getScalarSizeInBits();

          // First, do we have to sign extend this value in-place using
          // shifts?
          if (OriginalSize % 8 != 0) {
            unsigned ExtendBy = 8 - (OriginalSize % 8);

            llvm::Register ConstReg = MRI.createGenericVirtualRegister(LLT::integer(32));
            B.buildConstant(ConstReg, ExtendBy);

            if (LoadTy.getSizeInBits() < OriginalSize + ExtendBy) {
              LoadTy = LLT::integer(OriginalSize + ExtendBy);
              llvm::Register ZExtRes = MRI.createGenericVirtualRegister(LoadTy);
              B.buildZExt(ZExtRes, Res);
              Res = ZExtRes;
            }

            llvm::Register ShlRes = MRI.cloneVirtualRegister(Res);
            B.buildShl(ShlRes, Res, ConstReg);

            llvm::Register AShrRes = MRI.cloneVirtualRegister(ShlRes);
            B.buildAShr(AShrRes, ShlRes, ConstReg);

            Res = AShrRes;
            // Update the size to reflect the extension
            OriginalSize = (OriginalSize + 7) & ~7;
          }

          // Second, do we still have to SEXT it further?
          if (StoreSizeInBits > OriginalSize) {
            LoadTy = LLT::integer(StoreSizeInBits);
            llvm::Register SextRes = MRI.createGenericVirtualRegister(LoadTy);
            B.buildSExt(SextRes, Res);

            Res = SextRes;
          }
        } else {
          // G_ZEXT, simple
          unsigned NewSize = StoreSizeInBytes - Offset;
          assert(NewSize > LoadTy.getSizeInBytes());

          LoadTy = LLT::integer(NewSize * 8);
          llvm::Register ExtRes = MRI.createGenericVirtualRegister(LoadTy);
          B.buildZExt(ExtRes, Res);

          Res = ExtRes;
        }
      }
    } else {
      assert(SizeModificationOp->getOpcode() == TargetOpcode::G_TRUNC);

      // The value was truncated before storing. There are two cases now:
      //  - we need to truncate this value to the correct size
      //  - we need to ignore this one, as it is "out of range"
      if (Offset >= StoreSizeInBytes) {
        continue;
      }
      LoadTy = LLT::integer((StoreSizeInBytes - Offset) * 8);

      Res = MRI.createGenericVirtualRegister(LoadTy);
      B.buildTrunc(Res, Value);
    }

    llvm::MachineMemOperand *NewMMO = MI.getMF()->getMachineMemOperand(
        &MMO, MMO.getOffset() + Offset, LoadTy);

    // Stores the (potentially modified) pointer register
    llvm::Register AddrReg = PointerReg;

    // Add the offset to the pointer reg if the offset is not zero
    if (Offset != 0) {
      llvm::Register NewPointerReg =
          MRI.createGenericVirtualRegister(MRI.getType(AddrReg));

      // Get the pointer size from the pointer register type
      const LLT PtrTy = MRI.getType(AddrReg);
      const LLT IntTy = LLT::integer(PtrTy.getSizeInBits());
      llvm::Register CstReg = MRI.createGenericVirtualRegister(IntTy);
      B.buildConstant(CstReg, Offset);

      B.buildPtrAdd(NewPointerReg, AddrReg, CstReg);

      AddrReg = NewPointerReg;
    }
    B.buildStore(Res, AddrReg, *NewMMO);
  }

  MI.eraseFromParent();
  return;
}

// Matches following pattern:
// (s24) = G_TRUNC (s32)
// G_STORE (s24), ptr
bool PISAPreLegalizerCombinerImpl::matchTruncatedStore(MachineInstr &MI) const {
  llvm::GStore &Store = cast<GStore>(MI);
  llvm::MachineInstr *TruncMI = getDefIgnoringCopies(Store.getValueReg(), MRI);
  if (!TruncMI)
    return false;
  if (TruncMI->getOpcode() != TargetOpcode::G_TRUNC)
    return false;
  if (!isPowerOf2_32(
          MRI.getType(TruncMI->getOperand(1).getReg()).getSizeInBits()))
    return false;

  llvm::TypeSize Size = Store.getMemSizeInBits().getValue();
  uint64_t Align = Store.getMMO().getAlign().value();

  if (isPowerOf2_32(Size))
    return false;

  uint64_t NextPow2 = NextPowerOf2(Size);
  llvm::TypeSize TruncSize = MRI.getType(TruncMI->getOperand(1).getReg()).getSizeInBits();
  if (TruncSize != NextPow2)
    return false;
  uint64_t Remainder = NextPow2 - Size;
  if (NextPow2 % Remainder != 0)
    return false;

  if (Remainder % 8 != 0 || Remainder > 64 || !isPowerOf2_32(Remainder))
    return false;

  // We will create a vector store of smaller elements, which has to be aligned
  // to the bit width of the vector element.
  if (Align % (Remainder / 8) != 0)
    return false;

  uint64_t VecSize = NextPow2 / Remainder;
  if (VecSize > 4)
    return false;

  return true;
}

// Changes the type of a non-power-of-2 store to a vector of smaller elements,
// if it comes from a trunc instruction.
void PISAPreLegalizerCombinerImpl::applyTruncatedStore(MachineInstr &MI) const {
  llvm::GStore &Store = cast<GStore>(MI);

  llvm::MachineInstr *TruncMI = getDefIgnoringCopies(Store.getValueReg(), MRI);
  assert(TruncMI && TruncMI->getOpcode() == TargetOpcode::G_TRUNC);

  llvm::MachineOperand TruncVal = TruncMI->getOperand(1);
  llvm::TypeSize Size = Store.getMemSizeInBits().getValue();
  uint64_t NextPow2 = NextPowerOf2(Size);
  uint64_t Remainder = NextPow2 - Size;
  uint64_t VecSize = NextPow2 / Remainder;
  uint64_t VecSizeSmall = Size / Remainder;
  llvm::LLT VecType = LLT::fixed_vector(VecSize, LLT::integer(Remainder));
  llvm::LLT VecTypeSmall = LLT::fixed_vector(VecSizeSmall, LLT::integer(Remainder));
  llvm::Register TmpDst = MRI.createGenericVirtualRegister(VecType);
  llvm::Register TmpDstSmall = MRI.createGenericVirtualRegister(VecTypeSmall);
  B.buildBitcast(TmpDst, TruncVal);
  B.buildShuffleVector(TmpDstSmall, TmpDst, B.buildUndef(VecType), {0, 1, 2});

  llvm::Register Addr = Store.getPointerReg();
  llvm::MachineMemOperand &MMO = Store.getMMO();
  llvm::MachineMemOperand *NewMMO =
      MI.getMF()->getMachineMemOperand(&MMO, MMO.getOffset(), VecTypeSmall);
  B.buildStore(TmpDstSmall, Addr, *NewMMO);
  MI.eraseFromParent();
}

// Matches a G_ZEXT that extends the result of a load that has a non-power-of-2
// type. It might be profitable to do a wider load and mask out the bits.
// This saves us from using second load instruction and few arithmetic
// instructions (shl, and, or) to zero extend.
bool PISAPreLegalizerCombinerImpl::matchExtendedLoad(MachineInstr &MI) const {
  llvm::MachineInstr *DefMI = getDefIgnoringCopies(MI.getOperand(1).getReg(), MRI);
  if (!DefMI)
    return false;
  if (DefMI->getOpcode() != TargetOpcode::G_LOAD)
    return false;
  GLoad &Load = cast<GLoad>(*DefMI);

  llvm::TypeSize Size = Load.getMemSizeInBits().getValue();
  uint64_t Align = Load.getMMO().getAlign().value();

  if (isPowerOf2_32(Size))
    return false;

  llvm::TypeSize ZextSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
  if (!isPowerOf2_32(ZextSize))
    return false;

  if (ZextSize > 64)
    return false;

  if (Align % (ZextSize / 8) != 0)
    return false;

  return true;
}

void PISAPreLegalizerCombinerImpl::applyExtendedLoad(MachineInstr &MI) const {
  GLoad *LoadMI =
      cast<GLoad>(getDefIgnoringCopies(MI.getOperand(1).getReg(), MRI));
  assert(LoadMI);

  llvm::TypeSize DestSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
  llvm::TypeSize LoadSize = LoadMI->getMemSizeInBits().getValue();

  APInt Mask = APInt::getLowBitsSet(DestSize, LoadSize);

  llvm::LLT NewLoadSize = LLT::integer(DestSize);
  llvm::Register LoadDst = MRI.createGenericVirtualRegister(NewLoadSize);

  llvm::MachineOperand Addr = LoadMI->getOperand(1);
  llvm::MachineMemOperand &MMO = LoadMI->getMMO();
  llvm::MachineMemOperand *NewMMO = MI.getMF()->getMachineMemOperand(&MMO, MMO.getOffset(),
                                                  MRI.getType(LoadDst));

  B.buildLoad(LoadDst, Addr, *NewMMO);
  llvm::Register MaskReg = MRI.createGenericVirtualRegister(LLT::integer(DestSize));
  B.buildConstant(MaskReg, Mask.getZExtValue());
  B.buildAnd(MI.getOperand(0).getReg(), LoadDst, MaskReg);
  MI.eraseFromParent();
}

// i32 %lo = G_TRUNC i32 a 16
// i32 %shift = G_LSHR i32 a 16
// %hi = G_TRUNC i32 %shift to i16
// => %1 = G_BITCAST i32 a to <2 x 16>
// => %lo = G_EXTRACT_VECTOR_ELT <2 x 16> %1, 0
// => %hi = G_EXTRACT_VECTOR_ELT <2 x 16> %1, 1
bool PISAPreLegalizerCombinerImpl::matchTruncatedShift(MachineInstr &MI) const {
  llvm::MachineInstr &TruncMI = MI;
  llvm::MachineInstr &LshMI = *getDefIgnoringCopies(TruncMI.getOperand(1).getReg(), MRI);
  if (LshMI.getOpcode() == TargetOpcode::G_LSHR) {
    llvm::LLT WideTy = MRI.getType(TruncMI.getOperand(1).getReg());
    llvm::LLT NarrowTy = MRI.getType(TruncMI.getOperand(0).getReg());
    unsigned WideSize = WideTy.getScalarSizeInBits();
    unsigned NarrowSize = NarrowTy.getScalarSizeInBits();
    if (WideTy.isScalar() && NarrowTy.isScalar()) {
      if (isPowerOf2_32(WideSize) && isPowerOf2_32(NarrowSize)) {
        llvm::Register ShiftReg = LshMI.getOperand(2).getReg();
        if (std::optional<llvm::ValueAndVReg> Shift =
                getIConstantVRegValWithLookThrough(ShiftReg, MRI)) {
          if (Shift.has_value()) {
            uint64_t ShiftVal = Shift->Value.getZExtValue();
            if ((NarrowSize + ShiftVal) == WideSize)
              return true;
          }
        }
      }
    }
  }
  return false;
}
void PISAPreLegalizerCombinerImpl::applyTruncatedShift(MachineInstr &MI) const {
  llvm::MachineInstr &TruncMI = MI;
  llvm::MachineInstr &LshMI = *getDefIgnoringCopies(TruncMI.getOperand(1).getReg(), MRI);
  llvm::LLT WideTy = MRI.getType(TruncMI.getOperand(1).getReg());
  llvm::LLT NarrowTy = MRI.getType(TruncMI.getOperand(0).getReg());
  unsigned WideSize = WideTy.getScalarSizeInBits();
  unsigned NarrowSize = NarrowTy.getScalarSizeInBits();

  unsigned VecLen = WideSize / NarrowSize;
  llvm::LLT VecTy = LLT::fixed_vector(VecLen, NarrowTy);
  llvm::Register VecReg = MRI.createGenericVirtualRegister(VecTy);

  // trunc to extract high part
  llvm::MachineInstrBuilder BitCast = B.buildBitcast(VecReg, LshMI.getOperand(1));
  B.buildExtractVectorElementConstant(TruncMI.getOperand(0), VecReg,
                                      VecLen - 1);

  // find trunc to extract low part (if any)
  llvm::Register ShiftSrc = LshMI.getOperand(1).getReg();
  for (llvm::MachineInstr &UseMI : MRI.use_instructions(ShiftSrc)) {
    if (UseMI.getOpcode() == TargetOpcode::G_TRUNC) {
      llvm::TypeSize DstSize = MRI.getType(UseMI.getOperand(0).getReg()).getSizeInBits();
      if (DstSize == NarrowSize) {
        assert(MDT && "machine dominator pass must be available");
        if (!MDT->dominates(&UseMI, &TruncMI) &&
            !MDT->dominates(&TruncMI, &UseMI))
          continue; // can not optimize out low part
        MachineIRBuilder MIB(UseMI);
        llvm::MachineInstrBuilder Lo = MIB.buildExtractVectorElementConstant(UseMI.getOperand(0),
                                                        VecReg, 0);
        if (MDT->dominates(&UseMI, &TruncMI))
          BitCast->moveBefore(Lo);
        UseMI.eraseFromParent();
      }
    }
  }
  TruncMI.eraseFromParent();
}

// A(16) = G_EXTRACT_VECTOR_ELT ARG(32), 0
// B(16) = G_EXTRACT_VECTOR_ELT ARG(32), 1
// C(<2x16>) = G_BUILD_VECTOR A, B
// D(32) = G_BITCAST C(<2x16>)
// => D(32) = COPY ARG(32)
bool PISAPreLegalizerCombinerImpl::matchRedundantMovesPre(
    MachineInstr &MI) const {
  llvm::MachineInstr &BitcastMI = MI;
  llvm::Register DstReg = BitcastMI.getOperand(0).getReg();
  llvm::Register SrcReg = BitcastMI.getOperand(1).getReg();
  if (MRI.getType(DstReg).isVector() || !MRI.getType(SrcReg).isVector())
    return false;

  llvm::MachineInstr &BuildVecMI = *getDefIgnoringCopies(SrcReg, MRI);
  if (BuildVecMI.getOpcode() != TargetOpcode::G_BUILD_VECTOR)
    return false;

  unsigned Mask = 0;
  Register VecReg = 0;
  for (unsigned I = 1; I < BuildVecMI.getNumOperands(); I++) {
    llvm::MachineInstr &ExtractMI =
        *getDefIgnoringCopies(BuildVecMI.getOperand(I).getReg(), MRI);
    if (ExtractMI.getOpcode() != TargetOpcode::G_EXTRACT_VECTOR_ELT)
      return false;
    if (I == 1) {
      VecReg = ExtractMI.getOperand(1).getReg();
    } else if (VecReg != ExtractMI.getOperand(1).getReg()) {
      // must extract indices from the same vector
      return false;
    }
    llvm::Register IndexReg = ExtractMI.getOperand(2).getReg();
    std::optional<llvm::ValueAndVReg> Index = getIConstantVRegValWithLookThrough(IndexReg, MRI);
    if (!Index.has_value())
      return false;
    // indices must be in the same order
    if (Index->Value.getZExtValue() != (I - 1))
      return false;
    Mask |= 1 << Index->Value.getZExtValue();
  }
  // check that all indices have been extracted
  if (Mask != (1u << MRI.getType(SrcReg).getNumElements()) - 1)
    return false;

  // do not consider creating <2x16> from e.g. <3x16>
  if (MRI.getType(DstReg).getSizeInBits() !=
      MRI.getType(VecReg).getSizeInBits())
    return false;

  return true;
}
void PISAPreLegalizerCombinerImpl::applyRedundantMovesPre(
    MachineInstr &MI) const {
  llvm::MachineInstr &BitcastMI = MI;
  llvm::Register DstReg = BitcastMI.getOperand(0).getReg();
  llvm::Register SrcReg = BitcastMI.getOperand(1).getReg();
  llvm::MachineInstr &BuildVecMI = *getDefIgnoringCopies(SrcReg, MRI);
  llvm::MachineInstr &ExtractMI =
      *getDefIgnoringCopies(BuildVecMI.getOperand(1).getReg(), MRI);
  llvm::Register VecReg = ExtractMI.getOperand(1).getReg();

  if (MRI.getType(DstReg) == MRI.getType(VecReg))
    B.buildCopy(DstReg, VecReg);
  else
    B.buildBitcast(DstReg, VecReg);
  MI.eraseFromParent();
}

/// Returns divisor if operation is frcp intrinsic or fdiv with dividend equal
/// to constant one.
static MachineInstr *getReciprocalDivisor(MachineInstr *MI,
                                          MachineRegisterInfo &MRI) {
  if (MI->getOpcode() == TargetOpcode::G_FDIV) {
    // Check for 1.0 / x pattern. Avoid m_GFCstOrSplat because
    // getFConstantVRegValWithLookThrough loses bfloat16 semantics.
    Register Dividend = MI->getOperand(1).getReg();
    llvm::MachineInstr *DivDefMI = getDefIgnoringCopies(Dividend, MRI);
    if (!DivDefMI || DivDefMI->getOpcode() != TargetOpcode::G_FCONSTANT)
      return nullptr;
    if (!DivDefMI->getOperand(1).getFPImm()->isExactlyValue(1.0))
      return nullptr;
    return getDefIgnoringCopies(MI->getOperand(2).getReg(), MRI);
  }

  llvm::GIntrinsic *GI = dyn_cast<GIntrinsic>(MI);
  if (GI && GI->is(Intrinsic::pisa_frcp))
    return getDefIgnoringCopies(MI->getOperand(2).getReg(), MRI);

  return nullptr;
}

// 1/(sqrt(x))         -> frsqrt(x)
// frcp(sqrt(x))       -> frsqrt(x)
// sqrt(1/(x))         -> frsqrt(x)
// sqrt(frcp(x))       -> frsqrt(x)
// 1/(fabs(sqrt(x)))   -> fabs(frsqrt(x))
// frcp(fabs(sqrt(x))) -> fabs(frsqrt(x))
bool PISAPreLegalizerCombinerImpl::matchRcpSqrtToRsqrt(
    MachineInstr &MI,
    std::function<void(MachineIRBuilder &)> &MatchInfo) const {
  std::function<MachineInstr *(MachineInstr *)> GetFrcpSrc =
      [=](MachineInstr *MI) -> MachineInstr * {
    if (!MI || !MI->getFlag(MachineInstr::FmContract))
      return nullptr;
    return getReciprocalDivisor(MI, MRI);
  };
  std::function<MachineInstr *(MachineInstr *)> GetSqrtSrc =
      [=](MachineInstr *MI) -> MachineInstr * {
    if (!MI || !MI->getFlag(MachineInstr::FmContract))
      return nullptr;
    if (MI->getOpcode() == TargetOpcode::G_FSQRT)
      return getDefIgnoringCopies(MI->getOperand(1).getReg(), MRI);
    return nullptr;
  };

  bool WrapInFAbs = false;
  MachineInstr *Divisor = GetFrcpSrc(&MI);
  if (Divisor) {
    // result of sqrt cannot be snan, so Intrinsic::fabs is replaced with
    // Intrinsic::pisa_fabs. We won't get G_FABS here.
    llvm::GIntrinsic *GI = dyn_cast<GIntrinsic>(Divisor);
    if (GI && GI->is(Intrinsic::pisa_fabs)) {
      WrapInFAbs = true;
      Divisor = getDefIgnoringCopies(GI->getOperand(2).getReg(), MRI);
    }
  }

  MachineInstr *InnerMI = GetSqrtSrc(Divisor);
  if (!InnerMI) {
    WrapInFAbs = false;
    InnerMI = GetFrcpSrc(GetSqrtSrc(&MI));
  }
  if (!InnerMI)
    return false;

  LLT DstTy = MRI.getType(MI.getOperand(0).getReg());
  MatchInfo = [InnerMI, WrapInFAbs, DstTy, &MI](MachineIRBuilder &B) {
    Register Src = InnerMI->getOperand(0).getReg();
    if (!WrapInFAbs) {
      B.buildIntrinsic(Intrinsic::pisa_frsqrt, {MI.getOperand(0)})
          .addUse(Src)
          .setMIFlags(MI.getFlags());
      return;
    }
    // 1/|sqrt(x)| == |frsqrt(x)|
    llvm::MachineInstrBuilder Frsqrt = B.buildIntrinsic(Intrinsic::pisa_frsqrt, {DstTy})
                      .addUse(Src)
                      .setMIFlags(MI.getFlags());
    B.buildIntrinsic(Intrinsic::pisa_fabs, {MI.getOperand(0)})
        .addUse(Frsqrt.getReg(0))
        .setMIFlags(MI.getFlags());
  };
  return true;
}

// s32 %floor = G_FFLOOR %x
// s32 %dst   = G_FSUB %x, %floor
// => s32 %dst = frc %x
bool PISAPreLegalizerCombinerImpl::matchSubFloorToFrc(
    MachineInstr &MI,
    std::function<void(MachineIRBuilder &)> &MatchInfo) const {
  Register XReg, FloorReg;

  bool IsPlainFSub = MI.getOpcode() == TargetOpcode::G_FSUB;
  if (IsPlainFSub) {
    XReg = MI.getOperand(1).getReg();
    FloorReg = MI.getOperand(2).getReg();
  } else if (llvm::GIntrinsic *GI = dyn_cast<GIntrinsic>(&MI);
             GI && GI->is(Intrinsic::pisa_fsub)) {
    unsigned NumOps = MI.getNumOperands();
    llvm::RoundingMode Round = static_cast<RoundingMode>(MI.getOperand(NumOps - 2).getImm());
    int64_t Sat = MI.getOperand(NumOps - 1).getImm();
    if (Round != RoundingMode::TowardZero || Sat)
      return false;
    XReg = MI.getOperand(2).getReg();
    FloorReg = MI.getOperand(3).getReg();
  } else {
    return false;
  }

  llvm::TypeSize BitWidth = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
  if (BitWidth != 32)
    return false;

  MachineInstr *FloorMI = getDefIgnoringCopies(FloorReg, MRI);
  if (!FloorMI || FloorMI->getOpcode() != TargetOpcode::G_FFLOOR)
    return false;

  if (IsPlainFSub && (!MI.getFlag(MachineInstr::FmContract) ||
                      !FloorMI->getFlag(MachineInstr::FmContract)))
    return false;

  if (FloorMI->getOperand(1).getReg() != XReg)
    return false;

  MatchInfo = [&MI, XReg](MachineIRBuilder &B) {
    B.buildIntrinsic(Intrinsic::pisa_frc, {MI.getOperand(0)})
        .addUse(XReg)
        .setMIFlags(MI.getFlags());
  };
  return true;
}

// Fold `shl (zext (shl lane_id, S1)), S2` to `zext (shl lane_id, S1+S2)`.
// lane_id currently fits in 5 bits.
// The second shift could only appear from irtranslator pass.
//
// Pattern:
//   %5:_(s32) = G_CONSTANT i32 3
//   %4:_(s32) = G_INTRINSIC intrinsic(@llvm.pisa.lane.id)
//   %6:_(s32) = nuw nsw G_SHL %4:_, %5:_(s32)
//   %7:_(s64) = nneg G_ZEXT %6:_(s32)
//   %31:_(s64) = G_CONSTANT i64 2
//   %9:_(s64) = nuw nsw G_SHL %7:_, %31:_(s64)
// =>
//   %4:_(s32) = G_INTRINSIC intrinsic(@llvm.pisa.lane.id)
//   %5:_(s32) = G_CONSTANT i32 5
//   %6:_(s32) = nuw nsw G_SHL %4:_, %5:_(s32)
//   %7:_(s64) = nneg G_ZEXT %6:_(s32)
bool PISAPreLegalizerCombinerImpl::matchLaneIdLeftShiftChain(
    MachineInstr &MI,
    std::function<void(MachineIRBuilder &)> &MatchInfo) const {
  int64_t ShiftImm1, ShiftImm2;
  MachineInstr *ZExt = nullptr;
  MachineInstr *Intr = nullptr;
  if (!mi_match(MI, MRI, m_GShl(m_MInstr(ZExt), m_ICst(ShiftImm1))))
    return false;
  if (!mi_match(ZExt, MRI, m_GZExt(m_GShl(m_MInstr(Intr), m_ICst(ShiftImm2)))))
    return false;
  llvm::GIntrinsic *LaneId = dyn_cast<GIntrinsic>(Intr);
  if (!LaneId || LaneId->getIntrinsicID() != Intrinsic::pisa_lane_id)
    return false;

  llvm::LLT LaneIdTy = MRI.getType(LaneId->getOperand(0).getReg());
  constexpr int MaxNumBitsInLaneId = 5;
  if ((ShiftImm1 + ShiftImm2) >=
      (static_cast<int>(LaneIdTy.getSizeInBits()) - MaxNumBitsInLaneId - 1))
    return false;

  MatchInfo = [=, &MI](MachineIRBuilder &B) {
    llvm::Register NewShlReg = MRI.createGenericVirtualRegister(LaneIdTy);
    B.buildShl(NewShlReg, LaneId->getOperand(0),
               B.buildConstant(LaneIdTy, ShiftImm1 + ShiftImm2), MI.getFlags());
    B.buildZExt(MI.getOperand(0), NewShlReg, ZExt->getFlags());
  };
  return true;
}

// G_AND (G_ZEXT x), C -> G_ZEXT (G_AND x, trunc(C)).
// CodeGenPrepare can form the widened mask, but keeping the mask narrow lets
// address folding see the original arithmetic shape.
bool PISAPreLegalizerCombinerImpl::matchZExtAndToAndZExt(
    MachineInstr &MI,
    std::function<void(MachineIRBuilder &)> &MatchInfo) const {
  Register ZExtReg = MI.getOperand(1).getReg();
  Register MaskReg = MI.getOperand(2).getReg();
  llvm::MachineInstr *ZExtMI = getDefIgnoringCopies(ZExtReg, MRI);
  std::optional<llvm::ValueAndVReg> Mask = getIConstantVRegValWithLookThrough(MaskReg, MRI);

  if ((!ZExtMI || ZExtMI->getOpcode() != TargetOpcode::G_ZEXT ||
       !Mask.has_value())) {
    std::swap(ZExtReg, MaskReg);
    ZExtMI = getDefIgnoringCopies(ZExtReg, MRI);
    Mask = getIConstantVRegValWithLookThrough(MaskReg, MRI);
  }

  if (!ZExtMI || ZExtMI->getOpcode() != TargetOpcode::G_ZEXT ||
      !Mask.has_value())
    return false;

  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = ZExtMI->getOperand(1).getReg();
  LLT DstTy = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(SrcReg);
  if (!DstTy.isScalar() || !SrcTy.isScalar() || SrcTy.getSizeInBits() != 32 ||
      DstTy.getSizeInBits() != 64)
    return false;

  APInt NarrowMask = Mask->Value.zextOrTrunc(SrcTy.getSizeInBits());
  unsigned ZExtFlags = ZExtMI->getFlags();
  MatchInfo = [=](MachineIRBuilder &B) {
    llvm::MachineInstrBuilder NarrowMaskReg = B.buildConstant(SrcTy, NarrowMask);
    llvm::MachineInstrBuilder NarrowAnd = B.buildAnd(SrcTy, SrcReg, NarrowMaskReg);
    B.buildZExt(DstReg, NarrowAnd, ZExtFlags);
  };
  return true;
}

// Given the following pattern
//
// %144:_(s8) = G_EXTRACT_VECTOR_ELT %4:_(<32 x s8>), %145:_(s32)
// %146:_(s32) = G_ZEXT %144:_(s8)
// %147:_(s32) = nuw G_SHL %146:_, %23:_(s32)
// %148:_(s32) = disjoint G_OR %143:_, %147:
//
// record the following info using BaseReg (%148) as input
// - SrcVecReg = %4
// - SrcVecIdx = 1 << %145
// - bits of s32 (typeof(BaseReg)) written to by above - return value
static uint64_t getCoveredBits(Register BaseReg, Register &SrcVecReg,
                               uint64_t *SrcVecIdx, MachineRegisterInfo &MRI) {
  llvm::MachineInstr &SrcMI = *getDefIgnoringCopies(BaseReg, MRI);
  switch (SrcMI.getOpcode()) {
  case TargetOpcode::G_EXTRACT_VECTOR_ELT: {
    llvm::Register VecReg = SrcMI.getOperand(1).getReg();
    if (MRI.getType(VecReg).getNumElements() > 64)
      return 0; // not supported
    llvm::Register IdxReg = SrcMI.getOperand(2).getReg();
    if (SrcVecReg && (SrcVecReg != VecReg))
      return 0; // extracting from different source vector ?
    SrcVecReg = VecReg;
    std::optional<llvm::ValueAndVReg> Index = getIConstantVRegValWithLookThrough(IdxReg, MRI);
    if (!Index.has_value())
      return 0; // not a constant
    *SrcVecIdx |= (1ull << Index->Value.getZExtValue());
    return ~0;
  } break;
  case TargetOpcode::G_ZEXT: {
    llvm::Register SrcReg = SrcMI.getOperand(1).getReg();
    llvm::MachineInstr &ExtractMI = *getDefIgnoringCopies(SrcReg, MRI);
    if (ExtractMI.getOpcode() != TargetOpcode::G_EXTRACT_VECTOR_ELT)
      return 0;
    if (getCoveredBits(SrcReg, SrcVecReg, SrcVecIdx, MRI))
      return (1ull << MRI.getType(SrcReg).getScalarSizeInBits()) - 1;
  } break;
  case TargetOpcode::G_SHL: {
    llvm::Register SrcReg = SrcMI.getOperand(1).getReg();
    llvm::Register ShiftReg = SrcMI.getOperand(2).getReg();
    llvm::MachineInstr &ExtMI = *getDefIgnoringCopies(SrcReg, MRI);
    if (ExtMI.getOpcode() != TargetOpcode::G_ZEXT)
      return 0;
    std::optional<llvm::ValueAndVReg> Shift = getIConstantVRegValWithLookThrough(ShiftReg, MRI);
    if (!Shift.has_value())
      return 0; // not a constant
    uint64_t ShiftValue = Shift->Value.getZExtValue();
    // make sure we are not swapping bits
    uint64_t OldSrcVecIdx = *SrcVecIdx;
    uint64_t Bits = getCoveredBits(SrcReg, SrcVecReg, SrcVecIdx, MRI);
    int BitSet = llvm::countr_zero(*SrcVecIdx & ~OldSrcVecIdx);
    uint64_t Offset = (BitSet * MRI.getType(SrcVecReg).getScalarSizeInBits()) %
                  MRI.getType(SrcReg).getSizeInBits();
    if (ShiftValue != Offset)
      return 0; // will not insert at proper offset
    return Bits << ShiftValue;
  } break;
  case TargetOpcode::G_OR: {
    llvm::Register LHSReg = SrcMI.getOperand(1).getReg();
    llvm::Register RHSReg = SrcMI.getOperand(2).getReg();
    unsigned LHSOp = getDefIgnoringCopies(LHSReg, MRI)->getOpcode();
    unsigned RHSOp = getDefIgnoringCopies(RHSReg, MRI)->getOpcode();
    if ((LHSOp != TargetOpcode::G_SHL) && (LHSOp != TargetOpcode::G_ZEXT) &&
        (LHSOp != TargetOpcode::G_OR))
      return 0;
    if ((RHSOp != TargetOpcode::G_SHL) && (RHSOp != TargetOpcode::G_ZEXT) &&
        (RHSOp != TargetOpcode::G_OR))
      return 0;
    uint64_t LHSBits = getCoveredBits(LHSReg, SrcVecReg, SrcVecIdx, MRI);
    uint64_t RHSBits = getCoveredBits(RHSReg, SrcVecReg, SrcVecIdx, MRI);
    if (LHSBits && RHSBits)
      return LHSBits | RHSBits;
  } break;
  default:
    break;
  }
  return 0;
}

bool PISAPreLegalizerCombinerImpl::matchExtractInsertToBitcast(
    MachineInstr &MI, Register &SaveReg) const {
  llvm::MachineInstr &BuildVectorMI = MI;
  llvm::Register DstReg = BuildVectorMI.getOperand(0).getReg();
  llvm::LLT BuildVectorTy = MRI.getType(DstReg);
  uint16_t NumElts = BuildVectorTy.getNumElements();

  Register SrcVecReg;     // G_EXTRACT_VECTOR_ELT
  uint64_t SrcVecIdx = 0; // index of G_EXTRACT_VECTOR_ELT
  for (unsigned I = 0; I < NumElts; I++) {
    llvm::Register BuildVectorEltReg = BuildVectorMI.getOperand(I + 1).getReg();
    uint64_t Bits = getCoveredBits(BuildVectorEltReg, SrcVecReg, &SrcVecIdx, MRI);
    if (!Bits)
      return false; // did not match
    if (BuildVectorTy.getSizeInBits() != MRI.getType(SrcVecReg).getSizeInBits())
      return false;
    if (!((1ull << I) & SrcVecIdx))
      return false; // indices are not in ascending order
    unsigned EltSize = BuildVectorTy.getScalarSizeInBits();
    uint64_t Mask = (EltSize == 64) ? (uint64_t)-1ll : (1ull << EltSize) - 1;
    if (Mask != Bits)
      return false; // did not cover full element
  }
  llvm::LLT SrcVecTy = MRI.getType(SrcVecReg);
  uint64_t Mask = (1ull << SrcVecTy.getNumElements()) - 1;
  if (Mask != SrcVecIdx)
    return false; // did not extract all indices

  SaveReg = SrcVecReg;
  return true;
}
void PISAPreLegalizerCombinerImpl::applyExtractInsertToBitcast(
    MachineInstr &MI, Register SrcVecReg) const {
  llvm::Register DstReg = MI.getOperand(0).getReg();
  if (MRI.getType(DstReg) == MRI.getType(SrcVecReg))
    B.buildCopy(DstReg, SrcVecReg);
  else
    B.buildBitcast(DstReg, SrcVecReg);
  MI.eraseFromParent();
}

// %23:_(<4 x s8>) = G_BITCAST %1:_(s32)
//  %2:_(s8) = G_EXTRACT_VECTOR_ELT %23:_(<4 x s8>), %25:_(s32)
//  %7:_(s32) = G_LSHR %1:_, %6:_(s32)
//  %8:_(s8) = G_TRUNC %7:_(s32)
//  %12:_(s32) = G_LSHR %1:_, %11:_(s32)
//  %13:_(s8) = G_TRUNC %12:_(s32)
//  %24:_(s32) = G_CONSTANT i32 3
//  %18:_(s8) = G_EXTRACT_VECTOR_ELT %23:_(<4 x s8>), %24:_(s32)
//  %19:_(<4 x s8>) = G_BUILD_VECTOR %2:_(s8), %8:_(s8), %13:_(s8), %18:_(s8)
// => %19:_(<4 x s8>) = G_BITCAST %1:_(s32)
bool PISAPreLegalizerCombinerImpl::matchExtractBuildVectorToBitcast(
    MachineInstr &MI, Register &SaveReg) const {
  llvm::MachineInstr &BuildMI = MI;
  llvm::LLT DstTy = MRI.getType(BuildMI.getOperand(0).getReg());

  Register CastSrcReg; // G_BITCAST %1
  for (unsigned I = 1; I < BuildMI.getNumOperands(); I++) {
    llvm::Register EltReg = BuildMI.getOperand(I).getReg();
    llvm::MachineInstr &ExtMI = *getDefIgnoringCopies(EltReg, MRI);
    if (ExtMI.getOpcode() == TargetOpcode::G_EXTRACT_VECTOR_ELT) {
      llvm::Register NumReg = ExtMI.getOperand(2).getReg();
      std::optional<llvm::ValueAndVReg> NumValue = getIConstantVRegValWithLookThrough(NumReg, MRI);
      if (!NumValue.has_value() || (NumValue->Value != (I - 1)))
        return false;
      llvm::MachineInstr &CastMI = *getDefIgnoringCopies(ExtMI.getOperand(1).getReg(), MRI);
      if (CastMI.getOpcode() != TargetOpcode::G_BITCAST)
        return false;
      llvm::Register CastReg = CastMI.getOperand(1).getReg();
      if (CastSrcReg && (CastSrcReg != CastReg))
        return false;
      if (DstTy.getSizeInBits() != MRI.getType(CastReg).getSizeInBits())
        return false;
      CastSrcReg = CastReg;
    } else if (ExtMI.getOpcode() == TargetOpcode::G_TRUNC) {
      llvm::MachineInstr &ShiftMI = *getDefIgnoringCopies(ExtMI.getOperand(1).getReg(), MRI);
      if (ShiftMI.getOpcode() != TargetOpcode::G_LSHR)
        return false;
      std::optional<llvm::ValueAndVReg> ShiftValue = getIConstantVRegValWithLookThrough(
          ShiftMI.getOperand(2).getReg(), MRI);
      if (!ShiftValue.has_value() ||
          (ShiftValue->Value != ((I - 1) * DstTy.getScalarSizeInBits())))
        return false;
      llvm::Register ShiftReg = ShiftMI.getOperand(1).getReg();
      if (CastSrcReg && (CastSrcReg != ShiftReg))
        return false;
      CastSrcReg = ShiftReg;
    } else
      return false;
  }
  SaveReg = CastSrcReg;
  return true;
}
void PISAPreLegalizerCombinerImpl::applyExtractBuildVectorToBitcast(
    MachineInstr &MI, Register CastSrcReg) const {
  llvm::Register DstReg = MI.getOperand(0).getReg();
  B.buildBitcast(DstReg, CastSrcReg);
  MI.eraseFromParent();
}

// Reduce predicates in s32.
// %12:_(s1) = G_CONSTANT i1 true
// %3:_(<3 x s1>) = G_FCMP floatpred(oeq), %0:regv3_32b(<3 x s32>), %1:_
// %4:_(s1) = G_EXTRACT_VECTOR_ELT %3:_(<3 x s1>), %5:_(s32)
// %6:_(s1) = G_EXTRACT_VECTOR_ELT %3:_(<3 x s1>), %7:_(s32)
// %8:_(s1) = G_EXTRACT_VECTOR_ELT %3:_(<3 x s1>), %9:_(s32)
// %10:_(s1) = G_AND %4:_, %6:_
// %11:_(s1) = G_AND %10:_, %8:_
// %13:_(s1) = G_ICMP intpred(eq), %11:_(s1), %12:_
// => %3:_(<3 x s1>) = G_FCMP floatpred(oeq), %0:regv3_32b(<3 x s32>), %1:_
//    %18:_(<3 x s32>) = G_SEXT %3:_(<3 x s1>)
//    %20:_(s32) = G_CONSTANT i32 0
//    %19:_(s32) = G_EXTRACT_VECTOR_ELT %18:_(<3 x s32>), %20:_(s32)
//    %23:_(s32) = G_CONSTANT i32 1
//    %21:_(s32) = G_EXTRACT_VECTOR_ELT %18:_(<3 x s32>), %23:_(s32)
//    %22:_(s32) = G_AND %19:_, %21:_
//    %26:_(s32) = G_CONSTANT i32 2
//    %24:_(s32) = G_EXTRACT_VECTOR_ELT %18:_(<3 x s32>), %26:_(s32)
//    %25:_(s32) = G_AND %22:_, %24:_
//    %27:_(s32) = G_CONSTANT i32 -1
//    %13:_(s1) = G_ICMP intpred(eq), %25:_(s32), %27:_
// Supports patterns with sext too:
// %13:_(s8) = G_CONSTANT i8 -1
// %3:_(<3 x s1>) = G_FCMP floatpred(oeq), %0:regv3_32b(<3 x s32>), %1:_
// %4:_(<3 x s8>) = G_SEXT %3:_(<3 x s1>)
// %5:_(s8) = G_EXTRACT_VECTOR_ELT %4:_(<3 x s8>), %6:_(s32)
// %7:_(s8) = G_EXTRACT_VECTOR_ELT %4:_(<3 x s8>), %8:_(s32)
// %9:_(s8) = G_EXTRACT_VECTOR_ELT %4:_(<3 x s8>), %10:_(s32)
// %11:_(s8) = G_AND %5:_, %7:_
// %12:_(s8) = G_AND %11:_, %9:_
// %14:_(s1) = G_ICMP intpred(eq), %12:_(s8), %13:_
bool PISAPreLegalizerCombinerImpl::matchReducePredicates(
    MachineInstr &MI,
    std::function<void(MachineIRBuilder &)> &MatchInfo) const {

  llvm::Register DstReg = MI.getOperand(0).getReg();
  if (!MRI.hasOneUse(DstReg))
    return false;

  llvm::LLT DstTy = MRI.getType(DstReg);
  if (!DstTy.isScalar() || (DstTy.getSizeInBits() == 32))
    return false;

  // If type is greater than s1, it must be interpreted as true/false.
  if ((DstTy.getSizeInBits() > 1) &&
      !(mi_match(*MRI.use_instr_begin(DstReg), MRI,
                 m_GICmp(m_Pred(), m_Reg(),
                         m_any_of(m_SpecificICst(0), m_SpecificICst(-1))))))
    return false;

  MachineInstr *VecCmpMI = nullptr;
  APInt Mask;
  unsigned ReductionOpcode = MI.getOpcode();

  std::function<bool(MachineInstr *)> Match = [&](MachineInstr *MI) {
    if (!MI)
      return false;
    // Next step in reduction?
    Register LHS, RHS, Dst = MI->getOperand(0).getReg();
    if (mi_match(Dst, MRI,
                 m_OneUse(m_BinOp(ReductionOpcode, m_Reg(LHS), m_Reg(RHS))))) {
      return Match(getDefIgnoringCopies(LHS, MRI)) &&
             Match(getDefIgnoringCopies(RHS, MRI));
    }
    // Extract from vector?
    int64_t Idx;
    if (mi_match(Dst, MRI,
                 m_OneUse(m_BinOp(TargetOpcode::G_EXTRACT_VECTOR_ELT,
                                  m_Reg(LHS), m_ICst(Idx))))) {
      if (!Match(getDefIgnoringCopies(LHS, MRI)))
        return false;
      Mask.setBit(Idx);
      return true;
    }
    // Sext?
    if (MI->getOpcode() == TargetOpcode::G_SEXT) {
      return Match(getDefIgnoringCopies(MI->getOperand(1).getReg(), MRI));
    }
    // Cmp?
    if (mi_match(Dst, MRI,
                 m_any_of(m_GICmp(m_Pred(), m_Reg(), m_Reg()),
                          m_GFCmp(m_Pred(), m_Reg(), m_Reg())))) {
      if (VecCmpMI)
        return VecCmpMI == MI;
      llvm::LLT VecType = MRI.getType(MI->getOperand(0).getReg());
      if (!VecType.isFixedVector())
        return false;
      // Found cmp producing vector of predicates.
      VecCmpMI = MI;
      Mask = APInt::getZero(VecType.getNumElements());
      return true;
    }
    return false;
  };

  if (!Match(&MI) || !Mask.isAllOnes())
    return false;

  MatchInfo = [DstReg, Mask = std::move(Mask), VecCmpMI, ReductionOpcode,
               this](MachineIRBuilder &B) {
    const LLT S32 = LLT::integer(32);

    // Build reduction in s32.
    llvm::Register ExtendedVector = MRI.createGenericVirtualRegister(
        LLT::fixed_vector(Mask.getBitWidth(), S32));
    B.buildSExt(ExtendedVector, VecCmpMI->getOperand(0).getReg());

    Register Reduction = MRI.createGenericVirtualRegister(S32);
    B.buildExtractVectorElement(Reduction, ExtendedVector,
                                B.buildConstant(S32, 0));

    for (unsigned I = 1; I < Mask.getBitWidth(); ++I) {
      llvm::Register SecondSrc = MRI.createGenericVirtualRegister(S32);
      llvm::Register Dst = MRI.createGenericVirtualRegister(S32);
      B.buildExtractVectorElement(SecondSrc, ExtendedVector,
                                  B.buildConstant(S32, I));
      B.buildInstr(ReductionOpcode, {Dst}, {Reduction, SecondSrc});
      Reduction = Dst;
    }

    // Replace next use if possible.
    llvm::MachineRegisterInfo::use_instr_iterator UseMI = MRI.use_instr_begin(DstReg);
    switch (UseMI->getOpcode()) {
    case TargetOpcode::G_ICMP: {
      llvm::CmpInst::Predicate Pred = (CmpInst::Predicate)UseMI->getOperand(1).getPredicate();
      if (mi_match(UseMI->getOperand(3).getReg(), MRI, m_SpecificICst(-1)))
        Pred = CmpInst::getInversePredicate(Pred);
      B.buildICmp(Pred, UseMI->getOperand(0).getReg(), Reduction,
                  B.buildConstant(S32, 0));
      UseMI->eraseFromParent();
    } break;
    default: {
      B.buildICmp(CmpInst::ICMP_NE, DstReg, Reduction, B.buildConstant(S32, 0));
    } break;
    }
  };
  return true;
}

// %9:_(s1) = G_CONSTANT i1 true
// %8:_(s1) = G_ICMP intpred(eq), %15:_(s32), %22:_
// %10:_(s1) = G_ICMP intpred(eq), %8:_(s1), %9:_
// %11:_(s32) = G_SEXT %10:_(s1)
// => %8:_(s1) = G_ICMP intpred(eq), %15:_(s32), %22:_
//    %11:_(s32) = G_SEXT %8:_(s1)
// or
// %12:_(s1) = G_CONSTANT i1 true
// %11:_(s1) = G_AND %10:_, %35:_
// %13:_(s1) = G_ICMP intpred(eq), %11:_(s1), %12:_
// %14:_(s32) = G_SEXT %13:_(s1)
// => %11:_(s1) = G_AND %10:_, %35:_
//    %14:_(s32) = G_SEXT %11:_(s1)
bool PISAPreLegalizerCombinerImpl::matchCmpInt1(
    MachineInstr &ICmpMI,
    std::function<void(MachineIRBuilder &)> &MatchInfo) const {

  MachineInstr *PrevMI = nullptr;
  CmpInst::Predicate Pred;
  int64_t IsTrue;

  if (!mi_match(
          ICmpMI, MRI,
          m_GICmp(m_Pred(Pred),
                  m_all_of(m_SpecificType(LLT::integer(1)), m_MInstr(PrevMI)),
                  m_ICst(IsTrue))))
    return false;

  if (Pred != CmpInst::ICMP_EQ)
    return false;

  if (!PrevMI)
    return false;

  llvm::Register DstReg = ICmpMI.getOperand(0).getReg();
  MatchInfo = [DstReg, PrevMI, IsTrue, this](MachineIRBuilder &B) {
    llvm::Register PrevDstReg = PrevMI->getOperand(0).getReg();

    if (IsTrue) {
      if (MRI.hasOneUse(PrevDstReg))
        PrevMI->getOperand(0).setReg(DstReg);
      else
        B.buildCopy(DstReg, PrevDstReg);
    } else {
      B.buildNot(DstReg, PrevDstReg);
    }
  };
  return true;
}

// Pass boilerplate
// ================

class PISAPreLegalizerCombiner : public MachineFunctionPass {
  PISAPreLegalizerCombinerImplRuleConfig RuleConfig;

public:
  static char ID;

  PISAPreLegalizerCombiner();

  StringRef getPassName() const override { return "PISAPreLegalizerCombiner"; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};
} // end anonymous namespace

void PISAPreLegalizerCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  AU.addRequired<GISelValueTrackingAnalysisLegacy>();
  AU.addPreserved<GISelValueTrackingAnalysisLegacy>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();

  AU.addRequired<GISelCSEAnalysisWrapperPass>();
  AU.addPreserved<GISelCSEAnalysisWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

PISAPreLegalizerCombiner::PISAPreLegalizerCombiner() : MachineFunctionPass(ID) {
  initializePISAPreLegalizerCombinerPass(*PassRegistry::getPassRegistry());
  if (!RuleConfig.parseCommandLineOption())
    report_fatal_error("Invalid rule identifier");
}

bool PISAPreLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;

  llvm::TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOptLevel::None && !skipFunction(F);
  GISelValueTracking *KB =
      &getAnalysis<GISelValueTrackingAnalysisLegacy>().get(MF);
  MachineDominatorTree *MDT =
      &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  CombinerInfo CInfo(
      /*AllowIllegalOps=*/true, /*ShouldLegalizeIllegal=*/false,
      /*LegalizerInfo=*/nullptr, EnableOpt, F.hasOptSize(), F.hasMinSize());
  // Enable CSE.
  GISelCSEAnalysisWrapper &Wrapper =
      getAnalysis<GISelCSEAnalysisWrapperPass>().getCSEWrapper();
  llvm::GISelCSEInfo *CSEInfo = &Wrapper.get(TPC.getCSEConfig());

  const PISASubtarget &STI = MF.getSubtarget<PISASubtarget>();
  PISAPreLegalizerCombinerImpl Impl(MF, CInfo, *KB, CSEInfo, RuleConfig, STI,
                                    MDT, STI.getLegalizerInfo());
  return Impl.combineMachineInstrs();
}

// select(trunc(wide -> i1), N 1, N 0) => trunc(and(wide, 1), N)
// Avoids introducing an illegal i1 compare when lowering bool-to-int.
bool PISAPreLegalizerCombinerImpl::matchSelectTruncOneZero(
    MachineInstr &MI,
    std::function<void(MachineIRBuilder &)> &MatchInfo) const {
  assert(MI.getOpcode() == TargetOpcode::G_SELECT);
  GSelect &Sel = cast<GSelect>(MI);

  // Condition must be i1.
  Register Cond = Sel.getCondReg();
  if (MRI.getType(Cond) != LLT::scalar(1))
    return false;

  // Condition must come from a G_TRUNC.
  MachineInstr *TruncMI = getDefIgnoringCopies(Cond, MRI);
  if (!TruncMI || TruncMI->getOpcode() != TargetOpcode::G_TRUNC)
    return false;

  Register Wide = TruncMI->getOperand(1).getReg();
  LLT WideTy = MRI.getType(Wide);
  LLT DstTy = MRI.getType(Sel.getReg(0));

  // Only handle scalar integers where dst fits in the wide source.
  if (!DstTy.isScalar() || !WideTy.isScalar())
    return false;
  if (DstTy.getScalarSizeInBits() >= WideTy.getScalarSizeInBits())
    return false;

  // True value must be 1, false value must be 0.
  std::optional<llvm::ValueAndVReg> TrueOpt =
      getIConstantVRegValWithLookThrough(Sel.getTrueReg(), MRI);
  std::optional<llvm::ValueAndVReg> FalseOpt =
      getIConstantVRegValWithLookThrough(Sel.getFalseReg(), MRI);
  if (!TrueOpt || !FalseOpt)
    return false;
  if (!TrueOpt->Value.isOne() || !FalseOpt->Value.isZero())
    return false;

  Register DstReg = Sel.getReg(0);
  MatchInfo = [=](MachineIRBuilder &B) {
    llvm::MachineInstrBuilder One = B.buildConstant(WideTy, 1);
    llvm::MachineInstrBuilder And = B.buildAnd(WideTy, Wide, One);
    B.buildTrunc(DstReg, And);
  };
  return true;
}

char PISAPreLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(PISAPreLegalizerCombiner, DEBUG_TYPE,
                      "Combine PISA machine instrs before legalization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelValueTrackingAnalysisLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(GISelCSEAnalysisWrapperPass)
INITIALIZE_PASS_END(PISAPreLegalizerCombiner, DEBUG_TYPE,
                    "Combine PISA machine instrs before legalization", false,
                    false)

namespace llvm {
FunctionPass *createPISAPreLegalizerCombiner() {
  return new PISAPreLegalizerCombiner();
}
} // end namespace llvm
