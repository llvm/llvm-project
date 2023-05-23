//===-- X86FixupVectorConstants.cpp - optimize constant generation  -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file examines all full size vector constant pool loads and attempts to
// replace them with smaller constant pool entries, including:
// * Converting AVX512 memory-fold instructions to their broadcast-fold form
// * TODO: Broadcasting of full width loads.
// * TODO: Sign/Zero extension of full width loads.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrFoldTables.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineConstantPool.h"

using namespace llvm;

#define DEBUG_TYPE "x86-fixup-vector-constants"

STATISTIC(NumInstChanges, "Number of instructions changes");

namespace {
class X86FixupVectorConstantsPass : public MachineFunctionPass {
public:
  static char ID;

  X86FixupVectorConstantsPass() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "X86 Fixup Vector Constants";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  bool processInstruction(MachineFunction &MF, MachineBasicBlock &MBB,
                          MachineInstr &MI);

  // This pass runs after regalloc and doesn't support VReg operands.
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

private:
  const X86InstrInfo *TII = nullptr;
  const X86Subtarget *ST = nullptr;
  const MCSchedModel *SM = nullptr;
};
} // end anonymous namespace

char X86FixupVectorConstantsPass::ID = 0;

INITIALIZE_PASS(X86FixupVectorConstantsPass, DEBUG_TYPE, DEBUG_TYPE, false, false)

FunctionPass *llvm::createX86FixupVectorConstants() {
  return new X86FixupVectorConstantsPass();
}

static const Constant *getConstantFromPool(const MachineInstr &MI,
                                           const MachineOperand &Op) {
  if (!Op.isCPI() || Op.getOffset() != 0)
    return nullptr;

  ArrayRef<MachineConstantPoolEntry> Constants =
      MI.getParent()->getParent()->getConstantPool()->getConstants();
  const MachineConstantPoolEntry &ConstantEntry = Constants[Op.getIndex()];

  // Bail if this is a machine constant pool entry, we won't be able to dig out
  // anything useful.
  if (ConstantEntry.isMachineConstantPoolEntry())
    return nullptr;

  return ConstantEntry.Val.ConstVal;
}

// Attempt to extract the full width of bits data from the constant.
static std::optional<APInt> extractConstantBits(const Constant *C) {
  unsigned NumBits = C->getType()->getPrimitiveSizeInBits();

  if (auto *CInt = dyn_cast<ConstantInt>(C))
    return CInt->getValue();

  if (auto *CFP = dyn_cast<ConstantFP>(C))
    return CFP->getValue().bitcastToAPInt();

  if (auto *CV = dyn_cast<ConstantVector>(C)) {
    if (auto *CVSplat = CV->getSplatValue(/*AllowUndefs*/ true)) {
      if (std::optional<APInt> Bits = extractConstantBits(CVSplat)) {
        assert((NumBits % Bits->getBitWidth()) == 0 && "Illegal splat");
        return APInt::getSplat(NumBits, *Bits);
      }
    }
  }

  if (auto *CDS = dyn_cast<ConstantDataSequential>(C)) {
    bool IsInteger = CDS->getElementType()->isIntegerTy();
    bool IsFloat = CDS->getElementType()->isHalfTy() ||
                   CDS->getElementType()->isBFloatTy() ||
                   CDS->getElementType()->isFloatTy() ||
                   CDS->getElementType()->isDoubleTy();
    if (IsInteger || IsFloat) {
      APInt Bits = APInt::getZero(NumBits);
      unsigned EltBits = CDS->getElementType()->getPrimitiveSizeInBits();
      for (unsigned I = 0, E = CDS->getNumElements(); I != E; ++I) {
        if (IsInteger)
          Bits.insertBits(CDS->getElementAsAPInt(I), I * EltBits);
        else
          Bits.insertBits(CDS->getElementAsAPFloat(I).bitcastToAPInt(),
                          I * EltBits);
      }
      return Bits;
    }
  }

  return std::nullopt;
}

// Attempt to compute the splat width of bits data by normalizing the splat to
// remove undefs.
static std::optional<APInt> getSplatableConstant(const Constant *C,
                                                 unsigned SplatBitWidth) {
  const Type *Ty = C->getType();
  assert((Ty->getPrimitiveSizeInBits() % SplatBitWidth) == 0 &&
         "Illegal splat width");

  if (std::optional<APInt> Bits = extractConstantBits(C))
    if (Bits->isSplat(SplatBitWidth))
      return Bits->trunc(SplatBitWidth);

  // Detect general splats with undefs.
  // TODO: Do we need to handle NumEltsBits > SplatBitWidth splitting?
  if (auto *CV = dyn_cast<ConstantVector>(C)) {
    unsigned NumOps = CV->getNumOperands();
    unsigned NumEltsBits = Ty->getScalarSizeInBits();
    unsigned NumScaleOps = SplatBitWidth / NumEltsBits;
    if ((SplatBitWidth % NumEltsBits) == 0) {
      // Collect the elements and ensure that within the repeated splat sequence
      // they either match or are undef.
      SmallVector<Constant *, 16> Sequence(NumScaleOps, nullptr);
      for (unsigned Idx = 0; Idx != NumOps; ++Idx) {
        if (Constant *Elt = CV->getAggregateElement(Idx)) {
          if (isa<UndefValue>(Elt))
            continue;
          unsigned SplatIdx = Idx % NumScaleOps;
          if (!Sequence[SplatIdx] || Sequence[SplatIdx] == Elt) {
            Sequence[SplatIdx] = Elt;
            continue;
          }
        }
        return std::nullopt;
      }
      // Extract the constant bits forming the splat and insert into the bits
      // data, leave undef as zero.
      APInt SplatBits = APInt::getZero(SplatBitWidth);
      for (unsigned I = 0; I != NumScaleOps; ++I) {
        if (!Sequence[I])
          continue;
        if (std::optional<APInt> Bits = extractConstantBits(Sequence[I])) {
          SplatBits.insertBits(*Bits, I * Bits->getBitWidth());
          continue;
        }
        return std::nullopt;
      }
      return SplatBits;
    }
  }

  return std::nullopt;
}

// Attempt to rebuild a normalized splat vector constant of the requested splat
// width, built up of potentially smaller scalar values.
// NOTE: We don't always bother converting to scalars if the vector length is 1.
static Constant *rebuildSplatableConstant(const Constant *C,
                                          unsigned SplatBitWidth) {
  std::optional<APInt> Splat = getSplatableConstant(C, SplatBitWidth);
  if (!Splat)
    return nullptr;

  // Determine scalar size to use for the constant splat vector, clamping as we
  // might have found a splat smaller than the original constant data.
  const Type *OriginalType = C->getType();
  Type *SclTy = OriginalType->getScalarType();
  unsigned NumSclBits = SclTy->getPrimitiveSizeInBits();
  NumSclBits = std::min<unsigned>(NumSclBits, SplatBitWidth);

  if (NumSclBits == 8) {
    SmallVector<uint8_t> RawBits;
    for (unsigned I = 0; I != SplatBitWidth; I += 8)
      RawBits.push_back(Splat->extractBits(8, I).getZExtValue());
    return ConstantDataVector::get(OriginalType->getContext(), RawBits);
  }

  if (NumSclBits == 16) {
    SmallVector<uint16_t> RawBits;
    for (unsigned I = 0; I != SplatBitWidth; I += 16)
      RawBits.push_back(Splat->extractBits(16, I).getZExtValue());
    if (SclTy->is16bitFPTy())
      return ConstantDataVector::getFP(SclTy, RawBits);
    return ConstantDataVector::get(OriginalType->getContext(), RawBits);
  }

  if (NumSclBits == 32) {
    SmallVector<uint32_t> RawBits;
    for (unsigned I = 0; I != SplatBitWidth; I += 32)
      RawBits.push_back(Splat->extractBits(32, I).getZExtValue());
    if (SclTy->isFloatTy())
      return ConstantDataVector::getFP(SclTy, RawBits);
    return ConstantDataVector::get(OriginalType->getContext(), RawBits);
  }

  // Fallback to i64 / double.
  SmallVector<uint64_t> RawBits;
  for (unsigned I = 0; I != SplatBitWidth; I += 64)
    RawBits.push_back(Splat->extractBits(64, I).getZExtValue());
  if (SclTy->isDoubleTy())
    return ConstantDataVector::getFP(SclTy, RawBits);
  return ConstantDataVector::get(OriginalType->getContext(), RawBits);
}

bool X86FixupVectorConstantsPass::processInstruction(MachineFunction &MF,
                                                     MachineBasicBlock &MBB,
                                                     MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  MachineConstantPool *CP  = MI.getParent()->getParent()->getConstantPool();

  auto ConvertToBroadcast = [&](unsigned OpBcst256, unsigned OpBcst128,
                                unsigned OpBcst64, unsigned OpBcst32,
                                unsigned OpBcst16, unsigned OpBcst8,
                                unsigned OperandNo) {
    assert(MI.getNumOperands() >= (OperandNo + X86::AddrNumOperands) &&
           "Unexpected number of operands!");

    MachineOperand &CstOp = MI.getOperand(OperandNo + X86::AddrDisp);
    if (auto *C = getConstantFromPool(MI, CstOp)) {
      // Attempt to detect a suitable splat from increasing splat widths.
      std::pair<unsigned, unsigned> Broadcasts[] = {
          {8, OpBcst8},   {16, OpBcst16},   {32, OpBcst32},
          {64, OpBcst64}, {128, OpBcst128}, {256, OpBcst256},
      };
      for (auto [BitWidth, OpBcst] : Broadcasts) {
        if (OpBcst) {
          // Construct a suitable splat constant and adjust the MI to
          // use the new constant pool entry.
          if (Constant *NewCst = rebuildSplatableConstant(C, BitWidth)) {
            unsigned NewCPI =
                CP->getConstantPoolIndex(NewCst, Align(BitWidth / 8));
            MI.setDesc(TII->get(OpBcst));
            CstOp.setIndex(NewCPI);
            return true;
          }
        }
      }
    }
    return false;
  };

  // Attempt to find a AVX512 mapping from a full width memory-fold instruction
  // to a broadcast-fold instruction variant.
  if ((MI.getDesc().TSFlags & X86II::EncodingMask) == X86II::EVEX) {
    unsigned OpBcst32 = 0, OpBcst64 = 0;
    unsigned OpNoBcst32 = 0, OpNoBcst64 = 0;
    if (const X86MemoryFoldTableEntry *Mem2Bcst =
            llvm::lookupBroadcastFoldTable(Opc, 32)) {
      OpBcst32 = Mem2Bcst->DstOp;
      OpNoBcst32 = Mem2Bcst->Flags & TB_INDEX_MASK;
    }
    if (const X86MemoryFoldTableEntry *Mem2Bcst =
            llvm::lookupBroadcastFoldTable(Opc, 64)) {
      OpBcst64 = Mem2Bcst->DstOp;
      OpNoBcst64 = Mem2Bcst->Flags & TB_INDEX_MASK;
    }
    assert(((OpBcst32 == 0) || (OpBcst64 == 0) || (OpNoBcst32 == OpNoBcst64)) &&
           "OperandNo mismatch");

    if (OpBcst32 || OpBcst64) {
      unsigned OpNo = OpBcst32 == 0 ? OpNoBcst64 : OpNoBcst32;
      return ConvertToBroadcast(0, 0, OpBcst64, OpBcst32, 0, 0, OpNo);
    }
  }

  return false;
}

bool X86FixupVectorConstantsPass::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "Start X86FixupVectorConstants\n";);
  bool Changed = false;
  ST = &MF.getSubtarget<X86Subtarget>();
  TII = ST->getInstrInfo();
  SM = &ST->getSchedModel();

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (processInstruction(MF, MBB, MI)) {
        ++NumInstChanges;
        Changed = true;
      }
    }
  }
  LLVM_DEBUG(dbgs() << "End X86FixupVectorConstants\n";);
  return Changed;
}
