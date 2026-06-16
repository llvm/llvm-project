//===- lib/CodeGen/MachineStableHash.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stable hashing for MachineInstr and MachineOperand. Useful or getting a
// hash across runs, modules, etc.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineStableHash.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StableHashing.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/StructuralHash.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "machine-stable-hash"

using namespace llvm;

STATISTIC(StableHashBailingMachineBasicBlock,
          "Number of encountered unsupported MachineOperands that were "
          "MachineBasicBlocks while computing stable hashes");
STATISTIC(StableHashBailingConstantPoolIndex,
          "Number of encountered unsupported MachineOperands that were "
          "ConstantPoolIndex while computing stable hashes");
STATISTIC(StableHashBailingTargetIndexNoName,
          "Number of encountered unsupported MachineOperands that were "
          "TargetIndex with no name");
STATISTIC(StableHashBailingGlobalAddress,
          "Number of encountered unsupported MachineOperands that were "
          "GlobalAddress while computing stable hashes");
STATISTIC(StableHashBailingBlockAddress,
          "Number of encountered unsupported MachineOperands that were "
          "BlockAddress while computing stable hashes");
STATISTIC(StableHashBailingMetadataUnsupported,
          "Number of encountered unsupported MachineOperands that were "
          "Metadata of an unsupported kind while computing stable hashes");

stable_hash llvm::stableHashValue(const MachineOperand &MO) {
  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    if (MO.getReg().isVirtual()) {
      const MachineRegisterInfo &MRI = MO.getParent()->getMF()->getRegInfo();
      SmallVector<stable_hash> DefOpcodes;
      for (auto &Def : MRI.def_instructions(MO.getReg()))
        DefOpcodes.push_back(Def.getOpcode());
      return stable_hash_combine(DefOpcodes);
    }

    // Register operands don't have target flags.
    return stable_hash_combine(MO.getType(), MO.getReg().id(), MO.getSubReg(),
                               MO.isDef());
  case MachineOperand::MO_Immediate:
    return stable_hash_combine(MO.getType(), MO.getTargetFlags(), MO.getImm());
  case MachineOperand::MO_CImmediate:
  case MachineOperand::MO_FPImmediate: {
    auto Val = MO.isCImm() ? MO.getCImm()->getValue()
                           : MO.getFPImm()->getValueAPF().bitcastToAPInt();
    auto ValHash = stable_hash_combine(
        ArrayRef<stable_hash>(Val.getRawData(), Val.getNumWords()));
    return stable_hash_combine(MO.getType(), MO.getTargetFlags(), ValHash);
  }

  case MachineOperand::MO_MachineBasicBlock:
    ++StableHashBailingMachineBasicBlock;
    return 0;
  case MachineOperand::MO_ConstantPoolIndex:
    ++StableHashBailingConstantPoolIndex;
    return 0;
  case MachineOperand::MO_BlockAddress:
    ++StableHashBailingBlockAddress;
    return 0;
  case MachineOperand::MO_Metadata:
    ++StableHashBailingMetadataUnsupported;
    return 0;
  case MachineOperand::MO_GlobalAddress: {
    const GlobalValue *GV = MO.getGlobal();
    stable_hash GVHash = 0;
    if (auto *GVar = dyn_cast<GlobalVariable>(GV))
      GVHash = StructuralHash(*GVar);
    if (!GVHash) {
      if (!GV->hasName()) {
        ++StableHashBailingGlobalAddress;
        return 0;
      }
      GVHash = stable_hash_name(GV->getName());
    }

    return stable_hash_combine(MO.getType(), MO.getTargetFlags(), GVHash,
                               MO.getOffset());
  }

  case MachineOperand::MO_TargetIndex: {
    if (const char *Name = MO.getTargetIndexName())
      return stable_hash_combine(MO.getType(), MO.getTargetFlags(),
                                 stable_hash_name(Name), MO.getOffset());
    ++StableHashBailingTargetIndexNoName;
    return 0;
  }

  case MachineOperand::MO_FrameIndex:
  case MachineOperand::MO_JumpTableIndex:
    return stable_hash_combine(MO.getType(), MO.getTargetFlags(),
                               MO.getIndex());

  case MachineOperand::MO_ExternalSymbol:
    return stable_hash_combine(MO.getType(), MO.getTargetFlags(),
                               MO.getOffset(),
                               stable_hash_name(MO.getSymbolName()));

  case MachineOperand::MO_RegisterMask:
  case MachineOperand::MO_RegisterLiveOut: {
    if (const MachineInstr *MI = MO.getParent()) {
      if (const MachineBasicBlock *MBB = MI->getParent()) {
        if (const MachineFunction *MF = MBB->getParent()) {
          const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
          unsigned RegMaskSize =
              MachineOperand::getRegMaskSize(TRI->getNumRegs());
          const uint32_t *RegMask =
              MO.isRegMask() ? MO.getRegMask() : MO.getRegLiveOut();
          std::vector<llvm::stable_hash> RegMaskHashes(RegMask,
                                                       RegMask + RegMaskSize);
          return stable_hash_combine(MO.getType(), MO.getTargetFlags(),
                                     stable_hash_combine(RegMaskHashes));
        }
      }
    }

    assert(0 && "MachineOperand not associated with any MachineFunction");
    return stable_hash_combine(MO.getType(), MO.getTargetFlags());
  }

  case MachineOperand::MO_ShuffleMask: {
    std::vector<llvm::stable_hash> ShuffleMaskHashes;

    llvm::transform(
        MO.getShuffleMask(), std::back_inserter(ShuffleMaskHashes),
        [](int S) -> llvm::stable_hash { return llvm::stable_hash(S); });

    return stable_hash_combine(MO.getType(), MO.getTargetFlags(),
                               stable_hash_combine(ShuffleMaskHashes));
  }
  case MachineOperand::MO_MCSymbol: {
    auto SymbolName = MO.getMCSymbol()->getName();
    return stable_hash_combine(MO.getType(), MO.getTargetFlags(),
                               stable_hash_name(SymbolName));
  }
  case MachineOperand::MO_LaneMask: {
    return stable_hash_combine(MO.getType(), MO.getTargetFlags(),
                               MO.getLaneMask().getAsInteger());
  }
  case MachineOperand::MO_CFIIndex:
    return stable_hash_combine(MO.getType(), MO.getTargetFlags(),
                               MO.getCFIIndex());
  case MachineOperand::MO_IntrinsicID:
    return stable_hash_combine(MO.getType(), MO.getTargetFlags(),
                               MO.getIntrinsicID());
  case MachineOperand::MO_Predicate:
    return stable_hash_combine(MO.getType(), MO.getTargetFlags(),
                               MO.getPredicate());
  case MachineOperand::MO_DbgInstrRef:
    return stable_hash_combine(MO.getType(), MO.getInstrRefInstrIndex(),
                               MO.getInstrRefOpIndex());
  }
  llvm_unreachable("Invalid machine operand type");
}

/// A stable hash value for machine instructions.
/// Returns 0 if no stable hash could be computed.
/// The hashing and equality testing functions ignore definitions so this is
/// useful for CSE, etc.
stable_hash llvm::stableHashValue(const MachineInstr &MI, bool HashVRegs,
                                  bool HashConstantPoolIndices,
                                  bool HashMemOperands) {
  // Build up a buffer of hash code components.
  SmallVector<stable_hash, 16> HashComponents;
  HashComponents.reserve(MI.getNumOperands() + MI.getNumMemOperands() + 2);
  HashComponents.push_back(MI.getOpcode());
  HashComponents.push_back(MI.getFlags());
  for (const MachineOperand &MO : MI.operands()) {
    if (!HashVRegs && MO.isReg() && MO.isDef() && MO.getReg().isVirtual())
      continue; // Skip virtual register defs.

    if (MO.isCPI()) {
      HashComponents.push_back(stable_hash_combine(
          MO.getType(), MO.getTargetFlags(), MO.getIndex()));
      continue;
    }

    stable_hash StableHash = stableHashValue(MO);
    if (!StableHash)
      return 0;
    HashComponents.push_back(StableHash);
  }

  for (const auto *Op : MI.memoperands()) {
    if (!HashMemOperands)
      break;
    HashComponents.push_back(static_cast<unsigned>(Op->getSize().getValue()));
    HashComponents.push_back(static_cast<unsigned>(Op->getFlags()));
    HashComponents.push_back(static_cast<unsigned>(Op->getOffset()));
    HashComponents.push_back(static_cast<unsigned>(Op->getSuccessOrdering()));
    HashComponents.push_back(static_cast<unsigned>(Op->getAddrSpace()));
    HashComponents.push_back(static_cast<unsigned>(Op->getSyncScopeID()));
    HashComponents.push_back(static_cast<unsigned>(Op->getBaseAlign().value()));
    HashComponents.push_back(static_cast<unsigned>(Op->getFailureOrdering()));
  }

  return stable_hash_combine(HashComponents);
}

stable_hash llvm::stableHashValue(const MachineBasicBlock &MBB) {
  SmallVector<stable_hash> HashComponents;
  // TODO: Hash more stuff like block alignment and branch probabilities.
  for (const auto &MI : MBB)
    HashComponents.push_back(stableHashValue(MI));
  return stable_hash_combine(HashComponents);
}

stable_hash llvm::stableHashValue(const MachineFunction &MF) {
  SmallVector<stable_hash> HashComponents;
  // TODO: Hash lots more stuff like function alignment and stack objects.
  for (const auto &MBB : MF)
    HashComponents.push_back(stableHashValue(MBB));
  return stable_hash_combine(HashComponents);
}

// The generic stableHashValue() API is conservative: it can return 0 when an
// operand cannot be hashed with the stability needed by compiler analyses. The
// print-changed hash is diagnostic-only, so it is more permissive and hashes
// those operands using the best local identity available. This keeps
// -print-changed=hash-func/hash-bb useful for change detection without changing
// the contract of the generic stable hash APIs.
// TODO: Prefer stable symbolic identities over local numbering where available
// to reduce false positives from renumbering.
static stable_hash
hashMachineOperandForChangePrinter(const MachineOperand &MO) {
  stable_hash H = stable_hash_combine(
      MO.getType(), static_cast<unsigned>(MO.getTargetFlags()));

  switch (MO.getType()) {
  case MachineOperand::MO_Register: {
    H = stable_hash_combine(
        H, stable_hash_combine(MO.getReg().id(), MO.isDef(), MO.isImplicit()),
        stable_hash_combine(MO.isDead(), MO.isKill()));
    if (MO.getSubReg())
      H = stable_hash_combine(H, MO.getSubReg());
    return H;
  }
  case MachineOperand::MO_Immediate:
    return stable_hash_combine(H, MO.getImm());
  case MachineOperand::MO_CImmediate: {
    APInt Val = MO.getCImm()->getValue();
    return stable_hash_combine(H, static_cast<stable_hash>(hash_value(Val)));
  }
  case MachineOperand::MO_FPImmediate: {
    APInt Val = MO.getFPImm()->getValueAPF().bitcastToAPInt();
    return stable_hash_combine(H, static_cast<stable_hash>(hash_value(Val)));
  }
  case MachineOperand::MO_MachineBasicBlock:
    return stable_hash_combine(H, MO.getMBB()->getNumber());
  case MachineOperand::MO_FrameIndex:
    return stable_hash_combine(H, MO.getIndex());
  case MachineOperand::MO_ConstantPoolIndex:
    return stable_hash_combine(H, MO.getIndex(), MO.getOffset());
  case MachineOperand::MO_TargetIndex:
    return stable_hash_combine(H, MO.getIndex(), MO.getOffset());
  case MachineOperand::MO_JumpTableIndex:
    return stable_hash_combine(H, MO.getIndex());
  case MachineOperand::MO_ExternalSymbol:
    return stable_hash_combine(
        H, MO.getOffset(),
        static_cast<stable_hash>(hash_value(StringRef(MO.getSymbolName()))));
  case MachineOperand::MO_GlobalAddress:
    return stable_hash_combine(
        H, static_cast<stable_hash>(hash_value(MO.getGlobal()->getName())),
        MO.getOffset());
  case MachineOperand::MO_BlockAddress:
    return stable_hash_combine(
        H,
        static_cast<stable_hash>(
            hash_value(MO.getBlockAddress()->getFunction()->getName())),
        MO.getBlockAddress()->getBasicBlock()->hasName()
            ? static_cast<stable_hash>(
                  hash_value(MO.getBlockAddress()->getBasicBlock()->getName()))
            : static_cast<stable_hash>(
                  hash_value(MO.getBlockAddress()->getBasicBlock())),
        MO.getOffset());
  case MachineOperand::MO_Metadata:
    return stable_hash_combine(H, hash_value(MO.getMetadata()));
  case MachineOperand::MO_MCSymbol:
    return stable_hash_combine(
        H, static_cast<stable_hash>(hash_value(MO.getMCSymbol()->getName())));
  case MachineOperand::MO_RegisterMask:
  case MachineOperand::MO_RegisterLiveOut: {
    const MachineInstr *MI = MO.getParent();
    const MachineBasicBlock *MBB = MI ? MI->getParent() : nullptr;
    const MachineFunction *MF = MBB ? MBB->getParent() : nullptr;
    const TargetRegisterInfo *TRI =
        MF ? MF->getSubtarget().getRegisterInfo() : nullptr;
    if (!TRI)
      return H;
    unsigned RegMaskSize = MachineOperand::getRegMaskSize(TRI->getNumRegs());
    const uint32_t *RegMask =
        MO.isRegMask() ? MO.getRegMask() : MO.getRegLiveOut();
    SmallVector<stable_hash, 32> RegMaskHashes(RegMask, RegMask + RegMaskSize);
    return stable_hash_combine(H, stable_hash_combine(RegMaskHashes));
  }
  case MachineOperand::MO_CFIIndex:
    return stable_hash_combine(H, MO.getCFIIndex());
  case MachineOperand::MO_IntrinsicID:
    return stable_hash_combine(H, MO.getIntrinsicID());
  case MachineOperand::MO_Predicate:
    return stable_hash_combine(H, MO.getPredicate());
  case MachineOperand::MO_ShuffleMask: {
    SmallVector<stable_hash, 8> ShuffleMaskHashes;
    llvm::transform(MO.getShuffleMask(), std::back_inserter(ShuffleMaskHashes),
                    [](int S) { return stable_hash(S); });
    return stable_hash_combine(H, stable_hash_combine(ShuffleMaskHashes));
  }
  case MachineOperand::MO_DbgInstrRef:
    return stable_hash_combine(H, MO.getInstrRefInstrIndex(),
                               MO.getInstrRefOpIndex());
  case MachineOperand::MO_LaneMask:
    return stable_hash_combine(H, MO.getLaneMask().getAsInteger());
  }
  llvm_unreachable("Invalid machine operand type");
}

static stable_hash hashLocationSizeForChangePrinter(LocationSize Size) {
  if (!Size.hasValue())
    return 0;

  TypeSize TySize = Size.getValue();
  return stable_hash_combine(/*HasValue=*/true, TySize.getKnownMinValue(),
                             TySize.isScalable());
}

static stable_hash
hashMachineMemOperandForChangePrinter(const MachineMemOperand &MMO) {
  AAMDNodes AAInfo = MMO.getAAInfo();
  SmallVector<stable_hash, 16> HashComponents;
  HashComponents.push_back(static_cast<unsigned>(MMO.getFlags()));
  HashComponents.push_back(hashLocationSizeForChangePrinter(MMO.getSize()));
  HashComponents.push_back(MMO.getOffset());
  HashComponents.push_back(MMO.getAddrSpace());
  HashComponents.push_back(MMO.getBaseAlign().value());
  HashComponents.push_back(static_cast<unsigned>(MMO.getSyncScopeID()));
  HashComponents.push_back(static_cast<unsigned>(MMO.getSuccessOrdering()));
  HashComponents.push_back(static_cast<unsigned>(MMO.getFailureOrdering()));
  HashComponents.push_back(MMO.getMemoryType().isValid()
                               ? MMO.getMemoryType().getUniqueRAWLLTData()
                               : uint64_t(0));
  HashComponents.push_back(
      static_cast<stable_hash>(hash_value(MMO.getOpaqueValue())));
  HashComponents.push_back(static_cast<stable_hash>(hash_value(AAInfo.TBAA)));
  HashComponents.push_back(
      static_cast<stable_hash>(hash_value(AAInfo.TBAAStruct)));
  HashComponents.push_back(static_cast<stable_hash>(hash_value(AAInfo.Scope)));
  HashComponents.push_back(
      static_cast<stable_hash>(hash_value(AAInfo.NoAlias)));
  HashComponents.push_back(
      static_cast<stable_hash>(hash_value(AAInfo.NoAliasAddrSpace)));
  HashComponents.push_back(
      static_cast<stable_hash>(hash_value(MMO.getRanges())));
  return stable_hash_combine(HashComponents);
}

stable_hash llvm::stableHashValueForChangePrinter(const MachineInstr &MI) {
  stable_hash H =
      stable_hash_combine(MI.getOpcode(), MI.getFlags(), MI.getNumOperands(),
                          MI.getNumMemOperands());
  for (const MachineOperand &MO : MI.operands())
    H = stable_hash_combine(H, hashMachineOperandForChangePrinter(MO));
  for (const MachineMemOperand *MMO : MI.memoperands())
    H = stable_hash_combine(H, hashMachineMemOperandForChangePrinter(*MMO));
  return H;
}

static stable_hash stableHashMachineBasicBlockForChangePrinter(
    const MachineBasicBlock &MBB,
    MachineBasicBlockStableHashInfo *Details = nullptr) {
  if (Details)
    Details->MBB = &MBB;

  stable_hash MBBHash = 0;
  for (const MachineInstr &MI : MBB) {
    stable_hash MIHash = stableHashValueForChangePrinter(MI);
    MBBHash = stable_hash_combine(MBBHash, MIHash);
  }

  if (Details)
    Details->Hash = MBBHash;
  return MBBHash;
}

static stable_hash hashMachineFunctionForChangePrinter(
    const MachineFunction &MF,
    MachineFunctionStableHashInfo *Details = nullptr) {
  if (Details)
    Details->Blocks.reserve(MF.size());

  stable_hash MFHash = 0;
  for (const MachineBasicBlock &MBB : MF) {
    MachineBasicBlockStableHashInfo *BlockDetails = nullptr;
    if (Details) {
      Details->Blocks.emplace_back();
      BlockDetails = &Details->Blocks.back();
    }
    MFHash = stable_hash_combine(
        MFHash, stableHashMachineBasicBlockForChangePrinter(MBB, BlockDetails));
  }

  if (Details)
    Details->Hash = MFHash;
  return MFHash;
}

stable_hash
llvm::stableHashValueForChangePrinter(const MachineBasicBlock &MBB) {
  return stableHashMachineBasicBlockForChangePrinter(MBB);
}

stable_hash llvm::stableHashValueForChangePrinter(const MachineFunction &MF) {
  return hashMachineFunctionForChangePrinter(MF);
}

MachineFunctionStableHashInfo
llvm::stableHashValueWithDetailsForChangePrinter(const MachineFunction &MF) {
  MachineFunctionStableHashInfo Details;
  hashMachineFunctionForChangePrinter(MF, &Details);
  return Details;
}
