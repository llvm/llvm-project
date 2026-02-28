//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// BPF-specific BTF debug info generation. Handles CO-RE relocations,
// .maps section processing, and BPF instruction lowering.
//
//===----------------------------------------------------------------------===//

#include "BPFBTFDebug.h"
#include "BPF.h"
#include "BPFCORE.h"
#include "MCTargetDesc/BPFMCTargetDesc.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

using namespace llvm;

BPFBTFDebug::BPFBTFDebug(AsmPrinter *AP)
    : BTFDebug(AP), MapDefNotCollected(true) {}

void BPFBTFDebug::visitMapDefType(const DIType *Ty, uint32_t &TypeId) {
  if (!Ty || DIToIdMap.find(Ty) != DIToIdMap.end()) {
    TypeId = DIToIdMap[Ty];
    return;
  }

  uint32_t TmpId;
  switch (Ty->getTag()) {
  case dwarf::DW_TAG_typedef:
  case dwarf::DW_TAG_const_type:
  case dwarf::DW_TAG_volatile_type:
  case dwarf::DW_TAG_restrict_type:
  case dwarf::DW_TAG_pointer_type:
    visitMapDefType(dyn_cast<DIDerivedType>(Ty)->getBaseType(), TmpId);
    break;
  case dwarf::DW_TAG_array_type:
    // Visit nested map array and jump to the element type
    visitMapDefType(dyn_cast<DICompositeType>(Ty)->getBaseType(), TmpId);
    break;
  case dwarf::DW_TAG_structure_type: {
    const DICompositeType *CTy = dyn_cast<DICompositeType>(Ty);
    const DINodeArray Elements = CTy->getElements();
    for (const auto *Element : Elements) {
      const auto *MemberType = cast<DIDerivedType>(Element);
      const DIType *MemberBaseType =
          tryRemoveAtomicType(MemberType->getBaseType());

      // Check if the visited composite type is a wrapper, and the member
      // represents the actual map definition.
      const auto *MemberCTy = dyn_cast<DICompositeType>(MemberBaseType);
      if (MemberCTy) {
        visitMapDefType(MemberBaseType, TmpId);
      } else {
        visitTypeEntry(MemberBaseType);
      }
    }
    break;
  }
  default:
    break;
  }

  // Visit this type, struct or a const/typedef/volatile/restrict type
  visitTypeEntry(Ty, TypeId, false, false);
}

void BPFBTFDebug::processMapDefGlobals() {
  const Module *M = MMI->getModule();
  for (const GlobalVariable &Global : M->globals()) {
    StringRef SecName;
    std::optional<SectionKind> GVKind;

    if (!Global.isDeclarationForLinker())
      GVKind = TargetLoweringObjectFile::getKindForGlobal(&Global, Asm->TM);

    if (Global.isDeclarationForLinker())
      SecName = Global.hasSection() ? Global.getSection() : "";
    else if (GVKind->isCommon())
      SecName = ".bss";
    else {
      TargetLoweringObjectFile *TLOF = Asm->TM.getObjFileLowering();
      MCSection *Sec = TLOF->SectionForGlobal(&Global, Asm->TM);
      SecName = Sec->getName();
    }

    // Only process .maps globals in this pass.
    if (!SecName.starts_with(".maps"))
      continue;

    SmallVector<DIGlobalVariableExpression *, 1> GVs;
    Global.getDebugInfo(GVs);
    if (GVs.size() == 0)
      continue;

    uint32_t GVTypeId = 0;
    DIGlobalVariable *DIGlobal = nullptr;
    for (auto *GVE : GVs) {
      DIGlobal = GVE->getVariable();
      visitMapDefType(DIGlobal->getType(), GVTypeId);
      break;
    }

    auto Linkage = Global.getLinkage();
    if (Linkage != GlobalValue::InternalLinkage &&
        Linkage != GlobalValue::ExternalLinkage &&
        Linkage != GlobalValue::WeakAnyLinkage &&
        Linkage != GlobalValue::WeakODRLinkage &&
        Linkage != GlobalValue::ExternalWeakLinkage)
      continue;

    uint32_t GVarInfo;
    if (Linkage == GlobalValue::InternalLinkage) {
      GVarInfo = BTF::VAR_STATIC;
    } else if (Global.hasInitializer()) {
      GVarInfo = BTF::VAR_GLOBAL_ALLOCATED;
    } else {
      GVarInfo = BTF::VAR_GLOBAL_EXTERNAL;
    }

    auto VarEntry =
        std::make_unique<BTFKindVar>(Global.getName(), GVTypeId, GVarInfo);
    uint32_t VarId = addType(std::move(VarEntry));

    processDeclAnnotations(DIGlobal->getAnnotations(), VarId, -1);

    if (SecName.empty())
      continue;

    auto [It, Inserted] = DataSecEntries.try_emplace(std::string(SecName));
    if (Inserted)
      It->second = std::make_unique<BTFKindDataSec>(Asm, std::string(SecName));

    const DataLayout &DL = Global.getDataLayout();
    uint32_t Size = Global.getGlobalSize(DL);

    It->second->addDataSecEntry(VarId, Asm->getSymbol(&Global), Size);

    if (Global.hasInitializer())
      processGlobalInitializer(Global.getInitializer());
  }
}

void BPFBTFDebug::processGlobals() {
  // First process .maps globals if not yet done.
  if (MapDefNotCollected) {
    processMapDefGlobals();
    MapDefNotCollected = false;
  }
  // Then process all non-.maps globals via the base class.
  BTFDebug::processGlobals();
}

void BPFBTFDebug::generatePatchImmReloc(const MCSymbol *ORSym, uint32_t RootId,
                                         const GlobalVariable *GVar,
                                         bool IsAma) {
  BTFFieldReloc FieldReloc;
  FieldReloc.Label = ORSym;
  FieldReloc.TypeID = RootId;

  StringRef AccessPattern = GVar->getName();
  size_t FirstDollar = AccessPattern.find_first_of('$');
  if (IsAma) {
    size_t FirstColon = AccessPattern.find_first_of(':');
    size_t SecondColon = AccessPattern.find_first_of(':', FirstColon + 1);
    StringRef IndexPattern = AccessPattern.substr(FirstDollar + 1);
    StringRef RelocKindStr = AccessPattern.substr(FirstColon + 1,
        SecondColon - FirstColon);
    StringRef PatchImmStr = AccessPattern.substr(SecondColon + 1,
        FirstDollar - SecondColon);

    FieldReloc.OffsetNameOff = addString(IndexPattern);
    FieldReloc.RelocKind = std::stoull(std::string(RelocKindStr));
    PatchImms[GVar] = std::make_pair(std::stoll(std::string(PatchImmStr)),
                                     FieldReloc.RelocKind);
  } else {
    StringRef RelocStr = AccessPattern.substr(FirstDollar + 1);
    FieldReloc.OffsetNameOff = addString("0");
    FieldReloc.RelocKind = std::stoull(std::string(RelocStr));
    PatchImms[GVar] = std::make_pair(RootId, FieldReloc.RelocKind);
  }
  FieldRelocTable[SecNameOff].push_back(FieldReloc);
}

void BPFBTFDebug::processGlobalValue(const MachineOperand &MO) {
  if (MO.isGlobal()) {
    const GlobalValue *GVal = MO.getGlobal();
    auto *GVar = dyn_cast<GlobalVariable>(GVal);
    if (!GVar) {
      processFuncPrototypes(dyn_cast<Function>(GVal));
      return;
    }

    if (!GVar->hasAttribute(BPFCoreSharedInfo::AmaAttr) &&
        !GVar->hasAttribute(BPFCoreSharedInfo::TypeIdAttr))
      return;

    MCSymbol *ORSym = OS.getContext().createTempSymbol();
    OS.emitLabel(ORSym);

    MDNode *MDN = GVar->getMetadata(LLVMContext::MD_preserve_access_index);
    uint32_t RootId = populateType(dyn_cast<DIType>(MDN));
    generatePatchImmReloc(ORSym, RootId, GVar,
                          GVar->hasAttribute(BPFCoreSharedInfo::AmaAttr));
  }
}

void BPFBTFDebug::beginFunctionImpl(const MachineFunction *MF) {
  if (MapDefNotCollected) {
    processMapDefGlobals();
    MapDefNotCollected = false;
  }

  BTFDebug::beginFunctionImpl(MF);
}

void BPFBTFDebug::processBeginInstruction(const MachineInstr *MI) {
  // Handle CO-RE and extern function relocations.
  if (MI->getOpcode() == BPF::LD_imm64) {
    processGlobalValue(MI->getOperand(1));
  } else if (MI->getOpcode() == BPF::CORE_LD64 ||
             MI->getOpcode() == BPF::CORE_LD32 ||
             MI->getOpcode() == BPF::CORE_ST ||
             MI->getOpcode() == BPF::CORE_SHIFT) {
    processGlobalValue(MI->getOperand(3));
  } else if (MI->getOpcode() == BPF::JAL) {
    const MachineOperand &MO = MI->getOperand(0);
    if (MO.isGlobal()) {
      processFuncPrototypes(dyn_cast<Function>(MO.getGlobal()));
    }
  }
}

bool BPFBTFDebug::InstLower(const MachineInstr *MI, MCInst &OutMI) {
  if (MI->getOpcode() == BPF::LD_imm64) {
    const MachineOperand &MO = MI->getOperand(1);
    if (MO.isGlobal()) {
      const GlobalVariable *GVar = dyn_cast<GlobalVariable>(MO.getGlobal());
      if (GVar) {
        // Emit "mov ri, <imm>" for patched insns.
        auto IMGIt = PatchImms.find(GVar);
        if (IMGIt != PatchImms.end()) {
          auto Imm = IMGIt->second;
          auto Reloc = Imm.second;
          if (Reloc == BTF::ENUM_VALUE_EXISTENCE || Reloc == BTF::ENUM_VALUE ||
              Reloc == BTF::BTF_TYPE_ID_LOCAL || Reloc == BTF::BTF_TYPE_ID_REMOTE)
            OutMI.setOpcode(BPF::LD_imm64);
          else
            OutMI.setOpcode(BPF::MOV_ri);
          OutMI.addOperand(MCOperand::createReg(MI->getOperand(0).getReg()));
          OutMI.addOperand(MCOperand::createImm(Imm.first));
          return true;
        }
      }
    }
  } else if (MI->getOpcode() == BPF::CORE_LD64 ||
             MI->getOpcode() == BPF::CORE_LD32 ||
             MI->getOpcode() == BPF::CORE_ST ||
             MI->getOpcode() == BPF::CORE_SHIFT) {
    const MachineOperand &MO = MI->getOperand(3);
    if (MO.isGlobal()) {
      const GlobalValue *GVal = MO.getGlobal();
      auto *GVar = dyn_cast<GlobalVariable>(GVal);
      if (GVar && GVar->hasAttribute(BPFCoreSharedInfo::AmaAttr)) {
        uint32_t Imm = PatchImms[GVar].first;
        OutMI.setOpcode(MI->getOperand(1).getImm());
        if (MI->getOperand(0).isImm())
          OutMI.addOperand(MCOperand::createImm(MI->getOperand(0).getImm()));
        else
          OutMI.addOperand(MCOperand::createReg(MI->getOperand(0).getReg()));
        OutMI.addOperand(MCOperand::createReg(MI->getOperand(2).getReg()));
        OutMI.addOperand(MCOperand::createImm(Imm));
        return true;
      }
    }
  }
  return false;
}

void BPFBTFDebug::endModule() {
  // BPF_TRAP may lose its call site during MachineIR optimization,
  // so ensure its prototype is always emitted.
  for (const Function &F : *MMI->getModule()) {
    if (F.getName() == BPF_TRAP)
      processFuncPrototypes(&F);
  }

  BTFDebug::endModule();
}
