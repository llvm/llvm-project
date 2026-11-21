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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

using namespace llvm;

#define DEBUG_TYPE "btf-debug"

#define GET_CC_REGISTER_LISTS
#include "BPFGenCallingConv.inc"

static const DIType *stripDITypeAttributes(const DIType *Ty) {
  while (const auto *DTy = dyn_cast_or_null<DIDerivedType>(Ty)) {
    switch (DTy->getTag()) {
    case dwarf::DW_TAG_atomic_type:
    case dwarf::DW_TAG_const_type:
    case dwarf::DW_TAG_restrict_type:
    case dwarf::DW_TAG_typedef:
    case dwarf::DW_TAG_volatile_type:
      Ty = DTy->getBaseType();
      break;
    default:
      return Ty;
    }
  }
  return Ty;
}

static bool sourceArgMatchesIRType(const DIType *SourceTy, Type *IRTy) {
  SourceTy = stripDITypeAttributes(SourceTy);

  // All pointers are opaque in LLVM IR, so any source-level pointer matches any
  // IR pointer regardless of pointee type.
  if (const auto *DTy = dyn_cast<DIDerivedType>(SourceTy))
    return DTy->getTag() == dwarf::DW_TAG_pointer_type && IRTy->isPointerTy();

  if (const auto *BTy = dyn_cast<DIBasicType>(SourceTy)) {
    uint64_t SizeInBits = BTy->getSizeInBits();
    if (BTy->getEncoding() == dwarf::DW_ATE_float)
      return IRTy->isFloatingPointTy() &&
             IRTy->getPrimitiveSizeInBits() == SizeInBits;
    // _Bool is 8 bits in DWARF/source but lowered to i1 in LLVM IR.
    if (BTy->getEncoding() == dwarf::DW_ATE_boolean && IRTy->isIntegerTy(1))
      return true;
    return IRTy->isIntegerTy(SizeInBits);
  }

  const auto *CTy = dyn_cast<DICompositeType>(SourceTy);
  if (!CTy)
    return false;

  switch (CTy->getTag()) {
  case dwarf::DW_TAG_enumeration_type:
    return IRTy->isIntegerTy(CTy->getSizeInBits());
  default:
    return false;
  }
}

/// Collect the physical register each source argument lives in by scanning
/// DBG_VALUE instructions in the entry block.  A DBG_VALUE is only recorded
/// when its register either (a) has not been redefined by any preceding
/// non-debug instruction (i.e. it still holds the caller-passed value), or
/// (b) was most recently loaded from the stack via $r11 (a stack-passed
/// argument beyond the first five register args). For each argument only the
/// first eligible DBG_VALUE is recorded, since that is its entry location.
///
/// There is another case where DBG_VALUE is not emitted due to
/// AssignmentTrackingAnalysis which determines that a variable is
/// always stack-homed, and describes the variable via MachineFunction's
/// VariableDbgInfo (setVariableDbgInfo with a frame index). To recover the
/// register for those arguments, we also track stores of un-redefined physical
/// registers to stack frame objects during the entry-block walk (using
/// MachineMemOperands to identify the target frame index), then match them
/// against VariableDbgInfo entries after the scan.
static SmallVector<std::pair<uint32_t, Register>, 8>
collectNocallEntryArgRegs(const MachineFunction &MF) {
  SmallDenseMap<uint32_t, Register> EntryRegMap;
  const DISubprogram *SP = MF.getFunction().getSubprogram();
  SmallDenseSet<Register> DefinedRegs, StackLoadRegs;

  // Build a reverse map from IR alloca to frame index so we can
  // identify which frame object a store targets via its MachineMemOperand.
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  SmallDenseMap<const Value *, int> AllocaToFI;
  for (int I = 0, N = MFI.getObjectIndexEnd(); I < N; ++I)
    if (const AllocaInst *AI = MFI.getObjectAllocation(I))
      AllocaToFI[AI] = I;

  // Maps frame index → first physical register stored there before
  // that register is redefined.
  SmallDenseMap<int, Register> FrameIndexToReg;

  for (const MachineInstr &MI : MF.front()) {
    if (MI.isDebugValue()) {
      // Skip indirect DBG_VALUEs — the register is a base address for a
      // memory location, not the argument value itself.
      if (MI.isIndirectDebugValue())
        continue;

      const DILocalVariable *DV = MI.getDebugVariable();
      if (!DV || !DV->getArg() || DV->getScope()->getSubprogram() != SP)
        continue;

      uint32_t Arg = DV->getArg();
      const MachineOperand &MO = MI.getDebugOperand(0);
      if (!MO.isReg() || !MO.getReg().isPhysical())
        continue;

      if (!DefinedRegs.contains(MO.getReg()) ||
          StackLoadRegs.contains(MO.getReg()))
        EntryRegMap.try_emplace(Arg, MO.getReg());
      continue;
    }

    // Track stores of unredefined physical registers to stack frame
    // objects.  Use MachineMemOperands to identify the target frame
    // index rather than assuming a particular addressing mode.
    if (MI.mayStore() && !MI.isCall() && MI.getOperand(0).isReg()) {
      Register SrcReg = MI.getOperand(0).getReg();
      if (SrcReg.isPhysical() && !DefinedRegs.contains(SrcReg)) {
        for (const MachineMemOperand *MMO : MI.memoperands()) {
          const Value *V = MMO->getValue();
          if (!V)
            continue;
          auto It = AllocaToFI.find(V);
          if (It != AllocaToFI.end())
            FrameIndexToReg.try_emplace(It->second, SrcReg);
        }
      }
    }

    for (const MachineOperand &MO : MI.operands())
      if (MO.isReg() && MO.isDef() && MO.getReg().isPhysical()) {
        DefinedRegs.insert(MO.getReg());
        StackLoadRegs.erase(MO.getReg());
      }

    // Detect stack argument loads: $rX = LDD $r11, offset.
    if (MI.getOpcode() == BPF::LDD && MI.getOperand(1).getReg() == BPF::R11)
      StackLoadRegs.insert(MI.getOperand(0).getReg());
  }

  // Check VariableDbgInfo for args that AssignmentTrackingAnalysis described
  // via setVariableDbgInfo (single-loc stack-homed variables) rather than
  // DBG_VALUE instructions.
  for (const auto &VI : MF.getVariableDbgInfo()) {
    if (!VI.Var || !VI.Var->getArg() || !VI.inStackSlot())
      continue;
    if (VI.Var->getScope()->getSubprogram() != SP)
      continue;
    uint32_t Arg = VI.Var->getArg();
    if (EntryRegMap.count(Arg))
      continue;
    auto It = FrameIndexToReg.find(VI.getStackSlot());
    if (It != FrameIndexToReg.end())
      EntryRegMap[Arg] = It->second;
  }

  SmallVector<std::pair<uint32_t, Register>, 8> AliveArgs(EntryRegMap.begin(),
                                                          EntryRegMap.end());
  llvm::sort(AliveArgs, llvm::less_first());
  return AliveArgs;
}

/// Check whether the optimized IR signature matches the surviving source
/// arguments precisely enough to emit a filtered BTF prototype.
/// Requires exact IR/source arg count match, matching types, and correct
/// BPF register order (R1..R5) for register args.
static bool canUseNocallOptimizedSignature(
    const MachineFunction &MF, DITypeArray Elements,
    ArrayRef<std::pair<uint32_t, Register>> AliveArgs,
    const TargetRegisterInfo &TRI) {
  if (MF.getFunction().arg_size() != AliveArgs.size()) {
    LLVM_DEBUG(dbgs() << "BTF skip " << MF.getName() << ": IR arg count ("
                      << MF.getFunction().arg_size() << ") != alive arg count ("
                      << AliveArgs.size() << ")\n");
    return false;
  }

  auto ArgIt = MF.getFunction().arg_begin();
  for (unsigned I = 0, N = AliveArgs.size(); I < N; ++I, ++ArgIt) {
    auto [ArgNo, Reg] = AliveArgs[I];
    if (!sourceArgMatchesIRType(Elements[ArgNo], ArgIt->getType())) {
      LLVM_DEBUG(dbgs() << "BTF skip " << MF.getName()
                        << ": type mismatch for source arg " << ArgNo
                        << " at IR position " << I << "\n");
      return false;
    }

    if (I >= std::size(CC_BPF64_ArgRegs))
      continue;

    int DwarfReg = TRI.getDwarfRegNum(Reg, false);
    if (DwarfReg != static_cast<int>(I + 1)) {
      LLVM_DEBUG(dbgs() << "BTF skip " << MF.getName() << ": arg " << ArgNo
                        << " in DWARF reg " << DwarfReg << ", expected "
                        << (I + 1) << "\n");
      return false;
    }
  }

  return true;
}

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
      const DIType *MemberBaseType = MemberType->getBaseType();

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

  // The trap call may have been removed during machine-level optimization.
  for (const Function &F : *MMI->getModule())
    if (F.getName() == BPF_TRAP)
      processFuncPrototypes(&F);
}

void BPFBTFDebug::generatePatchImmReloc(const MCSymbol *ORSym, uint32_t RootId,
                                        const GlobalVariable *GVar,
                                        bool IsAma) {
  BTFFieldReloc FieldReloc;
  FieldReloc.Label = ORSym;
  FieldReloc.TypeID = RootId;

  if (IsAma) {
    auto Reloc = BPFCoreSharedInfo::AmaRelocation::decode(GVar->getName());
    FieldReloc.OffsetNameOff = addString(Reloc.AccessString);
    FieldReloc.RelocKind = Reloc.Kind;
    PatchImms[GVar] = {Reloc.PatchImm, Reloc.Kind};
  } else {
    StringRef RelocStr = GVar->getName().split('$').second;
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

uint32_t BPFBTFDebug::processFunctionType(
    const MachineFunction *MF, const DISubprogram *SP,
    const SmallDenseMap<uint32_t, StringRef> &FuncArgNames) {
  uint32_t ProtoTypeId = 0;
  uint8_t Scope = SP->isLocalToUnit() ? BTF::FUNC_STATIC : BTF::FUNC_GLOBAL;
  bool IsNocall = SP->getType()->getCC() == dwarf::DW_CC_nocall;
  bool VoidReturn = MF->getFunction().getReturnType()->isVoidTy();

  if (IsNocall) {
    // For DW_CC_nocall functions, try to build a FUNC_PROTO reflecting
    // the true ABI: only parameters that survived optimization and whose
    // first 5 arguments map to the correct BPF registers (R1-R5).
    const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
    DITypeArray Elements = SP->getType()->getTypeArray();
    SmallVector<std::pair<uint32_t, Register>, 8> AliveArgs =
        collectNocallEntryArgRegs(*MF);

    if (canUseNocallOptimizedSignature(*MF, Elements, AliveArgs, *TRI)) {
      SmallVector<uint32_t, 8> AliveParamIndices;
      SmallDenseMap<uint32_t, uint32_t> ArgIndexMap;
      for (auto [I, ArgReg] : llvm::enumerate(AliveArgs)) {
        AliveParamIndices.push_back(ArgReg.first);
        ArgIndexMap[ArgReg.first] = I;
      }

      if (!VoidReturn)
        visitTypeEntry(Elements[0]);
      for (uint32_t ArgNo : AliveParamIndices)
        visitTypeEntry(Elements[ArgNo]);

      auto TypeEntry = std::make_unique<BTFTypeFuncProto>(
          SP->getType(), AliveParamIndices.size(), FuncArgNames, true,
          AliveParamIndices, VoidReturn);
      ProtoTypeId = addType(std::move(TypeEntry));
      return processDISubprogram(SP, ProtoTypeId, Scope, &ArgIndexMap);
    }
  }

  // Fall back to the full source prototype, still voiding the return
  // type if compiler removed it.
  visitSubroutineType(SP->getType(), true, FuncArgNames, ProtoTypeId,
                      VoidReturn);
  if (!ProtoTypeId)
    return 0;
  return processDISubprogram(SP, ProtoTypeId, Scope);
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
        if (GVar->hasAttribute(BPFCoreSharedInfo::AmaAttr) ||
            GVar->hasAttribute(BPFCoreSharedInfo::TypeIdAttr)) {
          auto Imm = PatchImms[GVar];
          auto Reloc = Imm.second;
          if (Reloc == BTF::ENUM_VALUE_EXISTENCE || Reloc == BTF::ENUM_VALUE ||
              Reloc == BTF::BTF_TYPE_ID_LOCAL ||
              Reloc == BTF::BTF_TYPE_ID_REMOTE)
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
