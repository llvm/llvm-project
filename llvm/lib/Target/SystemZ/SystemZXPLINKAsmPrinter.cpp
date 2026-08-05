//===-- SystemZXPLINKAsmPrinter.cpp - SystemZ XPLINK asm printer ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemZXPLINKAsmPrinter class.
//
//===----------------------------------------------------------------------===//

#include "SystemZXPLINKAsmPrinter.h"
#include "MCTargetDesc/SystemZMCTargetDesc.h"
#include "MCTargetDesc/SystemZTargetStreamer.h"
#include "SystemZFrameLowering.h"
#include "SystemZInstrInfo.h"
#include "SystemZMCInstLower.h"
#include "SystemZMachineFunctionInfo.h"
#include "SystemZSubtarget.h"
#include "SystemZTargetObjectFile.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCSymbolGOFF.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/ConvertEBCDIC.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;

SystemZXPLINKAsmPrinter::SystemZXPLINKAsmPrinter(
    TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer)
    : SystemZAsmPrinter(TM, std::move(Streamer)),
      ADATable(TM.getPointerSize(0)) {}

bool SystemZXPLINKAsmPrinter::doInitialization(Module &M) {
  SM.reset();

  // In HLASM, the only way to represent aliases is to use the
  // extra-label-at-definition strategy. This is similar to the AIX
  // implementation with the additional caveat that all symbol attributes must
  // be emitted before the label is emitted.
  // Construct an aliasing list for each GlobalObject.
  for (const auto &Alias : M.aliases()) {
    const GlobalObject *Aliasee = Alias.getAliaseeObject();
    if (!Aliasee)
      OutContext.reportError(
          {}, "Alias without a base object is not yet supported on z/OS.");

    bool IsFunc = isa<Function>(Aliasee->stripPointerCasts());
    if (IsFunc) {
      if (Alias.hasWeakLinkage() || Alias.hasLinkOnceLinkage())
        OutContext.reportError({},
                                "Weak alias/reference not supported on z/OS");

      GOAliasMap[Aliasee].push_back(&Alias);
    } else
      OutContext.reportError(
          {}, "Only aliases to functions is supported in GOFF.");
  }
  return AsmPrinter::doInitialization(M);
}

// The XPLINK ABI requires that a no-op encoding the call type is emitted after
// each call to a subroutine. This information can be used by the called
// function to determine its entry point, e.g. for generating a backtrace. The
// call type is encoded as a register number in the bcr instruction. See
// enumeration CallType for the possible values.
void SystemZXPLINKAsmPrinter::emitCallInformation(CallType CT) {
  EmitToStreamer(*OutStreamer,
                 MCInstBuilder(SystemZ::BCRAsm)
                     .addImm(0)
                     .addReg(SystemZMC::GR64Regs[static_cast<unsigned>(CT)]));
}

uint32_t SystemZXPLINKAsmPrinter::AssociatedDataAreaTable::insert(
    const MCSymbol *Sym, unsigned SlotKind) {
  auto Key = std::make_pair(Sym, SlotKind);
  auto It = Displacements.find(Key);

  if (It != Displacements.end())
    return (*It).second;

  // Determine length of descriptor.
  uint32_t Length;
  switch (SlotKind) {
  case SystemZII::MO_ADA_DIRECT_FUNC_DESC:
    Length = 2 * PointerSize;
    break;
  default:
    Length = PointerSize;
    break;
  }

  uint32_t Displacement = NextDisplacement;
  Displacements[std::make_pair(Sym, SlotKind)] = NextDisplacement;
  NextDisplacement += Length;

  return Displacement;
}

uint32_t SystemZXPLINKAsmPrinter::AssociatedDataAreaTable::insert(
    const MachineOperand MO) {
  MCSymbol *Sym;
  if (MO.getType() == MachineOperand::MO_GlobalAddress) {
    const GlobalValue *GV = MO.getGlobal();
    Sym = MO.getParent()->getMF()->getTarget().getSymbol(GV);
    assert(Sym && "No symbol");
  } else if (MO.getType() == MachineOperand::MO_ExternalSymbol) {
    const char *SymName = MO.getSymbolName();
    Sym = MO.getParent()->getMF()->getContext().getOrCreateSymbol(SymName);
    assert(Sym && "No symbol");
  } else
    llvm_unreachable("Unexpected operand type");

  unsigned ADAslotType = MO.getTargetFlags();
  return insert(Sym, ADAslotType);
}

void SystemZXPLINKAsmPrinter::emitInstruction(const MachineInstr *MI) {
  SystemZMCInstLower Lower(MF->getContext(), *this);
  MCInst LoweredMI;

  switch (MI->getOpcode()) {
  case SystemZ::CallBRASL_XPLINK64:
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(SystemZ::BRASL)
                       .addReg(SystemZ::R7D)
                       .addExpr(Lower.getExpr(MI->getOperand(0),
                                              SystemZ::S_None)));
    emitCallInformation(CallType::BRASL7);
    return;

  case SystemZ::CallBASR_XPLINK64:
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(SystemZ::BASR)
                       .addReg(SystemZ::R7D)
                       .addReg(MI->getOperand(0).getReg()));
    emitCallInformation(CallType::BASR76);
    return;

  case SystemZ::Return_XPLINK:
    LoweredMI = MCInstBuilder(SystemZ::B)
                    .addReg(SystemZ::R7D)
                    .addImm(2)
                    .addReg(0);
    break;

  case SystemZ::CondReturn_XPLINK:
    LoweredMI = MCInstBuilder(SystemZ::BC)
                    .addImm(MI->getOperand(0).getImm())
                    .addImm(MI->getOperand(1).getImm())
                    .addReg(SystemZ::R7D)
                    .addImm(2)
                    .addReg(0);
    break;

  case SystemZ::CallBASR_STACKEXT:
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(SystemZ::BASR)
                       .addReg(SystemZ::R3D)
                       .addReg(MI->getOperand(0).getReg()));
    emitCallInformation(CallType::BASR33);
    return;

  case SystemZ::ADA_ENTRY_VALUE:
  case SystemZ::ADA_ENTRY: {
    const SystemZSubtarget &Subtarget = MF->getSubtarget<SystemZSubtarget>();
    const SystemZInstrInfo *TII = Subtarget.getInstrInfo();
    uint32_t Disp = ADATable.insert(MI->getOperand(1));
    Register TargetReg = MI->getOperand(0).getReg();

    Register ADAReg = MI->getOperand(2).getReg();
    Disp += MI->getOperand(3).getImm();
    bool LoadAddr = MI->getOpcode() == SystemZ::ADA_ENTRY;

    unsigned Op0 = LoadAddr ? SystemZ::LA : SystemZ::LG;
    unsigned Op = TII->getOpcodeForOffset(Op0, Disp);

    Register IndexReg = 0;
    if (!Op) {
      if (TargetReg != ADAReg) {
        IndexReg = TargetReg;
        // Use TargetReg to store displacement.
        EmitToStreamer(
            *OutStreamer,
            MCInstBuilder(SystemZ::LLILF).addReg(TargetReg).addImm(Disp));
      } else
        EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::ALGFI)
                                        .addReg(TargetReg)
                                        .addReg(TargetReg)
                                        .addImm(Disp));
      Disp = 0;
      Op = Op0;
    }
    EmitToStreamer(*OutStreamer, MCInstBuilder(Op)
                                    .addReg(TargetReg)
                                    .addReg(ADAReg)
                                    .addImm(Disp)
                                    .addReg(IndexReg));
    return;
  }

  default:
    SystemZAsmPrinter::emitInstruction(MI);
    return;
  }
  EmitToStreamer(*OutStreamer, LoweredMI);
}

void SystemZXPLINKAsmPrinter::emitXXStructorList(const DataLayout &DL,
                                                  const Constant *List,
                                                  bool IsCtor) {
  assert(TM.getTargetTriple().isOSBinFormatGOFF() && "Only GOFF supported");

  SmallVector<Structor, 8> Structors;
  preprocessXXStructorList(DL, List, Structors);
  if (Structors.empty())
    return;

  const Align Align = llvm::Align(4);
  const TargetLoweringObjectFileGOFF &Obj =
      static_cast<const TargetLoweringObjectFileGOFF &>(getObjFileLowering());
  for (Structor &S : Structors) {
    MCSectionGOFF *Section =
        static_cast<MCSectionGOFF *>(Obj.getStaticXtorSection(S.Priority));
    OutStreamer->switchSection(Section);
    if (OutStreamer->getCurrentSection() != OutStreamer->getPreviousSection())
      emitAlignment(Align);

    // The priority is provided as an input to getStaticXtorSection(), and is
    // recalculated within that function as `Prio` going to going into the
    // PR section.
    // This priority retrieved via the `SortKey` below is the recalculated
    // Priority.
    uint32_t XtorPriority = Section->getPRAttributes().SortKey;

    const GlobalValue *GV = dyn_cast<GlobalValue>(S.Func->stripPointerCasts());
    assert(GV && "C++ xxtor pointer was not a GlobalValue!");
    MCSymbolGOFF *Symbol = static_cast<MCSymbolGOFF *>(getSymbol(GV));

    // @@SQINIT entry: { unsigned prio; void (*ctor)();  void (*dtor)(); }

    unsigned PointerSizeInBytes = DL.getPointerSize();

    auto &Ctx = OutStreamer->getContext();
    const MCExpr *ADAFuncRefExpr;
    unsigned SlotKind = SystemZII::MO_ADA_DIRECT_FUNC_DESC;

    MCSectionGOFF *ADASection =
        static_cast<MCSectionGOFF *>(Obj.getADASection());
    assert(ADASection && "ADA section must exist for GOFF targets!");
    const MCSymbol *ADASym = ADASection->getBeginSymbol();
    assert(ADASym && "ADA symbol should already be set!");

    ADAFuncRefExpr = MCBinaryExpr::createAdd(
        MCSpecifierExpr::create(MCSymbolRefExpr::create(ADASym, OutContext),
                                SystemZ::S_QCon, OutContext),
        MCConstantExpr::create(ADATable.insert(Symbol, SlotKind), Ctx), Ctx);

    emitInt32(XtorPriority);
    if (IsCtor) {
      OutStreamer->emitValue(ADAFuncRefExpr, PointerSizeInBytes);
      OutStreamer->emitIntValue(0, PointerSizeInBytes);
    } else {
      OutStreamer->emitIntValue(0, PointerSizeInBytes);
      OutStreamer->emitValue(ADAFuncRefExpr, PointerSizeInBytes);
    }
  }
}

void SystemZXPLINKAsmPrinter::emitEndOfAsmFile(Module &M) {
  auto *ZOS = getTargetStreamer();
  emitADASection();
  emitIDRLSection(M);
  // On z/OS, we need to associate an external data reference with an ED
  // symbol, for which we use the the ED of the ADA. We also need to mark the
  // reference as being to data, otherwise we cannot bind with code generated
  // by XL.
  for (auto &GO : M.global_objects()) {
    if (auto *GV = dyn_cast<GlobalVariable>(&GO)) {
      if (!GV->hasInitializer()) {
        MCSymbol *Sym = getSymbol(GV);
        ZOS->emitADA(Sym, OutContext.getObjectFileInfo()->getADASection());
        OutStreamer->emitSymbolAttribute(Sym, MCSA_ELF_TypeObject);
      }
    }
  }
}

void SystemZXPLINKAsmPrinter::emitADASection() {
  OutStreamer->pushSection();

  const unsigned PointerSize = getDataLayout().getPointerSize();
  OutStreamer->switchSection(getObjFileLowering().getADASection());

  auto *ZOS = getTargetStreamer();
  unsigned EmittedBytes = 0;
  for (auto &Entry : ADATable.getTable()) {
    const MCSymbol *Sym;
    unsigned SlotKind;
    std::tie(Sym, SlotKind) = Entry.first;
    unsigned Offset = Entry.second;
    assert(Offset == EmittedBytes && "Offset not as expected");
    (void)EmittedBytes;
#define EMIT_COMMENT(Str)                                                      \
  OutStreamer->AddComment(Twine("Offset ")                                     \
                              .concat(utostr(Offset))                          \
                              .concat(" " Str " ")                             \
                              .concat(Sym->getName()));
    switch (SlotKind) {
    case SystemZII::MO_ADA_DIRECT_FUNC_DESC:
      // Language Environment DLL logic requires function descriptors, for
      // imported functions, that are placed in the ADA to be 8 byte aligned.
      EMIT_COMMENT("function descriptor of");
      OutStreamer->emitValue(
          MCSpecifierExpr::create(MCSymbolRefExpr::create(Sym, OutContext),
                                  SystemZ::S_RCon, OutContext),
          PointerSize);
      OutStreamer->emitValue(
          MCSpecifierExpr::create(MCSymbolRefExpr::create(Sym, OutContext),
                                  SystemZ::S_VCon, OutContext),
          PointerSize);
      EmittedBytes += PointerSize * 2;
      break;
    case SystemZII::MO_ADA_DATA_SYMBOL_ADDR:
      EMIT_COMMENT("pointer to data symbol");
      OutStreamer->emitValue(
          MCSpecifierExpr::create(MCSymbolRefExpr::create(Sym, OutContext),
                                  SystemZ::S_None, OutContext),
          PointerSize);
      EmittedBytes += PointerSize;
      break;
    case SystemZII::MO_ADA_INDIRECT_FUNC_DESC: {
      MCSymbol *Alias = OutContext.getOrCreateSymbol(
          Twine(Sym->getName()).concat("@indirect"));
      OutStreamer->emitSymbolAttribute(Alias, MCSA_IndirectSymbol);
      OutStreamer->emitSymbolAttribute(Alias, MCSA_ELF_TypeFunction);
      OutStreamer->emitSymbolAttribute(Alias, MCSA_Global);
      OutStreamer->emitSymbolAttribute(Alias, MCSA_Extern);
      MCSymbolGOFF *GOFFSym =
          static_cast<llvm::MCSymbolGOFF *>(const_cast<llvm::MCSymbol *>(Sym));
      ZOS->emitExternalName(Alias, GOFFSym->getExternalName());
      EMIT_COMMENT("pointer to function descriptor");
      OutStreamer->emitValue(
          MCSpecifierExpr::create(MCSymbolRefExpr::create(Alias, OutContext),
                                  SystemZ::S_VCon, OutContext),
          PointerSize);
      EmittedBytes += PointerSize;
      break;
    }
    default:
      llvm_unreachable("Unexpected slot kind");
    }
#undef EMIT_COMMENT
  }
  OutStreamer->popSection();
}

static std::string getProductID(Module &M) {
  std::string ProductID;
  if (auto *MD = M.getModuleFlag("zos_product_id"))
    ProductID = cast<MDString>(MD)->getString().str();
  if (ProductID.empty())
    ProductID = "LLVM";
  return ProductID;
}

static uint32_t getProductVersion(Module &M) {
  if (auto *VersionVal = mdconst::extract_or_null<ConstantInt>(
          M.getModuleFlag("zos_product_major_version")))
    return VersionVal->getZExtValue();
  return LLVM_VERSION_MAJOR;
}

static uint32_t getProductRelease(Module &M) {
  if (auto *ReleaseVal = mdconst::extract_or_null<ConstantInt>(
          M.getModuleFlag("zos_product_minor_version")))
    return ReleaseVal->getZExtValue();
  return LLVM_VERSION_MINOR;
}

static uint32_t getProductPatch(Module &M) {
  if (auto *PatchVal = mdconst::extract_or_null<ConstantInt>(
          M.getModuleFlag("zos_product_patchlevel")))
    return PatchVal->getZExtValue();
  return LLVM_VERSION_PATCH;
}

static time_t getTranslationTime(Module &M) {
  std::time_t Time = 0;
  if (auto *Val = mdconst::extract_or_null<ConstantInt>(
          M.getModuleFlag("zos_translation_time"))) {
    long SecondsSinceEpoch = Val->getSExtValue();
    Time = static_cast<time_t>(SecondsSinceEpoch);
  }
  return Time;
}

void SystemZXPLINKAsmPrinter::emitIDRLSection(Module &M) {
  OutStreamer->pushSection();
  OutStreamer->switchSection(getObjFileLowering().getIDRLSection());
  constexpr unsigned IDRLDataLength = 30;
  std::time_t Time = getTranslationTime(M);

  uint32_t ProductVersion = getProductVersion(M);
  uint32_t ProductRelease = getProductRelease(M);

  std::string ProductID = getProductID(M);

  SmallString<IDRLDataLength + 1> TempStr;
  raw_svector_ostream O(TempStr);
  O << formatv("{0,-10}{1,0-2:d}{2,0-2:d}{3:%Y%m%d%H%M%S}{4,0-2}",
               ProductID.substr(0, 10).c_str(), ProductVersion, ProductRelease,
               llvm::sys::toUtcTime(Time), "0");
  SmallString<IDRLDataLength> Data;
  ConverterEBCDIC::convertToEBCDIC(TempStr, Data);

  OutStreamer->emitInt8(0);               // Reserved.
  OutStreamer->emitInt8(3);               // Format.
  OutStreamer->emitInt16(IDRLDataLength); // Length.
  OutStreamer->emitBytes(Data.str());
  OutStreamer->popSection();
}

void SystemZXPLINKAsmPrinter::emitFunctionBodyEnd() {
  // Emit symbol for the end of function if the z/OS target streamer
  // is used. This is needed to calculate the size of the function.
  auto *ZOS = getTargetStreamer();
  OutStreamer->emitLabel(ZOS->DeferredPPA1.back().FnEnd);
}

// Determine the end of the prolog and the instructions which updates the stack
// register, and attach symbols to those instructions.
static void determinePrologueStackUpdateSym(MachineFunction *MF,
                                            MCSymbol *&EndOfPrologSym,
                                            MCSymbol *&StackUpdateSym) {
  EndOfPrologSym = nullptr;
  StackUpdateSym = nullptr;

  // Scan the basic block for the FENCE instruction which marks the end
  // of the prologue. We know
  // the prologue is spread at most across the first 3 basic blocks. Also record
  // the first instruction updating the stack pointer.
  const SystemZSubtarget &STI = MF->getSubtarget<SystemZSubtarget>();
  auto &Regs = STI.getSpecialRegisters<SystemZXPLINK64Registers>();
  MachineInstr *EndOfPrologMI = nullptr;
  MachineInstr *StackUpdateMI = nullptr;
  unsigned BBCount = 1;

  for (auto &MBB : *MF) {
    for (auto &I : MBB) {
      if (I.getOpcode() == SystemZ::FENCE)
        EndOfPrologMI = &I;
      else if (!StackUpdateMI) {
        unsigned Opcode = I.getOpcode();
        // TODO: We can instead emit a pseudo instruction in
        // SystemZFrameLowering to represent a stack adjustment instruction, and
        // check for that here, instead of having to check for multiple
        // instructions.
        if ((Opcode == SystemZ::AGHI || Opcode == SystemZ::AGFI) &&
            I.getOperand(0).getReg() == Regs.getStackPointerRegister())
          StackUpdateMI = &I;
      }
    }

    // Prologue can be a max of 3 BBs if we need to call stack extension code
    if (EndOfPrologMI || BBCount == 3)
      break;

    ++BBCount;
  }

  // Leaf functions do not have a prologue.
  if (EndOfPrologMI == nullptr)
    return;

#ifdef EXPENSIVE_CHECKS
  // Check that the prolog length is valid.
  auto *TII = STI.getInstrInfo();
  size_t Size = 0;

  for (auto &MBB : *MF) {
    bool TerminateLoop = false;
    for (auto &I : MBB) {
      Size += TII->getInstSizeInBytes(I);
      if (&I == EndOfPrologMI) {
        TerminateLoop = true;
        break;
      }
    }
    if (TerminateLoop)
      break;
  }
  if (Size > 128)
    report_fatal_error(
        Twine(MF->getName()).concat(": Prolog exceeds 128 bytes"));
#endif

  // Attach a temporary symbol to mark the end of the prolog.
  EndOfPrologSym = MF->getContext().createTempSymbol("end_of_prologue");
  EndOfPrologMI->setPostInstrSymbol(*MF, EndOfPrologSym);

  if (StackUpdateMI) {
    StackUpdateSym = MF->getContext().createTempSymbol("stack_update");
    StackUpdateMI->setPreInstrSymbol(*MF, StackUpdateSym);
  }
}

void SystemZXPLINKAsmPrinter::calculatePPA1() {
  auto *ZOS = getTargetStreamer();
  assert(ZOS->PPA2Sym != nullptr && "PPA2 Symbol not defined");

  SystemZTargetzOSStreamer::PPA1Info Info;

  const TargetRegisterInfo *TRI = MF->getRegInfo().getTargetRegisterInfo();
  const SystemZSubtarget &Subtarget = MF->getSubtarget<SystemZSubtarget>();

  const SystemZMachineFunctionInfo *ZFI =
      MF->getInfo<SystemZMachineFunctionInfo>();
  const auto *ZFL = static_cast<const SystemZXPLINKFrameLowering *>(
      Subtarget.getFrameLowering());
  const MachineFrameInfo &MFFrame = MF->getFrameInfo();

  // Get saved GPR/FPR/VPR masks.
  const std::vector<CalleeSavedInfo> &CSI = MFFrame.getCalleeSavedInfo();
  uint16_t SavedGPRMask = 0;
  uint16_t SavedFPRMask = 0;
  uint8_t SavedVRMask = 0;
  int64_t OffsetFPR = 0;
  int64_t OffsetVR = 0;
  const int64_t TopOfStack =
      MFFrame.getOffsetAdjustment() + MFFrame.getStackSize();

  // Loop over the spilled registers. The CalleeSavedInfo can't be used because
  // it does not contain all spilled registers.
  for (unsigned I = ZFI->getSpillGPRRegs().LowGPR,
                E = ZFI->getSpillGPRRegs().HighGPR;
       I && E && I <= E; ++I) {
    unsigned V = TRI->getEncodingValue((Register)I);
    assert(V < 16 && "GPR index out of range");
    SavedGPRMask |= 1 << (15 - V);
  }

  for (auto &CS : CSI) {
    unsigned Reg = CS.getReg();
    unsigned I = TRI->getEncodingValue(Reg);

    if (SystemZ::FP64BitRegClass.contains(Reg)) {
      assert(I < 16 && "FPR index out of range");
      SavedFPRMask |= 1 << (15 - I);
      int64_t Temp = MFFrame.getObjectOffset(CS.getFrameIdx());
      if (Temp < OffsetFPR)
        OffsetFPR = Temp;
    } else if (SystemZ::VR128BitRegClass.contains(Reg)) {
      assert(I >= 16 && I <= 23 && "VPR index out of range");
      unsigned BitNum = I - 16;
      SavedVRMask |= 1 << (7 - BitNum);
      int64_t Temp = MFFrame.getObjectOffset(CS.getFrameIdx());
      if (Temp < OffsetVR)
        OffsetVR = Temp;
    }
  }

  // Adjust the offset.
  OffsetFPR += (OffsetFPR < 0) ? TopOfStack : 0;
  OffsetVR += (OffsetVR < 0) ? TopOfStack : 0;

  // Get alloca register.
  uint8_t FrameReg = TRI->getEncodingValue(TRI->getFrameRegister(*MF));
  uint8_t AllocaReg = ZFL->hasFP(*MF) ? FrameReg : 0;
  assert(AllocaReg < 16 && "Can't have alloca register larger than 15");

  MCSymbol *PersonalityRoutine = nullptr;
  MCSymbol *GCCEH = nullptr;
  uint64_t PersonalityADADisp = 0;
  uint64_t GCCEHADADisp = 0;
  if (!MF->getLandingPads().empty()) {
    const Function *Per = dyn_cast<Function>(
        MF->getFunction().getPersonalityFn()->stripPointerCasts());
    PersonalityRoutine = Per ? MF->getTarget().getSymbol(Per) : nullptr;
    if (PersonalityRoutine) {
      GCCEH = MF->getContext().getOrCreateSymbol(
          Twine("GCC_except_table") + Twine(MF->getFunctionNumber()));
      PersonalityADADisp = ADATable.insert(
          PersonalityRoutine, SystemZII::MO_ADA_INDIRECT_FUNC_DESC);
      GCCEHADADisp = ADATable.insert(GCCEH, SystemZII::MO_ADA_DATA_SYMBOL_ADDR);
    }
  }

  // Get the name of the function, with suffix _.
  std::string N(MF->getFunction().hasName()
                    ? Twine(MF->getFunction().getName()).concat("_").str()
                    : "");

  // Calculate the lables for the prolog size and the stack update symbol.
  MCSymbol *EndOfPrologSym;
  MCSymbol *StackUpdateSym;
  determinePrologueStackUpdateSym(MF, EndOfPrologSym, StackUpdateSym);

  // Save the calculated values.
  if (MF->getFunction().hasFnAttribute("zos-ppa1-name"))
    Info.Name =
        MF->getFunction().getFnAttribute("zos-ppa1-name").getValueAsString();
  else if (MF->getFunction().hasName())
    Info.Name = MF->getFunction().getName();

  Info.PPA1 = OutContext.createTempSymbol(Twine("PPA1_").concat(N), true);
  Info.EPMarker = OutContext.createTempSymbol(Twine("EPM_").concat(N), true);
  Info.FnEnd = OutContext.createTempSymbol(Twine(N).concat("end_"));
  Info.Fn = CurrentFnSym;
  Info.EndOfProlog = EndOfPrologSym;
  Info.StackUpdate = StackUpdateSym;
  Info.PersonalityADADisp = PersonalityADADisp;
  Info.GCCEHADADisp = GCCEHADADisp;
  Info.OffsetFPR = OffsetFPR;
  Info.OffsetVR = OffsetVR;
  Info.CallFrameSize = MFFrame.getMaxCallFrameSize();
  Info.SizeOfFnParams = ZFI->getSizeOfFnParams();
  Info.SavedGPRMask = SavedGPRMask;
  Info.SavedFPRMask = SavedFPRMask;
  Info.SavedVRMask = SavedVRMask;
  Info.FrameReg = FrameReg;
  Info.AllocaReg = AllocaReg;
  Info.IsVarArg = MF->getFunction().isVarArg();
  Info.HasStackProtector = MFFrame.hasStackProtectorIndex();

  ZOS->DeferredPPA1.push_back(Info);
}

void SystemZXPLINKAsmPrinter::emitStartOfAsmFile(Module &M) {
  emitPPA2(M);
  AsmPrinter::emitStartOfAsmFile(M);
}

void SystemZXPLINKAsmPrinter::emitPPA2(Module &M) {
  auto *ZOS = getTargetStreamer();
  OutStreamer->pushSection();
  OutStreamer->switchSection(getObjFileLowering().getTextSection());
  MCContext &OutContext = OutStreamer->getContext();
  // Make CELQSTRT symbol.
  const char *StartSymbolName = "CELQSTRT";
  MCSymbol *CELQSTRT = OutContext.getOrCreateSymbol(StartSymbolName);
  OutStreamer->emitSymbolAttribute(CELQSTRT, MCSA_OSLinkage);
  OutStreamer->emitSymbolAttribute(CELQSTRT, MCSA_Global);

  // Create symbol and assign to streamer field for use in PPA1.
  ZOS->PPA2Sym = OutContext.createTempSymbol("PPA2", false);
  MCSymbol *PPA2Sym = ZOS->PPA2Sym;
  MCSymbol *DateVersionSym = OutContext.createTempSymbol("DVS", false);

  std::time_t Time = getTranslationTime(M);
  SmallString<14> CompilationTimeEBCDIC, CompilationTime;
  CompilationTime = formatv("{0:%Y%m%d%H%M%S}", llvm::sys::toUtcTime(Time));

  uint32_t ProductVersion = getProductVersion(M),
           ProductRelease = getProductRelease(M),
           ProductPatch = getProductPatch(M);

  SmallString<6> VersionEBCDIC, Version;
  Version = formatv("{0,0-2:d}{1,0-2:d}{2,0-2:d}", ProductVersion,
                    ProductRelease, ProductPatch);

  ConverterEBCDIC::convertToEBCDIC(CompilationTime, CompilationTimeEBCDIC);
  ConverterEBCDIC::convertToEBCDIC(Version, VersionEBCDIC);

  enum class PPA2MemberId : uint8_t {
    // See z/OS Language Environment Vendor Interfaces v2r5, p.23, for
    // complete list. Only the C runtime is supported by this backend.
    LE_C_Runtime = 3,
  };
  enum class PPA2MemberSubId : uint8_t {
    // List of languages using the LE C runtime implementation.
    C = 0x00,
    CXX = 0x01,
    Swift = 0x03,
    Go = 0x60,
    LLVMBasedLang = 0xe7,
  };
  // PPA2 Flags
  enum class PPA2Flags : uint8_t {
    CompileForBinaryFloatingPoint = 0x80,
    CompiledWithXPLink = 0x01,
    CompiledUnitASCII = 0x04,
    HasServiceInfo = 0x20,
  };

  PPA2MemberSubId MemberSubId = PPA2MemberSubId::LLVMBasedLang;
  if (auto *MD = M.getModuleFlag("zos_cu_language")) {
    StringRef Language = cast<MDString>(MD)->getString();
    MemberSubId = StringSwitch<PPA2MemberSubId>(Language)
                      .Case("C", PPA2MemberSubId::C)
                      .Case("C++", PPA2MemberSubId::CXX)
                      .Case("Swift", PPA2MemberSubId::Swift)
                      .Case("Go", PPA2MemberSubId::Go)
                      .Default(PPA2MemberSubId::LLVMBasedLang);
  }

  // Emit PPA2 section.
  OutStreamer->emitLabel(PPA2Sym);
  OutStreamer->emitInt8(static_cast<uint8_t>(PPA2MemberId::LE_C_Runtime));
  OutStreamer->emitInt8(static_cast<uint8_t>(MemberSubId));
  OutStreamer->emitInt8(0x22); // Member defined, c370_plist+c370_env
  OutStreamer->emitInt8(0x04); // Control level 4 (XPLink)
  OutStreamer->emitAbsoluteSymbolDiff(CELQSTRT, PPA2Sym, 4);
  OutStreamer->emitInt32(0x00000000);
  OutStreamer->emitAbsoluteSymbolDiff(DateVersionSym, PPA2Sym, 4);
  OutStreamer->emitInt32(
      0x00000000); // Offset to main entry point, always 0 (so says TR).
  uint8_t Flgs = static_cast<uint8_t>(PPA2Flags::CompileForBinaryFloatingPoint);
  Flgs |= static_cast<uint8_t>(PPA2Flags::CompiledWithXPLink);

  bool IsASCII = true;
  if (auto *MD = M.getModuleFlag("zos_le_char_mode")) {
    const StringRef &CharMode = cast<MDString>(MD)->getString();
    if (CharMode == "ebcdic")
      IsASCII = false;
    else if (CharMode != "ascii")
      OutContext.reportError(
          {}, "Only ascii or ebcdic are allowed for zos_le_char_mode");
  }
  if (IsASCII)
    Flgs |= static_cast<uint8_t>(
        PPA2Flags::CompiledUnitASCII); // Setting bit for ASCII char. mode.

  OutStreamer->emitInt8(Flgs);
  OutStreamer->emitInt8(0x00);    // Reserved.
                                  // No MD5 signature before timestamp.
                                  // No FLOAT(AFP(VOLATILE)).
                                  // Remaining 5 flag bits reserved.
  OutStreamer->emitInt16(0x0000); // 16 Reserved flag bits.

  // Emit date and version section.
  OutStreamer->emitLabel(DateVersionSym);
  OutStreamer->emitBytes(CompilationTimeEBCDIC.str());
  OutStreamer->emitBytes(VersionEBCDIC.str());

  OutStreamer->emitInt16(0x0000); // Service level string length.

  // The binder requires that the offset to the PPA2 be emitted in a different,
  // specially-named section.
  OutStreamer->switchSection(getObjFileLowering().getPPA2ListSection());
  // Emit 8 byte alignment.
  // Emit pointer to PPA2 label.
  OutStreamer->AddComment("A(PPA2-CELQSTRT)");
  OutStreamer->emitAbsoluteSymbolDiff(PPA2Sym, CELQSTRT, 8);
  OutStreamer->popSection();
}

void SystemZXPLINKAsmPrinter::emitGlobalAlias(const Module &M,
                                               const GlobalAlias &GA) {
  if (!TM.getTargetTriple().isOSzOS())
    return AsmPrinter::emitGlobalAlias(M, GA);

  // Aliased function labels have already been emitted for z/OS
}

const MCExpr *SystemZXPLINKAsmPrinter::lowerConstant(const Constant *CV,
                                                      const Constant *BaseCV,
                                                      uint64_t Offset) {
  const GlobalAlias *GA = dyn_cast<GlobalAlias>(CV);
  const GlobalVariable *GV = dyn_cast<GlobalVariable>(CV);
  const Function *FV = dyn_cast<Function>(CV);
  bool IsFunc = !GV && (FV || (GA && isa<Function>(GA->getAliaseeObject())));

  MCSymbol *Sym = NULL;

  if (GA)
    Sym = getSymbol(GA);
  else if (IsFunc)
    Sym = getSymbol(FV);
  else if (GV)
    Sym = getSymbol(GV);

  if (IsFunc) {
    OutStreamer->emitSymbolAttribute(Sym, MCSA_ELF_TypeFunction);
    if (FV->hasExternalLinkage())
      return MCSpecifierExpr::create(MCSymbolRefExpr::create(Sym, OutContext),
                                      SystemZ::S_VCon, OutContext);
    // Trigger creation of function descriptor in ADA for internal
    // functions.
    unsigned Disp = ADATable.insert(Sym, SystemZII::MO_ADA_DIRECT_FUNC_DESC);
    return MCBinaryExpr::createAdd(
        MCSpecifierExpr::create(
            MCSymbolRefExpr::create(
                getObjFileLowering().getADASection()->getBeginSymbol(),
                OutContext),
            SystemZ::S_None, OutContext),
        MCConstantExpr::create(Disp, OutContext), OutContext);
  }
  if (Sym) {
    OutStreamer->emitSymbolAttribute(Sym, MCSA_ELF_TypeObject);
    return MCSymbolRefExpr::create(Sym, OutContext);
  }
  return AsmPrinter::lowerConstant(CV);
}

void SystemZXPLINKAsmPrinter::emitFunctionEntryLabel() {
  auto *ZOS = getTargetStreamer();
  calculatePPA1();

  // EntryPoint Marker
  const MachineFrameInfo &MFFrame = MF->getFrameInfo();
  bool IsUsingAlloca = MFFrame.hasVarSizedObjects();
  uint32_t DSASize = MFFrame.getStackSize();
  bool IsLeaf = DSASize == 0 && MFFrame.getCalleeSavedInfo().empty();

  // Set Flags.
  uint8_t Flags = 0;
  if (IsLeaf)
    Flags |= 0x08;
  if (IsUsingAlloca)
    Flags |= 0x04;

  // Combine into top 27 bits of DSASize and bottom 5 bits of Flags.
  uint32_t DSAAndFlags = DSASize & 0xFFFFFFE0; // (x/32) << 5
  DSAAndFlags |= Flags;

  // Emit entry point marker section.
  OutStreamer->AddComment("XPLINK Routine Layout Entry");
  OutStreamer->emitLabel(ZOS->DeferredPPA1.back().EPMarker);
  OutStreamer->AddComment("Eyecatcher 0x00C300C500C500");
  OutStreamer->emitIntValueInHex(0x00C300C500C500, 7); // Eyecatcher.
  OutStreamer->AddComment("Mark Type C'1'");
  OutStreamer->emitInt8(0xF1); // Mark Type.
  OutStreamer->AddComment("Offset to PPA1");
  OutStreamer->emitAbsoluteSymbolDiff(ZOS->DeferredPPA1.back().PPA1,
                                      ZOS->DeferredPPA1.back().EPMarker, 4);
  if (OutStreamer->isVerboseAsm()) {
    OutStreamer->AddComment("DSA Size 0x" + Twine::utohexstr(DSASize));
    OutStreamer->AddComment("Entry Flags");
    if (Flags & 0x08)
      OutStreamer->AddComment("  Bit 1: 1 = Leaf function");
    else
      OutStreamer->AddComment("  Bit 1: 0 = Non-leaf function");
    if (Flags & 0x04)
      OutStreamer->AddComment("  Bit 2: 1 = Uses alloca");
    else
      OutStreamer->AddComment("  Bit 2: 0 = Does not use alloca");
  }
  OutStreamer->emitInt32(DSAAndFlags);

  ZOS->emitADA(CurrentFnSym, getObjFileLowering().getADASection());

  AsmPrinter::emitFunctionEntryLabel();

  const Function *F = &MF->getFunction();
  // Emit aliasing label for function entry point label.
  for (const GlobalAlias *Alias : GOAliasMap[F]) {
    MCSymbol *Sym = getSymbol(Alias);
    OutStreamer->emitSymbolAttribute(Sym, MCSA_ELF_TypeFunction);
    emitVisibility(Sym, Alias->getVisibility());
    emitLinkage(Alias, Sym);
    OutStreamer->emitLabel(Sym);
  }
}
