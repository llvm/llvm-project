//===- llvm/CodeGen/TargetLoweringObjectFileImpl.cpp - Object File Info ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements classes used to handle lowerings specific to common
// object file formats.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/Comdat.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSectionWasm.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/SectionKind.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <cassert>
#include <string>

using namespace llvm;
using namespace dwarf;

//===----------------------------------------------------------------------===//
//                                  ELF
//===----------------------------------------------------------------------===//

MCSymbol *TargetLoweringObjectFileELF::getCFIPersonalitySymbol(
    const GlobalValue *GV, const TargetMachine &TM,
    MachineModuleInfo *MMI) const {
  unsigned Encoding = getPersonalityEncoding();
  if ((Encoding & 0x80) == DW_EH_PE_indirect)
    return getContext().getOrCreateSymbol(StringRef("DW.ref.") +
                                          TM.getSymbol(GV)->getName());
  if ((Encoding & 0x70) == DW_EH_PE_absptr)
    return TM.getSymbol(GV);
  report_fatal_error("We do not support this DWARF encoding yet!");
}

void TargetLoweringObjectFileELF::emitPersonalityValue(
    MCStreamer &Streamer, const DataLayout &DL, const MCSymbol *Sym) const {
  SmallString<64> NameData("DW.ref.");
  NameData += Sym->getName();
  MCSymbolELF *Label =
      cast<MCSymbolELF>(getContext().getOrCreateSymbol(NameData));
  Streamer.EmitSymbolAttribute(Label, MCSA_Hidden);
  Streamer.EmitSymbolAttribute(Label, MCSA_Weak);
  unsigned Flags = ELF::SHF_ALLOC | ELF::SHF_WRITE | ELF::SHF_GROUP;
  MCSection *Sec = getContext().getELFNamedSection(".data", Label->getName(),
                                                   ELF::SHT_PROGBITS, Flags, 0);
  unsigned Size = DL.getPointerSize();
  Streamer.SwitchSection(Sec);
  Streamer.EmitValueToAlignment(DL.getPointerABIAlignment());
  Streamer.EmitSymbolAttribute(Label, MCSA_ELF_TypeObject);
  const MCExpr *E = MCConstantExpr::create(Size, getContext());
  Streamer.emitELFSize(Label, E);
  Streamer.EmitLabel(Label);

  Streamer.EmitSymbolValue(Sym, Size);
}

const MCExpr *TargetLoweringObjectFileELF::getTTypeGlobalReference(
    const GlobalValue *GV, unsigned Encoding, const TargetMachine &TM,
    MachineModuleInfo *MMI, MCStreamer &Streamer) const {
  if (Encoding & DW_EH_PE_indirect) {
    MachineModuleInfoELF &ELFMMI = MMI->getObjFileInfo<MachineModuleInfoELF>();

    MCSymbol *SSym = getSymbolWithGlobalValueBase(GV, ".DW.stub", TM);

    // Add information about the stub reference to ELFMMI so that the stub
    // gets emitted by the asmprinter.
    MachineModuleInfoImpl::StubValueTy &StubSym = ELFMMI.getGVStubEntry(SSym);
    if (!StubSym.getPointer()) {
      MCSymbol *Sym = TM.getSymbol(GV);
      StubSym = MachineModuleInfoImpl::StubValueTy(Sym, !GV->hasLocalLinkage());
    }

    return TargetLoweringObjectFile::
      getTTypeReference(MCSymbolRefExpr::create(SSym, getContext()),
                        Encoding & ~DW_EH_PE_indirect, Streamer);
  }

  return TargetLoweringObjectFile::getTTypeGlobalReference(GV, Encoding, TM,
                                                           MMI, Streamer);
}

static SectionKind
getELFKindForNamedSection(StringRef Name, SectionKind K) {
  // N.B.: The defaults used in here are no the same ones used in MC.
  // We follow gcc, MC follows gas. For example, given ".section .eh_frame",
  // both gas and MC will produce a section with no flags. Given
  // section(".eh_frame") gcc will produce:
  //
  //   .section   .eh_frame,"a",@progbits
  
  // TODO: to support Win->ELF cross compilation with coverage properly,
  // need to pass the module pointer to the following call.
  if (Name == getInstrProfCoverageSectionName())
    return SectionKind::getMetadata();

  if (Name.empty() || Name[0] != '.') return K;

  // Some lame default implementation based on some magic section names.
  if (Name == ".bss" ||
      Name.startswith(".bss.") ||
      Name.startswith(".gnu.linkonce.b.") ||
      Name.startswith(".llvm.linkonce.b.") ||
      Name == ".sbss" ||
      Name.startswith(".sbss.") ||
      Name.startswith(".gnu.linkonce.sb.") ||
      Name.startswith(".llvm.linkonce.sb."))
    return SectionKind::getBSS();

  if (Name == ".tdata" ||
      Name.startswith(".tdata.") ||
      Name.startswith(".gnu.linkonce.td.") ||
      Name.startswith(".llvm.linkonce.td."))
    return SectionKind::getThreadData();

  if (Name == ".tbss" ||
      Name.startswith(".tbss.") ||
      Name.startswith(".gnu.linkonce.tb.") ||
      Name.startswith(".llvm.linkonce.tb."))
    return SectionKind::getThreadBSS();

  return K;
}

static unsigned getELFSectionType(StringRef Name, SectionKind K) {
  // Use SHT_NOTE for section whose name starts with ".note" to allow
  // emitting ELF notes from C variable declaration.
  // See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=77609
  if (Name.startswith(".note"))
    return ELF::SHT_NOTE;

  if (Name == ".init_array")
    return ELF::SHT_INIT_ARRAY;

  if (Name == ".fini_array")
    return ELF::SHT_FINI_ARRAY;

  if (Name == ".preinit_array")
    return ELF::SHT_PREINIT_ARRAY;

  if (K.isBSS() || K.isThreadBSS())
    return ELF::SHT_NOBITS;

  return ELF::SHT_PROGBITS;
}

static unsigned getELFSectionFlags(SectionKind K) {
  unsigned Flags = 0;

  if (!K.isMetadata())
    Flags |= ELF::SHF_ALLOC;

  if (K.isText())
    Flags |= ELF::SHF_EXECINSTR;

  if (K.isExecuteOnly())
    Flags |= ELF::SHF_ARM_PURECODE;

  if (K.isWriteable())
    Flags |= ELF::SHF_WRITE;

  if (K.isThreadLocal())
    Flags |= ELF::SHF_TLS;

  if (K.isMergeableCString() || K.isMergeableConst())
    Flags |= ELF::SHF_MERGE;

  if (K.isMergeableCString())
    Flags |= ELF::SHF_STRINGS;

  return Flags;
}

static const Comdat *getELFComdat(const GlobalValue *GV) {
  const Comdat *C = GV->getComdat();
  if (!C)
    return nullptr;

  if (C->getSelectionKind() != Comdat::Any)
    report_fatal_error("ELF COMDATs only support SelectionKind::Any, '" +
                       C->getName() + "' cannot be lowered.");

  return C;
}

static const MCSymbolELF *getAssociatedSymbol(const GlobalObject *GO,
                                              const TargetMachine &TM) {
  MDNode *MD = GO->getMetadata(LLVMContext::MD_associated);
  if (!MD)
    return nullptr;

  auto *VM = dyn_cast<ValueAsMetadata>(MD->getOperand(0));
  if (!VM)
    report_fatal_error("MD_associated operand is not ValueAsMetadata");

  GlobalObject *OtherGO = dyn_cast<GlobalObject>(VM->getValue());
  return OtherGO ? dyn_cast<MCSymbolELF>(TM.getSymbol(OtherGO)) : nullptr;
}

MCSection *TargetLoweringObjectFileELF::getExplicitSectionGlobal(
    const GlobalObject *GO, SectionKind Kind, const TargetMachine &TM) const {
  StringRef SectionName = GO->getSection();

  // Infer section flags from the section name if we can.
  Kind = getELFKindForNamedSection(SectionName, Kind);

  StringRef Group = "";
  unsigned Flags = getELFSectionFlags(Kind);
  if (const Comdat *C = getELFComdat(GO)) {
    Group = C->getName();
    Flags |= ELF::SHF_GROUP;
  }

  // A section can have at most one associated section. Put each global with
  // MD_associated in a unique section.
  unsigned UniqueID = MCContext::GenericSectionID;
  const MCSymbolELF *AssociatedSymbol = getAssociatedSymbol(GO, TM);
  if (AssociatedSymbol) {
    UniqueID = NextUniqueID++;
    Flags |= ELF::SHF_LINK_ORDER;
  }

  MCSectionELF *Section = getContext().getELFSection(
      SectionName, getELFSectionType(SectionName, Kind), Flags,
      /*EntrySize=*/0, Group, UniqueID, AssociatedSymbol);
  // Make sure that we did not get some other section with incompatible sh_link.
  // This should not be possible due to UniqueID code above.
  assert(Section->getAssociatedSymbol() == AssociatedSymbol);
  return Section;
}

/// Return the section prefix name used by options FunctionsSections and
/// DataSections.
static StringRef getSectionPrefixForGlobal(SectionKind Kind) {
  if (Kind.isText())
    return ".text";
  if (Kind.isReadOnly())
    return ".rodata";
  if (Kind.isBSS())
    return ".bss";
  if (Kind.isThreadData())
    return ".tdata";
  if (Kind.isThreadBSS())
    return ".tbss";
  if (Kind.isData())
    return ".data";
  assert(Kind.isReadOnlyWithRel() && "Unknown section kind");
  return ".data.rel.ro";
}

static MCSectionELF *selectELFSectionForGlobal(
    MCContext &Ctx, const GlobalObject *GO, SectionKind Kind, Mangler &Mang,
    const TargetMachine &TM, bool EmitUniqueSection, unsigned Flags,
    unsigned *NextUniqueID, const MCSymbolELF *AssociatedSymbol) {
  unsigned EntrySize = 0;
  if (Kind.isMergeableCString()) {
    if (Kind.isMergeable2ByteCString()) {
      EntrySize = 2;
    } else if (Kind.isMergeable4ByteCString()) {
      EntrySize = 4;
    } else {
      EntrySize = 1;
      assert(Kind.isMergeable1ByteCString() && "unknown string width");
    }
  } else if (Kind.isMergeableConst()) {
    if (Kind.isMergeableConst4()) {
      EntrySize = 4;
    } else if (Kind.isMergeableConst8()) {
      EntrySize = 8;
    } else if (Kind.isMergeableConst16()) {
      EntrySize = 16;
    } else {
      assert(Kind.isMergeableConst32() && "unknown data width");
      EntrySize = 32;
    }
  }

  StringRef Group = "";
  if (const Comdat *C = getELFComdat(GO)) {
    Flags |= ELF::SHF_GROUP;
    Group = C->getName();
  }

  bool UniqueSectionNames = TM.getUniqueSectionNames();
  SmallString<128> Name;
  if (Kind.isMergeableCString()) {
    // We also need alignment here.
    // FIXME: this is getting the alignment of the character, not the
    // alignment of the global!
    unsigned Align = GO->getParent()->getDataLayout().getPreferredAlignment(
        cast<GlobalVariable>(GO));

    std::string SizeSpec = ".rodata.str" + utostr(EntrySize) + ".";
    Name = SizeSpec + utostr(Align);
  } else if (Kind.isMergeableConst()) {
    Name = ".rodata.cst";
    Name += utostr(EntrySize);
  } else {
    Name = getSectionPrefixForGlobal(Kind);
  }

  if (const auto *F = dyn_cast<Function>(GO)) {
    const auto &OptionalPrefix = F->getSectionPrefix();
    if (OptionalPrefix)
      Name += *OptionalPrefix;
  }

  if (EmitUniqueSection && UniqueSectionNames) {
    Name.push_back('.');
    TM.getNameWithPrefix(Name, GO, Mang, true);
  }
  unsigned UniqueID = MCContext::GenericSectionID;
  if (EmitUniqueSection && !UniqueSectionNames) {
    UniqueID = *NextUniqueID;
    (*NextUniqueID)++;
  }
  // Use 0 as the unique ID for execute-only text
  if (Kind.isExecuteOnly())
    UniqueID = 0;
  return Ctx.getELFSection(Name, getELFSectionType(Name, Kind), Flags,
                           EntrySize, Group, UniqueID, AssociatedSymbol);
}

MCSection *TargetLoweringObjectFileELF::SelectSectionForGlobal(
    const GlobalObject *GO, SectionKind Kind, const TargetMachine &TM) const {
  unsigned Flags = getELFSectionFlags(Kind);

  // If we have -ffunction-section or -fdata-section then we should emit the
  // global value to a uniqued section specifically for it.
  bool EmitUniqueSection = false;
  if (!(Flags & ELF::SHF_MERGE) && !Kind.isCommon()) {
    if (Kind.isText())
      EmitUniqueSection = TM.getFunctionSections();
    else
      EmitUniqueSection = TM.getDataSections();
  }
  EmitUniqueSection |= GO->hasComdat();

  const MCSymbolELF *AssociatedSymbol = getAssociatedSymbol(GO, TM);
  if (AssociatedSymbol) {
    EmitUniqueSection = true;
    Flags |= ELF::SHF_LINK_ORDER;
  }

  MCSectionELF *Section = selectELFSectionForGlobal(
      getContext(), GO, Kind, getMangler(), TM, EmitUniqueSection, Flags,
      &NextUniqueID, AssociatedSymbol);
  assert(Section->getAssociatedSymbol() == AssociatedSymbol);
  return Section;
}

MCSection *TargetLoweringObjectFileELF::getSectionForJumpTable(
    const Function &F, const TargetMachine &TM) const {
  // If the function can be removed, produce a unique section so that
  // the table doesn't prevent the removal.
  const Comdat *C = F.getComdat();
  bool EmitUniqueSection = TM.getFunctionSections() || C;
  if (!EmitUniqueSection)
    return ReadOnlySection;

  return selectELFSectionForGlobal(getContext(), &F, SectionKind::getReadOnly(),
                                   getMangler(), TM, EmitUniqueSection,
                                   ELF::SHF_ALLOC, &NextUniqueID,
                                   /* AssociatedSymbol */ nullptr);
}

bool TargetLoweringObjectFileELF::shouldPutJumpTableInFunctionSection(
    bool UsesLabelDifference, const Function &F) const {
  // We can always create relative relocations, so use another section
  // that can be marked non-executable.
  return false;
}

/// Given a mergeable constant with the specified size and relocation
/// information, return a section that it should be placed in.
MCSection *TargetLoweringObjectFileELF::getSectionForConstant(
    const DataLayout &DL, SectionKind Kind, const Constant *C,
    unsigned &Align) const {
  if (Kind.isMergeableConst4() && MergeableConst4Section)
    return MergeableConst4Section;
  if (Kind.isMergeableConst8() && MergeableConst8Section)
    return MergeableConst8Section;
  if (Kind.isMergeableConst16() && MergeableConst16Section)
    return MergeableConst16Section;
  if (Kind.isMergeableConst32() && MergeableConst32Section)
    return MergeableConst32Section;
  if (Kind.isReadOnly())
    return ReadOnlySection;

  assert(Kind.isReadOnlyWithRel() && "Unknown section kind");
  return DataRelROSection;
}

static MCSectionELF *getStaticStructorSection(MCContext &Ctx, bool UseInitArray,
                                              bool IsCtor, unsigned Priority,
                                              const MCSymbol *KeySym) {
  std::string Name;
  unsigned Type;
  unsigned Flags = ELF::SHF_ALLOC | ELF::SHF_WRITE;
  StringRef COMDAT = KeySym ? KeySym->getName() : "";

  if (KeySym)
    Flags |= ELF::SHF_GROUP;

  if (UseInitArray) {
    if (IsCtor) {
      Type = ELF::SHT_INIT_ARRAY;
      Name = ".init_array";
    } else {
      Type = ELF::SHT_FINI_ARRAY;
      Name = ".fini_array";
    }
    if (Priority != 65535) {
      Name += '.';
      Name += utostr(Priority);
    }
  } else {
    // The default scheme is .ctor / .dtor, so we have to invert the priority
    // numbering.
    if (IsCtor)
      Name = ".ctors";
    else
      Name = ".dtors";
    if (Priority != 65535) {
      Name += '.';
      Name += utostr(65535 - Priority);
    }
    Type = ELF::SHT_PROGBITS;
  }

  return Ctx.getELFSection(Name, Type, Flags, 0, COMDAT);
}

MCSection *TargetLoweringObjectFileELF::getStaticCtorSection(
    unsigned Priority, const MCSymbol *KeySym) const {
  return getStaticStructorSection(getContext(), UseInitArray, true, Priority,
                                  KeySym);
}

MCSection *TargetLoweringObjectFileELF::getStaticDtorSection(
    unsigned Priority, const MCSymbol *KeySym) const {
  return getStaticStructorSection(getContext(), UseInitArray, false, Priority,
                                  KeySym);
}

const MCExpr *TargetLoweringObjectFileELF::lowerRelativeReference(
    const GlobalValue *LHS, const GlobalValue *RHS,
    const TargetMachine &TM) const {
  // We may only use a PLT-relative relocation to refer to unnamed_addr
  // functions.
  if (!LHS->hasGlobalUnnamedAddr() || !LHS->getValueType()->isFunctionTy())
    return nullptr;

  // Basic sanity checks.
  if (LHS->getType()->getPointerAddressSpace() != 0 ||
      RHS->getType()->getPointerAddressSpace() != 0 || LHS->isThreadLocal() ||
      RHS->isThreadLocal())
    return nullptr;

  return MCBinaryExpr::createSub(
      MCSymbolRefExpr::create(TM.getSymbol(LHS), PLTRelativeVariantKind,
                              getContext()),
      MCSymbolRefExpr::create(TM.getSymbol(RHS), getContext()), getContext());
}

void
TargetLoweringObjectFileELF::InitializeELF(bool UseInitArray_) {
  UseInitArray = UseInitArray_;
  MCContext &Ctx = getContext();
  if (!UseInitArray) {
    StaticCtorSection = Ctx.getELFSection(".ctors", ELF::SHT_PROGBITS,
                                          ELF::SHF_ALLOC | ELF::SHF_WRITE);

    StaticDtorSection = Ctx.getELFSection(".dtors", ELF::SHT_PROGBITS,
                                          ELF::SHF_ALLOC | ELF::SHF_WRITE);
    return;
  }

  StaticCtorSection = Ctx.getELFSection(".init_array", ELF::SHT_INIT_ARRAY,
                                        ELF::SHF_WRITE | ELF::SHF_ALLOC);
  StaticDtorSection = Ctx.getELFSection(".fini_array", ELF::SHT_FINI_ARRAY,
                                        ELF::SHF_WRITE | ELF::SHF_ALLOC);
}

//===----------------------------------------------------------------------===//
//                                 MachO
//===----------------------------------------------------------------------===//

TargetLoweringObjectFileMachO::TargetLoweringObjectFileMachO()
  : TargetLoweringObjectFile() {
  SupportIndirectSymViaGOTPCRel = true;
}

void TargetLoweringObjectFileMachO::Initialize(MCContext &Ctx,
                                               const TargetMachine &TM) {
  TargetLoweringObjectFile::Initialize(Ctx, TM);
  if (TM.getRelocationModel() == Reloc::Static) {
    StaticCtorSection = Ctx.getMachOSection("__TEXT", "__constructor", 0,
                                            SectionKind::getData());
    StaticDtorSection = Ctx.getMachOSection("__TEXT", "__destructor", 0,
                                            SectionKind::getData());
  } else {
    StaticCtorSection = Ctx.getMachOSection("__DATA", "__mod_init_func",
                                            MachO::S_MOD_INIT_FUNC_POINTERS,
                                            SectionKind::getData());
    StaticDtorSection = Ctx.getMachOSection("__DATA", "__mod_term_func",
                                            MachO::S_MOD_TERM_FUNC_POINTERS,
                                            SectionKind::getData());
  }
}

/// emitModuleFlags - Perform code emission for module flags.
void TargetLoweringObjectFileMachO::emitModuleFlags(
    MCStreamer &Streamer, ArrayRef<Module::ModuleFlagEntry> ModuleFlags,
    const TargetMachine &TM) const {
  unsigned VersionVal = 0;
  unsigned ImageInfoFlags = 0;
  MDNode *LinkerOptions = nullptr;
  StringRef SectionVal;

  for (const auto &MFE : ModuleFlags) {
    // Ignore flags with 'Require' behavior.
    if (MFE.Behavior == Module::Require)
      continue;

    StringRef Key = MFE.Key->getString();
    Metadata *Val = MFE.Val;

    if (Key == "Objective-C Image Info Version") {
      VersionVal = mdconst::extract<ConstantInt>(Val)->getZExtValue();
    } else if (Key == "Objective-C Garbage Collection" ||
               Key == "Objective-C GC Only" ||
               Key == "Objective-C Is Simulated" ||
               Key == "Objective-C Class Properties" ||
               Key == "Objective-C Image Swift Version") {
      ImageInfoFlags |= mdconst::extract<ConstantInt>(Val)->getZExtValue();
    } else if (Key == "Objective-C Image Info Section") {
      SectionVal = cast<MDString>(Val)->getString();
    } else if (Key == "Linker Options") {
      LinkerOptions = cast<MDNode>(Val);
    }
  }

  // Emit the linker options if present.
  if (LinkerOptions) {
    for (const auto &Option : LinkerOptions->operands()) {
      SmallVector<std::string, 4> StrOptions;
      for (const auto &Piece : cast<MDNode>(Option)->operands())
        StrOptions.push_back(cast<MDString>(Piece)->getString());
      Streamer.EmitLinkerOptions(StrOptions);
    }
  }

  // The section is mandatory. If we don't have it, then we don't have GC info.
  if (SectionVal.empty()) return;

  StringRef Segment, Section;
  unsigned TAA = 0, StubSize = 0;
  bool TAAParsed;
  std::string ErrorCode =
    MCSectionMachO::ParseSectionSpecifier(SectionVal, Segment, Section,
                                          TAA, TAAParsed, StubSize);
  if (!ErrorCode.empty())
    // If invalid, report the error with report_fatal_error.
    report_fatal_error("Invalid section specifier '" + Section + "': " +
                       ErrorCode + ".");

  // Get the section.
  MCSectionMachO *S = getContext().getMachOSection(
      Segment, Section, TAA, StubSize, SectionKind::getData());
  Streamer.SwitchSection(S);
  Streamer.EmitLabel(getContext().
                     getOrCreateSymbol(StringRef("L_OBJC_IMAGE_INFO")));
  Streamer.EmitIntValue(VersionVal, 4);
  Streamer.EmitIntValue(ImageInfoFlags, 4);
  Streamer.AddBlankLine();
}

static void checkMachOComdat(const GlobalValue *GV) {
  const Comdat *C = GV->getComdat();
  if (!C)
    return;

  report_fatal_error("MachO doesn't support COMDATs, '" + C->getName() +
                     "' cannot be lowered.");
}

MCSection *TargetLoweringObjectFileMachO::getExplicitSectionGlobal(
    const GlobalObject *GO, SectionKind Kind, const TargetMachine &TM) const {
  // Parse the section specifier and create it if valid.
  StringRef Segment, Section;
  unsigned TAA = 0, StubSize = 0;
  bool TAAParsed;

  checkMachOComdat(GO);

  std::string ErrorCode =
    MCSectionMachO::ParseSectionSpecifier(GO->getSection(), Segment, Section,
                                          TAA, TAAParsed, StubSize);
  if (!ErrorCode.empty()) {
    // If invalid, report the error with report_fatal_error.
    report_fatal_error("Global variable '" + GO->getName() +
                       "' has an invalid section specifier '" +
                       GO->getSection() + "': " + ErrorCode + ".");
  }

  // Get the section.
  MCSectionMachO *S =
      getContext().getMachOSection(Segment, Section, TAA, StubSize, Kind);

  // If TAA wasn't set by ParseSectionSpecifier() above,
  // use the value returned by getMachOSection() as a default.
  if (!TAAParsed)
    TAA = S->getTypeAndAttributes();

  // Okay, now that we got the section, verify that the TAA & StubSize agree.
  // If the user declared multiple globals with different section flags, we need
  // to reject it here.
  if (S->getTypeAndAttributes() != TAA || S->getStubSize() != StubSize) {
    // If invalid, report the error with report_fatal_error.
    report_fatal_error("Global variable '" + GO->getName() +
                       "' section type or attributes does not match previous"
                       " section specifier");
  }

  return S;
}

MCSection *TargetLoweringObjectFileMachO::SelectSectionForGlobal(
    const GlobalObject *GO, SectionKind Kind, const TargetMachine &TM) const {
  checkMachOComdat(GO);

  // Handle thread local data.
  if (Kind.isThreadBSS()) return TLSBSSSection;
  if (Kind.isThreadData()) return TLSDataSection;

  if (Kind.isText())
    return GO->isWeakForLinker() ? TextCoalSection : TextSection;

  // If this is weak/linkonce, put this in a coalescable section, either in text
  // or data depending on if it is writable.
  if (GO->isWeakForLinker()) {
    if (Kind.isReadOnly())
      return ConstTextCoalSection;
    return DataCoalSection;
  }

  // FIXME: Alignment check should be handled by section classifier.
  if (Kind.isMergeable1ByteCString() &&
      GO->getParent()->getDataLayout().getPreferredAlignment(
          cast<GlobalVariable>(GO)) < 32)
    return CStringSection;

  // Do not put 16-bit arrays in the UString section if they have an
  // externally visible label, this runs into issues with certain linker
  // versions.
  if (Kind.isMergeable2ByteCString() && !GO->hasExternalLinkage() &&
      GO->getParent()->getDataLayout().getPreferredAlignment(
          cast<GlobalVariable>(GO)) < 32)
    return UStringSection;

  // With MachO only variables whose corresponding symbol starts with 'l' or
  // 'L' can be merged, so we only try merging GVs with private linkage.
  if (GO->hasPrivateLinkage() && Kind.isMergeableConst()) {
    if (Kind.isMergeableConst4())
      return FourByteConstantSection;
    if (Kind.isMergeableConst8())
      return EightByteConstantSection;
    if (Kind.isMergeableConst16())
      return SixteenByteConstantSection;
  }

  // Otherwise, if it is readonly, but not something we can specially optimize,
  // just drop it in .const.
  if (Kind.isReadOnly())
    return ReadOnlySection;

  // If this is marked const, put it into a const section.  But if the dynamic
  // linker needs to write to it, put it in the data segment.
  if (Kind.isReadOnlyWithRel())
    return ConstDataSection;

  // Put zero initialized globals with strong external linkage in the
  // DATA, __common section with the .zerofill directive.
  if (Kind.isBSSExtern())
    return DataCommonSection;

  // Put zero initialized globals with local linkage in __DATA,__bss directive
  // with the .zerofill directive (aka .lcomm).
  if (Kind.isBSSLocal())
    return DataBSSSection;

  // Otherwise, just drop the variable in the normal data section.
  return DataSection;
}

MCSection *TargetLoweringObjectFileMachO::getSectionForConstant(
    const DataLayout &DL, SectionKind Kind, const Constant *C,
    unsigned &Align) const {
  // If this constant requires a relocation, we have to put it in the data
  // segment, not in the text segment.
  if (Kind.isData() || Kind.isReadOnlyWithRel())
    return ConstDataSection;

  if (Kind.isMergeableConst4())
    return FourByteConstantSection;
  if (Kind.isMergeableConst8())
    return EightByteConstantSection;
  if (Kind.isMergeableConst16())
    return SixteenByteConstantSection;
  return ReadOnlySection;  // .const
}

const MCExpr *TargetLoweringObjectFileMachO::getTTypeGlobalReference(
    const GlobalValue *GV, unsigned Encoding, const TargetMachine &TM,
    MachineModuleInfo *MMI, MCStreamer &Streamer) const {
  // The mach-o version of this method defaults to returning a stub reference.

  if (Encoding & DW_EH_PE_indirect) {
    MachineModuleInfoMachO &MachOMMI =
      MMI->getObjFileInfo<MachineModuleInfoMachO>();

    MCSymbol *SSym = getSymbolWithGlobalValueBase(GV, "$non_lazy_ptr", TM);

    // Add information about the stub reference to MachOMMI so that the stub
    // gets emitted by the asmprinter.
    MachineModuleInfoImpl::StubValueTy &StubSym = MachOMMI.getGVStubEntry(SSym);
    if (!StubSym.getPointer()) {
      MCSymbol *Sym = TM.getSymbol(GV);
      StubSym = MachineModuleInfoImpl::StubValueTy(Sym, !GV->hasLocalLinkage());
    }

    return TargetLoweringObjectFile::
      getTTypeReference(MCSymbolRefExpr::create(SSym, getContext()),
                        Encoding & ~DW_EH_PE_indirect, Streamer);
  }

  return TargetLoweringObjectFile::getTTypeGlobalReference(GV, Encoding, TM,
                                                           MMI, Streamer);
}

MCSymbol *TargetLoweringObjectFileMachO::getCFIPersonalitySymbol(
    const GlobalValue *GV, const TargetMachine &TM,
    MachineModuleInfo *MMI) const {
  // The mach-o version of this method defaults to returning a stub reference.
  MachineModuleInfoMachO &MachOMMI =
    MMI->getObjFileInfo<MachineModuleInfoMachO>();

  MCSymbol *SSym = getSymbolWithGlobalValueBase(GV, "$non_lazy_ptr", TM);

  // Add information about the stub reference to MachOMMI so that the stub
  // gets emitted by the asmprinter.
  MachineModuleInfoImpl::StubValueTy &StubSym = MachOMMI.getGVStubEntry(SSym);
  if (!StubSym.getPointer()) {
    MCSymbol *Sym = TM.getSymbol(GV);
    StubSym = MachineModuleInfoImpl::StubValueTy(Sym, !GV->hasLocalLinkage());
  }

  return SSym;
}

const MCExpr *TargetLoweringObjectFileMachO::getIndirectSymViaGOTPCRel(
    const MCSymbol *Sym, const MCValue &MV, int64_t Offset,
    MachineModuleInfo *MMI, MCStreamer &Streamer) const {
  // Although MachO 32-bit targets do not explicitly have a GOTPCREL relocation
  // as 64-bit do, we replace the GOT equivalent by accessing the final symbol
  // through a non_lazy_ptr stub instead. One advantage is that it allows the
  // computation of deltas to final external symbols. Example:
  //
  //    _extgotequiv:
  //       .long   _extfoo
  //
  //    _delta:
  //       .long   _extgotequiv-_delta
  //
  // is transformed to:
  //
  //    _delta:
  //       .long   L_extfoo$non_lazy_ptr-(_delta+0)
  //
  //       .section        __IMPORT,__pointers,non_lazy_symbol_pointers
  //    L_extfoo$non_lazy_ptr:
  //       .indirect_symbol        _extfoo
  //       .long   0
  //
  MachineModuleInfoMachO &MachOMMI =
    MMI->getObjFileInfo<MachineModuleInfoMachO>();
  MCContext &Ctx = getContext();

  // The offset must consider the original displacement from the base symbol
  // since 32-bit targets don't have a GOTPCREL to fold the PC displacement.
  Offset = -MV.getConstant();
  const MCSymbol *BaseSym = &MV.getSymB()->getSymbol();

  // Access the final symbol via sym$non_lazy_ptr and generate the appropriated
  // non_lazy_ptr stubs.
  SmallString<128> Name;
  StringRef Suffix = "$non_lazy_ptr";
  Name += MMI->getModule()->getDataLayout().getPrivateGlobalPrefix();
  Name += Sym->getName();
  Name += Suffix;
  MCSymbol *Stub = Ctx.getOrCreateSymbol(Name);

  MachineModuleInfoImpl::StubValueTy &StubSym = MachOMMI.getGVStubEntry(Stub);
  if (!StubSym.getPointer())
    StubSym = MachineModuleInfoImpl::
      StubValueTy(const_cast<MCSymbol *>(Sym), true /* access indirectly */);

  const MCExpr *BSymExpr =
    MCSymbolRefExpr::create(BaseSym, MCSymbolRefExpr::VK_None, Ctx);
  const MCExpr *LHS =
    MCSymbolRefExpr::create(Stub, MCSymbolRefExpr::VK_None, Ctx);

  if (!Offset)
    return MCBinaryExpr::createSub(LHS, BSymExpr, Ctx);

  const MCExpr *RHS =
    MCBinaryExpr::createAdd(BSymExpr, MCConstantExpr::create(Offset, Ctx), Ctx);
  return MCBinaryExpr::createSub(LHS, RHS, Ctx);
}

static bool canUsePrivateLabel(const MCAsmInfo &AsmInfo,
                               const MCSection &Section) {
  if (!AsmInfo.isSectionAtomizableBySymbols(Section))
    return true;

  // If it is not dead stripped, it is safe to use private labels.
  const MCSectionMachO &SMO = cast<MCSectionMachO>(Section);
  if (SMO.hasAttribute(MachO::S_ATTR_NO_DEAD_STRIP))
    return true;

  return false;
}

void TargetLoweringObjectFileMachO::getNameWithPrefix(
    SmallVectorImpl<char> &OutName, const GlobalValue *GV,
    const TargetMachine &TM) const {
  bool CannotUsePrivateLabel = true;
  if (auto *GO = GV->getBaseObject()) {
    SectionKind GOKind = TargetLoweringObjectFile::getKindForGlobal(GO, TM);
    const MCSection *TheSection = SectionForGlobal(GO, GOKind, TM);
    CannotUsePrivateLabel =
        !canUsePrivateLabel(*TM.getMCAsmInfo(), *TheSection);
  }
  getMangler().getNameWithPrefix(OutName, GV, CannotUsePrivateLabel);
}

//===----------------------------------------------------------------------===//
//                                  COFF
//===----------------------------------------------------------------------===//

static unsigned
getCOFFSectionFlags(SectionKind K, const TargetMachine &TM) {
  unsigned Flags = 0;
  bool isThumb = TM.getTargetTriple().getArch() == Triple::thumb;

  if (K.isMetadata())
    Flags |=
      COFF::IMAGE_SCN_MEM_DISCARDABLE;
  else if (K.isText())
    Flags |=
      COFF::IMAGE_SCN_MEM_EXECUTE |
      COFF::IMAGE_SCN_MEM_READ |
      COFF::IMAGE_SCN_CNT_CODE |
      (isThumb ? COFF::IMAGE_SCN_MEM_16BIT : (COFF::SectionCharacteristics)0);
  else if (K.isBSS())
    Flags |=
      COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA |
      COFF::IMAGE_SCN_MEM_READ |
      COFF::IMAGE_SCN_MEM_WRITE;
  else if (K.isThreadLocal())
    Flags |=
      COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
      COFF::IMAGE_SCN_MEM_READ |
      COFF::IMAGE_SCN_MEM_WRITE;
  else if (K.isReadOnly() || K.isReadOnlyWithRel())
    Flags |=
      COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
      COFF::IMAGE_SCN_MEM_READ;
  else if (K.isWriteable())
    Flags |=
      COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
      COFF::IMAGE_SCN_MEM_READ |
      COFF::IMAGE_SCN_MEM_WRITE;

  return Flags;
}

static const GlobalValue *getComdatGVForCOFF(const GlobalValue *GV) {
  const Comdat *C = GV->getComdat();
  assert(C && "expected GV to have a Comdat!");

  StringRef ComdatGVName = C->getName();
  const GlobalValue *ComdatGV = GV->getParent()->getNamedValue(ComdatGVName);
  if (!ComdatGV)
    report_fatal_error("Associative COMDAT symbol '" + ComdatGVName +
                       "' does not exist.");

  if (ComdatGV->getComdat() != C)
    report_fatal_error("Associative COMDAT symbol '" + ComdatGVName +
                       "' is not a key for its COMDAT.");

  return ComdatGV;
}

static int getSelectionForCOFF(const GlobalValue *GV) {
  if (const Comdat *C = GV->getComdat()) {
    const GlobalValue *ComdatKey = getComdatGVForCOFF(GV);
    if (const auto *GA = dyn_cast<GlobalAlias>(ComdatKey))
      ComdatKey = GA->getBaseObject();
    if (ComdatKey == GV) {
      switch (C->getSelectionKind()) {
      case Comdat::Any:
        return COFF::IMAGE_COMDAT_SELECT_ANY;
      case Comdat::ExactMatch:
        return COFF::IMAGE_COMDAT_SELECT_EXACT_MATCH;
      case Comdat::Largest:
        return COFF::IMAGE_COMDAT_SELECT_LARGEST;
      case Comdat::NoDuplicates:
        return COFF::IMAGE_COMDAT_SELECT_NODUPLICATES;
      case Comdat::SameSize:
        return COFF::IMAGE_COMDAT_SELECT_SAME_SIZE;
      }
    } else {
      return COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE;
    }
  }
  return 0;
}

MCSection *TargetLoweringObjectFileCOFF::getExplicitSectionGlobal(
    const GlobalObject *GO, SectionKind Kind, const TargetMachine &TM) const {
  int Selection = 0;
  unsigned Characteristics = getCOFFSectionFlags(Kind, TM);
  StringRef Name = GO->getSection();
  StringRef COMDATSymName = "";
  if (GO->hasComdat()) {
    Selection = getSelectionForCOFF(GO);
    const GlobalValue *ComdatGV;
    if (Selection == COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE)
      ComdatGV = getComdatGVForCOFF(GO);
    else
      ComdatGV = GO;

    if (!ComdatGV->hasPrivateLinkage()) {
      MCSymbol *Sym = TM.getSymbol(ComdatGV);
      COMDATSymName = Sym->getName();
      Characteristics |= COFF::IMAGE_SCN_LNK_COMDAT;
    } else {
      Selection = 0;
    }
  }

  return getContext().getCOFFSection(Name, Characteristics, Kind, COMDATSymName,
                                     Selection);
}

static const char *getCOFFSectionNameForUniqueGlobal(SectionKind Kind) {
  if (Kind.isText())
    return ".text";
  if (Kind.isBSS())
    return ".bss";
  if (Kind.isThreadLocal())
    return ".tls$";
  if (Kind.isReadOnly() || Kind.isReadOnlyWithRel())
    return ".rdata";
  return ".data";
}

MCSection *TargetLoweringObjectFileCOFF::SelectSectionForGlobal(
    const GlobalObject *GO, SectionKind Kind, const TargetMachine &TM) const {
  // If we have -ffunction-sections then we should emit the global value to a
  // uniqued section specifically for it.
  bool EmitUniquedSection;
  if (Kind.isText())
    EmitUniquedSection = TM.getFunctionSections();
  else
    EmitUniquedSection = TM.getDataSections();

  if ((EmitUniquedSection && !Kind.isCommon()) || GO->hasComdat()) {
    const char *Name = getCOFFSectionNameForUniqueGlobal(Kind);
    unsigned Characteristics = getCOFFSectionFlags(Kind, TM);

    Characteristics |= COFF::IMAGE_SCN_LNK_COMDAT;
    int Selection = getSelectionForCOFF(GO);
    if (!Selection)
      Selection = COFF::IMAGE_COMDAT_SELECT_NODUPLICATES;
    const GlobalValue *ComdatGV;
    if (GO->hasComdat())
      ComdatGV = getComdatGVForCOFF(GO);
    else
      ComdatGV = GO;

    unsigned UniqueID = MCContext::GenericSectionID;
    if (EmitUniquedSection)
      UniqueID = NextUniqueID++;

    if (!ComdatGV->hasPrivateLinkage()) {
      MCSymbol *Sym = TM.getSymbol(ComdatGV);
      StringRef COMDATSymName = Sym->getName();
      return getContext().getCOFFSection(Name, Characteristics, Kind,
                                         COMDATSymName, Selection, UniqueID);
    } else {
      SmallString<256> TmpData;
      getMangler().getNameWithPrefix(TmpData, GO, /*CannotUsePrivateLabel=*/true);
      return getContext().getCOFFSection(Name, Characteristics, Kind, TmpData,
                                         Selection, UniqueID);
    }
  }

  if (Kind.isText())
    return TextSection;

  if (Kind.isThreadLocal())
    return TLSDataSection;

  if (Kind.isReadOnly() || Kind.isReadOnlyWithRel())
    return ReadOnlySection;

  // Note: we claim that common symbols are put in BSSSection, but they are
  // really emitted with the magic .comm directive, which creates a symbol table
  // entry but not a section.
  if (Kind.isBSS() || Kind.isCommon())
    return BSSSection;

  return DataSection;
}

void TargetLoweringObjectFileCOFF::getNameWithPrefix(
    SmallVectorImpl<char> &OutName, const GlobalValue *GV,
    const TargetMachine &TM) const {
  bool CannotUsePrivateLabel = false;
  if (GV->hasPrivateLinkage() &&
      ((isa<Function>(GV) && TM.getFunctionSections()) ||
       (isa<GlobalVariable>(GV) && TM.getDataSections())))
    CannotUsePrivateLabel = true;

  getMangler().getNameWithPrefix(OutName, GV, CannotUsePrivateLabel);
}

MCSection *TargetLoweringObjectFileCOFF::getSectionForJumpTable(
    const Function &F, const TargetMachine &TM) const {
  // If the function can be removed, produce a unique section so that
  // the table doesn't prevent the removal.
  const Comdat *C = F.getComdat();
  bool EmitUniqueSection = TM.getFunctionSections() || C;
  if (!EmitUniqueSection)
    return ReadOnlySection;

  // FIXME: we should produce a symbol for F instead.
  if (F.hasPrivateLinkage())
    return ReadOnlySection;

  MCSymbol *Sym = TM.getSymbol(&F);
  StringRef COMDATSymName = Sym->getName();

  SectionKind Kind = SectionKind::getReadOnly();
  const char *Name = getCOFFSectionNameForUniqueGlobal(Kind);
  unsigned Characteristics = getCOFFSectionFlags(Kind, TM);
  Characteristics |= COFF::IMAGE_SCN_LNK_COMDAT;
  unsigned UniqueID = NextUniqueID++;

  return getContext().getCOFFSection(Name, Characteristics, Kind, COMDATSymName,
                                     COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE, UniqueID);
}

void TargetLoweringObjectFileCOFF::emitModuleFlags(
    MCStreamer &Streamer, ArrayRef<Module::ModuleFlagEntry> ModuleFlags,
    const TargetMachine &TM) const {
  MDNode *LinkerOptions = nullptr;

  for (const auto &MFE : ModuleFlags) {
    StringRef Key = MFE.Key->getString();
    if (Key == "Linker Options")
      LinkerOptions = cast<MDNode>(MFE.Val);
  }

  if (LinkerOptions) {
    // Emit the linker options to the linker .drectve section.  According to the
    // spec, this section is a space-separated string containing flags for
    // linker.
    MCSection *Sec = getDrectveSection();
    Streamer.SwitchSection(Sec);
    for (const auto &Option : LinkerOptions->operands()) {
      for (const auto &Piece : cast<MDNode>(Option)->operands()) {
        // Lead with a space for consistency with our dllexport implementation.
        std::string Directive(" ");
        Directive.append(cast<MDString>(Piece)->getString());
        Streamer.EmitBytes(Directive);
      }
    }
  }
}

void TargetLoweringObjectFileCOFF::Initialize(MCContext &Ctx,
                                              const TargetMachine &TM) {
  TargetLoweringObjectFile::Initialize(Ctx, TM);
  const Triple &T = TM.getTargetTriple();
  if (T.isKnownWindowsMSVCEnvironment() || T.isWindowsItaniumEnvironment()) {
    StaticCtorSection =
        Ctx.getCOFFSection(".CRT$XCU", COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                           COFF::IMAGE_SCN_MEM_READ,
                           SectionKind::getReadOnly());
    StaticDtorSection =
        Ctx.getCOFFSection(".CRT$XTX", COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                           COFF::IMAGE_SCN_MEM_READ,
                           SectionKind::getReadOnly());
  } else {
    StaticCtorSection = Ctx.getCOFFSection(
        ".ctors", COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                      COFF::IMAGE_SCN_MEM_READ | COFF::IMAGE_SCN_MEM_WRITE,
        SectionKind::getData());
    StaticDtorSection = Ctx.getCOFFSection(
        ".dtors", COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                      COFF::IMAGE_SCN_MEM_READ | COFF::IMAGE_SCN_MEM_WRITE,
        SectionKind::getData());
  }
}

MCSection *TargetLoweringObjectFileCOFF::getStaticCtorSection(
    unsigned Priority, const MCSymbol *KeySym) const {
  return getContext().getAssociativeCOFFSection(
      cast<MCSectionCOFF>(StaticCtorSection), KeySym, 0);
}

MCSection *TargetLoweringObjectFileCOFF::getStaticDtorSection(
    unsigned Priority, const MCSymbol *KeySym) const {
  return getContext().getAssociativeCOFFSection(
      cast<MCSectionCOFF>(StaticDtorSection), KeySym, 0);
}

void TargetLoweringObjectFileCOFF::emitLinkerFlagsForGlobal(
    raw_ostream &OS, const GlobalValue *GV) const {
  emitLinkerFlagsForGlobalCOFF(OS, GV, getTargetTriple(), getMangler());
}

//===----------------------------------------------------------------------===//
//                                  Wasm
//===----------------------------------------------------------------------===//

static const Comdat *getWasmComdat(const GlobalValue *GV) {
  const Comdat *C = GV->getComdat();
  if (!C)
    return nullptr;

  if (C->getSelectionKind() != Comdat::Any)
    report_fatal_error("Wasm COMDATs only support SelectionKind::Any, '" +
                       C->getName() + "' cannot be lowered.");

  return C;
}

MCSection *TargetLoweringObjectFileWasm::getExplicitSectionGlobal(
    const GlobalObject *GO, SectionKind Kind, const TargetMachine &TM) const {
  llvm_unreachable("getExplicitSectionGlobal not yet implemented");
  return nullptr;
}

static MCSectionWasm *
selectWasmSectionForGlobal(MCContext &Ctx, const GlobalObject *GO,
                           SectionKind Kind, Mangler &Mang,
                           const TargetMachine &TM, bool EmitUniqueSection,
                           unsigned Flags, unsigned *NextUniqueID) {
  StringRef Group = "";
  if (getWasmComdat(GO))
    llvm_unreachable("comdat not yet supported for wasm");

  bool UniqueSectionNames = TM.getUniqueSectionNames();
  SmallString<128> Name = getSectionPrefixForGlobal(Kind);

  if (const auto *F = dyn_cast<Function>(GO)) {
    const auto &OptionalPrefix = F->getSectionPrefix();
    if (OptionalPrefix)
      Name += *OptionalPrefix;
  }

  if (EmitUniqueSection && UniqueSectionNames) {
    Name.push_back('.');
    TM.getNameWithPrefix(Name, GO, Mang, true);
  }
  unsigned UniqueID = MCContext::GenericSectionID;
  if (EmitUniqueSection && !UniqueSectionNames) {
    UniqueID = *NextUniqueID;
    (*NextUniqueID)++;
  }
  return Ctx.getWasmSection(Name, /*Type=*/0, Flags,
                            Group, UniqueID);
}

MCSection *TargetLoweringObjectFileWasm::SelectSectionForGlobal(
    const GlobalObject *GO, SectionKind Kind, const TargetMachine &TM) const {

  if (Kind.isCommon())
    report_fatal_error("mergable sections not supported yet on wasm");

  // If we have -ffunction-section or -fdata-section then we should emit the
  // global value to a uniqued section specifically for it.
  bool EmitUniqueSection = false;
  if (Kind.isText())
    EmitUniqueSection = TM.getFunctionSections();
  else
    EmitUniqueSection = TM.getDataSections();
  EmitUniqueSection |= GO->hasComdat();

  return selectWasmSectionForGlobal(getContext(), GO, Kind, getMangler(), TM,
                                    EmitUniqueSection, /*Flags=*/0,
                                    &NextUniqueID);
}

bool TargetLoweringObjectFileWasm::shouldPutJumpTableInFunctionSection(
    bool UsesLabelDifference, const Function &F) const {
  // We can always create relative relocations, so use another section
  // that can be marked non-executable.
  return false;
}

const MCExpr *TargetLoweringObjectFileWasm::lowerRelativeReference(
    const GlobalValue *LHS, const GlobalValue *RHS,
    const TargetMachine &TM) const {
  // We may only use a PLT-relative relocation to refer to unnamed_addr
  // functions.
  if (!LHS->hasGlobalUnnamedAddr() || !LHS->getValueType()->isFunctionTy())
    return nullptr;

  // Basic sanity checks.
  if (LHS->getType()->getPointerAddressSpace() != 0 ||
      RHS->getType()->getPointerAddressSpace() != 0 || LHS->isThreadLocal() ||
      RHS->isThreadLocal())
    return nullptr;

  return MCBinaryExpr::createSub(
      MCSymbolRefExpr::create(TM.getSymbol(LHS), MCSymbolRefExpr::VK_None,
                              getContext()),
      MCSymbolRefExpr::create(TM.getSymbol(RHS), getContext()), getContext());
}

void
TargetLoweringObjectFileWasm::InitializeWasm() {
  // TODO: Initialize StaticCtorSection and StaticDtorSection.
}
