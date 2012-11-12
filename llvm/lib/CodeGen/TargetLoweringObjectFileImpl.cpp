//===-- llvm/CodeGen/TargetLoweringObjectFileImpl.cpp - Object File Info --===//
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

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
using namespace llvm;
using namespace dwarf;

//===----------------------------------------------------------------------===//
//                                  ELF
//===----------------------------------------------------------------------===//

MCSymbol *
TargetLoweringObjectFileELF::getCFIPersonalitySymbol(const GlobalValue *GV,
                                                     Mangler *Mang,
                                                MachineModuleInfo *MMI) const {
  unsigned Encoding = getPersonalityEncoding();
  switch (Encoding & 0x70) {
  default:
    report_fatal_error("We do not support this DWARF encoding yet!");
  case dwarf::DW_EH_PE_absptr:
    return  Mang->getSymbol(GV);
  case dwarf::DW_EH_PE_pcrel: {
    return getContext().GetOrCreateSymbol(StringRef("DW.ref.") +
                                          Mang->getSymbol(GV)->getName());
  }
  }
}

void TargetLoweringObjectFileELF::emitPersonalityValue(MCStreamer &Streamer,
                                                       const TargetMachine &TM,
                                                       const MCSymbol *Sym) const {
  SmallString<64> NameData("DW.ref.");
  NameData += Sym->getName();
  MCSymbol *Label = getContext().GetOrCreateSymbol(NameData);
  Streamer.EmitSymbolAttribute(Label, MCSA_Hidden);
  Streamer.EmitSymbolAttribute(Label, MCSA_Weak);
  StringRef Prefix = ".data.";
  NameData.insert(NameData.begin(), Prefix.begin(), Prefix.end());
  unsigned Flags = ELF::SHF_ALLOC | ELF::SHF_WRITE | ELF::SHF_GROUP;
  const MCSection *Sec = getContext().getELFSection(NameData,
                                                    ELF::SHT_PROGBITS,
                                                    Flags,
                                                    SectionKind::getDataRel(),
                                                    0, Label->getName());
  unsigned Size = TM.getDataLayout()->getPointerSize();
  Streamer.SwitchSection(Sec);
  Streamer.EmitValueToAlignment(TM.getDataLayout()->getPointerABIAlignment());
  Streamer.EmitSymbolAttribute(Label, MCSA_ELF_TypeObject);
  const MCExpr *E = MCConstantExpr::Create(Size, getContext());
  Streamer.EmitELFSize(Label, E);
  Streamer.EmitLabel(Label);

  Streamer.EmitSymbolValue(Sym, Size);
}

static SectionKind
getELFKindForNamedSection(StringRef Name, SectionKind K) {
  // N.B.: The defaults used in here are no the same ones used in MC.
  // We follow gcc, MC follows gas. For example, given ".section .eh_frame",
  // both gas and MC will produce a section with no flags. Given
  // section(".eh_frame") gcc will produce:
  //
  //   .section   .eh_frame,"a",@progbits
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


static unsigned
getELFSectionFlags(SectionKind K) {
  unsigned Flags = 0;

  if (!K.isMetadata())
    Flags |= ELF::SHF_ALLOC;

  if (K.isText())
    Flags |= ELF::SHF_EXECINSTR;

  if (K.isWriteable())
    Flags |= ELF::SHF_WRITE;

  if (K.isThreadLocal())
    Flags |= ELF::SHF_TLS;

  // K.isMergeableConst() is left out to honour PR4650
  if (K.isMergeableCString() || K.isMergeableConst4() ||
      K.isMergeableConst8() || K.isMergeableConst16())
    Flags |= ELF::SHF_MERGE;

  if (K.isMergeableCString())
    Flags |= ELF::SHF_STRINGS;

  return Flags;
}


const MCSection *TargetLoweringObjectFileELF::
getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const {
  StringRef SectionName = GV->getSection();

  // Infer section flags from the section name if we can.
  Kind = getELFKindForNamedSection(SectionName, Kind);

  return getContext().getELFSection(SectionName,
                                    getELFSectionType(SectionName, Kind),
                                    getELFSectionFlags(Kind), Kind);
}

/// getSectionPrefixForGlobal - Return the section prefix name used by options
/// FunctionsSections and DataSections.
static const char *getSectionPrefixForGlobal(SectionKind Kind) {
  if (Kind.isText())                 return ".text.";
  if (Kind.isReadOnly())             return ".rodata.";
  if (Kind.isBSS())                  return ".bss.";

  if (Kind.isThreadData())           return ".tdata.";
  if (Kind.isThreadBSS())            return ".tbss.";

  if (Kind.isDataNoRel())            return ".data.";
  if (Kind.isDataRelLocal())         return ".data.rel.local.";
  if (Kind.isDataRel())              return ".data.rel.";
  if (Kind.isReadOnlyWithRelLocal()) return ".data.rel.ro.local.";

  assert(Kind.isReadOnlyWithRel() && "Unknown section kind");
  return ".data.rel.ro.";
}


const MCSection *TargetLoweringObjectFileELF::
SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                       Mangler *Mang, const TargetMachine &TM) const {
  // If we have -ffunction-section or -fdata-section then we should emit the
  // global value to a uniqued section specifically for it.
  bool EmitUniquedSection;
  if (Kind.isText())
    EmitUniquedSection = TM.getFunctionSections();
  else
    EmitUniquedSection = TM.getDataSections();

  // If this global is linkonce/weak and the target handles this by emitting it
  // into a 'uniqued' section name, create and return the section now.
  if ((GV->isWeakForLinker() || EmitUniquedSection) &&
      !Kind.isCommon()) {
    const char *Prefix;
    Prefix = getSectionPrefixForGlobal(Kind);

    SmallString<128> Name(Prefix, Prefix+strlen(Prefix));
    MCSymbol *Sym = Mang->getSymbol(GV);
    Name.append(Sym->getName().begin(), Sym->getName().end());
    StringRef Group = "";
    unsigned Flags = getELFSectionFlags(Kind);
    if (GV->isWeakForLinker()) {
      Group = Sym->getName();
      Flags |= ELF::SHF_GROUP;
    }

    return getContext().getELFSection(Name.str(),
                                      getELFSectionType(Name.str(), Kind),
                                      Flags, Kind, 0, Group);
  }

  if (Kind.isText()) return TextSection;

  if (Kind.isMergeable1ByteCString() ||
      Kind.isMergeable2ByteCString() ||
      Kind.isMergeable4ByteCString()) {

    // We also need alignment here.
    // FIXME: this is getting the alignment of the character, not the
    // alignment of the global!
    unsigned Align =
      TM.getDataLayout()->getPreferredAlignment(cast<GlobalVariable>(GV));

    const char *SizeSpec = ".rodata.str1.";
    if (Kind.isMergeable2ByteCString())
      SizeSpec = ".rodata.str2.";
    else if (Kind.isMergeable4ByteCString())
      SizeSpec = ".rodata.str4.";
    else
      assert(Kind.isMergeable1ByteCString() && "unknown string width");


    std::string Name = SizeSpec + utostr(Align);
    return getContext().getELFSection(Name, ELF::SHT_PROGBITS,
                                      ELF::SHF_ALLOC |
                                      ELF::SHF_MERGE |
                                      ELF::SHF_STRINGS,
                                      Kind);
  }

  if (Kind.isMergeableConst()) {
    if (Kind.isMergeableConst4() && MergeableConst4Section)
      return MergeableConst4Section;
    if (Kind.isMergeableConst8() && MergeableConst8Section)
      return MergeableConst8Section;
    if (Kind.isMergeableConst16() && MergeableConst16Section)
      return MergeableConst16Section;
    return ReadOnlySection;  // .const
  }

  if (Kind.isReadOnly())             return ReadOnlySection;

  if (Kind.isThreadData())           return TLSDataSection;
  if (Kind.isThreadBSS())            return TLSBSSSection;

  // Note: we claim that common symbols are put in BSSSection, but they are
  // really emitted with the magic .comm directive, which creates a symbol table
  // entry but not a section.
  if (Kind.isBSS() || Kind.isCommon()) return BSSSection;

  if (Kind.isDataNoRel())            return DataSection;
  if (Kind.isDataRelLocal())         return DataRelLocalSection;
  if (Kind.isDataRel())              return DataRelSection;
  if (Kind.isReadOnlyWithRelLocal()) return DataRelROLocalSection;

  assert(Kind.isReadOnlyWithRel() && "Unknown section kind");
  return DataRelROSection;
}

/// getSectionForConstant - Given a mergeable constant with the
/// specified size and relocation information, return a section that it
/// should be placed in.
const MCSection *TargetLoweringObjectFileELF::
getSectionForConstant(SectionKind Kind) const {
  if (Kind.isMergeableConst4() && MergeableConst4Section)
    return MergeableConst4Section;
  if (Kind.isMergeableConst8() && MergeableConst8Section)
    return MergeableConst8Section;
  if (Kind.isMergeableConst16() && MergeableConst16Section)
    return MergeableConst16Section;
  if (Kind.isReadOnly())
    return ReadOnlySection;

  if (Kind.isReadOnlyWithRelLocal()) return DataRelROLocalSection;
  assert(Kind.isReadOnlyWithRel() && "Unknown section kind");
  return DataRelROSection;
}

const MCExpr *TargetLoweringObjectFileELF::
getExprForDwarfGlobalReference(const GlobalValue *GV, Mangler *Mang,
                               MachineModuleInfo *MMI,
                               unsigned Encoding, MCStreamer &Streamer) const {

  if (Encoding & dwarf::DW_EH_PE_indirect) {
    MachineModuleInfoELF &ELFMMI = MMI->getObjFileInfo<MachineModuleInfoELF>();

    SmallString<128> Name;
    Mang->getNameWithPrefix(Name, GV, true);
    Name += ".DW.stub";

    // Add information about the stub reference to ELFMMI so that the stub
    // gets emitted by the asmprinter.
    MCSymbol *SSym = getContext().GetOrCreateSymbol(Name.str());
    MachineModuleInfoImpl::StubValueTy &StubSym = ELFMMI.getGVStubEntry(SSym);
    if (StubSym.getPointer() == 0) {
      MCSymbol *Sym = Mang->getSymbol(GV);
      StubSym = MachineModuleInfoImpl::StubValueTy(Sym, !GV->hasLocalLinkage());
    }

    return TargetLoweringObjectFile::
      getExprForDwarfReference(SSym, Encoding & ~dwarf::DW_EH_PE_indirect, Streamer);
  }

  return TargetLoweringObjectFile::
    getExprForDwarfGlobalReference(GV, Mang, MMI, Encoding, Streamer);
}

const MCSection *
TargetLoweringObjectFileELF::getStaticCtorSection(unsigned Priority) const {
  // The default scheme is .ctor / .dtor, so we have to invert the priority
  // numbering.
  if (Priority == 65535)
    return StaticCtorSection;

  if (UseInitArray) {
    std::string Name = std::string(".init_array.") + utostr(Priority);
    return getContext().getELFSection(Name, ELF::SHT_INIT_ARRAY,
                                      ELF::SHF_ALLOC | ELF::SHF_WRITE,
                                      SectionKind::getDataRel());
  } else {
    std::string Name = std::string(".ctors.") + utostr(65535 - Priority);
    return getContext().getELFSection(Name, ELF::SHT_PROGBITS,
                                      ELF::SHF_ALLOC |ELF::SHF_WRITE,
                                      SectionKind::getDataRel());
  }
}

const MCSection *
TargetLoweringObjectFileELF::getStaticDtorSection(unsigned Priority) const {
  // The default scheme is .ctor / .dtor, so we have to invert the priority
  // numbering.
  if (Priority == 65535)
    return StaticDtorSection;

  if (UseInitArray) {
    std::string Name = std::string(".fini_array.") + utostr(Priority);
    return getContext().getELFSection(Name, ELF::SHT_FINI_ARRAY,
                                      ELF::SHF_ALLOC | ELF::SHF_WRITE,
                                      SectionKind::getDataRel());
  } else {
    std::string Name = std::string(".dtors.") + utostr(65535 - Priority);
    return getContext().getELFSection(Name, ELF::SHT_PROGBITS,
                                      ELF::SHF_ALLOC |ELF::SHF_WRITE,
                                      SectionKind::getDataRel());
  }
}

void
TargetLoweringObjectFileELF::InitializeELF(bool UseInitArray_) {
  UseInitArray = UseInitArray_;
  if (!UseInitArray)
    return;

  StaticCtorSection =
    getContext().getELFSection(".init_array", ELF::SHT_INIT_ARRAY,
                               ELF::SHF_WRITE |
                               ELF::SHF_ALLOC,
                               SectionKind::getDataRel());
  StaticDtorSection =
    getContext().getELFSection(".fini_array", ELF::SHT_FINI_ARRAY,
                               ELF::SHF_WRITE |
                               ELF::SHF_ALLOC,
                               SectionKind::getDataRel());
}

//===----------------------------------------------------------------------===//
//                                 MachO
//===----------------------------------------------------------------------===//

/// emitModuleFlags - Emit the module flags that specify the garbage collection
/// information.
void TargetLoweringObjectFileMachO::
emitModuleFlags(MCStreamer &Streamer,
                ArrayRef<Module::ModuleFlagEntry> ModuleFlags,
                Mangler *Mang, const TargetMachine &TM) const {
  unsigned VersionVal = 0;
  unsigned ImageInfoFlags = 0;
  StringRef SectionVal;

  for (ArrayRef<Module::ModuleFlagEntry>::iterator
         i = ModuleFlags.begin(), e = ModuleFlags.end(); i != e; ++i) {
    const Module::ModuleFlagEntry &MFE = *i;

    // Ignore flags with 'Require' behavior.
    if (MFE.Behavior == Module::Require)
      continue;

    StringRef Key = MFE.Key->getString();
    Value *Val = MFE.Val;

    if (Key == "Objective-C Image Info Version")
      VersionVal = cast<ConstantInt>(Val)->getZExtValue();
    else if (Key == "Objective-C Garbage Collection" ||
             Key == "Objective-C GC Only" ||
             Key == "Objective-C Is Simulated")
      ImageInfoFlags |= cast<ConstantInt>(Val)->getZExtValue();
    else if (Key == "Objective-C Image Info Section")
      SectionVal = cast<MDString>(Val)->getString();
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
  const MCSectionMachO *S =
    getContext().getMachOSection(Segment, Section, TAA, StubSize,
                                 SectionKind::getDataNoRel());
  Streamer.SwitchSection(S);
  Streamer.EmitLabel(getContext().
                     GetOrCreateSymbol(StringRef("L_OBJC_IMAGE_INFO")));
  Streamer.EmitIntValue(VersionVal, 4);
  Streamer.EmitIntValue(ImageInfoFlags, 4);
  Streamer.AddBlankLine();
}

const MCSection *TargetLoweringObjectFileMachO::
getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const {
  // Parse the section specifier and create it if valid.
  StringRef Segment, Section;
  unsigned TAA = 0, StubSize = 0;
  bool TAAParsed;
  std::string ErrorCode =
    MCSectionMachO::ParseSectionSpecifier(GV->getSection(), Segment, Section,
                                          TAA, TAAParsed, StubSize);
  if (!ErrorCode.empty()) {
    // If invalid, report the error with report_fatal_error.
    report_fatal_error("Global variable '" + GV->getName() +
                       "' has an invalid section specifier '" +
                       GV->getSection() + "': " + ErrorCode + ".");
  }

  // Get the section.
  const MCSectionMachO *S =
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
    report_fatal_error("Global variable '" + GV->getName() +
                       "' section type or attributes does not match previous"
                       " section specifier");
  }

  return S;
}

const MCSection *TargetLoweringObjectFileMachO::
SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                       Mangler *Mang, const TargetMachine &TM) const {

  // Handle thread local data.
  if (Kind.isThreadBSS()) return TLSBSSSection;
  if (Kind.isThreadData()) return TLSDataSection;

  if (Kind.isText())
    return GV->isWeakForLinker() ? TextCoalSection : TextSection;

  // If this is weak/linkonce, put this in a coalescable section, either in text
  // or data depending on if it is writable.
  if (GV->isWeakForLinker()) {
    if (Kind.isReadOnly())
      return ConstTextCoalSection;
    return DataCoalSection;
  }

  // FIXME: Alignment check should be handled by section classifier.
  if (Kind.isMergeable1ByteCString() &&
      TM.getDataLayout()->getPreferredAlignment(cast<GlobalVariable>(GV)) < 32)
    return CStringSection;

  // Do not put 16-bit arrays in the UString section if they have an
  // externally visible label, this runs into issues with certain linker
  // versions.
  if (Kind.isMergeable2ByteCString() && !GV->hasExternalLinkage() &&
      TM.getDataLayout()->getPreferredAlignment(cast<GlobalVariable>(GV)) < 32)
    return UStringSection;

  if (Kind.isMergeableConst()) {
    if (Kind.isMergeableConst4())
      return FourByteConstantSection;
    if (Kind.isMergeableConst8())
      return EightByteConstantSection;
    if (Kind.isMergeableConst16() && SixteenByteConstantSection)
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

const MCSection *
TargetLoweringObjectFileMachO::getSectionForConstant(SectionKind Kind) const {
  // If this constant requires a relocation, we have to put it in the data
  // segment, not in the text segment.
  if (Kind.isDataRel() || Kind.isReadOnlyWithRel())
    return ConstDataSection;

  if (Kind.isMergeableConst4())
    return FourByteConstantSection;
  if (Kind.isMergeableConst8())
    return EightByteConstantSection;
  if (Kind.isMergeableConst16() && SixteenByteConstantSection)
    return SixteenByteConstantSection;
  return ReadOnlySection;  // .const
}

/// shouldEmitUsedDirectiveFor - This hook allows targets to selectively decide
/// not to emit the UsedDirective for some symbols in llvm.used.
// FIXME: REMOVE this (rdar://7071300)
bool TargetLoweringObjectFileMachO::
shouldEmitUsedDirectiveFor(const GlobalValue *GV, Mangler *Mang) const {
  /// On Darwin, internally linked data beginning with "L" or "l" does not have
  /// the directive emitted (this occurs in ObjC metadata).
  if (!GV) return false;

  // Check whether the mangled name has the "Private" or "LinkerPrivate" prefix.
  if (GV->hasLocalLinkage() && !isa<Function>(GV)) {
    // FIXME: ObjC metadata is currently emitted as internal symbols that have
    // \1L and \0l prefixes on them.  Fix them to be Private/LinkerPrivate and
    // this horrible hack can go away.
    MCSymbol *Sym = Mang->getSymbol(GV);
    if (Sym->getName()[0] == 'L' || Sym->getName()[0] == 'l')
      return false;
  }

  return true;
}

const MCExpr *TargetLoweringObjectFileMachO::
getExprForDwarfGlobalReference(const GlobalValue *GV, Mangler *Mang,
                               MachineModuleInfo *MMI, unsigned Encoding,
                               MCStreamer &Streamer) const {
  // The mach-o version of this method defaults to returning a stub reference.

  if (Encoding & DW_EH_PE_indirect) {
    MachineModuleInfoMachO &MachOMMI =
      MMI->getObjFileInfo<MachineModuleInfoMachO>();

    SmallString<128> Name;
    Mang->getNameWithPrefix(Name, GV, true);
    Name += "$non_lazy_ptr";

    // Add information about the stub reference to MachOMMI so that the stub
    // gets emitted by the asmprinter.
    MCSymbol *SSym = getContext().GetOrCreateSymbol(Name.str());
    MachineModuleInfoImpl::StubValueTy &StubSym =
      GV->hasHiddenVisibility() ? MachOMMI.getHiddenGVStubEntry(SSym) :
                                  MachOMMI.getGVStubEntry(SSym);
    if (StubSym.getPointer() == 0) {
      MCSymbol *Sym = Mang->getSymbol(GV);
      StubSym = MachineModuleInfoImpl::StubValueTy(Sym, !GV->hasLocalLinkage());
    }

    return TargetLoweringObjectFile::
      getExprForDwarfReference(SSym, Encoding & ~dwarf::DW_EH_PE_indirect, Streamer);
  }

  return TargetLoweringObjectFile::
    getExprForDwarfGlobalReference(GV, Mang, MMI, Encoding, Streamer);
}

MCSymbol *TargetLoweringObjectFileMachO::
getCFIPersonalitySymbol(const GlobalValue *GV, Mangler *Mang,
                        MachineModuleInfo *MMI) const {
  // The mach-o version of this method defaults to returning a stub reference.
  MachineModuleInfoMachO &MachOMMI =
    MMI->getObjFileInfo<MachineModuleInfoMachO>();

  SmallString<128> Name;
  Mang->getNameWithPrefix(Name, GV, true);
  Name += "$non_lazy_ptr";

  // Add information about the stub reference to MachOMMI so that the stub
  // gets emitted by the asmprinter.
  MCSymbol *SSym = getContext().GetOrCreateSymbol(Name.str());
  MachineModuleInfoImpl::StubValueTy &StubSym = MachOMMI.getGVStubEntry(SSym);
  if (StubSym.getPointer() == 0) {
    MCSymbol *Sym = Mang->getSymbol(GV);
    StubSym = MachineModuleInfoImpl::StubValueTy(Sym, !GV->hasLocalLinkage());
  }

  return SSym;
}

//===----------------------------------------------------------------------===//
//                                  COFF
//===----------------------------------------------------------------------===//

static unsigned
getCOFFSectionFlags(SectionKind K) {
  unsigned Flags = 0;

  if (K.isMetadata())
    Flags |=
      COFF::IMAGE_SCN_MEM_DISCARDABLE;
  else if (K.isText())
    Flags |=
      COFF::IMAGE_SCN_MEM_EXECUTE |
      COFF::IMAGE_SCN_MEM_READ |
      COFF::IMAGE_SCN_CNT_CODE;
  else if (K.isBSS ())
    Flags |=
      COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA |
      COFF::IMAGE_SCN_MEM_READ |
      COFF::IMAGE_SCN_MEM_WRITE;
  else if (K.isThreadLocal())
    Flags |=
      COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
      COFF::IMAGE_SCN_MEM_READ |
      COFF::IMAGE_SCN_MEM_WRITE;
  else if (K.isReadOnly())
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

const MCSection *TargetLoweringObjectFileCOFF::
getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const {
  return getContext().getCOFFSection(GV->getSection(),
                                     getCOFFSectionFlags(Kind),
                                     Kind);
}

static const char *getCOFFSectionPrefixForUniqueGlobal(SectionKind Kind) {
  if (Kind.isText())
    return ".text$";
  if (Kind.isBSS ())
    return ".bss$";
  if (Kind.isThreadLocal())
    return ".tls$";
  if (Kind.isWriteable())
    return ".data$";
  return ".rdata$";
}


const MCSection *TargetLoweringObjectFileCOFF::
SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                       Mangler *Mang, const TargetMachine &TM) const {

  // If this global is linkonce/weak and the target handles this by emitting it
  // into a 'uniqued' section name, create and return the section now.
  if (GV->isWeakForLinker()) {
    const char *Prefix = getCOFFSectionPrefixForUniqueGlobal(Kind);
    SmallString<128> Name(Prefix, Prefix+strlen(Prefix));
    MCSymbol *Sym = Mang->getSymbol(GV);
    Name.append(Sym->getName().begin() + 1, Sym->getName().end());

    unsigned Characteristics = getCOFFSectionFlags(Kind);

    Characteristics |= COFF::IMAGE_SCN_LNK_COMDAT;

    return getContext().getCOFFSection(Name.str(), Characteristics,
                          COFF::IMAGE_COMDAT_SELECT_ANY, Kind);
  }

  if (Kind.isText())
    return getTextSection();

  if (Kind.isThreadLocal())
    return getTLSDataSection();

  return getDataSection();
}

