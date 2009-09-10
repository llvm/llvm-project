//===-- llvm/Target/TargetLoweringObjectFile.cpp - Object File Info -------===//
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

#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Mangler.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//                              Generic Code
//===----------------------------------------------------------------------===//

TargetLoweringObjectFile::TargetLoweringObjectFile() : Ctx(0) {
  TextSection = 0;
  DataSection = 0;
  BSSSection = 0;
  ReadOnlySection = 0;
  StaticCtorSection = 0;
  StaticDtorSection = 0;
  LSDASection = 0;
  EHFrameSection = 0;

  DwarfAbbrevSection = 0;
  DwarfInfoSection = 0;
  DwarfLineSection = 0;
  DwarfFrameSection = 0;
  DwarfPubNamesSection = 0;
  DwarfPubTypesSection = 0;
  DwarfDebugInlineSection = 0;
  DwarfStrSection = 0;
  DwarfLocSection = 0;
  DwarfARangesSection = 0;
  DwarfRangesSection = 0;
  DwarfMacroInfoSection = 0;
}

TargetLoweringObjectFile::~TargetLoweringObjectFile() {
}

static bool isSuitableForBSS(const GlobalVariable *GV) {
  Constant *C = GV->getInitializer();
  
  // Must have zero initializer.
  if (!C->isNullValue())
    return false;
  
  // Leave constant zeros in readonly constant sections, so they can be shared.
  if (GV->isConstant())
    return false;
  
  // If the global has an explicit section specified, don't put it in BSS.
  if (!GV->getSection().empty())
    return false;
  
  // If -nozero-initialized-in-bss is specified, don't ever use BSS.
  if (NoZerosInBSS)
    return false;
  
  // Otherwise, put it in BSS!
  return true;
}

/// IsNullTerminatedString - Return true if the specified constant (which is
/// known to have a type that is an array of 1/2/4 byte elements) ends with a
/// nul value and contains no other nuls in it.
static bool IsNullTerminatedString(const Constant *C) {
  const ArrayType *ATy = cast<ArrayType>(C->getType());
  
  // First check: is we have constant array of i8 terminated with zero
  if (const ConstantArray *CVA = dyn_cast<ConstantArray>(C)) {
    if (ATy->getNumElements() == 0) return false;

    ConstantInt *Null =
      dyn_cast<ConstantInt>(CVA->getOperand(ATy->getNumElements()-1));
    if (Null == 0 || Null->getZExtValue() != 0)
      return false; // Not null terminated.
    
    // Verify that the null doesn't occur anywhere else in the string.
    for (unsigned i = 0, e = ATy->getNumElements()-1; i != e; ++i)
      // Reject constantexpr elements etc.
      if (!isa<ConstantInt>(CVA->getOperand(i)) ||
          CVA->getOperand(i) == Null)
        return false;
    return true;
  }

  // Another possibility: [1 x i8] zeroinitializer
  if (isa<ConstantAggregateZero>(C))
    return ATy->getNumElements() == 1;

  return false;
}

/// getKindForGlobal - This is a top-level target-independent classifier for
/// a global variable.  Given an global variable and information from TM, it
/// classifies the global in a variety of ways that make various target
/// implementations simpler.  The target implementation is free to ignore this
/// extra info of course.
SectionKind TargetLoweringObjectFile::getKindForGlobal(const GlobalValue *GV,
                                                       const TargetMachine &TM){
  assert(!GV->isDeclaration() && !GV->hasAvailableExternallyLinkage() &&
         "Can only be used for global definitions");
  
  Reloc::Model ReloModel = TM.getRelocationModel();
  
  // Early exit - functions should be always in text sections.
  const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);
  if (GVar == 0)
    return SectionKind::getText();
  
  // Handle thread-local data first.
  if (GVar->isThreadLocal()) {
    if (isSuitableForBSS(GVar))
      return SectionKind::getThreadBSS();
    return SectionKind::getThreadData();
  }

  // Variable can be easily put to BSS section.
  if (isSuitableForBSS(GVar))
    return SectionKind::getBSS();

  Constant *C = GVar->getInitializer();
  
  // If the global is marked constant, we can put it into a mergable section,
  // a mergable string section, or general .data if it contains relocations.
  if (GVar->isConstant()) {
    // If the initializer for the global contains something that requires a
    // relocation, then we may have to drop this into a wriable data section
    // even though it is marked const.
    switch (C->getRelocationInfo()) {
    default: llvm_unreachable("unknown relocation info kind");
    case Constant::NoRelocation:
      // If initializer is a null-terminated string, put it in a "cstring"
      // section of the right width.
      if (const ArrayType *ATy = dyn_cast<ArrayType>(C->getType())) {
        if (const IntegerType *ITy = 
              dyn_cast<IntegerType>(ATy->getElementType())) {
          if ((ITy->getBitWidth() == 8 || ITy->getBitWidth() == 16 ||
               ITy->getBitWidth() == 32) &&
              IsNullTerminatedString(C)) {
            if (ITy->getBitWidth() == 8)
              return SectionKind::getMergeable1ByteCString();
            if (ITy->getBitWidth() == 16)
              return SectionKind::getMergeable2ByteCString();
                                         
            assert(ITy->getBitWidth() == 32 && "Unknown width");
            return SectionKind::getMergeable4ByteCString();
          }
        }
      }
        
      // Otherwise, just drop it into a mergable constant section.  If we have
      // a section for this size, use it, otherwise use the arbitrary sized
      // mergable section.
      switch (TM.getTargetData()->getTypeAllocSize(C->getType())) {
      case 4:  return SectionKind::getMergeableConst4();
      case 8:  return SectionKind::getMergeableConst8();
      case 16: return SectionKind::getMergeableConst16();
      default: return SectionKind::getMergeableConst();
      }
      
    case Constant::LocalRelocation:
      // In static relocation model, the linker will resolve all addresses, so
      // the relocation entries will actually be constants by the time the app
      // starts up.  However, we can't put this into a mergable section, because
      // the linker doesn't take relocations into consideration when it tries to
      // merge entries in the section.
      if (ReloModel == Reloc::Static)
        return SectionKind::getReadOnly();
              
      // Otherwise, the dynamic linker needs to fix it up, put it in the
      // writable data.rel.local section.
      return SectionKind::getReadOnlyWithRelLocal();
              
    case Constant::GlobalRelocations:
      // In static relocation model, the linker will resolve all addresses, so
      // the relocation entries will actually be constants by the time the app
      // starts up.  However, we can't put this into a mergable section, because
      // the linker doesn't take relocations into consideration when it tries to
      // merge entries in the section.
      if (ReloModel == Reloc::Static)
        return SectionKind::getReadOnly();
      
      // Otherwise, the dynamic linker needs to fix it up, put it in the
      // writable data.rel section.
      return SectionKind::getReadOnlyWithRel();
    }
  }

  // Okay, this isn't a constant.  If the initializer for the global is going
  // to require a runtime relocation by the dynamic linker, put it into a more
  // specific section to improve startup time of the app.  This coalesces these
  // globals together onto fewer pages, improving the locality of the dynamic
  // linker.
  if (ReloModel == Reloc::Static)
    return SectionKind::getDataNoRel();

  switch (C->getRelocationInfo()) {
  default: llvm_unreachable("unknown relocation info kind");
  case Constant::NoRelocation:
    return SectionKind::getDataNoRel();
  case Constant::LocalRelocation:
    return SectionKind::getDataRelLocal();
  case Constant::GlobalRelocations:
    return SectionKind::getDataRel();
  }
}

/// SectionForGlobal - This method computes the appropriate section to emit
/// the specified global variable or function definition.  This should not
/// be passed external (or available externally) globals.
const MCSection *TargetLoweringObjectFile::
SectionForGlobal(const GlobalValue *GV, SectionKind Kind, Mangler *Mang,
                 const TargetMachine &TM) const {
  // Select section name.
  if (GV->hasSection())
    return getExplicitSectionGlobal(GV, Kind, Mang, TM);
  
  
  // Use default section depending on the 'type' of global
  return SelectSectionForGlobal(GV, Kind, Mang, TM);
}


// Lame default implementation. Calculate the section name for global.
const MCSection *
TargetLoweringObjectFile::SelectSectionForGlobal(const GlobalValue *GV,
                                                 SectionKind Kind,
                                                 Mangler *Mang,
                                                 const TargetMachine &TM) const{
  assert(!Kind.isThreadLocal() && "Doesn't support TLS");
  
  if (Kind.isText())
    return getTextSection();
  
  if (Kind.isBSS() && BSSSection != 0)
    return BSSSection;
  
  if (Kind.isReadOnly() && ReadOnlySection != 0)
    return ReadOnlySection;

  return getDataSection();
}

/// getSectionForConstant - Given a mergable constant with the
/// specified size and relocation information, return a section that it
/// should be placed in.
const MCSection *
TargetLoweringObjectFile::getSectionForConstant(SectionKind Kind) const {
  if (Kind.isReadOnly() && ReadOnlySection != 0)
    return ReadOnlySection;
  
  return DataSection;
}



//===----------------------------------------------------------------------===//
//                                  ELF
//===----------------------------------------------------------------------===//
typedef StringMap<const MCSectionELF*> ELFUniqueMapTy;

TargetLoweringObjectFileELF::~TargetLoweringObjectFileELF() {
  // If we have the section uniquing map, free it.
  delete (ELFUniqueMapTy*)UniquingMap;
}

const MCSection *TargetLoweringObjectFileELF::
getELFSection(StringRef Section, unsigned Type, unsigned Flags,
              SectionKind Kind, bool IsExplicit) const {
  if (UniquingMap == 0)
    UniquingMap = new ELFUniqueMapTy();
  ELFUniqueMapTy &Map = *(ELFUniqueMapTy*)UniquingMap;
  
  // Do the lookup, if we have a hit, return it.
  const MCSectionELF *&Entry = Map[Section];
  if (Entry) return Entry;
  
  return Entry = MCSectionELF::Create(Section, Type, Flags, Kind, IsExplicit,
                                      getContext());
}

void TargetLoweringObjectFileELF::Initialize(MCContext &Ctx,
                                             const TargetMachine &TM) {
  if (UniquingMap != 0)
    ((ELFUniqueMapTy*)UniquingMap)->clear();
  TargetLoweringObjectFile::Initialize(Ctx, TM);

  BSSSection = 
    getELFSection(".bss", MCSectionELF::SHT_NOBITS,
                  MCSectionELF::SHF_WRITE | MCSectionELF::SHF_ALLOC,
                  SectionKind::getBSS());
    
  TextSection = 
    getELFSection(".text", MCSectionELF::SHT_PROGBITS,
                  MCSectionELF::SHF_EXECINSTR | MCSectionELF::SHF_ALLOC,
                  SectionKind::getText());

  DataSection = 
    getELFSection(".data", MCSectionELF::SHT_PROGBITS,
                  MCSectionELF::SHF_WRITE | MCSectionELF::SHF_ALLOC,
                  SectionKind::getDataRel());

  ReadOnlySection = 
    getELFSection(".rodata", MCSectionELF::SHT_PROGBITS, 
                  MCSectionELF::SHF_ALLOC, 
                  SectionKind::getReadOnly());

  TLSDataSection = 
    getELFSection(".tdata", MCSectionELF::SHT_PROGBITS,
                  MCSectionELF::SHF_ALLOC | MCSectionELF::SHF_TLS | 
                  MCSectionELF::SHF_WRITE, SectionKind::getThreadData());
  
  TLSBSSSection = 
    getELFSection(".tbss", MCSectionELF::SHT_NOBITS,
                  MCSectionELF::SHF_ALLOC | MCSectionELF::SHF_TLS |
                  MCSectionELF::SHF_WRITE, SectionKind::getThreadBSS());

  DataRelSection = 
    getELFSection(".data.rel", MCSectionELF::SHT_PROGBITS,
                  MCSectionELF::SHF_ALLOC | MCSectionELF::SHF_WRITE,
                  SectionKind::getDataRel());

  DataRelLocalSection = 
    getELFSection(".data.rel.local", MCSectionELF::SHT_PROGBITS,
                  MCSectionELF::SHF_ALLOC | MCSectionELF::SHF_WRITE,
                  SectionKind::getDataRelLocal());

  DataRelROSection = 
    getELFSection(".data.rel.ro", MCSectionELF::SHT_PROGBITS,
                  MCSectionELF::SHF_ALLOC | MCSectionELF::SHF_WRITE,
                  SectionKind::getReadOnlyWithRel());

  DataRelROLocalSection = 
    getELFSection(".data.rel.ro.local", MCSectionELF::SHT_PROGBITS,
                  MCSectionELF::SHF_ALLOC | MCSectionELF::SHF_WRITE,
                  SectionKind::getReadOnlyWithRelLocal());
    
  MergeableConst4Section = 
    getELFSection(".rodata.cst4", MCSectionELF::SHT_PROGBITS,
                  MCSectionELF::SHF_ALLOC | MCSectionELF::SHF_MERGE,
                  SectionKind::getMergeableConst4());

  MergeableConst8Section = 
    getELFSection(".rodata.cst8", MCSectionELF::SHT_PROGBITS,
                  MCSectionELF::SHF_ALLOC | MCSectionELF::SHF_MERGE,
                  SectionKind::getMergeableConst8());

  MergeableConst16Section = 
    getELFSection(".rodata.cst16", MCSectionELF::SHT_PROGBITS,
                  MCSectionELF::SHF_ALLOC | MCSectionELF::SHF_MERGE,
                  SectionKind::getMergeableConst16());

  StaticCtorSection =
    getELFSection(".ctors", MCSectionELF::SHT_PROGBITS, 
                  MCSectionELF::SHF_ALLOC | MCSectionELF::SHF_WRITE,
                  SectionKind::getDataRel());

  StaticDtorSection =
    getELFSection(".dtors", MCSectionELF::SHT_PROGBITS,
                  MCSectionELF::SHF_ALLOC | MCSectionELF::SHF_WRITE,
                  SectionKind::getDataRel());
  
  // Exception Handling Sections.
  
  // FIXME: We're emitting LSDA info into a readonly section on ELF, even though
  // it contains relocatable pointers.  In PIC mode, this is probably a big
  // runtime hit for C++ apps.  Either the contents of the LSDA need to be
  // adjusted or this should be a data section.
  LSDASection =
    getELFSection(".gcc_except_table", MCSectionELF::SHT_PROGBITS,
                  MCSectionELF::SHF_ALLOC, SectionKind::getReadOnly());
  EHFrameSection =
    getELFSection(".eh_frame", MCSectionELF::SHT_PROGBITS, 
                  MCSectionELF::SHF_ALLOC | MCSectionELF::SHF_WRITE,
                  SectionKind::getDataRel());
  
  // Debug Info Sections.
  DwarfAbbrevSection = 
    getELFSection(".debug_abbrev", MCSectionELF::SHT_PROGBITS, 0, 
                  SectionKind::getMetadata());
  DwarfInfoSection = 
    getELFSection(".debug_info", MCSectionELF::SHT_PROGBITS, 0, 
                  SectionKind::getMetadata());
  DwarfLineSection = 
    getELFSection(".debug_line", MCSectionELF::SHT_PROGBITS, 0, 
                  SectionKind::getMetadata());
  DwarfFrameSection = 
    getELFSection(".debug_frame", MCSectionELF::SHT_PROGBITS, 0, 
                  SectionKind::getMetadata());
  DwarfPubNamesSection = 
    getELFSection(".debug_pubnames", MCSectionELF::SHT_PROGBITS, 0, 
                  SectionKind::getMetadata());
  DwarfPubTypesSection = 
    getELFSection(".debug_pubtypes", MCSectionELF::SHT_PROGBITS, 0, 
                  SectionKind::getMetadata());
  DwarfStrSection = 
    getELFSection(".debug_str", MCSectionELF::SHT_PROGBITS, 0,
                  SectionKind::getMetadata());
  DwarfLocSection = 
    getELFSection(".debug_loc", MCSectionELF::SHT_PROGBITS, 0, 
                  SectionKind::getMetadata());
  DwarfARangesSection = 
    getELFSection(".debug_aranges", MCSectionELF::SHT_PROGBITS, 0, 
                  SectionKind::getMetadata());
  DwarfRangesSection = 
    getELFSection(".debug_ranges", MCSectionELF::SHT_PROGBITS, 0, 
                  SectionKind::getMetadata());
  DwarfMacroInfoSection = 
    getELFSection(".debug_macinfo", MCSectionELF::SHT_PROGBITS, 0, 
                  SectionKind::getMetadata());
}


static SectionKind 
getELFKindForNamedSection(const char *Name, SectionKind K) {
  if (Name[0] != '.') return K;
  
  // Some lame default implementation based on some magic section names.
  if (strcmp(Name, ".bss") == 0 ||
      strncmp(Name, ".bss.", 5) == 0 ||
      strncmp(Name, ".gnu.linkonce.b.", 16) == 0 ||
      strncmp(Name, ".llvm.linkonce.b.", 17) == 0 ||
      strcmp(Name, ".sbss") == 0 ||
      strncmp(Name, ".sbss.", 6) == 0 ||
      strncmp(Name, ".gnu.linkonce.sb.", 17) == 0 ||
      strncmp(Name, ".llvm.linkonce.sb.", 18) == 0)
    return SectionKind::getBSS();
  
  if (strcmp(Name, ".tdata") == 0 ||
      strncmp(Name, ".tdata.", 7) == 0 ||
      strncmp(Name, ".gnu.linkonce.td.", 17) == 0 ||
      strncmp(Name, ".llvm.linkonce.td.", 18) == 0)
    return SectionKind::getThreadData();
  
  if (strcmp(Name, ".tbss") == 0 ||
      strncmp(Name, ".tbss.", 6) == 0 ||
      strncmp(Name, ".gnu.linkonce.tb.", 17) == 0 ||
      strncmp(Name, ".llvm.linkonce.tb.", 18) == 0)
    return SectionKind::getThreadBSS();
  
  return K;
}


static unsigned
getELFSectionType(const char *Name, SectionKind K) {

  if (strcmp(Name, ".init_array") == 0)
    return MCSectionELF::SHT_INIT_ARRAY;

  if (strcmp(Name, ".fini_array") == 0)
    return MCSectionELF::SHT_FINI_ARRAY;

  if (strcmp(Name, ".preinit_array") == 0)
    return MCSectionELF::SHT_PREINIT_ARRAY;

  if (K.isBSS() || K.isThreadBSS())
    return MCSectionELF::SHT_NOBITS;

  return MCSectionELF::SHT_PROGBITS;
}


static unsigned
getELFSectionFlags(SectionKind K) {
  unsigned Flags = 0;

  if (!K.isMetadata())
    Flags |= MCSectionELF::SHF_ALLOC;
  
  if (K.isText())
    Flags |= MCSectionELF::SHF_EXECINSTR;
  
  if (K.isWriteable())
    Flags |= MCSectionELF::SHF_WRITE;
  
  if (K.isThreadLocal())
    Flags |= MCSectionELF::SHF_TLS;
  
  // K.isMergeableConst() is left out to honour PR4650
  if (K.isMergeableCString() || K.isMergeableConst4() ||
      K.isMergeableConst8() || K.isMergeableConst16())
    Flags |= MCSectionELF::SHF_MERGE;

  if (K.isMergeableCString())
    Flags |= MCSectionELF::SHF_STRINGS;

  return Flags;
}


const MCSection *TargetLoweringObjectFileELF::
getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind, 
                         Mangler *Mang, const TargetMachine &TM) const {
  const char *SectionName = GV->getSection().c_str();

  // Infer section flags from the section name if we can.
  Kind = getELFKindForNamedSection(SectionName, Kind);
  
  return getELFSection(SectionName, 
                       getELFSectionType(SectionName, Kind), 
                       getELFSectionFlags(Kind), Kind, true);
}

static const char *getSectionPrefixForUniqueGlobal(SectionKind Kind) {
  if (Kind.isText())                 return ".gnu.linkonce.t.";
  if (Kind.isReadOnly())             return ".gnu.linkonce.r.";
  
  if (Kind.isThreadData())           return ".gnu.linkonce.td.";
  if (Kind.isThreadBSS())            return ".gnu.linkonce.tb.";
  
  if (Kind.isBSS())                  return ".gnu.linkonce.b.";
  if (Kind.isDataNoRel())            return ".gnu.linkonce.d.";
  if (Kind.isDataRelLocal())         return ".gnu.linkonce.d.rel.local.";
  if (Kind.isDataRel())              return ".gnu.linkonce.d.rel.";
  if (Kind.isReadOnlyWithRelLocal()) return ".gnu.linkonce.d.rel.ro.local.";
  
  assert(Kind.isReadOnlyWithRel() && "Unknown section kind");
  return ".gnu.linkonce.d.rel.ro.";
}

const MCSection *TargetLoweringObjectFileELF::
SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                       Mangler *Mang, const TargetMachine &TM) const {
  
  // If this global is linkonce/weak and the target handles this by emitting it
  // into a 'uniqued' section name, create and return the section now.
  if (GV->isWeakForLinker()) {
    const char *Prefix = getSectionPrefixForUniqueGlobal(Kind);
    std::string Name = Mang->makeNameProper(GV->getNameStr());

    return getELFSection((Prefix+Name).c_str(), 
                         getELFSectionType((Prefix+Name).c_str(), Kind), 
                         getELFSectionFlags(Kind), 
                         Kind);
  }
  
  if (Kind.isText()) return TextSection;
  
  if (Kind.isMergeable1ByteCString() ||
      Kind.isMergeable2ByteCString() ||
      Kind.isMergeable4ByteCString()) {
    
    // We also need alignment here.
    // FIXME: this is getting the alignment of the character, not the
    // alignment of the global!
    unsigned Align = 
      TM.getTargetData()->getPreferredAlignment(cast<GlobalVariable>(GV));
    
    const char *SizeSpec = ".rodata.str1.";
    if (Kind.isMergeable2ByteCString())
      SizeSpec = ".rodata.str2.";
    else if (Kind.isMergeable4ByteCString())
      SizeSpec = ".rodata.str4.";
    else
      assert(Kind.isMergeable1ByteCString() && "unknown string width");
    
    
    std::string Name = SizeSpec + utostr(Align);
    return getELFSection(Name.c_str(), MCSectionELF::SHT_PROGBITS, 
                         MCSectionELF::SHF_ALLOC | 
                         MCSectionELF::SHF_MERGE | 
                         MCSectionELF::SHF_STRINGS, 
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
  
  if (Kind.isBSS())                  return BSSSection;
  
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

//===----------------------------------------------------------------------===//
//                                 MachO
//===----------------------------------------------------------------------===//

typedef StringMap<const MCSectionMachO*> MachOUniqueMapTy;

TargetLoweringObjectFileMachO::~TargetLoweringObjectFileMachO() {
  // If we have the MachO uniquing map, free it.
  delete (MachOUniqueMapTy*)UniquingMap;
}


const MCSectionMachO *TargetLoweringObjectFileMachO::
getMachOSection(const StringRef &Segment, const StringRef &Section,
                unsigned TypeAndAttributes,
                unsigned Reserved2, SectionKind Kind) const {
  // We unique sections by their segment/section pair.  The returned section
  // may not have the same flags as the requested section, if so this should be
  // diagnosed by the client as an error.
  
  // Create the map if it doesn't already exist.
  if (UniquingMap == 0)
    UniquingMap = new MachOUniqueMapTy();
  MachOUniqueMapTy &Map = *(MachOUniqueMapTy*)UniquingMap;
  
  // Form the name to look up.
  SmallString<64> Name;
  Name.append(Segment.begin(), Segment.end());
  Name.push_back(',');
  Name.append(Section.begin(), Section.end());
  
  // Do the lookup, if we have a hit, return it.
  const MCSectionMachO *&Entry = Map[StringRef(Name.data(), Name.size())];
  if (Entry) return Entry;

  // Otherwise, return a new section.
  return Entry = MCSectionMachO::Create(Segment, Section, TypeAndAttributes,
                                        Reserved2, Kind, getContext());
}


void TargetLoweringObjectFileMachO::Initialize(MCContext &Ctx,
                                               const TargetMachine &TM) {
  if (UniquingMap != 0)
    ((MachOUniqueMapTy*)UniquingMap)->clear();
  TargetLoweringObjectFile::Initialize(Ctx, TM);
  
  TextSection // .text
    = getMachOSection("__TEXT", "__text",
                      MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                      SectionKind::getText());
  DataSection // .data
    = getMachOSection("__DATA", "__data", 0, SectionKind::getDataRel());
  
  CStringSection // .cstring
    = getMachOSection("__TEXT", "__cstring", MCSectionMachO::S_CSTRING_LITERALS,
                      SectionKind::getMergeable1ByteCString());
  UStringSection
    = getMachOSection("__TEXT","__ustring", 0,
                      SectionKind::getMergeable2ByteCString());
  FourByteConstantSection // .literal4
    = getMachOSection("__TEXT", "__literal4", MCSectionMachO::S_4BYTE_LITERALS,
                      SectionKind::getMergeableConst4());
  EightByteConstantSection // .literal8
    = getMachOSection("__TEXT", "__literal8", MCSectionMachO::S_8BYTE_LITERALS,
                      SectionKind::getMergeableConst8());
  
  // ld_classic doesn't support .literal16 in 32-bit mode, and ld64 falls back
  // to using it in -static mode.
  SixteenByteConstantSection = 0;
  if (TM.getRelocationModel() != Reloc::Static &&
      TM.getTargetData()->getPointerSize() == 32)
    SixteenByteConstantSection =   // .literal16
      getMachOSection("__TEXT", "__literal16",MCSectionMachO::S_16BYTE_LITERALS,
                      SectionKind::getMergeableConst16());
  
  ReadOnlySection  // .const
    = getMachOSection("__TEXT", "__const", 0, SectionKind::getReadOnly());
  
  TextCoalSection
    = getMachOSection("__TEXT", "__textcoal_nt",
                      MCSectionMachO::S_COALESCED |
                      MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                      SectionKind::getText());
  ConstTextCoalSection
    = getMachOSection("__TEXT", "__const_coal", MCSectionMachO::S_COALESCED,
                      SectionKind::getText());
  ConstDataCoalSection
    = getMachOSection("__DATA","__const_coal", MCSectionMachO::S_COALESCED,
                      SectionKind::getText());
  ConstDataSection  // .const_data
    = getMachOSection("__DATA", "__const", 0,
                      SectionKind::getReadOnlyWithRel());
  DataCoalSection
    = getMachOSection("__DATA","__datacoal_nt", MCSectionMachO::S_COALESCED,
                      SectionKind::getDataRel());

  
  LazySymbolPointerSection
    = getMachOSection("__DATA", "__la_symbol_ptr",
                      MCSectionMachO::S_LAZY_SYMBOL_POINTERS,
                      SectionKind::getMetadata());
  NonLazySymbolPointerSection
    = getMachOSection("__DATA", "__nl_symbol_ptr",
                      MCSectionMachO::S_NON_LAZY_SYMBOL_POINTERS,
                      SectionKind::getMetadata());
  
  if (TM.getRelocationModel() == Reloc::Static) {
    StaticCtorSection
      = getMachOSection("__TEXT", "__constructor", 0,SectionKind::getDataRel());
    StaticDtorSection
      = getMachOSection("__TEXT", "__destructor", 0, SectionKind::getDataRel());
  } else {
    StaticCtorSection
      = getMachOSection("__DATA", "__mod_init_func",
                        MCSectionMachO::S_MOD_INIT_FUNC_POINTERS,
                        SectionKind::getDataRel());
    StaticDtorSection
      = getMachOSection("__DATA", "__mod_term_func", 
                        MCSectionMachO::S_MOD_TERM_FUNC_POINTERS,
                        SectionKind::getDataRel());
  }
  
  // Exception Handling.
  LSDASection = getMachOSection("__DATA", "__gcc_except_tab", 0,
                                SectionKind::getDataRel());
  EHFrameSection =
    getMachOSection("__TEXT", "__eh_frame",
                    MCSectionMachO::S_COALESCED |
                    MCSectionMachO::S_ATTR_NO_TOC |
                    MCSectionMachO::S_ATTR_STRIP_STATIC_SYMS |
                    MCSectionMachO::S_ATTR_LIVE_SUPPORT,
                    SectionKind::getReadOnly());

  // Debug Information.
  DwarfAbbrevSection = 
    getMachOSection("__DWARF", "__debug_abbrev", MCSectionMachO::S_ATTR_DEBUG,
                    SectionKind::getMetadata());
  DwarfInfoSection =  
    getMachOSection("__DWARF", "__debug_info", MCSectionMachO::S_ATTR_DEBUG,
                    SectionKind::getMetadata());
  DwarfLineSection =  
    getMachOSection("__DWARF", "__debug_line", MCSectionMachO::S_ATTR_DEBUG,
                    SectionKind::getMetadata());
  DwarfFrameSection =  
    getMachOSection("__DWARF", "__debug_frame", MCSectionMachO::S_ATTR_DEBUG,
                    SectionKind::getMetadata());
  DwarfPubNamesSection =  
    getMachOSection("__DWARF", "__debug_pubnames", MCSectionMachO::S_ATTR_DEBUG,
                    SectionKind::getMetadata());
  DwarfPubTypesSection =  
    getMachOSection("__DWARF", "__debug_pubtypes", MCSectionMachO::S_ATTR_DEBUG,
                    SectionKind::getMetadata());
  DwarfStrSection =  
    getMachOSection("__DWARF", "__debug_str", MCSectionMachO::S_ATTR_DEBUG,
                    SectionKind::getMetadata());
  DwarfLocSection =  
    getMachOSection("__DWARF", "__debug_loc", MCSectionMachO::S_ATTR_DEBUG,
                    SectionKind::getMetadata());
  DwarfARangesSection =  
    getMachOSection("__DWARF", "__debug_aranges", MCSectionMachO::S_ATTR_DEBUG,
                    SectionKind::getMetadata());
  DwarfRangesSection =  
    getMachOSection("__DWARF", "__debug_ranges", MCSectionMachO::S_ATTR_DEBUG,
                    SectionKind::getMetadata());
  DwarfMacroInfoSection =  
    getMachOSection("__DWARF", "__debug_macinfo", MCSectionMachO::S_ATTR_DEBUG,
                    SectionKind::getMetadata());
  DwarfDebugInlineSection = 
    getMachOSection("__DWARF", "__debug_inlined", MCSectionMachO::S_ATTR_DEBUG,
                    SectionKind::getMetadata());
}

const MCSection *TargetLoweringObjectFileMachO::
getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind, 
                         Mangler *Mang, const TargetMachine &TM) const {
  // Parse the section specifier and create it if valid.
  StringRef Segment, Section;
  unsigned TAA, StubSize;
  std::string ErrorCode =
    MCSectionMachO::ParseSectionSpecifier(GV->getSection(), Segment, Section,
                                          TAA, StubSize);
  if (!ErrorCode.empty()) {
    // If invalid, report the error with llvm_report_error.
    llvm_report_error("Global variable '" + GV->getNameStr() +
                      "' has an invalid section specifier '" + GV->getSection()+
                      "': " + ErrorCode + ".");
    // Fall back to dropping it into the data section.
    return DataSection;
  }
  
  // Get the section.
  const MCSectionMachO *S =
    getMachOSection(Segment, Section, TAA, StubSize, Kind);
  
  // Okay, now that we got the section, verify that the TAA & StubSize agree.
  // If the user declared multiple globals with different section flags, we need
  // to reject it here.
  if (S->getTypeAndAttributes() != TAA || S->getStubSize() != StubSize) {
    // If invalid, report the error with llvm_report_error.
    llvm_report_error("Global variable '" + GV->getNameStr() +
                      "' section type or attributes does not match previous"
                      " section specifier");
  }
  
  return S;
}

const MCSection *TargetLoweringObjectFileMachO::
SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                       Mangler *Mang, const TargetMachine &TM) const {
  assert(!Kind.isThreadLocal() && "Darwin doesn't support TLS");
  
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
  if (Kind.isMergeable1ByteCString() ||
      Kind.isMergeable2ByteCString()) {
    if (TM.getTargetData()->getPreferredAlignment(
                                              cast<GlobalVariable>(GV)) < 32) {
      if (Kind.isMergeable1ByteCString())
        return CStringSection;
      assert(Kind.isMergeable2ByteCString());
      return UStringSection;
    }
  }
  
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
  
  // Otherwise, just drop the variable in the normal data section.
  return DataSection;
}

const MCSection *
TargetLoweringObjectFileMachO::getSectionForConstant(SectionKind Kind) const {
  // If this constant requires a relocation, we have to put it in the data
  // segment, not in the text segment.
  if (Kind.isDataRel())
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
    const std::string &Name = Mang->getMangledName(GV);
    if (Name[0] == 'L' || Name[0] == 'l')
      return false;
  }
  
  return true;
}


//===----------------------------------------------------------------------===//
//                                  COFF
//===----------------------------------------------------------------------===//

typedef StringMap<const MCSectionCOFF*> COFFUniqueMapTy;

TargetLoweringObjectFileCOFF::~TargetLoweringObjectFileCOFF() {
  delete (COFFUniqueMapTy*)UniquingMap;
}


const MCSection *TargetLoweringObjectFileCOFF::
getCOFFSection(const char *Name, bool isDirective, SectionKind Kind) const {
  // Create the map if it doesn't already exist.
  if (UniquingMap == 0)
    UniquingMap = new MachOUniqueMapTy();
  COFFUniqueMapTy &Map = *(COFFUniqueMapTy*)UniquingMap;
  
  // Do the lookup, if we have a hit, return it.
  const MCSectionCOFF *&Entry = Map[Name];
  if (Entry) return Entry;
  
  return Entry = MCSectionCOFF::Create(Name, isDirective, Kind, getContext());
}

void TargetLoweringObjectFileCOFF::Initialize(MCContext &Ctx,
                                              const TargetMachine &TM) {
  if (UniquingMap != 0)
    ((COFFUniqueMapTy*)UniquingMap)->clear();
  TargetLoweringObjectFile::Initialize(Ctx, TM);
  TextSection = getCOFFSection("\t.text", true, SectionKind::getText());
  DataSection = getCOFFSection("\t.data", true, SectionKind::getDataRel());
  StaticCtorSection =
    getCOFFSection(".ctors", false, SectionKind::getDataRel());
  StaticDtorSection =
    getCOFFSection(".dtors", false, SectionKind::getDataRel());
  
  // FIXME: We're emitting LSDA info into a readonly section on COFF, even
  // though it contains relocatable pointers.  In PIC mode, this is probably a
  // big runtime hit for C++ apps.  Either the contents of the LSDA need to be
  // adjusted or this should be a data section.
  LSDASection =
    getCOFFSection(".gcc_except_table", false, SectionKind::getReadOnly());
  EHFrameSection =
    getCOFFSection(".eh_frame", false, SectionKind::getDataRel());
  
  // Debug info.
  // FIXME: Don't use 'directive' mode here.
  DwarfAbbrevSection =  
    getCOFFSection("\t.section\t.debug_abbrev,\"dr\"",
                   true, SectionKind::getMetadata());
  DwarfInfoSection =    
    getCOFFSection("\t.section\t.debug_info,\"dr\"",
                   true, SectionKind::getMetadata());
  DwarfLineSection =    
    getCOFFSection("\t.section\t.debug_line,\"dr\"",
                   true, SectionKind::getMetadata());
  DwarfFrameSection =   
    getCOFFSection("\t.section\t.debug_frame,\"dr\"",
                   true, SectionKind::getMetadata());
  DwarfPubNamesSection =
    getCOFFSection("\t.section\t.debug_pubnames,\"dr\"",
                   true, SectionKind::getMetadata());
  DwarfPubTypesSection =
    getCOFFSection("\t.section\t.debug_pubtypes,\"dr\"",
                   true, SectionKind::getMetadata());
  DwarfStrSection =     
    getCOFFSection("\t.section\t.debug_str,\"dr\"",
                   true, SectionKind::getMetadata());
  DwarfLocSection =     
    getCOFFSection("\t.section\t.debug_loc,\"dr\"",
                   true, SectionKind::getMetadata());
  DwarfARangesSection = 
    getCOFFSection("\t.section\t.debug_aranges,\"dr\"",
                   true, SectionKind::getMetadata());
  DwarfRangesSection =  
    getCOFFSection("\t.section\t.debug_ranges,\"dr\"",
                   true, SectionKind::getMetadata());
  DwarfMacroInfoSection = 
    getCOFFSection("\t.section\t.debug_macinfo,\"dr\"",
                   true, SectionKind::getMetadata());
}

const MCSection *TargetLoweringObjectFileCOFF::
getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind, 
                         Mangler *Mang, const TargetMachine &TM) const {
  return getCOFFSection(GV->getSection().c_str(), false, Kind);
}

static const char *getCOFFSectionPrefixForUniqueGlobal(SectionKind Kind) {
  if (Kind.isText())
    return ".text$linkonce";
  if (Kind.isWriteable())
    return ".data$linkonce";
  return ".rdata$linkonce";
}


const MCSection *TargetLoweringObjectFileCOFF::
SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                       Mangler *Mang, const TargetMachine &TM) const {
  assert(!Kind.isThreadLocal() && "Doesn't support TLS");
  
  // If this global is linkonce/weak and the target handles this by emitting it
  // into a 'uniqued' section name, create and return the section now.
  if (GV->isWeakForLinker()) {
    const char *Prefix = getCOFFSectionPrefixForUniqueGlobal(Kind);
    std::string Name = Mang->makeNameProper(GV->getNameStr());
    return getCOFFSection((Prefix+Name).c_str(), false, Kind);
  }
  
  if (Kind.isText())
    return getTextSection();
  
  return getDataSection();
}

