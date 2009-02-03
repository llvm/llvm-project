//===-- DarwinTargetAsmInfo.cpp - Darwin asm properties ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on Darwin-based targets
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Target/DarwinTargetAsmInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"

using namespace llvm;

DarwinTargetAsmInfo::DarwinTargetAsmInfo(const TargetMachine &TM) 
  : TargetAsmInfo(TM) {

  CStringSection_ = getUnnamedSection("\t.cstring",
                                SectionFlags::Mergeable | SectionFlags::Strings);
  FourByteConstantSection = getUnnamedSection("\t.literal4\n",
                                              SectionFlags::Mergeable);
  EightByteConstantSection = getUnnamedSection("\t.literal8\n",
                                               SectionFlags::Mergeable);

  // Note: 16-byte constant section is subtarget specific and should be provided
  // there, if needed.
  SixteenByteConstantSection = 0;

  ReadOnlySection = getUnnamedSection("\t.const\n", SectionFlags::None);

  TextCoalSection =
    getNamedSection("\t__TEXT,__textcoal_nt,coalesced,pure_instructions",
                    SectionFlags::Code);
  ConstTextCoalSection = getNamedSection("\t__TEXT,__const_coal,coalesced",
                                         SectionFlags::None);
  ConstDataCoalSection = getNamedSection("\t__DATA,__const_coal,coalesced",
                                         SectionFlags::None);
  ConstDataSection = getUnnamedSection(".const_data", SectionFlags::None);
  DataCoalSection = getNamedSection("\t__DATA,__datacoal_nt,coalesced",
                                    SectionFlags::Writeable);
}

/// emitUsedDirectiveFor - On Darwin, internally linked data beginning with
/// the PrivateGlobalPrefix or the LessPrivateGlobalPrefix does not have the
/// directive emitted (this occurs in ObjC metadata).

bool
DarwinTargetAsmInfo::emitUsedDirectiveFor(const GlobalValue* GV,
                                          Mangler *Mang) const {
  if (GV==0)
    return false;
  if (GV->hasLocalLinkage() && !isa<Function>(GV) &&
      ((strlen(getPrivateGlobalPrefix()) != 0 &&
        Mang->getValueName(GV).substr(0,strlen(getPrivateGlobalPrefix())) ==
          getPrivateGlobalPrefix()) ||
       (strlen(getLessPrivateGlobalPrefix()) != 0 &&
        Mang->getValueName(GV).substr(0,strlen(getLessPrivateGlobalPrefix())) ==
          getLessPrivateGlobalPrefix())))
    return false;
  return true;
}

const Section*
DarwinTargetAsmInfo::SelectSectionForGlobal(const GlobalValue *GV) const {
  SectionKind::Kind Kind = SectionKindForGlobal(GV);
  bool isWeak = GV->mayBeOverridden();
  bool isNonStatic = TM.getRelocationModel() != Reloc::Static;

  switch (Kind) {
   case SectionKind::Text:
    if (isWeak)
      return TextCoalSection;
    else
      return TextSection;
   case SectionKind::Data:
   case SectionKind::ThreadData:
   case SectionKind::BSS:
   case SectionKind::ThreadBSS:
    if (cast<GlobalVariable>(GV)->isConstant())
      return (isWeak ? ConstDataCoalSection : ConstDataSection);
    else
      return (isWeak ? DataCoalSection : DataSection);
   case SectionKind::ROData:
    return (isWeak ? ConstDataCoalSection :
            (isNonStatic ? ConstDataSection : getReadOnlySection()));
   case SectionKind::RODataMergeStr:
    return (isWeak ?
            ConstTextCoalSection :
            MergeableStringSection(cast<GlobalVariable>(GV)));
   case SectionKind::RODataMergeConst:
    return (isWeak ?
            ConstDataCoalSection:
            MergeableConstSection(cast<GlobalVariable>(GV)));
   default:
    assert(0 && "Unsuported section kind for global");
  }

  // FIXME: Do we have any extra special weird cases?
  return NULL;
}

const Section*
DarwinTargetAsmInfo::MergeableStringSection(const GlobalVariable *GV) const {
  const TargetData *TD = TM.getTargetData();
  Constant *C = cast<GlobalVariable>(GV)->getInitializer();
  const Type *Ty = cast<ArrayType>(C->getType())->getElementType();

  unsigned Size = TD->getTypePaddedSize(Ty);
  if (Size) {
    unsigned Align = TD->getPreferredAlignment(GV);
    if (Align <= 32)
      return getCStringSection_();
  }

  return getReadOnlySection();
}

const Section*
DarwinTargetAsmInfo::MergeableConstSection(const GlobalVariable *GV) const {
  Constant *C = GV->getInitializer();

  return MergeableConstSection(C->getType());
}

inline const Section*
DarwinTargetAsmInfo::MergeableConstSection(const Type *Ty) const {
  const TargetData *TD = TM.getTargetData();

  unsigned Size = TD->getTypePaddedSize(Ty);
  if (Size == 4)
    return FourByteConstantSection;
  else if (Size == 8)
    return EightByteConstantSection;
  else if (Size == 16 && SixteenByteConstantSection)
    return SixteenByteConstantSection;

  return getReadOnlySection();
}

const Section*
DarwinTargetAsmInfo::SelectSectionForMachineConst(const Type *Ty) const {
  const Section* S = MergeableConstSection(Ty);

  // Handle weird special case, when compiling PIC stuff.
  if (S == getReadOnlySection() &&
      TM.getRelocationModel() != Reloc::Static)
    return ConstDataSection;

  return S;
}

std::string
DarwinTargetAsmInfo::UniqueSectionForGlobal(const GlobalValue* GV,
                                               SectionKind::Kind kind) const {
  assert(0 && "Darwin does not use unique sections");
  return "";
}
