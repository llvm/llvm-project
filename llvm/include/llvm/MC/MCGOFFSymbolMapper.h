//===- MCGOFFSymbolMapper.h - Maps MC section/symbol to GOFF symbols ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Maps a section or a symbol to the GOFF symbols it is composed of, and their
// attributes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCGOFFSYMBOLMAPPER_H
#define LLVM_MC_MCGOFFSYMBOLMAPPER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/Support/Alignment.h"
#include <string>
#include <utility>

namespace llvm {
class MCAssembler;
class MCContext;
class MCSectionGOFF;

// An "External Symbol Definition" in the GOFF file has a type, and depending on
// the type a different subset of the fields is used.
//
// Unlike other formats, a 2 dimensional structure is used to define the
// location of data. For example, the equivalent of the ELF .text section is
// made up of a Section Definition (SD) and a class (Element Definition; ED).
// The name of the SD symbol depends on the application, while the class has the
// predefined name C_CODE64.
//
// Data can be placed into this structure in 2 ways. First, the data (in a text
// record) can be associated with an ED symbol. To refer to data, a Label
// Definition (LD) is used to give an offset into the data a name. When binding,
// the whole data is pulled into the resulting executable, and the addresses
// given by the LD symbols are resolved.
//
// The alternative is to use a Part Defiition (PR). In this case, the data (in a
// text record) is associated with the part. When binding, only the data of
// referenced PRs is pulled into the resulting binary.
//
// Both approaches are used, which means that the equivalent of a section in ELF
// results in 3 GOFF symbol, either SD/ED/LD or SD/ED/PR. Moreover, certain
// sections are fine with just defining SD/ED symbols. The SymbolMapper takes
// care of all those details.

// Attributes for SD symbols.
struct SDAttr {
  GOFF::ESDTaskingBehavior TaskingBehavior = GOFF::ESD_TA_Unspecified;
  GOFF::ESDBindingScope BindingScope = GOFF::ESD_BSC_Unspecified;
};

// Attributes for ED symbols.
struct EDAttr {
  bool IsReadOnly = false;
  GOFF::ESDExecutable Executable = GOFF::ESD_EXE_Unspecified;
  GOFF::ESDAmode Amode;
  GOFF::ESDRmode Rmode;
  GOFF::ESDNameSpaceId NameSpace = GOFF::ESD_NS_NormalName;
  GOFF::ESDTextStyle TextStyle = GOFF::ESD_TS_ByteOriented;
  GOFF::ESDBindingAlgorithm BindAlgorithm = GOFF::ESD_BA_Concatenate;
  GOFF::ESDLoadingBehavior LoadBehavior = GOFF::ESD_LB_Initial;
  GOFF::ESDReserveQwords ReservedQwords = GOFF::ESD_RQ_0;
  GOFF::ESDAlignment Alignment = GOFF::ESD_ALIGN_Doubleword;
};

// Attributes for LD symbols.
struct LDAttr {
  bool IsRenamable = false;
  GOFF::ESDExecutable Executable = GOFF::ESD_EXE_Unspecified;
  GOFF::ESDNameSpaceId NameSpace = GOFF::ESD_NS_NormalName;
  GOFF::ESDBindingStrength BindingStrength = GOFF::ESD_BST_Strong;
  GOFF::ESDLinkageType Linkage = GOFF::ESD_LT_XPLink;
  GOFF::ESDAmode Amode;
  GOFF::ESDBindingScope BindingScope = GOFF::ESD_BSC_Unspecified;
};

// Attributes for PR symbols.
struct PRAttr {
  bool IsRenamable = false;
  bool IsReadOnly = false; // ???? Not documented.
  GOFF::ESDExecutable Executable = GOFF::ESD_EXE_Unspecified;
  GOFF::ESDNameSpaceId NameSpace = GOFF::ESD_NS_NormalName;
  GOFF::ESDLinkageType Linkage = GOFF::ESD_LT_XPLink;
  GOFF::ESDAmode Amode;
  GOFF::ESDBindingScope BindingScope = GOFF::ESD_BSC_Unspecified;
  GOFF::ESDDuplicateSymbolSeverity DuplicateSymbolSeverity =
      GOFF::ESD_DSS_NoWarning;
  GOFF::ESDAlignment Alignment = GOFF::ESD_ALIGN_Byte;
  uint32_t SortKey = 0;
};

struct GOFFSectionData {
  // Name and attributes of SD symbol.
  StringRef SDName;
  SDAttr SDAttributes;

  // Name and attributes of ED symbol.
  StringRef EDName;
  EDAttr EDAttributes;

  // Name and attributes of LD or PR symbol.
  StringRef LDorPRName;
  LDAttr LDAttributes;
  PRAttr PRAttributes;

  // Indicates if there is a LD or PR symbol.
  enum { None, LD, PR } Tag;

  // Indicates if the SD symbol is to root symbol (aka the Csect Code).
  bool IsSDRootSD;
};

class GOFFSymbolMapper {
  MCContext &Ctx;

  std::string RootSDName;
  SDAttr RootSDAttributes;

  std::string ADALDName;

  StringRef BaseName;

  bool IsCsectCodeNameEmpty;
  bool Is64Bit;
  bool UsesXPLINK;

public:
  GOFFSymbolMapper(MCContext &Ctx);
  GOFFSymbolMapper(MCAssembler &Asm);

  // Required order: .text first, then .ada.
  std::pair<GOFFSectionData, bool> getSection(const MCSectionGOFF &Section);

  void setBaseName();
  void determineRootSD(StringRef CSectCodeName);
  llvm::StringRef getRootSDName() const;
  const SDAttr &getRootSD() const;
};

} // namespace llvm

#endif
