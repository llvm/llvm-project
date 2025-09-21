//===- MCGOFFAttributes.h - Attributes of GOFF symbols --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the various attribute collections defining GOFF symbols.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCGOFFATTRIBUTES_H
#define LLVM_MC_MCGOFFATTRIBUTES_H

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/GOFF.h"
#include <cstdint>

namespace llvm {
namespace GOFF {
// An "External Symbol Definition" in the GOFF file has a type, and depending on
// the type a different subset of the fields is used.
//
// Unlike other formats, a 2 dimensional structure is used to define the
// location of data. For example, the equivalent of the ELF .text section is
// made up of a Section Definition (SD) and a class (Element Definition; ED).
// The name of the SD symbol depends on the application, while the class has the
// predefined name C_CODE/C_CODE64 in AMODE31 and AMODE64 respectively.
//
// Data can be placed into this structure in 2 ways. First, the data (in a text
// record) can be associated with an ED symbol. To refer to data, a Label
// Definition (LD) is used to give an offset into the data a name. When binding,
// the whole data is pulled into the resulting executable, and the addresses
// given by the LD symbols are resolved.
//
// The alternative is to use a Part Definition (PR). In this case, the data (in
// a text record) is associated with the part. When binding, only the data of
// referenced PRs is pulled into the resulting binary.
//
// Both approaches are used. SD, ED, and PR elements are modelled by nested
// MCSectionGOFF instances, while LD elements are associated with MCSymbolGOFF
// instances.

// Attributes for SD symbols.
struct SDAttr {
  GOFF::ESDTaskingBehavior TaskingBehavior = GOFF::ESD_TA_Unspecified;
  GOFF::ESDBindingScope BindingScope = GOFF::ESD_BSC_Unspecified;
};

// Attributes for ED symbols.
struct EDAttr {
  bool IsReadOnly = false;
  GOFF::ESDRmode Rmode;
  GOFF::ESDNameSpaceId NameSpace = GOFF::ESD_NS_NormalName;
  GOFF::ESDTextStyle TextStyle = GOFF::ESD_TS_ByteOriented;
  GOFF::ESDBindingAlgorithm BindAlgorithm = GOFF::ESD_BA_Concatenate;
  GOFF::ESDLoadingBehavior LoadBehavior = GOFF::ESD_LB_Initial;
  GOFF::ESDReserveQwords ReservedQwords = GOFF::ESD_RQ_0;
  GOFF::ESDAlignment Alignment = GOFF::ESD_ALIGN_Doubleword;
  uint8_t FillByteValue = 0;
};

// Attributes for LD symbols.
struct LDAttr {
  bool IsRenamable = false;
  GOFF::ESDExecutable Executable = GOFF::ESD_EXE_Unspecified;
  GOFF::ESDBindingStrength BindingStrength = GOFF::ESD_BST_Strong;
  GOFF::ESDLinkageType Linkage = GOFF::ESD_LT_XPLink;
  GOFF::ESDAmode Amode;
  GOFF::ESDBindingScope BindingScope = GOFF::ESD_BSC_Unspecified;
};

// Attributes for PR symbols.
struct PRAttr {
  bool IsRenamable = false;
  GOFF::ESDExecutable Executable = GOFF::ESD_EXE_Unspecified;
  GOFF::ESDLinkageType Linkage = GOFF::ESD_LT_XPLink;
  GOFF::ESDBindingScope BindingScope = GOFF::ESD_BSC_Unspecified;
  uint32_t SortKey = 0;
};

// Predefined GOFF class names.
constexpr StringLiteral CLASS_CODE = "C_CODE64";
constexpr StringLiteral CLASS_WSA = "C_WSA64";
constexpr StringLiteral CLASS_DATA = "C_DATA64";
constexpr StringLiteral CLASS_PPA2 = "C_@@QPPA2";

} // namespace GOFF
} // namespace llvm

#endif
