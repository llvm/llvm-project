//===---- MachOObjectFormat.h - MachO format details for ORC ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ORC-specific MachO object format details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_MACHOOBJECTFORMAT_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_MACHOOBJECTFORMAT_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
namespace orc {

// FIXME: Move these to BinaryFormat?

// MachO section names.

extern StringRef MachODataCommonSectionName;
extern StringRef MachODataDataSectionName;
extern StringRef MachOEHFrameSectionName;
extern StringRef MachOCStringSectionName;
extern StringRef MachOModInitFuncSectionName;
extern StringRef MachOObjCCatListSectionName;
extern StringRef MachOObjCCatList2SectionName;
extern StringRef MachOObjCClassListSectionName;
extern StringRef MachOObjCClassNameSectionName;
extern StringRef MachOObjCClassRefsSectionName;
extern StringRef MachOObjCConstSectionName;
extern StringRef MachOObjCDataSectionName;
extern StringRef MachOObjCImageInfoSectionName;
extern StringRef MachOObjCMethNameSectionName;
extern StringRef MachOObjCMethTypeSectionName;
extern StringRef MachOObjCNLCatListSectionName;
extern StringRef MachOObjCNLClassListSectionName;
extern StringRef MachOObjCProtoListSectionName;
extern StringRef MachOObjCProtoRefsSectionName;
extern StringRef MachOObjCSelRefsSectionName;
extern StringRef MachOSwift5ProtoSectionName;
extern StringRef MachOSwift5ProtosSectionName;
extern StringRef MachOSwift5TypesSectionName;
extern StringRef MachOSwift5TypeRefSectionName;
extern StringRef MachOSwift5FieldMetadataSectionName;
extern StringRef MachOSwift5EntrySectionName;
extern StringRef MachOTextTextSectionName;
extern StringRef MachOThreadBSSSectionName;
extern StringRef MachOThreadDataSectionName;
extern StringRef MachOThreadVarsSectionName;
extern StringRef MachOUnwindInfoSectionName;

extern StringRef MachOInitSectionNames[22];

bool isMachOInitializerSection(StringRef SegName, StringRef SecName);
bool isMachOInitializerSection(StringRef QualifiedName);

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_MACHOOBJECTFORMAT_H
