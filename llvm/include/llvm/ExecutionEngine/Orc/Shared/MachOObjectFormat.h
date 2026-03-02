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
#include "llvm/Support/Compiler.h"

namespace llvm {
namespace orc {

// FIXME: Move these to BinaryFormat?

// MachO section names.

LLVM_ABI extern StringRef MachODataCommonSectionName;
LLVM_ABI extern StringRef MachODataDataSectionName;
LLVM_ABI extern StringRef MachOEHFrameSectionName;
LLVM_ABI extern StringRef MachOCompactUnwindSectionName;
LLVM_ABI extern StringRef MachOCStringSectionName;
LLVM_ABI extern StringRef MachOModInitFuncSectionName;
LLVM_ABI extern StringRef MachOObjCCatListSectionName;
LLVM_ABI extern StringRef MachOObjCCatList2SectionName;
LLVM_ABI extern StringRef MachOObjCClassListSectionName;
LLVM_ABI extern StringRef MachOObjCClassNameSectionName;
LLVM_ABI extern StringRef MachOObjCClassRefsSectionName;
LLVM_ABI extern StringRef MachOObjCConstSectionName;
LLVM_ABI extern StringRef MachOObjCDataSectionName;
LLVM_ABI extern StringRef MachOObjCImageInfoSectionName;
LLVM_ABI extern StringRef MachOObjCMethNameSectionName;
LLVM_ABI extern StringRef MachOObjCMethTypeSectionName;
LLVM_ABI extern StringRef MachOObjCNLCatListSectionName;
LLVM_ABI extern StringRef MachOObjCNLClassListSectionName;
LLVM_ABI extern StringRef MachOObjCProtoListSectionName;
LLVM_ABI extern StringRef MachOObjCProtoRefsSectionName;
LLVM_ABI extern StringRef MachOObjCSelRefsSectionName;
LLVM_ABI extern StringRef MachOSwift5ProtoSectionName;
LLVM_ABI extern StringRef MachOSwift5ProtosSectionName;
LLVM_ABI extern StringRef MachOSwift5TypesSectionName;
LLVM_ABI extern StringRef MachOSwift5TypeRefSectionName;
LLVM_ABI extern StringRef MachOSwift5FieldMetadataSectionName;
LLVM_ABI extern StringRef MachOSwift5EntrySectionName;
LLVM_ABI extern StringRef MachOTextTextSectionName;
LLVM_ABI extern StringRef MachOThreadBSSSectionName;
LLVM_ABI extern StringRef MachOThreadDataSectionName;
LLVM_ABI extern StringRef MachOThreadVarsSectionName;
LLVM_ABI extern StringRef MachOUnwindInfoSectionName;

LLVM_ABI extern StringRef MachOInitSectionNames[22];

LLVM_ABI bool isMachOInitializerSection(StringRef SegName, StringRef SecName);
LLVM_ABI bool isMachOInitializerSection(StringRef QualifiedName);

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_MACHOOBJECTFORMAT_H
