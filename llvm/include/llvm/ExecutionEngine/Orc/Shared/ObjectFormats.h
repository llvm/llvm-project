//===------ ObjectFormats.h - Object format details for ORC -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ORC-specific object format details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_OBJECTFORMATS_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_OBJECTFORMATS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
namespace orc {

// MachO section names.

LLVM_FUNC_ABI extern StringRef MachODataCommonSectionName;
LLVM_FUNC_ABI extern StringRef MachODataDataSectionName;
LLVM_FUNC_ABI extern StringRef MachOEHFrameSectionName;
LLVM_FUNC_ABI extern StringRef MachOCompactUnwindInfoSectionName;
LLVM_FUNC_ABI extern StringRef MachOModInitFuncSectionName;
LLVM_FUNC_ABI extern StringRef MachOObjCCatListSectionName;
LLVM_FUNC_ABI extern StringRef MachOObjCCatList2SectionName;
LLVM_FUNC_ABI extern StringRef MachOObjCClassListSectionName;
LLVM_FUNC_ABI extern StringRef MachOObjCClassNameSectionName;
LLVM_FUNC_ABI extern StringRef MachOObjCClassRefsSectionName;
LLVM_FUNC_ABI extern StringRef MachOObjCConstSectionName;
LLVM_FUNC_ABI extern StringRef MachOObjCDataSectionName;
LLVM_FUNC_ABI extern StringRef MachOObjCImageInfoSectionName;
LLVM_FUNC_ABI extern StringRef MachOObjCMethNameSectionName;
LLVM_FUNC_ABI extern StringRef MachOObjCMethTypeSectionName;
LLVM_FUNC_ABI extern StringRef MachOObjCNLCatListSectionName;
LLVM_FUNC_ABI extern StringRef MachOObjCSelRefsSectionName;
LLVM_FUNC_ABI extern StringRef MachOSwift5ProtoSectionName;
LLVM_FUNC_ABI extern StringRef MachOSwift5ProtosSectionName;
LLVM_FUNC_ABI extern StringRef MachOSwift5TypesSectionName;
LLVM_FUNC_ABI extern StringRef MachOSwift5TypeRefSectionName;
LLVM_FUNC_ABI extern StringRef MachOSwift5FieldMetadataSectionName;
LLVM_FUNC_ABI extern StringRef MachOSwift5EntrySectionName;
LLVM_FUNC_ABI extern StringRef MachOThreadBSSSectionName;
LLVM_FUNC_ABI extern StringRef MachOThreadDataSectionName;
LLVM_FUNC_ABI extern StringRef MachOThreadVarsSectionName;

LLVM_FUNC_ABI extern StringRef MachOInitSectionNames[19];

// ELF section names.
LLVM_FUNC_ABI extern StringRef ELFEHFrameSectionName;

LLVM_FUNC_ABI extern StringRef ELFInitArrayFuncSectionName;
LLVM_FUNC_ABI extern StringRef ELFInitFuncSectionName;
LLVM_FUNC_ABI extern StringRef ELFFiniArrayFuncSectionName;
LLVM_FUNC_ABI extern StringRef ELFFiniFuncSectionName;
LLVM_FUNC_ABI extern StringRef ELFCtorArrayFuncSectionName;
LLVM_FUNC_ABI extern StringRef ELFDtorArrayFuncSectionName;

LLVM_FUNC_ABI extern StringRef ELFInitSectionNames[3];

LLVM_FUNC_ABI extern StringRef ELFThreadBSSSectionName;
LLVM_FUNC_ABI extern StringRef ELFThreadDataSectionName;

LLVM_FUNC_ABI bool isMachOInitializerSection(StringRef SegName, StringRef SecName);
LLVM_FUNC_ABI bool isMachOInitializerSection(StringRef QualifiedName);

LLVM_FUNC_ABI bool isELFInitializerSection(StringRef SecName);

LLVM_FUNC_ABI bool isCOFFInitializerSection(StringRef Name);

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_MEMORYFLAGS_H
