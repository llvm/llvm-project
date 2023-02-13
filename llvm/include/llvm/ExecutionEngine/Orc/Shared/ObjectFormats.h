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

namespace llvm {
namespace orc {

// MachO section names.
extern StringRef MachODataCommonSectionName;
extern StringRef MachODataDataSectionName;
extern StringRef MachOEHFrameSectionName;
extern StringRef MachOCompactUnwindInfoSectionName;
extern StringRef MachOModInitFuncSectionName;
extern StringRef MachOObjCClassListSectionName;
extern StringRef MachOObjCImageInfoSectionName;
extern StringRef MachOObjCSelRefsSectionName;
extern StringRef MachOSwift5ProtoSectionName;
extern StringRef MachOSwift5ProtosSectionName;
extern StringRef MachOSwift5TypesSectionName;
extern StringRef MachOThreadBSSSectionName;
extern StringRef MachOThreadDataSectionName;
extern StringRef MachOThreadVarsSectionName;
extern StringRef MachOInitSectionNames[6];

// ELF section names.
extern StringRef ELFEHFrameSectionName;
extern StringRef ELFInitArrayFuncSectionName;

extern StringRef ELFThreadBSSSectionName;
extern StringRef ELFThreadDataSectionName;

bool isMachOInitializerSection(StringRef SegName, StringRef SecName);
bool isMachOInitializerSection(StringRef QualifiedName);

bool isELFInitializerSection(StringRef SecName);

bool isCOFFInitializerSection(StringRef Name);

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_MEMORYFLAGS_H
