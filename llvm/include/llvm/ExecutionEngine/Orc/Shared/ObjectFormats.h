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
#include "llvm/ExecutionEngine/Orc/Shared/MachOObjectFormat.h"

namespace llvm {
namespace orc {

// ELF section names.
extern StringRef ELFEHFrameSectionName;

extern StringRef ELFInitArrayFuncSectionName;
extern StringRef ELFInitFuncSectionName;
extern StringRef ELFFiniArrayFuncSectionName;
extern StringRef ELFFiniFuncSectionName;
extern StringRef ELFCtorArrayFuncSectionName;
extern StringRef ELFDtorArrayFuncSectionName;

extern StringRef ELFInitSectionNames[3];

extern StringRef ELFThreadBSSSectionName;
extern StringRef ELFThreadDataSectionName;

bool isELFInitializerSection(StringRef SecName);

bool isCOFFInitializerSection(StringRef Name);

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_OBJECTFORMATS_H
