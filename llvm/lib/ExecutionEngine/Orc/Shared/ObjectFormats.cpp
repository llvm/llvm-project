//===---------- ObjectFormats.cpp - Object format details for ORC ---------===//
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

#include "llvm/ExecutionEngine/Orc/Shared/ObjectFormats.h"

namespace llvm {
namespace orc {

StringRef ELFEHFrameSectionName = ".eh_frame";

StringRef ELFInitArrayFuncSectionName = ".init_array";
StringRef ELFInitFuncSectionName = ".init";
StringRef ELFFiniArrayFuncSectionName = ".fini_array";
StringRef ELFFiniFuncSectionName = ".fini";
StringRef ELFCtorArrayFuncSectionName = ".ctors";
StringRef ELFDtorArrayFuncSectionName = ".dtors";

StringRef ELFInitSectionNames[3]{
    ELFInitArrayFuncSectionName,
    ELFInitFuncSectionName,
    ELFCtorArrayFuncSectionName,
};

StringRef ELFThreadBSSSectionName = ".tbss";
StringRef ELFThreadDataSectionName = ".tdata";

bool isMachOInitializerSection(StringRef QualifiedName) {
  for (auto &InitSection : MachOInitSectionNames)
    if (InitSection == QualifiedName)
      return true;
  return false;
}

bool isELFInitializerSection(StringRef SecName) {
  for (StringRef InitSection : ELFInitSectionNames) {
    StringRef Name = SecName;
    if (Name.consume_front(InitSection) && (Name.empty() || Name[0] == '.'))
      return true;
  }
  return false;
}

bool isCOFFInitializerSection(StringRef SecName) {
  return SecName.starts_with(".CRT");
}

} // namespace orc
} // namespace llvm
