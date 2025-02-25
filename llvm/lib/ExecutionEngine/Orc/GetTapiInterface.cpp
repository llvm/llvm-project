//===--------- GetTapiInterface.cpp - Get interface from TAPI file --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/GetTapiInterface.h"

#define DEBUG_TYPE "orc"

namespace llvm::orc {

Expected<SymbolNameSet> getInterfaceFromTapiFile(ExecutionSession &ES,
                                                 object::TapiUniversal &TU) {
  SymbolNameSet Symbols;

  auto CPUType = MachO::getCPUType(ES.getTargetTriple());
  if (!CPUType)
    return CPUType.takeError();

  auto CPUSubType = MachO::getCPUSubType(ES.getTargetTriple());
  if (!CPUSubType)
    return CPUSubType.takeError();

  auto &TUIF = TU.getInterfaceFile();
  auto ArchInterface =
      TUIF.extract(MachO::getArchitectureFromCpuType(*CPUType, *CPUSubType));
  if (!ArchInterface)
    return ArchInterface.takeError();

  for (auto *Sym : (*ArchInterface)->exports())
    Symbols.insert(ES.intern(Sym->getName()));

  return Symbols;
}

} // namespace llvm::orc
