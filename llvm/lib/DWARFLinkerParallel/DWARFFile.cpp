//=== DWARFFile.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DWARFLinkerParallel/DWARFFile.h"
#include "DWARFLinkerGlobalData.h"

llvm::dwarflinker_parallel::DWARFFile::DWARFFile(
    StringRef Name, std::unique_ptr<DWARFContext> Dwarf,
    std::unique_ptr<AddressesMap> Addresses,
    const std::vector<std::string> &Warnings,
    DWARFFile::UnloadCallbackTy UnloadFunc)
    : FileName(Name), Dwarf(std::move(Dwarf)), Addresses(std::move(Addresses)),
      Warnings(Warnings), UnloadFunc(UnloadFunc) {}
