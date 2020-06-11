//===- SymbolizableObjectFile.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the SymbolizableObjectFile class.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_DEBUGINFO_SYMBOLIZE_SYMBOLIZABLEOBJECTFILE_H
#define LLVM_LIB_DEBUGINFO_SYMBOLIZE_SYMBOLIZABLEOBJECTFILE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableModule.h"
#include "llvm/Support/ErrorOr.h"
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <system_error>

namespace llvm {

class DataExtractor;

namespace symbolize {

class SymbolizableObjectFile : public SymbolizableModule {
public:
  static Expected<std::unique_ptr<SymbolizableObjectFile>>
  create(const object::ObjectFile *Obj, std::unique_ptr<DIContext> DICtx,
         bool UntagAddresses);

  DILineInfo symbolizeCode(object::SectionedAddress ModuleOffset,
                           DILineInfoSpecifier LineInfoSpecifier,
                           bool UseSymbolTable) const override;
  DIInliningInfo symbolizeInlinedCode(object::SectionedAddress ModuleOffset,
                                      DILineInfoSpecifier LineInfoSpecifier,
                                      bool UseSymbolTable) const override;
  DIGlobal symbolizeData(object::SectionedAddress ModuleOffset) const override;
  std::vector<DILocal>
  symbolizeFrame(object::SectionedAddress ModuleOffset) const override;

  // Return true if this is a 32-bit x86 PE COFF module.
  bool isWin32Module() const override;

  // Returns the preferred base of the module, i.e. where the loader would place
  // it in memory assuming there were no conflicts.
  uint64_t getModulePreferredBase() const override;

private:
  bool shouldOverrideWithSymbolTable(FunctionNameKind FNKind,
                                     bool UseSymbolTable) const;

  bool getNameFromSymbolTable(object::SymbolRef::Type Type, uint64_t Address,
                              std::string &Name, uint64_t &Addr,
                              uint64_t &Size) const;
  // For big-endian PowerPC64 ELF, OpdAddress is the address of the .opd
  // (function descriptor) section and OpdExtractor refers to its contents.
  Error addSymbol(const object::SymbolRef &Symbol, uint64_t SymbolSize,
                  DataExtractor *OpdExtractor = nullptr,
                  uint64_t OpdAddress = 0);
  Error addCoffExportSymbols(const object::COFFObjectFile *CoffObj);

  /// Search for the first occurence of specified Address in ObjectFile.
  uint64_t getModuleSectionIndexForAddress(uint64_t Address) const;

  const object::ObjectFile *Module;
  std::unique_ptr<DIContext> DebugInfoContext;
  bool UntagAddresses;

  struct SymbolDesc {
    uint64_t Addr;
    // If size is 0, assume that symbol occupies the whole memory range up to
    // the following symbol.
    uint64_t Size;

    bool operator<(const SymbolDesc &RHS) const {
      return Addr != RHS.Addr ? Addr < RHS.Addr : Size < RHS.Size;
    }
  };
  std::vector<std::pair<SymbolDesc, StringRef>> Functions;
  std::vector<std::pair<SymbolDesc, StringRef>> Objects;

  SymbolizableObjectFile(const object::ObjectFile *Obj,
                         std::unique_ptr<DIContext> DICtx,
                         bool UntagAddresses);
};

} // end namespace symbolize

} // end namespace llvm

#endif // LLVM_LIB_DEBUGINFO_SYMBOLIZE_SYMBOLIZABLEOBJECTFILE_H
