//===- TapiFile.h - Text-based Dynamic Library Stub -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the TapiFile interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_TAPIFILE_H
#define LLVM_OBJECT_TAPIFILE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/TextAPI/Architecture.h"

namespace llvm {

class raw_ostream;

namespace MachO {

class InterfaceFile;

}

namespace object {

class TapiFile : public SymbolicFile {
public:
  TapiFile(MemoryBufferRef Source, const MachO::InterfaceFile &interface,
           MachO::Architecture Arch);
  ~TapiFile() override;

  void moveSymbolNext(DataRefImpl &DRI) const override;

  Error printSymbolName(raw_ostream &OS, DataRefImpl DRI) const override;

  Expected<uint32_t> getSymbolFlags(DataRefImpl DRI) const override;

  basic_symbol_iterator symbol_begin() const override;

  basic_symbol_iterator symbol_end() const override;

  static bool classof(const Binary *v) { return v->isTapiFile(); }

  bool is64Bit() const override { return MachO::is64Bit(Arch); }

private:
  struct Symbol {
    StringRef Prefix;
    StringRef Name;
    uint32_t Flags;

    constexpr Symbol(StringRef Prefix, StringRef Name, uint32_t Flags)
        : Prefix(Prefix), Name(Name), Flags(Flags) {}
  };

  std::vector<Symbol> Symbols;
  MachO::Architecture Arch;
};

} // end namespace object.
} // end namespace llvm.

#endif // LLVM_OBJECT_TAPIFILE_H
