//===- PrettyClassDefinitionDumper.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_PRETTYCLASSDEFINITIONDUMPER_H
#define LLVM_TOOLS_LLVMPDBDUMP_PRETTYCLASSDEFINITIONDUMPER_H

#include "llvm/DebugInfo/PDB/PDBSymDumper.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"

#include <list>
#include <memory>
#include <unordered_map>

namespace llvm {
namespace pdb {

class LinePrinter;

class ClassDefinitionDumper : public PDBSymDumper {
public:
  ClassDefinitionDumper(LinePrinter &P);

  void start(const PDBSymbolTypeUDT &Exe);

  void dump(const PDBSymbolTypeBaseClass &Symbol) override;
  void dump(const PDBSymbolData &Symbol) override;
  void dump(const PDBSymbolTypeEnum &Symbol) override;
  void dump(const PDBSymbolFunc &Symbol) override;
  void dump(const PDBSymbolTypeTypedef &Symbol) override;
  void dump(const PDBSymbolTypeUDT &Symbol) override;
  void dump(const PDBSymbolTypeVTable &Symbol) override;

private:
  LinePrinter &Printer;

  struct SymbolGroup {
    SymbolGroup() {}
    SymbolGroup(SymbolGroup &&Other) {
      Functions = std::move(Other.Functions);
      Data = std::move(Other.Data);
      Unknown = std::move(Other.Unknown);
    }

    std::list<std::unique_ptr<PDBSymbolFunc>> Functions;
    std::list<std::unique_ptr<PDBSymbolData>> Data;
    std::list<std::unique_ptr<PDBSymbol>> Unknown;
    SymbolGroup(const SymbolGroup &other) = delete;
    SymbolGroup &operator=(const SymbolGroup &other) = delete;
  };
  typedef std::unordered_map<int, SymbolGroup> SymbolGroupByAccess;

  int dumpAccessGroup(PDB_MemberAccess Access, const SymbolGroup &Group);
};
}
}
#endif
