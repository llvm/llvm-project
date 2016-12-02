//===- ModuleSymbolTable.h - symbol table for in-memory IR ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class represents a symbol table built from in-memory IR. It provides
// access to GlobalValues and should only be used if such access is required
// (e.g. in the LTO implementation).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_MODULESYMBOLTABLE_H
#define LLVM_OBJECT_MODULESYMBOLTABLE_H

#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Object/SymbolicFile.h"
#include <string>
#include <utility>

namespace llvm {

class GlobalValue;

class ModuleSymbolTable {
public:
  typedef std::pair<std::string, uint32_t> AsmSymbol;
  typedef PointerUnion<GlobalValue *, AsmSymbol *> Symbol;

private:
  Module *FirstMod = nullptr;

  SpecificBumpPtrAllocator<AsmSymbol> AsmSymbols;
  std::vector<Symbol> SymTab;
  Mangler Mang;

public:
  ArrayRef<Symbol> symbols() const { return SymTab; }
  void addModule(Module *M);

  void printSymbolName(raw_ostream &OS, Symbol S) const;
  uint32_t getSymbolFlags(Symbol S) const;

  /// Parse inline ASM and collect the symbols that are defined or referenced in
  /// the current module.
  ///
  /// For each found symbol, call \p AsmSymbol with the name of the symbol found
  /// and the associated flags.
  static void CollectAsmSymbols(
      const Triple &TheTriple, StringRef InlineAsm,
      function_ref<void(StringRef, object::BasicSymbolRef::Flags)> AsmSymbol);
};

}

#endif
