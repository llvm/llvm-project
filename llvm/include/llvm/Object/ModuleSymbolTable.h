//===- ModuleSymbolTable.h - symbol table for in-memory IR ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace llvm {

class GlobalValue;
class Module;

class ModuleSymbolTable {
public:
  using AsmSymbol = std::pair<std::string, uint32_t>;
  using Symbol = PointerUnion<GlobalValue *, AsmSymbol *>;

private:
  Module *FirstMod = nullptr;

  SpecificBumpPtrAllocator<AsmSymbol> AsmSymbols;
  std::vector<Symbol> SymTab;
  Mangler Mang;

public:
  ArrayRef<Symbol> symbols() const { return SymTab; }
  LLVM_ABI void addModule(Module *M);

  LLVM_ABI void printSymbolName(raw_ostream &OS, Symbol S) const;
  LLVM_ABI uint32_t getSymbolFlags(Symbol S) const;

  /// Parse inline ASM and collect the symbols that are defined or referenced in
  /// the current module.
  ///
  /// For each found symbol, call \p AsmSymbol with the name of the symbol found
  /// and the associated flags.
  LLVM_ABI static void CollectAsmSymbols(
      const Module &M,
      function_ref<void(StringRef, object::BasicSymbolRef::Flags)> AsmSymbol);

  /// Parse inline ASM and collect the symvers directives that are defined in
  /// the current module.
  ///
  /// For each found symbol, call \p AsmSymver with the name of the symbol and
  /// its alias.
  LLVM_ABI static void
  CollectAsmSymvers(const Module &M,
                    function_ref<void(StringRef, StringRef)> AsmSymver);

  /// Emit module flags for symbols and symvers defined in global
  /// inline assembly. If these flags are present, CollectAsmSymbols
  /// and CollectAsmSymvers will use them instead of trying to parse
  /// assembly again.
  ///
  /// This allows LLVM IR tools to build a symbol table for an IR
  /// module without knowing exact CPU and Features required to parse
  /// its global inline assembly.
  LLVM_ABI static bool EmitModuleFlags(Module &M, StringRef CPU,
                                       StringRef Features);
};

} // end namespace llvm

#endif // LLVM_OBJECT_MODULESYMBOLTABLE_H
