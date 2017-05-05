//===- NativeCompilandSymbol.h - native impl for compiland syms -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_NATIVE_NATIVECOMPILANDSYMBOL_H
#define LLVM_DEBUGINFO_PDB_NATIVE_NATIVECOMPILANDSYMBOL_H

#include "llvm/DebugInfo/PDB/Native/DbiModuleDescriptor.h"
#include "llvm/DebugInfo/PDB/Native/NativeRawSymbol.h"

namespace llvm {
namespace pdb {

class NativeCompilandSymbol : public NativeRawSymbol {
public:
  NativeCompilandSymbol(NativeSession &Session, DbiModuleDescriptor MI);
  PDB_SymType getSymTag() const override;
  bool isEditAndContinueEnabled() const override;
  uint32_t getLexicalParentId() const override;
  std::string getLibraryName() const override;
  std::string getName() const override;

private:
  DbiModuleDescriptor Module;
};

} // namespace pdb
} // namespace llvm

#endif
