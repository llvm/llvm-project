//===- IRObjectFile.h - LLVM IR object file implementation ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the IRObjectFile template class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_IROBJECTFILE_H
#define LLVM_OBJECT_IROBJECTFILE_H

#include "llvm/ADT/PointerUnion.h"
#include "llvm/Object/ModuleSymbolTable.h"
#include "llvm/Object/SymbolicFile.h"

namespace llvm {
class Mangler;
class Module;
class GlobalValue;
class Triple;

namespace object {
class ObjectFile;

class IRObjectFile : public SymbolicFile {
  std::vector<std::unique_ptr<Module>> Mods;
  ModuleSymbolTable SymTab;
  IRObjectFile(MemoryBufferRef Object,
               std::vector<std::unique_ptr<Module>> Mods);

public:
  ~IRObjectFile() override;
  void moveSymbolNext(DataRefImpl &Symb) const override;
  std::error_code printSymbolName(raw_ostream &OS,
                                  DataRefImpl Symb) const override;
  uint32_t getSymbolFlags(DataRefImpl Symb) const override;
  basic_symbol_iterator symbol_begin() const override;
  basic_symbol_iterator symbol_end() const override;

  StringRef getTargetTriple() const;

  static inline bool classof(const Binary *v) {
    return v->isIR();
  }

  /// \brief Finds and returns bitcode embedded in the given object file, or an
  /// error code if not found.
  static ErrorOr<MemoryBufferRef> findBitcodeInObject(const ObjectFile &Obj);

  /// \brief Finds and returns bitcode in the given memory buffer (which may
  /// be either a bitcode file or a native object file with embedded bitcode),
  /// or an error code if not found.
  static ErrorOr<MemoryBufferRef>
  findBitcodeInMemBuffer(MemoryBufferRef Object);

  static Expected<std::unique_ptr<IRObjectFile>> create(MemoryBufferRef Object,
                                                        LLVMContext &Context);
};
}
}

#endif
