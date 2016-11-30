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
#include "llvm/Object/SymbolicFile.h"

namespace llvm {
class Mangler;
class Module;
class GlobalValue;
class Triple;

namespace object {
class ObjectFile;

class IRObjectFile : public SymbolicFile {
  std::unique_ptr<Module> M;
  std::unique_ptr<Mangler> Mang;
  typedef std::pair<std::string, uint32_t> AsmSymbol;
  SpecificBumpPtrAllocator<AsmSymbol> AsmSymbols;

  typedef PointerUnion<GlobalValue *, AsmSymbol *> Sym;
  std::vector<Sym> SymTab;
  static Sym getSym(DataRefImpl &Symb) {
    return *reinterpret_cast<Sym *>(Symb.p);
  }

public:
  IRObjectFile(MemoryBufferRef Object, std::unique_ptr<Module> M);
  ~IRObjectFile() override;
  void moveSymbolNext(DataRefImpl &Symb) const override;
  std::error_code printSymbolName(raw_ostream &OS,
                                  DataRefImpl Symb) const override;
  uint32_t getSymbolFlags(DataRefImpl Symb) const override;
  GlobalValue *getSymbolGV(DataRefImpl Symb);
  const GlobalValue *getSymbolGV(DataRefImpl Symb) const {
    return const_cast<IRObjectFile *>(this)->getSymbolGV(Symb);
  }
  basic_symbol_iterator symbol_begin() const override;
  basic_symbol_iterator symbol_end() const override;

  const Module &getModule() const {
    return const_cast<IRObjectFile*>(this)->getModule();
  }
  Module &getModule() {
    return *M;
  }
  std::unique_ptr<Module> takeModule();

  StringRef getTargetTriple() const;

  static inline bool classof(const Binary *v) {
    return v->isIR();
  }

  /// \brief Finds and returns bitcode embedded in the given object file, or an
  /// error code if not found.
  static ErrorOr<MemoryBufferRef> findBitcodeInObject(const ObjectFile &Obj);

  /// Parse inline ASM and collect the symbols that are not defined in
  /// the current module.
  ///
  /// For each found symbol, call \p AsmUndefinedRefs with the name of the
  /// symbol found and the associated flags.
  static void CollectAsmUndefinedRefs(
      const Triple &TheTriple, StringRef InlineAsm,
      function_ref<void(StringRef, BasicSymbolRef::Flags)> AsmUndefinedRefs);

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
