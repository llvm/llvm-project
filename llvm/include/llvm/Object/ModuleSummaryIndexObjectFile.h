//===- ModuleSummaryIndexObjectFile.h - Summary index file implementation -=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ModuleSummaryIndexObjectFile template class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_MODULESUMMARYINDEXOBJECTFILE_H
#define LLVM_OBJECT_MODULESUMMARYINDEXOBJECTFILE_H

#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Object/SymbolicFile.h"

namespace llvm {
class ModuleSummaryIndex;
class Module;

namespace object {
class ObjectFile;

/// This class is used to read just the module summary index related
/// sections out of the given object (which may contain a single module's
/// bitcode or be a combined index bitcode file). It builds a ModuleSummaryIndex
/// object.
class ModuleSummaryIndexObjectFile : public SymbolicFile {
  std::unique_ptr<ModuleSummaryIndex> Index;

public:
  ModuleSummaryIndexObjectFile(MemoryBufferRef Object,
                               std::unique_ptr<ModuleSummaryIndex> I);
  ~ModuleSummaryIndexObjectFile() override;

  // TODO: Walk through GlobalValueMap entries for symbols.
  // However, currently these interfaces are not used by any consumers.
  void moveSymbolNext(DataRefImpl &Symb) const override {
    llvm_unreachable("not implemented");
  }
  std::error_code printSymbolName(raw_ostream &OS,
                                  DataRefImpl Symb) const override {
    llvm_unreachable("not implemented");
    return std::error_code();
  }
  uint32_t getSymbolFlags(DataRefImpl Symb) const override {
    llvm_unreachable("not implemented");
    return 0;
  }
  basic_symbol_iterator symbol_begin_impl() const override {
    llvm_unreachable("not implemented");
    return basic_symbol_iterator(BasicSymbolRef());
  }
  basic_symbol_iterator symbol_end_impl() const override {
    llvm_unreachable("not implemented");
    return basic_symbol_iterator(BasicSymbolRef());
  }

  const ModuleSummaryIndex &getIndex() const {
    return const_cast<ModuleSummaryIndexObjectFile *>(this)->getIndex();
  }
  ModuleSummaryIndex &getIndex() { return *Index; }
  std::unique_ptr<ModuleSummaryIndex> takeIndex();

  static inline bool classof(const Binary *v) {
    return v->isModuleSummaryIndex();
  }

  /// \brief Finds and returns bitcode embedded in the given object file, or an
  /// error code if not found.
  static ErrorOr<MemoryBufferRef> findBitcodeInObject(const ObjectFile &Obj);

  /// \brief Finds and returns bitcode in the given memory buffer (which may
  /// be either a bitcode file or a native object file with embedded bitcode),
  /// or an error code if not found.
  static ErrorOr<MemoryBufferRef>
  findBitcodeInMemBuffer(MemoryBufferRef Object);

  /// \brief Looks for summary sections in the given memory buffer,
  /// returns true if found, else false.
  static bool hasGlobalValueSummaryInMemBuffer(
      MemoryBufferRef Object,
      const DiagnosticHandlerFunction &DiagnosticHandler);

  /// \brief Parse module summary index in the given memory buffer.
  /// Return new ModuleSummaryIndexObjectFile instance containing parsed module
  /// summary/index.
  static ErrorOr<std::unique_ptr<ModuleSummaryIndexObjectFile>>
  create(MemoryBufferRef Object,
         const DiagnosticHandlerFunction &DiagnosticHandler);
};
}

/// Parse the module summary index out of an IR file and return the module
/// summary index object if found, or nullptr if not.
ErrorOr<std::unique_ptr<ModuleSummaryIndex>> getModuleSummaryIndexForFile(
    StringRef Path, const DiagnosticHandlerFunction &DiagnosticHandler);
}

#endif
