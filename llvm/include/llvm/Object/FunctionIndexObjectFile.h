//===- FunctionIndexObjectFile.h - Function index file implementation -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the FunctionIndexObjectFile template class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_FUNCTIONINDEXOBJECTFILE_H
#define LLVM_OBJECT_FUNCTIONINDEXOBJECTFILE_H

#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Object/SymbolicFile.h"

namespace llvm {
class FunctionInfoIndex;
class Module;

namespace object {
class ObjectFile;

/// This class is used to read just the function summary index related
/// sections out of the given object (which may contain a single module's
/// bitcode or be a combined index bitcode file). It builds a FunctionInfoIndex
/// object.
class FunctionIndexObjectFile : public SymbolicFile {
  std::unique_ptr<FunctionInfoIndex> Index;

public:
  FunctionIndexObjectFile(MemoryBufferRef Object,
                          std::unique_ptr<FunctionInfoIndex> I);
  ~FunctionIndexObjectFile() override;

  // TODO: Walk through FunctionMap entries for function symbols.
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

  const FunctionInfoIndex &getIndex() const {
    return const_cast<FunctionIndexObjectFile *>(this)->getIndex();
  }
  FunctionInfoIndex &getIndex() { return *Index; }
  std::unique_ptr<FunctionInfoIndex> takeIndex();

  static inline bool classof(const Binary *v) { return v->isFunctionIndex(); }

  /// \brief Finds and returns bitcode embedded in the given object file, or an
  /// error code if not found.
  static ErrorOr<MemoryBufferRef> findBitcodeInObject(const ObjectFile &Obj);

  /// \brief Finds and returns bitcode in the given memory buffer (which may
  /// be either a bitcode file or a native object file with embedded bitcode),
  /// or an error code if not found.
  static ErrorOr<MemoryBufferRef>
  findBitcodeInMemBuffer(MemoryBufferRef Object);

  /// \brief Looks for function summary in the given memory buffer,
  /// returns true if found, else false.
  static bool
  hasFunctionSummaryInMemBuffer(MemoryBufferRef Object,
                                DiagnosticHandlerFunction DiagnosticHandler);

  /// \brief Parse function index in the given memory buffer.
  /// Return new FunctionIndexObjectFile instance containing parsed function
  /// summary/index.
  static ErrorOr<std::unique_ptr<FunctionIndexObjectFile>>
  create(MemoryBufferRef Object, DiagnosticHandlerFunction DiagnosticHandler,
         bool IsLazy = false);

  /// \brief Parse the function summary information for function with the
  /// given name out of the given buffer. Parsed information is
  /// stored on the index object saved in this object.
  std::error_code
  findFunctionSummaryInMemBuffer(MemoryBufferRef Object,
                                 DiagnosticHandlerFunction DiagnosticHandler,
                                 StringRef FunctionName);
};
}

/// Parse the function index out of an IR file and return the function
/// index object if found, or nullptr if not.
ErrorOr<std::unique_ptr<FunctionInfoIndex>>
getFunctionIndexForFile(StringRef Path,
                        DiagnosticHandlerFunction DiagnosticHandler);
}

#endif
