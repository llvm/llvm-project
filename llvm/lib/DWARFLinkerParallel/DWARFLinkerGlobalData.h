//===- DWARFLinkerGlobalData.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERGLOBALDATA_H
#define LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERGLOBALDATA_H

#include "llvm/DWARFLinkerParallel/DWARFLinker.h"
#include "llvm/DWARFLinkerParallel/StringPool.h"
#include "llvm/Support/PerThreadBumpPtrAllocator.h"
#include <map>

namespace llvm {

class DWARFDie;

namespace dwarflinker_parallel {

using TranslatorFuncTy = std::function<StringRef(StringRef)>;
using MessageHandlerTy = std::function<void(
    const Twine &Warning, StringRef Context, const DWARFDie *DIE)>;

/// linking options
struct DWARFLinkerOptions {
  /// DWARF version for the output.
  uint16_t TargetDWARFVersion = 0;

  /// Generate processing log to the standard output.
  bool Verbose = false;

  /// Print statistics.
  bool Statistics = false;

  /// Verify the input DWARF.
  bool VerifyInputDWARF = false;

  /// Do not emit output.
  bool NoOutput = false;

  /// Do not unique types according to ODR
  bool NoODR = false;

  /// Update index tables.
  bool UpdateIndexTablesOnly = false;

  /// Whether we want a static variable to force us to keep its enclosing
  /// function.
  bool KeepFunctionForStatic = false;

  /// Allow to generate valid, but non deterministic output.
  bool AllowNonDeterministicOutput = false;

  /// Number of threads.
  unsigned Threads = 1;

  /// The accelerator table kinds
  SmallVector<DWARFLinker::AccelTableKind, 1> AccelTables;

  /// Prepend path for the clang modules.
  std::string PrependPath;

  /// input verification handler(it might be called asynchronously).
  DWARFLinker::InputVerificationHandlerTy InputVerificationHandler = nullptr;

  /// A list of all .swiftinterface files referenced by the debug
  /// info, mapping Module name to path on disk. The entries need to
  /// be uniqued and sorted and there are only few entries expected
  /// per compile unit, which is why this is a std::map.
  /// this is dsymutil specific fag.
  ///
  /// (it might be called asynchronously).
  DWARFLinker::SwiftInterfacesMapTy *ParseableSwiftInterfaces = nullptr;

  /// A list of remappings to apply to file paths.
  ///
  /// (it might be called asynchronously).
  DWARFLinker::ObjectPrefixMapTy *ObjectPrefixMap = nullptr;
};

class DWARFLinkerImpl;

/// This class keeps data and services common for the whole linking process.
class LinkingGlobalData {
  friend DWARFLinkerImpl;

public:
  /// Returns global per-thread allocator.
  parallel::PerThreadBumpPtrAllocator &getAllocator() { return Allocator; }

  /// Returns global string pool.
  StringPool &getStringPool() { return Strings; }

  /// Set translation function.
  void setTranslator(TranslatorFuncTy Translator) {
    this->Translator = Translator;
  }

  /// Translate specified string.
  StringRef translateString(StringRef String) {
    if (Translator)
      return Translator(String);

    return String;
  }

  /// Returns linking options.
  const DWARFLinkerOptions &getOptions() const { return Options; }

  /// Set warning handler.
  void setWarningHandler(MessageHandlerTy Handler) { WarningHandler = Handler; }

  /// Set error handler.
  void setErrorHandler(MessageHandlerTy Handler) { ErrorHandler = Handler; }

  /// Report warning.
  void warn(const Twine &Warning, StringRef Context,
            const DWARFDie *DIE = nullptr) {
    if (WarningHandler)
      (WarningHandler)(Warning, Context, DIE);
  }

  /// Report warning.
  void warn(Error Warning, StringRef Context, const DWARFDie *DIE = nullptr) {
    handleAllErrors(std::move(Warning), [&](ErrorInfoBase &Info) {
      warn(Info.message(), Context, DIE);
    });
  }

  /// Report error.
  void error(const Twine &Err, StringRef Context,
             const DWARFDie *DIE = nullptr) {
    if (ErrorHandler)
      (ErrorHandler)(Err, Context, DIE);
  }

  /// Report error.
  void error(Error Err, StringRef Context, const DWARFDie *DIE = nullptr) {
    handleAllErrors(std::move(Err), [&](ErrorInfoBase &Info) {
      error(Info.message(), Context, DIE);
    });
  }

protected:
  parallel::PerThreadBumpPtrAllocator Allocator;
  StringPool Strings;
  TranslatorFuncTy Translator;
  DWARFLinkerOptions Options;
  MessageHandlerTy WarningHandler;
  MessageHandlerTy ErrorHandler;
};

} // end of namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_LIB_DWARFLINKERPARALLEL_DWARFLINKERGLOBALDATA_H
