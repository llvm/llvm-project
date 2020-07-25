//===------------ TPCDynamicLibrarySearchGenerator.h ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support loading and searching of dynamic libraries in a target process via
// the TargetProcessControl class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_TPCDYNAMICLIBRARYSEARCHGENERATOR_H
#define LLVM_EXECUTIONENGINE_ORC_TPCDYNAMICLIBRARYSEARCHGENERATOR_H

#include "llvm/ExecutionEngine/Orc/TargetProcessControl.h"

namespace llvm {
namespace orc {

class TPCDynamicLibrarySearchGenerator : public JITDylib::DefinitionGenerator {
public:
  /// Create a DynamicLibrarySearchGenerator that searches for symbols in the
  /// library with the given handle.
  ///
  /// If the Allow predicate is given then only symbols matching the predicate
  /// will be searched for. If the predicate is not given then all symbols will
  /// be searched for.
  TPCDynamicLibrarySearchGenerator(TargetProcessControl &TPC,
                                   TargetProcessControl::DylibHandle H)
      : TPC(TPC), H(H) {}

  /// Permanently loads the library at the given path and, on success, returns
  /// a DynamicLibrarySearchGenerator that will search it for symbol definitions
  /// in the library. On failure returns the reason the library failed to load.
  static Expected<std::unique_ptr<TPCDynamicLibrarySearchGenerator>>
  Load(TargetProcessControl &TPC, const char *LibraryPath);

  /// Creates a TPCDynamicLibrarySearchGenerator that searches for symbols in
  /// the target process.
  static Expected<std::unique_ptr<TPCDynamicLibrarySearchGenerator>>
  GetForTargetProcess(TargetProcessControl &TPC) {
    return Load(TPC, nullptr);
  }

  Error tryToGenerate(LookupKind K, JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet &Symbols) override;

private:
  TargetProcessControl &TPC;
  TargetProcessControl::DylibHandle H;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_TPCDYNAMICLIBRARYSEARCHGENERATOR_H
