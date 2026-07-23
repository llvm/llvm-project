//===-- BacktraceTools.h - Backtrace symbolication tools -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tools for dumping symbol tables and symbolicating backtraces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_BACKTRACETOOLS_H
#define LLVM_EXECUTIONENGINE_ORC_BACKTRACETOOLS_H

#include "llvm/ExecutionEngine/Orc/LinkGraphLinkingLayer.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <mutex>
#include <string>

namespace llvm::orc {

/// Dumps symbol tables from LinkGraphs to enable backtrace symbolication.
///
/// This plugin appends symbol information to a file in the following format:
///   "<link graph name>"
///   <address> <symbol name>
///   <address> <symbol name>
///   ...
///
/// Where addresses are in hexadecimal and symbol names are for defined symbols.
class LLVM_ABI SymbolTableDumpPlugin : public LinkGraphLinkingLayer::Plugin {
public:
  /// Create a SymbolTableDumpPlugin that will append symbol information
  /// to the file at the given path.
  static Expected<std::shared_ptr<SymbolTableDumpPlugin>>
  Create(StringRef Path);

  /// Create a SymbolTableDumpPlugin. The resulting object is in an invalid
  /// state if, upon return, EC != std::error_code().
  /// Prefer SymbolTableDumpPlugin::Create.
  SymbolTableDumpPlugin(StringRef Path, std::error_code &EC);

  SymbolTableDumpPlugin(const SymbolTableDumpPlugin &) = delete;
  SymbolTableDumpPlugin &operator=(const SymbolTableDumpPlugin &) = delete;
  SymbolTableDumpPlugin(SymbolTableDumpPlugin &&) = delete;
  SymbolTableDumpPlugin &operator=(SymbolTableDumpPlugin &&) = delete;

  void modifyPassConfig(MaterializationResponsibility &MR,
                        jitlink::LinkGraph &G,
                        jitlink::PassConfiguration &Config) override;

  Error notifyFailed(MaterializationResponsibility &MR) override {
    return Error::success();
  }

  Error notifyRemovingResources(JITDylib &JD, ResourceKey K) override {
    return Error::success();
  }

  void notifyTransferringResources(JITDylib &JD, ResourceKey DstKey,
                                   ResourceKey SrcKey) override {}

private:
  raw_fd_ostream OutputStream;
  std::mutex DumpMutex;
};

/// A class for symbolicating backtraces using a previously dumped symbol table.
class LLVM_ABI DumpedSymbolTable {
public:
  /// Create a DumpedSymbolTable from the given path.
  static Expected<DumpedSymbolTable> Create(StringRef Path);

  /// Given a backtrace, try to symbolicate any unsymbolicated lines using the
  /// symbol addresses in the dumped symbol table.
  std::string symbolicate(StringRef Backtrace);

private:
  DumpedSymbolTable(std::unique_ptr<MemoryBuffer> SymtabBuffer);

  void parseBuffer();

  struct SymbolInfo {
    StringRef SymName;
    StringRef GraphName;
  };

  std::map<uint64_t, SymbolInfo> SymbolInfos;
  std::unique_ptr<MemoryBuffer> SymtabBuffer;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_BACKTRACETOOLS_H
