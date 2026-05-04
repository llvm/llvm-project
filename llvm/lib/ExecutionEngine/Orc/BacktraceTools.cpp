//===------- BacktraceTools.cpp - Backtrace symbolication tools ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/BacktraceTools.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm::orc {

Expected<std::shared_ptr<SymbolTableDumpPlugin>>
SymbolTableDumpPlugin::Create(StringRef Path) {
  std::error_code EC;
  auto P = std::make_shared<SymbolTableDumpPlugin>(Path, EC);
  if (EC)
    return createFileError(Path, EC);
  return P;
}

SymbolTableDumpPlugin::SymbolTableDumpPlugin(StringRef Path,
                                             std::error_code &EC)
    : OutputStream(Path, EC) {}

void SymbolTableDumpPlugin::modifyPassConfig(
    MaterializationResponsibility &MR, jitlink::LinkGraph &G,
    jitlink::PassConfiguration &Config) {

  Config.PostAllocationPasses.push_back([this](jitlink::LinkGraph &G) -> Error {
    std::scoped_lock<std::mutex> Lock(DumpMutex);

    OutputStream << "\"" << G.getName() << "\"\n";
    for (auto &Sec : G.sections()) {
      // NoAlloc symbols don't exist in the executing process, so can't
      // contribute to symbolication. (Note: We leave Finalize-liftime symbols
      // in for now in case of crashes during finalization, but we should
      // probably make this optional).
      if (Sec.getMemLifetime() == MemLifetime::NoAlloc)
        continue;

      // Write out named symbols. Anonymous symbols are skipped, since they
      // don't add any information for symbolication purposes.
      for (auto *Sym : Sec.symbols()) {
        if (Sym->hasName())
          OutputStream << formatv("{0:x}", Sym->getAddress().getValue()) << " "
                       << Sym->getName() << "\n";
      }
    }

    OutputStream.flush();
    return Error::success();
  });
}

Expected<DumpedSymbolTable> DumpedSymbolTable::Create(StringRef Path) {
  auto MB = MemoryBuffer::getFile(Path);
  if (!MB)
    return createFileError(Path, MB.getError());

  return DumpedSymbolTable(std::move(*MB));
}

DumpedSymbolTable::DumpedSymbolTable(std::unique_ptr<MemoryBuffer> SymtabBuffer)
    : SymtabBuffer(std::move(SymtabBuffer)) {
  parseBuffer();
}

void DumpedSymbolTable::parseBuffer() {
  // Read the symbol table file
  SmallVector<StringRef, 0> Rows;
  SymtabBuffer->getBuffer().split(Rows, '\n');

  StringRef CurGraph = "<unidentified>";
  for (auto Row : Rows) {
    Row = Row.trim();
    if (Row.empty())
      continue;

    // Check for graph name line (enclosed in quotes)
    if (Row.starts_with("\"") && Row.ends_with("\"")) {
      CurGraph = Row.trim('"');
      continue;
    }

    // Parse "address symbol_name" lines, ignoring malformed lines.
    size_t SpacePos = Row.find(' ');
    if (SpacePos == StringRef::npos)
      continue;

    StringRef AddrStr = Row.substr(0, SpacePos);
    StringRef SymName = Row.substr(SpacePos + 1);

    uint64_t Addr;
    if (AddrStr.starts_with("0x"))
      AddrStr = AddrStr.drop_front(2);
    if (AddrStr.getAsInteger(16, Addr))
      continue; // Skip malformed lines

    SymbolInfos[Addr] = {SymName, CurGraph};
  }
}

std::string DumpedSymbolTable::symbolicate(StringRef Backtrace) {
  // Symbolicate the backtrace by replacing rows with empty symbol names
  SmallVector<StringRef, 0> BacktraceRows;
  Backtrace.split(BacktraceRows, '\n');

  std::string Result;
  raw_string_ostream Out(Result);
  for (auto Row : BacktraceRows) {
    // Look for a row ending with a hex number. If there's only one column, or
    // if the last column is not a hex number, then just reproduce the input
    // row.
    auto [RowStart, AddrCol] = Row.rtrim().rsplit(' ');
    auto AddrStr = AddrCol.starts_with("0x") ? AddrCol.drop_front(2) : AddrCol;

    uint64_t Addr;
    if (AddrStr.empty() || AddrStr.getAsInteger(16, Addr)) {
      Out << Row << "\n";
      continue;
    }

    // Search for the address
    auto I = SymbolInfos.upper_bound(Addr);

    // If no JIT symbol entry within 2Gb then skip.
    if (I == SymbolInfos.begin() || (Addr - std::prev(I)->first >= 1U << 31)) {
      Out << Row << "\n";
      continue;
    }

    // Found a symbol. Output modified line.
    auto &[SymAddr, SymInfo] = *std::prev(I);
    Out << RowStart << " " << AddrCol << " " << SymInfo.SymName;
    if (auto Delta = Addr - SymAddr)
      Out << " + " << formatv("{0}", Delta);
    Out << " (" << SymInfo.GraphName << ")\n";
  }

  return Result;
}

} // namespace llvm::orc
