//===---------- DebugUtils.cpp - Utilities for debugging ORC JITs ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/DebugUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {

DumpObjects::DumpObjects(std::string DumpDir, std::string IdentifierOverride)
    : DumpDir(std::move(DumpDir)),
      IdentifierOverride(std::move(IdentifierOverride)) {

  /// Discard any trailing separators.
  while (!this->DumpDir.empty() &&
         sys::path::is_separator(this->DumpDir.back()))
    this->DumpDir.pop_back();
}

Expected<std::unique_ptr<MemoryBuffer>>
DumpObjects::operator()(std::unique_ptr<MemoryBuffer> Obj) {
  size_t Idx = 1;

  std::string DumpPathStem;
  raw_string_ostream(DumpPathStem)
      << DumpDir << (DumpDir.empty() ? "" : "/") << getBufferIdentifier(*Obj);

  std::string DumpPath = DumpPathStem + ".o";
  while (sys::fs::exists(DumpPath)) {
    DumpPath.clear();
    raw_string_ostream(DumpPath) << DumpPathStem << "." << (++Idx) << ".o";
  }

  LLVM_DEBUG({
    dbgs() << "Dumping object buffer [ " << (const void *)Obj->getBufferStart()
           << " -- " << (const void *)(Obj->getBufferEnd() - 1) << " ] to "
           << DumpPath << "\n";
  });

  std::error_code EC;
  raw_fd_ostream DumpStream(DumpPath, EC);
  if (EC)
    return errorCodeToError(EC);
  DumpStream.write(Obj->getBufferStart(), Obj->getBufferSize());

  return std::move(Obj);
}

StringRef DumpObjects::getBufferIdentifier(MemoryBuffer &B) {
  if (!IdentifierOverride.empty())
    return IdentifierOverride;
  StringRef Identifier = B.getBufferIdentifier();
  Identifier.consume_back(".o");
  return Identifier;
}

} // End namespace orc.
} // End namespace llvm.
