//===------------------- Hashing.cpp - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides hashing functions for UnitID, CapabilityRunKey, and content hashing.
// Uses BLAKE3 for fast, secure hashing.
//
//===----------------------------------------------------------------------===//
#include "Utils/Hashing.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::advisor;

std::string llvm::advisor::hashString(StringRef Data) {
  BLAKE3 Hasher;
  Hasher.update(Data);
  BLAKE3Result<> Result = Hasher.final();
  return toHex(Result, true);
}

Expected<std::string> llvm::advisor::hashFile(StringRef Path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer = MemoryBuffer::getFile(Path);
  if (!Buffer)
    return createStringError(Buffer.getError(), "cannot read '%s'",
                             Path.str().c_str());
  return hashString((*Buffer)->getBuffer());
}

std::string llvm::advisor::hashJSON(const json::Value &Value) {
  std::string Storage;
  raw_string_ostream OS(Storage);
  OS << Value;
  return hashString(OS.str());
}

std::string llvm::advisor::computeUnitID(const UnitRecord &Unit) {
  BLAKE3 Hasher;
  Hasher.update(Unit.SourcePath);
  Hasher.update("\0");
  Hasher.update(Unit.Language);
  Hasher.update("\0");
  Hasher.update(Unit.TargetTriple);
  for (const std::string &Arg : Unit.Arguments) {
    Hasher.update("\0");
    Hasher.update(Arg);
  }
  BLAKE3Result<> Result = Hasher.final();
  return toHex(Result, true);
}

std::string llvm::advisor::computeSnapshotID(StringRef SourceRoot,
                                             StringRef BuildRoot,
                                             uint64_t CreatedUnix) {
  std::string Data;
  raw_string_ostream OS(Data);
  OS << SourceRoot << '\0' << BuildRoot << '\0' << CreatedUnix;
  return hashString(OS.str());
}

std::string llvm::advisor::computeCapabilityRunKey(const UnitRecord &Unit,
                                                   StringRef CapabilityID,
                                                   StringRef CapabilityVersion,
                                                   StringRef InputDigest) {
  BLAKE3 Hasher;
  Hasher.update(Unit.ID);
  Hasher.update("\0");
  Hasher.update(CapabilityID);
  Hasher.update("\0");
  Hasher.update(CapabilityVersion);
  Hasher.update("\0");
  Hasher.update(InputDigest);
  BLAKE3Result<> Result = Hasher.final();
  return toHex(Result, true);
}
