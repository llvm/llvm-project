//===- C13DebugFragmentVisitor.cpp -------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "C13DebugFragmentVisitor.h"

#include "llvm/DebugInfo/CodeView/ModuleDebugFileChecksumFragment.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugLineFragment.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/DebugInfo/PDB/Native/StringTable.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

C13DebugFragmentVisitor::C13DebugFragmentVisitor(PDBFile &F) : F(F) {}

C13DebugFragmentVisitor::~C13DebugFragmentVisitor() {}

Error C13DebugFragmentVisitor::visitUnknown(
    codeview::ModuleDebugUnknownFragment &Fragment) {
  return Error::success();
}

Error C13DebugFragmentVisitor::visitFileChecksums(
    codeview::ModuleDebugFileChecksumFragment &Checksums) {
  assert(!this->Checksums.hasValue());
  this->Checksums = Checksums;
  return Error::success();
}

Error C13DebugFragmentVisitor::visitLines(
    codeview::ModuleDebugLineFragment &Lines) {
  this->Lines.push_back(Lines);
  return Error::success();
}

Error C13DebugFragmentVisitor::finished() {
  if (!Checksums.hasValue()) {
    assert(Lines.empty());
    return Error::success();
  }
  if (auto EC = handleFileChecksums())
    return EC;

  if (auto EC = handleLines())
    return EC;

  return Error::success();
}

Expected<StringRef>
C13DebugFragmentVisitor::getNameFromStringTable(uint32_t Offset) {
  auto ST = F.getStringTable();
  if (!ST)
    return ST.takeError();

  return ST->getStringForID(Offset);
}

Expected<StringRef>
C13DebugFragmentVisitor::getNameFromChecksumsBuffer(uint32_t Offset) {
  assert(Checksums.hasValue());

  auto Array = Checksums->getArray();
  auto ChecksumIter = Array.at(Offset);
  if (ChecksumIter == Array.end())
    return make_error<RawError>(raw_error_code::invalid_format);
  const auto &Entry = *ChecksumIter;
  return getNameFromStringTable(Entry.FileNameOffset);
}
