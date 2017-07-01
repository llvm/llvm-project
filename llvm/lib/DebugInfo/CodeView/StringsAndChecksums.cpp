//===- StringsAndChecksums.cpp --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/StringsAndChecksums.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/DebugChecksumsSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugStringTableSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugSubsectionRecord.h"
#include "llvm/Support/Error.h"
#include <cassert>

using namespace llvm;
using namespace llvm::codeview;

StringsAndChecksumsRef::StringsAndChecksumsRef() = default;

StringsAndChecksumsRef::StringsAndChecksumsRef(
    const DebugStringTableSubsectionRef &Strings)
    : Strings(&Strings) {}

StringsAndChecksumsRef::StringsAndChecksumsRef(
    const DebugStringTableSubsectionRef &Strings,
    const DebugChecksumsSubsectionRef &Checksums)
    : Strings(&Strings), Checksums(&Checksums) {}

void StringsAndChecksumsRef::initializeStrings(
    const DebugSubsectionRecord &SR) {
  assert(SR.kind() == DebugSubsectionKind::StringTable);
  assert(!Strings && "Found a string table even though we already have one!");

  OwnedStrings = llvm::make_unique<DebugStringTableSubsectionRef>();
  consumeError(OwnedStrings->initialize(SR.getRecordData()));
  Strings = OwnedStrings.get();
}

void StringsAndChecksumsRef::setChecksums(
    const DebugChecksumsSubsectionRef &CS) {
  OwnedChecksums = llvm::make_unique<DebugChecksumsSubsectionRef>();
  *OwnedChecksums = CS;
  Checksums = OwnedChecksums.get();
}

void StringsAndChecksumsRef::initializeChecksums(
    const DebugSubsectionRecord &FCR) {
  assert(FCR.kind() == DebugSubsectionKind::FileChecksums);
  if (Checksums)
    return;

  OwnedChecksums = llvm::make_unique<DebugChecksumsSubsectionRef>();
  consumeError(OwnedChecksums->initialize(FCR.getRecordData()));
  Checksums = OwnedChecksums.get();
}
