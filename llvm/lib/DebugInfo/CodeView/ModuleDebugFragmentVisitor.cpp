//===- ModuleDebugFragmentVisitor.cpp ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/ModuleDebugFragmentVisitor.h"

#include "llvm/DebugInfo/CodeView/ModuleDebugFileChecksumFragment.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugFragmentRecord.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugInlineeLinesFragment.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugLineFragment.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugUnknownFragment.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/BinaryStreamRef.h"

using namespace llvm;
using namespace llvm::codeview;

Error llvm::codeview::visitModuleDebugFragment(
    const ModuleDebugFragmentRecord &R, ModuleDebugFragmentVisitor &V) {
  BinaryStreamReader Reader(R.getRecordData());
  switch (R.kind()) {
  case ModuleDebugFragmentKind::Lines: {
    ModuleDebugLineFragmentRef Fragment;
    if (auto EC = Fragment.initialize(Reader))
      return EC;

    return V.visitLines(Fragment);
  }
  case ModuleDebugFragmentKind::FileChecksums: {
    ModuleDebugFileChecksumFragmentRef Fragment;
    if (auto EC = Fragment.initialize(Reader))
      return EC;

    return V.visitFileChecksums(Fragment);
  }
  case ModuleDebugFragmentKind::InlineeLines: {
    ModuleDebugInlineeLineFragmentRef Fragment;
    if (auto EC = Fragment.initialize(Reader))
      return EC;
    return V.visitInlineeLines(Fragment);
  }
  default: {
    ModuleDebugUnknownFragmentRef Fragment(R.kind(), R.getRecordData());
    return V.visitUnknown(Fragment);
  }
  }
}
