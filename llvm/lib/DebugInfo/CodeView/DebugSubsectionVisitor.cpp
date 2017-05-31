//===- DebugSubsectionVisitor.cpp ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/DebugSubsectionVisitor.h"

#include "llvm/DebugInfo/CodeView/DebugChecksumsSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugInlineeLinesSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugLinesSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugSubsectionRecord.h"
#include "llvm/DebugInfo/CodeView/DebugUnknownSubsection.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/BinaryStreamRef.h"

using namespace llvm;
using namespace llvm::codeview;

Error llvm::codeview::visitDebugSubsection(const DebugSubsectionRecord &R,
                                           DebugSubsectionVisitor &V) {
  BinaryStreamReader Reader(R.getRecordData());
  switch (R.kind()) {
  case DebugSubsectionKind::Lines: {
    DebugLinesSubsectionRef Fragment;
    if (auto EC = Fragment.initialize(Reader))
      return EC;

    return V.visitLines(Fragment);
  }
  case DebugSubsectionKind::FileChecksums: {
    DebugChecksumsSubsectionRef Fragment;
    if (auto EC = Fragment.initialize(Reader))
      return EC;

    return V.visitFileChecksums(Fragment);
  }
  case DebugSubsectionKind::InlineeLines: {
    DebugInlineeLinesSubsectionRef Fragment;
    if (auto EC = Fragment.initialize(Reader))
      return EC;
    return V.visitInlineeLines(Fragment);
  }
  default: {
    DebugUnknownSubsectionRef Fragment(R.kind(), R.getRecordData());
    return V.visitUnknown(Fragment);
  }
  }
}
