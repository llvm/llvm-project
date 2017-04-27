//===- ModuleDebugFragmentVisitor.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFRAGMENTVISITOR_H
#define LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFRAGMENTVISITOR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/CodeViewError.h"
#include "llvm/DebugInfo/CodeView/Line.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugFileChecksumFragment.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugFragmentRecord.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugLineFragment.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugUnknownFragment.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/BinaryStreamRef.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include <cstdint>

namespace llvm {

namespace codeview {

class ModuleDebugFragmentVisitor {
public:
  virtual ~ModuleDebugFragmentVisitor() = default;

  virtual Error visitUnknown(ModuleDebugUnknownFragment &Unknown) {
    return Error::success();
  }
  virtual Error visitLines(ModuleDebugLineFragment &Lines) {
    return Error::success();
  }

  virtual Error visitFileChecksums(ModuleDebugFileChecksumFragment &Checksums) {
    return Error::success();
  }
};

Error visitModuleDebugFragment(const ModuleDebugFragmentRecord &R,
                               ModuleDebugFragmentVisitor &V);
} // end namespace codeview

} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFRAGMENTVISITOR_H
