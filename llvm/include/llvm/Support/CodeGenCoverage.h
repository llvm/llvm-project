//== llvm/Support/CodeGenCoverage.h ------------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file This file provides rule coverage tracking for tablegen-erated CodeGen.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CODEGENCOVERAGE_H
#define LLVM_SUPPORT_CODEGENCOVERAGE_H

#include "llvm/ADT/BitVector.h"

namespace llvm {
class LLVMContext;
class MemoryBuffer;

class CodeGenCoverage {
protected:
  BitVector RuleCoverage;

public:
  CodeGenCoverage();

  void setCovered(uint64_t RuleID);
  bool isCovered(uint64_t RuleID);

  bool parse(MemoryBuffer &Buffer, StringRef BackendName);
  bool emit(StringRef FilePrefix, StringRef BackendName) const;
  void reset();
};
} // end namespace llvm

#endif // ifndef LLVM_SUPPORT_CODEGENCOVERAGE_H
