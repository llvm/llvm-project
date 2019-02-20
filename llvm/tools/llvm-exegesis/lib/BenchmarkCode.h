//===-- BenchmarkCode.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_BENCHMARKCODE_H
#define LLVM_TOOLS_LLVM_EXEGESIS_BENCHMARKCODE_H

#include "RegisterValue.h"
#include "llvm/MC/MCInst.h"
#include <string>
#include <vector>

namespace llvm {
namespace exegesis {

// A collection of instructions that are to be assembled, executed and measured.
struct BenchmarkCode {
  // The sequence of instructions that are to be repeated.
  std::vector<llvm::MCInst> Instructions;

  // Before the code is executed some instructions are added to setup the
  // registers initial values.
  std::vector<RegisterValue> RegisterInitialValues;

  // We also need to provide the registers that are live on entry for the
  // assembler to generate proper prologue/epilogue.
  std::vector<unsigned> LiveIns;

  // Informations about how this configuration was built.
  std::string Info;
};

} // namespace exegesis
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_EXEGESIS_BENCHMARKCODE_H
