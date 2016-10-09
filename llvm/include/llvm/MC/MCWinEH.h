//===- MCWinEH.h - Windows Unwinding Support --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCWINEH_H
#define LLVM_MC_MCWINEH_H

#include <vector>

namespace llvm {
class MCSection;
class MCStreamer;
class MCSymbol;

namespace WinEH {
struct Instruction {
  const MCSymbol *Label;
  const unsigned Offset;
  const unsigned Register;
  const unsigned Operation;

  Instruction(unsigned Op, MCSymbol *L, unsigned Reg, unsigned Off)
    : Label(L), Offset(Off), Register(Reg), Operation(Op) {}
};

struct FrameInfo {
  const MCSymbol *Begin = nullptr;
  const MCSymbol *End = nullptr;
  const MCSymbol *ExceptionHandler = nullptr;
  const MCSymbol *Function = nullptr;
  const MCSymbol *PrologEnd = nullptr;
  const MCSymbol *Symbol = nullptr;
  const MCSection *TextSection = nullptr;

  bool HandlesUnwind = false;
  bool HandlesExceptions = false;

  int LastFrameInst = -1;
  const FrameInfo *ChainedParent = nullptr;
  std::vector<Instruction> Instructions;

  FrameInfo() = default;
  FrameInfo(const MCSymbol *Function, const MCSymbol *BeginFuncEHLabel)
      : Begin(BeginFuncEHLabel), Function(Function) {}
  FrameInfo(const MCSymbol *Function, const MCSymbol *BeginFuncEHLabel,
            const FrameInfo *ChainedParent)
      : Begin(BeginFuncEHLabel), Function(Function),
        ChainedParent(ChainedParent) {}
};
}
}

#endif
