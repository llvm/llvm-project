//===- MCWin64EH.h - Machine Code Win64 EH support --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations to support the Win64 Exception Handling
// scheme in MC.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCWIN64EH_H
#define LLVM_MC_MCWIN64EH_H

#include "llvm/Support/Win64EH.h"
#include <cassert>
#include <vector>

namespace llvm {
  class StringRef;
  class MCStreamer;
  class MCSymbol;

  class MCWin64EHInstruction {
  public:
    typedef Win64EH::UnwindOpcodes OpType;
  private:
    OpType Operation;
    MCSymbol *Label;
    unsigned Offset;
    unsigned Register;
  public:
    MCWin64EHInstruction(OpType Op, MCSymbol *L, unsigned Reg)
      : Operation(Op), Label(L), Offset(0), Register(Reg) {
     assert(Op == Win64EH::UOP_PushNonVol);
    }
    MCWin64EHInstruction(MCSymbol *L, unsigned Size)
      : Operation(Size>128 ? Win64EH::UOP_AllocLarge : Win64EH::UOP_AllocSmall),
        Label(L), Offset(Size) { }
    MCWin64EHInstruction(OpType Op, MCSymbol *L, unsigned Reg, unsigned Off)
      : Operation(Op), Label(L), Offset(Off), Register(Reg) {
      assert(Op == Win64EH::UOP_SetFPReg ||
             Op == Win64EH::UOP_SaveNonVol ||
             Op == Win64EH::UOP_SaveNonVolBig ||
             Op == Win64EH::UOP_SaveXMM128 ||
             Op == Win64EH::UOP_SaveXMM128Big);
    }
    MCWin64EHInstruction(OpType Op, MCSymbol *L, bool Code)
      : Operation(Op), Label(L), Offset(Code ? 1 : 0) {
      assert(Op == Win64EH::UOP_PushMachFrame);
    }
    OpType getOperation() const { return Operation; }
    MCSymbol *getLabel() const { return Label; }
    unsigned getOffset() const { return Offset; }
    unsigned getSize() const { return Offset; }
    unsigned getRegister() const { return Register; }
    bool isPushCodeFrame() const { return Offset == 1; }
  };

  struct MCWin64EHUnwindInfo {
    MCWin64EHUnwindInfo() : Begin(0), End(0), ExceptionHandler(0),
                            Function(0), PrologEnd(0), Symbol(0),
                            HandlesUnwind(false), HandlesExceptions(false),
                            LastFrameInst(-1), ChainedParent(0),
                            Instructions() {}
    MCSymbol *Begin;
    MCSymbol *End;
    const MCSymbol *ExceptionHandler;
    const MCSymbol *Function;
    MCSymbol *PrologEnd;
    MCSymbol *Symbol;
    bool HandlesUnwind;
    bool HandlesExceptions;
    int LastFrameInst;
    MCWin64EHUnwindInfo *ChainedParent;
    std::vector<MCWin64EHInstruction> Instructions;
  };

  class MCWin64EHUnwindEmitter {
  public:
    static StringRef GetSectionSuffix(const MCSymbol *func);
    //
    // This emits the unwind info sections (.pdata and .xdata in PE/COFF).
    //
    static void Emit(MCStreamer &streamer);
    static void EmitUnwindInfo(MCStreamer &streamer, MCWin64EHUnwindInfo *info);
  };
} // end namespace llvm

#endif
