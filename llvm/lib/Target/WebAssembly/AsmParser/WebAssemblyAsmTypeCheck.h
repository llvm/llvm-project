//==- WebAssemblyAsmTypeCheck.h - Assembler for WebAssembly -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file is part of the WebAssembly Assembler.
///
/// It contains code to translate a parsed .s file into MCInsts.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_ASMPARSER_TYPECHECK_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_ASMPARSER_TYPECHECK_H

#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCSymbol.h"

namespace llvm {

class WebAssemblyAsmTypeCheck final {
  MCAsmParser &Parser;
  const MCInstrInfo &MII;

  SmallVector<wasm::ValType, 8> Stack;
  SmallVector<SmallVector<wasm::ValType, 4>, 8> BrStack;
  SmallVector<wasm::ValType, 16> LocalTypes;
  SmallVector<wasm::ValType, 4> ReturnTypes;
  wasm::WasmSignature LastSig;
  bool Unreachable = false;
  bool Is64;

  void dumpTypeStack(Twine Msg);
  bool typeError(SMLoc ErrorLoc, const Twine &Msg);
  bool popType(SMLoc ErrorLoc, std::optional<wasm::ValType> EVT);
  bool popRefType(SMLoc ErrorLoc);
  bool getLocal(SMLoc ErrorLoc, const MCOperand &LocalOp, wasm::ValType &Type);
  bool checkEnd(SMLoc ErrorLoc, bool PopVals = false);
  bool checkBr(SMLoc ErrorLoc, size_t Level);
  bool checkSig(SMLoc ErrorLoc, const wasm::WasmSignature &Sig);
  bool getSymRef(SMLoc ErrorLoc, const MCOperand &SymOp,
                 const MCSymbolRefExpr *&SymRef);
  bool getGlobal(SMLoc ErrorLoc, const MCOperand &GlobalOp,
                 wasm::ValType &Type);
  bool getTable(SMLoc ErrorLoc, const MCOperand &TableOp, wasm::ValType &Type);
  bool getSignature(SMLoc ErrorLoc, const MCOperand &SigOp,
                    wasm::WasmSymbolType Type, const wasm::WasmSignature *&Sig);

public:
  WebAssemblyAsmTypeCheck(MCAsmParser &Parser, const MCInstrInfo &MII,
                          bool Is64);

  void funcDecl(const wasm::WasmSignature &Sig);
  void localDecl(const SmallVectorImpl<wasm::ValType> &Locals);
  void setLastSig(const wasm::WasmSignature &Sig) { LastSig = Sig; }
  bool endOfFunction(SMLoc ErrorLoc);
  bool typeCheck(SMLoc ErrorLoc, const MCInst &Inst, OperandVector &Operands);

  void clear() {
    Stack.clear();
    BrStack.clear();
    LocalTypes.clear();
    ReturnTypes.clear();
    Unreachable = false;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_WEBASSEMBLY_ASMPARSER_TYPECHECK_H
