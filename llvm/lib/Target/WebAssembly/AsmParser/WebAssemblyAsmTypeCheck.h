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

#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/MC/MCSymbol.h"

namespace llvm {

class WebAssemblyAsmTypeCheck final {
  MCAsmParser &Parser;
  const MCInstrInfo &MII;

  SmallVector<wasm::ValType, 8> Stack;
  SmallVector<wasm::ValType, 16> LocalTypes;
  SmallVector<wasm::ValType, 4> ReturnTypes;
  wasm::WasmSignature LastSig;
  bool TypeErrorThisFunction = false;
  bool Unreachable = false;
  bool is64;

  void dumpTypeStack(Twine Msg);
  bool typeError(SMLoc ErrorLoc, const Twine &Msg);
  bool popType(SMLoc ErrorLoc, Optional<wasm::ValType> EVT);
  bool getLocal(SMLoc ErrorLoc, const MCInst &Inst, wasm::ValType &Type);
  bool checkEnd(SMLoc ErrorLoc, bool PopVals = false);
  bool checkSig(SMLoc ErrorLoc, const wasm::WasmSignature &Sig);
  bool getSymRef(SMLoc ErrorLoc, const MCInst &Inst,
                 const MCSymbolRefExpr *&SymRef);
  bool getGlobal(SMLoc ErrorLoc, const MCInst &Inst, wasm::ValType &Type);

public:
  WebAssemblyAsmTypeCheck(MCAsmParser &Parser, const MCInstrInfo &MII, bool is64);

  void funcDecl(const wasm::WasmSignature &Sig);
  void localDecl(const SmallVector<wasm::ValType, 4> &Locals);
  void setLastSig(const wasm::WasmSignature &Sig) { LastSig = Sig; }
  bool endOfFunction(SMLoc ErrorLoc);
  bool typeCheck(SMLoc ErrorLoc, const MCInst &Inst);

  void Clear() {
    Stack.clear();
    LocalTypes.clear();
    ReturnTypes.clear();
    TypeErrorThisFunction = false;
    Unreachable = false;
  }
};

} // end namespace llvm

#endif  // LLVM_LIB_TARGET_WEBASSEMBLY_ASMPARSER_TYPECHECK_H
