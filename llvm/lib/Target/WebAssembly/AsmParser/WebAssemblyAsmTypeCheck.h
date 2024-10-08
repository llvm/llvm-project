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
#include <variant>

namespace llvm {

class WebAssemblyAsmTypeCheck final {
  MCAsmParser &Parser;
  const MCInstrInfo &MII;

  struct Ref : public std::monostate {};
  struct Any : public std::monostate {};
  struct Polymorphic : public std::monostate {};
  using StackType = std::variant<wasm::ValType, Ref, Any, Polymorphic>;
  SmallVector<StackType, 16> Stack;
  struct BlockInfo {
    wasm::WasmSignature Sig;
    size_t StackStartPos;
    bool IsLoop;
  };
  SmallVector<BlockInfo, 8> BlockInfoStack;
  SmallVector<wasm::ValType, 16> LocalTypes;
  wasm::WasmSignature LastSig;
  bool Is64;

  // checkTypes checks 'Types' against the value stack. popTypes checks 'Types'
  // against the value stack and also pops them.
  //
  // If ExactMatch is true, 'Types' will be compared against not only the top of
  // the value stack but the whole remaining value stack
  // (TODO: This should be the whole remaining value stack "at the the current
  // block level", which has not been implemented yet)
  bool checkTypes(SMLoc ErrorLoc, ArrayRef<wasm::ValType> Types,
                  bool ExactMatch = false);
  bool checkTypes(SMLoc ErrorLoc, ArrayRef<StackType> Types,
                  bool ExactMatch = false);
  bool popTypes(SMLoc ErrorLoc, ArrayRef<wasm::ValType> Types,
                bool ExactMatch = false);
  bool popTypes(SMLoc ErrorLoc, ArrayRef<StackType> Types,
                bool ExactMatch = false);
  bool popType(SMLoc ErrorLoc, StackType Type);
  bool popRefType(SMLoc ErrorLoc);
  bool popAnyType(SMLoc ErrorLoc);
  void pushTypes(ArrayRef<wasm::ValType> Types);
  void pushType(StackType Type) { Stack.push_back(Type); }
  bool match(StackType TypeA, StackType TypeB);
  std::string getTypesString(ArrayRef<wasm::ValType> Types,
                             size_t StartPos = 0);
  std::string getTypesString(ArrayRef<StackType> Types, size_t StartPos = 0);
  SmallVector<StackType, 4>
  valTypesToStackTypes(ArrayRef<wasm::ValType> ValTypes);

  void dumpTypeStack(Twine Msg);
  bool typeError(SMLoc ErrorLoc, const Twine &Msg);
  bool getLocal(SMLoc ErrorLoc, const MCOperand &LocalOp, wasm::ValType &Type);
  bool checkSig(SMLoc ErrorLoc, const wasm::WasmSignature &Sig);
  bool getSymRef(SMLoc ErrorLoc, const MCOperand &SymOp,
                 const MCSymbolRefExpr *&SymRef);
  bool getGlobal(SMLoc ErrorLoc, const MCOperand &GlobalOp,
                 wasm::ValType &Type);
  bool getTable(SMLoc ErrorLoc, const MCOperand &TableOp, wasm::ValType &Type);
  bool getSignature(SMLoc ErrorLoc, const MCOperand &SigOp,
                    wasm::WasmSymbolType Type, const wasm::WasmSignature *&Sig);
  bool checkTryTable(SMLoc ErrorLoc, const MCInst &Inst);

public:
  WebAssemblyAsmTypeCheck(MCAsmParser &Parser, const MCInstrInfo &MII,
                          bool Is64);

  void funcDecl(const wasm::WasmSignature &Sig);
  void localDecl(const SmallVectorImpl<wasm::ValType> &Locals);
  void setLastSig(const wasm::WasmSignature &Sig) { LastSig = Sig; }
  bool endOfFunction(SMLoc ErrorLoc, bool ExactMatch);
  bool typeCheck(SMLoc ErrorLoc, const MCInst &Inst, OperandVector &Operands);

  void clear() {
    Stack.clear();
    BlockInfoStack.clear();
    LocalTypes.clear();
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_WEBASSEMBLY_ASMPARSER_TYPECHECK_H
